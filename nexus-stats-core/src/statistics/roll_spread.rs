use crate::math::MulAdd;

macro_rules! impl_roll_spread {
    ($name:ident, $builder:ident, $ty:ty) => {
        /// Roll's implicit spread estimator.
        ///
        /// Estimates effective bid-ask spread from the autocovariance of
        /// consecutive price changes: `spread = 2·√(-Cov(Δp_t, Δp_{t-1}))`.
        ///
        /// When autocovariance is non-negative (trending market), spread
        /// is undefined and `spread()` returns `None`.
        ///
        /// Hasbrouck (2009) adjustment uses autocorrelation:
        /// `spread_h = spread · √(1 + ρ)`.
        ///
        /// Roll (1984).
        ///
        /// # Parameters
        ///
        /// - `alpha` — EW decay factor for the autocovariance estimator.
        ///
        /// # Examples
        ///
        /// ```
        /// use nexus_stats_core::statistics::RollSpreadF64;
        ///
        /// let mut rs = RollSpreadF64::builder()
        ///     .alpha(0.05)
        ///     .build()
        ///     .unwrap();
        ///
        /// // Feed alternating prices (mean-reverting → negative autocov)
        /// for i in 0..200 {
        ///     let price = 100.0 + if i % 2 == 0 { 0.5 } else { -0.5 };
        ///     rs.update(price).unwrap();
        /// }
        /// assert!(rs.spread().is_some());
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            alpha: $ty,
            one_minus_alpha: $ty,
            ew_cov: $ty,
            ew_var: $ty,
            prev_price: $ty,
            prev_diff: $ty,
            count: u64,
            min_samples: u64,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            alpha: Option<$ty>,
            min_samples: u64,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    alpha: Option::None,
                    min_samples: 30,
                }
            }

            /// Feeds a trade price.
            ///
            /// # Errors
            ///
            /// Returns `DataError::NotANumber` if the price is NaN, or
            /// `DataError::Infinite` if the price is infinite.
            #[inline]
            pub fn update(&mut self, price: $ty) -> Result<(), crate::DataError> {
                check_finite!(price);
                self.count += 1;

                if self.count == 1 {
                    self.prev_price = price;
                    return Ok(());
                }

                let diff = price - self.prev_price;
                self.prev_price = price;

                if self.count == 2 {
                    self.prev_diff = diff;
                    return Ok(());
                }

                self.ew_cov = self
                    .alpha
                    .fma(diff * self.prev_diff, self.one_minus_alpha * self.ew_cov);
                self.ew_var = self
                    .alpha
                    .fma(diff * diff, self.one_minus_alpha * self.ew_var);
                self.prev_diff = diff;
                Ok(())
            }

            /// Roll's spread: `2·√(-cov)`.
            ///
            /// Returns `None` if not primed or autocovariance is non-negative
            /// (trending market — spread undefined).
            #[inline]
            #[must_use]
            pub fn spread(&self) -> Option<$ty> {
                if !self.is_primed() {
                    return Option::None;
                }
                if self.ew_cov >= 0.0 as $ty {
                    return Option::None;
                }
                #[allow(clippy::cast_possible_truncation)]
                Option::Some(2.0 as $ty * crate::math::sqrt((-self.ew_cov) as f64) as $ty)
            }

            /// Hasbrouck's adjusted spread: `spread · √(1 + ρ)` where
            /// `ρ = cov / var` is the first-order autocorrelation.
            ///
            /// Returns `None` if Roll spread is `None`, variance is zero,
            /// or `1 + ρ <= 0` (extreme mean-reversion).
            #[inline]
            #[must_use]
            pub fn hasbrouck_spread(&self) -> Option<$ty> {
                let s = self.spread()?;
                if self.ew_var <= 0.0 as $ty {
                    return Option::None;
                }
                let rho = self.ew_cov / self.ew_var;
                let factor = 1.0 as $ty + rho;
                if factor <= 0.0 as $ty {
                    return Option::None;
                }
                #[allow(clippy::cast_possible_truncation)]
                Option::Some(s * crate::math::sqrt(factor as f64) as $ty)
            }

            /// Raw exponentially weighted autocovariance, or `None` if not primed.
            #[inline]
            #[must_use]
            pub fn autocovariance(&self) -> Option<$ty> {
                if self.is_primed() {
                    Option::Some(self.ew_cov)
                } else {
                    Option::None
                }
            }

            /// Number of prices seen.
            #[inline]
            #[must_use]
            pub fn count(&self) -> u64 {
                self.count
            }

            /// Whether enough samples have been observed.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                self.count >= self.min_samples
            }

            /// Resets to uninitialized state. Parameters unchanged.
            #[inline]
            pub fn reset(&mut self) {
                self.ew_cov = 0.0 as $ty;
                self.ew_var = 0.0 as $ty;
                self.prev_price = 0.0 as $ty;
                self.prev_diff = 0.0 as $ty;
                self.count = 0;
            }
        }

        impl $builder {
            /// EW decay factor (required, in (0, 1)).
            #[inline]
            #[must_use]
            pub fn alpha(mut self, alpha: $ty) -> Self {
                self.alpha = Option::Some(alpha);
                self
            }

            /// Minimum prices before results are valid. Default: 30.
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, min: u64) -> Self {
                self.min_samples = min;
                self
            }

            /// Builds the Roll spread estimator.
            ///
            /// # Errors
            ///
            /// - Alpha must have been set and be in (0, 1).
            #[inline]
            pub fn build(self) -> Result<$name, crate::ConfigError> {
                let alpha = self.alpha.ok_or(crate::ConfigError::Missing("alpha"))?;
                if alpha <= 0.0 as $ty || alpha >= 1.0 as $ty || !alpha.is_finite() {
                    return Err(crate::ConfigError::Invalid(
                        "RollSpread alpha must be in (0, 1)",
                    ));
                }

                Ok($name {
                    alpha,
                    one_minus_alpha: 1.0 as $ty - alpha,
                    ew_cov: 0.0 as $ty,
                    ew_var: 0.0 as $ty,
                    prev_price: 0.0 as $ty,
                    prev_diff: 0.0 as $ty,
                    count: 0,
                    min_samples: self.min_samples,
                })
            }
        }
    };
}

impl_roll_spread!(RollSpreadF64, RollSpreadF64Builder, f64);
impl_roll_spread!(RollSpreadF32, RollSpreadF32Builder, f32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_reverting_spread() {
        let mut rs = RollSpreadF64::builder()
            .alpha(0.05)
            .min_samples(10)
            .build()
            .unwrap();

        for i in 0..200 {
            let price = 100.0 + if i % 2 == 0 { 0.5 } else { -0.5 };
            rs.update(price).unwrap();
        }

        let spread = rs.spread();
        assert!(spread.is_some(), "mean-reverting should produce a spread");
        assert!(spread.unwrap() > 0.0, "spread should be positive");
    }

    #[test]
    fn trending_no_spread() {
        let mut rs = RollSpreadF64::builder()
            .alpha(0.05)
            .min_samples(10)
            .build()
            .unwrap();

        for i in 0..200 {
            rs.update(100.0 + i as f64 * 0.1).unwrap();
        }

        assert!(
            rs.spread().is_none(),
            "trending series should have no spread (positive autocov)"
        );
    }

    #[test]
    fn hasbrouck_vs_roll() {
        let mut rs = RollSpreadF64::builder()
            .alpha(0.05)
            .min_samples(10)
            .build()
            .unwrap();

        // Trend + bounce: negative autocov but ρ > -1
        for i in 0..200 {
            let bounce = if i % 2 == 0 { 0.2 } else { -0.2 };
            rs.update(100.0 + (i as f64) * 0.1 + bounce).unwrap();
        }

        let roll = rs.spread().unwrap();
        let hasbrouck = rs.hasbrouck_spread().unwrap();
        assert!(
            hasbrouck > 0.0 && hasbrouck <= roll * 1.5,
            "Hasbrouck ({hasbrouck}) should be positive and reasonable vs Roll ({roll})"
        );
    }

    #[test]
    fn autocovariance_negative() {
        let mut rs = RollSpreadF64::builder()
            .alpha(0.05)
            .min_samples(10)
            .build()
            .unwrap();

        for i in 0..200 {
            let price = 100.0 + if i % 2 == 0 { 0.5 } else { -0.5 };
            rs.update(price).unwrap();
        }

        let cov = rs.autocovariance().unwrap();
        assert!(
            cov < 0.0,
            "mean-reverting should have negative autocov, got {cov}"
        );
    }

    #[test]
    fn reset_clears() {
        let mut rs = RollSpreadF64::builder()
            .alpha(0.05)
            .min_samples(10)
            .build()
            .unwrap();

        for i in 0..50 {
            let price = 100.0 + if i % 2 == 0 { 0.5 } else { -0.5 };
            rs.update(price).unwrap();
        }
        rs.reset();
        assert_eq!(rs.count(), 0);
        assert!(rs.spread().is_none());
    }

    #[test]
    fn nan_rejected() {
        let mut rs = RollSpreadF64::builder().alpha(0.05).build().unwrap();
        assert!(matches!(
            rs.update(f64::NAN),
            Err(crate::DataError::NotANumber)
        ));
    }

    #[test]
    fn inf_rejected() {
        let mut rs = RollSpreadF64::builder().alpha(0.05).build().unwrap();
        assert!(matches!(
            rs.update(f64::INFINITY),
            Err(crate::DataError::Infinite)
        ));
    }

    #[test]
    fn builder_validation() {
        assert!(matches!(
            RollSpreadF64::builder().build(),
            Err(crate::ConfigError::Missing("alpha"))
        ));
        assert!(matches!(
            RollSpreadF64::builder().alpha(0.0).build(),
            Err(crate::ConfigError::Invalid(_))
        ));
        assert!(matches!(
            RollSpreadF64::builder().alpha(1.0).build(),
            Err(crate::ConfigError::Invalid(_))
        ));
    }
}

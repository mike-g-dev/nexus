macro_rules! impl_bipower {
    ($name:ident, $builder:ident, $ty:ty, $pi_over_2:expr) => {
        /// Bipower variation — jump-robust volatility estimator.
        ///
        /// Estimates the continuous component of quadratic variation
        /// using products of consecutive absolute returns. Difference
        /// with realized variance isolates the jump component.
        ///
        /// Barndorff-Nielsen & Shephard (2004).
        ///
        /// # Examples
        ///
        /// ```
        /// use nexus_stats_core::statistics::BipowerVariationF64;
        ///
        /// let mut bv = BipowerVariationF64::new();
        /// for i in 0..100 {
        ///     bv.update(100.0 + (i as f64) * 0.01).unwrap();
        /// }
        /// assert!(bv.bipower_variation().is_some());
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            sum_bv: $ty,
            sum_rv: $ty,
            prev_abs_diff: $ty,
            prev_price: $ty,
            count: u64,
            min_samples: u64,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            min_samples: u64,
        }

        impl $name {
            /// Creates a new bipower variation tracker with default min_samples (30).
            #[inline]
            #[must_use]
            pub const fn new() -> Self {
                Self {
                    sum_bv: 0.0 as $ty,
                    sum_rv: 0.0 as $ty,
                    prev_abs_diff: 0.0 as $ty,
                    prev_price: 0.0 as $ty,
                    count: 0,
                    min_samples: 30,
                }
            }

            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder { min_samples: 30 }
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
                let abs_diff = if diff < 0.0 as $ty { -diff } else { diff };

                self.sum_rv += diff * diff;

                if self.count >= 3 {
                    self.sum_bv += abs_diff * self.prev_abs_diff;
                }

                self.prev_abs_diff = abs_diff;
                self.prev_price = price;
                Ok(())
            }

            /// Bipower variation: `(π/2) · Σ|Δp_t|·|Δp_{t-1}| / n`.
            ///
            /// Returns `None` if not primed.
            #[inline]
            #[must_use]
            pub fn bipower_variation(&self) -> Option<$ty> {
                if !self.is_primed() || self.count < 3 {
                    return Option::None;
                }
                let n = (self.count - 2) as $ty;
                Option::Some($pi_over_2 * self.sum_bv / n)
            }

            /// Realized variance: `Σ(Δp_t)² / n`.
            ///
            /// Returns `None` if not primed.
            #[inline]
            #[must_use]
            pub fn realized_variance(&self) -> Option<$ty> {
                if !self.is_primed() || self.count < 2 {
                    return Option::None;
                }
                let n = (self.count - 1) as $ty;
                Option::Some(self.sum_rv / n)
            }

            /// Jump variation: `max(RV - BV, 0)`.
            ///
            /// Returns `None` if not primed.
            #[inline]
            #[must_use]
            pub fn jump_variation(&self) -> Option<$ty> {
                let rv = self.realized_variance()?;
                let bv = self.bipower_variation()?;
                let jv = rv - bv;
                if jv > 0.0 as $ty {
                    Option::Some(jv)
                } else {
                    Option::Some(0.0 as $ty)
                }
            }

            /// Jump ratio: `max(RV - BV, 0) / RV`. Range [0, 1].
            ///
            /// Returns `None` if not primed or RV is zero.
            #[inline]
            #[must_use]
            pub fn jump_ratio(&self) -> Option<$ty> {
                let rv = self.realized_variance()?;
                if rv <= 0.0 as $ty {
                    return Option::None;
                }
                let jv = self.jump_variation()?;
                Option::Some(jv / rv)
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
                self.sum_bv = 0.0 as $ty;
                self.sum_rv = 0.0 as $ty;
                self.prev_abs_diff = 0.0 as $ty;
                self.prev_price = 0.0 as $ty;
                self.count = 0;
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $builder {
            /// Minimum prices before results are valid. Default: 30.
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, min: u64) -> Self {
                self.min_samples = min;
                self
            }

            /// Builds the bipower variation tracker.
            #[inline]
            pub fn build(self) -> $name {
                $name {
                    sum_bv: 0.0 as $ty,
                    sum_rv: 0.0 as $ty,
                    prev_abs_diff: 0.0 as $ty,
                    prev_price: 0.0 as $ty,
                    count: 0,
                    min_samples: self.min_samples,
                }
            }
        }
    };
}

impl_bipower!(
    BipowerVariationF64,
    BipowerVariationF64Builder,
    f64,
    core::f64::consts::FRAC_PI_2
);
impl_bipower!(
    BipowerVariationF32,
    BipowerVariationF32Builder,
    f32,
    core::f32::consts::FRAC_PI_2
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_series() {
        let mut bv = BipowerVariationF64::new();
        for i in 0..100 {
            bv.update(100.0 + (i as f64) * 0.01).unwrap();
        }
        let bipower = bv.bipower_variation().unwrap();
        let rv = bv.realized_variance().unwrap();
        assert!(bipower > 0.0, "bipower should be positive for smooth trend");
        assert!(
            bipower < rv * 2.0,
            "smooth series: BV should be comparable to RV, got BV={bipower}, RV={rv}"
        );
    }

    #[test]
    fn series_with_jump() {
        let mut bv = BipowerVariationF64::new();
        for i in 0..50 {
            bv.update(100.0 + (i as f64) * 0.01).unwrap();
        }
        bv.update(110.0).unwrap(); // jump
        for i in 51..100 {
            bv.update(110.0 + ((i - 51) as f64) * 0.01).unwrap();
        }
        let jv = bv.jump_variation().unwrap();
        assert!(jv > 0.0, "jump variation should be positive, got {jv}");
    }

    #[test]
    fn jump_ratio_range() {
        let mut bv = BipowerVariationF64::new();
        for i in 0..50 {
            bv.update(100.0 + (i as f64) * 0.01).unwrap();
        }
        bv.update(110.0).unwrap();
        for i in 51..100 {
            bv.update(110.0 + ((i - 51) as f64) * 0.01).unwrap();
        }
        let ratio = bv.jump_ratio().unwrap();
        assert!(
            (0.0..=1.0).contains(&ratio),
            "jump ratio should be in [0, 1], got {ratio}"
        );
    }

    #[test]
    fn rv_matches_manual() {
        let mut bv = BipowerVariationF64::new();
        let prices = [100.0, 101.0, 99.0, 102.0];
        for &p in &prices {
            bv.update(p).unwrap();
        }
        // diffs: 1, -2, 3 → sum_sq = 1+4+9 = 14, n = 3 → RV = 14/3
        let min_bv = BipowerVariationF64::builder().min_samples(2).build();
        let mut bv2 = min_bv;
        for &p in &prices {
            bv2.update(p).unwrap();
        }
        let rv = bv2.realized_variance().unwrap();
        let expected = 14.0 / 3.0;
        assert!(
            (rv - expected).abs() < 1e-10,
            "RV should be {expected}, got {rv}"
        );
    }

    #[test]
    fn bv_scaling() {
        let mut bv = BipowerVariationF64::builder().min_samples(4).build();
        let prices = [100.0, 101.0, 99.0, 102.0, 100.0];
        for &p in &prices {
            bv.update(p).unwrap();
        }
        let bipower = bv.bipower_variation().unwrap();
        // sum_bv: |1|*|-2| + |-2|*|3| + |3|*|-2| = 2+6+6 = 14, n_bv = 3
        // BV = (π/2) * 14/3
        let expected = core::f64::consts::FRAC_PI_2 * 14.0 / 3.0;
        assert!(
            (bipower - expected).abs() < 1e-10,
            "BV should be {expected}, got {bipower}"
        );
    }

    #[test]
    fn reset_clears() {
        let mut bv = BipowerVariationF64::new();
        for i in 0..50 {
            bv.update(100.0 + (i as f64) * 0.1).unwrap();
        }
        bv.reset();
        assert_eq!(bv.count(), 0);
        assert!(bv.bipower_variation().is_none());
        assert!(bv.realized_variance().is_none());
    }

    #[test]
    fn nan_rejected() {
        let mut bv = BipowerVariationF64::new();
        assert!(matches!(
            bv.update(f64::NAN),
            Err(crate::DataError::NotANumber)
        ));
    }

    #[test]
    fn inf_rejected() {
        let mut bv = BipowerVariationF64::new();
        assert!(matches!(
            bv.update(f64::INFINITY),
            Err(crate::DataError::Infinite)
        ));
    }
}

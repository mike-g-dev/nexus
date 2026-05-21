use crate::statistics::{EwmaVarF32, EwmaVarF64};

macro_rules! impl_zscore_norm {
    ($name:ident, $builder:ident, $ty:ty, $ewma_var_name:ident, $sd_floor:expr) => {
        /// Z-score normalizer backed by EWMA variance.
        ///
        /// Returns `(sample - mean) / std_dev` using exponentially weighted
        /// estimates of mean and standard deviation. Useful for online feature
        /// normalization before feeding into adaptive filters or learners.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("use nexus_stats_core::normalization::", stringify!($name), ";")]
        ///
        #[doc = concat!("let mut zs = ", stringify!($name), "::builder().span(20).build().unwrap();")]
        /// for i in 0..100 {
        #[doc = concat!("    let _ = zs.update(100.0 as ", stringify!($ty), " + i as ", stringify!($ty), ");")]
        /// }
        /// let z = zs.update(150.0).unwrap();
        /// assert!(z.is_some());
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            inner: $ewma_var_name,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            alpha: Option<$ty>,
            min_samples: u64,
            seed_mean: Option<$ty>,
            seed_variance: Option<$ty>,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    alpha: Option::None,
                    min_samples: 2,
                    seed_mean: Option::None,
                    seed_variance: Option::None,
                }
            }

            /// Feeds a sample. Returns the z-score once primed.
            ///
            /// If the standard deviation is zero (constant stream), returns `0.0`.
            ///
            /// # Errors
            ///
            /// Returns `DataError::NotANumber` if the sample is NaN, or
            /// `DataError::Infinite` if the sample is infinite.
            #[inline]
            pub fn update(&mut self, sample: $ty) -> Result<Option<$ty>, crate::DataError> {
                check_finite!(sample);
                self.inner.update(sample)?;
                Ok(self.compute_zscore(sample))
            }

            /// Normalizes an arbitrary value against current statistics
            /// without updating state.
            ///
            /// Returns `None` if not primed.
            #[inline]
            #[must_use]
            pub fn normalize(&self, value: $ty) -> Option<$ty> {
                self.compute_zscore(value)
            }

            #[inline]
            fn compute_zscore(&self, value: $ty) -> Option<$ty> {
                if !self.inner.is_primed() {
                    return Option::None;
                }
                let mean = self.inner.mean().unwrap();
                let sd = self.inner.std_dev().unwrap();
                // Guard: FMA with non-exact alpha can produce tiny non-zero
                // variance from numerical noise on constant input. Use a
                // relative floor scaled to the data magnitude.
                let scale = if mean > 0.0 as $ty { mean } else { -(mean) };
                let floor = (scale.max(1.0 as $ty)) * $sd_floor;
                if sd > floor {
                    Option::Some((value - mean) / sd)
                } else {
                    Option::Some(0.0 as $ty)
                }
            }

            /// Current smoothed mean, or `None` if not primed.
            #[inline]
            #[must_use]
            pub fn mean(&self) -> Option<$ty> {
                self.inner.mean()
            }

            /// Current exponentially weighted standard deviation, or `None` if not primed.
            #[inline]
            #[must_use]
            pub fn std_dev(&self) -> Option<$ty> {
                self.inner.std_dev()
            }

            /// Number of samples processed.
            #[inline]
            #[must_use]
            pub fn count(&self) -> u64 {
                self.inner.count()
            }

            /// Whether enough samples have been observed.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                self.inner.is_primed()
            }

            /// Resets accumulated state. Parameters unchanged.
            #[inline]
            pub fn reset(&mut self) {
                self.inner.reset();
            }
        }

        impl $builder {
            /// Direct smoothing factor. Must be in (0, 1) exclusive.
            #[inline]
            #[must_use]
            pub fn alpha(mut self, alpha: $ty) -> Self {
                self.alpha = Option::Some(alpha);
                self
            }

            /// Samples for weight to decay by half.
            #[inline]
            #[must_use]
            pub fn halflife(mut self, halflife: $ty) -> Self {
                let ln2 = core::f64::consts::LN_2 as $ty;
                let alpha = 1.0 as $ty - crate::math::exp((-ln2 / halflife) as f64) as $ty;
                self.alpha = Option::Some(alpha);
                self
            }

            /// Number of samples for center of mass (pandas convention).
            #[inline]
            #[must_use]
            pub fn span(mut self, n: u64) -> Self {
                let alpha = 2.0 as $ty / (n as $ty + 1.0 as $ty);
                self.alpha = Option::Some(alpha);
                self
            }

            /// Minimum samples before values are valid. Default: 2.
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, min: u64) -> Self {
                self.min_samples = min;
                self
            }

            /// Pre-loads mean and variance from calibration data.
            #[inline]
            #[must_use]
            pub fn seed(mut self, mean: $ty, variance: $ty) -> Self {
                self.seed_mean = Option::Some(mean);
                self.seed_variance = Option::Some(variance);
                self
            }

            /// Builds the z-score normalizer.
            ///
            /// # Errors
            ///
            /// Returns `ConfigError` if alpha is missing or not in (0, 1).
            pub fn build(self) -> Result<$name, crate::ConfigError> {
                let mut ewma_builder = $ewma_var_name::builder();
                if let Option::Some(a) = self.alpha {
                    ewma_builder = ewma_builder.alpha(a);
                }
                ewma_builder = ewma_builder.min_samples(self.min_samples);
                if let (Option::Some(m), Option::Some(v)) = (self.seed_mean, self.seed_variance) {
                    ewma_builder = ewma_builder.seed(m, v);
                }
                let inner = ewma_builder.build()?;
                Ok($name { inner })
            }
        }
    };
}

impl_zscore_norm!(ZScoreNormF64, ZScoreNormF64Builder, f64, EwmaVarF64, 1e-14);
impl_zscore_norm!(ZScoreNormF32, ZScoreNormF32Builder, f32, EwmaVarF32, 1e-5);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_scaled() {
        let mut zs = ZScoreNormF64::builder().alpha(0.1).build().unwrap();
        for i in 0..500 {
            let _ = zs.update(100.0 + (i % 10) as f64);
        }
        let z = zs.update(105.0).unwrap().unwrap();
        assert!(
            z.abs() < 3.0,
            "z-score of near-mean value should be moderate, got {z}"
        );
    }

    #[test]
    fn warmup_returns_none() {
        let mut zs = ZScoreNormF64::builder().alpha(0.1).build().unwrap();
        assert!(zs.update(1.0).unwrap().is_none());
        assert!(!zs.is_primed());
    }

    #[test]
    fn normalize_without_update() {
        let mut zs = ZScoreNormF64::builder()
            .alpha(0.1)
            .seed(100.0, 25.0)
            .build()
            .unwrap();
        let z = zs.normalize(105.0).unwrap();
        assert!((z - 1.0).abs() < 0.01, "expected z ≈ 1.0, got {z}");
        let z2 = zs.update(105.0).unwrap().unwrap();
        assert!(z != z2, "update should change internal state");
    }

    #[test]
    fn zero_variance() {
        let mut zs = ZScoreNormF64::builder().alpha(0.1).build().unwrap();
        for _ in 0..100 {
            let z = zs.update(42.0).unwrap();
            if let Option::Some(v) = z {
                assert!(
                    v.abs() < 1e-10,
                    "constant stream z-score should be 0.0, got {v}"
                );
            }
        }
    }

    #[test]
    fn rejects_nan_inf() {
        let mut zs = ZScoreNormF64::builder().alpha(0.1).build().unwrap();
        assert!(zs.update(f64::NAN).is_err());
        assert!(zs.update(f64::INFINITY).is_err());
        assert!(zs.update(f64::NEG_INFINITY).is_err());
        assert_eq!(zs.count(), 0);
    }

    #[test]
    fn reset_clears() {
        let mut zs = ZScoreNormF64::builder().alpha(0.1).build().unwrap();
        for i in 0..50 {
            let _ = zs.update(i as f64);
        }
        assert!(zs.is_primed());
        zs.reset();
        assert_eq!(zs.count(), 0);
        assert!(!zs.is_primed());
        assert!(zs.mean().is_none());
    }
}

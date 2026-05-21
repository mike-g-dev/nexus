use crate::monitoring::{WindowedMaxF32, WindowedMaxF64, WindowedMinF32, WindowedMinF64};

macro_rules! impl_minmax_norm {
    ($name:ident, $builder:ident, $ty:ty, $windowed_min:ident, $windowed_max:ident) => {
        /// Windowed min-max normalizer.
        ///
        /// Scales values to [0, 1] using the windowed minimum and maximum
        /// over a sliding timestamp window: `(sample - min) / (max - min)`.
        ///
        /// When the range is zero (all values equal), returns 0.5.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("use nexus_stats_core::normalization::", stringify!($name), ";")]
        ///
        #[doc = concat!("let mut mm = ", stringify!($name), "::builder().window(100).build().unwrap();")]
        #[doc = concat!("let _ = mm.update(0, 10.0 as ", stringify!($ty), ");")]
        #[doc = concat!("let _ = mm.update(1, 20.0 as ", stringify!($ty), ");")]
        #[doc = concat!("let v = mm.update(2, 15.0 as ", stringify!($ty), ").unwrap();")]
        /// assert!(v.is_some());
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            min_tracker: $windowed_min,
            max_tracker: $windowed_max,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            window: Option<u64>,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    window: Option::None,
                }
            }

            /// Feeds a sample at the given timestamp. Returns normalized value
            /// in [0, 1] once at least one sample has been observed.
            ///
            /// # Errors
            ///
            /// Returns `DataError::NotANumber` if the sample is NaN, or
            /// `DataError::Infinite` if the sample is infinite.
            #[inline]
            pub fn update(
                &mut self,
                timestamp: u64,
                sample: $ty,
            ) -> Result<Option<$ty>, crate::DataError> {
                check_finite!(sample);
                self.min_tracker.update(timestamp, sample)?;
                self.max_tracker.update(timestamp, sample)?;
                Ok(self.compute_normalized(sample))
            }

            /// Normalizes an arbitrary value against current windowed min/max
            /// without updating state.
            ///
            /// Returns `None` if not primed.
            #[inline]
            #[must_use]
            pub fn normalize(&self, value: $ty) -> Option<$ty> {
                self.compute_normalized(value)
            }

            #[inline]
            fn compute_normalized(&self, value: $ty) -> Option<$ty> {
                let min = self.min_tracker.min()?;
                let max = self.max_tracker.max()?;
                let range = max - min;
                if range > 0.0 as $ty {
                    Option::Some((value - min) / range)
                } else {
                    Option::Some(0.5 as $ty)
                }
            }

            /// Current windowed minimum, or `None` if empty.
            #[inline]
            #[must_use]
            pub fn min(&self) -> Option<$ty> {
                self.min_tracker.min()
            }

            /// Current windowed maximum, or `None` if empty.
            #[inline]
            #[must_use]
            pub fn max(&self) -> Option<$ty> {
                self.max_tracker.max()
            }

            /// Current range (max - min), or `None` if empty.
            #[inline]
            #[must_use]
            pub fn range(&self) -> Option<$ty> {
                let min = self.min_tracker.min()?;
                let max = self.max_tracker.max()?;
                Option::Some(max - min)
            }

            /// Number of samples processed (minimum of both trackers).
            #[inline]
            #[must_use]
            pub fn count(&self) -> u64 {
                self.min_tracker.count().min(self.max_tracker.count())
            }

            /// Whether at least one sample has been observed.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                self.min_tracker.count() > 0 && self.max_tracker.count() > 0
            }

            /// Resets accumulated state. Window size preserved.
            #[inline]
            pub fn reset(&mut self) {
                self.min_tracker.reset();
                self.max_tracker.reset();
            }
        }

        impl $builder {
            /// Timestamp window size (required, must be positive).
            #[inline]
            #[must_use]
            pub fn window(mut self, window: u64) -> Self {
                self.window = Option::Some(window);
                self
            }

            /// Builds the min-max normalizer.
            ///
            /// # Errors
            ///
            /// Returns `ConfigError` if window is missing or zero.
            pub fn build(self) -> Result<$name, crate::ConfigError> {
                let window = self
                    .window
                    .ok_or(crate::ConfigError::Missing("window"))?;
                let min_tracker = $windowed_min::new(window)?;
                let max_tracker = $windowed_max::new(window)?;
                Ok($name {
                    min_tracker,
                    max_tracker,
                })
            }
        }
    };
}

impl_minmax_norm!(
    MinMaxNormF64,
    MinMaxNormF64Builder,
    f64,
    WindowedMinF64,
    WindowedMaxF64
);
impl_minmax_norm!(
    MinMaxNormF32,
    MinMaxNormF32Builder,
    f32,
    WindowedMinF32,
    WindowedMaxF32
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_to_unit() {
        let mut mm = MinMaxNormF64::builder().window(1000).build().unwrap();
        let _ = mm.update(0, 10.0).unwrap();
        let _ = mm.update(1, 20.0).unwrap();
        let v = mm.update(2, 30.0).unwrap().unwrap();
        assert!((v - 1.0).abs() < 1e-10, "30 should map to 1.0, got {v}");
        let mid = mm.normalize(20.0).unwrap();
        assert!((mid - 0.5).abs() < 1e-10, "20 should map to 0.5, got {mid}");
    }

    #[test]
    fn windowed_eviction() {
        let mut mm = MinMaxNormF64::builder().window(30).build().unwrap();
        // Wide range initially
        let _ = mm.update(0, 0.0).unwrap();
        let _ = mm.update(1, 100.0).unwrap();
        let v1 = mm.update(2, 50.0).unwrap().unwrap();
        assert!(
            (v1 - 0.5).abs() < 1e-10,
            "50 in [0,100] should be 0.5, got {v1}"
        );
        // Feed narrow-range values after old extrema expire
        for t in 35..60 {
            let _ = mm.update(t, 40.0 + (t % 10) as f64).unwrap();
        }
        let range = mm.range().unwrap();
        assert!(
            range < 50.0,
            "range should have narrowed after window eviction, got {range}"
        );
    }

    #[test]
    fn constant_returns_half() {
        let mut mm = MinMaxNormF64::builder().window(100).build().unwrap();
        for i in 0..20u64 {
            let v = mm.update(i, 42.0).unwrap().unwrap();
            assert!(
                (v - 0.5).abs() < 1e-10,
                "constant stream should return 0.5, got {v}"
            );
        }
    }

    #[test]
    fn normalize_without_update() {
        let mut mm = MinMaxNormF64::builder().window(100).build().unwrap();
        let _ = mm.update(0, 0.0).unwrap();
        let _ = mm.update(1, 100.0).unwrap();
        let v = mm.normalize(75.0).unwrap();
        assert!(
            (v - 0.75).abs() < 1e-10,
            "75 in [0,100] should be 0.75, got {v}"
        );
    }

    #[test]
    fn single_sample() {
        let mut mm = MinMaxNormF64::builder().window(100).build().unwrap();
        let v = mm.update(0, 42.0).unwrap().unwrap();
        assert!(
            (v - 0.5).abs() < 1e-10,
            "single sample: range=0 → 0.5, got {v}"
        );
    }

    #[test]
    fn rejects_nan_inf() {
        let mut mm = MinMaxNormF64::builder().window(100).build().unwrap();
        assert!(mm.update(0, f64::NAN).is_err());
        assert!(mm.update(0, f64::INFINITY).is_err());
        assert!(mm.update(0, f64::NEG_INFINITY).is_err());
        assert_eq!(mm.count(), 0);
    }
}

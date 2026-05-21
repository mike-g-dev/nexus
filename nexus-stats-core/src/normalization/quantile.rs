extern crate alloc;
use crate::statistics::{PercentileF32, PercentileF64};
use alloc::boxed::Box;

macro_rules! impl_quantile_norm {
    ($name:ident, $builder:ident, $ty:ty, $percentile_name:ident) => {
        /// Quantile normalizer via P² estimator grid.
        ///
        /// Maps values to approximate uniform [0, 1] by maintaining a grid
        /// of P² percentile estimators at uniformly spaced quantile points.
        /// The normalized value is the interpolated rank within the grid.
        ///
        /// O(resolution) per update. Requires `alloc`.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("use nexus_stats_core::normalization::", stringify!($name), ";")]
        ///
        #[doc = concat!("let mut qn = ", stringify!($name), "::builder().resolution(9).build().unwrap();")]
        /// for i in 0..500 {
        #[doc = concat!("    let _ = qn.update(i as ", stringify!($ty), ");")]
        /// }
        /// let v = qn.update(250.0).unwrap();
        /// assert!(v.is_some());
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            grid: Box<[$percentile_name]>,
            quantile_points: Box<[$ty]>,
            resolution: usize,
            count: u64,
            min_samples: u64,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            resolution: Option<usize>,
            min_samples: Option<u64>,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    resolution: Option::None,
                    min_samples: Option::None,
                }
            }

            /// Feeds a sample to all grid estimators. Returns the approximate
            /// normalized rank in [0, 1] once primed.
            ///
            /// # Errors
            ///
            /// Returns `DataError::NotANumber` if the sample is NaN, or
            /// `DataError::Infinite` if the sample is infinite.
            #[inline]
            pub fn update(&mut self, sample: $ty) -> Result<Option<$ty>, crate::DataError> {
                check_finite!(sample);
                for p in self.grid.iter_mut() {
                    p.update(sample)?;
                }
                self.count += 1;

                if !self.is_primed() {
                    return Ok(Option::None);
                }

                Ok(Option::Some(self.rank(sample)))
            }

            /// Ranks an arbitrary value against current grid without updating state.
            ///
            /// Returns `None` if not primed.
            #[inline]
            #[must_use]
            pub fn normalize(&self, value: $ty) -> Option<$ty> {
                if !self.is_primed() {
                    return Option::None;
                }
                Option::Some(self.rank(value))
            }

            #[allow(clippy::float_cmp, clippy::suboptimal_flops)]
            fn rank(&self, value: $ty) -> $ty {
                let n = self.resolution;

                let first_q = self.grid[0].percentile().unwrap();
                if value <= first_q {
                    let frac = self.quantile_points[0];
                    if first_q == value {
                        return frac;
                    }
                    return frac * 0.5 as $ty;
                }

                let last_q = self.grid[n - 1].percentile().unwrap();
                if value >= last_q {
                    let frac = self.quantile_points[n - 1];
                    if last_q == value {
                        return frac;
                    }
                    return frac + (1.0 as $ty - frac) * 0.5 as $ty;
                }

                for i in 1..n {
                    let q_prev = self.grid[i - 1].percentile().unwrap();
                    let q_curr = self.grid[i].percentile().unwrap();

                    if value <= q_curr {
                        let range = q_curr - q_prev;
                        if range > 0.0 as $ty {
                            let t = (value - q_prev) / range;
                            let p_prev = self.quantile_points[i - 1];
                            let p_curr = self.quantile_points[i];
                            return p_prev + t * (p_curr - p_prev);
                        }
                        return self.quantile_points[i];
                    }
                }

                self.quantile_points[n - 1]
            }

            /// Number of quantile grid points.
            #[inline]
            #[must_use]
            pub fn resolution(&self) -> usize {
                self.resolution
            }

            /// Total samples processed.
            #[inline]
            #[must_use]
            pub fn count(&self) -> u64 {
                self.count
            }

            /// Whether all grid estimators are primed.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                self.count >= self.min_samples
            }

            /// Resets all grid estimators. Resolution preserved.
            #[inline]
            pub fn reset(&mut self) {
                for p in self.grid.iter_mut() {
                    p.reset();
                }
                self.count = 0;
            }
        }

        impl $builder {
            /// Number of quantile grid points (required, >= 2).
            ///
            /// Grid points are placed at `1/(n+1), 2/(n+1), ..., n/(n+1)`.
            /// Resolution 9 → grid at 0.1, 0.2, ..., 0.9.
            #[inline]
            #[must_use]
            pub fn resolution(mut self, n: usize) -> Self {
                self.resolution = Option::Some(n);
                self
            }

            /// Minimum samples before `normalize` returns values.
            /// Default: derived from the most extreme grid quantile.
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, n: u64) -> Self {
                self.min_samples = Option::Some(n);
                self
            }

            /// Builds the quantile normalizer.
            ///
            /// # Errors
            ///
            /// Returns `ConfigError` if resolution is missing or < 2.
            pub fn build(self) -> Result<$name, crate::ConfigError> {
                let resolution = self
                    .resolution
                    .ok_or(crate::ConfigError::Missing("resolution"))?;
                if resolution < 2 {
                    return Err(crate::ConfigError::Invalid(
                        "resolution must be >= 2",
                    ));
                }

                let denom = (resolution + 1) as $ty;
                let mut grid = alloc::vec::Vec::with_capacity(resolution);
                let mut quantile_points = alloc::vec::Vec::with_capacity(resolution);

                let mut max_min_samples: u64 = 5;

                for i in 1..=resolution {
                    let p = i as $ty / denom;
                    quantile_points.push(p);
                    let estimator = $percentile_name::new(p)?;
                    let ms = Self::adaptive_min_samples(p);
                    if ms > max_min_samples {
                        max_min_samples = ms;
                    }
                    grid.push(estimator);
                }

                let min_samples = self.min_samples.unwrap_or(max_min_samples).max(max_min_samples);

                Ok($name {
                    grid: grid.into_boxed_slice(),
                    quantile_points: quantile_points.into_boxed_slice(),
                    resolution,
                    count: 0,
                    min_samples,
                })
            }

            fn adaptive_min_samples(p: $ty) -> u64 {
                let denom = p * (1.0 as $ty - p);
                if denom <= 0.0 as $ty {
                    return 1000;
                }
                ((1.0 as $ty / denom) as u64).max(5)
            }
        }
    };
}

impl_quantile_norm!(QuantileNormF64, QuantileNormF64Builder, f64, PercentileF64);
impl_quantile_norm!(QuantileNormF32, QuantileNormF32Builder, f32, PercentileF32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_input_uniform_output() {
        let mut qn = QuantileNormF64::builder().resolution(9).build().unwrap();
        for i in 0..2000 {
            let _ = qn.update(i as f64);
        }
        let low = qn.normalize(200.0).unwrap();
        let mid = qn.normalize(1000.0).unwrap();
        let high = qn.normalize(1800.0).unwrap();

        assert!(low < 0.2, "200/2000 should be low rank, got {low}");
        assert!(
            mid > 0.35 && mid < 0.65,
            "1000/2000 should be ~0.5, got {mid}"
        );
        assert!(high > 0.8, "1800/2000 should be high rank, got {high}");
    }

    #[test]
    fn resolution_affects_granularity() {
        let mut coarse = QuantileNormF64::builder().resolution(3).build().unwrap();
        let mut fine = QuantileNormF64::builder().resolution(19).build().unwrap();
        for i in 0..2000 {
            let v = i as f64;
            let _ = coarse.update(v);
            let _ = fine.update(v);
        }
        assert_eq!(coarse.resolution(), 3);
        assert_eq!(fine.resolution(), 19);
        let vc = coarse.normalize(500.0).unwrap();
        let vf = fine.normalize(500.0).unwrap();
        assert!(vc > 0.0 && vc < 1.0, "coarse should be in (0,1), got {vc}");
        assert!(vf > 0.0 && vf < 1.0, "fine should be in (0,1), got {vf}");
    }

    #[test]
    fn normalize_without_update() {
        let mut qn = QuantileNormF64::builder().resolution(9).build().unwrap();
        for i in 0..1000 {
            let _ = qn.update(i as f64);
        }
        let before_count = qn.count();
        let _ = qn.normalize(500.0);
        assert_eq!(
            qn.count(),
            before_count,
            "normalize should not change count"
        );
    }

    #[test]
    fn warmup_returns_none() {
        let mut qn = QuantileNormF64::builder().resolution(9).build().unwrap();
        assert!(qn.update(1.0).unwrap().is_none());
        assert!(!qn.is_primed());
    }

    #[test]
    fn rejects_nan_inf() {
        let mut qn = QuantileNormF64::builder().resolution(4).build().unwrap();
        assert!(qn.update(f64::NAN).is_err());
        assert!(qn.update(f64::INFINITY).is_err());
        assert!(qn.update(f64::NEG_INFINITY).is_err());
        assert_eq!(qn.count(), 0);
    }

    #[test]
    fn reset_clears() {
        let mut qn = QuantileNormF64::builder().resolution(4).build().unwrap();
        for i in 0..200 {
            let _ = qn.update(i as f64);
        }
        assert!(qn.count() > 0);
        qn.reset();
        assert_eq!(qn.count(), 0);
        assert!(!qn.is_primed());
    }
}

extern crate alloc;
use alloc::boxed::Box;
use alloc::vec;
use nexus_stats_core::DataError;

macro_rules! impl_predictive_info {
    ($name:ident, $builder:ident, $ty:ty) => {
        /// Streaming mutual information estimator via binned MI.
        ///
        /// Estimates I(X; Y) from joint and marginal frequency counts
        /// over equi-width bins. Miller-Madow bias correction adjusts
        /// for finite-sample underestimation.
        ///
        /// O(K²) per query (K = bin count). O(1) per update.
        ///
        /// # Parameters
        ///
        /// - `bins` — number of bins per dimension (total cells = bins²)
        /// - `x_range` — `(min, max)` for the X variable
        /// - `y_range` — `(min, max)` for the Y variable
        ///
        /// # Examples
        ///
        /// ```
        /// use nexus_stats_detection::signal::PredictiveInfoBoundF64;
        ///
        /// let mut pib = PredictiveInfoBoundF64::builder()
        ///     .bins(10)
        ///     .x_range(0.0, 100.0)
        ///     .y_range(0.0, 100.0)
        ///     .build()
        ///     .unwrap();
        ///
        /// // Feed identical values → high MI
        /// for i in 0..1000 {
        ///     let x = (i % 100) as f64;
        ///     pib.update(x, x).unwrap();
        /// }
        /// assert!(pib.mutual_information().unwrap() > 0.0);
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            bins: usize,
            x_min: $ty,
            x_max: $ty,
            y_min: $ty,
            y_max: $ty,
            x_width: $ty,
            y_width: $ty,
            joint: Box<[u64]>,
            marginal_x: Box<[u64]>,
            marginal_y: Box<[u64]>,
            count: u64,
            min_samples: u64,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            bins: Option<usize>,
            x_range: Option<($ty, $ty)>,
            y_range: Option<($ty, $ty)>,
            min_samples: Option<u64>,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    bins: Option::None,
                    x_range: Option::None,
                    y_range: Option::None,
                    min_samples: Option::None,
                }
            }

            /// Feeds an (x, y) observation.
            ///
            /// Values outside the configured range are clamped to the
            /// nearest bin (first or last).
            ///
            /// # Errors
            ///
            /// Returns `DataError::NotANumber` if either value is NaN, or
            /// `DataError::Infinite` if either value is infinite.
            #[inline]
            pub fn update(&mut self, x: $ty, y: $ty) -> Result<(), DataError> {
                check_finite!(x);
                check_finite!(y);

                let xi = self.bin_x(x);
                let yi = self.bin_y(y);

                self.joint[xi * self.bins + yi] += 1;
                self.marginal_x[xi] += 1;
                self.marginal_y[yi] += 1;
                self.count += 1;
                Ok(())
            }

            /// Mutual information I(X; Y) in nats, with Miller-Madow correction.
            ///
            /// Returns `None` if not primed.
            #[must_use]
            pub fn mutual_information(&self) -> Option<$ty> {
                if !self.is_primed() {
                    return Option::None;
                }
                let n = self.count as $ty;

                let mut mi = 0.0 as $ty;
                for xi in 0..self.bins {
                    let nx = self.marginal_x[xi];
                    if nx == 0 {
                        continue;
                    }
                    for yi in 0..self.bins {
                        let ny = self.marginal_y[yi];
                        let nxy = self.joint[xi * self.bins + yi];
                        if nxy == 0 || ny == 0 {
                            continue;
                        }
                        let p_xy = nxy as $ty / n;
                        let p_x = nx as $ty / n;
                        let p_y = ny as $ty / n;
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            mi += p_xy
                                * nexus_stats_core::math::ln((p_xy / (p_x * p_y)) as f64) as $ty;
                        }
                    }
                }

                let occupied = self.count_occupied_cells() as $ty;
                let correction = (occupied - 1.0 as $ty) / (2.0 as $ty * n);
                Option::Some(mi + correction)
            }

            /// Mutual information in bits (log base 2).
            ///
            /// Returns `None` if not primed.
            #[inline]
            #[must_use]
            pub fn mutual_information_bits(&self) -> Option<$ty> {
                #[allow(clippy::cast_possible_truncation)]
                self.mutual_information()
                    .map(|mi| mi / nexus_stats_core::math::ln(2.0) as $ty)
            }

            /// Normalized MI: I(X;Y) / min(H(X), H(Y)). Range [0, 1].
            ///
            /// Returns `None` if not primed or if either marginal entropy is zero.
            #[must_use]
            pub fn normalized_mi(&self) -> Option<$ty> {
                let mi = self.mutual_information()?;
                let hx = self.marginal_entropy_x()?;
                let hy = self.marginal_entropy_y()?;
                let min_h = if hx < hy { hx } else { hy };
                if min_h <= 0.0 as $ty {
                    return Option::None;
                }
                let nmi = mi / min_h;
                let clamped = if nmi < 0.0 as $ty {
                    0.0 as $ty
                } else if nmi > 1.0 as $ty {
                    1.0 as $ty
                } else {
                    nmi
                };
                Option::Some(clamped)
            }

            fn marginal_entropy_x(&self) -> Option<$ty> {
                if self.count == 0 {
                    return Option::None;
                }
                let n = self.count as $ty;
                let mut h = 0.0 as $ty;
                for xi in 0..self.bins {
                    let c = self.marginal_x[xi];
                    if c > 0 {
                        let p = c as $ty / n;
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            h -= p * nexus_stats_core::math::ln(p as f64) as $ty;
                        }
                    }
                }
                Option::Some(h)
            }

            fn marginal_entropy_y(&self) -> Option<$ty> {
                if self.count == 0 {
                    return Option::None;
                }
                let n = self.count as $ty;
                let mut h = 0.0 as $ty;
                for yi in 0..self.bins {
                    let c = self.marginal_y[yi];
                    if c > 0 {
                        let p = c as $ty / n;
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            h -= p * nexus_stats_core::math::ln(p as f64) as $ty;
                        }
                    }
                }
                Option::Some(h)
            }

            fn count_occupied_cells(&self) -> u64 {
                self.joint.iter().filter(|&&c| c > 0).count() as u64
            }

            fn bin_x(&self, x: $ty) -> usize {
                let clamped = if x < self.x_min {
                    self.x_min
                } else if x > self.x_max {
                    self.x_max
                } else {
                    x
                };
                let idx = ((clamped - self.x_min) / self.x_width) as usize;
                if idx >= self.bins { self.bins - 1 } else { idx }
            }

            fn bin_y(&self, y: $ty) -> usize {
                let clamped = if y < self.y_min {
                    self.y_min
                } else if y > self.y_max {
                    self.y_max
                } else {
                    y
                };
                let idx = ((clamped - self.y_min) / self.y_width) as usize;
                if idx >= self.bins { self.bins - 1 } else { idx }
            }

            /// Total observations.
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

            /// Resets counts. Configuration and allocation preserved.
            #[inline]
            pub fn reset(&mut self) {
                self.joint.fill(0);
                self.marginal_x.fill(0);
                self.marginal_y.fill(0);
                self.count = 0;
            }
        }

        impl $builder {
            /// Number of bins per dimension (required, 2..=256).
            #[inline]
            #[must_use]
            pub fn bins(mut self, bins: usize) -> Self {
                self.bins = Option::Some(bins);
                self
            }

            /// Range for the X variable (required, max > min).
            #[inline]
            #[must_use]
            pub fn x_range(mut self, min: $ty, max: $ty) -> Self {
                self.x_range = Option::Some((min, max));
                self
            }

            /// Range for the Y variable (required, max > min).
            #[inline]
            #[must_use]
            pub fn y_range(mut self, min: $ty, max: $ty) -> Self {
                self.y_range = Option::Some((min, max));
                self
            }

            /// Minimum samples before MI is valid. Default: bins².
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, min: u64) -> Self {
                self.min_samples = Option::Some(min);
                self
            }

            /// Builds the mutual information estimator.
            ///
            /// # Errors
            ///
            /// - `bins` must be in [2, 256].
            /// - `x_range` and `y_range` must have `max > min`, both finite.
            pub fn build(self) -> Result<$name, nexus_stats_core::ConfigError> {
                let bins = self
                    .bins
                    .ok_or(nexus_stats_core::ConfigError::Missing("bins"))?;
                if bins < 2 {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "PredictiveInfoBound bins must be >= 2",
                    ));
                }
                if bins > 256 {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "PredictiveInfoBound bins must be <= 256",
                    ));
                }

                let (x_min, x_max) = self
                    .x_range
                    .ok_or(nexus_stats_core::ConfigError::Missing("x_range"))?;
                if x_max <= x_min || !x_min.is_finite() || !x_max.is_finite() {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "x_range must have max > min, both finite",
                    ));
                }

                let (y_min, y_max) = self
                    .y_range
                    .ok_or(nexus_stats_core::ConfigError::Missing("y_range"))?;
                if y_max <= y_min || !y_min.is_finite() || !y_max.is_finite() {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "y_range must have max > min, both finite",
                    ));
                }

                let x_width = (x_max - x_min) / bins as $ty;
                let y_width = (y_max - y_min) / bins as $ty;
                let min_samples = self.min_samples.unwrap_or((bins * bins) as u64);
                if min_samples < 1 {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "PredictiveInfoBound min_samples must be >= 1",
                    ));
                }

                Ok($name {
                    bins,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    x_width,
                    y_width,
                    joint: vec![0u64; bins * bins].into_boxed_slice(),
                    marginal_x: vec![0u64; bins].into_boxed_slice(),
                    marginal_y: vec![0u64; bins].into_boxed_slice(),
                    count: 0,
                    min_samples,
                })
            }
        }
    };
}

impl_predictive_info!(PredictiveInfoBoundF64, PredictiveInfoBoundF64Builder, f64);
impl_predictive_info!(PredictiveInfoBoundF32, PredictiveInfoBoundF32Builder, f32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_variables() {
        let mut pib = PredictiveInfoBoundF64::builder()
            .bins(10)
            .x_range(0.0, 10.0)
            .y_range(0.0, 10.0)
            .min_samples(10)
            .build()
            .unwrap();

        // X cycles 0..10 sequentially, Y uses a co-prime stride that
        // produces a uniform marginal over bins, independent of X.
        // With 10 bins and stride 7 (coprime to 10), every (x,y) cell
        // is hit equally over a full 100-step cycle.
        for i in 0..10_000u64 {
            let x = (i % 10) as f64 + 0.5;
            let y = ((i / 10) % 10) as f64 + 0.5;
            pib.update(x, y).unwrap();
        }

        let mi = pib.mutual_information().unwrap();
        assert!(
            mi.abs() < 0.1,
            "independent variables should have MI ≈ 0, got {mi}"
        );
    }

    #[test]
    fn identical_variables() {
        let mut pib = PredictiveInfoBoundF64::builder()
            .bins(10)
            .x_range(0.0, 10.0)
            .y_range(0.0, 10.0)
            .min_samples(10)
            .build()
            .unwrap();

        for i in 0..1000 {
            let x = (i % 10) as f64 + 0.5;
            pib.update(x, x).unwrap();
        }

        let mi = pib.mutual_information().unwrap();
        let hx = pib.marginal_entropy_x().unwrap();
        assert!(
            (mi - hx).abs() < 0.2,
            "identical variables: MI ({mi}) should ≈ H(X) ({hx})"
        );
    }

    #[test]
    fn linear_relationship() {
        let mut pib_tight = PredictiveInfoBoundF64::builder()
            .bins(10)
            .x_range(0.0, 10.0)
            .y_range(0.0, 20.0)
            .min_samples(10)
            .build()
            .unwrap();

        for i in 0..1000 {
            let x = (i % 10) as f64 + 0.5;
            let y = 2.0 * x;
            pib_tight.update(x, y).unwrap();
        }

        let mi_tight = pib_tight.mutual_information().unwrap();
        assert!(
            mi_tight > 0.5,
            "tight linear relationship should have high MI, got {mi_tight}"
        );
    }

    #[test]
    fn clamping() {
        let mut pib = PredictiveInfoBoundF64::builder()
            .bins(5)
            .x_range(0.0, 10.0)
            .y_range(0.0, 10.0)
            .min_samples(2)
            .build()
            .unwrap();

        pib.update(-5.0, 15.0).unwrap();
        pib.update(100.0, -100.0).unwrap();
        assert_eq!(pib.count(), 2);
    }

    #[test]
    fn miller_madow_correction() {
        let mut pib = PredictiveInfoBoundF64::builder()
            .bins(4)
            .x_range(0.0, 4.0)
            .y_range(0.0, 4.0)
            .min_samples(2)
            .build()
            .unwrap();

        for i in 0..20 {
            let x = (i % 4) as f64 + 0.5;
            pib.update(x, x).unwrap();
        }

        let mi = pib.mutual_information().unwrap();
        assert!(mi > 0.0, "MI with correction should be positive, got {mi}");
    }

    #[test]
    fn bits_vs_nats() {
        let mut pib = PredictiveInfoBoundF64::builder()
            .bins(5)
            .x_range(0.0, 5.0)
            .y_range(0.0, 5.0)
            .min_samples(5)
            .build()
            .unwrap();

        for i in 0..500 {
            let x = (i % 5) as f64 + 0.5;
            pib.update(x, x).unwrap();
        }

        let nats = pib.mutual_information().unwrap();
        let bits = pib.mutual_information_bits().unwrap();
        let ratio = bits / nats;
        let expected_ratio = 1.0 / (2.0_f64).ln();
        assert!(
            (ratio - expected_ratio).abs() < 0.01,
            "bits/nats ratio should be 1/ln(2), got {ratio}"
        );
    }

    #[test]
    fn normalized_mi_range() {
        let mut pib = PredictiveInfoBoundF64::builder()
            .bins(5)
            .x_range(0.0, 5.0)
            .y_range(0.0, 5.0)
            .min_samples(5)
            .build()
            .unwrap();

        for i in 0..500 {
            let x = (i % 5) as f64 + 0.5;
            pib.update(x, x).unwrap();
        }

        let nmi = pib.normalized_mi().unwrap();
        assert!(
            (0.0..=1.0).contains(&nmi),
            "normalized MI should be in [0, 1], got {nmi}"
        );
    }

    #[test]
    fn reset_clears() {
        let mut pib = PredictiveInfoBoundF64::builder()
            .bins(5)
            .x_range(0.0, 5.0)
            .y_range(0.0, 5.0)
            .min_samples(5)
            .build()
            .unwrap();

        for i in 0..100 {
            let x = (i % 5) as f64 + 0.5;
            pib.update(x, x).unwrap();
        }

        pib.reset();
        assert_eq!(pib.count(), 0);
        assert!(pib.mutual_information().is_none());
    }

    #[test]
    fn nan_rejected() {
        let mut pib = PredictiveInfoBoundF64::builder()
            .bins(5)
            .x_range(0.0, 5.0)
            .y_range(0.0, 5.0)
            .build()
            .unwrap();
        assert!(matches!(
            pib.update(f64::NAN, 1.0),
            Err(DataError::NotANumber)
        ));
        assert!(matches!(
            pib.update(1.0, f64::NAN),
            Err(DataError::NotANumber)
        ));
    }

    #[test]
    fn inf_rejected() {
        let mut pib = PredictiveInfoBoundF64::builder()
            .bins(5)
            .x_range(0.0, 5.0)
            .y_range(0.0, 5.0)
            .build()
            .unwrap();
        assert!(matches!(
            pib.update(f64::INFINITY, 1.0),
            Err(DataError::Infinite)
        ));
    }

    #[test]
    fn builder_validation() {
        assert!(matches!(
            PredictiveInfoBoundF64::builder()
                .x_range(0.0, 10.0)
                .y_range(0.0, 10.0)
                .build(),
            Err(nexus_stats_core::ConfigError::Missing("bins"))
        ));

        assert!(matches!(
            PredictiveInfoBoundF64::builder()
                .bins(1)
                .x_range(0.0, 10.0)
                .y_range(0.0, 10.0)
                .build(),
            Err(nexus_stats_core::ConfigError::Invalid(_))
        ));

        assert!(matches!(
            PredictiveInfoBoundF64::builder()
                .bins(257)
                .x_range(0.0, 10.0)
                .y_range(0.0, 10.0)
                .build(),
            Err(nexus_stats_core::ConfigError::Invalid(_))
        ));

        assert!(matches!(
            PredictiveInfoBoundF64::builder()
                .bins(10)
                .x_range(10.0, 0.0)
                .y_range(0.0, 10.0)
                .build(),
            Err(nexus_stats_core::ConfigError::Invalid(_))
        ));

        assert!(matches!(
            PredictiveInfoBoundF64::builder()
                .bins(10)
                .x_range(0.0, 10.0)
                .y_range(10.0, 0.0)
                .build(),
            Err(nexus_stats_core::ConfigError::Invalid(_))
        ));
    }
}

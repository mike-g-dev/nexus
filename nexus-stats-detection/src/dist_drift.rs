extern crate alloc;
use alloc::boxed::Box;

macro_rules! impl_dist_drift {
    ($name:ident, $builder:ident, $ty:ty, $ln:path) => {
        /// Distribution drift metrics via reference/live histograms.
        ///
        /// Maintains two equi-width histograms (reference and live) and
        /// computes three divergence measures:
        ///
        /// - KL divergence: KL(live || reference), in nats
        /// - Jensen-Shannon divergence: symmetric, bounded in [0, ln2]
        /// - Wasserstein-1 distance: earth mover's distance
        ///
        /// Out-of-range samples are clamped to boundary bins.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("use nexus_stats_detection::detection::", stringify!($name), ";")]
        ///
        #[doc = concat!("let mut drift = ", stringify!($name), "::builder()")]
        ///     .num_bins(10)
        ///     .min_val(0.0)
        ///     .max_val(100.0)
        ///     .build()
        ///     .unwrap();
        ///
        /// // Build reference distribution
        /// for i in 0..1000 {
        #[doc = concat!("    drift.update_reference((i % 100) as ", stringify!($ty), ").unwrap();")]
        /// }
        ///
        /// // Feed live data from same distribution
        /// for i in 0..1000 {
        #[doc = concat!("    drift.update((i % 100) as ", stringify!($ty), ").unwrap();")]
        /// }
        ///
        /// let kl = drift.kl_divergence().unwrap();
        #[doc = concat!("assert!(kl < 0.01 as ", stringify!($ty), ");")]
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            reference: Box<[u64]>,
            live: Box<[u64]>,
            num_bins: usize,
            min_val: $ty,
            max_val: $ty,
            bin_width: $ty,
            ref_total: u64,
            live_total: u64,
            min_samples: u64,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            num_bins: Option<usize>,
            min_val: Option<$ty>,
            max_val: Option<$ty>,
            min_samples: u64,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    num_bins: Option::None,
                    min_val: Option::None,
                    max_val: Option::None,
                    min_samples: 1,
                }
            }

            #[allow(clippy::as_conversions)]
            fn bin_index(&self, sample: $ty) -> usize {
                let frac = (sample - self.min_val) / self.bin_width;
                if frac < (0.0 as $ty) {
                    0
                } else {
                    (frac as usize).min(self.num_bins - 1)
                }
            }

            /// Feeds a sample into the reference histogram.
            ///
            /// # Errors
            ///
            /// Returns `DataError::NotANumber` if the sample is NaN, or
            /// `DataError::Infinite` if the sample is infinite.
            #[inline]
            pub fn update_reference(&mut self, sample: $ty) -> Result<(), nexus_stats_core::DataError> {
                check_finite!(sample);
                let idx = self.bin_index(sample);
                self.reference[idx] += 1;
                self.ref_total += 1;
                Ok(())
            }

            /// Feeds a sample into the live histogram.
            ///
            /// # Errors
            ///
            /// Returns `DataError::NotANumber` if the sample is NaN, or
            /// `DataError::Infinite` if the sample is infinite.
            #[inline]
            pub fn update(&mut self, sample: $ty) -> Result<(), nexus_stats_core::DataError> {
                check_finite!(sample);
                let idx = self.bin_index(sample);
                self.live[idx] += 1;
                self.live_total += 1;
                Ok(())
            }

            /// KL divergence: KL(live || reference), in nats.
            ///
            /// Uses Laplace smoothing to avoid log(0). Returns `None` if
            /// either histogram has fewer than `min_samples` observations.
            #[must_use]
            pub fn kl_divergence(&self) -> Option<$ty> {
                if !self.is_primed() {
                    return Option::None;
                }
                let smooth = 1.0 as $ty;
                let n = self.num_bins as $ty;
                let p_total = self.live_total as $ty + smooth * n;
                let q_total = self.ref_total as $ty + smooth * n;

                let mut kl = 0.0 as $ty;
                for i in 0..self.num_bins {
                    let p = (self.live[i] as $ty + smooth) / p_total;
                    let q = (self.reference[i] as $ty + smooth) / q_total;
                    kl += p * $ln(p / q);
                }
                Option::Some(kl)
            }

            /// Jensen-Shannon divergence, bounded in [0, ln2].
            ///
            /// Symmetric: JS(live, reference) = JS(reference, live).
            /// Returns `None` if not primed.
            #[must_use]
            #[allow(clippy::suboptimal_flops)]
            pub fn js_divergence(&self) -> Option<$ty> {
                if !self.is_primed() {
                    return Option::None;
                }
                let smooth = 1.0 as $ty;
                let n = self.num_bins as $ty;
                let p_total = self.live_total as $ty + smooth * n;
                let q_total = self.ref_total as $ty + smooth * n;

                let mut js = 0.0 as $ty;
                for i in 0..self.num_bins {
                    let p = (self.live[i] as $ty + smooth) / p_total;
                    let q = (self.reference[i] as $ty + smooth) / q_total;
                    let m = 0.5 as $ty * (p + q);
                    js += 0.5 as $ty * p * $ln(p / m)
                        + 0.5 as $ty * q * $ln(q / m);
                }
                Option::Some(js)
            }

            /// Wasserstein-1 (earth mover's) distance.
            ///
            /// Returns `None` if not primed.
            #[must_use]
            #[allow(clippy::suboptimal_flops)]
            pub fn wasserstein1(&self) -> Option<$ty> {
                if !self.is_primed() {
                    return Option::None;
                }
                let mut cdf_p = 0.0 as $ty;
                let mut cdf_q = 0.0 as $ty;
                let mut w1 = 0.0 as $ty;
                let p_total = self.live_total as $ty;
                let q_total = self.ref_total as $ty;

                for i in 0..self.num_bins {
                    cdf_p += self.live[i] as $ty / p_total;
                    cdf_q += self.reference[i] as $ty / q_total;
                    let diff = cdf_p - cdf_q;
                    w1 += (if diff < 0.0 as $ty { -diff } else { diff }) * self.bin_width;
                }
                Option::Some(w1)
            }

            /// Number of histogram bins.
            #[inline]
            #[must_use]
            pub fn num_bins(&self) -> usize {
                self.num_bins
            }

            /// Minimum value of the histogram range.
            #[inline]
            #[must_use]
            pub fn min_val(&self) -> $ty {
                self.min_val
            }

            /// Maximum value of the histogram range.
            #[inline]
            #[must_use]
            pub fn max_val(&self) -> $ty {
                self.max_val
            }

            /// Live sample count.
            #[inline]
            #[must_use]
            pub fn count(&self) -> u64 {
                self.live_total
            }

            /// Reference sample count.
            #[inline]
            #[must_use]
            pub fn reference_count(&self) -> u64 {
                self.ref_total
            }

            /// Whether both histograms have at least `min_samples` observations.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                self.ref_total >= self.min_samples && self.live_total >= self.min_samples
            }

            /// Resets the reference histogram.
            #[inline]
            pub fn reset_reference(&mut self) {
                for bin in &mut *self.reference {
                    *bin = 0;
                }
                self.ref_total = 0;
            }

            /// Resets the live histogram.
            #[inline]
            pub fn reset_live(&mut self) {
                for bin in &mut *self.live {
                    *bin = 0;
                }
                self.live_total = 0;
            }

            /// Resets both histograms.
            #[inline]
            pub fn reset(&mut self) {
                self.reset_reference();
                self.reset_live();
            }
        }

        impl $builder {
            /// Number of histogram bins (required, >= 2).
            #[inline]
            #[must_use]
            pub fn num_bins(mut self, n: usize) -> Self {
                self.num_bins = Option::Some(n);
                self
            }

            /// Minimum value of the histogram range (required, finite).
            #[inline]
            #[must_use]
            pub fn min_val(mut self, v: $ty) -> Self {
                self.min_val = Option::Some(v);
                self
            }

            /// Maximum value of the histogram range (required, finite, > min_val).
            #[inline]
            #[must_use]
            pub fn max_val(mut self, v: $ty) -> Self {
                self.max_val = Option::Some(v);
                self
            }

            /// Minimum samples in each histogram before divergence queries
            /// return values. Default: 1.
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, n: u64) -> Self {
                self.min_samples = n;
                self
            }

            /// Builds the distribution drift tracker.
            ///
            /// # Errors
            ///
            /// Returns `ConfigError` if required fields are missing, bins < 2,
            /// or min_val >= max_val.
            pub fn build(self) -> Result<$name, nexus_stats_core::ConfigError> {
                let num_bins = self
                    .num_bins
                    .ok_or(nexus_stats_core::ConfigError::Missing("num_bins"))?;
                if num_bins < 2 {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "num_bins must be >= 2",
                    ));
                }
                let min_val = self
                    .min_val
                    .ok_or(nexus_stats_core::ConfigError::Missing("min_val"))?;
                if !min_val.is_finite() {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "min_val must be finite",
                    ));
                }
                let max_val = self
                    .max_val
                    .ok_or(nexus_stats_core::ConfigError::Missing("max_val"))?;
                if !max_val.is_finite() {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "max_val must be finite",
                    ));
                }
                if max_val <= min_val {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "max_val must be > min_val",
                    ));
                }

                let bin_width = (max_val - min_val) / num_bins as $ty;

                Ok($name {
                    reference: alloc::vec![0u64; num_bins].into_boxed_slice(),
                    live: alloc::vec![0u64; num_bins].into_boxed_slice(),
                    num_bins,
                    min_val,
                    max_val,
                    bin_width,
                    ref_total: 0,
                    live_total: 0,
                    min_samples: self.min_samples,
                })
            }
        }
    };
}

impl_dist_drift!(
    DistDriftF64,
    DistDriftF64Builder,
    f64,
    nexus_stats_core::math::ln
);
impl_dist_drift!(
    DistDriftF32,
    DistDriftF32Builder,
    f32,
    nexus_stats_core::math::ln_f32
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_distributions_zero_divergence() {
        let mut drift = DistDriftF64::builder()
            .num_bins(10)
            .min_val(0.0)
            .max_val(100.0)
            .build()
            .unwrap();
        for i in 0..1000u64 {
            drift.update_reference((i % 100) as f64).unwrap();
            drift.update((i % 100) as f64).unwrap();
        }
        let kl = drift.kl_divergence().unwrap();
        let js = drift.js_divergence().unwrap();
        let w1 = drift.wasserstein1().unwrap();
        assert!(kl.abs() < 1e-10, "KL should be ~0, got {kl}");
        assert!(js.abs() < 1e-10, "JS should be ~0, got {js}");
        assert!(w1.abs() < 1e-10, "W1 should be ~0, got {w1}");
    }

    #[test]
    fn uniform_vs_concentrated() {
        let mut drift = DistDriftF64::builder()
            .num_bins(10)
            .min_val(0.0)
            .max_val(100.0)
            .build()
            .unwrap();
        for i in 0..1000 {
            drift.update_reference((i % 100) as f64).unwrap();
        }
        for _ in 0..1000 {
            drift.update(50.0).unwrap();
        }
        let kl = drift.kl_divergence().unwrap();
        let js = drift.js_divergence().unwrap();
        assert!(
            kl > 1.0,
            "KL should be large for concentrated vs uniform, got {kl}"
        );
        assert!(js > 0.1, "JS should be significant, got {js}");
    }

    #[test]
    fn js_bounded() {
        let mut drift = DistDriftF64::builder()
            .num_bins(10)
            .min_val(0.0)
            .max_val(100.0)
            .build()
            .unwrap();
        for _ in 0..500 {
            drift.update_reference(10.0).unwrap();
        }
        for _ in 0..500 {
            drift.update(90.0).unwrap();
        }
        let js = drift.js_divergence().unwrap();
        let ln2 = nexus_stats_core::math::ln(2.0);
        assert!(js >= 0.0, "JS should be non-negative, got {js}");
        assert!(js <= ln2 + 1e-10, "JS should be <= ln(2) ≈ {ln2}, got {js}");
    }

    #[test]
    fn js_symmetric() {
        let mut drift_ab = DistDriftF64::builder()
            .num_bins(10)
            .min_val(0.0)
            .max_val(100.0)
            .build()
            .unwrap();
        let mut drift_ba = DistDriftF64::builder()
            .num_bins(10)
            .min_val(0.0)
            .max_val(100.0)
            .build()
            .unwrap();
        for i in 0..500 {
            let a = (i % 50) as f64;
            let b = (i % 100) as f64;
            drift_ab.update_reference(a).unwrap();
            drift_ab.update(b).unwrap();
            drift_ba.update_reference(b).unwrap();
            drift_ba.update(a).unwrap();
        }
        let js_ab = drift_ab.js_divergence().unwrap();
        let js_ba = drift_ba.js_divergence().unwrap();
        assert!(
            (js_ab - js_ba).abs() < 1e-10,
            "JS should be symmetric: {js_ab} vs {js_ba}"
        );
    }

    #[test]
    fn kl_asymmetric() {
        let mut drift_ab = DistDriftF64::builder()
            .num_bins(10)
            .min_val(0.0)
            .max_val(100.0)
            .build()
            .unwrap();
        let mut drift_ba = DistDriftF64::builder()
            .num_bins(10)
            .min_val(0.0)
            .max_val(100.0)
            .build()
            .unwrap();
        for _ in 0..1000 {
            drift_ab.update_reference(50.0).unwrap();
            drift_ba.update(50.0).unwrap();
        }
        for i in 0..1000 {
            drift_ab.update((i % 100) as f64).unwrap();
            drift_ba.update_reference((i % 100) as f64).unwrap();
        }
        let kl_ab = drift_ab.kl_divergence().unwrap();
        let kl_ba = drift_ba.kl_divergence().unwrap();
        assert!(
            (kl_ab - kl_ba).abs() > 0.01,
            "KL should be asymmetric: {kl_ab} vs {kl_ba}"
        );
    }

    #[test]
    fn wasserstein_shifted() {
        let mut drift = DistDriftF64::builder()
            .num_bins(100)
            .min_val(0.0)
            .max_val(100.0)
            .build()
            .unwrap();
        for i in 0..10_000 {
            drift.update_reference((i % 50) as f64).unwrap();
            drift.update(((i % 50) + 10) as f64).unwrap();
        }
        let w1 = drift.wasserstein1().unwrap();
        assert!(
            (w1 - 10.0).abs() < 2.0,
            "W1 should be ≈ 10 for shift=10, got {w1}"
        );
    }

    #[test]
    fn out_of_range_clamped() {
        let mut drift = DistDriftF64::builder()
            .num_bins(5)
            .min_val(0.0)
            .max_val(10.0)
            .build()
            .unwrap();
        drift.update_reference(-100.0).unwrap();
        drift.update_reference(200.0).unwrap();
        drift.update(-50.0).unwrap();
        drift.update(150.0).unwrap();
        assert_eq!(drift.reference_count(), 2);
        assert_eq!(drift.count(), 2);
    }

    #[test]
    fn rejects_nan_inf() {
        let mut drift = DistDriftF64::builder()
            .num_bins(5)
            .min_val(0.0)
            .max_val(10.0)
            .build()
            .unwrap();
        assert!(drift.update(f64::NAN).is_err());
        assert!(drift.update(f64::INFINITY).is_err());
        assert!(drift.update_reference(f64::NAN).is_err());
        assert!(drift.update_reference(f64::NEG_INFINITY).is_err());
        assert_eq!(drift.count(), 0);
        assert_eq!(drift.reference_count(), 0);
    }

    #[test]
    fn reset_clears() {
        let mut drift = DistDriftF64::builder()
            .num_bins(5)
            .min_val(0.0)
            .max_val(10.0)
            .build()
            .unwrap();
        for i in 0..100 {
            drift.update_reference(i as f64 % 10.0).unwrap();
            drift.update(i as f64 % 10.0).unwrap();
        }
        assert_eq!(drift.reference_count(), 100);
        assert_eq!(drift.count(), 100);

        drift.reset_live();
        assert_eq!(drift.count(), 0);
        assert_eq!(drift.reference_count(), 100);

        drift.reset_reference();
        assert_eq!(drift.reference_count(), 0);

        for i in 0..50 {
            drift.update_reference(i as f64 % 10.0).unwrap();
            drift.update(i as f64 % 10.0).unwrap();
        }
        drift.reset();
        assert_eq!(drift.count(), 0);
        assert_eq!(drift.reference_count(), 0);
    }

    #[test]
    fn not_primed_returns_none() {
        let mut drift = DistDriftF64::builder()
            .num_bins(5)
            .min_val(0.0)
            .max_val(10.0)
            .min_samples(100)
            .build()
            .unwrap();
        for i in 0..50 {
            drift.update_reference(i as f64 % 10.0).unwrap();
            drift.update(i as f64 % 10.0).unwrap();
        }
        assert!(!drift.is_primed());
        assert!(drift.kl_divergence().is_none());
        assert!(drift.js_divergence().is_none());
        assert!(drift.wasserstein1().is_none());
    }
}

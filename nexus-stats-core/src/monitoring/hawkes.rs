macro_rules! impl_hawkes {
    ($name:ident, $builder:ident, $ty:ty) => {
        /// Hawkes process intensity estimator.
        ///
        /// Self-exciting point process: each event increases the intensity,
        /// which then decays exponentially. Models bursty arrivals where
        /// events cluster (trade arrivals, order bursts, alert cascades).
        ///
        /// λ(t) = μ + α · Σ exp(-β · (t - t_i))
        ///
        /// Recursive form: O(1) per event.
        ///
        /// # Parameters
        ///
        /// - `mu` (μ) — baseline intensity (events per unit time)
        /// - `alpha` (α) — excitation per event
        /// - `beta` (β) — decay rate (higher = faster decay)
        ///
        /// Stability requires α < β (branching ratio α/β < 1).
        ///
        /// # Examples
        ///
        /// ```
        /// use nexus_stats_core::monitoring::HawkesIntensityF64;
        ///
        /// let mut h = HawkesIntensityF64::builder()
        ///     .mu(1.0)
        ///     .alpha(0.5)
        ///     .beta(1.0)
        ///     .build()
        ///     .unwrap();
        ///
        /// h.update(0);
        /// h.update(100);
        /// assert!(h.intensity() > 1.0); // above baseline after event
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            mu: $ty,
            alpha: $ty,
            beta: $ty,
            excitation: $ty,
            last_time: u64,
            count: u64,
            min_samples: u64,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            mu: Option<$ty>,
            alpha: Option<$ty>,
            beta: Option<$ty>,
            min_samples: u64,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    mu: Option::None,
                    alpha: Option::None,
                    beta: Option::None,
                    min_samples: 2,
                }
            }

            /// Records an event at the given timestamp.
            ///
            /// Timestamps are u64 (matching `Clock::stamp()`). The delta
            /// between timestamps is computed with saturating subtraction.
            #[inline]
            pub fn update(&mut self, time: u64) {
                self.count += 1;

                if self.count == 1 {
                    self.excitation = self.alpha;
                    self.last_time = time;
                    return;
                }

                let dt = time.saturating_sub(self.last_time);
                #[allow(clippy::cast_possible_truncation)]
                let decay = crate::math::exp(-(self.beta * dt as $ty) as f64) as $ty;
                self.excitation = crate::math::MulAdd::fma(decay, self.excitation, self.alpha);
                self.last_time = time;
            }

            /// Current intensity λ at last event time: `μ + excitation`.
            #[inline]
            #[must_use]
            pub fn intensity(&self) -> $ty {
                if self.count == 0 {
                    self.mu
                } else {
                    self.mu + self.excitation
                }
            }

            /// Intensity at an arbitrary time (without recording an event).
            ///
            /// Decays excitation from last event time to `time`.
            #[inline]
            #[must_use]
            pub fn intensity_at(&self, time: u64) -> $ty {
                if self.count == 0 {
                    return self.mu;
                }
                let dt = time.saturating_sub(self.last_time);
                #[allow(clippy::cast_possible_truncation)]
                let decay = crate::math::exp(-(self.beta * dt as $ty) as f64) as $ty;
                crate::math::MulAdd::fma(decay, self.excitation, self.mu)
            }

            /// Baseline intensity μ.
            #[inline]
            #[must_use]
            pub fn baseline(&self) -> $ty {
                self.mu
            }

            /// Branching ratio α/β. Must be < 1 for stability.
            #[inline]
            #[must_use]
            pub fn branching_ratio(&self) -> $ty {
                self.alpha / self.beta
            }

            /// Number of events recorded.
            #[inline]
            #[must_use]
            pub fn count(&self) -> u64 {
                self.count
            }

            /// Whether enough events have been observed.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                self.count >= self.min_samples
            }

            /// Resets to empty state. Parameters unchanged.
            #[inline]
            pub fn reset(&mut self) {
                self.excitation = 0.0 as $ty;
                self.last_time = 0;
                self.count = 0;
            }
        }

        impl $builder {
            /// Baseline intensity μ (required, > 0).
            #[inline]
            #[must_use]
            pub fn mu(mut self, mu: $ty) -> Self {
                self.mu = Option::Some(mu);
                self
            }

            /// Excitation per event α (required, >= 0).
            #[inline]
            #[must_use]
            pub fn alpha(mut self, alpha: $ty) -> Self {
                self.alpha = Option::Some(alpha);
                self
            }

            /// Decay rate β (required, > 0, must be > α for stability).
            #[inline]
            #[must_use]
            pub fn beta(mut self, beta: $ty) -> Self {
                self.beta = Option::Some(beta);
                self
            }

            /// Minimum events before is_primed. Default: 2.
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, min: u64) -> Self {
                self.min_samples = min;
                self
            }

            /// Builds the Hawkes intensity estimator.
            ///
            /// # Errors
            ///
            /// - `mu` must be positive and finite.
            /// - `alpha` must be non-negative and finite.
            /// - `beta` must be positive and finite.
            /// - `alpha` must be < `beta` (branching ratio < 1).
            #[inline]
            pub fn build(self) -> Result<$name, crate::ConfigError> {
                let mu = self.mu.ok_or(crate::ConfigError::Missing("mu"))?;
                if mu <= 0.0 as $ty || !mu.is_finite() {
                    return Err(crate::ConfigError::Invalid(
                        "Hawkes mu must be positive and finite",
                    ));
                }

                let alpha = self.alpha.ok_or(crate::ConfigError::Missing("alpha"))?;
                if alpha < 0.0 as $ty || !alpha.is_finite() {
                    return Err(crate::ConfigError::Invalid(
                        "Hawkes alpha must be non-negative and finite",
                    ));
                }

                let beta = self.beta.ok_or(crate::ConfigError::Missing("beta"))?;
                if beta <= 0.0 as $ty || !beta.is_finite() {
                    return Err(crate::ConfigError::Invalid(
                        "Hawkes beta must be positive and finite",
                    ));
                }

                if alpha >= beta {
                    return Err(crate::ConfigError::Invalid(
                        "Hawkes alpha must be < beta (branching ratio < 1)",
                    ));
                }

                Ok($name {
                    mu,
                    alpha,
                    beta,
                    excitation: 0.0 as $ty,
                    last_time: 0,
                    count: 0,
                    min_samples: self.min_samples,
                })
            }
        }
    };
}

impl_hawkes!(HawkesIntensityF64, HawkesIntensityF64Builder, f64);
impl_hawkes!(HawkesIntensityF32, HawkesIntensityF32Builder, f32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_without_events() {
        let h = HawkesIntensityF64::builder()
            .mu(5.0)
            .alpha(0.5)
            .beta(1.0)
            .build()
            .unwrap();

        assert!((h.intensity() - 5.0).abs() < 1e-10);
        assert!((h.intensity_at(1_000_000) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn excitation_spike() {
        let mut h = HawkesIntensityF64::builder()
            .mu(1.0)
            .alpha(0.5)
            .beta(1.0)
            .build()
            .unwrap();

        h.update(0);
        let after_event = h.intensity();
        assert!(
            after_event > 1.0,
            "intensity should spike above baseline after event, got {after_event}"
        );
    }

    #[test]
    fn decay_over_time() {
        let mut h = HawkesIntensityF64::builder()
            .mu(1.0)
            .alpha(0.5)
            .beta(1.0)
            .build()
            .unwrap();

        h.update(0);
        let far_future = h.intensity_at(1_000_000);
        assert!(
            (far_future - 1.0).abs() < 0.01,
            "intensity should decay to baseline, got {far_future}"
        );
    }

    #[test]
    fn burst_intensifies() {
        let mut h = HawkesIntensityF64::builder()
            .mu(1.0)
            .alpha(0.5)
            .beta(1.0)
            .build()
            .unwrap();

        h.update(0);
        let after_one = h.intensity();

        h.update(1);
        let after_two = h.intensity();

        h.update(2);
        let after_three = h.intensity();

        assert!(
            after_three > after_two && after_two > after_one,
            "rapid events should increase intensity: {after_one} < {after_two} < {after_three}"
        );
    }

    #[test]
    fn branching_ratio_value() {
        let h = HawkesIntensityF64::builder()
            .mu(1.0)
            .alpha(0.3)
            .beta(1.0)
            .build()
            .unwrap();

        assert!((h.branching_ratio() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn stability_validation() {
        let result = HawkesIntensityF64::builder()
            .mu(1.0)
            .alpha(1.0)
            .beta(1.0)
            .build();
        assert!(matches!(result, Err(crate::ConfigError::Invalid(_))));

        let result = HawkesIntensityF64::builder()
            .mu(1.0)
            .alpha(2.0)
            .beta(1.0)
            .build();
        assert!(matches!(result, Err(crate::ConfigError::Invalid(_))));
    }

    #[test]
    fn reset_clears() {
        let mut h = HawkesIntensityF64::builder()
            .mu(1.0)
            .alpha(0.5)
            .beta(1.0)
            .build()
            .unwrap();

        h.update(0);
        h.update(10);
        h.reset();
        assert_eq!(h.count(), 0);
        assert!((h.intensity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn same_timestamp_events() {
        let mut h = HawkesIntensityF64::builder()
            .mu(1.0)
            .alpha(0.5)
            .beta(1.0)
            .build()
            .unwrap();

        h.update(100);
        let after_one = h.intensity();

        h.update(100);
        let after_two = h.intensity();

        assert!(
            after_two > after_one,
            "same-time events should stack: {after_one} < {after_two}"
        );
    }
}

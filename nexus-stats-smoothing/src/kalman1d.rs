use nexus_stats_core::math::MulAdd;
macro_rules! impl_kalman1d {
    ($name:ident, $builder:ident, $ty:ty) => {
        /// 1D Kalman filter with constant-velocity model.
        ///
        /// Tracks position and velocity from noisy measurements.
        /// Automatically balances process noise (system uncertainty) against
        /// measurement noise (sensor uncertainty).
        ///
        /// # Timing assumption
        ///
        /// This filter assumes **dt = 1** between consecutive measurements.
        /// For variable-interval data, either:
        /// - Scale `process_noise` proportionally to the actual interval
        /// - Pre-normalize timestamps so samples arrive at uniform intervals
        ///
        /// # Use Cases
        /// - Smoothing noisy position/latency measurements
        /// - Estimating rate of change (velocity) from noisy data
        /// - Predictive filtering (forecast next value)
        #[derive(Debug, Clone)]
        pub struct $name {
            // State: [position, velocity]
            x0: $ty,
            x1: $ty,
            // Covariance: symmetric 2x2 → 3 values (P00, P01, P11)
            p00: $ty,
            p01: $ty,
            p11: $ty,
            // Noise parameters
            q: $ty, // process noise
            r: $ty, // measurement noise
            count: u64,
            min_samples: u64,
            initialized: bool,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            q: Option<$ty>,
            r: Option<$ty>,
            min_samples: u64,
            seed_pos: Option<$ty>,
            seed_vel: Option<$ty>,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    q: Option::None,
                    r: Option::None,
                    min_samples: 1,
                    seed_pos: Option::None,
                    seed_vel: Option::None,
                }
            }

            /// Feeds a measurement. Returns `(position, velocity)` once primed.
            ///
            /// Assumes dt = 1 between measurements. For variable dt, scale
            /// the process noise or pre-process timestamps.
            ///
            /// # Errors
            ///
            /// Returns `DataError::NotANumber` if the measurement is NaN, or
            /// `DataError::Infinite` if the measurement is infinite.
            #[inline]
            pub fn update(
                &mut self,
                measurement: $ty,
            ) -> Result<Option<($ty, $ty)>, nexus_stats_core::DataError> {
                check_finite!(measurement);
                self.count += 1;

                if !self.initialized {
                    // Initialize from first measurement
                    self.x0 = measurement;
                    self.x1 = 0.0 as $ty;
                    self.p00 = self.r;
                    self.p01 = 0.0 as $ty;
                    self.p11 = 1.0 as $ty;
                    self.initialized = true;

                    return if self.count >= self.min_samples {
                        Ok(Option::Some((self.x0, self.x1)))
                    } else {
                        Ok(Option::None)
                    };
                }

                // Predict step (constant velocity model, dt=1)
                // x_pred = F * x = [x0 + x1, x1]
                let pred_x0 = self.x0 + self.x1;
                let pred_x1 = self.x1;

                // P_pred = F * P * F' + Q
                let pred_p00 = (2.0 as $ty).fma(self.p01, self.p00) + self.p11 + self.q;
                let pred_p01 = self.p01 + self.p11;
                let pred_p11 = self.p11 + self.q;

                // Update step
                // Innovation: y = z - H * x_pred (H = [1, 0])
                let y = measurement - pred_x0;

                // Innovation covariance: S = H * P_pred * H' + R = P00 + R
                let s_inv = (1.0 as $ty) / (pred_p00 + self.r).max(<$ty>::EPSILON);

                // Kalman gain: K = P_pred * H' / S = [P00/S, P01/S]
                let k0 = pred_p00 * s_inv;
                let k1 = pred_p01 * s_inv;

                // State update: x = x_pred + K * y
                self.x0 = k0.fma(y, pred_x0);
                self.x1 = k1.fma(y, pred_x1);

                // Covariance update: P = (I - K*H) * P_pred
                let one_minus_k0 = 1.0 as $ty - k0;
                self.p00 = one_minus_k0 * pred_p00;
                self.p01 = one_minus_k0 * pred_p01;
                self.p11 = pred_p11 - k1 * pred_p01;

                if self.count >= self.min_samples {
                    Ok(Option::Some((self.x0, self.x1)))
                } else {
                    Ok(Option::None)
                }
            }

            /// Estimated position, or `None` if not primed.
            #[inline]
            #[must_use]
            pub fn position(&self) -> Option<$ty> {
                if self.count >= self.min_samples {
                    Option::Some(self.x0)
                } else {
                    Option::None
                }
            }

            /// Estimated velocity, or `None` if not primed.
            #[inline]
            #[must_use]
            pub fn velocity(&self) -> Option<$ty> {
                if self.count >= self.min_samples {
                    Option::Some(self.x1)
                } else {
                    Option::None
                }
            }

            /// Position uncertainty (P00).
            #[inline]
            #[must_use]
            pub fn uncertainty(&self) -> $ty {
                self.p00
            }

            /// Number of measurements processed.
            #[inline]
            #[must_use]
            pub fn count(&self) -> u64 {
                self.count
            }

            /// Whether the filter has reached `min_samples`.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                self.count >= self.min_samples
            }

            /// Resets to uninitialized state.
            #[inline]
            pub fn reset(&mut self) {
                self.x0 = 0.0 as $ty;
                self.x1 = 0.0 as $ty;
                self.p00 = 1.0 as $ty;
                self.p01 = 0.0 as $ty;
                self.p11 = 1.0 as $ty;
                self.count = 0;
                self.initialized = false;
            }
        }

        impl $builder {
            /// Process noise variance. Higher = more reactive to changes.
            ///
            /// The filter assumes dt=1 between samples. For variable-interval
            /// data, scale this value proportionally to the actual interval.
            #[inline]
            #[must_use]
            pub fn process_noise(mut self, q: $ty) -> Self {
                self.q = Option::Some(q);
                self
            }

            /// Measurement noise variance. Higher = smoother output.
            #[inline]
            #[must_use]
            pub fn measurement_noise(mut self, r: $ty) -> Self {
                self.r = Option::Some(r);
                self
            }

            /// Minimum measurements before output is valid. Default: 1.
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, min: u64) -> Self {
                self.min_samples = min;
                self
            }

            /// Pre-load position and velocity from calibration.
            #[inline]
            #[must_use]
            pub fn seed(mut self, position: $ty, velocity: $ty) -> Self {
                self.seed_pos = Option::Some(position);
                self.seed_vel = Option::Some(velocity);
                self
            }

            /// Builds the Kalman filter.
            ///
            /// # Errors
            ///
            /// - process_noise and measurement_noise must have been set.
            /// - Both must be positive.
            #[inline]
            pub fn build(self) -> Result<$name, nexus_stats_core::ConfigError> {
                let q = self
                    .q
                    .ok_or(nexus_stats_core::ConfigError::Missing("process_noise"))?;
                let r = self
                    .r
                    .ok_or(nexus_stats_core::ConfigError::Missing("measurement_noise"))?;
                if q <= 0.0 as $ty {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "process_noise must be positive",
                    ));
                }
                if r <= 0.0 as $ty {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "measurement_noise must be positive",
                    ));
                }

                let (x0, x1, count, initialized) =
                    if let (Some(pos), Some(vel)) = (self.seed_pos, self.seed_vel) {
                        (pos, vel, self.min_samples, true)
                    } else {
                        (0.0 as $ty, 0.0 as $ty, 0, false)
                    };

                Ok($name {
                    x0,
                    x1,
                    p00: 1.0 as $ty,
                    p01: 0.0 as $ty,
                    p11: 1.0 as $ty,
                    q,
                    r,
                    count,
                    min_samples: self.min_samples,
                    initialized,
                })
            }
        }
    };
}

impl_kalman1d!(Kalman1dF64, Kalman1dF64Builder, f64);
impl_kalman1d!(Kalman1dF32, Kalman1dF32Builder, f32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_on_constant() {
        let mut kf = Kalman1dF64::builder()
            .process_noise(0.01)
            .measurement_noise(1.0)
            .build()
            .unwrap();

        for _ in 0..100 {
            kf.update(50.0).unwrap();
        }

        let pos = kf.position().unwrap();
        assert!(
            (pos - 50.0).abs() < 1.0,
            "should converge to ~50, got {pos}"
        );
    }

    #[test]
    fn tracks_linear_trend() {
        let mut kf = Kalman1dF64::builder()
            .process_noise(0.1)
            .measurement_noise(1.0)
            .build()
            .unwrap();

        for i in 0..100 {
            kf.update(i as f64 * 10.0).unwrap();
        }

        let vel = kf.velocity().unwrap();
        assert!(
            (vel - 10.0).abs() < 2.0,
            "velocity should be ~10, got {vel}"
        );
    }

    #[test]
    fn high_process_noise_reactive() {
        let mut reactive = Kalman1dF64::builder()
            .process_noise(10.0)
            .measurement_noise(1.0)
            .build()
            .unwrap();
        let mut smooth = Kalman1dF64::builder()
            .process_noise(0.001)
            .measurement_noise(1.0)
            .build()
            .unwrap();

        for _ in 0..20 {
            reactive.update(100.0).unwrap();
            smooth.update(100.0).unwrap();
        }
        // Both at 100. Now jump.
        reactive.update(200.0).unwrap();
        smooth.update(200.0).unwrap();

        let r_pos = reactive.position().unwrap();
        let s_pos = smooth.position().unwrap();
        assert!(
            r_pos > s_pos,
            "reactive ({r_pos}) should track faster than smooth ({s_pos})"
        );
    }

    #[test]
    fn uncertainty_decreases() {
        let mut kf = Kalman1dF64::builder()
            .process_noise(0.01)
            .measurement_noise(1.0)
            .build()
            .unwrap();

        kf.update(50.0).unwrap();
        let u1 = kf.uncertainty();

        for _ in 0..50 {
            kf.update(50.0).unwrap();
        }
        let u2 = kf.uncertainty();

        assert!(u2 < u1, "uncertainty should decrease, was {u1} now {u2}");
    }

    #[test]
    fn seeded_startup() {
        let kf = Kalman1dF64::builder()
            .process_noise(0.01)
            .measurement_noise(1.0)
            .seed(100.0, 5.0)
            .build()
            .unwrap();

        assert!(kf.is_primed());
        let pos = kf.position().unwrap();
        assert!((pos - 100.0).abs() < 1e-10);
    }

    #[test]
    fn reset() {
        let mut kf = Kalman1dF64::builder()
            .process_noise(0.01)
            .measurement_noise(1.0)
            .build()
            .unwrap();

        for _ in 0..50 {
            kf.update(100.0).unwrap();
        }
        kf.reset();
        assert_eq!(kf.count(), 0);
    }

    #[test]
    fn f32_basic() {
        let mut kf = Kalman1dF32::builder()
            .process_noise(0.1)
            .measurement_noise(1.0)
            .build()
            .unwrap();

        kf.update(50.0).unwrap();
        assert!(kf.position().is_some());
    }

    #[test]
    fn seed_zero_zero_works() {
        let mut kf = Kalman1dF64::builder()
            .process_noise(0.01)
            .measurement_noise(1.0)
            .seed(0.0, 0.0)
            .build()
            .unwrap();

        assert!(kf.is_primed());
        // First update should apply predict+update, not re-initialize
        let (pos, _vel) = kf.update(10.0).unwrap().unwrap();
        assert!(pos > 0.0, "should track toward 10, got {pos}");
    }

    #[test]
    fn errors_without_process_noise() {
        let result = Kalman1dF64::builder().measurement_noise(1.0).build();
        assert!(matches!(
            result,
            Err(nexus_stats_core::ConfigError::Missing("process_noise"))
        ));
    }

    #[test]
    fn rejects_nan_and_inf() {
        let mut kf = Kalman1dF64::builder()
            .process_noise(0.01)
            .measurement_noise(1.0)
            .build()
            .unwrap();
        assert!(matches!(
            kf.update(f64::NAN),
            Err(nexus_stats_core::DataError::NotANumber)
        ));
        assert!(matches!(
            kf.update(f64::INFINITY),
            Err(nexus_stats_core::DataError::Infinite)
        ));
        assert!(matches!(
            kf.update(f64::NEG_INFINITY),
            Err(nexus_stats_core::DataError::Infinite)
        ));
        assert_eq!(kf.count(), 0);
    }
}

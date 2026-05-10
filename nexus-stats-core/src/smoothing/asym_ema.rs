use crate::math::MulAdd;
macro_rules! impl_asym_ema_float {
    ($name:ident, $builder:ident, $ty:ty) => {
        /// Asymmetric EMA — different smoothing factors for rising vs falling.
        ///
        /// Uses `alpha_up` when the new sample exceeds the current value,
        /// `alpha_down` when it's below. This allows fast attack / slow decay
        /// or vice versa.
        ///
        /// # Use Cases
        /// - Fast attack / slow decay for peak tracking
        /// - Slow attack / fast decay for trough tracking
        /// - Asymmetric noise filtering
        #[derive(Debug, Clone)]
        pub struct $name {
            alpha_up: $ty,
            alpha_down: $ty,
            // Precomputed `1.0 - alpha_*` for the `update` hot path; saves a
            // subtraction per sample at the cost of two `$ty` fields per
            // instance. Set in `build()` and held constant for the
            // instance's lifetime.
            one_minus_alpha_up: $ty,
            one_minus_alpha_down: $ty,
            value: $ty,
            count: u64,
            min_samples: u64,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            alpha_up: Option<$ty>,
            alpha_down: Option<$ty>,
            min_samples: u64,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    alpha_up: Option::None,
                    alpha_down: Option::None,
                    min_samples: 1,
                }
            }

            /// Feeds a sample. Returns smoothed value once primed.
            ///
            /// # Errors
            ///
            /// Returns `DataError::NotANumber` if the sample is NaN, or
            /// `DataError::Infinite` if the sample is infinite.
            #[inline]
            pub fn update(&mut self, sample: $ty) -> Result<Option<$ty>, crate::DataError> {
                check_finite!(sample);
                self.count += 1;

                if self.count == 1 {
                    self.value = sample;
                } else {
                    let (alpha, one_minus) = if sample > self.value {
                        (self.alpha_up, self.one_minus_alpha_up)
                    } else {
                        (self.alpha_down, self.one_minus_alpha_down)
                    };
                    self.value = alpha.fma(sample, one_minus * self.value);
                }

                if self.count >= self.min_samples {
                    Ok(Option::Some(self.value))
                } else {
                    Ok(Option::None)
                }
            }

            /// Current smoothed value, or `None` if not primed.
            #[inline]
            #[must_use]
            pub fn value(&self) -> Option<$ty> {
                if self.count >= self.min_samples {
                    Option::Some(self.value)
                } else {
                    Option::None
                }
            }

            /// Number of samples processed.
            #[inline]
            #[must_use]
            pub fn count(&self) -> u64 {
                self.count
            }

            /// Whether the EMA has reached `min_samples`.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                self.count >= self.min_samples
            }

            /// Resets to uninitialized state.
            #[inline]
            pub fn reset(&mut self) {
                self.value = 0.0 as $ty;
                self.count = 0;
            }
        }

        impl $builder {
            /// Smoothing factor when sample > current value.
            #[inline]
            #[must_use]
            pub fn alpha_up(mut self, alpha: $ty) -> Self {
                self.alpha_up = Option::Some(alpha);
                self
            }

            /// Smoothing factor when sample <= current value.
            #[inline]
            #[must_use]
            pub fn alpha_down(mut self, alpha: $ty) -> Self {
                self.alpha_down = Option::Some(alpha);
                self
            }

            /// Span for rising smoothing (alpha_up = 2/(n+1)).
            #[inline]
            #[must_use]
            pub fn span_up(mut self, n: u64) -> Self {
                self.alpha_up = Option::Some(2.0 as $ty / (n as $ty + 1.0 as $ty));
                self
            }

            /// Span for falling smoothing (alpha_down = 2/(n+1)).
            #[inline]
            #[must_use]
            pub fn span_down(mut self, n: u64) -> Self {
                self.alpha_down = Option::Some(2.0 as $ty / (n as $ty + 1.0 as $ty));
                self
            }

            /// Minimum samples before value is valid. Default: 1.
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, min: u64) -> Self {
                self.min_samples = min;
                self
            }

            /// Builds the asymmetric EMA.
            ///
            /// # Errors
            ///
            /// - Both alpha_up and alpha_down must have been set.
            /// - Both must be in (0, 1) exclusive.
            #[inline]
            pub fn build(self) -> Result<$name, crate::ConfigError> {
                let alpha_up = self
                    .alpha_up
                    .ok_or(crate::ConfigError::Missing("alpha_up"))?;
                let alpha_down = self
                    .alpha_down
                    .ok_or(crate::ConfigError::Missing("alpha_down"))?;
                if !(alpha_up > 0.0 as $ty && alpha_up < 1.0 as $ty) {
                    return Err(crate::ConfigError::Invalid("alpha_up must be in (0, 1)"));
                }
                if !(alpha_down > 0.0 as $ty && alpha_down < 1.0 as $ty) {
                    return Err(crate::ConfigError::Invalid("alpha_down must be in (0, 1)"));
                }

                Ok($name {
                    alpha_up,
                    alpha_down,
                    one_minus_alpha_up: 1.0 as $ty - alpha_up,
                    one_minus_alpha_down: 1.0 as $ty - alpha_down,
                    value: 0.0 as $ty,
                    count: 0,
                    min_samples: self.min_samples,
                })
            }
        }
    };
}

macro_rules! impl_asym_ema_int {
    ($name:ident, $builder:ident, $ty:ty, $acc_ty:ty) => {
        /// Asymmetric EMA (integer) — different shift factors for rising vs falling.
        #[derive(Debug, Clone)]
        pub struct $name {
            acc: $acc_ty,
            shift_up: u32,
            shift_down: u32,
            span_up: u64,
            span_down: u64,
            count: u64,
            min_samples: u64,
            initialized: bool,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            span_up: Option<u64>,
            span_down: Option<u64>,
            min_samples: u64,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    span_up: Option::None,
                    span_down: Option::None,
                    min_samples: 1,
                }
            }

            /// Feeds a sample. Returns smoothed value once primed.
            #[inline]
            #[must_use]
            pub fn update(&mut self, sample: $ty) -> Option<$ty> {
                self.count += 1;

                if !self.initialized {
                    // Use the larger shift for initial accumulator
                    let shift = self.shift_up.max(self.shift_down);
                    self.acc = (sample as $acc_ty) << shift;
                    self.initialized = true;
                } else {
                    let current = (self.acc >> self.shift_up.max(self.shift_down)) as $ty;
                    let shift = if sample > current {
                        self.shift_up
                    } else {
                        self.shift_down
                    };
                    let sample_shifted = (sample as $acc_ty) << shift;
                    // Normalize accumulator to the active shift
                    let acc_at_shift = if shift == self.shift_up {
                        self.acc >> (self.shift_up.max(self.shift_down) - shift)
                    } else {
                        self.acc >> (self.shift_up.max(self.shift_down) - shift)
                    };
                    let new_acc = acc_at_shift + ((sample_shifted - acc_at_shift) >> shift);
                    self.acc = new_acc << (self.shift_up.max(self.shift_down) - shift);
                }

                if self.count >= self.min_samples {
                    let shift = self.shift_up.max(self.shift_down);
                    Option::Some((self.acc >> shift) as $ty)
                } else {
                    Option::None
                }
            }

            /// Current smoothed value, or `None` if not primed.
            #[inline]
            #[must_use]
            pub fn value(&self) -> Option<$ty> {
                if self.count >= self.min_samples && self.initialized {
                    let shift = self.shift_up.max(self.shift_down);
                    Option::Some((self.acc >> shift) as $ty)
                } else {
                    Option::None
                }
            }

            /// Effective spans after rounding.
            #[inline]
            #[must_use]
            pub fn effective_span_up(&self) -> u64 {
                self.span_up
            }

            /// Effective span for falling direction.
            #[inline]
            #[must_use]
            pub fn effective_span_down(&self) -> u64 {
                self.span_down
            }

            /// Number of samples processed.
            #[inline]
            #[must_use]
            pub fn count(&self) -> u64 {
                self.count
            }

            /// Whether the EMA has reached `min_samples`.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                self.count >= self.min_samples
            }

            /// Resets to uninitialized state.
            #[inline]
            pub fn reset(&mut self) {
                self.acc = 0;
                self.count = 0;
                self.initialized = false;
            }
        }

        impl $builder {
            /// Span for rising direction. Rounded up to next `2^k - 1`.
            #[inline]
            #[must_use]
            pub fn span_up(mut self, n: u64) -> Self {
                self.span_up = Option::Some(n);
                self
            }

            /// Span for falling direction. Rounded up to next `2^k - 1`.
            #[inline]
            #[must_use]
            pub fn span_down(mut self, n: u64) -> Self {
                self.span_down = Option::Some(n);
                self
            }

            /// Minimum samples before value is valid. Default: 1.
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, min: u64) -> Self {
                self.min_samples = min;
                self
            }

            /// Builds the asymmetric EMA.
            ///
            /// # Errors
            ///
            /// - Both span_up and span_down must have been set and >= 1.
            #[inline]
            pub fn build(self) -> Result<$name, crate::ConfigError> {
                let req_up = self.span_up.ok_or(crate::ConfigError::Missing("span_up"))?;
                let req_down = self
                    .span_down
                    .ok_or(crate::ConfigError::Missing("span_down"))?;
                if req_up < 1 {
                    return Err(crate::ConfigError::Invalid("span_up must be >= 1"));
                }
                if req_down < 1 {
                    return Err(crate::ConfigError::Invalid("span_down must be >= 1"));
                }

                let eff_up = crate::smoothing::ema::next_power_of_two_minus_one(req_up);
                let eff_down = crate::smoothing::ema::next_power_of_two_minus_one(req_down);

                Ok($name {
                    acc: 0,
                    shift_up: crate::smoothing::ema::log2_of_span_plus_one(eff_up),
                    shift_down: crate::smoothing::ema::log2_of_span_plus_one(eff_down),
                    span_up: eff_up,
                    span_down: eff_down,
                    count: 0,
                    min_samples: self.min_samples,
                    initialized: false,
                })
            }
        }
    };
}

impl_asym_ema_float!(AsymEmaF64, AsymEmaF64Builder, f64);
impl_asym_ema_float!(AsymEmaF32, AsymEmaF32Builder, f32);
impl_asym_ema_int!(AsymEmaI64, AsymEmaI64Builder, i64, i128);
impl_asym_ema_int!(AsymEmaI32, AsymEmaI32Builder, i32, i64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_attack_slow_decay() {
        let mut ema = AsymEmaF64::builder()
            .alpha_up(0.9) // fast attack
            .alpha_down(0.1) // slow decay
            .build()
            .unwrap();

        ema.update(0.0).unwrap(); // initialize
        ema.update(100.0).unwrap(); // fast attack
        let after_attack = ema.value().unwrap();

        ema.update(0.0).unwrap(); // slow decay
        let after_decay = ema.value().unwrap();

        // Attack should move a lot, decay should move little
        assert!(
            after_attack > 50.0,
            "fast attack should jump, got {after_attack}"
        );
        assert!(
            after_decay > 30.0,
            "slow decay should hold, got {after_decay}"
        );
    }

    #[test]
    fn asymmetric_response() {
        let mut fast_up = AsymEmaF64::builder()
            .alpha_up(0.9)
            .alpha_down(0.1)
            .build()
            .unwrap();
        let mut fast_down = AsymEmaF64::builder()
            .alpha_up(0.1)
            .alpha_down(0.9)
            .build()
            .unwrap();

        fast_up.update(50.0).unwrap();
        fast_down.update(50.0).unwrap();

        fast_up.update(100.0).unwrap();
        fast_down.update(100.0).unwrap();

        // fast_up should be closer to 100
        assert!(fast_up.value().unwrap() > fast_down.value().unwrap());
    }

    #[test]
    fn priming() {
        let mut ema = AsymEmaF64::builder()
            .alpha_up(0.5)
            .alpha_down(0.5)
            .min_samples(5)
            .build()
            .unwrap();

        for _ in 0..4 {
            assert!(ema.update(100.0).unwrap().is_none());
        }
        assert!(ema.update(100.0).unwrap().is_some());
    }

    #[test]
    fn reset() {
        let mut ema = AsymEmaF64::builder()
            .alpha_up(0.5)
            .alpha_down(0.5)
            .build()
            .unwrap();
        ema.update(100.0).unwrap();
        ema.reset();
        assert_eq!(ema.count(), 0);
        assert!(ema.value().is_none());
    }

    #[test]
    fn i64_basic() {
        let mut ema = AsymEmaI64::builder()
            .span_up(3)
            .span_down(7)
            .build()
            .unwrap();

        let _ = ema.update(100);
        let _ = ema.update(200);
        assert!(ema.value().is_some());
    }

    #[test]
    fn f32_basic() {
        let mut ema = AsymEmaF32::builder()
            .alpha_up(0.5)
            .alpha_down(0.3)
            .build()
            .unwrap();

        assert!(ema.update(100.0).unwrap().is_some());
    }

    #[test]
    fn errors_without_alpha_up() {
        let result = AsymEmaF64::builder().alpha_down(0.5).build();
        assert!(matches!(
            result,
            Err(crate::ConfigError::Missing("alpha_up"))
        ));
    }

    #[test]
    fn rejects_nan_and_inf() {
        let mut ema = AsymEmaF64::builder()
            .alpha_up(0.5)
            .alpha_down(0.3)
            .build()
            .unwrap();
        assert!(matches!(
            ema.update(f64::NAN),
            Err(crate::DataError::NotANumber)
        ));
        assert!(matches!(
            ema.update(f64::INFINITY),
            Err(crate::DataError::Infinite)
        ));
        assert!(matches!(
            ema.update(f64::NEG_INFINITY),
            Err(crate::DataError::Infinite)
        ));
        assert_eq!(ema.count(), 0);
    }
}

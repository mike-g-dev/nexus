use crate::math::MulAdd;
macro_rules! impl_jitter_float {
    ($name:ident, $builder:ident, $ty:ty) => {
        /// Jitter tracker — smoothed absolute deviation between consecutive samples.
        ///
        /// Internally tracks an EMA of absolute consecutive deltas and an EMA of
        /// values for computing the jitter ratio.
        ///
        /// # Use Cases
        /// - Network jitter (variation in inter-packet delay)
        /// - Latency jitter (variation in response times)
        /// - Clock stability monitoring
        #[derive(Debug, Clone)]
        pub struct $name {
            alpha: $ty,
            one_minus_alpha: $ty,
            jitter: $ty,
            mean: $ty,
            last_sample: $ty,
            last_deviation: $ty,
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
            seed_value: Option<$ty>,
            seed_jitter: Option<$ty>,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    alpha: Option::None,
                    min_samples: 2,
                    seed_value: Option::None,
                    seed_jitter: Option::None,
                }
            }

            /// Feeds a sample. Returns smoothed jitter once primed.
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
                    self.last_sample = sample;
                    self.mean = sample;
                    return Ok(Option::None);
                }

                let abs_delta = (sample - self.last_sample).abs();
                self.last_deviation = abs_delta;
                self.last_sample = sample;

                if self.count == 2 {
                    self.jitter = abs_delta;
                    self.mean = self.alpha.fma(sample, self.one_minus_alpha * self.mean);
                } else {
                    self.jitter = self
                        .alpha
                        .fma(abs_delta, self.one_minus_alpha * self.jitter);
                    self.mean = self.alpha.fma(sample, self.one_minus_alpha * self.mean);
                }

                if self.count >= self.min_samples {
                    Ok(Option::Some(self.jitter))
                } else {
                    Ok(Option::None)
                }
            }

            /// Current smoothed jitter (absolute deviation), or `None` if not primed.
            #[inline]
            #[must_use]
            pub fn jitter(&self) -> Option<$ty> {
                if self.count >= self.min_samples {
                    Option::Some(self.jitter)
                } else {
                    Option::None
                }
            }

            /// Jitter as a fraction of the smoothed mean, or `None` if not primed
            /// or mean is near zero (absolute value < epsilon).
            #[inline]
            #[must_use]
            pub fn jitter_ratio(&self) -> Option<$ty> {
                if self.count >= self.min_samples && self.mean.abs() > <$ty>::EPSILON {
                    Option::Some(self.jitter / self.mean)
                } else {
                    Option::None
                }
            }

            /// Raw absolute deviation of the last two samples, or `None` if < 2 samples.
            #[inline]
            #[must_use]
            pub fn last_deviation(&self) -> Option<$ty> {
                if self.count >= 2 {
                    Option::Some(self.last_deviation)
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

            /// Whether enough data has been collected.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                self.count >= self.min_samples
            }

            /// Resets to empty state. Parameters unchanged.
            #[inline]
            pub fn reset(&mut self) {
                self.jitter = 0.0 as $ty;
                self.mean = 0.0 as $ty;
                self.last_sample = 0.0 as $ty;
                self.last_deviation = 0.0 as $ty;
                self.count = 0;
            }
        }

        impl $builder {
            /// Smoothing factor for jitter EMA.
            #[inline]
            #[must_use]
            pub fn alpha(mut self, alpha: $ty) -> Self {
                self.alpha = Option::Some(alpha);
                self
            }

            /// Halflife for jitter smoothing.
            #[inline]
            #[must_use]
            #[cfg(any(feature = "std", feature = "libm"))]
            pub fn halflife(mut self, halflife: $ty) -> Self {
                let ln2 = core::f64::consts::LN_2 as $ty;
                self.alpha =
                    Option::Some(1.0 as $ty - crate::math::exp((-ln2 / halflife) as f64) as $ty);
                self
            }

            /// Span for jitter smoothing.
            #[inline]
            #[must_use]
            pub fn span(mut self, n: u64) -> Self {
                self.alpha = Option::Some(2.0 as $ty / (n as $ty + 1.0 as $ty));
                self
            }

            /// Minimum samples before jitter is valid. Default: 2.
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, min: u64) -> Self {
                self.min_samples = min;
                self
            }

            /// Pre-loads the last sample value and smoothed jitter from calibration data.
            ///
            /// When seeded, `is_primed()` returns true immediately and the
            /// next `update()` computes a deviation against `value`.
            #[inline]
            #[must_use]
            pub fn seed(mut self, value: $ty, jitter: $ty) -> Self {
                self.seed_value = Option::Some(value);
                self.seed_jitter = Option::Some(jitter);
                self
            }

            /// Builds the jitter tracker.
            ///
            /// # Errors
            ///
            /// - Alpha must have been set.
            /// - Alpha must be in (0, 1) exclusive.
            #[inline]
            pub fn build(self) -> Result<$name, crate::ConfigError> {
                let alpha = self.alpha.ok_or(crate::ConfigError::Missing("alpha"))?;
                if !(alpha > 0.0 as $ty && alpha < 1.0 as $ty) {
                    return Err(crate::ConfigError::Invalid(
                        "Jitter alpha must be in (0, 1)",
                    ));
                }

                let (last_sample, jitter, mean, count) = match (self.seed_value, self.seed_jitter) {
                    (Some(v), Some(j)) => (v, j, v, self.min_samples),
                    _ => (0.0 as $ty, 0.0 as $ty, 0.0 as $ty, 0),
                };

                Ok($name {
                    alpha,
                    one_minus_alpha: 1.0 as $ty - alpha,
                    jitter,
                    mean,
                    last_sample,
                    last_deviation: 0.0 as $ty,
                    count,
                    min_samples: self.min_samples,
                })
            }
        }
    };
}

impl_jitter_float!(JitterF64, JitterF64Builder, f64);
impl_jitter_float!(JitterF32, JitterF32Builder, f32);

macro_rules! impl_jitter_int {
    ($name:ident, $builder:ident, $ty:ty, $acc_ty:ty) => {
        /// Jitter tracker (integer variant) — fixed-point EMA of absolute deltas.
        ///
        /// Uses kernel-style bit-shift arithmetic. `jitter_ratio()` is not
        /// available on integer types (integer division loses too much precision).
        #[derive(Debug, Clone)]
        pub struct $name {
            acc: $acc_ty,
            shift: u32,
            span: u64,
            last_sample: $ty,
            last_deviation: $ty,
            count: u64,
            min_samples: u64,
            initialized: bool,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            span: Option<u64>,
            min_samples: u64,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    span: Option::None,
                    min_samples: 2,
                }
            }

            /// Feeds a sample. Returns smoothed jitter once primed.
            #[inline]
            #[must_use]
            pub fn update(&mut self, sample: $ty) -> Option<$ty> {
                self.count += 1;

                if self.count == 1 {
                    self.last_sample = sample;
                    return Option::None;
                }

                let abs_delta = (sample - self.last_sample).abs();
                self.last_deviation = abs_delta;
                self.last_sample = sample;

                if !self.initialized {
                    self.acc = (abs_delta as $acc_ty) << self.shift;
                    self.initialized = true;
                } else {
                    let delta_shifted = (abs_delta as $acc_ty) << self.shift;
                    self.acc += (delta_shifted - self.acc) >> self.shift;
                }

                if self.count >= self.min_samples {
                    Option::Some((self.acc >> self.shift) as $ty)
                } else {
                    Option::None
                }
            }

            /// Current smoothed jitter, or `None` if not primed.
            #[inline]
            #[must_use]
            pub fn jitter(&self) -> Option<$ty> {
                if self.count >= self.min_samples && self.initialized {
                    Option::Some((self.acc >> self.shift) as $ty)
                } else {
                    Option::None
                }
            }

            /// Raw absolute deviation of the last two samples, or `None` if < 2.
            #[inline]
            #[must_use]
            pub fn last_deviation(&self) -> Option<$ty> {
                if self.count >= 2 {
                    Option::Some(self.last_deviation)
                } else {
                    Option::None
                }
            }

            /// Effective span after rounding.
            #[inline]
            #[must_use]
            pub fn effective_span(&self) -> u64 {
                self.span
            }

            /// Number of samples processed.
            #[inline]
            #[must_use]
            pub fn count(&self) -> u64 {
                self.count
            }

            /// Whether enough data has been collected.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                self.count >= self.min_samples
            }

            /// Resets to empty state.
            #[inline]
            pub fn reset(&mut self) {
                self.acc = 0;
                self.last_sample = 0;
                self.last_deviation = 0;
                self.count = 0;
                self.initialized = false;
            }
        }

        impl $builder {
            /// Smoothing span. Rounded up to next `2^k - 1`.
            #[inline]
            #[must_use]
            pub fn span(mut self, n: u64) -> Self {
                self.span = Option::Some(n);
                self
            }

            /// Minimum samples before jitter is valid. Default: 2.
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, min: u64) -> Self {
                self.min_samples = min;
                self
            }

            /// Builds the jitter tracker.
            ///
            /// # Errors
            ///
            /// - Span must have been set and >= 1.
            #[inline]
            pub fn build(self) -> Result<$name, crate::ConfigError> {
                let requested = self.span.ok_or(crate::ConfigError::Missing("span"))?;
                if requested < 1 {
                    return Err(crate::ConfigError::Invalid("Jitter span must be >= 1"));
                }

                let effective = crate::smoothing::ema::next_power_of_two_minus_one(requested);
                let shift = crate::smoothing::ema::log2_of_span_plus_one(effective);

                Ok($name {
                    acc: 0,
                    shift,
                    span: effective,
                    last_sample: 0,
                    last_deviation: 0,
                    count: 0,
                    min_samples: self.min_samples,
                    initialized: false,
                })
            }
        }
    };
}

impl_jitter_int!(JitterI64, JitterI64Builder, i64, i128);
impl_jitter_int!(JitterI32, JitterI32Builder, i32, i64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn constant_input_zero_jitter() {
        let mut j = JitterF64::builder().alpha(0.3).build().unwrap();
        for _ in 0..100 {
            let _ = j.update(100.0).unwrap();
        }
        let jitter = j.jitter().unwrap();
        assert!(
            jitter.abs() < 1e-10,
            "constant input should have ~zero jitter, got {jitter}"
        );
    }

    #[test]
    fn alternating_input_high_jitter() {
        let mut j = JitterF64::builder().alpha(0.5).build().unwrap();
        for i in 0..50 {
            let _ = j.update(if i % 2 == 0 { 100.0 } else { 200.0 }).unwrap();
        }
        let jitter = j.jitter().unwrap();
        assert!(
            jitter > 50.0,
            "alternating input should have high jitter, got {jitter}"
        );
    }

    #[test]
    fn jitter_ratio_correctness() {
        let mut j = JitterF64::builder().alpha(0.3).build().unwrap();
        for i in 0..100 {
            let _ = j.update(100.0 + (i % 10) as f64).unwrap();
        }
        let ratio = j.jitter_ratio().unwrap();
        assert!(
            ratio > 0.0 && ratio < 1.0,
            "ratio should be reasonable, got {ratio}"
        );
    }

    #[test]
    fn priming() {
        let mut j = JitterF64::builder()
            .alpha(0.3)
            .min_samples(5)
            .build()
            .unwrap();
        for _ in 0..4 {
            assert!(j.update(100.0).unwrap().is_none());
        }
        assert!(j.update(100.0).unwrap().is_some());
    }

    #[test]
    fn reset() {
        let mut j = JitterF64::builder().alpha(0.3).build().unwrap();
        for _ in 0..10 {
            let _ = j.update(100.0).unwrap();
        }
        j.reset();
        assert_eq!(j.count(), 0);
        assert!(j.jitter().is_none());
    }

    #[test]
    fn i64_basic() {
        let mut j = JitterI64::builder().span(7).build().unwrap();
        let _ = j.update(100);
        let _ = j.update(110);
        let _ = j.update(105);
        assert!(j.jitter().is_some());
    }

    #[test]
    fn i32_basic() {
        let mut j = JitterI32::builder().span(3).build().unwrap();
        let _ = j.update(50);
        let _ = j.update(60);
        assert!(j.jitter().is_some());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn f32_basic() {
        let mut j = JitterF32::builder().alpha(0.5).build().unwrap();
        let _ = j.update(100.0).unwrap();
        let _ = j.update(110.0).unwrap();
        assert_eq!(j.last_deviation(), Some(10.0));
    }

    #[test]
    fn seeded_is_primed() {
        let j = JitterF64::builder()
            .alpha(0.3)
            .seed(100.0, 5.0)
            .build()
            .unwrap();

        assert!(j.is_primed());
        assert!((j.jitter().unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn seeded_next_update_uses_seed_value() {
        let mut j = JitterF64::builder()
            .alpha(0.3)
            .seed(100.0, 5.0)
            .build()
            .unwrap();

        // Next update should compute deviation from seeded last_sample=100
        let result = j.update(110.0).unwrap();
        assert!(result.is_some());
        // Deviation is |110-100|=10, smoothed jitter = 0.3*10 + 0.7*5 = 6.5
        let jitter = result.unwrap();
        assert!((jitter - 6.5).abs() < 1e-10);
    }

    #[test]
    fn errors_without_alpha() {
        let result = JitterF64::builder().build();
        assert!(matches!(result, Err(crate::ConfigError::Missing("alpha"))));
    }

    #[test]
    fn rejects_nan_and_inf() {
        let mut j = JitterF64::builder().alpha(0.3).build().unwrap();
        assert!(matches!(
            j.update(f64::NAN),
            Err(crate::DataError::NotANumber)
        ));
        assert!(matches!(
            j.update(f64::INFINITY),
            Err(crate::DataError::Infinite)
        ));
        assert!(matches!(
            j.update(f64::NEG_INFINITY),
            Err(crate::DataError::Infinite)
        ));

        let mut j32 = JitterF32::builder().alpha(0.3).build().unwrap();
        assert!(matches!(
            j32.update(f32::NAN),
            Err(crate::DataError::NotANumber)
        ));
    }
}

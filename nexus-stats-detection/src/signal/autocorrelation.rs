// Checking m2 == 0.0 is intentional: zero variance means all samples
// are identical, and correlation is undefined. This is exact, not approximate.
#![allow(clippy::float_cmp)]

// Online Autocorrelation at Fixed Lag
//
// Maintains a circular buffer of size `lag` for delayed values, plus
// Welford-style accumulators for variance and the cross-moment
// between x(t) and x(t-lag).
//
// r(k) = cross_m / m2 — the 1/N normalization cancels.

extern crate alloc;
use alloc::boxed::Box;
use alloc::vec;

/// Online autocorrelation at a configurable lag.
///
/// Maintains a circular buffer of previous values and computes
/// the autocorrelation coefficient between x(t) and x(t-lag) using
/// Welford-style running accumulators.
///
/// # Use Cases
/// - "Is this signal trending or mean-reverting?" (positive vs negative lag-1)
/// - Detecting periodicity at a known lag
/// - Stationarity monitoring
///
/// # Complexity
/// - O(1) per update, heap-allocated circular buffer.
///
/// # Examples
///
/// ```
/// use nexus_stats_detection::signal::AutocorrelationF64;
///
/// // Strongly periodic signal: lag-1 autocorrelation of alternating values
/// let mut ac = AutocorrelationF64::builder().lag(1).build().unwrap();
/// for i in 0..200u64 {
///     ac.update(if i % 2 == 0 { 1.0 } else { -1.0 }).unwrap();
/// }
/// let r = ac.correlation().unwrap();
/// assert!(r < -0.9, "alternating signal should have negative lag-1 autocorrelation");
/// ```
#[derive(Debug, Clone)]
pub struct AutocorrelationF64 {
    buffer: Box<[f64]>,
    lag: usize,
    head: usize,
    count: u64,
    mean: f64,
    m2: f64,
    cross_m: f64,
}

/// Builder for [`AutocorrelationF64`].
#[derive(Debug, Clone)]
pub struct AutocorrelationF64Builder {
    lag: Option<usize>,
}

impl AutocorrelationF64 {
    /// Creates a builder.
    #[inline]
    #[must_use]
    pub fn builder() -> AutocorrelationF64Builder {
        AutocorrelationF64Builder { lag: None }
    }

    /// Feeds a sample.
    ///
    /// # Errors
    ///
    /// Returns `DataError::NotANumber` if the sample is NaN, or
    /// `DataError::Infinite` if the sample is infinite.
    #[inline]
    pub fn update(&mut self, sample: f64) -> Result<(), nexus_stats_core::DataError> {
        check_finite!(sample);
        self.count += 1;

        // Welford update for running mean and variance
        let delta = sample - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = sample - self.mean;
        self.m2 += delta * delta2;

        // Cross-moment: accumulate (x_new - mean)(x_lagged - mean)
        // once we have at least lag+1 observations
        if self.count > self.lag as u64 {
            let x_lagged = self.buffer[self.head];
            self.cross_m += (sample - self.mean) * (x_lagged - self.mean);
        }

        // Store in circular buffer
        self.buffer[self.head] = sample;
        self.head = (self.head + 1) % self.lag;
        Ok(())
    }

    /// Autocorrelation coefficient in \[-1, 1\], or `None` if fewer
    /// than `lag + 2` samples.
    ///
    /// Defined as γ(k)/γ(0) where γ(k) is the autocovariance at lag k
    /// and γ(0) is the variance. Returns `None` if variance is zero.
    #[inline]
    #[must_use]
    pub fn correlation(&self) -> Option<f64> {
        if self.count < (self.lag as u64 + 2) {
            return None;
        }
        if self.m2 == 0.0 {
            return None;
        }
        // cross_m accumulated over (count - lag) pairs,
        // m2 accumulated over (count - 1) samples.
        // Normalize both to get comparable per-observation values.
        let n_pairs = (self.count - self.lag as u64) as f64;
        let n_samples = (self.count - 1) as f64;
        Some(self.cross_m * n_samples / (self.m2 * n_pairs))
    }

    /// Raw autocovariance at the configured lag, or `None` if not primed.
    #[inline]
    #[must_use]
    pub fn covariance(&self) -> Option<f64> {
        if self.count < (self.lag as u64 + 2) {
            return None;
        }
        let n_pairs = (self.count - self.lag as u64) as f64;
        Some(self.cross_m / n_pairs)
    }

    /// The configured lag.
    #[inline]
    #[must_use]
    pub fn lag(&self) -> usize {
        self.lag
    }

    /// Number of observations processed.
    #[inline]
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Whether enough data has been collected (>= lag + 2).
    #[inline]
    #[must_use]
    pub fn is_primed(&self) -> bool {
        self.count >= self.lag as u64 + 2
    }

    /// Resets to empty state. Configuration and buffer allocation preserved.
    #[inline]
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.head = 0;
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.cross_m = 0.0;
    }
}

impl AutocorrelationF64Builder {
    /// Sets the lag (required, >= 1).
    #[inline]
    #[must_use]
    pub fn lag(mut self, lag: usize) -> Self {
        self.lag = Some(lag);
        self
    }

    /// Builds the autocorrelation tracker.
    ///
    /// # Errors
    /// Returns `ConfigError` if lag is missing or < 1.
    #[inline]
    pub fn build(self) -> Result<AutocorrelationF64, nexus_stats_core::ConfigError> {
        let lag = self
            .lag
            .ok_or(nexus_stats_core::ConfigError::Missing("lag"))?;
        if lag < 1 {
            return Err(nexus_stats_core::ConfigError::Invalid("lag must be >= 1"));
        }
        Ok(AutocorrelationF64 {
            buffer: vec![0.0; lag].into_boxed_slice(),
            lag,
            head: 0,
            count: 0,
            mean: 0.0,
            m2: 0.0,
            cross_m: 0.0,
        })
    }
}

/// Online autocorrelation at a configurable lag (integer input variant).
///
/// Accepts integer samples, accumulates internally in f64.
/// All query methods return f64.
///
/// # Complexity
/// - O(1) per update, heap-allocated circular buffer.
///
/// # Examples
///
/// ```
/// use nexus_stats_detection::signal::AutocorrelationI64;
///
/// let mut ac = AutocorrelationI64::builder().lag(1).build().unwrap();
/// for i in 0..200 { ac.update(if i % 2 == 0 { 1 as i64 } else { -1 as i64 }); }
/// let r = ac.correlation().unwrap();
/// assert!(r < -0.9);
/// ```
#[derive(Debug, Clone)]
pub struct AutocorrelationI64 {
    buffer: Box<[f64]>,
    lag: usize,
    head: usize,
    count: u64,
    mean: f64,
    m2: f64,
    cross_m: f64,
}

/// Builder for [`AutocorrelationI64`].
#[derive(Debug, Clone)]
pub struct AutocorrelationI64Builder {
    lag: Option<usize>,
}

impl AutocorrelationI64 {
    /// Creates a builder.
    #[inline]
    #[must_use]
    pub fn builder() -> AutocorrelationI64Builder {
        AutocorrelationI64Builder { lag: None }
    }

    /// Feeds a sample.
    #[inline]
    pub fn update(&mut self, sample: i64) {
        #[allow(clippy::cast_lossless, clippy::cast_possible_truncation)]
        let x = sample as f64;
        self.count += 1;

        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        if self.count > self.lag as u64 {
            let x_lagged = self.buffer[self.head];
            self.cross_m += (x - self.mean) * (x_lagged - self.mean);
        }

        self.buffer[self.head] = x;
        self.head = (self.head + 1) % self.lag;
    }

    /// Autocorrelation coefficient in \[-1, 1\], or `None` if fewer
    /// than `lag + 2` samples or variance is zero.
    #[inline]
    #[must_use]
    pub fn correlation(&self) -> Option<f64> {
        if self.count < (self.lag as u64 + 2) {
            return None;
        }
        if self.m2 == 0.0 {
            return None;
        }
        let n_pairs = (self.count - self.lag as u64) as f64;
        let n_samples = (self.count - 1) as f64;
        Some(self.cross_m * n_samples / (self.m2 * n_pairs))
    }

    /// Raw autocovariance at the configured lag, or `None` if not primed.
    #[inline]
    #[must_use]
    pub fn covariance(&self) -> Option<f64> {
        if self.count < (self.lag as u64 + 2) {
            return None;
        }
        let n_pairs = (self.count - self.lag as u64) as f64;
        Some(self.cross_m / n_pairs)
    }

    /// The configured lag.
    #[inline]
    #[must_use]
    pub fn lag(&self) -> usize {
        self.lag
    }

    /// Number of observations processed.
    #[inline]
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Whether enough data has been collected (>= lag + 2).
    #[inline]
    #[must_use]
    pub fn is_primed(&self) -> bool {
        self.count >= self.lag as u64 + 2
    }

    /// Resets to empty state. Configuration and buffer allocation preserved.
    #[inline]
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.head = 0;
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.cross_m = 0.0;
    }
}

impl AutocorrelationI64Builder {
    /// Sets the lag (required, >= 1).
    #[inline]
    #[must_use]
    pub fn lag(mut self, lag: usize) -> Self {
        self.lag = Some(lag);
        self
    }

    /// Builds the autocorrelation tracker.
    ///
    /// # Errors
    /// Returns `ConfigError` if lag is missing or < 1.
    #[inline]
    pub fn build(self) -> Result<AutocorrelationI64, nexus_stats_core::ConfigError> {
        let lag = self
            .lag
            .ok_or(nexus_stats_core::ConfigError::Missing("lag"))?;
        if lag < 1 {
            return Err(nexus_stats_core::ConfigError::Invalid("lag must be >= 1"));
        }
        Ok(AutocorrelationI64 {
            buffer: vec![0.0; lag].into_boxed_slice(),
            lag,
            head: 0,
            count: 0,
            mean: 0.0,
            m2: 0.0,
            cross_m: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alternating_negative_lag1() {
        let mut ac = AutocorrelationF64::builder().lag(1).build().unwrap();
        for i in 0..1000u64 {
            ac.update(if i % 2 == 0 { 1.0 } else { -1.0 }).unwrap();
        }
        let r = ac.correlation().unwrap();
        assert!(r < -0.9, "alternating should be strongly negative, got {r}");
    }

    #[test]
    fn trending_positive_lag1() {
        let mut ac = AutocorrelationF64::builder().lag(1).build().unwrap();
        for i in 0..1000u64 {
            ac.update(i as f64).unwrap();
        }
        let r = ac.correlation().unwrap();
        assert!(
            r > 0.9,
            "monotone trend should have positive lag-1, got {r}"
        );
    }

    #[test]
    fn lag10_periodic() {
        let mut ac = AutocorrelationF64::builder().lag(10).build().unwrap();
        for i in 0..2000u64 {
            ac.update((i % 10) as f64).unwrap();
        }
        let r = ac.correlation().unwrap();
        assert!(
            r > 0.8,
            "period-10 signal should correlate at lag 10, got {r}"
        );
    }

    #[test]
    fn constant_input_zero_variance() {
        let mut ac = AutocorrelationF64::builder().lag(1).build().unwrap();
        for _ in 0..100 {
            ac.update(42.0).unwrap();
        }
        assert!(ac.correlation().is_none());
    }

    #[test]
    fn not_primed_until_lag_plus_2() {
        let mut ac = AutocorrelationF64::builder().lag(5).build().unwrap();
        for i in 0..6 {
            ac.update(i as f64).unwrap();
            assert!(!ac.is_primed(), "should not be primed at count {}", i + 1);
        }
        ac.update(6.0).unwrap();
        assert!(ac.is_primed(), "should be primed at count 7 (lag+2)");
    }

    #[test]
    fn covariance_sign_matches_correlation() {
        let mut ac = AutocorrelationF64::builder().lag(1).build().unwrap();
        for i in 0..500u64 {
            ac.update(i as f64).unwrap();
        }
        let corr = ac.correlation().unwrap();
        let cov = ac.covariance().unwrap();
        assert!(
            corr.signum() == cov.signum(),
            "corr={corr}, cov={cov} — signs should match"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut ac = AutocorrelationF64::builder().lag(1).build().unwrap();
        for i in 0..100 {
            ac.update(i as f64).unwrap();
        }
        ac.reset();
        assert_eq!(ac.count(), 0);
        assert!(!ac.is_primed());
        assert!(ac.correlation().is_none());
    }

    #[test]
    fn lag_accessor() {
        let ac = AutocorrelationF64::builder().lag(7).build().unwrap();
        assert_eq!(ac.lag(), 7);
    }

    #[test]
    fn i64_alternating() {
        let mut ac = AutocorrelationI64::builder().lag(1).build().unwrap();
        for i in 0..1000i64 {
            ac.update(if i % 2 == 0 { 100 } else { -100 });
        }
        let r = ac.correlation().unwrap();
        assert!(r < -0.9, "i64 alternating got {r}");
    }

    #[test]
    fn rejects_nan_and_inf() {
        let mut ac = AutocorrelationF64::builder().lag(1).build().unwrap();
        assert_eq!(
            ac.update(f64::NAN),
            Err(nexus_stats_core::DataError::NotANumber)
        );
        assert_eq!(
            ac.update(f64::INFINITY),
            Err(nexus_stats_core::DataError::Infinite)
        );
        assert_eq!(ac.count(), 0);
    }

    #[test]
    fn builder_requires_lag() {
        let result = AutocorrelationF64::builder().build();
        assert!(matches!(
            result,
            Err(nexus_stats_core::ConfigError::Missing("lag"))
        ));
    }

    #[test]
    fn builder_rejects_zero_lag() {
        let result = AutocorrelationF64::builder().lag(0).build();
        assert!(result.is_err());
    }
}

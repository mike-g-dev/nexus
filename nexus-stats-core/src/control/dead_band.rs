/// Dead band filter — suppresses small changes, reports significant ones.
///
/// Only emits a new value when the sample deviates from the last reported
/// value by more than the threshold. Prevents noisy oscillation around
/// a stable value from generating unnecessary updates.
///
/// # Use Cases
/// - Reducing update frequency for slowly-changing metrics
/// - Hysteresis-free change suppression
/// - Sensor noise filtering
#[derive(Debug, Clone)]
pub struct DeadBandF64 {
    threshold: f64,
    last_reported: f64,
    initialized: bool,
}

impl DeadBandF64 {
    /// Creates a new dead band filter with the given threshold.
    #[inline]
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            last_reported: 0.0,
            initialized: false,
        }
    }

    /// Feeds a sample. Returns `Ok(Some(value))` if the change exceeds
    /// the threshold, `Ok(None)` if suppressed.
    ///
    /// The first sample is always reported.
    ///
    /// # Errors
    ///
    /// Returns `DataError::NotANumber` if the sample is NaN, or
    /// `DataError::Infinite` if the sample is infinite.
    #[inline]
    pub fn update(&mut self, sample: f64) -> Result<Option<f64>, crate::DataError> {
        check_finite!(sample);
        if !self.initialized {
            self.last_reported = sample;
            self.initialized = true;
            return Ok(Option::Some(sample));
        }

        let delta = sample - self.last_reported;
        let abs_delta = if delta < 0.0 { 0.0 - delta } else { delta };

        if abs_delta > self.threshold {
            self.last_reported = sample;
            Ok(Option::Some(sample))
        } else {
            Ok(Option::None)
        }
    }

    /// Last reported value, or `None` if no sample has been processed.
    #[inline]
    #[must_use]
    pub fn last_reported(&self) -> Option<f64> {
        if self.initialized {
            Option::Some(self.last_reported)
        } else {
            Option::None
        }
    }

    /// Resets to uninitialized state.
    #[inline]
    pub fn reset(&mut self) {
        self.last_reported = 0.0;
        self.initialized = false;
    }
}

/// Dead band filter — suppresses small changes, reports significant ones.
///
/// Only emits a new value when the sample deviates from the last reported
/// value by more than the threshold. Prevents noisy oscillation around
/// a stable value from generating unnecessary updates.
///
/// # Use Cases
/// - Reducing update frequency for slowly-changing metrics
/// - Hysteresis-free change suppression
/// - Sensor noise filtering
#[derive(Debug, Clone)]
pub struct DeadBandI64 {
    threshold: u64,
    last_reported: i64,
    initialized: bool,
}

impl DeadBandI64 {
    /// Creates a new dead band filter with the given threshold.
    #[inline]
    #[must_use]
    pub fn new(threshold: u64) -> Self {
        Self {
            threshold,
            last_reported: 0,
            initialized: false,
        }
    }

    /// Feeds a sample. Returns `Some(value)` if the change exceeds
    /// the threshold, `None` if suppressed.
    ///
    /// The first sample is always reported.
    #[inline]
    #[must_use]
    pub fn update(&mut self, sample: i64) -> Option<i64> {
        if !self.initialized {
            self.last_reported = sample;
            self.initialized = true;
            return Option::Some(sample);
        }

        if sample.abs_diff(self.last_reported) > self.threshold {
            self.last_reported = sample;
            Option::Some(sample)
        } else {
            Option::None
        }
    }

    /// Last reported value, or `None` if no sample has been processed.
    #[inline]
    #[must_use]
    pub fn last_reported(&self) -> Option<i64> {
        if self.initialized {
            Option::Some(self.last_reported)
        } else {
            Option::None
        }
    }

    /// Resets to uninitialized state.
    #[inline]
    pub fn reset(&mut self) {
        self.last_reported = 0;
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn first_sample_always_reported() {
        let mut db = DeadBandF64::new(5.0);
        assert_eq!(db.update(100.0).unwrap(), Some(100.0));
    }

    #[test]
    fn small_changes_suppressed() {
        let mut db = DeadBandF64::new(5.0);
        let _ = db.update(100.0).unwrap();
        assert_eq!(db.update(103.0).unwrap(), None); // within threshold
        assert_eq!(db.update(99.0).unwrap(), None); // within threshold
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn large_changes_reported() {
        let mut db = DeadBandF64::new(5.0);
        let _ = db.update(100.0).unwrap();
        assert_eq!(db.update(110.0).unwrap(), Some(110.0)); // exceeds threshold
    }

    #[test]
    fn i64_basic() {
        let mut db = DeadBandI64::new(10);
        assert_eq!(db.update(100), Some(100));
        assert_eq!(db.update(105), None);
        assert_eq!(db.update(115), Some(115));
    }

    #[test]
    fn reset() {
        let mut db = DeadBandF64::new(5.0);
        let _ = db.update(100.0).unwrap();
        db.reset();
        assert!(db.last_reported().is_none());
    }

    #[test]
    fn rejects_nan_and_inf() {
        let mut db = DeadBandF64::new(5.0);
        assert_eq!(db.update(f64::NAN), Err(crate::DataError::NotANumber));
        assert_eq!(db.update(f64::INFINITY), Err(crate::DataError::Infinite));
        assert_eq!(
            db.update(f64::NEG_INFINITY),
            Err(crate::DataError::Infinite)
        );
    }
}

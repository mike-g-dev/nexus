/// Hysteresis filter — Schmitt trigger with separate high/low thresholds.
///
/// Transitions to `true` when sample exceeds the high threshold.
/// Transitions to `false` when sample drops below the low threshold.
/// Between the thresholds, the state is unchanged — preventing oscillation.
///
/// # Use Cases
/// - Thermostat logic (turn on at low, turn off at high)
/// - Alert suppression (don't flap at boundary)
/// - Binary state from a noisy analog signal
#[derive(Debug, Clone)]
pub struct HysteresisF64 {
    low: f64,
    high: f64,
    state: bool,
}

impl HysteresisF64 {
    /// Creates a new hysteresis filter.
    ///
    /// `low_threshold` must be less than `high_threshold`.
    #[inline]
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn new(low_threshold: f64, high_threshold: f64) -> Result<Self, crate::ConfigError> {
        // Negated form rejects NaN (all NaN comparisons are false, so !(NaN < x) → true → reject).
        if !(low_threshold < high_threshold) {
            return Err(crate::ConfigError::Invalid(
                "low threshold must be less than high",
            ));
        }
        Ok(Self {
            low: low_threshold,
            high: high_threshold,
            state: false,
        })
    }

    /// Feeds a sample. Returns the current state.
    ///
    /// # Errors
    ///
    /// Returns `DataError::NotANumber` if the sample is NaN, or
    /// `DataError::Infinite` if the sample is infinite.
    #[inline]
    pub fn update(&mut self, sample: f64) -> Result<bool, crate::DataError> {
        check_finite!(sample);
        if sample >= self.high {
            self.state = true;
        } else if sample <= self.low {
            self.state = false;
        }
        Ok(self.state)
    }

    /// Current state.
    #[inline]
    #[must_use]
    pub fn state(&self) -> bool {
        self.state
    }

    /// Resets state to false.
    #[inline]
    pub fn reset(&mut self) {
        self.state = false;
    }
}

/// Hysteresis filter — Schmitt trigger with separate high/low thresholds.
///
/// Transitions to `true` when sample exceeds the high threshold.
/// Transitions to `false` when sample drops below the low threshold.
/// Between the thresholds, the state is unchanged — preventing oscillation.
///
/// # Use Cases
/// - Thermostat logic (turn on at low, turn off at high)
/// - Alert suppression (don't flap at boundary)
/// - Binary state from a noisy analog signal
#[derive(Debug, Clone)]
pub struct HysteresisI64 {
    low: i64,
    high: i64,
    state: bool,
}

impl HysteresisI64 {
    /// Creates a new hysteresis filter.
    ///
    /// `low_threshold` must be less than `high_threshold`.
    #[inline]
    pub fn new(low_threshold: i64, high_threshold: i64) -> Result<Self, crate::ConfigError> {
        if low_threshold >= high_threshold {
            return Err(crate::ConfigError::Invalid(
                "low threshold must be less than high",
            ));
        }
        Ok(Self {
            low: low_threshold,
            high: high_threshold,
            state: false,
        })
    }

    /// Feeds a sample. Returns the current state.
    #[inline]
    #[must_use]
    pub fn update(&mut self, sample: i64) -> bool {
        if sample >= self.high {
            self.state = true;
        } else if sample <= self.low {
            self.state = false;
        }
        self.state
    }

    /// Current state.
    #[inline]
    #[must_use]
    pub fn state(&self) -> bool {
        self.state
    }

    /// Resets state to false.
    #[inline]
    pub fn reset(&mut self) {
        self.state = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rising_crosses_high() {
        let mut h = HysteresisF64::new(30.0, 70.0).unwrap();
        assert!(!h.update(50.0).unwrap()); // between thresholds, starts false
        assert!(h.update(80.0).unwrap()); // crosses high
    }

    #[test]
    fn falling_crosses_low() {
        let mut h = HysteresisF64::new(30.0, 70.0).unwrap();
        let _ = h.update(80.0).unwrap(); // true
        assert!(h.update(50.0).unwrap()); // between, stays true
        assert!(!h.update(20.0).unwrap()); // crosses low
    }

    #[test]
    fn no_oscillation_at_boundary() {
        let mut h = HysteresisF64::new(30.0, 70.0).unwrap();
        let _ = h.update(80.0).unwrap(); // true

        // Oscillate between thresholds — state should not change
        for _ in 0..10 {
            assert!(h.update(50.0).unwrap());
            assert!(h.update(60.0).unwrap());
            assert!(h.update(40.0).unwrap());
        }
    }

    #[test]
    fn i64_basic() {
        let mut h = HysteresisI64::new(30, 70).unwrap();
        assert!(!h.update(50));
        assert!(h.update(75));
        assert!(h.update(50)); // between, stays true
        assert!(!h.update(25));
    }

    #[test]
    fn reset() {
        let mut h = HysteresisF64::new(30.0, 70.0).unwrap();
        let _ = h.update(80.0).unwrap();
        h.reset();
        assert!(!h.state());
    }

    #[test]
    fn rejects_invalid_thresholds() {
        assert!(matches!(
            HysteresisF64::new(70.0, 30.0),
            Err(crate::ConfigError::Invalid(_))
        ));
    }

    #[test]
    fn rejects_nan_and_inf() {
        let mut h = HysteresisF64::new(30.0, 70.0).unwrap();
        assert_eq!(h.update(f64::NAN), Err(crate::DataError::NotANumber));
        assert_eq!(h.update(f64::INFINITY), Err(crate::DataError::Infinite));
        assert_eq!(h.update(f64::NEG_INFINITY), Err(crate::DataError::Infinite));
    }
}

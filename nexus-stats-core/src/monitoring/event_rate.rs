/// Smoothed event rate tracker (unsigned integer timestamps).
///
/// Uses a fixed-point bit-shift EMA of inter-arrival times, inverted
/// on query to produce a rate (events per unit time). Timestamps are
/// unsigned ticks (e.g. nanoseconds from `Instant`).
///
/// # Use Cases
/// - Message throughput monitoring
/// - Order rate tracking
/// - Adaptive rate limiting input
#[derive(Debug, Clone)]
pub struct EventRateU64 {
    acc: i128,
    shift: u32,
    span: u64,
    last_timestamp: u64,
    count: u64,
    min_samples: u64,
    initialized: bool,
}

/// Builder for [`EventRateU64`].
#[derive(Debug, Clone)]
pub struct EventRateU64Builder {
    span: Option<u64>,
    min_samples: u64,
}

impl EventRateU64 {
    /// Creates a builder.
    #[inline]
    #[must_use]
    pub fn builder() -> EventRateU64Builder {
        EventRateU64Builder {
            span: None,
            min_samples: 2,
        }
    }

    /// Updates with an event at the given tick.
    #[inline]
    pub fn update(&mut self, timestamp: u64) {
        self.count += 1;

        if self.count == 1 {
            self.last_timestamp = timestamp;
            return;
        }

        let dt = timestamp.wrapping_sub(self.last_timestamp) as i64;
        self.last_timestamp = timestamp;

        if self.initialized {
            let dt_shifted = (dt as i128) << self.shift;
            self.acc += (dt_shifted - self.acc) >> self.shift;
        } else {
            self.acc = (dt as i128) << self.shift;
            self.initialized = true;
        }
    }

    /// Current smoothed event rate (events per unit time).
    ///
    /// Returns `None` if not primed or if the smoothed interval is zero.
    #[inline]
    #[must_use]
    pub fn rate(&self) -> Option<f64> {
        let interval = self.interval()?;
        if interval == 0 {
            None
        } else {
            Some(1.0 / interval as f64)
        }
    }

    /// Current smoothed inter-event interval in ticks, or `None` if < 2 events.
    #[inline]
    #[must_use]
    pub fn interval(&self) -> Option<u64> {
        if self.count >= self.min_samples && self.initialized {
            Some((self.acc >> self.shift) as u64)
        } else {
            None
        }
    }

    /// Effective span after rounding to `2^k - 1`.
    #[inline]
    #[must_use]
    pub fn effective_span(&self) -> u64 {
        self.span
    }

    /// Number of events recorded.
    #[inline]
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Whether the tracker has reached `min_samples`.
    #[inline]
    #[must_use]
    pub fn is_primed(&self) -> bool {
        self.count >= self.min_samples
    }

    /// Resets to uninitialized state.
    #[inline]
    pub fn reset(&mut self) {
        self.acc = 0;
        self.last_timestamp = 0;
        self.count = 0;
        self.initialized = false;
    }
}

impl EventRateU64Builder {
    /// Smoothing span. Rounded up to next `2^k - 1`.
    #[inline]
    #[must_use]
    pub fn span(mut self, n: u64) -> Self {
        self.span = Some(n);
        self
    }

    /// Minimum events before rate is valid. Default: 2.
    #[inline]
    #[must_use]
    pub fn min_samples(mut self, min: u64) -> Self {
        self.min_samples = min;
        self
    }

    /// Builds the event rate tracker.
    ///
    /// # Errors
    ///
    /// - Span must have been set and >= 1.
    #[inline]
    pub fn build(self) -> Result<EventRateU64, crate::ConfigError> {
        let requested = self.span.ok_or(crate::ConfigError::Missing("span"))?;
        if requested < 1 {
            return Err(crate::ConfigError::Invalid("EventRate span must be >= 1"));
        }

        let effective = crate::smoothing::ema::next_power_of_two_minus_one(requested);
        let shift = crate::smoothing::ema::log2_of_span_plus_one(effective);

        Ok(EventRateU64 {
            acc: 0,
            shift,
            span: effective,
            last_timestamp: 0,
            count: 0,
            min_samples: self.min_samples,
            initialized: false,
        })
    }
}

/// Smoothed event rate tracker (signed integer timestamps).
///
/// Identical to [`EventRateU64`] but accepts `i64` timestamps for
/// compatibility with signed time representations (e.g. offsets,
/// relative ticks).
#[derive(Debug, Clone)]
pub struct EventRateI64 {
    acc: i128,
    shift: u32,
    span: u64,
    last_timestamp: i64,
    count: u64,
    min_samples: u64,
    initialized: bool,
}

/// Builder for [`EventRateI64`].
#[derive(Debug, Clone)]
pub struct EventRateI64Builder {
    span: Option<u64>,
    min_samples: u64,
}

impl EventRateI64 {
    /// Creates a builder.
    #[inline]
    #[must_use]
    pub fn builder() -> EventRateI64Builder {
        EventRateI64Builder {
            span: None,
            min_samples: 2,
        }
    }

    /// Updates with an event at the given tick.
    #[inline]
    pub fn update(&mut self, timestamp: i64) {
        self.count += 1;

        if self.count == 1 {
            self.last_timestamp = timestamp;
            return;
        }

        let dt = timestamp - self.last_timestamp;
        self.last_timestamp = timestamp;

        if self.initialized {
            let dt_shifted = (dt as i128) << self.shift;
            self.acc += (dt_shifted - self.acc) >> self.shift;
        } else {
            self.acc = (dt as i128) << self.shift;
            self.initialized = true;
        }
    }

    /// Current smoothed event rate (events per unit time).
    ///
    /// Returns `None` if not primed or if the smoothed interval is zero.
    #[inline]
    #[must_use]
    pub fn rate(&self) -> Option<f64> {
        let interval = self.interval()?;
        if interval == 0 {
            None
        } else {
            Some(1.0 / interval as f64)
        }
    }

    /// Current smoothed inter-event interval in ticks, or `None` if < 2 events.
    #[inline]
    #[must_use]
    pub fn interval(&self) -> Option<i64> {
        if self.count >= self.min_samples && self.initialized {
            Some((self.acc >> self.shift) as i64)
        } else {
            None
        }
    }

    /// Effective span after rounding to `2^k - 1`.
    #[inline]
    #[must_use]
    pub fn effective_span(&self) -> u64 {
        self.span
    }

    /// Number of events recorded.
    #[inline]
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Whether the tracker has reached `min_samples`.
    #[inline]
    #[must_use]
    pub fn is_primed(&self) -> bool {
        self.count >= self.min_samples
    }

    /// Resets to uninitialized state.
    #[inline]
    pub fn reset(&mut self) {
        self.acc = 0;
        self.last_timestamp = 0;
        self.count = 0;
        self.initialized = false;
    }
}

impl EventRateI64Builder {
    /// Smoothing span. Rounded up to next `2^k - 1`.
    #[inline]
    #[must_use]
    pub fn span(mut self, n: u64) -> Self {
        self.span = Some(n);
        self
    }

    /// Minimum events before rate is valid. Default: 2.
    #[inline]
    #[must_use]
    pub fn min_samples(mut self, min: u64) -> Self {
        self.min_samples = min;
        self
    }

    /// Builds the event rate tracker.
    ///
    /// # Errors
    ///
    /// - Span must have been set and >= 1.
    #[inline]
    pub fn build(self) -> Result<EventRateI64, crate::ConfigError> {
        let requested = self.span.ok_or(crate::ConfigError::Missing("span"))?;
        if requested < 1 {
            return Err(crate::ConfigError::Invalid("EventRate span must be >= 1"));
        }

        let effective = crate::smoothing::ema::next_power_of_two_minus_one(requested);
        let shift = crate::smoothing::ema::log2_of_span_plus_one(effective);

        Ok(EventRateI64 {
            acc: 0,
            shift,
            span: effective,
            last_timestamp: 0,
            count: 0,
            min_samples: self.min_samples,
            initialized: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_rate_u64() {
        let mut er = EventRateU64::builder().span(15).build().unwrap();

        for i in 0..100u64 {
            er.update(i * 10);
        }

        let interval = er.interval().unwrap();
        assert_eq!(interval, 10, "interval should converge to 10");
        let rate = er.rate().unwrap();
        assert!(
            (rate - 0.1).abs() < 0.001,
            "rate should be ~0.1, got {rate}"
        );
    }

    #[test]
    fn constant_rate_i64() {
        let mut er = EventRateI64::builder().span(15).build().unwrap();

        for i in 0..100i64 {
            er.update(i * 10);
        }

        let interval = er.interval().unwrap();
        assert_eq!(interval, 10, "interval should converge to 10");
        let rate = er.rate().unwrap();
        assert!(
            (rate - 0.1).abs() < 0.001,
            "rate should be ~0.1, got {rate}"
        );
    }

    #[test]
    fn burst_increases_rate() {
        let mut er = EventRateU64::builder().span(7).build().unwrap();

        for i in 0..20u64 {
            er.update(i * 1000);
        }
        let normal_rate = er.rate().unwrap();

        for i in 0..20u64 {
            er.update(20_000 + i * 100);
        }
        let burst_rate = er.rate().unwrap();

        assert!(
            burst_rate > normal_rate,
            "burst rate ({burst_rate}) should exceed normal ({normal_rate})"
        );
    }

    #[test]
    fn priming() {
        let mut er = EventRateU64::builder()
            .span(7)
            .min_samples(5)
            .build()
            .unwrap();

        for i in 0..4u64 {
            er.update(i * 10);
            assert!(!er.is_primed());
            assert!(er.rate().is_none());
        }
        er.update(40);
        assert!(er.is_primed());
        assert!(er.rate().is_some());
    }

    #[test]
    fn reset_u64() {
        let mut er = EventRateU64::builder().span(7).build().unwrap();
        for i in 0..10u64 {
            er.update(i * 10);
        }
        er.reset();
        assert_eq!(er.count(), 0);
        assert!(er.rate().is_none());
    }

    #[test]
    fn reset_i64() {
        let mut er = EventRateI64::builder().span(7).build().unwrap();
        for i in 0..10i64 {
            er.update(i * 10);
        }
        er.reset();
        assert_eq!(er.count(), 0);
        assert!(er.rate().is_none());
    }

    #[test]
    fn zero_interval_returns_none() {
        let mut er = EventRateU64::builder().span(7).build().unwrap();
        er.update(100);
        er.update(100);
        assert!(er.rate().is_none());
    }

    #[test]
    fn errors_without_span() {
        let result = EventRateU64::builder().build();
        assert!(matches!(result, Err(crate::ConfigError::Missing("span"))));
    }

    #[test]
    fn effective_span_rounds_up() {
        let er = EventRateU64::builder().span(10).build().unwrap();
        assert_eq!(er.effective_span(), 15);
    }
}

//! Exponential backoff with optional jitter and deadline.
//!
//! A small struct that tracks retry state. The user controls the loop —
//! no closures, no future wrapping, no magic.
//!
//! ```ignore
//! use nexus_async_rt::Backoff;
//! use std::time::{Duration, Instant};
//!
//! let mut backoff = Backoff::builder()
//!     .initial(Duration::from_millis(10))
//!     .max_delay(Duration::from_secs(5))
//!     .max_retries(10)
//!     .deadline(Instant::now() + Duration::from_secs(30))
//!     .jitter(0.25)
//!     .build();
//!
//! loop {
//!     match try_connect().await {
//!         Ok(conn) => break conn,
//!         Err(e) => backoff.wait(e).await?,
//!     }
//! }
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Global counter for per-instance jitter seed.
static BACKOFF_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Retries exhausted — either max retries reached or deadline passed.
///
/// Wraps the last error from the failed operation.
#[derive(Debug)]
pub struct Exhausted<E>(pub E);

impl<E: std::fmt::Display> std::fmt::Display for Exhausted<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "backoff exhausted: {}", self.0)
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for Exhausted<E> {}

/// Exponential backoff with optional jitter and deadline.
///
/// Tracks current delay, retry count, and optional wall-clock deadline.
/// Call `.wait(err).await` to check exhaustion, sleep if retries remain,
/// and advance state. Returns `Err(Exhausted(err))` when done.
pub struct Backoff {
    initial: Duration,
    current: Duration,
    max_delay: Duration,
    max_retries: Option<u32>,
    deadline: Option<Instant>,
    retries: u32,
    jitter: f64,
    /// Per-instance seed so identically-configured instances produce
    /// different jitter sequences (mitigates thundering herd).
    seed: u64,
}

impl Backoff {
    /// Create a builder.
    #[must_use]
    pub fn builder() -> BackoffBuilder {
        BackoffBuilder::new()
    }

    /// Check exhaustion, sleep for the current delay, then advance.
    ///
    /// Returns `Ok(())` if retries remain (caller should loop).
    /// Returns `Err(Exhausted(err))` if max retries or deadline
    /// reached — the caller should escalate or fail over.
    ///
    /// If a deadline is set, the sleep is capped to not exceed it.
    ///
    /// # Panics
    ///
    /// Panics if called outside [`Runtime::block_on`](crate::Runtime::block_on).
    pub async fn wait<E>(&mut self, err: E) -> Result<(), Exhausted<E>> {
        if self.is_exhausted() {
            return Err(Exhausted(err));
        }

        let delay = self.effective_delay();

        // Re-check: deadline may have been reached while computing delay.
        if delay.is_zero() && self.deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(Exhausted(err));
        }

        crate::context::sleep(delay).await;
        self.advance();
        Ok(())
    }

    /// Advance the backoff state without sleeping.
    ///
    /// Useful when the caller manages timing externally.
    pub fn advance(&mut self) {
        self.retries += 1;
        self.current = self
            .current
            .checked_mul(2)
            .map_or(self.max_delay, |next| next.min(self.max_delay));
    }

    /// Whether retries are exhausted (max retries or deadline).
    ///
    /// Returns `false` if neither limit is set.
    pub fn is_exhausted(&self) -> bool {
        if self.max_retries.is_some_and(|max| self.retries >= max) {
            return true;
        }
        if self.deadline.is_some_and(|d| Instant::now() >= d) {
            return true;
        }
        false
    }

    /// Current retry count.
    pub fn retries(&self) -> u32 {
        self.retries
    }

    /// The next delay (before jitter).
    pub fn current_delay(&self) -> Duration {
        self.current
    }

    /// Time remaining until deadline, if one is set.
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }

    /// Reset delay and retry count to initial state.
    ///
    /// The deadline is **not** reset — it is wall-clock absolute.
    pub fn reset(&mut self) {
        self.current = self.initial;
        self.retries = 0;
    }

    /// Compute the effective delay: jittered, then capped to deadline.
    fn effective_delay(&self) -> Duration {
        let delay = self.jittered_delay();
        self.deadline.map_or(delay, |d| {
            delay.min(d.saturating_duration_since(Instant::now()))
        })
    }

    /// Compute the jittered delay for the current step.
    fn jittered_delay(&self) -> Duration {
        if self.jitter == 0.0 {
            return self.current;
        }

        // Jitter: multiply by (1.0 ± jitter).
        // Per-instance seed mixed with retry count so identically-configured
        // instances produce different sequences (mitigates thundering herd).
        let hash = {
            let a = self.retries as u64;
            let b = self.current.as_nanos() as u64;
            a.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(b)
                .wrapping_add(self.seed)
        };
        // Map to [-1.0, 1.0]
        let normalized = (hash as f64 / u64::MAX as f64).mul_add(2.0, -1.0);
        let factor = self.jitter.mul_add(normalized, 1.0);
        let jittered_nanos = self.current.as_nanos() as f64 * factor;
        Duration::from_nanos(jittered_nanos.max(0.0) as u64)
    }
}

impl std::fmt::Debug for Backoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backoff")
            .field("current", &self.current)
            .field("retries", &self.retries)
            .field("max_delay", &self.max_delay)
            .field("max_retries", &self.max_retries)
            .field("deadline", &self.deadline)
            .field("jitter", &self.jitter)
            .finish()
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Builder for [`Backoff`].
pub struct BackoffBuilder {
    initial: Duration,
    max_delay: Duration,
    max_retries: Option<u32>,
    deadline: Option<Instant>,
    jitter: f64,
}

impl BackoffBuilder {
    #[must_use]
    fn new() -> Self {
        Self {
            initial: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            max_retries: None,
            deadline: None,
            jitter: 0.0,
        }
    }

    /// Initial delay. Default: 100ms.
    #[must_use]
    pub fn initial(mut self, d: Duration) -> Self {
        self.initial = d;
        self
    }

    /// Maximum delay cap. Default: 30s.
    #[must_use]
    pub fn max_delay(mut self, d: Duration) -> Self {
        self.max_delay = d;
        self
    }

    /// Maximum number of retries. Default: unlimited.
    #[must_use]
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = Some(n);
        self
    }

    /// Wall-clock deadline. After this instant, [`Backoff::wait`] returns
    /// `Err(Exhausted)`. Sleep durations are capped to not exceed it.
    ///
    /// Default: no deadline.
    #[must_use]
    pub fn deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Jitter factor (0.0 to 1.0). Default: 0.0 (no jitter).
    ///
    /// Each delay is randomly scaled by `1.0 ± jitter`. For example,
    /// `jitter(0.25)` means a 100ms delay becomes 75ms–125ms.
    #[must_use]
    pub fn jitter(mut self, factor: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&factor),
            "jitter must be between 0.0 and 1.0, got {factor}"
        );
        self.jitter = factor;
        self
    }

    /// Build the backoff.
    ///
    /// # Panics
    ///
    /// Panics if `initial` exceeds `max_delay`.
    #[must_use]
    pub fn build(self) -> Backoff {
        assert!(
            self.initial <= self.max_delay,
            "initial delay ({:?}) must not exceed max_delay ({:?})",
            self.initial,
            self.max_delay,
        );
        Backoff {
            initial: self.initial,
            current: self.initial,
            max_delay: self.max_delay,
            max_retries: self.max_retries,
            deadline: self.deadline,
            retries: 0,
            jitter: self.jitter,
            seed: BACKOFF_COUNTER.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl Default for BackoffBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(
    unused_must_use,
    clippy::float_cmp,
    dead_code,
    clippy::ref_option,
    clippy::redundant_closure_for_method_calls,
    clippy::let_underscore_future,
    clippy::semicolon_if_nothing_returned
)]
mod tests {
    use super::*;

    #[test]
    fn doubles_each_step() {
        let mut b = Backoff::builder()
            .initial(Duration::from_millis(10))
            .max_delay(Duration::from_secs(10))
            .build();

        assert_eq!(b.current_delay(), Duration::from_millis(10));
        b.advance();
        assert_eq!(b.current_delay(), Duration::from_millis(20));
        b.advance();
        assert_eq!(b.current_delay(), Duration::from_millis(40));
        b.advance();
        assert_eq!(b.current_delay(), Duration::from_millis(80));
    }

    #[test]
    fn caps_at_max() {
        let mut b = Backoff::builder()
            .initial(Duration::from_secs(1))
            .max_delay(Duration::from_secs(5))
            .build();

        b.advance(); // 2s
        b.advance(); // 4s
        b.advance(); // 5s (capped)
        assert_eq!(b.current_delay(), Duration::from_secs(5));
        b.advance(); // still 5s
        assert_eq!(b.current_delay(), Duration::from_secs(5));
    }

    #[test]
    fn exhausted_after_max_retries() {
        let mut b = Backoff::builder()
            .initial(Duration::from_millis(1))
            .max_retries(3)
            .build();

        assert!(!b.is_exhausted());
        b.advance();
        assert!(!b.is_exhausted());
        b.advance();
        assert!(!b.is_exhausted());
        b.advance();
        assert!(b.is_exhausted());
    }

    #[test]
    fn unlimited_retries() {
        let mut b = Backoff::builder()
            .initial(Duration::from_millis(1))
            .max_delay(Duration::from_millis(1))
            .build();

        for _ in 0..10_000 {
            b.advance();
        }
        assert!(!b.is_exhausted());
        assert_eq!(b.retries(), 10_000);
    }

    #[test]
    fn reset_restores_initial() {
        let mut b = Backoff::builder()
            .initial(Duration::from_millis(10))
            .max_retries(5)
            .build();

        b.advance();
        b.advance();
        b.advance();
        assert_eq!(b.retries(), 3);
        assert_eq!(b.current_delay(), Duration::from_millis(80));

        b.reset();
        assert_eq!(b.retries(), 0);
        assert_eq!(b.current_delay(), Duration::from_millis(10));
    }

    #[test]
    fn jitter_stays_in_range() {
        let mut b = Backoff::builder()
            .initial(Duration::from_millis(100))
            .max_delay(Duration::from_secs(10))
            .jitter(0.5)
            .build();

        // Run several iterations — jittered delay should be within ±50%.
        // Allow 1ns tolerance for float→integer truncation.
        for _ in 0..20 {
            let delay = b.jittered_delay();
            let base = b.current_delay().as_nanos();
            let actual = delay.as_nanos();
            let lo = (base as f64 * 0.5) as u128;
            let hi = (base as f64 * 1.5) as u128 + 1;
            assert!(
                actual >= lo && actual <= hi,
                "delay {actual}ns out of range [{lo}, {hi}] for base {base}ns"
            );
            b.advance();
        }
    }

    #[test]
    #[should_panic(expected = "jitter must be between")]
    fn jitter_out_of_range_panics() {
        let _ = Backoff::builder().jitter(1.5).build();
    }

    #[test]
    #[should_panic(expected = "initial delay")]
    fn initial_exceeds_max_delay_panics() {
        Backoff::builder()
            .initial(Duration::from_secs(60))
            .max_delay(Duration::from_secs(5))
            .build();
    }

    #[test]
    fn default_values() {
        let b = Backoff::builder().build();
        assert_eq!(b.current_delay(), Duration::from_millis(100));
        assert_eq!(b.max_delay, Duration::from_secs(30));
        assert!(!b.is_exhausted());
        assert!(b.remaining().is_none());
    }

    #[test]
    fn deadline_exhausts() {
        // Deadline in the past — immediately exhausted.
        let b = Backoff::builder()
            .initial(Duration::from_millis(10))
            .deadline(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
            .build();

        assert!(b.is_exhausted());
    }

    #[test]
    fn deadline_remaining() {
        let deadline = Instant::now() + Duration::from_secs(60);
        let b = Backoff::builder()
            .initial(Duration::from_millis(10))
            .deadline(deadline)
            .build();

        let remaining = b.remaining().expect("should have remaining");
        assert!(remaining > Duration::ZERO);
        assert!(remaining <= Duration::from_secs(60));
    }

    #[test]
    fn effective_delay_capped_by_deadline() {
        // Deadline 50ms from now, but current delay is 10s.
        let b = Backoff::builder()
            .initial(Duration::from_millis(50))
            .max_delay(Duration::from_secs(10))
            .deadline(Instant::now() + Duration::from_millis(50))
            .build();

        let delay = b.effective_delay();
        // Should be capped to ~50ms, not 10s.
        assert!(delay <= Duration::from_millis(55));
    }

    #[test]
    fn reset_does_not_clear_deadline() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut b = Backoff::builder()
            .initial(Duration::from_millis(10))
            .deadline(deadline)
            .build();

        b.advance();
        b.advance();
        b.reset();

        // Deadline still set.
        assert!(b.remaining().is_some());
        assert_eq!(b.retries(), 0);
        assert_eq!(b.current_delay(), Duration::from_millis(10));
    }

    #[test]
    fn advance_does_not_overflow_large_delay() {
        let mut b = Backoff::builder()
            .initial(Duration::from_secs(u64::MAX / 4))
            .max_delay(Duration::from_secs(u64::MAX / 4))
            .build();

        // Should not panic — checked_mul saturates to max_delay.
        b.advance();
        assert_eq!(b.current_delay(), Duration::from_secs(u64::MAX / 4));
    }

    #[test]
    fn different_instances_different_jitter() {
        let a = Backoff::builder()
            .initial(Duration::from_millis(100))
            .jitter(0.5)
            .build();
        let b = Backoff::builder()
            .initial(Duration::from_millis(100))
            .jitter(0.5)
            .build();

        // Different seeds → different jitter values.
        assert_ne!(a.seed, b.seed);
    }
}

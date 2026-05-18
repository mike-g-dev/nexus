//! Multi-producer single-consumer channel.
//!
//! Wraps [`queue::mpsc`](crate::queue::mpsc) with backoff and parking.
//!
//! # Philosophy
//!
//! **Senders use brief backoff.** They spin, yield, then return error if still
//! full. Never make syscalls - keeps the hot path fast.
//!
//! **Receivers can block.** They use `park_timeout` to wait for messages
//! without burning CPU. The timeout ensures they periodically check for
//! disconnection.
//!
//! # Example
//!
//! ```
//! use nexus_logbuf::channel::mpsc;
//! use std::thread;
//!
//! let (tx, mut rx) = mpsc::channel(4096);
//!
//! for i in 0..4 {
//!     let mut tx = tx.clone();
//!     thread::spawn(move || {
//!         let payload = i.to_string();
//!         let mut claim = tx.send(payload.len()).unwrap();
//!         claim.copy_from_slice(payload.as_bytes());
//!         claim.commit();
//!         tx.notify();
//!     });
//! }
//!
//! drop(tx); // Drop original sender
//!
//! let mut count = 0;
//! while let Ok(_record) = rx.recv(None) {
//!     count += 1;
//!     if count == 4 { break; }
//! }
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use crossbeam_utils::Backoff;

use crate::queue::mpsc as queue;

/// Default park timeout for receivers.
const DEFAULT_PARK_TIMEOUT: Duration = Duration::from_millis(100);

/// Creates a bounded MPSC channel.
///
/// Capacity is rounded up to the next power of two.
///
/// # Panics
///
/// Panics if `capacity` is less than 16 bytes.
pub fn channel(capacity: usize) -> (Sender, Receiver) {
    let (producer, consumer) = queue::new(capacity);

    let parker = crossbeam_utils::sync::Parker::new();
    let unparker = parker.unparker().clone();

    let shared = Arc::new(ChannelShared {
        receiver_waiting: AtomicBool::new(false),
        receiver_unparker: unparker,
        sender_count: AtomicUsize::new(1),
        receiver_disconnected: AtomicBool::new(false),
    });

    (
        Sender {
            inner: producer,
            shared: Arc::clone(&shared),
        },
        Receiver {
            inner: consumer,
            parker,
            shared,
        },
    )
}

/// Shared state between senders and receiver.
struct ChannelShared {
    /// True if receiver is parked and waiting.
    receiver_waiting: AtomicBool,
    /// Unparker for the receiver.
    receiver_unparker: crossbeam_utils::sync::Unparker,
    /// Number of active senders.
    sender_count: AtomicUsize,
    /// True if receiver has been dropped.
    receiver_disconnected: AtomicBool,
}

// ============================================================================
// Sender
// ============================================================================

/// Sending half of the MPSC channel.
///
/// This type is `Clone` - multiple senders can exist concurrently.
///
/// **Never blocks with syscalls.** Uses brief backoff (spin + yield) then
/// returns error if buffer is full.
pub struct Sender {
    inner: queue::Producer,
    shared: Arc<ChannelShared>,
}

impl Clone for Sender {
    fn clone(&self) -> Self {
        self.shared.sender_count.fetch_add(1, Ordering::Relaxed);
        Self {
            inner: self.inner.clone(),
            shared: Arc::clone(&self.shared),
        }
    }
}

/// Error returned from [`Sender::send`] when the receiver has been dropped.
///
/// `send` has only one runtime failure mode — the receiver is gone. Passing
/// `len == 0` is a precondition violation and panics (see
/// [`queue::Producer::try_claim`] for details).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelClosed;

impl std::fmt::Display for ChannelClosed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("channel disconnected")
    }
}

impl std::error::Error for ChannelClosed {}

/// Error returned from [`Sender::try_send`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrySendError {
    /// The buffer is full.
    Full,
    /// The receiver has been dropped.
    Disconnected,
}

impl std::fmt::Display for TrySendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "channel full"),
            Self::Disconnected => write!(f, "channel disconnected"),
        }
    }
}

impl std::error::Error for TrySendError {}

impl Sender {
    /// Claims space for a record, spinning until space is available.
    ///
    /// **Never makes syscalls.** Spins and yields until the buffer has space
    /// or the receiver disconnects.
    ///
    /// After receiving a [`WriteClaim`](queue::WriteClaim), write your payload
    /// and call [`commit()`](queue::WriteClaim::commit) to publish. Then call
    /// [`notify()`](Self::notify) to wake a parked receiver.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelClosed`] if the receiver was dropped.
    ///
    /// # Panics
    ///
    /// Panics if `len == 0` (see [`queue::Producer::try_claim`]).
    #[inline]
    pub fn send(&mut self, len: usize) -> Result<queue::WriteClaim<'_>, ChannelClosed> {
        // Precondition check before any state inspection — `len == 0` is a
        // contract violation regardless of channel state, and the doc
        // contract is honest only if it panics unconditionally.
        assert!(len > 0, "payload length must be non-zero");
        if self.shared.receiver_disconnected.load(Ordering::Relaxed) {
            return Err(ChannelClosed);
        }

        let backoff = Backoff::new();

        loop {
            // Polonius / NLL successor workaround.
            //
            // The naive form
            //   `if let Ok(claim) = self.inner.try_claim(len) { return Ok(claim); }`
            // fails to compile because the borrow checker holds the
            // `&mut self.inner` borrow for the entire `if let` body — and
            // therefore for the whole loop iteration — even on the
            // early-return path. Without the early-return-as-narrowing
            // analysis Polonius provides, the compiler can't see that the
            // borrow is dead at the `return` statement, so the next loop
            // iteration's `try_claim` is rejected as a conflicting reborrow.
            //
            // The transmute<'a, 'a> here is a runtime no-op — same type,
            // same lifetime — that bypasses the borrow checker for a
            // pattern that is actually sound. When Polonius lands in
            // stable rustc, this can be rewritten to the natural form and
            // the unsafe block deleted.
            //
            // SAFETY: The transmute is between two identical types with
            // identical lifetimes. The early return guarantees the
            // original `&mut self.inner` borrow is dead before the
            // returned claim is used, and the loop only re-borrows after
            // the previous iteration's borrow has fully expired.
            // SAFETY: Polonius workaround — transmute is between identical types
            // with identical lifetimes. Early return guarantees the original
            // &mut self.inner borrow is dead before the claim is used.
            unsafe {
                let inner_ptr: *mut queue::Producer = &raw mut self.inner;
                if let Ok(claim) = (*inner_ptr).try_claim(len) {
                    return Ok(std::mem::transmute::<
                        queue::WriteClaim<'_>,
                        queue::WriteClaim<'_>,
                    >(claim));
                }
                // BufferFull — wait for receiver to drain.
                backoff.snooze();
                if self.shared.receiver_disconnected.load(Ordering::Relaxed) {
                    return Err(ChannelClosed);
                }
                // Reset backoff after it completes to keep spinning
                if backoff.is_completed() {
                    backoff.reset();
                }
            }
        }
    }

    /// Attempts to claim space for a record without any waiting.
    ///
    /// # Errors
    ///
    /// - [`TrySendError::Full`] if buffer is full
    /// - [`TrySendError::Disconnected`] if receiver was dropped
    ///
    /// # Panics
    ///
    /// Panics if `len == 0` (see [`queue::Producer::try_claim`]).
    #[inline]
    pub fn try_send(&mut self, len: usize) -> Result<queue::WriteClaim<'_>, TrySendError> {
        // Precondition check before any state inspection — see `send` for why.
        assert!(len > 0, "payload length must be non-zero");
        if self.shared.receiver_disconnected.load(Ordering::Relaxed) {
            return Err(TrySendError::Disconnected);
        }

        match self.inner.try_claim(len) {
            Ok(claim) => Ok(claim),
            Err(crate::BufferFull) => Err(TrySendError::Full),
        }
    }

    /// Notifies the receiver that data is available.
    ///
    /// Call this after committing a write to wake a parked receiver.
    /// Cheap no-op if receiver isn't parked.
    #[inline]
    pub fn notify(&self) {
        if self.shared.receiver_waiting.load(Ordering::Relaxed) {
            self.shared.receiver_unparker.unpark();
        }
    }

    /// Returns the capacity of the underlying buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns `true` if the receiver has been dropped.
    #[inline]
    pub fn is_disconnected(&self) -> bool {
        self.shared.receiver_disconnected.load(Ordering::Relaxed)
    }
}

impl Drop for Sender {
    fn drop(&mut self) {
        let prev = self.shared.sender_count.fetch_sub(1, Ordering::Relaxed);
        if prev == 1 {
            // Last sender - wake receiver so it can observe disconnection
            self.shared.receiver_unparker.unpark();
        }
    }
}

impl std::fmt::Debug for Sender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sender")
            .field("capacity", &self.capacity())
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Receiver
// ============================================================================

/// Receiving half of the MPSC channel.
///
/// **Can block with syscalls.** Uses `park_timeout` to wait for messages
/// without burning CPU.
pub struct Receiver {
    inner: queue::Consumer,
    parker: crossbeam_utils::sync::Parker,
    shared: Arc<ChannelShared>,
}

/// Error returned from [`Receiver::recv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecvError {
    /// The timeout elapsed before a message arrived.
    ///
    /// Only returned when a timeout was specified.
    Timeout,
    /// All senders have been dropped and the buffer is empty.
    Disconnected,
}

impl std::fmt::Display for RecvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout => write!(f, "receive timed out"),
            Self::Disconnected => write!(f, "channel disconnected"),
        }
    }
}

impl std::error::Error for RecvError {}

impl Receiver {
    /// Blocks until a message is available or the optional timeout elapses.
    ///
    /// - `None` — block forever (or until disconnected)
    /// - `Some(Duration::ZERO)` — single try, no spinning
    /// - `Some(duration)` — block up to `duration`
    ///
    /// Uses backoff (spin → yield) then parks.
    ///
    /// # Errors
    ///
    /// - [`RecvError::Timeout`] if timeout elapsed (only when `Some`)
    /// - [`RecvError::Disconnected`] if all senders were dropped and buffer is empty
    #[inline]
    pub fn recv(&mut self, timeout: Option<Duration>) -> Result<queue::ReadClaim<'_>, RecvError> {
        // Fast path for zero timeout - single try, no spinning
        if timeout == Some(Duration::ZERO) {
            // SAFETY: see Polonius pattern at the top of `Sender::send` in
            // this file. Same shape: early return frees the borrow before
            // any reuse.
            // SAFETY: Polonius workaround — same-type transmute, early return
            // ensures borrow is dead before the claim is used.
            unsafe {
                let inner_ptr: *mut queue::Consumer = &raw mut self.inner;
                if let Some(claim) = (*inner_ptr).try_claim() {
                    return Ok(std::mem::transmute::<
                        queue::ReadClaim<'_>,
                        queue::ReadClaim<'_>,
                    >(claim));
                }
            }
            if self.shared.sender_count.load(Ordering::Relaxed) == 0 {
                return Err(RecvError::Disconnected);
            }
            return Err(RecvError::Timeout);
        }

        let park_timeout = timeout.unwrap_or(DEFAULT_PARK_TIMEOUT);
        let backoff = Backoff::new();

        loop {
            // SAFETY: see Polonius pattern at the top of `Sender::send` in
            // this file. Same shape: early return frees the borrow before
            // the next loop iteration reuses it.
            // SAFETY: Polonius workaround — same-type transmute, early return
            // ensures borrow is dead before the next iteration reborrows.
            unsafe {
                let inner_ptr: *mut queue::Consumer = &raw mut self.inner;
                if let Some(claim) = (*inner_ptr).try_claim() {
                    return Ok(std::mem::transmute::<
                        queue::ReadClaim<'_>,
                        queue::ReadClaim<'_>,
                    >(claim));
                }
            }

            if self.shared.sender_count.load(Ordering::Relaxed) == 0 {
                return Err(RecvError::Disconnected);
            }

            // Backoff phase: spin/yield without syscalls
            if !backoff.is_completed() {
                backoff.snooze();
                continue;
            }

            // Park phase
            self.shared.receiver_waiting.store(true, Ordering::Relaxed);
            self.parker.park_timeout(park_timeout);
            self.shared.receiver_waiting.store(false, Ordering::Relaxed);

            // For Some(timeout), only park once then return Timeout
            // For None, loop back and try again
            if timeout.is_some() {
                // Final try after park.
                // SAFETY: see Polonius pattern at the top of
                // `Sender::send` in this file.
                // SAFETY: Polonius workaround — same-type transmute, early
                // return ensures borrow is dead before the claim is used.
                unsafe {
                    let inner_ptr: *mut queue::Consumer = &raw mut self.inner;
                    if let Some(claim) = (*inner_ptr).try_claim() {
                        return Ok(std::mem::transmute::<
                            queue::ReadClaim<'_>,
                            queue::ReadClaim<'_>,
                        >(claim));
                    }
                }

                if self.shared.sender_count.load(Ordering::Relaxed) == 0 {
                    return Err(RecvError::Disconnected);
                }

                return Err(RecvError::Timeout);
            }

            // None case: reset backoff and loop
            backoff.reset();
        }
    }

    /// Attempts to receive a message without blocking.
    ///
    /// Returns `None` if no message is available.
    #[inline]
    pub fn try_recv(&mut self) -> Option<queue::ReadClaim<'_>> {
        self.inner.try_claim()
    }

    /// Returns the capacity of the underlying buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns `true` if all senders have been dropped.
    #[inline]
    pub fn is_disconnected(&self) -> bool {
        self.shared.sender_count.load(Ordering::Relaxed) == 0
    }
}

impl Drop for Receiver {
    fn drop(&mut self) {
        self.shared
            .receiver_disconnected
            .store(true, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for Receiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Receiver")
            .field("capacity", &self.capacity())
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn basic_send_recv() {
        let (mut tx, mut rx) = channel(1024);

        let payload = b"hello world";
        let mut claim = tx.send(payload.len()).unwrap();
        claim.copy_from_slice(payload);
        claim.commit();
        tx.notify();

        let record = rx.recv(None).unwrap();
        assert_eq!(&*record, payload);
    }

    #[test]
    #[allow(clippy::redundant_clone)]
    fn sender_is_clone() {
        let (tx, _rx) = channel(1024);
        let _tx2 = tx.clone();
    }

    #[test]
    fn multiple_senders() {
        const SENDERS: usize = 4;
        const MESSAGES: usize = 100;

        let (tx, mut rx) = channel(4096);

        let handles: Vec<_> = (0..SENDERS)
            .map(|id| {
                let mut tx = tx.clone();
                thread::spawn(move || {
                    for i in 0..MESSAGES {
                        let payload = format!("{}:{}", id, i);
                        let mut claim = tx.send(payload.len()).unwrap();
                        claim.copy_from_slice(payload.as_bytes());
                        claim.commit();
                        tx.notify();
                    }
                })
            })
            .collect();

        drop(tx);

        let mut count = 0;
        while let Ok(_record) = rx.recv(None) {
            count += 1;
            if count == SENDERS * MESSAGES {
                break;
            }
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(count, SENDERS * MESSAGES);
    }

    #[test]
    fn disconnection_all_senders_dropped() {
        let (tx, mut rx) = channel(1024);

        drop(tx);

        match rx.recv(None) {
            Err(RecvError::Disconnected) => {}
            _ => panic!("expected Disconnected"),
        }
    }

    #[test]
    fn disconnection_receiver_dropped() {
        let (mut tx, rx) = channel(1024);

        drop(rx);

        match tx.send(8) {
            Err(ChannelClosed) => {}
            _ => panic!("expected ChannelClosed"),
        }
    }

    #[test]
    fn recv_timeout_works() {
        let (_tx, mut rx) = channel(1024);

        let start = std::time::Instant::now();
        let result = rx.recv(Some(Duration::from_millis(50)));
        let elapsed = start.elapsed();

        assert!(matches!(result, Err(RecvError::Timeout)));
        assert!(elapsed >= Duration::from_millis(40));
        assert!(elapsed < Duration::from_millis(200));
    }

    #[test]
    #[should_panic(expected = "payload length must be non-zero")]
    fn send_zero_panics() {
        let (mut tx, _rx) = channel(1024);
        let _ = tx.send(0);
    }

    #[test]
    #[should_panic(expected = "payload length must be non-zero")]
    fn try_send_zero_panics() {
        let (mut tx, _rx) = channel(1024);
        let _ = tx.try_send(0);
    }

    /// High-volume stress test with multiple senders.
    #[test]
    fn stress_multiple_senders() {
        const SENDERS: usize = 4;
        const MESSAGES_PER_SENDER: u64 = 10_000;
        const TOTAL: u64 = SENDERS as u64 * MESSAGES_PER_SENDER;
        const BUFFER_SIZE: usize = 64 * 1024;

        let (tx, mut rx) = channel(BUFFER_SIZE);

        let handles: Vec<_> = (0..SENDERS)
            .map(|sender_id| {
                let mut tx = tx.clone();
                thread::spawn(move || {
                    for i in 0..MESSAGES_PER_SENDER {
                        // Encode sender_id and sequence in payload
                        let mut payload = [0u8; 16];
                        payload[..8].copy_from_slice(&(sender_id as u64).to_le_bytes());
                        payload[8..].copy_from_slice(&i.to_le_bytes());

                        {
                            let mut claim = tx.send(16).unwrap();
                            claim.copy_from_slice(&payload);
                            claim.commit();
                        }
                        tx.notify();
                    }
                })
            })
            .collect();

        drop(tx);

        // Track per-sender sequence to verify ordering
        let consumer = thread::spawn(move || {
            let mut received = 0u64;
            let mut per_sender = vec![0u64; SENDERS];

            while received < TOTAL {
                match rx.recv(None) {
                    Ok(record) => {
                        let sender_id =
                            u64::from_le_bytes(record[..8].try_into().unwrap()) as usize;
                        let seq = u64::from_le_bytes(record[8..].try_into().unwrap());

                        // Each sender's messages should arrive in order
                        assert_eq!(
                            seq, per_sender[sender_id],
                            "sender {} out of order at {}",
                            sender_id, received
                        );
                        per_sender[sender_id] += 1;
                        received += 1;
                    }
                    Err(RecvError::Timeout) => unreachable!(),
                    Err(RecvError::Disconnected) => break,
                }
            }

            per_sender
        });

        for h in handles {
            h.join().unwrap();
        }

        let per_sender = consumer.join().unwrap();
        for (i, &count) in per_sender.iter().enumerate() {
            assert_eq!(count, MESSAGES_PER_SENDER, "sender {} count", i);
        }
    }
}

//! High-performance lock-free ring buffers for variable-length messages.
//!
//! This crate provides bounded SPSC and MPSC byte ring buffers optimized for
//! getting data off the hot path without disturbing it. No allocation, no
//! formatting, no syscalls on the producer side.
//!
//! # Modules
//!
//! - [`queue`]: Low-level ring buffer primitives. No blocking, maximum control.
//! - [`channel`]: Ergonomic channel API with backoff and parking for receivers.
//!
//! # Design
//!
//! - **Flat byte buffer** with free-running offsets, power-of-2 capacity
//! - **len-as-commit**: Record's len field is the commit marker (non-zero = ready)
//! - **Skip markers**: High bit of len distinguishes padding/aborted claims
//! - **Consumer zeroing**: Consumer zeros records before releasing space
//! - **Claim-based API**: `WriteClaim`/`ReadClaim` with RAII semantics
//!
//! # Channel Philosophy
//!
//! **Senders are never slowed down.** They use brief backoff (spin + yield) but
//! never syscall. If the buffer is full, they return an error immediately.
//!
//! **Receivers can block.** They use `park_timeout` to wait for messages without
//! burning CPU, but always with a timeout to check for disconnection.
//!
//! # Example (Queue API)
//!
//! ```
//! use nexus_logbuf::queue::spsc;
//!
//! let (mut producer, mut consumer) = spsc::new(4096);
//!
//! // Producer (hot path)
//! let payload = b"hello world";
//! if let Ok(mut claim) = producer.try_claim(payload.len()) {
//!     claim.copy_from_slice(payload);
//!     claim.commit();
//! }
//!
//! // Consumer (background thread)
//! if let Some(record) = consumer.try_claim() {
//!     assert_eq!(&*record, b"hello world");
//!     // record dropped here -> zeros region, advances head
//! }
//! ```

#![warn(missing_docs)]

pub mod channel;
pub mod queue;

// Re-export for convenience (queue is the primitive layer)
pub use queue::mpsc;
pub use queue::spsc;

/// Error returned from queue `try_claim` operations when the buffer has no
/// space for the requested record.
///
/// This is the only failure mode that `try_claim` can surface as an error:
/// passing `len == 0` is a precondition violation and panics (`len == 0` is
/// reserved by the wire format as the "uncommitted" sentinel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferFull;

impl std::fmt::Display for BufferFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("buffer full")
    }
}

impl std::error::Error for BufferFull {}

/// Align a value up to the next multiple of 8.
#[inline]
pub(crate) const fn align8(n: usize) -> usize {
    (n + 7) & !7
}

/// Record header constants.
///
/// The len field is a `usize` (system word) and uses the high bit as a skip
/// marker:
/// - `len == 0`: Not committed, consumer waits
/// - `len > 0, high bit clear`: Committed record, payload is `len` bytes
/// - `len high bit set`: Skip marker, advance by `len & LEN_MASK` bytes
pub(crate) const SKIP_BIT: usize = 1 << (usize::BITS - 1);
pub(crate) const LEN_MASK: usize = !SKIP_BIT;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align8_works() {
        assert_eq!(align8(0), 0);
        assert_eq!(align8(1), 8);
        assert_eq!(align8(7), 8);
        assert_eq!(align8(8), 8);
        assert_eq!(align8(9), 16);
        assert_eq!(align8(15), 16);
        assert_eq!(align8(16), 16);
    }
}

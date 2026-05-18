//! Multi-producer single-consumer byte ring buffer.
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ Shared:                                                                 │
//! │   head: CachePadded<AtomicUsize>  ← Consumer writes, producers read     │
//! │   tail: CachePadded<AtomicUsize>  ← Producers CAS to claim space        │
//! │   buffer: *mut u8                                                       │
//! │   capacity: usize                 (power of 2)                          │
//! │   mask: usize                     (capacity - 1)                        │
//! └─────────────────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
//! │ Producer (cloneable):           │   │ Consumer:                       │
//! │   cached_head: usize (local)    │   │   head: usize        (local)    │
//! │   shared: Arc<Shared>           │   │                                 │
//! └─────────────────────────────────┘   └─────────────────────────────────┘
//! ```
//!
//! # Differences from SPSC
//!
//! - Tail is atomic in shared state (not local to producer)
//! - Producers use CAS loop to claim space
//! - Producer is `Clone` - multiple producers allowed
//! - Synchronization: Relaxed CAS on tail, Release on len commit, Acquire on len read
//!
//! # Record Layout
//!
//! Same as SPSC - see [`crate::spsc`] for details.

use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::cell::Cell;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering, fence};

use crossbeam_utils::CachePadded;

use crate::{BufferFull, LEN_MASK, SKIP_BIT, align8};

/// Header size in bytes — one system word (`usize`).
///
/// On 64-bit this is 8 bytes, ensuring the payload starts at 8-byte alignment.
const HEADER_SIZE: usize = std::mem::size_of::<usize>();

/// Creates a bounded MPSC byte ring buffer.
///
/// Capacity is rounded up to the next power of two.
///
/// # Panics
///
/// Panics if `capacity` is zero or less than 16 bytes.
pub fn new(capacity: usize) -> (Producer, Consumer) {
    assert!(capacity >= 16, "capacity must be at least 16 bytes");

    let capacity = capacity.next_power_of_two();
    let mask = capacity - 1;

    // Allocate buffer, zero-initialized, 8-byte aligned for atomic len stamps
    let layout = Layout::from_size_align(capacity, 8)
        .expect("valid layout: capacity is a power of two >= 16, align is 8");
    // SAFETY: Layout is valid — capacity >= 16 (power of two), align is 8.
    let buffer_ptr = unsafe { alloc_zeroed(layout) };
    if buffer_ptr.is_null() {
        handle_alloc_error(layout);
    }

    let shared = Arc::new(Shared {
        head: CachePadded::new(AtomicUsize::new(0)),
        tail: CachePadded::new(AtomicUsize::new(0)),
        buffer: buffer_ptr,
        capacity,
        mask,
    });

    (
        Producer {
            cached_head: Cell::new(0),
            shared: Arc::clone(&shared),
        },
        Consumer {
            head: Cell::new(0),
            shared,
        },
    )
}

struct Shared {
    /// Consumer's read position. Updated by consumer, read by producers.
    head: CachePadded<AtomicUsize>,
    /// Producers' write position. CAS'd by producers.
    tail: CachePadded<AtomicUsize>,
    /// Buffer pointer.
    buffer: *mut u8,
    /// Buffer capacity (power of 2).
    capacity: usize,
    /// Mask for wrapping (capacity - 1).
    mask: usize,
}

// SAFETY: Buffer access is synchronized through atomic head/tail.
// Multiple producers coordinate via CAS on tail.
// Single consumer is enforced by API (Consumer is not Clone).
unsafe impl Send for Shared {}
// SAFETY: Producers access disjoint CAS-claimed regions; consumer
// accesses only committed regions released by the atomic len store.
unsafe impl Sync for Shared {}

impl Drop for Shared {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, 8)
            .expect("valid layout: capacity was validated at construction");
        // SAFETY: buffer was allocated with alloc_zeroed using this exact layout.
        // Shared is only dropped once (Arc prevents earlier drops).
        unsafe { dealloc(self.buffer, layout) };
    }
}

// ============================================================================
// Producer
// ============================================================================

/// Producer endpoint of the MPSC ring buffer.
///
/// This type is `Clone` - multiple producers can write concurrently.
/// Use [`try_claim`](Producer::try_claim) to claim space for writing.
#[derive(Clone)]
pub struct Producer {
    /// Cached head position (Rigtorp-style optimization, per-producer).
    cached_head: Cell<usize>,
    /// Shared state.
    shared: Arc<Shared>,
}

// SAFETY: Producer coordinates with other producers via atomic CAS on
// tail. Each producer's claimed region is disjoint after successful CAS.
unsafe impl Send for Producer {}

impl Producer {
    /// Attempts to claim space for a record with the given payload length.
    ///
    /// Returns a [`WriteClaim`] that can be written to and then committed.
    ///
    /// # Errors
    ///
    /// Returns [`BufferFull`] if the buffer has no space for the record.
    ///
    /// # Panics
    ///
    /// Panics if `len == 0`. The wire format reserves `len == 0` as the
    /// "uncommitted" sentinel — letting it through would silently hang the
    /// consumer. Aborting a non-zero claim is fully supported (drop the
    /// [`WriteClaim`] without committing); only claiming zero bytes upfront
    /// is forbidden.
    ///
    /// # Safety Contract
    ///
    /// `len` must not exceed `LEN_MASK`. On 64-bit this is ~9.2 exabytes
    /// (unreachable in practice). On 32-bit, records >2GB could set
    /// `SKIP_BIT` and corrupt the stream — enforced with `assert!`.
    #[inline]
    pub fn try_claim(&mut self, len: usize) -> Result<WriteClaim<'_>, BufferFull> {
        assert!(len > 0, "payload length must be non-zero");
        #[cfg(target_pointer_width = "32")]
        assert!(len <= LEN_MASK, "payload too large for 32-bit logbuf");
        #[cfg(not(target_pointer_width = "32"))]
        debug_assert!(len <= LEN_MASK, "payload too large");

        let record_size = align8(HEADER_SIZE + len);

        // CAS loop to claim space
        loop {
            let tail = self.shared.tail.load(Ordering::Relaxed);

            // Calculate used space. If cached_head is stale, used can exceed capacity.
            // saturating_sub handles this gracefully (returns 0 if stale).
            let used = tail.wrapping_sub(self.cached_head.get());
            let available = self.shared.capacity.saturating_sub(used);

            if available < record_size {
                // Reload head from shared state
                self.cached_head
                    .set(self.shared.head.load(Ordering::Relaxed));
                fence(Ordering::Acquire);

                let used = tail.wrapping_sub(self.cached_head.get());
                if used > self.shared.capacity || self.shared.capacity - used < record_size {
                    return Err(BufferFull);
                }
            }

            // Check if record fits before buffer end, or needs wrap
            let offset = tail & self.shared.mask;
            let space_to_end = self.shared.capacity - offset;

            if space_to_end < record_size {
                // Need to wrap. Check if we have space for padding + record at start.
                let total_needed = space_to_end + record_size;

                let used = tail.wrapping_sub(self.cached_head.get());
                let available = self.shared.capacity.saturating_sub(used);

                if available < total_needed {
                    // Reload and recheck
                    self.cached_head
                        .set(self.shared.head.load(Ordering::Relaxed));
                    fence(Ordering::Acquire);

                    let used = tail.wrapping_sub(self.cached_head.get());
                    if used > self.shared.capacity || self.shared.capacity - used < total_needed {
                        return Err(BufferFull);
                    }
                }

                // Try to claim the padding + record space
                let new_tail = tail.wrapping_add(total_needed);
                if self
                    .shared
                    .tail
                    .compare_exchange_weak(tail, new_tail, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    // We claimed the space. Write padding skip marker.
                    let buffer = self.shared.buffer;
                    let skip_len = space_to_end | SKIP_BIT;

                    // Release fence before writing skip marker
                    fence(Ordering::Release);
                    // SAFETY: offset is masked to [0, capacity), 8-byte aligned.
                    // CAS success guarantees exclusive ownership of this region.
                    let len_ptr = unsafe { buffer.add(offset) }.cast::<AtomicUsize>();
                    // SAFETY: len_ptr points to a valid, aligned usize within the buffer.
                    unsafe { &*len_ptr }.store(skip_len, Ordering::Relaxed);

                    return Ok(WriteClaim {
                        shared: &self.shared,
                        offset: 0, // Record starts at beginning after wrap
                        len,
                        record_size,
                        committed: false,
                    });
                }
                // CAS failed — another producer claimed first. Pause to
                // reduce pipeline pressure, then retry with fresh tail.
                core::hint::spin_loop();
                continue;
            }

            // Fits without wrapping
            let new_tail = tail.wrapping_add(record_size);
            if self
                .shared
                .tail
                .compare_exchange_weak(tail, new_tail, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(WriteClaim {
                    shared: &self.shared,
                    offset,
                    len,
                    record_size,
                    committed: false,
                });
            }
            // CAS failed — pause before retry.
            core::hint::spin_loop();
        }
    }

    /// Returns the capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.shared.capacity
    }

    /// Best-effort hint: returns `true` if the consumer has likely been dropped.
    ///
    /// Uses `Arc::strong_count` which is inherently racy. For reliable
    /// disconnection detection, use the channel layer (`channel::mpsc`).
    #[inline]
    pub fn is_disconnected(&self) -> bool {
        Arc::strong_count(&self.shared) == 1
    }
}

impl std::fmt::Debug for Producer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Producer")
            .field("capacity", &self.capacity())
            .finish_non_exhaustive()
    }
}

// ============================================================================
// WriteClaim
// ============================================================================

/// A claimed region for writing a record.
///
/// Dereferences to `&mut [u8]` for the payload region. Call [`commit`](WriteClaim::commit)
/// when done writing to publish the record. If dropped without committing, a skip
/// marker is written so the consumer can advance past the dead region.
///
/// # Important
///
/// Leaking a `WriteClaim` via [`mem::forget`](std::mem::forget) will permanently
/// block the consumer at this record's offset. This is not undefined behavior
/// but causes an unrecoverable deadlock. Always drop or explicitly abort claims.
pub struct WriteClaim<'a> {
    shared: &'a Shared,
    offset: usize,
    len: usize,
    record_size: usize,
    committed: bool,
}

impl WriteClaim<'_> {
    /// Commits the record, making it visible to the consumer.
    #[inline]
    pub fn commit(mut self) {
        self.do_commit();
        self.committed = true;
    }

    #[inline]
    fn do_commit(&mut self) {
        let buffer = self.shared.buffer;
        // SAFETY: offset is within [0, capacity), 8-byte aligned. Buffer is valid.
        let len_ptr = unsafe { buffer.add(self.offset) }.cast::<AtomicUsize>();

        // Release fence: ensures payload writes are visible before len store
        fence(Ordering::Release);
        // SAFETY: len_ptr points to a valid, aligned usize within the buffer.
        unsafe { &*len_ptr }.store(self.len, Ordering::Relaxed);
    }

    /// Returns the length of the payload region.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the payload is empty (always false, len must be > 0).
    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }
}

impl Deref for WriteClaim<'_> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        let buffer = self.shared.buffer;
        // SAFETY: offset + HEADER_SIZE is within the buffer. CAS atomically
        // advances tail past this region before the pointer is constructed,
        // guaranteeing disjoint regions — no two WriteClaims overlap in the buffer.
        let payload_ptr = unsafe { buffer.add(self.offset + HEADER_SIZE) };
        // SAFETY: payload_ptr is valid for self.len bytes, word-aligned,
        // and exclusively owned by this claim (disjoint CAS regions).
        // Lifetime tied to &self.
        unsafe { std::slice::from_raw_parts(payload_ptr, self.len) }
    }
}

impl DerefMut for WriteClaim<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let buffer = self.shared.buffer;
        // SAFETY: offset + HEADER_SIZE is within the buffer. CAS atomically
        // advances tail past this region before the pointer is constructed,
        // guaranteeing disjoint regions — no two WriteClaims overlap in the buffer.
        let payload_ptr = unsafe { buffer.add(self.offset + HEADER_SIZE) };
        // SAFETY: payload_ptr is valid for self.len bytes, word-aligned,
        // and exclusively owned by this claim (disjoint CAS regions).
        // Lifetime tied to &mut self.
        unsafe { std::slice::from_raw_parts_mut(payload_ptr, self.len) }
    }
}

impl Drop for WriteClaim<'_> {
    fn drop(&mut self) {
        if !self.committed {
            // Write skip marker so consumer can advance past this region
            let buffer = self.shared.buffer;
            // SAFETY: offset is within [0, capacity), 8-byte aligned. Buffer is valid.
            let len_ptr = unsafe { buffer.add(self.offset) }.cast::<AtomicUsize>();
            let skip_len = self.record_size | SKIP_BIT;

            fence(Ordering::Release);
            // SAFETY: len_ptr points to a valid, aligned usize within the buffer.
            unsafe { &*len_ptr }.store(skip_len, Ordering::Relaxed);
        }
    }
}

// ============================================================================
// Consumer
// ============================================================================

/// Consumer endpoint of the MPSC ring buffer.
///
/// Use [`try_claim`](Consumer::try_claim) to claim the next record for reading.
/// This type is NOT `Clone` - only one consumer is allowed.
pub struct Consumer {
    /// Local head position (free-running).
    head: Cell<usize>,
    /// Shared state.
    shared: Arc<Shared>,
}

// SAFETY: Consumer is only used from one thread (not Clone, &mut self API).
// Sending it to another thread is safe; using from multiple threads is not.
unsafe impl Send for Consumer {}

impl Consumer {
    /// Attempts to claim the next record for reading.
    ///
    /// Returns a [`ReadClaim`] if a record is available. The claim dereferences
    /// to `&[u8]` for the payload. When dropped, the record region is zeroed
    /// and the head is advanced.
    ///
    /// Returns `None` if no committed record is available.
    #[inline]
    pub fn try_claim(&mut self) -> Option<ReadClaim<'_>> {
        let buffer = self.shared.buffer;

        loop {
            let offset = self.head.get() & self.shared.mask;
            // SAFETY: offset is masked to [0, capacity), always 8-byte aligned
            // (head advances by align8'd record sizes). Buffer is valid.
            let len_ptr = unsafe { buffer.add(offset) }.cast::<AtomicUsize>();

            // Relaxed atomic load, then Acquire fence for payload visibility
            // SAFETY: len_ptr points to a valid, aligned usize within the buffer.
            let len_raw = unsafe { &*len_ptr }.load(Ordering::Relaxed);
            fence(Ordering::Acquire);

            if len_raw == 0 {
                // Not committed yet
                return None;
            }

            if len_raw & SKIP_BIT != 0 {
                // Skip marker: zero the region and advance
                let skip_size = len_raw & LEN_MASK;
                // Zero payload first, then stamp last (mirrors write path)
                if skip_size > HEADER_SIZE {
                    // SAFETY: offset + HEADER_SIZE .. offset + skip_size is within
                    // the buffer. Consumer has exclusive read access to this region.
                    unsafe {
                        ptr::write_bytes(
                            buffer.add(offset + HEADER_SIZE),
                            0,
                            skip_size - HEADER_SIZE,
                        );
                    }
                }
                // Ensure payload zeroing completes before clearing stamp
                fence(Ordering::Release);
                // SAFETY: len_ptr is still valid, computed above.
                unsafe { &*len_ptr }.store(0, Ordering::Relaxed);

                self.head.set(self.head.get().wrapping_add(skip_size));

                // Ensure stamp clear completes before head advance
                fence(Ordering::Release);
                self.shared.head.store(self.head.get(), Ordering::Relaxed);

                // Continue to check next position
                continue;
            }

            // Valid record
            let len = len_raw;
            let record_size = align8(HEADER_SIZE + len);

            return Some(ReadClaim {
                consumer: self,
                offset,
                len,
                record_size,
            });
        }
    }

    /// Returns the capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.shared.capacity
    }

    /// Best-effort hint: returns `true` if all producers have likely been dropped.
    ///
    /// See producer's [`is_disconnected`](Producer::is_disconnected) for caveats.
    #[inline]
    pub fn is_disconnected(&self) -> bool {
        Arc::strong_count(&self.shared) == 1
    }
}

impl std::fmt::Debug for Consumer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Consumer")
            .field("capacity", &self.capacity())
            .finish_non_exhaustive()
    }
}

// ============================================================================
// ReadClaim
// ============================================================================

/// A claimed record for reading.
///
/// Dereferences to `&[u8]` for the payload. When dropped, the record region
/// is zeroed and the head is advanced, freeing space for producers.
pub struct ReadClaim<'a> {
    consumer: &'a mut Consumer,
    offset: usize,
    len: usize,
    record_size: usize,
}

impl ReadClaim<'_> {
    /// Returns the length of the payload.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the payload is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Deref for ReadClaim<'_> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        let buffer = self.consumer.shared.buffer;
        // SAFETY: offset + HEADER_SIZE is within the buffer. The claim owns
        // exclusive read access via &mut Consumer borrow.
        let payload_ptr = unsafe { buffer.add(self.offset + HEADER_SIZE) };
        // SAFETY: payload_ptr is valid for self.len bytes. The producer has
        // finished writing (len was non-zero, preceded by Release fence).
        unsafe { std::slice::from_raw_parts(payload_ptr, self.len) }
    }
}

impl Drop for ReadClaim<'_> {
    fn drop(&mut self) {
        let buffer = self.consumer.shared.buffer;

        // Zero payload first, then stamp last (mirrors write path)
        if self.record_size > HEADER_SIZE {
            // SAFETY: offset + HEADER_SIZE .. offset + record_size is within
            // the buffer. Consumer owns this region exclusively.
            unsafe {
                ptr::write_bytes(
                    buffer.add(self.offset + HEADER_SIZE),
                    0,
                    self.record_size - HEADER_SIZE,
                );
            }
        }
        // Ensure payload zeroing completes before clearing stamp
        fence(Ordering::Release);
        // SAFETY: offset is within [0, capacity), 8-byte aligned. Buffer is valid.
        let len_ptr = unsafe { buffer.add(self.offset) }.cast::<AtomicUsize>();
        // SAFETY: len_ptr points to a valid, aligned usize within the buffer.
        unsafe { &*len_ptr }.store(0, Ordering::Relaxed);

        // Advance head
        let new_head = self.consumer.head.get().wrapping_add(self.record_size);
        self.consumer.head.set(new_head);

        // Ensure stamp clear completes before head advance
        fence(Ordering::Release);
        self.consumer.shared.head.store(new_head, Ordering::Relaxed);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_write_read() {
        let (mut prod, mut cons) = new(1024);

        let payload = b"hello world";
        let mut claim = prod.try_claim(payload.len()).unwrap();
        claim.copy_from_slice(payload);
        claim.commit();

        let record = cons.try_claim().unwrap();
        assert_eq!(&*record, payload);
    }

    #[test]
    fn empty_returns_none() {
        let (_, mut cons) = new(1024);
        assert!(cons.try_claim().is_none());
    }

    #[test]
    fn multiple_records() {
        let (mut prod, mut cons) = new(1024);

        for i in 0..10 {
            let payload = format!("message {}", i);
            let mut claim = prod.try_claim(payload.len()).unwrap();
            claim.copy_from_slice(payload.as_bytes());
            claim.commit();
        }

        for i in 0..10 {
            let record = cons.try_claim().unwrap();
            let expected = format!("message {}", i);
            assert_eq!(&*record, expected.as_bytes());
        }

        assert!(cons.try_claim().is_none());
    }

    #[test]
    #[allow(clippy::redundant_clone)]
    fn producer_is_clone() {
        let (prod, _cons) = new(1024);
        let _prod2 = prod.clone();
    }

    #[test]
    fn multiple_producers_single_consumer() {
        use std::thread;

        const PRODUCERS: usize = 4;
        const MESSAGES_PER_PRODUCER: u64 = 10_000;
        const TOTAL: u64 = PRODUCERS as u64 * MESSAGES_PER_PRODUCER;

        let (prod, mut cons) = new(64 * 1024);

        let handles: Vec<_> = (0..PRODUCERS)
            .map(|producer_id| {
                let mut prod = prod.clone();
                thread::spawn(move || {
                    for i in 0..MESSAGES_PER_PRODUCER {
                        // Encode producer_id and sequence in payload
                        let mut payload = [0u8; 16];
                        payload[..8].copy_from_slice(&(producer_id as u64).to_le_bytes());
                        payload[8..].copy_from_slice(&i.to_le_bytes());

                        loop {
                            match prod.try_claim(16) {
                                Ok(mut claim) => {
                                    claim.copy_from_slice(&payload);
                                    claim.commit();
                                    break;
                                }
                                Err(_) => std::hint::spin_loop(),
                            }
                        }
                    }
                })
            })
            .collect();

        // Drop original producer
        drop(prod);

        // Consumer: track per-producer sequence
        let consumer = thread::spawn(move || {
            let mut received = 0u64;
            let mut per_producer = vec![0u64; PRODUCERS];

            while received < TOTAL {
                if let Some(record) = cons.try_claim() {
                    let producer_id = u64::from_le_bytes(record[..8].try_into().unwrap()) as usize;
                    let seq = u64::from_le_bytes(record[8..].try_into().unwrap());

                    // Each producer's messages should arrive in order
                    assert_eq!(
                        seq, per_producer[producer_id],
                        "producer {} out of order",
                        producer_id
                    );
                    per_producer[producer_id] += 1;
                    received += 1;
                } else {
                    std::hint::spin_loop();
                }
            }

            per_producer
        });

        for h in handles {
            h.join().unwrap();
        }

        let per_producer = consumer.join().unwrap();
        for (i, &count) in per_producer.iter().enumerate() {
            assert_eq!(count, MESSAGES_PER_PRODUCER, "producer {} count", i);
        }
    }

    #[test]
    fn aborted_claim_creates_skip() {
        let (mut prod, mut cons) = new(1024);

        // Claim and drop without committing
        {
            let mut claim = prod.try_claim(10).unwrap();
            claim.copy_from_slice(b"0123456789");
            // drop without commit
        }

        // Write another record
        {
            let mut claim = prod.try_claim(5).unwrap();
            claim.copy_from_slice(b"hello");
            claim.commit();
        }

        // Consumer should skip the aborted record and read the committed one
        let record = cons.try_claim().unwrap();
        assert_eq!(&*record, b"hello");
    }

    #[test]
    fn wrap_around() {
        let (mut prod, mut cons) = new(64);

        // Fill with messages that will cause wrap-around
        for i in 0..20 {
            let payload = format!("msg{:02}", i);
            loop {
                match prod.try_claim(payload.len()) {
                    Ok(mut claim) => {
                        claim.copy_from_slice(payload.as_bytes());
                        claim.commit();
                        break;
                    }
                    Err(_) => {
                        // Drain some
                        while cons.try_claim().is_some() {}
                    }
                }
            }
        }
    }

    #[test]
    fn full_returns_error() {
        let (mut prod, _cons) = new(64);

        // Fill the buffer
        let mut count = 0;
        while let Ok(mut claim) = prod.try_claim(8) {
            claim.copy_from_slice(b"12345678");
            claim.commit();
            count += 1;
        }

        assert!(count > 0);
        assert!(prod.try_claim(8).is_err());
    }

    #[test]
    fn disconnection_detection() {
        let (prod, cons) = new(1024);

        assert!(!prod.is_disconnected());
        assert!(!cons.is_disconnected());

        drop(cons);
        assert!(prod.is_disconnected());
    }

    #[test]
    #[should_panic(expected = "capacity must be at least 16")]
    fn tiny_capacity_panics() {
        let _ = new(8);
    }

    #[test]
    #[should_panic(expected = "payload length must be non-zero")]
    fn zero_len_panics() {
        let (mut prod, _) = new(1024);
        let _ = prod.try_claim(0);
    }

    #[test]
    fn capacity_rounds_to_power_of_two() {
        let (prod, _) = new(100);
        assert_eq!(prod.capacity(), 128);

        let (prod, _) = new(1000);
        assert_eq!(prod.capacity(), 1024);
    }

    /// High-volume stress test with multiple producers.
    #[test]
    fn stress_multiple_producers() {
        use std::thread;

        const PRODUCERS: usize = 4;
        const COUNT_PER_PRODUCER: u64 = 100_000;
        const TOTAL: u64 = PRODUCERS as u64 * COUNT_PER_PRODUCER;
        const BUFFER_SIZE: usize = 64 * 1024;

        let (prod, mut cons) = new(BUFFER_SIZE);

        let handles: Vec<_> = (0..PRODUCERS)
            .map(|_| {
                let mut prod = prod.clone();
                thread::spawn(move || {
                    for i in 0..COUNT_PER_PRODUCER {
                        let payload = i.to_le_bytes();
                        loop {
                            match prod.try_claim(payload.len()) {
                                Ok(mut claim) => {
                                    claim.copy_from_slice(&payload);
                                    claim.commit();
                                    break;
                                }
                                Err(_) => std::hint::spin_loop(),
                            }
                        }
                    }
                })
            })
            .collect();

        drop(prod);

        let consumer = thread::spawn(move || {
            let mut received = 0u64;
            let mut sum = 0u64;
            while received < TOTAL {
                if let Some(record) = cons.try_claim() {
                    let value = u64::from_le_bytes((*record).try_into().unwrap());
                    sum = sum.wrapping_add(value);
                    received += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
            (received, sum)
        });

        for h in handles {
            h.join().unwrap();
        }

        let (received, sum) = consumer.join().unwrap();
        assert_eq!(received, TOTAL);

        // Each producer sends 0..COUNT_PER_PRODUCER
        // Sum per producer = COUNT_PER_PRODUCER * (COUNT_PER_PRODUCER - 1) / 2
        let expected_sum = PRODUCERS as u64 * COUNT_PER_PRODUCER * (COUNT_PER_PRODUCER - 1) / 2;
        assert_eq!(sum, expected_sum);
    }
}

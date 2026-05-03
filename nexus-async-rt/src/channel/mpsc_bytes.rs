//! Bounded cross-thread MPSC byte channel.
//!
//! Variable-length messages over `nexus_logbuf::mpsc`. Multiple senders
//! can write `&[u8]` into claim regions and commit. The single consumer
//! reads `ReadClaim` references that deref to `&[u8]`.
//!
//! Zero allocation on the send/recv hot path. Must be created inside
//! [`Runtime::block_on`](crate::Runtime::block_on).
//!
//! ```ignore
//! use nexus_async_rt::channel::mpsc_bytes;
//!
//! let (tx, mut rx) = mpsc_bytes::channel(64 * 1024);
//!
//! // Clone sender for multiple producers
//! let tx2 = tx.clone();
//!
//! // Claim, write, commit (zero-copy)
//! let mut claim = tx.claim(5).await?;
//! claim.copy_from_slice(b"hello");
//! claim.commit();
//!
//! // Or from another sender
//! tx2.send(b"world").await?;
//!
//! // Receive
//! let msg = rx.recv().await?;
//! assert_eq!(&*msg, b"hello");
//! drop(msg);  // advances consumer head
//! ```

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::task::{Poll, Waker};

use std::ops::{Deref, DerefMut};

// =============================================================================
// Waker primitives (same pattern as spsc_bytes / mpsc typed)
// =============================================================================

const EMPTY: u8 = 0;
const STORED: u8 = 1;
const REGISTERING: u8 = 2;

struct RxWakerSlot {
    task_ptr: std::sync::atomic::AtomicPtr<u8>,
    cross_ctx: *const crate::cross_wake::CrossWakeContext,
    state: AtomicU8,
}

unsafe impl Send for RxWakerSlot {}
unsafe impl Sync for RxWakerSlot {}

impl RxWakerSlot {
    fn new(cross_ctx: *const crate::cross_wake::CrossWakeContext) -> Self {
        Self {
            task_ptr: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            cross_ctx,
            state: AtomicU8::new(EMPTY),
        }
    }

    /// Register the receiver's task pointer. Single-registerer only.
    fn register(&self, task_ptr: *mut u8) {
        debug_assert!(
            !task_ptr.is_null(),
            "RxWakerSlot::register called with null task_ptr"
        );
        let prev = self.state.swap(REGISTERING, Ordering::Acquire);
        debug_assert_ne!(prev, REGISTERING);

        // BUG-2 (#168) fix: hold a refcount on the task while
        // registered so a sender that captures the pointer
        // mid-`wake()` can't have it freed underneath. Matched by
        // ref_dec in wake/clear/Drop.
        // SAFETY: caller (RecvFut::poll) just received task_ptr from
        // the active receiver task whose refcount is >= 1; the
        // debug_assert above catches the null case in development.
        unsafe { crate::task::ref_inc(task_ptr) };

        // Release any prior registration's ref. Always check prev_ptr —
        // not gated on `prev == STORED` — because a sender's `wake()` CAS
        // may have transitioned state STORED→EMPTY without yet taking
        // the task_ptr (the swap is the second step). In that race
        // window, prev_ptr is still non-null even though state was
        // EMPTY. Skipping the release leaks the ref. (BUG-2 follow-up.)
        //
        // SAFETY: prev_ptr (if non-null) was registered with a ref_inc;
        // we own that ref now and must release it.
        let prev_ptr = self.task_ptr.swap(task_ptr, Ordering::AcqRel);
        if !prev_ptr.is_null() {
            unsafe { release_slot_ref(prev_ptr, self.cross_ctx) };
        }

        self.state.store(STORED, Ordering::Release);
    }

    fn try_register_local(&self, waker: &Waker) -> bool {
        crate::waker::task_ptr_from_local_waker(waker).is_some_and(|task_ptr| {
            self.register(task_ptr);
            true
        })
    }

    fn wake(&self) -> bool {
        if self
            .state
            .compare_exchange(STORED, EMPTY, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            let task_ptr = self.task_ptr.swap(std::ptr::null_mut(), Ordering::Acquire);
            if !task_ptr.is_null() {
                // SAFETY: task_ptr is alive because `try_register_local`
                // ref_inc'd before storing — that ref keeps the task
                // allocated through the dispatch (see BUG-2 #168).
                let ctx = unsafe { &*self.cross_ctx };
                unsafe { crate::cross_wake::wake_task_cross_thread(task_ptr, ctx) };

                // BUG-2 fix: release the ref `try_register_local`
                // acquired. AFTER wake_task_cross_thread so the task is
                // alive for its deref.
                // SAFETY: we own the ref from `try_register_local`.
                unsafe { release_slot_ref(task_ptr, self.cross_ctx) };
                return true;
            }
        }
        false
    }

    fn has_waker(&self) -> bool {
        self.state.load(Ordering::Acquire) == STORED
    }

    fn clear(&self) {
        if self
            .state
            .compare_exchange(STORED, EMPTY, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            let task_ptr = self.task_ptr.swap(std::ptr::null_mut(), Ordering::Acquire);
            if !task_ptr.is_null() {
                // BUG-2 fix: release the ref `try_register_local` acquired.
                // SAFETY: we own the ref.
                unsafe { release_slot_ref(task_ptr, self.cross_ctx) };
            }
        }
    }
}

impl Drop for RxWakerSlot {
    fn drop(&mut self) {
        // BUG-2 (#168) fix: if still registered when dropped, release
        // our ref. See mpsc.rs for the full rationale.
        if *self.state.get_mut() == STORED {
            let task_ptr = *self.task_ptr.get_mut();
            if !task_ptr.is_null() {
                // SAFETY: we own the ref from `try_register_local`.
                unsafe { release_slot_ref(task_ptr, self.cross_ctx) };
            }
        }
    }
}

/// Release the slot's ref on `task_ptr`. If terminal, route via
/// [`crate::cross_wake::dispose_terminal`]. See `mpsc::release_slot_ref`
/// for the full design rationale.
///
/// `cross_ctx` is unused (dispose_terminal reads ctx from the task
/// header); kept on the signature for PR 1a consistency.
///
/// # Safety
///
/// `task_ptr` must point to a task on which `try_register_local`
/// previously called `ref_inc`.
unsafe fn release_slot_ref(
    task_ptr: *mut u8,
    _cross_ctx: *const crate::cross_wake::CrossWakeContext,
) {
    match unsafe { crate::task::ref_dec(task_ptr) } {
        crate::task::FreeAction::Retain => {}
        crate::task::FreeAction::FreeBox | crate::task::FreeAction::FreeSlab => {
            // SAFETY: task_ptr was alive until ref_dec; terminal but
            // not yet freed (dispose_terminal does the routing).
            unsafe { crate::cross_wake::dispose_terminal(task_ptr) };
        }
    }
}

struct FallbackWaker {
    state: AtomicU8,
    waker: UnsafeCell<Option<Waker>>,
}

unsafe impl Send for FallbackWaker {}
unsafe impl Sync for FallbackWaker {}

impl FallbackWaker {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(EMPTY),
            waker: UnsafeCell::new(None),
        }
    }

    fn register(&self, waker: &Waker) {
        let prev = self.state.swap(REGISTERING, Ordering::Acquire);
        debug_assert_ne!(prev, REGISTERING);
        unsafe { *self.waker.get() = Some(waker.clone()) };
        self.state.store(STORED, Ordering::Release);
    }

    fn wake(&self) -> bool {
        if self
            .state
            .compare_exchange(STORED, EMPTY, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            if let Some(w) = unsafe { (*self.waker.get()).take() } {
                w.wake();
                return true;
            }
        }
        false
    }

    fn has_waker(&self) -> bool {
        self.state.load(Ordering::Acquire) == STORED
    }
}

impl Drop for FallbackWaker {
    fn drop(&mut self) {
        *self.waker.get_mut() = None;
    }
}

// =============================================================================
// Sender waiter list (intrusive, same pattern as mpsc typed)
// =============================================================================

struct SenderWakerNode {
    waker: UnsafeCell<Option<Waker>>,
    next: std::sync::atomic::AtomicPtr<SenderWakerNode>,
    queued: AtomicBool,
    /// Set when the Sender is dropped while node is in the list.
    /// wake_one skips cancelled nodes.
    cancelled: AtomicBool,
}

unsafe impl Send for SenderWakerNode {}
unsafe impl Sync for SenderWakerNode {}

impl SenderWakerNode {
    fn new() -> Self {
        Self {
            waker: UnsafeCell::new(None),
            next: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            queued: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
        }
    }
}

/// Atomic head pointer for the sender waiter list.
/// Senders CAS-push their node. Receiver pops one and wakes it.
///
/// Each node in the list has its Arc refcount bumped on push and
/// decremented on pop, ensuring the node memory stays valid even
/// if the Sender is dropped while queued.
struct SenderWaitList {
    head: std::sync::atomic::AtomicPtr<SenderWakerNode>,
}

impl SenderWaitList {
    fn new() -> Self {
        Self {
            head: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Push a sender's waker node onto the list. Thread-safe.
    ///
    /// Clones the Arc (bumps refcount) to keep the node alive in the list
    /// independently of the Sender's lifetime.
    fn push(&self, node: &Arc<SenderWakerNode>) {
        let ptr = Arc::as_ptr(node).cast_mut();
        // Bump refcount: the list now holds a reference.
        std::mem::forget(Arc::clone(node));

        unsafe { (*ptr).queued.store(true, Ordering::Relaxed) };
        loop {
            let head = self.head.load(Ordering::Acquire);
            unsafe { (*ptr).next.store(head, Ordering::Relaxed) };
            if self
                .head
                .compare_exchange_weak(head, ptr, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Pop one node and wake it. Called by receiver (single thread).
    /// Skips cancelled nodes (senders that were dropped while queued).
    /// Returns true if a sender was woken.
    fn wake_one(&self) -> bool {
        let head = self.head.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if head.is_null() {
            return false;
        }

        let mut cursor = head;
        let mut woken = false;
        while !cursor.is_null() {
            let next = unsafe { (*cursor).next.load(Ordering::Acquire) };
            let cancelled = unsafe { (*cursor).cancelled.load(Ordering::Acquire) };

            unsafe {
                (*cursor).queued.store(false, Ordering::Release);
                (*cursor)
                    .next
                    .store(std::ptr::null_mut(), Ordering::Relaxed);
            }

            if !cancelled && !woken {
                let waker = unsafe { (*cursor).waker.get().read() };
                unsafe { (*cursor).waker.get().write(None) };
                // Drop the list's Arc refcount for this node.
                unsafe { Arc::decrement_strong_count(cursor) };
                if let Some(w) = waker {
                    w.wake();
                    woken = true;
                }
            } else if !cancelled {
                // Non-cancelled but already woke one -- re-push.
                // Keep the refcount (list still owns it).
                loop {
                    let cur_head = self.head.load(Ordering::Acquire);
                    unsafe { (*cursor).next.store(cur_head, Ordering::Relaxed) };
                    unsafe { (*cursor).queued.store(true, Ordering::Relaxed) };
                    if self
                        .head
                        .compare_exchange_weak(
                            cur_head,
                            cursor,
                            Ordering::AcqRel,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        break;
                    }
                }
            } else {
                // Cancelled: drop the list's Arc refcount.
                unsafe { Arc::decrement_strong_count(cursor) };
            }

            cursor = next;
        }

        woken
    }

    fn has_waiters(&self) -> bool {
        !self.head.load(Ordering::Acquire).is_null()
    }

    /// Wake all waiters. Called when receiver drops.
    fn wake_all(&self) {
        let mut node = self.head.swap(std::ptr::null_mut(), Ordering::AcqRel);
        while !node.is_null() {
            let next = unsafe { (*node).next.load(Ordering::Acquire) };
            let cancelled = unsafe { (*node).cancelled.load(Ordering::Acquire) };
            unsafe {
                (*node).next.store(std::ptr::null_mut(), Ordering::Relaxed);
                (*node).queued.store(false, Ordering::Release);
            }
            if !cancelled {
                let waker = unsafe { (*node).waker.get().read() };
                unsafe { (*node).waker.get().write(None) };
                if let Some(w) = waker {
                    w.wake();
                }
            }
            // Drop the list's Arc refcount.
            unsafe { Arc::decrement_strong_count(node) };
            node = next;
        }
    }
}

// =============================================================================
// Shared state
// =============================================================================

struct Inner {
    rx_slot: RxWakerSlot,
    rx_fallback: FallbackWaker,
    tx_waiters: SenderWaitList,
    _cross_wake_owner: Arc<crate::cross_wake::CrossWakeContext>,
    sender_count: AtomicUsize,
    rx_closed: AtomicBool,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Inner {
    fn wake_rx(&self) {
        if !self.rx_slot.wake() {
            self.rx_fallback.wake();
        }
    }

    fn has_rx_waker(&self) -> bool {
        self.rx_slot.has_waker() || self.rx_fallback.has_waker()
    }
}

// =============================================================================
// WriteClaim wrapper -- auto-notifies receiver on commit
// =============================================================================

/// A claimed write region in the byte channel. Dereferences to `&mut [u8]`.
///
/// Call [`.commit()`](WriteClaim::commit) to publish the record and
/// wake the receiver. Dropping without commit writes a skip marker (abort).
pub struct WriteClaim<'a> {
    inner: nexus_logbuf::queue::mpsc::WriteClaim<'a>,
    notify: &'a Inner,
}

impl WriteClaim<'_> {
    /// Commit the record, making it visible to the receiver.
    /// Automatically wakes the receiver if it's parked.
    pub fn commit(self) {
        let notify = self.notify;
        self.inner.commit();
        if notify.has_rx_waker() {
            notify.wake_rx();
        }
    }

    /// Payload length in bytes.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Always false (claims must have len > 0).
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Deref for WriteClaim<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.inner
    }
}

impl DerefMut for WriteClaim<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.inner
    }
}

// =============================================================================
// ReadClaim wrapper -- auto-wakes sender on drop (frees space)
// =============================================================================

/// A received message from the byte channel. Dereferences to `&[u8]`.
///
/// When dropped, the record region is freed (consumer head advances)
/// and a sender is woken if it was parked on a full buffer.
pub struct ReadClaim<'a> {
    inner: nexus_logbuf::queue::mpsc::ReadClaim<'a>,
    notify: &'a Inner,
}

impl ReadClaim<'_> {
    /// Payload length in bytes.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Always false.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Deref for ReadClaim<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.inner
    }
}

impl Drop for ReadClaim<'_> {
    fn drop(&mut self) {
        // The inner ReadClaim drops after this impl runs (field drop order),
        // which advances the consumer head and frees space. We wake a
        // sender BEFORE inner drops -- the sender will re-try and see space
        // once inner's drop completes. This ordering is acceptable because
        // the sender's try_claim will simply fail and re-park if the space
        // isn't freed yet. On the next poll it succeeds.
        if self.notify.tx_waiters.has_waiters() {
            self.notify.tx_waiters.wake_one();
        }
    }
}

// =============================================================================
// Error types
// =============================================================================

/// Claim failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum ClaimError {
    /// All receivers were dropped.
    Closed,
    /// Requested length exceeds buffer capacity (can never succeed).
    TooLarge,
    /// Requested length is zero (claims must be non-empty).
    ZeroLength,
}

impl std::fmt::Display for ClaimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => f.write_str("byte channel closed"),
            Self::TooLarge => f.write_str("message exceeds buffer capacity"),
            Self::ZeroLength => f.write_str("zero-length claim"),
        }
    }
}

impl std::error::Error for ClaimError {}

/// Receive failed -- all senders dropped and buffer empty.
#[derive(Debug)]
pub struct RecvError;

impl std::fmt::Display for RecvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("byte channel closed")
    }
}

impl std::error::Error for RecvError {}

// =============================================================================
// channel()
// =============================================================================

/// Create a bounded cross-thread MPSC byte channel.
///
/// `capacity` is the ring buffer size in bytes (rounded up to next power of two).
///
/// `Sender` is `Clone + Send` -- multiple producers allowed.
/// `Receiver` is `Send` -- single consumer.
///
/// # Panics
///
/// - Panics if called outside [`Runtime::block_on`](crate::Runtime::block_on).
pub fn channel(capacity: usize) -> (Sender, Receiver) {
    crate::context::assert_in_runtime("mpsc_bytes::channel() called outside Runtime::block_on");

    let cross_ctx = crate::cross_wake::cross_wake_context()
        .expect("mpsc_bytes::channel() requires runtime context");

    let (producer, consumer) = nexus_logbuf::queue::mpsc::new(capacity);
    let rx_slot = RxWakerSlot::new(Arc::as_ptr(&cross_ctx));

    let inner = Arc::new(Inner {
        rx_slot,
        rx_fallback: FallbackWaker::new(),
        tx_waiters: SenderWaitList::new(),
        _cross_wake_owner: cross_ctx,
        sender_count: AtomicUsize::new(1),
        rx_closed: AtomicBool::new(false),
    });

    (
        Sender {
            producer,
            inner: inner.clone(),
            wake_node: Arc::new(SenderWakerNode::new()),
        },
        Receiver { consumer, inner },
    )
}

// =============================================================================
// Sender
// =============================================================================

/// Sending half of a bounded MPSC byte channel.
///
/// `Clone + Send` -- multiple producers allowed.
pub struct Sender {
    producer: nexus_logbuf::queue::mpsc::Producer,
    inner: Arc<Inner>,
    /// Pre-allocated waker node for backpressure parking.
    /// Arc so the node survives in the waiter list after Sender drops.
    wake_node: Arc<SenderWakerNode>,
}

impl Sender {
    /// Claim `len` bytes for zero-copy writing.
    ///
    /// Waits if the buffer is full. Write into the returned `WriteClaim`,
    /// then call `.commit()` to publish. Drop without commit writes a
    /// skip marker (abort).
    ///
    /// Returns `Err(ClaimError::TooLarge)` immediately if `len` exceeds
    /// the buffer capacity (can never succeed).
    pub fn claim(&mut self, len: usize) -> ClaimFut<'_> {
        ClaimFut { sender: self, len }
    }

    /// Try to claim without waiting.
    pub fn try_claim(&mut self, len: usize) -> Result<WriteClaim<'_>, nexus_logbuf::TryClaimError> {
        let inner_claim = self.producer.try_claim(len)?;
        Ok(WriteClaim {
            inner: inner_claim,
            notify: &self.inner,
        })
    }
}

impl Clone for Sender {
    fn clone(&self) -> Self {
        self.inner.sender_count.fetch_add(1, Ordering::Relaxed);
        Self {
            producer: self.producer.clone(),
            inner: self.inner.clone(),
            wake_node: Arc::new(SenderWakerNode::new()),
        }
    }
}

impl Drop for Sender {
    fn drop(&mut self) {
        // Mark our wake node as cancelled. If it's in the waiter list,
        // wake_one/wake_all will skip it (they check cancelled with
        // Acquire before reading the waker). The waker is NOT touched
        // here — wake_one may be reading it concurrently on the
        // receiver thread.
        self.wake_node.cancelled.store(true, Ordering::Release);

        if self.inner.sender_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Last sender dropped -- wake receiver so it sees closed.
            self.inner.wake_rx();
        }
    }
}

// SAFETY: Inner uses atomic operations. Producer is Send. wake_node is owned.
unsafe impl Send for Sender {}

// =============================================================================
// ClaimFut
// =============================================================================

/// Future returned by [`Sender::claim`].
pub struct ClaimFut<'a> {
    sender: &'a mut Sender,
    len: usize,
}

impl<'a> Future for ClaimFut<'a> {
    type Output = Result<WriteClaim<'a>, ClaimError>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { &mut *std::pin::Pin::into_inner_unchecked(self) };
        // SAFETY: Extend the reborrow lifetime to 'a. This is sound because:
        // - ClaimFut holds &'a mut Sender, so the Sender lives for 'a
        // - WriteClaim borrows &mut Producer from that Sender
        // - The future won't be polled again after returning Ready
        let sender: &'a mut Sender = unsafe { &mut *(this.sender as *mut Sender) };

        if sender.inner.rx_closed.load(Ordering::Acquire) {
            return Poll::Ready(Err(ClaimError::Closed));
        }

        if this.len > sender.producer.capacity() {
            return Poll::Ready(Err(ClaimError::TooLarge));
        }

        match sender.producer.try_claim(this.len) {
            Ok(inner_claim) => Poll::Ready(Ok(WriteClaim {
                inner: inner_claim,
                notify: &sender.inner,
            })),
            Err(nexus_logbuf::TryClaimError::Full) => {
                let node = &sender.wake_node;
                if !node.queued.load(Ordering::Acquire) {
                    // Not in list yet -- safe to write waker, then push.
                    // SAFETY: exclusive access -- node not in any shared structure.
                    unsafe { *node.waker.get() = Some(cx.waker().clone()) };
                    sender.inner.tx_waiters.push(node);
                }
                Poll::Pending
            }
            Err(nexus_logbuf::TryClaimError::ZeroLength) => {
                Poll::Ready(Err(ClaimError::ZeroLength))
            }
        }
    }
}

unsafe impl Send for ClaimFut<'_> {}

// =============================================================================
// Receiver
// =============================================================================

/// Receiving half of a bounded MPSC byte channel.
///
/// `Send` but not `Clone` -- single consumer.
pub struct Receiver {
    consumer: nexus_logbuf::queue::mpsc::Consumer,
    inner: Arc<Inner>,
}

impl Receiver {
    /// Receive the next message. Returns a `ReadClaim` that derefs to `&[u8]`.
    ///
    /// Dropping the claim advances the consumer head and wakes a sender
    /// if it was blocked on a full buffer.
    pub fn recv(&mut self) -> RecvFut<'_> {
        RecvFut { receiver: self }
    }

    /// Try to receive without waiting.
    pub fn try_recv(&mut self) -> Option<ReadClaim<'_>> {
        let inner_claim = self.consumer.try_claim()?;
        Some(ReadClaim {
            inner: inner_claim,
            notify: &self.inner,
        })
    }
}

/// Future returned by [`Receiver::recv`].
pub struct RecvFut<'a> {
    receiver: &'a mut Receiver,
}

impl Drop for RecvFut<'_> {
    fn drop(&mut self) {
        self.receiver.inner.rx_slot.clear();
    }
}

impl<'a> Future for RecvFut<'a> {
    type Output = Result<ReadClaim<'a>, RecvError>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        // SAFETY: RecvFut is not Unpin-sensitive. We need &mut access to
        // receiver.consumer for try_claim, and the returned ReadClaim must
        // have lifetime 'a (tied to the Receiver, not this poll call).
        let this = unsafe { &mut *std::pin::Pin::into_inner_unchecked(self) };

        // SAFETY: Extend the reborrow lifetime to 'a. This is sound because:
        // - RecvFut holds &'a mut Receiver, so the Receiver lives for 'a
        // - ReadClaim borrows &mut Consumer from that Receiver
        // - The future won't be polled again after returning Ready
        let receiver: &'a mut Receiver = unsafe { &mut *(this.receiver as *mut Receiver) };

        // Try to claim.
        if let Some(inner_claim) = receiver.consumer.try_claim() {
            return Poll::Ready(Ok(ReadClaim {
                inner: inner_claim,
                notify: &receiver.inner,
            }));
        }

        // Empty + all senders dropped -> closed.
        if receiver.inner.sender_count.load(Ordering::Acquire) == 0 {
            return Poll::Ready(Err(RecvError));
        }

        // Park.
        if !receiver.inner.rx_slot.try_register_local(cx.waker()) {
            receiver.inner.rx_fallback.register(cx.waker());
        }

        Poll::Pending
    }
}

unsafe impl Send for RecvFut<'_> {}

impl Drop for Receiver {
    fn drop(&mut self) {
        self.inner.rx_closed.store(true, Ordering::Release);
        self.inner.tx_waiters.wake_all();
    }
}

unsafe impl Send for Receiver {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_channel(capacity: usize) -> (Sender, Receiver) {
        let poll = mio::Poll::new().unwrap();
        let mio_waker = Arc::new(mio::Waker::new(poll.registry(), mio::Token(usize::MAX)).unwrap());
        let cross_ctx = Arc::new(crate::cross_wake::CrossWakeContext {
            queue: crate::cross_wake::CrossWakeQueue::new(),
            mio_waker,
            parked: AtomicBool::new(false),
        });

        let (producer, consumer) = nexus_logbuf::queue::mpsc::new(capacity);
        let rx_slot = RxWakerSlot::new(Arc::as_ptr(&cross_ctx));

        let inner = Arc::new(Inner {
            rx_slot,
            rx_fallback: FallbackWaker::new(),
            tx_waiters: SenderWaitList::new(),
            _cross_wake_owner: cross_ctx,
            sender_count: AtomicUsize::new(1),
            rx_closed: AtomicBool::new(false),
        });

        (
            Sender {
                producer,
                inner: inner.clone(),
                wake_node: Arc::new(SenderWakerNode::new()),
            },
            Receiver { consumer, inner },
        )
    }

    fn try_send(tx: &mut Sender, data: &[u8]) {
        let mut claim = tx.try_claim(data.len()).unwrap();
        claim.copy_from_slice(data);
        claim.commit();
    }

    #[test]
    fn claim_commit_recv() {
        let (mut tx, mut rx) = test_channel(4096);
        try_send(&mut tx, b"hello");
        try_send(&mut tx, b"world");

        let msg = rx.try_recv().unwrap();
        assert_eq!(&*msg, b"hello");
        drop(msg);

        let msg = rx.try_recv().unwrap();
        assert_eq!(&*msg, b"world");
        drop(msg);

        assert!(rx.try_recv().is_none());
    }

    #[test]
    fn fifo_ordering() {
        let (mut tx, mut rx) = test_channel(4096);
        for i in 0u32..10 {
            try_send(&mut tx, &i.to_le_bytes());
        }
        for i in 0u32..10 {
            let msg = rx.try_recv().unwrap();
            assert_eq!(&*msg, &i.to_le_bytes());
        }
    }

    #[test]
    fn sender_drop_signals_closed() {
        let (mut tx, mut rx) = test_channel(4096);
        try_send(&mut tx, b"last");
        drop(tx);

        let msg = rx.try_recv().unwrap();
        assert_eq!(&*msg, b"last");
        drop(msg);

        assert!(rx.try_recv().is_none());
    }

    #[test]
    fn receiver_drop_signals_sender() {
        let (_tx, rx) = test_channel(4096);
        drop(rx);
        assert!(_tx.inner.rx_closed.load(Ordering::Acquire));
    }

    #[test]
    fn variable_length_messages() {
        let (mut tx, mut rx) = test_channel(8192);

        try_send(&mut tx, b"hi");
        try_send(&mut tx, &vec![0xABu8; 100]);
        try_send(&mut tx, &vec![0xCDu8; 1000]);

        let msg = rx.try_recv().unwrap();
        assert_eq!(msg.len(), 2);
        drop(msg);

        let msg = rx.try_recv().unwrap();
        assert_eq!(msg.len(), 100);
        drop(msg);

        let msg = rx.try_recv().unwrap();
        assert_eq!(msg.len(), 1000);
    }

    #[test]
    fn cross_thread_claim_send() {
        let (mut tx, mut rx) = test_channel(64 * 1024);

        let handle = std::thread::spawn(move || {
            for i in 0u64..100 {
                try_send(&mut tx, &i.to_le_bytes());
            }
        });

        handle.join().unwrap();

        for i in 0u64..100 {
            let msg = rx.try_recv().unwrap();
            assert_eq!(&*msg, &i.to_le_bytes());
        }
    }

    #[test]
    fn stress_sequential() {
        let (mut tx, mut rx) = test_channel(4096);
        let data = [0xFFu8; 32];

        let n = if cfg!(miri) { 100 } else { 10_000 };
        for _ in 0..n {
            try_send(&mut tx, &data);
            let msg = rx.try_recv().unwrap();
            assert_eq!(msg.len(), 32);
        }
    }

    #[test]
    fn claim_without_commit_aborts() {
        let (mut tx, mut rx) = test_channel(4096);

        // Claim and drop without commit -- skip marker.
        let claim = tx.try_claim(10).unwrap();
        drop(claim);

        // Next claim + commit should work.
        try_send(&mut tx, b"after_abort");

        let msg = rx.try_recv().unwrap();
        assert_eq!(&*msg, b"after_abort");
    }

    #[test]
    fn multiple_senders() {
        let (mut tx1, mut rx) = test_channel(64 * 1024);
        let mut tx2 = tx1.clone();

        try_send(&mut tx1, b"from_tx1");
        try_send(&mut tx2, b"from_tx2");
        try_send(&mut tx1, b"tx1_again");

        let msg = rx.try_recv().unwrap();
        assert_eq!(&*msg, b"from_tx1");
        drop(msg);

        let msg = rx.try_recv().unwrap();
        assert_eq!(&*msg, b"from_tx2");
        drop(msg);

        let msg = rx.try_recv().unwrap();
        assert_eq!(&*msg, b"tx1_again");
        drop(msg);

        assert!(rx.try_recv().is_none());
    }

    /// Sender dropped while its wake_node may be in the waiter list.
    /// Previously caused use-after-free when wake_one read freed memory.
    /// Fixed by Arc refcount on the node.
    #[test]
    fn sender_drop_while_queued() {
        let (mut tx1, mut rx) = test_channel(4096);
        let tx2 = tx1.clone();

        try_send(&mut tx1, b"data");

        // Drop tx2 -- its node may or may not be in the list.
        // Key test: this shouldn't crash even if the node IS in the list.
        drop(tx2);

        // Receiver pops -- should still work.
        let msg = rx.try_recv().unwrap();
        assert_eq!(&*msg, b"data");
        drop(msg);

        // tx1 can still send.
        try_send(&mut tx1, b"more");
        let msg = rx.try_recv().unwrap();
        assert_eq!(&*msg, b"more");
    }
}

// =============================================================================
// BUG-2 (#168) — UAF white-box test, same shape as mpsc.rs
// =============================================================================
//
// See `mpsc.rs::uaf_tests` for the full rationale. This file's
// `RxWakerSlot` shares the same fix and is verified by the same scenario.
#[cfg(test)]
mod uaf_tests {
    use super::*;
    use crate::cross_wake::wake_task_cross_thread;
    use crate::task::{self, FreeAction, Task};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::task::{Context, Poll};

    struct UafNoop;
    impl Future for UafNoop {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            Poll::Ready(())
        }
    }

    fn make_uaf_task() -> *mut u8 {
        let task = Box::new(Task::new_boxed(UafNoop, 0));
        Box::into_raw(task) as *mut u8
    }

    fn make_uaf_cross_ctx() -> Arc<crate::cross_wake::CrossWakeContext> {
        let poll = mio::Poll::new().unwrap();
        let mio_waker = Arc::new(mio::Waker::new(poll.registry(), mio::Token(usize::MAX)).unwrap());
        Arc::new(crate::cross_wake::CrossWakeContext {
            queue: crate::cross_wake::CrossWakeQueue::new(),
            mio_waker,
            parked: AtomicBool::new(false),
        })
    }

    #[test]
    fn waker_slot_uaf_when_task_freed_mid_dispatch() {
        let cross_ctx = make_uaf_cross_ctx();
        let task_ptr = make_uaf_task();
        assert_eq!(unsafe { task::ref_count(task_ptr) }, 1);

        let slot = RxWakerSlot::new(Arc::as_ptr(&cross_ctx));
        slot.register(task_ptr);

        assert!(
            slot.state
                .compare_exchange(STORED, EMPTY, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        );
        let captured = slot.task_ptr.swap(std::ptr::null_mut(), Ordering::Acquire);
        assert_eq!(captured, task_ptr);

        let pre_fix = match unsafe { task::complete_and_unref(task_ptr) } {
            FreeAction::FreeBox => {
                // PRE-FIX path: under regular cargo test, fail early
                // (avoid segfault from the deref below). Under miri,
                // trigger the UAF so the diagnostic trace fires.
                #[cfg(not(miri))]
                panic!(
                    "BUG-2 regression detected: register skipped ref_inc. \
                     Run under miri for the full UAF trace."
                );
                #[cfg(miri)]
                {
                    unsafe { task::free_task(task_ptr) };
                    true
                }
            }
            FreeAction::Retain => false,
            FreeAction::FreeSlab => panic!("box test must not yield FreeSlab"),
        };

        unsafe { wake_task_cross_thread(captured, &cross_ctx) };

        if !pre_fix {
            match unsafe { task::ref_dec(captured) } {
                FreeAction::FreeBox => unsafe { task::free_task(captured) },
                _ => panic!("post-fix cleanup must terminate the box task"),
            }
        }

        drop(slot);
    }

    /// Sensitive to the fix via explicit refcount assertions. FAILS pre-fix
    /// because `register` skips ref_inc and there's no Drop impl. PASSES
    /// post-fix because register ref_incs and Drop ref_decs.
    #[test]
    fn slot_drop_releases_ref_when_still_registered() {
        let cross_ctx = make_uaf_cross_ctx();
        let task_ptr = make_uaf_task();
        unsafe { task::ref_inc(task_ptr) };
        let action = unsafe { task::complete_and_unref(task_ptr) };
        assert!(matches!(action, FreeAction::Retain));
        let baseline_refcount = unsafe { task::ref_count(task_ptr) };
        assert_eq!(baseline_refcount, 1, "after complete_and_unref, refcount=1");

        let slot = RxWakerSlot::new(Arc::as_ptr(&cross_ctx));
        slot.register(task_ptr);
        let after_register = unsafe { task::ref_count(task_ptr) };

        drop(slot);
        let after_drop = unsafe { task::ref_count(task_ptr) };

        assert_eq!(
            after_register,
            after_drop + 1,
            "Post-fix Drop must release the ref that register acquired."
        );
        assert_eq!(
            after_register,
            baseline_refcount + 1,
            "Post-fix register must bump refcount by 1 — BUG-2 root cause."
        );

        // Cleanup: refcount = 1 + COMPLETED → final ref_dec yields FreeBox.
        let action = unsafe { task::ref_dec(task_ptr) };
        match action {
            FreeAction::FreeBox => unsafe { task::free_task(task_ptr) },
            other => panic!("expected FreeBox on final ref_dec, got {other:?}"),
        }
    }

    /// Race regression for John's review item 1 (BUG-2 follow-up).
    /// See `mpsc.rs::uaf_tests::register_during_wake_does_not_leak_ref`
    /// for the full design notes.
    #[test]
    fn register_during_wake_does_not_leak_ref() {
        let cross_ctx = make_uaf_cross_ctx();
        let task_ptr = make_uaf_task();

        unsafe { task::ref_inc(task_ptr) };
        let action = unsafe { task::complete_and_unref(task_ptr) };
        assert!(matches!(action, FreeAction::Retain));
        let baseline = unsafe { task::ref_count(task_ptr) };
        assert_eq!(baseline, 1);

        let slot = RxWakerSlot::new(Arc::as_ptr(&cross_ctx));

        slot.register(task_ptr);
        assert_eq!(
            unsafe { task::ref_count(task_ptr) },
            baseline + 1,
            "initial register must take a ref"
        );

        // Wake first half: CAS only.
        assert!(
            slot.state
                .compare_exchange(STORED, EMPTY, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        );

        // Race: re-register during the wake window.
        slot.register(task_ptr);
        assert_eq!(
            unsafe { task::ref_count(task_ptr) },
            baseline + 1,
            "race register must net to baseline+1; pre-fix this is baseline+2 (the leak)"
        );

        // Wake second half: swap + release.
        let captured = slot.task_ptr.swap(std::ptr::null_mut(), Ordering::Acquire);
        assert_eq!(captured, task_ptr);
        unsafe { release_slot_ref(captured, Arc::as_ptr(&cross_ctx)) };
        assert_eq!(
            unsafe { task::ref_count(task_ptr) },
            baseline,
            "post-wake refcount must be at baseline; pre-fix this is baseline+1"
        );

        drop(slot);
        assert_eq!(
            unsafe { task::ref_count(task_ptr) },
            baseline,
            "Drop on STORED-but-null slot is a no-op for refcount"
        );

        match unsafe { task::ref_dec(task_ptr) } {
            FreeAction::FreeBox => unsafe { task::free_task(task_ptr) },
            other => panic!("expected FreeBox, got {other:?}"),
        }
    }
}

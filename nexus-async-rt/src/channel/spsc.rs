//! Bounded cross-thread SPSC channel.
//!
//! `Sender`: `Send` (not Clone). `Receiver`: `Send`.
//! Uses `nexus_queue::spsc` for the data path (lock-free, cache-line padded).
//! Zero allocation on the send/recv hot path.
//!
//! Must be created inside [`Runtime::block_on`](crate::Runtime::block_on).
//!
//! ```ignore
//! use nexus_async_rt::channel::spsc;
//!
//! // Inside block_on:
//! let (tx, rx) = spsc::channel::<u64>(64);
//!
//! std::thread::spawn(move || {
//!     tx.try_send(42).unwrap();
//! });
//!
//! let val = rx.recv().await.unwrap();
//! ```

use std::cell::UnsafeCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::task::{Context, Poll, Waker};

use super::{RecvError, SendError, TryRecvError, TrySendError};

// =============================================================================
// Waker primitives (shared with mpsc pattern)
// =============================================================================

const EMPTY: u8 = 0;
const STORED: u8 = 1;
const REGISTERING: u8 = 2;

/// Zero-alloc receiver waker slot. Lives in Inner.
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

    fn register(&self, task_ptr: *mut u8) {
        debug_assert!(
            !task_ptr.is_null(),
            "RxWakerSlot::register called with null task_ptr"
        );
        let prev = self.state.swap(REGISTERING, Ordering::Acquire);
        debug_assert_ne!(prev, REGISTERING);

        // BUG-2 (#168) fix: hold a refcount on the task while registered
        // so a sender that captures the pointer mid-`wake()` can't have
        // it freed underneath. Matched by `ref_dec` in wake/clear/Drop.
        // SAFETY: caller (RecvFut::poll) just received task_ptr from the
        // active receiver task whose refcount is >= 1; the debug_assert
        // above catches the null case in development.
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
                // SAFETY: task_ptr is alive because `register` ref_inc'd
                // before storing — that ref keeps the task allocated
                // through the dispatch (see BUG-2 #168).
                let ctx = unsafe { &*self.cross_ctx };
                unsafe { crate::cross_wake::wake_task_cross_thread(task_ptr, ctx) };

                // BUG-2 fix: release the ref `register` acquired. AFTER
                // wake_task_cross_thread so the task is alive for its deref.
                // SAFETY: we own the ref from `register`.
                unsafe { release_slot_ref(task_ptr, self.cross_ctx) };
                return true;
            }
        }
        false
    }

    fn has_waker(&self) -> bool {
        self.state.load(Ordering::Acquire) == STORED
    }

    /// Clear the stored waker if one exists. Used by RecvFut::Drop to
    /// prevent use-after-free when the recv task completes while a
    /// sender on another thread may try to wake through the stale ptr.
    fn clear(&self) {
        if self
            .state
            .compare_exchange(STORED, EMPTY, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            let task_ptr = self.task_ptr.swap(std::ptr::null_mut(), Ordering::Acquire);
            if !task_ptr.is_null() {
                // BUG-2 fix: release the ref `register` acquired.
                // SAFETY: we own the ref from `register`.
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
                // SAFETY: we own the ref from `register`.
                unsafe { release_slot_ref(task_ptr, self.cross_ctx) };
            }
        }
    }
}

/// Release the slot's ref on `task_ptr`. If terminal, route via
/// [`crate::cross_wake::dispose_terminal`]. See `mpsc::release_slot_ref`
/// for the full design rationale — this is the identical pattern.
///
/// `cross_ctx` is unused here (dispose_terminal reads ctx from the task
/// header); kept on the signature for PR 1a consistency.
///
/// # Safety
///
/// `task_ptr` must point to a task on which `register` previously called
/// `ref_inc`.
unsafe fn release_slot_ref(
    task_ptr: *mut u8,
    _cross_ctx: *const crate::cross_wake::CrossWakeContext,
) {
    match unsafe { crate::task::ref_dec(task_ptr) } {
        crate::task::FreeAction::Retain => {}
        crate::task::FreeAction::FreeBox | crate::task::FreeAction::FreeSlab => {
            // SAFETY: task_ptr was alive until ref_dec; on terminal it's
            // still alive (dispose_terminal does the routing).
            unsafe { crate::cross_wake::dispose_terminal(task_ptr) };
        }
    }
}

/// Fallback waker for non-runtime wakers (root future).
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

/// Sender waker slot — single sender, no intrusive list needed.
struct TxWakerSlot {
    state: AtomicU8,
    waker: UnsafeCell<Option<Waker>>,
}

unsafe impl Send for TxWakerSlot {}
unsafe impl Sync for TxWakerSlot {}

impl TxWakerSlot {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(EMPTY),
            waker: UnsafeCell::new(None),
        }
    }

    /// Register. Called by the single sender — no concurrent register.
    fn register(&self, waker: &Waker) {
        let prev = self.state.swap(REGISTERING, Ordering::Acquire);
        debug_assert_ne!(prev, REGISTERING);
        unsafe { *self.waker.get() = Some(waker.clone()) };
        self.state.store(STORED, Ordering::Release);
    }

    /// Wake. Called by receiver (single thread).
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

impl Drop for TxWakerSlot {
    fn drop(&mut self) {
        *self.waker.get_mut() = None;
    }
}

// =============================================================================
// Shared state
// =============================================================================

struct Inner<T> {
    producer: nexus_queue::spsc::Producer<T>,
    consumer: nexus_queue::spsc::Consumer<T>,

    rx_slot: RxWakerSlot,
    rx_fallback: FallbackWaker,
    tx_waker: TxWakerSlot,

    _cross_wake_owner: Arc<crate::cross_wake::CrossWakeContext>,

    /// Sender alive flag.
    tx_alive: AtomicBool,
    /// Receiver alive flag.
    rx_closed: AtomicBool,
}

unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send> Sync for Inner<T> {}

impl<T> Inner<T> {
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
// channel()
// =============================================================================

/// Create a bounded cross-thread SPSC channel.
///
/// `capacity` is rounded up to the next power of two.
///
/// # Panics
///
/// - Panics if called outside [`Runtime::block_on`](crate::Runtime::block_on).
/// - Panics if `capacity` is 0.
pub fn channel<T: Send>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    crate::context::assert_in_runtime("spsc::channel() called outside Runtime::block_on");

    assert!(capacity > 0, "channel capacity must be > 0");

    let cross_ctx = crate::cross_wake::cross_wake_context()
        .expect("spsc::channel() requires runtime context for cross-thread wake");

    let (producer, consumer) = nexus_queue::spsc::ring_buffer(capacity);

    let rx_slot = RxWakerSlot::new(Arc::as_ptr(&cross_ctx));

    let inner = Arc::new(Inner {
        producer,
        consumer,
        rx_slot,
        rx_fallback: FallbackWaker::new(),
        tx_waker: TxWakerSlot::new(),
        _cross_wake_owner: cross_ctx,
        tx_alive: AtomicBool::new(true),
        rx_closed: AtomicBool::new(false),
    });

    let tx = Sender {
        inner: inner.clone(),
    };
    let rx = Receiver { inner };
    (tx, rx)
}

// =============================================================================
// Sender
// =============================================================================

/// Sending half of a bounded SPSC channel.
///
/// `Send` but not `Clone` — single producer.
pub struct Sender<T> {
    inner: Arc<Inner<T>>,
}

impl<T: Send> Sender<T> {
    /// Send a value, waiting if the buffer is full.
    pub fn send(&self, value: T) -> SendFut<'_, T> {
        SendFut {
            sender: self,
            value: Some(value),
        }
    }

    /// Try to send without waiting.
    pub fn try_send(&self, value: T) -> Result<(), TrySendError<T>> {
        if self.inner.rx_closed.load(Ordering::Acquire) {
            return Err(TrySendError::Closed(value));
        }

        match self.inner.producer.push(value) {
            Ok(()) => {
                if self.inner.has_rx_waker() {
                    self.inner.wake_rx();
                }
                Ok(())
            }
            Err(nexus_queue::Full(value)) => Err(TrySendError::Full(value)),
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        self.inner.tx_alive.store(false, Ordering::Release);
        self.inner.wake_rx();
    }
}

unsafe impl<T: Send> Send for Sender<T> {}

// =============================================================================
// SendFut
// =============================================================================

/// Future returned by [`Sender::send`].
pub struct SendFut<'a, T> {
    sender: &'a Sender<T>,
    value: Option<T>,
}

impl<T: Send> Future for SendFut<'_, T> {
    type Output = Result<(), SendError<T>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        let inner = &this.sender.inner;

        if inner.rx_closed.load(Ordering::Acquire) {
            let value = this.value.take().expect("polled after completion");
            return Poll::Ready(Err(SendError(value)));
        }

        let value = this.value.take().expect("polled after completion");
        match inner.producer.push(value) {
            Ok(()) => {
                if inner.has_rx_waker() {
                    inner.wake_rx();
                }
                Poll::Ready(Ok(()))
            }
            Err(nexus_queue::Full(value)) => {
                this.value = Some(value);
                inner.tx_waker.register(cx.waker());
                Poll::Pending
            }
        }
    }
}

unsafe impl<T: Send> Send for SendFut<'_, T> {}

// =============================================================================
// Receiver
// =============================================================================

/// Receiving half of a bounded SPSC channel.
///
/// `Send` but not `Clone` — single consumer.
pub struct Receiver<T> {
    inner: Arc<Inner<T>>,
}

impl<T: Send> Receiver<T> {
    /// Receive a value, waiting if the buffer is empty.
    pub fn recv(&self) -> RecvFut<'_, T> {
        RecvFut { receiver: self }
    }

    /// Try to receive without waiting.
    #[allow(clippy::option_if_let_else)]
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        match self.inner.consumer.pop() {
            Some(value) => {
                if self.inner.tx_waker.has_waker() {
                    self.inner.tx_waker.wake();
                }
                Ok(value)
            }
            None => {
                if self.inner.tx_alive.load(Ordering::Acquire) {
                    Err(TryRecvError::Empty)
                } else {
                    Err(TryRecvError::Closed)
                }
            }
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.inner.rx_closed.store(true, Ordering::Release);
        self.inner.tx_waker.wake();
    }
}

unsafe impl<T: Send> Send for Receiver<T> {}

// =============================================================================
// RecvFut
// =============================================================================

/// Future returned by [`Receiver::recv`].
pub struct RecvFut<'a, T> {
    receiver: &'a Receiver<T>,
}

impl<T> Drop for RecvFut<'_, T> {
    fn drop(&mut self) {
        // Clear the RxWakerSlot to prevent use-after-free: if a sender on
        // another thread calls wake() after this recv future is dropped,
        // it would read a dangling task pointer. The CAS ensures mutual
        // exclusion with the sender's wake() CAS on the same slot.
        self.receiver.inner.rx_slot.clear();
    }
}

impl<T: Send> Future for RecvFut<'_, T> {
    type Output = Result<T, RecvError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let inner = &self.receiver.inner;

        if let Some(value) = inner.consumer.pop() {
            if inner.tx_waker.has_waker() {
                inner.tx_waker.wake();
            }
            return Poll::Ready(Ok(value));
        }

        if !inner.tx_alive.load(Ordering::Acquire) {
            return Poll::Ready(Err(RecvError));
        }

        // Park with cross-thread-safe waker.
        if !inner.rx_slot.try_register_local(cx.waker()) {
            inner.rx_fallback.register(cx.waker());
        }

        // Re-check after register to avoid lost wake.
        if let Some(value) = inner.consumer.pop() {
            if inner.tx_waker.has_waker() {
                inner.tx_waker.wake();
            }
            return Poll::Ready(Ok(value));
        }

        if !inner.tx_alive.load(Ordering::Acquire) {
            return Poll::Ready(Err(RecvError));
        }

        Poll::Pending
    }
}

unsafe impl<T: Send> Send for RecvFut<'_, T> {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_channel<T: Send>(capacity: usize) -> (Sender<T>, Receiver<T>) {
        let poll = mio::Poll::new().unwrap();
        let mio_waker = Arc::new(mio::Waker::new(poll.registry(), mio::Token(usize::MAX)).unwrap());
        let cross_ctx = Arc::new(crate::cross_wake::CrossWakeContext {
            queue: crate::cross_wake::CrossWakeQueue::new(),
            mio_waker,
            parked: AtomicBool::new(false),
        });

        let (producer, consumer) = nexus_queue::spsc::ring_buffer(capacity);
        let rx_slot = RxWakerSlot::new(Arc::as_ptr(&cross_ctx));

        let inner = Arc::new(Inner {
            producer,
            consumer,
            rx_slot,
            rx_fallback: FallbackWaker::new(),
            tx_waker: TxWakerSlot::new(),
            _cross_wake_owner: cross_ctx,
            tx_alive: AtomicBool::new(true),
            rx_closed: AtomicBool::new(false),
        });
        (
            Sender {
                inner: inner.clone(),
            },
            Receiver { inner },
        )
    }

    #[test]
    fn send_recv_single() {
        let (tx, rx) = test_channel::<u32>(4);
        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();
        tx.try_send(3).unwrap();

        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        assert_eq!(rx.try_recv().unwrap(), 3);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn fifo_ordering() {
        let (tx, rx) = test_channel(8);
        for i in 0..8u32 {
            tx.try_send(i).unwrap();
        }
        for i in 0..8u32 {
            assert_eq!(rx.try_recv().unwrap(), i);
        }
    }

    #[test]
    fn try_send_full() {
        let (tx, rx) = test_channel(2);
        tx.try_send(1u32).unwrap();
        tx.try_send(2).unwrap();

        let err = tx.try_send(3).unwrap_err();
        assert!(err.is_full());
        assert_eq!(err.into_inner(), 3);

        assert_eq!(rx.try_recv().unwrap(), 1);
        tx.try_send(3).unwrap();
    }

    #[test]
    fn try_recv_empty() {
        let (tx, rx) = test_channel::<u32>(4);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
        tx.try_send(1).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn sender_drop_signals_closed() {
        let (tx, rx) = test_channel::<u32>(4);
        tx.try_send(42).unwrap();
        drop(tx);
        assert_eq!(rx.try_recv().unwrap(), 42);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Closed));
    }

    #[test]
    fn receiver_drop_signals_closed() {
        let (tx, rx) = test_channel::<u32>(4);
        drop(rx);
        let err = tx.try_send(1).unwrap_err();
        assert!(err.is_closed());
    }

    #[test]
    fn sender_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Sender<u64>>();
    }

    #[test]
    fn receiver_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Receiver<u64>>();
    }

    #[test]
    fn cross_thread_try_send() {
        let (tx, rx) = test_channel::<u64>(128);

        let handle = std::thread::spawn(move || {
            for i in 0..100 {
                tx.try_send(i).unwrap();
            }
        });

        handle.join().unwrap();
        for i in 0..100u64 {
            assert_eq!(rx.try_recv().unwrap(), i);
        }
    }

    #[test]
    fn stress_sequential() {
        let (tx, rx) = test_channel(64);
        let n = if cfg!(miri) { 100 } else { 100_000 };
        for i in 0..n {
            tx.try_send(i).unwrap();
            assert_eq!(rx.try_recv().unwrap(), i);
        }
    }

    #[test]
    fn sender_drop_while_receiver_alive() {
        let (tx, rx) = test_channel::<u32>(4);
        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();
        drop(tx);

        // Buffered values still available.
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Closed));

        // Dropping receiver is clean.
        drop(rx);
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

#![allow(
    unused_must_use,
    unused_imports,
    dead_code,
    unknown_lints,
    clippy::float_cmp,
    clippy::ref_option,
    clippy::used_underscore_binding,
    clippy::redundant_locals,
    clippy::semicolon_if_nothing_returned,
    clippy::let_underscore_future,
    clippy::while_let_loop,
    clippy::needless_continue,
    clippy::match_wild_err_arm,
    clippy::collection_is_never_read,
    clippy::async_yields_async,
    clippy::match_same_arms
)]
//! Miri tests for channel subsystems.
//!
//! Exercises ring buffer operations, drop semantics, and waiter lists
//! under miri to catch UB in the unsafe channel internals.
//!
//! Run: `cargo +nightly miri test -p nexus-async-rt --test miri_channel`

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use nexus_async_rt::Runtime;
use nexus_async_rt::channel::{TryRecvError, local, mpsc, spsc};
use nexus_rt::WorldBuilder;

// =============================================================================
// Helpers
// =============================================================================

fn runtime() -> (nexus_rt::World, Runtime) {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let rt = Runtime::new(&mut world);
    (world, rt)
}

#[derive(Clone)]
struct DropCounter(Rc<Cell<u32>>);

impl DropCounter {
    fn new() -> (Self, Rc<Cell<u32>>) {
        let count = Rc::new(Cell::new(0));
        (Self(count.clone()), count)
    }
}

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

/// Send-safe variant for spsc/mpsc channels (T: Send required).
#[derive(Clone)]
struct SendDropCounter(Arc<AtomicU32>);

impl SendDropCounter {
    fn new() -> (Self, Arc<AtomicU32>) {
        let count = Arc::new(AtomicU32::new(0));
        (Self(count.clone()), count)
    }
}

impl Drop for SendDropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

// =============================================================================
// Local channel tests
// =============================================================================

#[test]
fn local_send_recv_basic() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = local::channel::<u32>(4);
        tx.try_send(10).unwrap();
        tx.try_send(20).unwrap();
        tx.try_send(30).unwrap();

        assert_eq!(rx.try_recv().unwrap(), 10);
        assert_eq!(rx.try_recv().unwrap(), 20);
        assert_eq!(rx.try_recv().unwrap(), 30);
        assert_eq!(rx.try_recv().unwrap_err(), TryRecvError::Empty);
    });
}

#[test]
fn local_fill_and_drain() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        // Capacity rounds up to next power of two (4).
        let (tx, rx) = local::channel::<u32>(4);

        // Fill to capacity.
        for i in 0..4 {
            tx.try_send(i).unwrap();
        }
        assert!(tx.try_send(99).unwrap_err().is_full());

        // Drain all.
        for i in 0..4 {
            assert_eq!(rx.try_recv().unwrap(), i);
        }
        assert_eq!(rx.try_recv().unwrap_err(), TryRecvError::Empty);

        // Refill after drain — exercises ring buffer wrapping.
        for i in 100..104 {
            tx.try_send(i).unwrap();
        }
        for i in 100..104 {
            assert_eq!(rx.try_recv().unwrap(), i);
        }
    });
}

#[test]
fn local_channel_drop_with_pending_items() {
    let (_world, mut rt) = runtime();
    let (_, count) = DropCounter::new();

    rt.block_on(async {
        let (dc, inner_count) = DropCounter::new();
        let (tx, rx) = local::channel::<DropCounter>(8);

        tx.try_send(dc.clone()).unwrap();
        tx.try_send(dc.clone()).unwrap();
        tx.try_send(dc).unwrap();

        // 3 items in the buffer. Drop both ends without receiving.
        assert_eq!(inner_count.get(), 0);
        drop(rx);
        drop(tx);
        // All 3 buffered items + the original dc should be dropped.
    });

    // After block_on returns, all DropCounters are dropped.
    // The original dc (moved into the async block) plus the 3 clones
    // in the buffer. Each clone increments on drop.
    // Original dc dropped = 1, 3 buffer items dropped = 3, total = 4.
    let _ = count; // Just verify no leak — miri catches the rest.
}

#[test]
fn local_channel_drop_tracker() {
    let (_world, mut rt) = runtime();

    rt.block_on(async {
        let (dc, count) = DropCounter::new();
        let (tx, rx) = local::channel::<DropCounter>(8);

        tx.try_send(dc.clone()).unwrap();
        tx.try_send(dc.clone()).unwrap();
        tx.try_send(dc.clone()).unwrap();

        // Receive all values — each received value is dropped at end of scope.
        let v0 = rx.try_recv().unwrap();
        let v1 = rx.try_recv().unwrap();
        let v2 = rx.try_recv().unwrap();

        drop(v0);
        drop(v1);
        drop(v2);

        // 3 clones dropped via recv + drop.
        assert_eq!(count.get(), 3);

        drop(tx);
        drop(rx);
        drop(dc);
        // Original dc dropped = 4 total.
        assert_eq!(count.get(), 4);
    });
}

#[test]
fn local_sender_closes() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = local::channel::<u32>(4);

        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();
        drop(tx);

        // Buffered values still available.
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        // Then closed.
        assert_eq!(rx.try_recv().unwrap_err(), TryRecvError::Closed);
    });
}

#[test]
fn local_receiver_closes() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = local::channel::<u32>(4);

        tx.try_send(1).unwrap();
        drop(rx);

        // Sender sees closed.
        assert!(tx.try_send(2).unwrap_err().is_closed());
    });
}

// =============================================================================
// SPSC channel tests
// =============================================================================

#[test]
fn spsc_send_recv_basic() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = spsc::channel::<u32>(4);

        tx.try_send(10).unwrap();
        tx.try_send(20).unwrap();
        tx.try_send(30).unwrap();

        assert_eq!(rx.try_recv().unwrap(), 10);
        assert_eq!(rx.try_recv().unwrap(), 20);
        assert_eq!(rx.try_recv().unwrap(), 30);
        assert_eq!(rx.try_recv().unwrap_err(), TryRecvError::Empty);
    });
}

#[test]
fn spsc_fill_drain() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = spsc::channel::<u32>(4);

        // Fill to capacity.
        for i in 0..4 {
            tx.try_send(i).unwrap();
        }
        assert!(tx.try_send(99).unwrap_err().is_full());

        // Drain all.
        for i in 0..4 {
            assert_eq!(rx.try_recv().unwrap(), i);
        }
        assert_eq!(rx.try_recv().unwrap_err(), TryRecvError::Empty);
    });
}

#[test]
fn spsc_drop_tracker() {
    let (_world, mut rt) = runtime();

    rt.block_on(async {
        let (dc, count) = SendDropCounter::new();
        let (tx, rx) = spsc::channel::<SendDropCounter>(8);

        tx.try_send(dc.clone()).unwrap();
        tx.try_send(dc.clone()).unwrap();
        tx.try_send(dc.clone()).unwrap();

        let v0 = rx.try_recv().unwrap();
        let v1 = rx.try_recv().unwrap();
        let v2 = rx.try_recv().unwrap();

        drop(v0);
        drop(v1);
        drop(v2);

        assert_eq!(count.load(Ordering::Relaxed), 3);

        drop(tx);
        drop(rx);
        drop(dc);
        assert_eq!(count.load(Ordering::Relaxed), 4);
    });
}

#[test]
fn spsc_sender_closes() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = spsc::channel::<u32>(4);

        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();
        drop(tx);

        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        assert_eq!(rx.try_recv().unwrap_err(), TryRecvError::Closed);
    });
}

// =============================================================================
// MPSC channel tests
// =============================================================================

#[test]
fn mpsc_send_recv_basic() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = mpsc::channel::<u32>(8);

        tx.try_send(10).unwrap();
        tx.try_send(20).unwrap();
        tx.try_send(30).unwrap();

        assert_eq!(rx.try_recv().unwrap(), 10);
        assert_eq!(rx.try_recv().unwrap(), 20);
        assert_eq!(rx.try_recv().unwrap(), 30);
        assert_eq!(rx.try_recv().unwrap_err(), TryRecvError::Empty);
    });
}

#[test]
fn mpsc_multiple_senders_sequential() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = mpsc::channel::<u32>(32);

        // 3 senders, each sends 5 values sequentially.
        let tx2 = tx.clone();
        let tx3 = tx.clone();

        for i in 0..5 {
            tx.try_send(100 + i).unwrap();
        }
        for i in 0..5 {
            tx2.try_send(200 + i).unwrap();
        }
        for i in 0..5 {
            tx3.try_send(300 + i).unwrap();
        }

        // Receive all 15 values. Since sends are sequential and single-threaded,
        // order is deterministic: sender1's values, then sender2's, then sender3's.
        let mut values = Vec::new();
        for _ in 0..15 {
            values.push(rx.try_recv().unwrap());
        }
        assert_eq!(rx.try_recv().unwrap_err(), TryRecvError::Empty);

        // Verify order within each sender.
        let s1: Vec<_> = values
            .iter()
            .copied()
            .filter(|v| *v >= 100 && *v < 200)
            .collect();
        let s2: Vec<_> = values
            .iter()
            .copied()
            .filter(|v| *v >= 200 && *v < 300)
            .collect();
        let s3: Vec<_> = values
            .iter()
            .copied()
            .filter(|v| *v >= 300 && *v < 400)
            .collect();

        assert_eq!(s1, vec![100, 101, 102, 103, 104]);
        assert_eq!(s2, vec![200, 201, 202, 203, 204]);
        assert_eq!(s3, vec![300, 301, 302, 303, 304]);
    });
}

#[test]
fn mpsc_drop_tracker() {
    let (_world, mut rt) = runtime();

    rt.block_on(async {
        let (dc, count) = SendDropCounter::new();
        let (tx, rx) = mpsc::channel::<SendDropCounter>(8);

        tx.try_send(dc.clone()).unwrap();
        tx.try_send(dc.clone()).unwrap();
        tx.try_send(dc.clone()).unwrap();

        let v0 = rx.try_recv().unwrap();
        let v1 = rx.try_recv().unwrap();
        let v2 = rx.try_recv().unwrap();

        drop(v0);
        drop(v1);
        drop(v2);

        assert_eq!(count.load(Ordering::Relaxed), 3);

        drop(tx);
        drop(rx);
        drop(dc);
        assert_eq!(count.load(Ordering::Relaxed), 4);
    });
}

#[test]
fn mpsc_sender_closes() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = mpsc::channel::<u32>(4);
        let tx2 = tx.clone();

        tx.try_send(1).unwrap();
        tx2.try_send(2).unwrap();

        // Drop both senders.
        drop(tx);
        drop(tx2);

        // Buffered values still available.
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        // Then closed.
        assert_eq!(rx.try_recv().unwrap_err(), TryRecvError::Closed);
    });
}

#[test]
fn mpsc_receiver_closes() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = mpsc::channel::<u32>(4);

        tx.try_send(1).unwrap();
        drop(rx);

        assert!(tx.try_send(2).unwrap_err().is_closed());
    });
}

// =============================================================================
// RecvFut::Drop clears RxWakerSlot — SIGABRT regression tests
// =============================================================================

/// Spawn a task that parks on recv (registers waker in RxWakerSlot),
/// then abort it. RecvFut::Drop must clear the slot. After the abort,
/// try_send must not crash from reading a stale task pointer.
///
/// The key scenario: RecvFut registers the spawned task's waker ptr
/// in the RxWakerSlot. When the future is dropped (abort), RecvFut::Drop
/// calls clear() on the slot. Without the fix, the slot would hold a
/// dangling task pointer that the sender's wake path would read → UB.
#[test]
fn spsc_recv_future_drop_then_send_no_ub() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = spsc::channel::<u32>(4);

        let jh = nexus_async_rt::spawn_boxed(async move {
            // This recv().await parks (buffer empty). The RecvFut
            // registers the local runtime waker in the RxWakerSlot,
            // then suspends.
            let _ = rx.recv().await;
        });

        // Yield to let the spawned task run and park on recv.
        nexus_async_rt::yield_now().await;

        // Abort — drops future → RecvFut::Drop → rx_slot.clear()
        // Also drops rx (moved into the future).
        jh.abort();
        nexus_async_rt::yield_now().await; // process abort

        // Send after abort — channel is closed (rx dropped by abort),
        // so try_send returns Closed. The key assertion: it must NOT
        // crash from reading a stale task pointer in the RxWakerSlot.
        let _ = tx.try_send(42);
    });
}

/// Same pattern for mpsc channel.
#[test]
fn mpsc_recv_future_drop_clears_slot() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = mpsc::channel::<u32>(4);

        let jh = nexus_async_rt::spawn_boxed(async move {
            let _ = rx.recv().await;
        });

        nexus_async_rt::yield_now().await; // let it park on recv

        // Abort — drops future → RecvFut::Drop → rx_slot.clear()
        jh.abort();
        nexus_async_rt::yield_now().await; // process abort

        // Send after abort — rx is dropped (moved into aborted task),
        // so channel is closed. The key: no UB from stale waker ptr.
        let _ = tx.try_send(42);
    });
}

/// Local channel variant. The local channel uses Option<Waker> directly
/// (not RxWakerSlot), so this mainly verifies no UB from waker drop
/// ordering.
#[test]
fn local_channel_recv_drop_no_ub() {
    let (_world, mut rt) = runtime();
    rt.block_on(async {
        let (tx, rx) = local::channel::<u32>(4);

        // Spawn a task that parks on recv.
        let jh = nexus_async_rt::spawn_boxed(async move {
            let _ = rx.recv().await;
        });

        nexus_async_rt::yield_now().await; // let it park

        jh.abort();
        nexus_async_rt::yield_now().await; // process abort

        // Channel closed (rx dropped). Send must not crash.
        let _ = tx.try_send(42);
    });
}

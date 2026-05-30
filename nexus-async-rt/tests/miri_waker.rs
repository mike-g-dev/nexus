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
//! Miri tests for waker vtable lifecycle.
//!
//! Exercises clone_fn, wake_fn, wake_by_ref_fn, drop_fn through the
//! standard Waker API. Each test verifies refcount transitions and
//! correct routing of completed tasks to deferred free.
//!
//! Run: `cargo +nightly miri test -p nexus-async-rt --test miri_waker`

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

use nexus_async_rt::Executor;

// =============================================================================
// Test helpers
// =============================================================================

fn test_executor() -> Executor {
    Executor::new(16)
}

/// A future that yields `n` times via `wake_by_ref`, then returns `()`.
struct YieldN {
    remaining: u32,
}

impl YieldN {
    fn new(n: u32) -> Self {
        Self { remaining: n }
    }
}

impl Future for YieldN {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.remaining == 0 {
            Poll::Ready(())
        } else {
            self.remaining -= 1;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// A future that stores a waker clone on each poll, yields `n-1` times,
/// then completes on the nth poll.
struct StoreWakers {
    wakers: Rc<RefCell<Vec<Waker>>>,
    polls: u32,
    target: u32,
}

impl StoreWakers {
    fn new(wakers: Rc<RefCell<Vec<Waker>>>, target: u32) -> Self {
        Self {
            wakers,
            polls: 0,
            target,
        }
    }
}

impl Future for StoreWakers {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        self.wakers.borrow_mut().push(cx.waker().clone());
        self.polls += 1;
        if self.polls >= self.target {
            Poll::Ready(())
        } else {
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// A future that on first poll clones the waker and wakes it by value
/// (consuming the clone), then returns Pending. On second poll, returns
/// Ready. This exercises the wake_fn (by-value) vtable entry.
struct WakeByValueThenComplete {
    polls: u32,
}

impl WakeByValueThenComplete {
    fn new() -> Self {
        Self { polls: 0 }
    }
}

impl Future for WakeByValueThenComplete {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        self.polls += 1;
        if self.polls == 1 {
            // Clone the waker, then wake by value — consumes the clone,
            // which decrements refcount. The task should be re-queued.
            let cloned = cx.waker().clone();
            cloned.wake(); // by value: push to ready + ref_dec
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

/// Clone increments refcount — wakers created during self-waking are clones
/// of the task waker. Verify that a future yielding multiple times (creating
/// clones internally) completes and frees correctly.
#[test]
fn waker_clone_increments_refcount() {
    let mut exec = test_executor();
    let handle = exec.spawn_boxed(YieldN::new(5));

    // Each poll: task yields via wake_by_ref, re-queued.
    // 5 yields + 1 final poll = 6 polls needed.
    for _ in 0..6 {
        exec.poll();
    }

    assert!(handle.is_finished());
    drop(handle);
    exec.poll(); // drain deferred free
    assert_eq!(exec.task_count(), 0);
}

/// wake_by_ref does NOT consume the waker — the task is re-queued but
/// the waker remains valid. Verify re-queue across multiple polls.
#[test]
fn waker_wake_by_ref_does_not_consume() {
    let mut exec = test_executor();
    // Yields twice, completes on 3rd poll.
    let handle = exec.spawn_boxed(YieldN::new(2));

    // First poll: yields, re-queued via wake_by_ref.
    exec.poll();
    assert!(!handle.is_finished());

    // Second poll: yields again.
    exec.poll();
    assert!(!handle.is_finished());

    // Third poll: completes.
    exec.poll();
    assert!(handle.is_finished());

    drop(handle);
    exec.poll(); // drain deferred free
    assert_eq!(exec.task_count(), 0);
}

/// wake() by value consumes the waker (decrements refcount). The future
/// clones its waker and calls wake() on the clone during poll. This
/// exercises the wake_fn vtable entry (push to ready + ref_dec).
#[test]
fn waker_wake_by_value_consumes() {
    let mut exec = test_executor();
    let handle = exec.spawn_boxed(WakeByValueThenComplete::new());

    // First poll: future clones waker, calls clone.wake() (by value),
    // returns Pending. The wake re-queues the task.
    exec.poll();
    assert!(!handle.is_finished());

    // Second poll: task completes.
    exec.poll();
    assert!(handle.is_finished());

    drop(handle);
    exec.poll(); // drain deferred free
    assert_eq!(exec.task_count(), 0);
}

/// Dropping the JoinHandle after completion frees the task slot via
/// deferred free on the next poll cycle.
#[test]
fn waker_drop_after_completion_frees_slot() {
    let mut exec = test_executor();
    let handle = exec.spawn_boxed(async { 42u64 });

    exec.poll(); // task completes
    assert!(handle.is_finished());
    assert_eq!(exec.task_count(), 0); // live_count decremented on completion

    drop(handle); // drops JoinHandle's waker ref, pushes to deferred free
    exec.poll(); // drain deferred free — actually frees the slot
}

/// Waking a completed task is a no-op — the waker detects is_completed
/// and skips the ready queue push.
#[test]
fn waker_wake_completed_task_is_noop() {
    let stash: Rc<RefCell<Vec<Waker>>> = Rc::new(RefCell::new(Vec::new()));
    let mut exec = test_executor();

    // Future stores a waker clone, then completes immediately.
    let handle = exec.spawn_boxed(StoreWakers::new(stash.clone(), 1));

    exec.poll(); // task stores waker, completes.
    assert!(handle.is_finished());

    // The stashed waker points to a completed task.
    let wakers = stash.borrow();
    assert_eq!(wakers.len(), 1);

    // Wake the completed task — should be a no-op.
    wakers[0].wake_by_ref();

    // Poll again — no tasks should be re-queued.
    let polled = exec.poll();
    assert_eq!(polled, 0, "completed task should not be re-polled");

    drop(wakers);
    drop(stash); // drops the stashed waker clones
    drop(handle);
    exec.poll(); // drain deferred free
}

/// Multiple waker clones — only the last ref_dec triggers should_free.
/// Spawn a task that stores 10 waker clones. Task completes. Drop all
/// clones one by one. Verify task_count reaches 0 after cleanup.
#[test]
fn waker_multiple_clones_one_free() {
    let stash: Rc<RefCell<Vec<Waker>>> = Rc::new(RefCell::new(Vec::new()));
    let mut exec = test_executor();

    // Future stores 10 waker clones (one per poll, yields 9 times).
    // Actually, we want 10 clones stored. The future polls 10 times,
    // storing a clone each time, completing on the 10th.
    let handle = exec.spawn_boxed(StoreWakers::new(stash.clone(), 10));

    // Poll enough times to complete (10 polls: 9 yields + 1 completion).
    for _ in 0..10 {
        exec.poll();
    }
    assert!(handle.is_finished());

    // We now hold 10 waker clones + the JoinHandle's ref.
    let clone_count = stash.borrow().len();
    assert_eq!(clone_count, 10);

    // Drop clones one by one.
    for _ in 0..10 {
        stash.borrow_mut().pop();
    }
    assert!(stash.borrow().is_empty());

    // Drop the JoinHandle — last reference besides executor's.
    drop(handle);

    // Drain deferred free.
    exec.poll();
    assert_eq!(exec.task_count(), 0);
}

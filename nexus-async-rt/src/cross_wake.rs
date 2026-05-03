//! Cross-thread wake infrastructure.
//!
//! An intrusive MPSC queue (Vyukov style) for waking tasks from other
//! threads. Each task's header contains an `AtomicPtr<u8>` (`cross_next`)
//! used as the intrusive link — zero allocation per wake.
//!
//! The queue is paired with a `mio::Waker` (eventfd) to interrupt the
//! runtime's epoll when a cross-thread wake arrives.
//!
//! Local wakes (same thread) continue using the fast TLS Vec path.
//! Cross-thread wakes use this queue + eventfd. The executor drains
//! both on each poll cycle.

use std::cell::{Cell, UnsafeCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::task;

// =============================================================================
// TLS — cross-wake context accessible during block_on
// =============================================================================
//
// Two slots, separate scopes, separate consumers:
//
// - `CTX_CROSS_WAKE: *const Arc<CrossWakeContext>`
//   Pointer to the Runtime's `cross_wake` Arc field. Installed at
//   `run_loop` entry, cleared on exit. Used by `cross_wake_context()`
//   to clone the Arc for new cross-thread waker construction (channel
//   slots, tokio_compat). Scope is `block_on` because that's when
//   user task code runs and needs to construct cross-thread wakers.
//   Pointer-to-field — safe under `&mut self` during run_loop.
//
// - `CURRENT_RUNTIME_CTX: *const CrossWakeContext`
//   `Arc::as_ptr` value pointing into the Arc allocation (heap-stable,
//   doesn't dangle if the Runtime moves). Installed at
//   `RuntimeBuilder::build`, cleared via a guard field on Runtime.
//   Used by `dispose_terminal::on_owning_executor` to decide whether
//   a TaskRef::Drop is on the executor thread (defer) or a foreign
//   thread (queue). Scope is the Runtime's full lifetime because
//   TaskRef::Drop can fire post-`block_on` (e.g., JoinHandle dropped
//   between block_on calls, channel slots torn down during Runtime
//   drop) and during `Executor::drop` itself.

thread_local! {
    static CTX_CROSS_WAKE: Cell<*const Arc<CrossWakeContext>> =
        const { Cell::new(std::ptr::null()) };

    static CURRENT_RUNTIME_CTX: Cell<*const CrossWakeContext> =
        const { Cell::new(std::ptr::null()) };
}

/// Install the cross-wake context in TLS. Returns a guard that clears
/// it on drop.
pub(crate) fn install_cross_wake(ctx: &Arc<CrossWakeContext>) -> CrossWakeGuard {
    let prev = CTX_CROSS_WAKE.with(|c| c.replace(std::ptr::from_ref(ctx)));
    CrossWakeGuard { prev }
}

pub(crate) struct CrossWakeGuard {
    prev: *const Arc<CrossWakeContext>,
}

impl Drop for CrossWakeGuard {
    fn drop(&mut self) {
        CTX_CROSS_WAKE.with(|c| c.set(self.prev));
    }
}

/// Get the current runtime's cross-wake context. Returns None if
/// called outside `block_on`.
pub(crate) fn cross_wake_context() -> Option<Arc<CrossWakeContext>> {
    CTX_CROSS_WAKE.with(|c| {
        let ptr = c.get();
        if ptr.is_null() {
            None
        } else {
            // SAFETY: ptr was set by install_cross_wake and is valid
            // for the duration of block_on.
            Some(unsafe { (*ptr).clone() })
        }
    })
}

// =============================================================================
// Runtime-lifetime TLS install for `dispose_terminal`
// =============================================================================

/// Install the Runtime's cross-wake context as the current-thread
/// "owning executor" identity. Returns a guard that restores the
/// previous value on drop.
///
/// Stores `Arc::as_ptr(arc)` — the inner allocation address, stable
/// for the lifetime of the Arc. Doesn't dangle if the Runtime struct
/// moves (the inner stays put).
pub(crate) fn install_runtime_cross_wake(arc: &Arc<CrossWakeContext>) -> RuntimeCrossWakeGuard {
    let ptr = Arc::as_ptr(arc);
    let prev = CURRENT_RUNTIME_CTX.with(|c| c.replace(ptr));
    RuntimeCrossWakeGuard { prev }
}

/// RAII guard restoring the previous `CURRENT_RUNTIME_CTX` on drop.
///
/// Lives as a field on `Runtime`, placed after `executor` so it drops
/// after the executor finishes its task-freeing work (which may invoke
/// `dispose_terminal`).
pub(crate) struct RuntimeCrossWakeGuard {
    prev: *const CrossWakeContext,
}

impl Drop for RuntimeCrossWakeGuard {
    fn drop(&mut self) {
        CURRENT_RUNTIME_CTX.with(|c| c.set(self.prev));
    }
}

/// Read the currently-installed runtime cross-wake context pointer for
/// this thread, if any. Returns null when no Runtime is alive on the
/// thread.
///
/// Used by spawn paths (`Executor::spawn_boxed`, `alloc::slab_spawn`,
/// `SlabClaim::spawn`) to write the value into each new task's header.
#[inline]
pub(crate) fn current_runtime_ctx() -> *const CrossWakeContext {
    CURRENT_RUNTIME_CTX.with(Cell::get)
}

/// True if `ctx` matches the runtime currently installed on this thread.
///
/// False when no Runtime is alive here, or when a different runtime's
/// ctx is installed (which can't actually happen given
/// `RuntimePresenceGuard` — at most one Runtime per thread).
#[inline]
fn on_owning_executor(ctx: &CrossWakeContext) -> bool {
    let installed = CURRENT_RUNTIME_CTX.with(Cell::get);
    !installed.is_null() && std::ptr::eq(installed, ctx)
}

// =============================================================================
// dispose_terminal — unified terminal-drop routing
// =============================================================================

/// Dispose of a task whose final ref was just dropped (FreeBox / FreeSlab).
///
/// Routes to one of two on-thread paths and one off-thread path,
/// chosen by `task_ptr`'s header context and the current thread:
///
/// 1. **On-thread (null ctx OR ctx matches the current Runtime's TLS).**
///    Defer to end of the current poll cycle via
///    `crate::waker::try_defer_free` so `Executor::all_tasks`
///    bookkeeping stays consistent (the executor's drain frees the
///    task and removes the slab key in the right order). If the
///    `DEFERRED_FREE` TLS is null (called outside any poll cycle —
///    e.g., `JoinHandle` dropped after `block_on` returns, or a bare
///    `Executor` test where the handle outlives the poll), the slot
///    leaks until `Executor::drop` iterates `all_tasks` and reclaims
///    it. **Direct-freeing here is unsafe** because it would race
///    `Executor::drop`'s `all_tasks` iteration → double free. This is
///    the established pre-refactor behavior of `free_completed_slot`.
///
/// 2. **Off-thread (ctx non-null, doesn't match TLS).** Push to the
///    Runtime's cross-thread queue via `try_set_queued + push +
///    parked check`. The Runtime's `drain_cross_thread` recognizes
///    terminal entries and pushes them to `deferred_free`, which
///    updates `all_tasks` on the next poll cycle.
///
/// This is the ONE place that does terminal-drop routing for off-thread
/// holders. All other code (`TaskRef::Drop`, channel slot release
/// paths, `tokio_compat` terminal branches) goes through here.
///
/// # Safety
///
/// `task_ptr` must reference a task whose final ref was just dropped
/// (i.e., `ref_dec` returned `FreeBox` or `FreeSlab`). The header's
/// `cross_wake_ctx` must be either null or a valid Arc-backed pointer
/// (set at spawn time, kept alive transitively by any holder of a
/// TaskRef on this task).
pub(crate) unsafe fn dispose_terminal(task_ptr: *mut u8) {
    let ctx_ptr = unsafe { task::header_cross_wake_ctx(task_ptr) };

    // Null ctx (bare Executor or test-only Task::new_boxed): same
    // defer-or-leak fallback as the on-owning-executor branch. Direct
    // free is unsafe — see doc-comment.
    let on_executor = ctx_ptr.is_null() || {
        // SAFETY: ctx_ptr is Arc-backed and kept alive transitively
        // by whoever held the ref we just dropped.
        let ctx = unsafe { &*ctx_ptr };
        on_owning_executor(ctx)
    };

    if on_executor {
        // SAFETY: caller guarantees terminal state.
        let _ = unsafe { crate::waker::try_defer_free(task_ptr) };
        return;
    }

    // Off-thread: route via cross-queue. The executor's
    // drain_cross_thread recognizes terminal entries and pushes them
    // to deferred_free, which updates all_tasks on the next poll cycle.
    //
    // SAFETY: ctx_ptr non-null (bypassed the on_executor branch via
    // the && short-circuit above), Arc-backed and alive. try_set_queued
    // is an atomic on the task header (still alive — we own no ref but
    // the task header isn't freed because no one else has freed it
    // yet, and dispose_terminal is the path that would). queue.push
    // uses the cross_next field and is thread-safe.
    let ctx = unsafe { &*ctx_ptr };
    if unsafe { task::try_set_queued(task_ptr) } {
        unsafe { ctx.queue.push(task_ptr) };
        if ctx.parked.load(Ordering::Acquire) {
            let _ = ctx.mio_waker.wake();
        }
    }
}

// =============================================================================
// Intrusive MPSC queue (Vyukov)
// =============================================================================

/// Lock-free MPSC queue for cross-thread task wake notifications.
///
/// Producers (any thread) push task pointers via atomic swap on the tail.
/// The single consumer (runtime thread) drains via the head.
///
/// Each task's `cross_next` field (offset 32 in the header) serves as
/// the intrusive link pointer. No heap allocation per push.
///
/// Uses a stub node to avoid the empty-queue edge case. The stub is
/// just an `AtomicPtr` — not a real task.
pub(crate) struct CrossWakeQueue {
    /// Consumer reads from here. Only touched by the runtime thread.
    /// Wrapped in UnsafeCell so pop() can take &self (interior mutability).
    head: UnsafeCell<*mut u8>,
    /// Producers CAS here. Shared across threads.
    tail: AtomicPtr<u8>,
    /// Heap-allocated stub node. Stable address across moves.
    /// The stub is just an `AtomicPtr<u8>` (the "next" pointer).
    stub: *mut AtomicPtr<u8>,
}

// SAFETY: The queue is designed for cross-thread use.
// Producers push from any thread (atomic tail swap).
// Consumer pops from one thread (head is non-atomic).
unsafe impl Send for CrossWakeQueue {}
unsafe impl Sync for CrossWakeQueue {}

impl CrossWakeQueue {
    /// Create a new empty queue.
    pub(crate) fn new() -> Self {
        // Heap-allocate the stub so its address is stable after moves.
        let stub = Box::into_raw(Box::new(AtomicPtr::new(std::ptr::null_mut())));
        let stub_as_node = stub.cast::<u8>();
        Self {
            head: UnsafeCell::new(stub_as_node),
            tail: AtomicPtr::new(stub_as_node),
            stub,
        }
    }

    /// The stub's "task pointer" — the heap-allocated AtomicPtr.
    #[inline]
    fn stub_ptr(&self) -> *mut u8 {
        self.stub.cast::<u8>()
    }

    /// Get the `cross_next` pointer for a node. For real tasks this is
    /// the AtomicPtr at offset 32. For the stub it IS the stub allocation.
    #[inline]
    unsafe fn next_of(&self, node: *mut u8) -> &AtomicPtr<u8> {
        if node == self.stub_ptr() {
            // SAFETY: stub is a valid heap-allocated AtomicPtr.
            unsafe { &*self.stub }
        } else {
            // SAFETY: caller guarantees `node` is a valid task pointer.
            // cross_next returns a raw pointer; we dereference explicitly.
            unsafe { &*task::cross_next(node) }
        }
    }
}

impl Drop for CrossWakeQueue {
    fn drop(&mut self) {
        // SAFETY: stub was allocated via Box::into_raw in new().
        unsafe { drop(Box::from_raw(self.stub)) };
    }
}

impl CrossWakeQueue {
    /// Push a task pointer into the queue. Thread-safe (any thread).
    ///
    /// # Safety
    ///
    /// `task_ptr` must point to a live task with a valid `cross_next` field,
    /// OR must be the stub pointer (internal re-insertion).
    /// The task must not already be in this queue.
    pub(crate) unsafe fn push(&self, task_ptr: *mut u8) {
        // Clear next pointer on the node we're pushing.
        // SAFETY: task_ptr is either a valid task or the stub.
        unsafe { self.next_of(task_ptr) }.store(std::ptr::null_mut(), Ordering::Relaxed);

        // Atomically swap ourselves into the tail position.
        let prev = self.tail.swap(task_ptr, Ordering::AcqRel);

        // Link the previous tail to us. The consumer will see this
        // once the Release from our swap is visible.
        // SAFETY: prev is either the stub or a previously pushed task.
        unsafe { self.next_of(prev) }.store(task_ptr, Ordering::Release);
    }

    /// Pop a task pointer from the queue. Single-consumer only.
    ///
    /// Takes `&self` using interior mutability for `head` (UnsafeCell).
    /// The single-consumer guarantee ensures no concurrent access to `head`.
    ///
    /// Returns `None` if the queue is empty (or a producer hasn't
    /// finished linking yet — transient inconsistency).
    pub(crate) fn pop(&self) -> Option<*mut u8> {
        // SAFETY: single-consumer guarantee — only the runtime thread
        // calls pop(), never concurrently.
        let head_ref = unsafe { &mut *self.head.get() };
        let mut head = *head_ref;
        // SAFETY: head is either the stub or a previously pushed task.
        let mut next = unsafe { self.next_of(head) }.load(Ordering::Acquire);

        let stub = self.stub_ptr();

        // Skip the stub node.
        if head == stub {
            if next.is_null() {
                return None; // Queue is empty.
            }
            *head_ref = next;
            head = next;
            next = unsafe { self.next_of(head) }.load(Ordering::Acquire);
        }

        // Normal case: head has a next -> pop head, advance.
        if !next.is_null() {
            *head_ref = next;
            return Some(head);
        }

        // head is the last node. Check if tail == head.
        let tail = self.tail.load(Ordering::Acquire);
        if head != tail {
            // A producer swapped tail but hasn't linked next yet.
            // Transient inconsistency — return None, retry later.
            return None;
        }

        // Re-insert stub so we don't lose the tail reference.
        // SAFETY: stub is always valid.
        unsafe { self.push(stub) };

        // Now check if head got a next pointer (the stub push linked it).
        next = unsafe { self.next_of(head) }.load(Ordering::Acquire);
        if !next.is_null() {
            *head_ref = next;
            return Some(head);
        }

        None
    }
}

// =============================================================================
// Cross-thread waker data
// =============================================================================

/// Shared context for all cross-thread wakers in a runtime instance.
/// Created once per runtime, `Arc`-shared across all cross-thread wakers.
pub(crate) struct CrossWakeContext {
    /// The intrusive MPSC queue for cross-thread wake pushes.
    pub(crate) queue: CrossWakeQueue,
    /// The mio waker to interrupt epoll after pushing.
    pub(crate) mio_waker: Arc<mio::Waker>,
    /// Whether the runtime is currently parked in epoll_wait.
    /// Cross-thread senders read this to decide whether to poke
    /// the eventfd — skip the syscall when the runtime is actively
    /// polling (it will drain the inbox on the next iteration).
    pub(crate) parked: std::sync::atomic::AtomicBool,
}

// SAFETY: CrossWakeQueue is Send + Sync, Arc<mio::Waker> is Send + Sync.
unsafe impl Send for CrossWakeContext {}
unsafe impl Sync for CrossWakeContext {}

/// Wake a task via the cross-thread path: push to intrusive inbox,
/// conditionally poke eventfd. Zero allocation.
///
/// # Safety
///
/// `task_ptr` must point to a live task. `ctx` must be a valid
/// `CrossWakeContext` (guaranteed by channel lifetime).
pub(crate) unsafe fn wake_task_cross_thread(task_ptr: *mut u8, ctx: &CrossWakeContext) {
    // Don't wake completed tasks.
    if unsafe { task::is_completed(task_ptr) } {
        return;
    }

    // Dedup: atomic CAS on is_queued for thread safety.
    // SAFETY: task_ptr is a valid task.
    if !unsafe { task::try_set_queued(task_ptr) } {
        return;
    }

    // SAFETY: task_ptr valid, not already queued.
    unsafe { ctx.queue.push(task_ptr) };

    if ctx.parked.load(Ordering::Acquire) {
        let _ = ctx.mio_waker.wake();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::Task;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    struct Noop;
    impl Future for Noop {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            Poll::Ready(())
        }
    }

    fn make_task() -> *mut u8 {
        let task = Box::new(Task::new_boxed(Noop, 0));
        Box::into_raw(task) as *mut u8
    }

    unsafe fn free(ptr: *mut u8) {
        unsafe { task::free_task(ptr) };
    }

    #[test]
    fn queue_push_pop_single() {
        let q = CrossWakeQueue::new();
        let t1 = make_task();

        unsafe { q.push(t1) };
        assert_eq!(q.pop(), Some(t1));
        assert_eq!(q.pop(), None);

        unsafe { free(t1) };
    }

    #[test]
    fn queue_push_pop_multiple() {
        let q = CrossWakeQueue::new();
        let t1 = make_task();
        let t2 = make_task();
        let t3 = make_task();

        unsafe { q.push(t1) };
        unsafe { q.push(t2) };
        unsafe { q.push(t3) };

        assert_eq!(q.pop(), Some(t1));
        assert_eq!(q.pop(), Some(t2));
        assert_eq!(q.pop(), Some(t3));
        assert_eq!(q.pop(), None);

        unsafe { free(t1) };
        unsafe { free(t2) };
        unsafe { free(t3) };
    }

    #[test]
    fn queue_interleaved_push_pop() {
        let q = CrossWakeQueue::new();
        let t1 = make_task();
        let t2 = make_task();

        unsafe { q.push(t1) };
        assert_eq!(q.pop(), Some(t1));

        unsafe { q.push(t2) };
        assert_eq!(q.pop(), Some(t2));
        assert_eq!(q.pop(), None);

        unsafe { free(t1) };
        unsafe { free(t2) };
    }

    #[test]
    fn queue_empty() {
        let q = CrossWakeQueue::new();
        assert_eq!(q.pop(), None);
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn queue_reuse_after_drain() {
        let q = CrossWakeQueue::new();
        let t1 = make_task();

        for _ in 0..100 {
            unsafe { q.push(t1) };
            assert_eq!(q.pop(), Some(t1));
        }
        assert_eq!(q.pop(), None);

        unsafe { free(t1) };
    }

    // =========================================================================
    // dispose_terminal — routing under different ctx + thread states
    // =========================================================================

    fn make_ctx() -> Arc<CrossWakeContext> {
        // Real mio waker + queue. Construction is lightweight enough
        // for unit tests; miri tolerates the mio FFI here because
        // CrossWakeContext doesn't actually call wake() unless `parked`
        // is true (we keep it false in these tests).
        let poll = mio::Poll::new().expect("mio::Poll");
        let waker = mio::Waker::new(poll.registry(), mio::Token(0)).expect("mio::Waker");
        Arc::new(CrossWakeContext {
            queue: CrossWakeQueue::new(),
            mio_waker: Arc::new(waker),
            parked: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Build a real spawn-shaped task (refcount=2, HAS_JOIN, ctx in
    /// header) so dispose_terminal sees a production-like header.
    fn make_spawned_task(ctx: &Arc<CrossWakeContext>) -> *mut u8 {
        struct Noop;
        impl std::future::Future for Noop {
            type Output = u64;
            fn poll(self: std::pin::Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u64> {
                Poll::Ready(0)
            }
        }
        crate::task::box_spawn_joinable(Noop, 0, Arc::as_ptr(ctx))
    }

    /// Walk a freshly-spawned joinable task to terminal-eligible state:
    /// drop future, complete, clear HAS_JOIN, ref_dec the JoinHandle's
    /// ref. Final state: rc=0, COMPLETED, no lifecycle flags. Caller
    /// then invokes `dispose_terminal(ptr)` directly to exercise the
    /// routing — no re-acquire (would violate ref_inc's >=1 contract).
    unsafe fn drive_to_terminal(ptr: *mut u8) {
        unsafe {
            crate::task::drop_task_future(ptr);
            assert!(matches!(
                crate::task::complete_and_unref(ptr),
                crate::task::FreeAction::Retain
            )); // 2 refs → 1 (HAS_JOIN still set, rc=1)
            crate::task::clear_has_join(ptr);
            assert!(matches!(
                crate::task::ref_dec(ptr),
                crate::task::FreeAction::FreeBox
            )); // rc 1 → 0, COMPLETED, no lifecycle → terminal
            assert!(crate::task::is_terminal(ptr));
        }
    }

    #[test]
    fn dispose_terminal_null_ctx_no_tls_leaks() {
        // Null ctx + no DEFERRED_FREE TLS → leak. dispose_terminal must
        // NOT free directly (would race Executor::drop's all_tasks scan
        // for tasks registered with a bare Executor).
        struct Noop;
        impl std::future::Future for Noop {
            type Output = ();
            fn poll(self: std::pin::Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
                Poll::Ready(())
            }
        }
        // Task::new_boxed: cross_wake_ctx is null, rc=1, no HAS_JOIN.
        let task = Box::new(crate::task::Task::new_boxed(Noop, 0));
        let ptr = Box::into_raw(task) as *mut u8;

        unsafe {
            // Walk to terminal: rc 1 → 0, COMPLETED, no lifecycle.
            crate::task::drop_task_future(ptr);
            assert!(matches!(
                crate::task::complete_and_unref(ptr),
                crate::task::FreeAction::FreeBox
            ));
            assert!(crate::task::is_terminal(ptr));

            // No TLS installed. dispose_terminal: null ctx → on_executor
            // branch → try_defer_free returns false (no DEFERRED_FREE
            // TLS) → leak. Task header stays valid for our final free.
            dispose_terminal(ptr);
            assert!(crate::task::is_terminal(ptr));
            crate::task::free_task(ptr);
        }
    }

    #[test]
    fn dispose_terminal_on_executor_defers_when_tls_set() {
        // ctx matches CURRENT_RUNTIME_CTX TLS + DEFERRED_FREE TLS set
        // → push to deferred_free list. Real-world case: JoinHandle
        // dropped during a poll cycle.
        let ctx = make_ctx();
        let _guard = install_runtime_cross_wake(&ctx);
        let ptr = make_spawned_task(&ctx);

        // Set up DEFERRED_FREE TLS pointing at a local Vec.
        let mut deferred: Vec<*mut u8> = Vec::new();
        let mut ready: Vec<*mut u8> = Vec::new();
        let _poll_guard = crate::waker::set_poll_context(&raw mut ready, &raw mut deferred);

        unsafe {
            drive_to_terminal(ptr);
            dispose_terminal(ptr);
        }

        // Verify push to deferred_free.
        assert_eq!(deferred.len(), 1);
        assert_eq!(deferred[0], ptr);

        // Cleanup: free the task ourselves (we own the deferred entry).
        unsafe { crate::task::free_task(ptr) };
    }

    #[test]
    fn dispose_terminal_on_executor_leaks_when_tls_null() {
        // ctx matches TLS but DEFERRED_FREE TLS is null (e.g.,
        // JoinHandle dropped between block_on calls). dispose_terminal
        // leaves the task for Executor::drop to reclaim — we verify
        // "no double free" by freeing manually afterwards.
        let ctx = make_ctx();
        let _guard = install_runtime_cross_wake(&ctx);
        let ptr = make_spawned_task(&ctx);

        unsafe {
            drive_to_terminal(ptr);
            // No poll context installed → DEFERRED_FREE TLS stays null.
            dispose_terminal(ptr);
        }

        // Task header still valid (we leaked, didn't free).
        assert!(unsafe { crate::task::is_terminal(ptr) });

        unsafe { crate::task::free_task(ptr) };
    }

    #[test]
    fn dispose_terminal_off_thread_queues() {
        // ctx non-null, but CURRENT_RUNTIME_CTX TLS NOT installed →
        // on_owning_executor returns false → off-thread branch →
        // push to ctx.queue + parked check.
        let ctx = make_ctx();
        let ptr = make_spawned_task(&ctx);

        unsafe {
            drive_to_terminal(ptr);
            // No install_runtime_cross_wake → TLS null.
            dispose_terminal(ptr);
        }

        // Verify push to cross-queue.
        let popped = ctx.queue.pop();
        assert_eq!(popped, Some(ptr));
        assert!(unsafe { crate::task::is_queued(ptr) });

        unsafe {
            crate::task::clear_queued(ptr);
            crate::task::free_task(ptr);
        }
    }
}

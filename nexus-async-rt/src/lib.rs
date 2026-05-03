//! Single-threaded async runtime.
//!
//! Two spawn strategies:
//! - **`spawn_boxed()`** — Box-allocated. Default. No setup needed.
//! - **`spawn_slab()`** — Slab-allocated. Pre-allocated, zero-alloc
//!   hot path. Requires slab configured via [`RuntimeBuilder::slab_unbounded`] or [`RuntimeBuilder::slab_bounded`].
//!
//! ```ignore
//! use nexus_async_rt::*;
//! use nexus_slab::byte::unbounded::Slab;
//! use nexus_rt::WorldBuilder;
//!
//! let mut world = WorldBuilder::new().build();
//!
//! // Simple — Box-allocated tasks, no slab setup
//! let mut rt = Runtime::new(&mut world);
//! rt.block_on(async {
//!     spawn_boxed(async { /* Box-allocated */ });
//! });
//!
//! // Power user — with slab for hot-path tasks
//! // SAFETY: single-threaded runtime.
//! let slab = unsafe { Slab::<256>::with_chunk_capacity(64) };
//! let mut rt = Runtime::builder(&mut world)
//!     .slab_unbounded(slab)
//!     .build();
//! rt.block_on(async {
//!     spawn_boxed(async { /* Box-allocated, long-lived */ });
//!     spawn_slab(async { /* slab-allocated, hot path */ });
//! });
//! ```

// Single-threaded runtime — futures are intentionally !Send.
#![allow(clippy::future_not_send)]
#![cfg(unix)]

mod alloc;
mod backoff;
mod cancel;
pub mod channel;
mod context;
pub(crate) mod cross_wake;
mod io;
pub mod net;
mod runtime;
mod shutdown;
mod task;
mod timer;
#[cfg(feature = "tokio-compat")]
pub mod tokio_compat;
#[cfg(feature = "tokio-compat")]
pub use tokio_compat::{TokioJoinError, TokioJoinHandle, spawn_on_tokio};
mod waker;
mod world_ctx;

// Re-export slab type for convenience — users create the slab and hand it to the builder.
pub use alloc::SlabClaim;
pub use backoff::{Backoff, BackoffBuilder, Exhausted};
pub use cancel::{CancellationToken, DropGuard};
pub use context::{
    after, after_delay, event_time, interval, interval_at, io, shutdown_signal, sleep, sleep_until,
    timeout, timeout_at, with_world, with_world_ref, yield_now,
};
pub use io::IoHandle;
pub use net::{
    AsyncRead, AsyncWrite, OwnedReadHalf, OwnedWriteHalf, ReadHalf, TcpListener, TcpSocket,
    TcpStream, UdpSocket, WriteHalf,
};
pub use nexus_slab::byte::unbounded::Slab as ByteSlab;
pub use runtime::{Runtime, RuntimeBuilder, claim_slab, spawn_boxed, spawn_slab, try_claim_slab};
pub use shutdown::{ShutdownHandle, ShutdownSignal};
pub use task::{JoinHandle, TASK_HEADER_SIZE};
pub use timer::{Elapsed, Interval, MissedTickBehavior, Sleep, Timeout, TimerHandle, YieldNow};
pub use world_ctx::WorldCtx;

use std::future::Future;
use std::task::{Context, Poll};

use waker::set_poll_context;

/// Recommended minimum slab slot size.
///
/// The actual minimum depends on the task: header (72 bytes) + `max(size_of::<F>(),
/// size_of::<T>())`. ZST futures need only 72 bytes. 128 is a conservative default
/// that covers most small futures.
pub const MIN_SLOT_SIZE: usize = 128;

// =============================================================================
// Executor
// =============================================================================

/// Single-threaded async executor.
///
/// Manages task lifecycle: spawn, poll, complete, free. Tasks are
/// allocated via Box (default) or slab (via `spawn_slab`). Each
/// task's header contains a `free_fn` that knows how to deallocate
/// its own storage — the executor doesn't know or care which
/// allocator was used.
/// # UnsafeCell on `incoming` and `deferred_free`
///
/// These fields are wrapped in `UnsafeCell` to prevent a provenance
/// aliasing violation. During `poll()`, raw pointers to these Vecs are
/// stored in TLS for wakers to push into. Later in the same `poll()`,
/// `complete_task(&mut self)` takes `&mut self` — which under Rust's
/// aliasing rules asserts exclusive access to ALL fields. Without
/// `UnsafeCell`, this invalidates the TLS pointers because two `&mut`
/// paths to the same memory exist. `UnsafeCell` opts these fields out
/// of `&mut`'s exclusivity guarantee, telling the compiler they may be
/// accessed through other paths (the TLS raw pointers).
///
/// This is NOT a performance concern — `UnsafeCell` is zero-sized and
/// `get()` compiles to a no-op pointer cast. The only effect is that
/// the compiler won't optimize based on exclusive access to these fields.
pub struct Executor {
    /// Incoming ready tasks. Wakers and spawn push here.
    /// Swapped with `draining` at the start of each poll cycle.
    ///
    /// Wrapped in `UnsafeCell` because raw pointers to this Vec are stored
    /// in TLS during `poll()`. Without `UnsafeCell`, `&mut self` on methods
    /// like `complete_task` would invalidate the TLS pointer's provenance
    /// (exclusive `&mut` covers all non-UnsafeCell fields).
    incoming: std::cell::UnsafeCell<Vec<*mut u8>>,

    /// Tasks being drained this cycle. Iterated linearly.
    /// Does NOT need UnsafeCell — only accessed through `&mut self` in poll().
    draining: Vec<*mut u8>,

    /// All live task pointers. Slab-indexed for O(1) removal.
    all_tasks: slab::Slab<*mut u8>,

    /// Number of live tasks.
    live_count: usize,

    /// Maximum tasks to poll per cycle before yielding to IO.
    tasks_per_cycle: usize,

    /// Completed task slots awaiting deferred free.
    ///
    /// Same UnsafeCell rationale as `incoming` — TLS pointer stored during poll.
    deferred_free: std::cell::UnsafeCell<Vec<*mut u8>>,
}

/// Default poll limit.
const DEFAULT_TASKS_PER_CYCLE: usize = 64;

impl Executor {
    /// Create an executor.
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            incoming: std::cell::UnsafeCell::new(Vec::with_capacity(initial_capacity)),
            draining: Vec::with_capacity(initial_capacity),
            all_tasks: slab::Slab::with_capacity(initial_capacity),
            live_count: 0,
            tasks_per_cycle: DEFAULT_TASKS_PER_CYCLE,
            deferred_free: std::cell::UnsafeCell::new(Vec::new()),
        }
    }

    /// Reserve a tracker key for external allocation (slab spawn).
    pub(crate) fn next_tracker_key(&self) -> u32 {
        let key = self.all_tasks.vacant_key();
        debug_assert!(
            u32::try_from(key).is_ok(),
            "more than 4 billion concurrent tasks — tracker_key overflow"
        );
        key as u32
    }

    /// Spawn an async task via Box allocation. Returns a [`JoinHandle`]
    /// that can be awaited for the task's output.
    pub fn spawn_boxed<F>(&mut self, future: F) -> task::JoinHandle<F::Output>
    where
        F: Future + 'static,
        F::Output: 'static,
    {
        let tracker_key = self.all_tasks.vacant_key();
        debug_assert!(
            u32::try_from(tracker_key).is_ok(),
            "more than 4 billion concurrent tasks — tracker_key overflow"
        );
        // Read the runtime's cross-wake context from TLS — installed at
        // RuntimeBuilder::build, lifetime of Runtime. Null when no
        // Runtime is alive (e.g., direct Executor use in tests); the
        // task header's cross_wake_ctx becomes null and dispose_terminal
        // routes those tasks via its null-ctx fallback.
        let cross_wake_ctx = crate::cross_wake::current_runtime_ctx();
        let ptr = task::box_spawn_joinable(future, tracker_key as u32, cross_wake_ctx);

        self.enqueue(ptr);
        task::JoinHandle::new(ptr)
    }

    /// Spawn a task with a pre-allocated pointer (from slab).
    ///
    /// The task at `ptr` must have been constructed with joinable or
    /// fire-and-forget constructors and a valid `free_fn`.
    pub(crate) fn spawn_raw(&mut self, ptr: *mut u8) {
        self.enqueue(ptr);
    }

    /// Common enqueue logic for spawn and spawn_raw.
    fn enqueue(&mut self, ptr: *mut u8) {
        self.all_tasks.insert(ptr);
        unsafe { task::set_queued(ptr, true) };
        // SAFETY: single-threaded, no concurrent access during enqueue.
        unsafe { &mut *self.incoming.get() }.push(ptr);
        self.live_count += 1;
    }

    /// Drain the cross-thread wake inbox into the local ready queue.
    ///
    /// Called at the start of each poll cycle. Tasks pushed from other
    /// threads via `CrossWakeQueue::push` are moved into `incoming`.
    /// Completed tasks are routed to `deferred_free` instead — they
    /// were pushed for cleanup (not re-polling) by `cross_task_drop`.
    /// Drains at most `limit` tasks (remaining are picked up next cycle).
    pub(crate) fn drain_cross_thread(
        &mut self,
        inbox: &crate::cross_wake::CrossWakeQueue,
        limit: usize,
    ) {
        let mut drained = 0;
        while drained < limit {
            match inbox.pop() {
                Some(task_ptr) => {
                    // Clear QUEUED flag now that we've popped it.
                    unsafe { task::clear_queued(task_ptr) };

                    // Check if TERMINAL was reached (e.g., cross-thread waker
                    // produced TERMINAL via ref_dec while the task was queued).
                    // Only TERMINAL tasks go to deferred_free. Completed tasks
                    // with outstanding refs must NOT be freed prematurely.
                    if unsafe { task::is_terminal(task_ptr) } {
                        unsafe { &mut *self.deferred_free.get() }.push(task_ptr);
                    } else {
                        unsafe { &mut *self.incoming.get() }.push(task_ptr);
                    }
                    drained += 1;
                }
                None => break,
            }
        }
    }

    /// Poll all ready tasks once.
    pub fn poll(&mut self) -> usize {
        let mut completed = 0;

        // Drain deferred frees from last cycle.
        // SAFETY: single-threaded, TLS not yet set for this cycle.
        for ptr in unsafe { &mut *self.deferred_free.get() }.drain(..) {
            let key = unsafe { task::tracker_key(ptr) } as usize;
            // SAFETY: free_fn was set at spawn time.
            unsafe { task::free_task(ptr) };
            if self.all_tasks.contains(key) {
                self.all_tasks.remove(key);
            }
        }

        // SAFETY: single-threaded, swapping before TLS is set.
        std::mem::swap(unsafe { &mut *self.incoming.get() }, &mut self.draining);

        // Derive TLS pointers from UnsafeCell — NOT from &mut self field borrows.
        // This is critical: complete_task(&mut self) later in this function must
        // not invalidate the TLS pointers. UnsafeCell fields are excluded from
        // &mut self's exclusivity guarantee.
        let _guard = set_poll_context(self.incoming.get(), self.deferred_free.get());

        let limit = self.tasks_per_cycle.min(self.draining.len());
        let draining_ptr: *const Vec<*mut u8> = &raw const self.draining;
        let drain_slice = unsafe { &(&*draining_ptr)[..limit] };

        for &ptr in drain_slice {
            if unsafe { task::is_completed(ptr) } {
                continue;
            }

            unsafe { task::set_queued(ptr, false) };

            // SAFETY: ptr is a live task, ref_count >= 1 (executor holds a ref).
            // task_waker increments ref_count; drop after poll decrements it.
            let waker = unsafe { crate::waker::task_waker(ptr) };
            let mut cx = Context::from_waker(&waker);

            let poll_result = unsafe { task::poll_task(ptr, &mut cx) };

            drop(waker);

            match poll_result {
                Poll::Pending => {}
                Poll::Ready(()) => {
                    self.complete_task(ptr);
                    completed += 1;
                }
            }
        }

        if limit < self.draining.len() {
            // SAFETY: single-threaded, TLS guard is about to drop.
            unsafe { &mut *self.incoming.get() }.extend_from_slice(&self.draining[limit..]);
        }
        self.draining.clear();

        completed
    }

    /// Number of live tasks.
    pub fn task_count(&self) -> usize {
        self.live_count
    }

    /// Number of completed task slots awaiting deferred free.
    #[cfg(test)]
    pub fn deferred_free_count(&self) -> usize {
        // SAFETY: single-threaded, read-only snapshot.
        unsafe { &*self.deferred_free.get() }.len()
    }

    /// Returns `true` if any tasks are queued for polling.
    pub fn has_ready(&self) -> bool {
        // SAFETY: single-threaded, read-only snapshot.
        !unsafe { &*self.incoming.get() }.is_empty()
    }

    /// Set the maximum tasks to poll per cycle.
    pub fn set_tasks_per_cycle(&mut self, limit: usize) {
        self.tasks_per_cycle = limit;
    }

    /// Complete a task: handle joinable vs fire-and-forget paths.
    ///
    /// Uses `complete_and_unref` to atomically set COMPLETED and decrement
    /// the executor's reference in a single atomic operation — eliminating
    /// the race window that caused SIGABRT with cross-thread wakers.
    ///
    /// Three branches based on task state:
    /// - **Aborted:** drop F (still live — poll_join short-circuited), notify joiner
    /// - **Joinable (HAS_JOIN):** T is live in the union, don't touch it — JoinHandle owns it
    /// - **Fire-and-forget / detached:** drop the value (F or T) and free
    ///
    /// # Safety invariants
    ///
    /// `ptr` must point to a task that just returned `Poll::Ready(())` from poll_task.
    fn complete_task(&mut self, ptr: *mut u8) {
        let aborted = unsafe { task::is_aborted(ptr) };

        if aborted {
            // Aborted: poll_join saw ABORTED and returned Ready without polling F.
            // F is still live in the union. drop_fn still targets F.
            unsafe { task::drop_task_future(ptr) };
            self.live_count -= 1;

            if unsafe { task::has_join(ptr) } {
                let waker = unsafe { task::take_join_waker(ptr) };
                if let Some(w) = waker {
                    w.wake();
                }
            }

            match unsafe { task::complete_and_unref(ptr) } {
                task::FreeAction::Retain => {}
                task::FreeAction::FreeBox | task::FreeAction::FreeSlab => {
                    let key = unsafe { task::tracker_key(ptr) } as usize;
                    unsafe { task::free_task(ptr) };
                    self.all_tasks.remove(key);
                }
            }
        } else if unsafe { task::has_join(ptr) } {
            // Joinable: poll_join dropped F and wrote T. drop_fn = drop_output::<T>.
            // Don't drop T — JoinHandle will read it or drop it on handle drop.
            self.live_count -= 1;

            // Wake the joiner so it can poll the JoinHandle and read T.
            let waker = unsafe { task::take_join_waker(ptr) };
            if let Some(w) = waker {
                w.wake();
            }

            match unsafe { task::complete_and_unref(ptr) } {
                task::FreeAction::Retain => {}
                task::FreeAction::FreeBox | task::FreeAction::FreeSlab => {
                    // Terminal — JoinHandle already dropped (detached). Drop output.
                    unsafe { task::drop_task_future(ptr) };
                    let key = unsafe { task::tracker_key(ptr) } as usize;
                    unsafe { task::free_task(ptr) };
                    self.all_tasks.remove(key);
                }
            }
        } else {
            // Fire-and-forget or detached (HAS_JOIN cleared by JoinHandle::Drop).
            unsafe { task::drop_task_future(ptr) };
            self.live_count -= 1;

            match unsafe { task::complete_and_unref(ptr) } {
                task::FreeAction::Retain => {}
                task::FreeAction::FreeBox | task::FreeAction::FreeSlab => {
                    let key = unsafe { task::tracker_key(ptr) } as usize;
                    unsafe { task::free_task(ptr) };
                    self.all_tasks.remove(key);
                }
            }
        }
    }

    /// Returns raw pointers for TLS setup.
    ///
    /// Takes `&self` because `UnsafeCell::get()` only needs a shared reference.
    /// The raw pointers carry write provenance from the `UnsafeCell`.
    pub(crate) fn poll_context_ptrs(&self) -> (*mut Vec<*mut u8>, *mut Vec<*mut u8>) {
        (self.incoming.get(), self.deferred_free.get())
    }

    /// Cancel a task by ID.
    #[allow(dead_code)]
    pub(crate) fn cancel(&mut self, id: task::TaskId) {
        let ptr = id.0;
        // Skip if already completed (e.g. double-cancel or cancel after poll).
        if unsafe { task::is_completed(ptr) } {
            return;
        }
        // SAFETY: single-threaded, no TLS active during cancel.
        unsafe { &mut *self.incoming.get() }.retain(|p| *p != ptr);
        self.draining.retain(|p| *p != ptr);
        self.complete_task(ptr);
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        // Free deferred slots first (completed tasks whose last ref dropped).
        // SAFETY: &mut self in Drop, no concurrent access.
        for ptr in unsafe { &mut *self.deferred_free.get() }.drain(..) {
            let key = unsafe { task::tracker_key(ptr) } as usize;
            unsafe { task::free_task(ptr) };
            if self.all_tasks.contains(key) {
                self.all_tasks.remove(key);
            }
        }

        for (_, &ptr) in &self.all_tasks {
            if unsafe { task::is_terminal(ptr) } {
                // TERMINAL: completed, zero refs, all flags cleared.
                // This happens when a cross-thread waker produced TERMINAL
                // via ref_dec but the executor hadn't scanned yet.
                unsafe { task::free_task(ptr) };
                continue;
            }

            // Drop the future if not already completed.
            if !unsafe { task::is_completed(ptr) } {
                unsafe { task::drop_task_future(ptr) };
                // Use complete_and_unref to atomically set COMPLETED + dec ref.
                match unsafe { task::complete_and_unref(ptr) } {
                    task::FreeAction::Retain => {}
                    task::FreeAction::FreeBox | task::FreeAction::FreeSlab => {
                        unsafe { task::free_task(ptr) };
                        continue;
                    }
                }
            }

            // Task is completed but not TERMINAL — outstanding refs exist.
            let rc = unsafe { task::ref_count(ptr) };
            if rc > 0 {
                if std::thread::panicking() {
                    // Mid-unwind — must not double-panic (would abort the
                    // process via SIGABRT). Resources held by the task
                    // were already released by `drop_task_future` above
                    // (Aeron publishers, sockets, file handles all run
                    // their Drop impls there).
                    //
                    // Cleanup behavior differs by allocation type because
                    // of how the task memory gets reclaimed:
                    //
                    // - **Box tasks**: leaking is safe. The Box just sits
                    //   in process memory; outstanding cross-thread waker
                    //   refs that later run `ref_dec` see valid memory.
                    //   Memory is reclaimed at process exit.
                    //
                    // - **Slab tasks**: leaking is UNSAFE. After this
                    //   `Executor::drop` returns, the `_slab_guard`
                    //   field on Runtime drops, freeing the slab's
                    //   backing storage. Outstanding cross-thread waker
                    //   refs that later run `ref_dec` would access
                    //   freed slab memory → UAF.
                    //
                    //   For slab tasks, we wait briefly for cross-thread
                    //   wakers to drop their refs (this happens
                    //   asynchronously on producer threads — e.g.,
                    //   tokio's worker thread). If they settle within
                    //   the deadline, we free cleanly. If not, we abort
                    //   — the original SIGABRT we were trying to avoid,
                    //   but UAF would be worse.
                    if unsafe { task::is_slab_allocated(ptr) } {
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_millis(100);
                        while unsafe { task::ref_count(ptr) } > 0
                            && std::time::Instant::now() < deadline
                        {
                            std::thread::yield_now();
                        }
                        if unsafe { task::ref_count(ptr) } > 0 {
                            eprintln!(
                                "nexus-async-rt: slab task {ptr:p} has \
                                 outstanding refs after 100ms during unwinding \
                                 — aborting to avoid UAF on slab memory \
                                 release. Cross-thread waker producer thread \
                                 may be deadlocked or starved."
                            );
                            std::process::abort();
                        }
                        // Refs settled — free cleanly. Avoid the panic
                        // path below.
                        unsafe { task::free_task(ptr) };
                        continue;
                    }
                    // Box task — leak is safe.
                    eprintln!(
                        "nexus-async-rt: executor dropped with {rc} outstanding \
                         reference(s) during unwinding — suppressing panic to \
                         avoid abort. Task resources were released via \
                         drop_task_future; leaking box task allocation + waker \
                         bookkeeping memory."
                    );
                    continue;
                }

                // Normal shutdown (no panic in flight) — sanity-check the
                // user's lifetime discipline.
                #[cfg(debug_assertions)]
                panic!(
                    "executor dropped with {rc} outstanding reference(s) — \
                     all wakers and JoinHandles must be dropped before the Runtime"
                );
                #[cfg(not(debug_assertions))]
                eprintln!(
                    "nexus-async-rt: executor dropped with {rc} outstanding task \
                     reference(s) — leaking to avoid UB"
                );
                #[allow(unreachable_code)]
                {
                    continue;
                }
            }

            unsafe { task::free_task(ptr) };
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::pin::Pin;
    use task::Task;

    fn test_executor() -> Executor {
        Executor::new(16)
    }

    // =========================================================================
    // Basic spawn + poll
    // =========================================================================

    #[test]
    fn spawn_and_poll_single_task() {
        let mut exec = test_executor();
        let mut done = false;
        let flag = &raw mut done;

        exec.spawn_boxed(async move {
            // SAFETY: single-threaded, flag lives on stack.
            unsafe { *flag = true };
        });

        assert_eq!(exec.task_count(), 1);
        let completed = exec.poll();
        assert_eq!(completed, 1);
        assert!(done);
        assert_eq!(exec.task_count(), 0);
    }

    #[test]
    fn spawn_multiple_tasks() {
        let mut exec = test_executor();

        for _ in 0..8 {
            exec.spawn_boxed(async {});
        }

        assert_eq!(exec.task_count(), 8);
        let completed = exec.poll();
        assert_eq!(completed, 8);
        assert_eq!(exec.task_count(), 0);
    }

    // =========================================================================
    // Pending tasks
    // =========================================================================

    #[test]
    fn pending_task_not_completed() {
        let mut exec = test_executor();

        // A future that is always pending.
        exec.spawn_boxed(std::future::pending::<()>());

        let completed = exec.poll();
        assert_eq!(completed, 0);
        assert_eq!(exec.task_count(), 1);
    }

    // =========================================================================
    // Waker: re-queue via wake_by_ref
    // =========================================================================

    #[test]
    fn immediate_task_completes() {
        let mut exec = test_executor();

        exec.spawn_boxed(async {
            // Immediately ready.
        });

        let completed = exec.poll();
        assert_eq!(completed, 1);
        assert_eq!(exec.task_count(), 0);
    }

    // =========================================================================
    // Self-waking task
    // =========================================================================

    #[test]
    fn self_waking_task_polled_again() {
        use std::cell::Cell;
        use std::rc::Rc;

        let mut exec = test_executor();

        let counter = Rc::new(Cell::new(0u32));
        let c = counter.clone();

        exec.spawn_boxed(async move {
            struct SelfWake {
                counter: Rc<Cell<u32>>,
            }
            impl Future for SelfWake {
                type Output = ();
                fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
                    let n = self.counter.get();
                    self.counter.set(n + 1);
                    if n < 3 {
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    } else {
                        Poll::Ready(())
                    }
                }
            }
            SelfWake { counter: c }.await;
        });

        // Drain all polls.
        let mut total = 0;
        for _ in 0..10 {
            total += exec.poll();
            if exec.task_count() == 0 {
                break;
            }
        }
        assert_eq!(total, 1); // completed once
        assert_eq!(counter.get(), 4); // polled 4 times
    }

    // =========================================================================
    // Cancel
    // =========================================================================

    #[test]
    fn abort_task() {
        let mut exec = test_executor();
        let handle = exec.spawn_boxed(std::future::pending::<()>());

        assert_eq!(exec.task_count(), 1);
        assert!(handle.abort()); // was running, handle consumed
        exec.poll(); // abort takes effect on next poll
        assert_eq!(exec.task_count(), 0);
    }

    #[test]
    fn abort_frees_slot_for_reuse() {
        let mut exec = test_executor();
        let handle = exec.spawn_boxed(std::future::pending::<()>());
        handle.abort(); // consumes handle

        exec.poll(); // process abort + deferred free

        // Should be able to spawn again.
        exec.spawn_boxed(async {});
        assert_eq!(exec.task_count(), 1);
        exec.poll();
        assert_eq!(exec.task_count(), 0);
    }

    // =========================================================================
    // Poll limit (tasks_per_cycle)
    // =========================================================================

    #[test]
    fn poll_limit_respected() {
        let mut exec = test_executor();
        exec.set_tasks_per_cycle(2);

        for _ in 0..5 {
            exec.spawn_boxed(async {});
        }

        // Only 2 polled per cycle.
        let completed = exec.poll();
        assert_eq!(completed, 2);
        assert_eq!(exec.task_count(), 3);

        let completed = exec.poll();
        assert_eq!(completed, 2);
        assert_eq!(exec.task_count(), 1);

        let completed = exec.poll();
        assert_eq!(completed, 1);
        assert_eq!(exec.task_count(), 0);
    }

    // =========================================================================
    // Stale ready entries after cancel
    // =========================================================================

    #[test]
    fn cancel_with_stale_ready_entry() {
        use std::cell::Cell;
        use std::rc::Rc;

        let mut exec = test_executor();

        let polled = Rc::new(Cell::new(false));
        let p = polled.clone();

        // Spawn a self-waking task.
        struct WakeOnce(bool);
        impl Future for WakeOnce {
            type Output = ();
            fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
                if !self.0 {
                    self.0 = true;
                    cx.waker().wake_by_ref();
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            }
        }

        let handle = exec.spawn_boxed(WakeOnce(false));

        // First poll: sets is_queued again via wake_by_ref.
        exec.poll();

        // Abort while the task is in the ready queue (consumes handle).
        handle.abort();

        // Spawn a new task to prove we don't crash on the stale pointer.
        exec.spawn_boxed(async move {
            p.set(true);
        });

        exec.poll(); // processes abort + new task
        assert!(polled.get());
    }

    // =========================================================================
    // Refcount behavior
    // =========================================================================

    #[test]
    fn refcount_starts_at_one() {
        let task = Box::new(Task::new_boxed(async {}, 0));
        let ptr = Box::into_raw(task) as *mut u8;
        assert_eq!(unsafe { task::ref_count(ptr) }, 1);
        unsafe { task::free_task(ptr) };
    }

    #[test]
    fn executor_drop_cleans_up_queued_tasks() {
        let mut exec = test_executor();
        exec.spawn_boxed(std::future::pending::<()>());
        exec.spawn_boxed(std::future::pending::<()>());
        exec.poll(); // poll them once
        // Drop executor — should free all tasks without panic.
        drop(exec);
    }

    // =========================================================================
    // Dispatch latency (rough, not controlled)
    // =========================================================================

    #[test]
    #[ignore]
    fn dispatch_latency() {
        use std::time::Instant;

        struct Noop;
        impl Future for Noop {
            type Output = ();
            fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }

        let mut exec = test_executor();
        exec.spawn_boxed(Noop);

        // Warmup.
        for _ in 0..10_000 {
            exec.poll();
        }

        let iters = 100_000;
        let start = Instant::now();
        for _ in 0..iters {
            exec.poll();
        }
        let elapsed = start.elapsed();
        let ns_per = elapsed.as_nanos() / iters;
        println!("dispatch: {ns_per} ns/poll (Box-allocated)");
        black_box(ns_per);
    }
}

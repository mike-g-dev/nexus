//! Graceful shutdown support.
//!
//! [`ShutdownSignal`] is a future that resolves when a shutdown is
//! requested — either by a Unix signal (SIGTERM, SIGINT) or by
//! explicitly calling [`ShutdownHandle::trigger`].
//!
//! The Runtime checks the shutdown flag each poll cycle. When set,
//! the root future can observe it via the `ShutdownSignal` future
//! and begin connection draining.
//!
//! # Usage
//!
//! ```ignore
//! let mut rt = Runtime::new(&mut world);
//!
//! // Install signal handlers (call once at startup).
//! rt.install_signal_handlers();
//!
//! rt.block_on(async move {
//!     spawn_boxed(connection_tasks...);
//!
//!     // Wait for SIGTERM/SIGINT.
//!     nexus_async_rt::ShutdownSignal::current().await;
//!
//!     // Drain connections, flush buffers, etc.
//! });
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Waker};

/// Shared shutdown flag.
#[derive(Clone)]
pub struct ShutdownHandle {
    flag: Arc<AtomicBool>,
    /// Mio waker to break epoll_wait when shutdown is triggered.
    mio_waker: Option<Arc<mio::Waker>>,
    /// Task waker slot — the ShutdownSignal future registers here.
    /// Protected by Mutex because the signal handler thread may
    /// call wake(). Only contested at shutdown time (once per process).
    pub(crate) task_waker: Arc<std::sync::Mutex<Option<Waker>>>,
}

impl ShutdownHandle {
    pub(crate) fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            mio_waker: None,
            task_waker: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Set the mio waker. Called by Runtime during construction.
    pub(crate) fn set_mio_waker(&mut self, waker: Arc<mio::Waker>) {
        self.mio_waker = Some(waker);
    }

    /// Trigger shutdown programmatically.
    ///
    /// Sets the flag, wakes the registered task waker (if any), and
    /// breaks epoll_wait so the runtime loop re-polls the root future.
    pub fn trigger(&self) {
        self.flag.store(true, Ordering::Release);
        // Wake the task waker first — signal the future directly.
        if let Ok(mut guard) = self.task_waker.lock()
            && let Some(w) = guard.take()
        {
            w.wake();
        }
        if let Some(w) = &self.mio_waker {
            let _ = w.wake();
        }
    }

    /// Check if shutdown has been requested.
    pub fn is_shutdown(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Get the underlying flag Arc for signal handler registration.
    pub(crate) fn flag_ptr(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.flag)
    }

    /// Returns a future that completes when shutdown is triggered.
    pub fn signal(&self) -> ShutdownSignal {
        ShutdownSignal {
            flag: Arc::as_ptr(&self.flag),
            task_waker: self.task_waker.clone(),
        }
    }
}

/// Future that resolves when shutdown is triggered.
///
/// Registers (and updates) a waker on every poll so that
/// `ShutdownHandle::trigger()` (or a signal handler) can wake the
/// awaiting task directly. The waker is overwritten on each poll to
/// handle the case where the future is re-polled from a different
/// task context.
///
/// **Single waiter only.** Only one task may await `ShutdownSignal` at a
/// time. If a second task polls while a waker is already registered, the
/// waker is replaced (not duplicated). For multi-waiter shutdown, use
/// [`CancellationToken`](crate::CancellationToken) instead.
///
/// Holds a raw pointer to the AtomicBool flag, valid for the lifetime
/// of the Runtime (which outlives `block_on` which outlives all tasks).
pub struct ShutdownSignal {
    pub(crate) flag: *const AtomicBool,
    pub(crate) task_waker: Arc<std::sync::Mutex<Option<Waker>>>,
}

impl ShutdownSignal {
    /// Returns a [`ShutdownSignal`] future for the currently running runtime.
    ///
    /// The returned future resolves when shutdown is triggered — either by
    /// a Unix signal handler installed via
    /// [`Runtime::install_signal_handlers`](crate::Runtime::install_signal_handlers)
    /// (SIGTERM / SIGINT) or by an explicit
    /// [`ShutdownHandle::trigger`] call. Mirrors
    /// `tokio::runtime::Handle::current()`. Read as
    /// `ShutdownSignal::current().await` — "await the current shutdown
    /// signal".
    ///
    /// **Single waiter only** — see the type-level docs. For multi-waiter
    /// patterns, use [`CancellationToken`](crate::CancellationToken).
    ///
    /// # Panics
    ///
    /// Panics if called outside a [`Runtime::block_on`](crate::Runtime::block_on)
    /// context.
    #[must_use]
    pub fn current() -> ShutdownSignal {
        let (flag, waker_ptr) = crate::context::current_shutdown_ptrs();
        assert!(
            !flag.is_null(),
            "ShutdownSignal::current() called outside Runtime::block_on"
        );
        // Defense-in-depth: flag and waker_ptr are written together by
        // install(), so this should be unreachable — but a future refactor
        // that splits the install path would make a null waker_ptr deref UB.
        // Catch it at the call site instead.
        assert!(
            !waker_ptr.is_null(),
            "ShutdownSignal::current(): waker_ptr null while flag non-null (runtime install bug)"
        );
        // SAFETY: install() writes flag and waker_ptr together; both verified
        // non-null above. waker_ptr points to the Arc<Mutex<Option<Waker>>>
        // inside the Runtime's ShutdownHandle. Valid for Runtime lifetime
        // (block_on borrows &mut Runtime, which outlives all tasks).
        let task_waker = unsafe { (*waker_ptr).clone() };
        ShutdownSignal { flag, task_waker }
    }
}

impl Future for ShutdownSignal {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        // SAFETY: flag points to the AtomicBool inside the Runtime's
        // ShutdownHandle (Arc-allocated, heap-stable address). Valid for
        // Runtime lifetime — block_on borrows &mut Runtime which outlives
        // all tasks. AtomicBool access is inherently thread-safe.
        if unsafe { &*self.flag }.load(Ordering::Acquire) {
            return Poll::Ready(());
        }

        // Register (or update) the waker so trigger() can wake us.
        // Always update — the waker may have changed if the future was
        // re-polled from a different task context.
        if let Ok(mut guard) = self.task_waker.lock() {
            *guard = Some(cx.waker().clone());
        }

        // SAFETY: Same pointer, same invariant as above — flag is valid
        // for the Runtime lifetime.
        if unsafe { &*self.flag }.load(Ordering::Acquire) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// Install signal handlers for SIGTERM and SIGINT that trigger shutdown.
///
/// Uses `signal-hook` for safe, portable signal registration. The
/// handler atomically sets the flag. The mio waker breaks epoll_wait
/// so the runtime notices the flag promptly.
pub fn install_signal_handlers(flag: &Arc<AtomicBool>, mio_waker: &Arc<mio::Waker>) {
    let waker_ref = Arc::clone(mio_waker);

    // signal-hook provides safe registration with proper cleanup.
    // The closure runs in signal context — only async-signal-safe
    // operations (atomic store + eventfd write).
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(flag))
        .expect("failed to register SIGTERM handler");
    signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(flag))
        .expect("failed to register SIGINT handler");

    // signal-hook::flag::register sets the AtomicBool on signal, but
    // we also need to break epoll_wait. Register a second handler that
    // fires the mio waker.
    // SAFETY: The closure is async-signal-safe — mio::Waker::wake() is
    // an eventfd write (single syscall, no locks, no allocations). The
    // Arc is cloned before registration so the waker outlives the handler.
    unsafe {
        signal_hook::low_level::register(signal_hook::consts::SIGTERM, move || {
            let _ = waker_ref.wake();
        })
        .expect("failed to register SIGTERM waker");
    }
    let waker_ref2 = Arc::clone(mio_waker);
    // SAFETY: Same as above — async-signal-safe eventfd write only.
    unsafe {
        signal_hook::low_level::register(signal_hook::consts::SIGINT, move || {
            let _ = waker_ref2.wake();
        })
        .expect("failed to register SIGINT waker");
    }
}

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
    fn shutdown_handle_trigger() {
        let handle = ShutdownHandle::new();
        assert!(!handle.is_shutdown());
        handle.trigger();
        assert!(handle.is_shutdown());
    }

    #[test]
    fn shutdown_signal_resolves_after_trigger() {
        use crate::{Runtime, spawn_boxed};
        use nexus_rt::WorldBuilder;
        use std::cell::Cell;
        use std::rc::Rc;

        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let mut rt = Runtime::new(&mut world);
        let shutdown = rt.shutdown_handle();

        let done = Rc::new(Cell::new(false));
        let flag = done.clone();

        // Trigger shutdown from a spawned task after a short delay.
        let sh = shutdown.clone();
        rt.block_on(async move {
            spawn_boxed(async move {
                crate::context::sleep(std::time::Duration::from_millis(50)).await;
                sh.trigger();
            });

            // Root future waits for shutdown.
            shutdown.signal().await;
            flag.set(true);
        });

        assert!(done.get());
    }

    #[test]
    #[should_panic(expected = "called outside Runtime::block_on")]
    fn shutdown_signal_current_panics_outside_runtime() {
        // Pins the documented panic contract for
        // `ShutdownSignal::current()`. Symmetric to
        // `IoHandle::current_panics_outside_runtime` and
        // `WorldCtx::current_panics_outside_runtime`.
        let _ = ShutdownSignal::current();
    }

    #[test]
    fn shutdown_signal_current_resolves_after_trigger() {
        // Sister test to `shutdown_signal_resolves_after_trigger`, but
        // exercises the TLS-fetcher path (`ShutdownSignal::current()`)
        // instead of `handle.signal()`. Catches regressions in the
        // CTX_SHUTDOWN / CTX_SHUTDOWN_WAKER install/uninstall wiring.
        use crate::{Runtime, spawn_boxed};
        use nexus_rt::WorldBuilder;
        use std::cell::Cell;
        use std::rc::Rc;

        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let mut rt = Runtime::new(&mut world);
        let shutdown = rt.shutdown_handle();

        let done = Rc::new(Cell::new(false));
        let flag = done.clone();

        let sh = shutdown;
        rt.block_on(async move {
            spawn_boxed(async move {
                crate::context::sleep(std::time::Duration::from_millis(50)).await;
                sh.trigger();
            });

            // Fetch the signal via the TLS-based current() rather than
            // handle.signal() — this is the path users will hit.
            ShutdownSignal::current().await;
            flag.set(true);
        });

        assert!(done.get());
    }

    #[test]
    fn shutdown_signal_waker_updates_on_repoll() {
        // Verify the waker is updated on each poll (not stale from first poll).
        use std::task::{RawWaker, RawWakerVTable, Waker};

        let handle = ShutdownHandle::new();
        let mut signal = Box::pin(handle.signal());

        // First poll with noop waker — registers it.
        // SAFETY: all vtable fns are no-ops; null data is never deref'd.
        let noop = unsafe {
            static V: RawWakerVTable =
                RawWakerVTable::new(|p| RawWaker::new(p, &V), |_| {}, |_| {}, |_| {});
            Waker::from_raw(RawWaker::new(std::ptr::null(), &V))
        };
        let mut cx = Context::from_waker(&noop);
        assert_eq!(signal.as_mut().poll(&mut cx), Poll::Pending);

        // Second poll with a tracking waker — should overwrite.
        let woke = std::cell::Cell::new(false);
        let flag_ptr = &raw const woke as *const ();
        // SAFETY: flag_ptr points to the stack-local `woke` Cell which
        // outlives the waker. The vtable wake/wake_by_ref cast back to
        // Cell<bool> and set true. clone copies the raw pointer. drop
        // is a no-op.
        let tracking = unsafe {
            static V2: RawWakerVTable = RawWakerVTable::new(
                |p| RawWaker::new(p, &V2),
                // SAFETY: p is flag_ptr, valid for the test's lifetime.
                |p| unsafe { (*(p as *const std::cell::Cell<bool>)).set(true) },
                // SAFETY: p is flag_ptr, valid for the test's lifetime.
                |p| unsafe { (*(p as *const std::cell::Cell<bool>)).set(true) },
                |_| {},
            );
            Waker::from_raw(RawWaker::new(flag_ptr, &V2))
        };
        let mut cx2 = Context::from_waker(&tracking);
        assert_eq!(signal.as_mut().poll(&mut cx2), Poll::Pending);

        // Trigger shutdown — must wake the tracking waker, not the noop.
        handle.trigger();
        assert!(woke.get(), "latest waker must fire on trigger");
    }

    #[test]
    fn shutdown_signal_already_triggered() {
        // Trigger before first poll — immediate Ready, no waker registration.
        use std::task::{RawWaker, RawWakerVTable, Waker};

        let handle = ShutdownHandle::new();
        handle.trigger();

        let mut signal = Box::pin(handle.signal());
        // SAFETY: all vtable fns are no-ops; null data is never deref'd.
        let waker = unsafe {
            static V: RawWakerVTable =
                RawWakerVTable::new(|p| RawWaker::new(p, &V), |_| {}, |_| {}, |_| {});
            Waker::from_raw(RawWaker::new(std::ptr::null(), &V))
        };
        let mut cx = Context::from_waker(&waker);
        assert_eq!(signal.as_mut().poll(&mut cx), Poll::Ready(()));
    }
}

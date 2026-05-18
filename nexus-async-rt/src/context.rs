//! Thread-local runtime context.
//!
//! Two access shapes for runtime state, by intent:
//!
//! - **Handles for the current runtime** — [`IoHandle::current`](crate::IoHandle::current),
//!   [`WorldCtx::current`](crate::WorldCtx::current),
//!   [`ShutdownSignal::current`](crate::ShutdownSignal::current). Inherent
//!   `current()` methods on the type, mirroring `tokio::runtime::Handle::current()`.
//!   Use when you need the handle/future itself.
//! - **Future factories and value getters** — free functions [`sleep`],
//!   [`sleep_until`], [`interval`], [`interval_at`], [`after`],
//!   [`after_delay`], [`timeout`], [`timeout_at`], [`yield_now`],
//!   [`event_time`]. These produce a value and don't fit the `Type::current()`
//!   shape (the future is the API; there's no enclosing handle to fetch).
//!
//! All readers panic if called outside a [`Runtime::block_on`](crate::Runtime::block_on)
//! context. The TLS slots are installed by `block_on` and cleared on exit;
//! const-initialized for zero first-access cost.
//!
//! ```ignore
//! use nexus_async_rt::{spawn_boxed, sleep, WorldCtx, ShutdownSignal, TcpListener};
//!
//! rt.block_on(async {
//!     spawn_boxed(async {
//!         WorldCtx::current().with_world(|world| { /* ... */ });
//!         sleep(Duration::from_secs(1)).await;
//!         let listener = TcpListener::bind(addr);  // fetches IoHandle::current() internally
//!     });
//!     ShutdownSignal::current().await;
//! });
//! ```

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use crate::io::IoDriver;
use crate::timer::{TimerDriver, TimerHandle};

// =============================================================================
// TLS slots — const-initialized, zero first-access cost
// =============================================================================

thread_local! {
    static CTX_WORLD: Cell<*mut nexus_rt::World> =
        const { Cell::new(std::ptr::null_mut()) };
    static CTX_IO: Cell<*mut IoDriver> =
        const { Cell::new(std::ptr::null_mut()) };
    static CTX_TIMER: Cell<*mut TimerDriver> =
        const { Cell::new(std::ptr::null_mut()) };
    static CTX_EVENT_TIME: Cell<*const Cell<Instant>> =
        const { Cell::new(std::ptr::null()) };
    static CTX_SHUTDOWN: Cell<*const AtomicBool> =
        const { Cell::new(std::ptr::null()) };
    static CTX_SHUTDOWN_WAKER: Cell<*const Arc<std::sync::Mutex<Option<std::task::Waker>>>> =
        const { Cell::new(std::ptr::null()) };
}

// =============================================================================
// Install / clear (called by Runtime::block_on)
// =============================================================================

/// Install runtime context into TLS. Called by both `Runtime::block_on`
/// (root execution path) and `Runtime::shutdown_quiesce` (so cross-thread
/// wakes that fire during quiesce still find a runtime context).
/// The context stays installed until the returned guard is dropped.
pub(crate) fn install(
    world: *mut nexus_rt::World,
    io: *mut IoDriver,
    timer: *mut TimerDriver,
    event_time: *const Cell<Instant>,
    shutdown_flag: *const AtomicBool,
    shutdown_waker: *const Arc<std::sync::Mutex<Option<std::task::Waker>>>,
) -> ContextGuard {
    let prev = PrevContext {
        world: CTX_WORLD.with(|c| c.replace(world)),
        io: CTX_IO.with(|c| c.replace(io)),
        timer: CTX_TIMER.with(|c| c.replace(timer)),
        event_time: CTX_EVENT_TIME.with(|c| c.replace(event_time)),
        shutdown: CTX_SHUTDOWN.with(|c| c.replace(shutdown_flag)),
        shutdown_waker: CTX_SHUTDOWN_WAKER.with(|c| c.replace(shutdown_waker)),
    };
    ContextGuard { prev }
}

struct PrevContext {
    world: *mut nexus_rt::World,
    io: *mut IoDriver,
    timer: *mut TimerDriver,
    event_time: *const Cell<Instant>,
    shutdown: *const AtomicBool,
    shutdown_waker: *const Arc<std::sync::Mutex<Option<std::task::Waker>>>,
}

pub(crate) struct ContextGuard {
    prev: PrevContext,
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        CTX_WORLD.with(|c| c.set(self.prev.world));
        CTX_IO.with(|c| c.set(self.prev.io));
        CTX_TIMER.with(|c| c.set(self.prev.timer));
        CTX_EVENT_TIME.with(|c| c.set(self.prev.event_time));
        CTX_SHUTDOWN.with(|c| c.set(self.prev.shutdown));
        CTX_SHUTDOWN_WAKER.with(|c| c.set(self.prev.shutdown_waker));
    }
}

/// Assert that we're inside a runtime context. Panics with `msg` if not.
pub(crate) fn assert_in_runtime(msg: &str) {
    let ptr = CTX_WORLD.with(Cell::get);
    assert!(!ptr.is_null(), "{msg}");
}

// =============================================================================
// pub(crate) TLS readers — back the inherent `Type::current()` methods on
// `IoHandle`, `WorldCtx`, and `ShutdownSignal`. Kept in this module so the
// `CTX_*` thread-locals don't need to be exposed elsewhere in the crate.
// =============================================================================

/// Returns the raw `IoDriver` pointer installed for the current runtime, or
/// null if outside a runtime context.
pub(crate) fn current_io_ptr() -> *mut IoDriver {
    CTX_IO.with(Cell::get)
}

/// Returns the raw `World` pointer installed for the current runtime, or null
/// if outside a runtime context.
pub(crate) fn current_world_ptr() -> *mut nexus_rt::World {
    CTX_WORLD.with(Cell::get)
}

/// Returns `(flag, waker)` pointers for the current runtime's shutdown
/// machinery, or `(null, null)` if outside a runtime context. Both pointers
/// are non-null whenever the runtime is installed (`install` writes them
/// together).
pub(crate) fn current_shutdown_ptrs() -> (
    *const AtomicBool,
    *const Arc<std::sync::Mutex<Option<std::task::Waker>>>,
) {
    let flag = CTX_SHUTDOWN.with(Cell::get);
    let waker = CTX_SHUTDOWN_WAKER.with(Cell::get);
    (flag, waker)
}

/// Create a [`Sleep`](crate::Sleep) future that completes after `duration`.
///
/// # Panics
///
/// Panics if called outside a runtime context.
pub fn sleep(duration: Duration) -> crate::Sleep {
    let ptr = CTX_TIMER.with(Cell::get);
    assert!(!ptr.is_null(), "sleep() called outside Runtime::block_on");
    // SAFETY: ptr was installed by install() from a &mut TimerDriver owned
    // by the Runtime. Valid for Runtime lifetime (block_on borrows &mut self).
    // Single-threaded — no concurrent access.
    let handle = TimerHandle::new(unsafe { &mut *ptr });
    handle.sleep(duration)
}

/// Create a [`Sleep`](crate::Sleep) future that completes at `deadline`.
pub fn sleep_until(deadline: Instant) -> crate::Sleep {
    let ptr = CTX_TIMER.with(Cell::get);
    assert!(
        !ptr.is_null(),
        "sleep_until() called outside Runtime::block_on"
    );
    // SAFETY: ptr was installed by install() from a &mut TimerDriver owned
    // by the Runtime. Valid for Runtime lifetime. Single-threaded.
    let handle = TimerHandle::new(unsafe { &mut *ptr });
    handle.sleep_until(deadline)
}

/// Timestamp taken after the most recent IO poll cycle.
///
/// All events dispatched within the same cycle share this timestamp.
/// One clock read per cycle, not per event.
pub fn event_time() -> Instant {
    let ptr = CTX_EVENT_TIME.with(Cell::get);
    assert!(
        !ptr.is_null(),
        "event_time() called outside Runtime::block_on"
    );
    // SAFETY: ptr was installed by install() from a &Cell<Instant> owned
    // by the Runtime. Valid for Runtime lifetime. Cell::get() is a read
    // (no mutation), single-threaded.
    unsafe { &*ptr }.get()
}

/// Wrap a future with a deadline. Returns `Err(Elapsed)` if the
/// deadline expires before the future completes.
///
/// # Panics
///
/// Panics if called outside a runtime context.
pub fn timeout<F: std::future::Future>(duration: Duration, future: F) -> crate::timer::Timeout<F> {
    crate::timer::Timeout::new(future, sleep(duration))
}

/// Create an interval that ticks at a fixed period.
///
/// The first tick completes after `period`. Subsequent ticks are
/// spaced `period` apart. If processing takes longer than `period`,
/// behavior is controlled by [`MissedTickBehavior`](crate::MissedTickBehavior).
///
/// # Panics
///
/// Panics if `period` is zero. Polling the interval (via `tick().await`)
/// requires an active runtime context and will panic otherwise.
pub fn interval(period: Duration) -> crate::timer::Interval {
    crate::timer::Interval::new(period)
}

/// Run a future no earlier than `deadline`.
///
/// Waits until `deadline`, then polls the future. Useful for
/// scheduling deferred work at a specific time.
///
/// Polling requires an active runtime context.
pub async fn after<F: std::future::Future>(deadline: Instant, future: F) -> F::Output {
    sleep_until(deadline).await;
    future.await
}

/// Run a future after `duration` elapses.
///
/// Waits for `duration`, then polls the future.
///
/// Polling requires an active runtime context.
pub async fn after_delay<F: std::future::Future>(duration: Duration, future: F) -> F::Output {
    sleep(duration).await;
    future.await
}

/// Wrap a future with an absolute deadline. Returns `Err(Elapsed)` if
/// the deadline passes before the future completes.
///
/// Like [`timeout`] but takes an [`Instant`] instead of a [`Duration`].
///
/// # Panics
///
/// Panics if called outside a runtime context.
pub fn timeout_at<F: std::future::Future>(
    deadline: Instant,
    future: F,
) -> crate::timer::Timeout<F> {
    crate::timer::Timeout::new(future, sleep_until(deadline))
}

/// Create an interval that starts ticking at `start`, then every `period`.
///
/// If `start` is in the past, the first tick fires immediately.
///
/// # Panics
///
/// Panics if `period` is zero. Polling the interval requires an active
/// runtime context.
pub fn interval_at(start: Instant, period: Duration) -> crate::timer::Interval {
    crate::timer::Interval::new_at(start, period)
}

/// Cooperatively yield the current task.
///
/// Returns `Pending` once, wakes itself, then completes on the next
/// poll. Other ready tasks get a turn before this task resumes.
pub fn yield_now() -> crate::timer::YieldNow {
    crate::timer::YieldNow(false)
}

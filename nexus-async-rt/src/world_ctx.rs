//! Lightweight handle for synchronous [`World`](nexus_rt::World) access from
//! async tasks.
//!
//! [`WorldCtx`] wraps a raw pointer to a [`World`](nexus_rt::World). It is
//! [`Copy`] so tasks can capture it cheaply (8 bytes). The scoped
//! [`with_world`](WorldCtx::with_world) API runs a closure synchronously
//! inline during task poll — no await point needed.
//!
//! # Pre-resolved parameters
//!
//! Closures going through [`IntoHandler`](nexus_rt::IntoHandler) resolve
//! [`ResourceId`](nexus_rt::ResourceId)s at build time (one HashMap lookup
//! per type). At dispatch time, each resource access is a single pointer
//! deref. Build the handler before spawning, move it into the task, and
//! call [`Handler::run`](nexus_rt::Handler::run) inside `with_world`.

use nexus_rt::World;

/// [`Copy`] handle for synchronous [`World`] access from async tasks.
///
/// # Safety Contract
///
/// - **Single-threaded only.** No concurrent `with_world` calls.
/// - **World outlives tasks.** The [`World`] must not be dropped while
///   any task holds a `WorldCtx`.
///
/// Both invariants are enforced structurally by the single-threaded
/// executor: only one task polls at a time, and the user owns the
/// [`World`] alongside the executor in the same scope.
///
/// # Examples
///
/// ```ignore
/// use nexus_async_rt::{Executor, WorldCtx};
/// use nexus_rt::{WorldBuilder, Res, ResMut, IntoHandler, Handler};
///
/// let mut world = builder.build();
/// let ctx = WorldCtx::new(&mut world);
///
/// // Pre-resolve at setup — single HashMap lookup per type
/// let mut on_quote = (|mut books: ResMut<Books>, q: Quote| {
///     books.update(q);
/// }).into_handler(world.registry());
///
/// let mut executor = Executor::new(64);
/// executor.spawn_boxed(async move {
///     let data = read_socket().await;
///     // Single deref per resource at dispatch time
///     ctx.with_world(|world| on_quote.run(world, data));
/// });
///
/// while executor.task_count() > 0 { executor.poll(); }
/// ```
#[derive(Clone, Copy)]
pub struct WorldCtx {
    ptr: *mut World,
}

impl WorldCtx {
    /// Create a context handle from a mutable [`World`] reference.
    ///
    /// # Safety Contract (enforced by caller, not by the type system)
    ///
    /// - The [`World`] must outlive all tasks using this handle.
    /// - The caller must not use `&mut World` directly while tasks hold
    ///   a `WorldCtx` — all World access must go through `with_world`.
    /// - Single-threaded use only (no concurrent `with_world` calls).
    ///
    /// These invariants are structurally enforced by [`crate::Runtime`]:
    /// the World is created before the runtime, `block_on` takes
    /// `&mut self` preventing direct World access during execution,
    /// and the single-threaded executor prevents concurrent polls.
    pub fn new(world: &mut World) -> Self {
        Self {
            ptr: std::ptr::from_mut(world),
        }
    }

    /// Run a closure with exclusive [`World`] access.
    ///
    /// Executes synchronously inline — no await point. The closure
    /// has `&mut World` access for its duration.
    /// Returns the raw world pointer. Used by context module.
    pub(crate) fn as_ptr(&self) -> *mut World {
        self.ptr
    }

    /// Run a closure with exclusive [`World`] access.
    ///
    /// Executes synchronously inline — no await point. The closure has
    /// `&mut World` for its duration. Use when you need to mutate
    /// resources; see [`with_world_ref`](Self::with_world_ref) for
    /// read-only access.
    pub fn with_world<R>(&self, f: impl FnOnce(&mut World) -> R) -> R {
        // SAFETY: Single-threaded executor guarantees only one task polls
        // at a time, so only one with_world is active. World outlives all
        // tasks (caller invariant from WorldCtx::new).
        let world = unsafe { &mut *self.ptr };
        f(world)
    }

    /// Run a closure with shared [`World`] access.
    ///
    /// Use when you only need to read resources.
    pub fn with_world_ref<R>(&self, f: impl FnOnce(&World) -> R) -> R {
        // SAFETY: Same invariants as with_world. Shared ref is strictly
        // less powerful.
        let world = unsafe { &*self.ptr };
        f(world)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Executor;
    use nexus_rt::{Handler, IntoHandler, Res, ResMut, WorldBuilder};

    nexus_rt::new_resource!(Val(u64));
    nexus_rt::new_resource!(Out(u64));

    #[test]
    fn with_world_raw_access() {
        let mut wb = WorldBuilder::new();
        wb.register(Val(42));
        wb.register(Out(0));
        let mut world = wb.build();
        let ctx = WorldCtx::new(&mut world);

        let mut executor = Executor::new(4);
        executor.spawn_boxed(async move {
            ctx.with_world(|world| {
                let v = world.resource::<Val>().0;
                world.resource_mut::<Out>().0 = v + 10;
            });
        });

        while executor.task_count() > 0 {
            executor.poll();
        }
        assert_eq!(world.resource::<Out>().0, 52);
    }

    #[test]
    fn with_world_ref_read_only() {
        let mut wb = WorldBuilder::new();
        wb.register(Val(99));
        let mut world = wb.build();
        let ctx = WorldCtx::new(&mut world);

        let result = std::cell::Cell::new(0u64);
        let result_ptr = std::ptr::from_ref(&result);

        let mut executor = Executor::new(4);
        executor.spawn_boxed(async move {
            let v = ctx.with_world_ref(|world| world.resource::<Val>().0);
            // SAFETY: test-only, single-threaded, Cell is alive.
            unsafe { &*result_ptr }.set(v);
        });

        while executor.task_count() > 0 {
            executor.poll();
        }
        assert_eq!(result.get(), 99);
    }

    #[test]
    fn with_world_pre_resolved_handler() {
        let mut wb = WorldBuilder::new();
        wb.register(Val(42));
        wb.register(Out(0));
        let mut world = wb.build();
        let ctx = WorldCtx::new(&mut world);

        // Pre-resolve: HashMap lookups happen here, once
        let mut handler = (|val: Res<Val>, mut out: ResMut<Out>, event: u64| {
            out.0 = val.0 + event;
        })
        .into_handler(world.registry());

        let mut executor = Executor::new(4);
        executor.spawn_boxed(async move {
            ctx.with_world(|world| handler.run(world, 10));
        });

        while executor.task_count() > 0 {
            executor.poll();
        }
        assert_eq!(world.resource::<Out>().0, 52);
    }

    #[test]
    fn with_world_returns_value() {
        let mut wb = WorldBuilder::new();
        wb.register(Val(7));
        let mut world = wb.build();
        let ctx = WorldCtx::new(&mut world);

        let result = std::cell::Cell::new(0u64);
        let result_ptr = std::ptr::from_ref(&result);

        let mut executor = Executor::new(4);
        executor.spawn_boxed(async move {
            let v = ctx.with_world(|world| world.resource::<Val>().0 * 6);
            // SAFETY: test-only, single-threaded, Cell is alive.
            unsafe { &*result_ptr }.set(v);
        });

        while executor.task_count() > 0 {
            executor.poll();
        }
        assert_eq!(result.get(), 42);
    }

    #[test]
    fn multiple_tasks_share_ctx() {
        let mut wb = WorldBuilder::new();
        wb.register(Out(0));
        let mut world = wb.build();
        let ctx = WorldCtx::new(&mut world);

        let mut executor = Executor::new(4);

        for i in 1..=3u64 {
            let ctx = ctx; // Copy
            executor.spawn_boxed(async move {
                ctx.with_world(|world| {
                    world.resource_mut::<Out>().0 += i;
                });
            });
        }

        while executor.task_count() > 0 {
            executor.poll();
        }
        assert_eq!(world.resource::<Out>().0, 6); // 1 + 2 + 3
    }
}

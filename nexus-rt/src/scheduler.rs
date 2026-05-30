//! Staged scheduler with boolean propagation.
//!
//! The scheduler is installed as a **driver** via [`SchedulerBuilder`].
//! After event handlers process incoming data and write to resources,
//! the scheduler runs reconciliation [`System`]s in stage order. This
//! two-phase pattern (event → reconcile) separates reactive logic from
//! derived-state computation.
//!
//! # Staged model
//!
//! Systems are grouped into **stages**. Each stage runs all its systems,
//! then propagates a `bool` to the next stage: "did any system in this
//! stage return `true`?" If no system fired, downstream stages are
//! skipped.
//!
//! Topology is implicit in builder order — each `.then()` call creates
//! the next stage. No explicit edge declarations, no topological sort.
//!
//! # Monomorphization
//!
//! The schedule chain is fully monomorphized. Each `.root()` / `.then()`
//! wraps the accumulator in a `StageNode<Prev, S>` type layer. The
//! compiler inlines the recursive `run_schedule` calls, eliminating all
//! vtable dispatch. Same pattern as [`Pipeline`](crate::Pipeline) and
//! [`Dag`](crate::Dag).
//!
//! # Propagation model
//!
//! Each system returns `bool`. `true` means "my outputs changed, run
//! downstream." `false` means "nothing changed." Within a stage, all
//! systems always run regardless of individual return values — the
//! stage result is the OR of all system returns.
//!
//! # Sequence mechanics
//!
//! The global sequence counter is event-only — the scheduler never
//! calls [`next_sequence`](crate::World::next_sequence).
//!
//! # Invariants
//!
//! - **Stage order**: Stages execute in builder order. All systems
//!   within a stage run before any system in the next stage.
//! - **All systems in a stage run**: Even if one returns `false`,
//!   the rest still execute (side effects matter).
//! - **Deterministic**: Same inputs produce same results. No
//!   randomness, no thread-dependent ordering.
//! - **No sequence bump**: The scheduler never advances the global
//!   sequence. Event handlers own sequencing; the scheduler observes.
//! - **No system limit**: The nested type encodes the full schedule
//!   at compile time. No bitmask, no capacity constant.
//!
//! # Examples
//!
//! ```
//! use nexus_rt::{WorldBuilder, Res, ResMut, Installer, Resource};
//! use nexus_rt::scheduler::SchedulerBuilder;
//!
//! #[derive(Resource)]
//! struct Val(u64);
//!
//! fn step_a(mut val: ResMut<Val>) -> bool {
//!     val.0 += 1;
//!     true
//! }
//!
//! fn step_b(val: Res<Val>) -> bool {
//!     val.0 > 0
//! }
//!
//! let mut builder = WorldBuilder::new();
//! builder.register(Val(0));
//! let reg = builder.registry();
//!
//! let mut scheduler = builder.install_driver(
//!     SchedulerBuilder::new()
//!         .root(step_a, &reg)
//!         .then(step_b, &reg)
//! );
//! let mut world = builder.build();
//!
//! assert_eq!(scheduler.run(&mut world), 2);
//! ```

use crate::driver::Installer;
use crate::system::{IntoSystem, System};
use crate::world::{Registry, World, WorldBuilder};

// =============================================================================
// StageEnd — terminal node
// =============================================================================

#[doc(hidden)]
pub struct StageEnd;

// =============================================================================
// StageNode — one stage in the schedule chain
// =============================================================================

#[doc(hidden)]
pub struct StageNode<Prev, S> {
    prev: Prev,
    stage: S,
}

// =============================================================================
// RunSchedule — chain dispatch trait
// =============================================================================

#[doc(hidden)]
pub trait RunSchedule: Send {
    fn run_schedule(&mut self, world: &mut World) -> (usize, bool);
    fn system_count(&self) -> usize;
}

impl RunSchedule for StageEnd {
    #[inline(always)]
    fn run_schedule(&mut self, _world: &mut World) -> (usize, bool) {
        (0, true)
    }

    fn system_count(&self) -> usize {
        0
    }
}

impl<Prev: RunSchedule, S: StageRunner> RunSchedule for StageNode<Prev, S> {
    #[inline(always)]
    fn run_schedule(&mut self, world: &mut World) -> (usize, bool) {
        let (prev_ran, prev_fired) = self.prev.run_schedule(world);
        if !prev_fired {
            return (prev_ran, false);
        }
        let (stage_ran, stage_fired) = self.stage.run_all(world);
        (prev_ran + stage_ran, stage_fired)
    }

    fn system_count(&self) -> usize {
        self.prev.system_count() + self.stage.system_count()
    }
}

// =============================================================================
// StageRunner — per-stage dispatch trait
// =============================================================================

#[doc(hidden)]
pub trait StageRunner: Send {
    fn run_all(&mut self, world: &mut World) -> (usize, bool);
    fn system_count(&self) -> usize;
}

// =============================================================================
// Stage1..Stage8 — macro-generated stage wrappers
// =============================================================================

macro_rules! impl_stage {
    ($name:ident, $count:expr, $(($idx:tt, $S:ident)),+) => {
        #[doc(hidden)]
        pub struct $name<$($S),+>($(pub(crate) $S),+);

        impl<$($S: System),+> StageRunner for $name<$($S),+> {
            #[inline(always)]
            fn run_all(&mut self, world: &mut World) -> (usize, bool) {
                let mut fired = false;
                $(fired |= self.$idx.run(world);)+
                ($count, fired)
            }

            fn system_count(&self) -> usize {
                $count
            }
        }
    };
}

impl_stage!(Stage1, 1, (0, S0));
impl_stage!(Stage2, 2, (0, S0), (1, S1));
impl_stage!(Stage3, 3, (0, S0), (1, S1), (2, S2));
impl_stage!(Stage4, 4, (0, S0), (1, S1), (2, S2), (3, S3));
impl_stage!(Stage5, 5, (0, S0), (1, S1), (2, S2), (3, S3), (4, S4));
impl_stage!(
    Stage6,
    6,
    (0, S0),
    (1, S1),
    (2, S2),
    (3, S3),
    (4, S4),
    (5, S5)
);
impl_stage!(
    Stage7,
    7,
    (0, S0),
    (1, S1),
    (2, S2),
    (3, S3),
    (4, S4),
    (5, S5),
    (6, S6)
);
impl_stage!(
    Stage8,
    8,
    (0, S0),
    (1, S1),
    (2, S2),
    (3, S3),
    (4, S4),
    (5, S5),
    (6, S6),
    (7, S7)
);

// =============================================================================
// IntoStage — converts functions/tuples into stages
// =============================================================================

/// Converts a single function or tuple of functions into a stage.
///
/// The `Params` type parameter is inference-only — same pattern as
/// [`IntoHandler<E, Params>`](crate::IntoHandler). Never constructed,
/// just guides the compiler to the right impl.
pub trait IntoStage<Params> {
    /// The concrete stage type produced.
    type Stage: StageRunner + 'static;
    /// Convert into a stage, resolving system parameters from the registry.
    fn into_stage(self, registry: &Registry) -> Self::Stage;
}

impl<F, P, M> IntoStage<(P, M)> for F
where
    F: IntoSystem<P, M>,
    F::System: 'static,
{
    type Stage = Stage1<F::System>;

    fn into_stage(self, registry: &Registry) -> Self::Stage {
        Stage1(self.into_system(registry))
    }
}

macro_rules! impl_into_stage {
    ($stage:ident, $(($F:ident, $P:ident, $M:ident, $idx:tt)),+) => {
        impl<$($F, $P, $M),+> IntoStage<($(($P, $M),)+)> for ($($F,)+)
        where
            $($F: IntoSystem<$P, $M>, $F::System: 'static,)+
        {
            type Stage = $stage<$($F::System),+>;

            fn into_stage(self, registry: &Registry) -> Self::Stage {
                $stage($(self.$idx.into_system(registry)),+)
            }
        }
    };
}

impl_into_stage!(Stage2, (F0, P0, M0, 0), (F1, P1, M1, 1));
impl_into_stage!(Stage3, (F0, P0, M0, 0), (F1, P1, M1, 1), (F2, P2, M2, 2));
impl_into_stage!(
    Stage4,
    (F0, P0, M0, 0),
    (F1, P1, M1, 1),
    (F2, P2, M2, 2),
    (F3, P3, M3, 3)
);
impl_into_stage!(
    Stage5,
    (F0, P0, M0, 0),
    (F1, P1, M1, 1),
    (F2, P2, M2, 2),
    (F3, P3, M3, 3),
    (F4, P4, M4, 4)
);
impl_into_stage!(
    Stage6,
    (F0, P0, M0, 0),
    (F1, P1, M1, 1),
    (F2, P2, M2, 2),
    (F3, P3, M3, 3),
    (F4, P4, M4, 4),
    (F5, P5, M5, 5)
);
impl_into_stage!(
    Stage7,
    (F0, P0, M0, 0),
    (F1, P1, M1, 1),
    (F2, P2, M2, 2),
    (F3, P3, M3, 3),
    (F4, P4, M4, 4),
    (F5, P5, M5, 5),
    (F6, P6, M6, 6)
);
impl_into_stage!(
    Stage8,
    (F0, P0, M0, 0),
    (F1, P1, M1, 1),
    (F2, P2, M2, 2),
    (F3, P3, M3, 3),
    (F4, P4, M4, 4),
    (F5, P5, M5, 5),
    (F6, P6, M6, 6),
    (F7, P7, M7, 7)
);

// =============================================================================
// SchedulerBuilder — entry point
// =============================================================================

/// Entry point for building a monomorphized schedule.
///
/// # Examples
///
/// ```
/// use nexus_rt::{WorldBuilder, ResMut, Resource};
/// use nexus_rt::scheduler::SchedulerBuilder;
///
/// #[derive(Resource)]
/// struct Val(u64);
///
/// fn step_a(mut val: ResMut<Val>) -> bool {
///     val.0 += 1;
///     true
/// }
///
/// fn step_b(val: nexus_rt::Res<Val>) -> bool {
///     val.0 > 0
/// }
///
/// let mut builder = WorldBuilder::new();
/// builder.register(Val(0));
/// let reg = builder.registry();
///
/// let mut scheduler = builder.install_driver(
///     SchedulerBuilder::new()
///         .root(step_a, &reg)
///         .then(step_b, &reg)
/// );
/// let mut world = builder.build();
/// assert_eq!(scheduler.run(&mut world), 2);
/// ```
pub struct SchedulerBuilder;

impl SchedulerBuilder {
    /// Create a new scheduler builder.
    pub fn new() -> Self {
        Self
    }

    /// Create the first stage from a system or tuple of systems.
    pub fn root<S, Params>(self, stage: S, registry: &Registry) -> StageNode<StageEnd, S::Stage>
    where
        S: IntoStage<Params>,
    {
        StageNode {
            prev: StageEnd,
            stage: stage.into_stage(registry),
        }
    }
}

impl Default for SchedulerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// StageNode builder methods
// =============================================================================

impl<Prev, S> StageNode<Prev, S>
where
    Prev: RunSchedule + 'static,
    S: StageRunner + 'static,
{
    /// Append a stage that runs after the current chain.
    pub fn then<Next, Params>(
        self,
        stage: Next,
        registry: &Registry,
    ) -> StageNode<Self, Next::Stage>
    where
        Next: IntoStage<Params>,
    {
        StageNode {
            prev: self,
            stage: stage.into_stage(registry),
        }
    }
}

// =============================================================================
// Installer impl on StageNode
// =============================================================================

impl<Prev, S> Installer for StageNode<Prev, S>
where
    Self: RunSchedule + 'static,
{
    type Poller = SystemScheduler<Self>;

    fn install(self, _world: &mut WorldBuilder) -> Self::Poller {
        SystemScheduler { chain: self }
    }
}

// =============================================================================
// SystemScheduler<Chain> — final type
// =============================================================================

/// Monomorphized staged scheduler. Created via [`SchedulerBuilder`].
///
/// All system calls are direct (no vtable dispatch). The `Chain` type
/// parameter encodes the full schedule — the compiler inlines the
/// recursive `run_schedule` chain.
///
/// # Propagation
///
/// The first stage always runs. Each subsequent stage runs only if
/// the previous stage's systems returned at least one `true` (OR
/// semantics within a stage).
///
/// Does NOT call [`next_sequence`](World::next_sequence) — the global
/// sequence is event-only.
pub struct SystemScheduler<Chain> {
    chain: Chain,
}

impl<Chain: RunSchedule> SystemScheduler<Chain> {
    /// Run all stages with boolean propagation.
    ///
    /// Returns the number of systems that actually ran.
    pub fn run(&mut self, world: &mut World) -> usize {
        let (ran, _) = self.chain.run_schedule(world);
        ran
    }

    /// Returns the total number of systems across all stages.
    pub fn len(&self) -> usize {
        self.chain.system_count()
    }

    /// Returns `true` if the scheduler contains no systems.
    pub fn is_empty(&self) -> bool {
        self.chain.system_count() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResMut;

    // -- Helpers ----------------------------------------------------------

    fn increment(mut val: ResMut<u64>) -> bool {
        *val += 1;
        true
    }

    fn set_flag(mut flag: ResMut<bool>) -> bool {
        *flag = true;
        true
    }

    fn false_source() -> bool {
        false
    }

    fn should_not_run(mut val: ResMut<u64>) -> bool {
        *val = 999;
        true
    }

    fn source(mut val: ResMut<u64>) -> bool {
        *val += 1;
        *val <= 2
    }

    fn middle(mut val: ResMut<u64>) -> bool {
        *val += 10;
        true
    }

    fn leaf(mut val: ResMut<u64>) -> bool {
        *val += 100;
        true
    }

    fn double(mut val: ResMut<u64>) -> bool {
        *val *= 2;
        true
    }

    // -- Migrated tests ---------------------------------------------------

    #[test]
    fn single_root_always_runs() {
        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        let reg = builder.registry();
        let mut scheduler = builder.install_driver(SchedulerBuilder::new().root(increment, reg));
        let mut world = builder.build();

        assert_eq!(scheduler.run(&mut world), 1);
        assert_eq!(*world.resource::<u64>(), 1);
    }

    #[test]
    fn linear_chain_propagation() {
        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        let reg = builder.registry();
        let mut scheduler = builder.install_driver(
            SchedulerBuilder::new()
                .root(source, reg)
                .then(middle, reg)
                .then(leaf, reg),
        );
        let mut world = builder.build();

        assert_eq!(scheduler.run(&mut world), 3);
        assert_eq!(*world.resource::<u64>(), 111);
    }

    #[test]
    fn propagation_stops_on_false() {
        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        let reg = builder.registry();
        let mut scheduler = builder.install_driver(
            SchedulerBuilder::new()
                .root(false_source, reg)
                .then(should_not_run, reg),
        );
        let mut world = builder.build();

        assert_eq!(scheduler.run(&mut world), 1);
        assert_eq!(*world.resource::<u64>(), 0);
    }

    #[test]
    fn staged_diamond() {
        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        builder.register::<bool>(false);
        let reg = builder.registry();
        let mut scheduler = builder.install_driver(
            SchedulerBuilder::new()
                .root(increment, reg)
                .then((increment, set_flag), reg)
                .then(increment, reg),
        );
        let mut world = builder.build();

        assert_eq!(scheduler.run(&mut world), 4);
        assert!(*world.resource::<bool>());
        assert_eq!(*world.resource::<u64>(), 3);
    }

    #[test]
    fn multiple_roots() {
        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        let reg = builder.registry();
        let mut scheduler = builder
            .install_driver(SchedulerBuilder::new().root((increment, increment, increment), reg));
        let mut world = builder.build();

        assert_eq!(scheduler.run(&mut world), 3);
        assert_eq!(*world.resource::<u64>(), 3);
    }

    #[test]
    fn scheduler_does_not_bump_sequence() {
        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        let reg = builder.registry();
        let mut scheduler = builder.install_driver(SchedulerBuilder::new().root(increment, reg));
        let mut world = builder.build();

        let before = world.current_sequence();
        scheduler.run(&mut world);
        assert_eq!(world.current_sequence(), before);
    }

    #[test]
    fn mutations_visible_downstream() {
        let mut builder = WorldBuilder::new();
        builder.register::<u64>(1);
        let reg = builder.registry();
        let mut scheduler =
            builder.install_driver(SchedulerBuilder::new().root(double, reg).then(double, reg));
        let mut world = builder.build();

        scheduler.run(&mut world);
        assert_eq!(*world.resource::<u64>(), 4);
    }

    // -- New tests --------------------------------------------------------

    #[test]
    fn multi_system_stage_all_run() {
        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        builder.register::<bool>(false);
        let reg = builder.registry();
        let mut scheduler = builder
            .install_driver(SchedulerBuilder::new().root((increment, increment, set_flag), reg));
        let mut world = builder.build();

        assert_eq!(scheduler.run(&mut world), 3);
        assert_eq!(*world.resource::<u64>(), 2);
        assert!(*world.resource::<bool>());
    }

    #[test]
    fn stage_propagation_any_semantics() {
        fn false_increment(mut val: ResMut<u64>) -> bool {
            *val += 1;
            false
        }

        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        let reg = builder.registry();
        let mut scheduler = builder.install_driver(
            SchedulerBuilder::new()
                .root((false_increment, increment), reg)
                .then(increment, reg),
        );
        let mut world = builder.build();

        assert_eq!(scheduler.run(&mut world), 3);
        assert_eq!(*world.resource::<u64>(), 3);
    }

    #[test]
    fn stage_propagation_all_false_stops() {
        fn false_increment(mut val: ResMut<u64>) -> bool {
            *val += 1;
            false
        }

        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        let reg = builder.registry();
        let mut scheduler = builder.install_driver(
            SchedulerBuilder::new()
                .root((false_increment, false_increment), reg)
                .then(should_not_run, reg),
        );
        let mut world = builder.build();

        assert_eq!(scheduler.run(&mut world), 2);
        assert_eq!(*world.resource::<u64>(), 2);
    }

    #[test]
    fn void_systems_in_stage() {
        fn void_increment(mut val: ResMut<u64>) {
            *val += 1;
        }

        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        let reg = builder.registry();
        let mut scheduler = builder.install_driver(
            SchedulerBuilder::new()
                .root(void_increment, reg)
                .then(increment, reg),
        );
        let mut world = builder.build();

        assert_eq!(scheduler.run(&mut world), 2);
        assert_eq!(*world.resource::<u64>(), 2);
    }

    #[test]
    fn mixed_bool_void_stage() {
        fn void_increment(mut val: ResMut<u64>) {
            *val += 1;
        }

        fn false_increment(mut val: ResMut<u64>) -> bool {
            *val += 1;
            false
        }

        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        let reg = builder.registry();
        let mut scheduler = builder.install_driver(
            SchedulerBuilder::new()
                .root((false_increment, void_increment), reg)
                .then(increment, reg),
        );
        let mut world = builder.build();

        assert_eq!(scheduler.run(&mut world), 3);
        assert_eq!(*world.resource::<u64>(), 3);
    }

    #[test]
    fn scheduler_chain_is_send() {
        fn assert_send<T: Send>(_: &T) {}

        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        builder.register::<bool>(false);
        let reg = builder.registry();
        let scheduler = builder.install_driver(
            SchedulerBuilder::new()
                .root(increment, reg)
                .then((increment, set_flag), reg)
                .then(double, reg),
        );
        assert_send(&scheduler);
    }

    #[test]
    fn three_stage_linear() {
        let mut builder = WorldBuilder::new();
        builder.register::<u64>(0);
        builder.register::<bool>(false);
        let reg = builder.registry();
        let mut scheduler = builder.install_driver(
            SchedulerBuilder::new()
                .root(increment, reg)
                .then((increment, set_flag), reg)
                .then(double, reg),
        );
        let mut world = builder.build();

        assert_eq!(scheduler.run(&mut world), 4);
        assert!(*world.resource::<bool>());
        // u64: 0 +1 (root) +1 (stage1) = 2, then *2 (stage2) = 4
        assert_eq!(*world.resource::<u64>(), 4);
    }
}

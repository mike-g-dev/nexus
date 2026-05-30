#![allow(
    unused_must_use,
    dead_code,
    clippy::float_cmp,
    clippy::used_underscore_binding,
    clippy::items_after_statements
)]
//! Miri tests for World/ResourceId unsafe paths.
//!
//! Exercises type-erased resource storage via NonNull<u8>, Box reconstitution
//! on drop, and ResourceCell change detection — the only unsafe code in
//! nexus-rt.
//!
//! Run: `cargo +nightly miri test -p nexus-rt --test miri_tests`

use std::cell::Cell;

use nexus_rt::{Resource, WorldBuilder};

// =============================================================================
// Helper types
// =============================================================================

thread_local! {
    static DROP_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[derive(Resource)]
struct Counter(u64);

#[derive(Resource)]
struct Label(String);

#[derive(Resource)]
struct DropTracker(#[allow(dead_code)] u64);

impl Drop for DropTracker {
    fn drop(&mut self) {
        DROP_COUNT.with(|c| c.set(c.get() + 1));
    }
}

fn reset_drop_count() {
    DROP_COUNT.with(|c| c.set(0));
}

fn get_drop_count() -> usize {
    DROP_COUNT.with(Cell::get)
}

// =============================================================================
// Resource insert / get / get_mut cycle
// =============================================================================

#[test]
fn world_resource_insert_get_roundtrip() {
    let mut wb = WorldBuilder::new();
    wb.register(Counter(42));
    wb.register(Label("hello".into()));
    let world = wb.build();

    assert_eq!(world.resource::<Counter>().0, 42);
    assert_eq!(world.resource::<Label>().0, "hello");
}

#[test]
fn world_resource_mut() {
    let mut wb = WorldBuilder::new();
    wb.register(Counter(0));
    let mut world = wb.build();

    world.resource_mut::<Counter>().0 = 99;
    assert_eq!(world.resource::<Counter>().0, 99);
}

#[test]
fn world_multiple_resources_coexist() {
    let mut wb = WorldBuilder::new();
    wb.register(Counter(1));
    wb.register(Label("a".into()));
    let mut world = wb.build();

    // Mutate one, read both — verifies separate ResourceId pointers
    world.resource_mut::<Counter>().0 += 10;
    assert_eq!(world.resource::<Counter>().0, 11);
    assert_eq!(world.resource::<Label>().0, "a");
}

// =============================================================================
// Drop ordering when World is dropped
// =============================================================================

#[derive(Resource)]
struct DT1(#[allow(dead_code)] u64);
impl Drop for DT1 {
    fn drop(&mut self) {
        DROP_COUNT.with(|c| c.set(c.get() + 1));
    }
}
#[derive(Resource)]
struct DT2(#[allow(dead_code)] u64);
impl Drop for DT2 {
    fn drop(&mut self) {
        DROP_COUNT.with(|c| c.set(c.get() + 1));
    }
}
#[derive(Resource)]
struct DT3(#[allow(dead_code)] u64);
impl Drop for DT3 {
    fn drop(&mut self) {
        DROP_COUNT.with(|c| c.set(c.get() + 1));
    }
}

#[test]
fn world_drop_drops_resources() {
    reset_drop_count();

    {
        let mut wb = WorldBuilder::new();
        wb.register(DT1(1));
        wb.register(DT2(2));
        wb.register(DT3(3));
        let _world = wb.build();
        assert_eq!(get_drop_count(), 0);
    }
    // World dropped — all 3 drop-tracked resources should be dropped
    assert_eq!(get_drop_count(), 3);
}

#[test]
fn world_drop_drops_heap_resources() {
    // String has a heap allocation — verify no leak
    let mut wb = WorldBuilder::new();
    wb.register(Label("heap allocated string".into()));
    let world = wb.build();
    assert_eq!(world.resource::<Label>().0, "heap allocated string");
    drop(world);
    // Miri checks for leaks
}

// =============================================================================
// Resource replacement
// =============================================================================

/// Register a resource, build world, drop and rebuild — verifies
/// the full lifecycle through Box reconstitution.
#[test]
fn world_rebuild_after_drop() {
    let mut wb = WorldBuilder::new();
    wb.register(Counter(10));
    let world = wb.build();
    assert_eq!(world.resource::<Counter>().0, 10);
    drop(world);

    // Rebuild fresh — old allocation freed, new one created
    let mut wb2 = WorldBuilder::new();
    wb2.register(Counter(20));
    let world2 = wb2.build();
    assert_eq!(world2.resource::<Counter>().0, 20);
}

// =============================================================================
// Change detection (ResourceCell tick)
// =============================================================================

#[test]
fn world_change_detection() {
    let mut wb = WorldBuilder::new();
    wb.register(Counter(0));
    let mut world = wb.build();

    // Initial state — resource was just registered
    let changed_before = world.resource::<Counter>().0;
    assert_eq!(changed_before, 0);

    // Mutate via resource_mut — stamps the ResourceCell
    world.resource_mut::<Counter>().0 = 42;
    assert_eq!(world.resource::<Counter>().0, 42);
}

// =============================================================================
// Many resources — exercises the HashMap<TypeId, ResourceId> path
// =============================================================================

#[derive(Resource)]
struct R0(u64);
#[derive(Resource)]
struct R1(u64);
#[derive(Resource)]
struct R2(u64);
#[derive(Resource)]
struct R3(u64);
#[derive(Resource)]
struct R4(u64);
#[derive(Resource)]
struct R5(u64);
#[derive(Resource)]
struct R6(u64);
#[derive(Resource)]
struct R7(u64);

#[test]
fn world_many_resources() {
    let mut wb = WorldBuilder::new();
    wb.register(R0(0));
    wb.register(R1(1));
    wb.register(R2(2));
    wb.register(R3(3));
    wb.register(R4(4));
    wb.register(R5(5));
    wb.register(R6(6));
    wb.register(R7(7));
    let world = wb.build();

    assert_eq!(world.resource::<R0>().0, 0);
    assert_eq!(world.resource::<R7>().0, 7);
    assert_eq!(world.resource::<R3>().0, 3);
}

// =============================================================================
// Stress — alloc/read/mutate/drop cycle
// =============================================================================

#[test]
fn world_stress_register_mutate_drop() {
    reset_drop_count();

    for _ in 0..10 {
        let mut wb = WorldBuilder::new();
        wb.register(Counter(0));
        wb.register(Label("stress".into()));
        wb.register(DT1(99));
        let mut world = wb.build();

        for i in 0..5u64 {
            world.resource_mut::<Counter>().0 += i;
        }
        assert_eq!(world.resource::<Counter>().0, 10); // 0+1+2+3+4
    }

    assert_eq!(get_drop_count(), 10); // 10 DT1 instances
}

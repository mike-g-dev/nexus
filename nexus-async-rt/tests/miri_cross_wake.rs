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
//! Integration tests for cross-thread wake paths.
//!
//! Tests the cross_task_wake fix (#5) through the tokio_compat public API.
//! Uses real threads (tokio worker) + the nexus-async-rt executor.
//!
//! Run: `cargo test -p nexus-async-rt --test miri_cross_wake`
//!
//! Note: These tests use tokio and real threads — they are NOT miri-compatible.
//! The miri coverage for the cross-wake queue is in the source file's unit tests.

#![cfg(feature = "tokio-compat")]

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use nexus_async_rt::tokio_compat::{spawn_on_tokio, with_tokio};
use nexus_async_rt::{Runtime, spawn_boxed};
use nexus_rt::WorldBuilder;

// =============================================================================
// 1. spawn_on_tokio completes and delivers result
// =============================================================================

#[test]
fn cross_wake_spawn_on_tokio_completes() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let result = spawn_on_tokio(async { 42u64 }).await.unwrap();
        assert_eq!(result, 42);
    });
}

// =============================================================================
// 2. with_tokio wrapping tokio::time::sleep completes
// =============================================================================

#[test]
fn cross_wake_with_tokio_completes() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        with_tokio(|| tokio::time::sleep(Duration::from_millis(10))).await;
        flag.set(true);
    });

    assert!(done.get());
}

// =============================================================================
// 3. Multiple concurrent spawn_on_tokio tasks all complete
// =============================================================================

#[test]
fn cross_wake_multiple_tokio_spawns() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let mut handles = Vec::new();
        for i in 0u64..10 {
            handles.push(spawn_on_tokio(async move { i * 2 }));
        }

        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.unwrap());
        }

        results.sort_unstable();
        let expected: Vec<u64> = (0..10).map(|i| i * 2).collect();
        assert_eq!(results, expected);
    });
}

// =============================================================================
// 4. Dropping TokioJoinHandle before completion aborts without crash
// =============================================================================

#[test]
fn cross_wake_drop_handle_before_completion() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let completed = Arc::new(AtomicBool::new(false));
    let flag = completed.clone();

    rt.block_on(async move {
        {
            let _handle = spawn_on_tokio(async move {
                tokio::time::sleep(Duration::from_secs(60)).await;
                flag.store(true, Ordering::Relaxed);
            });
            // _handle drops here — task aborted
        }

        // Give tokio a moment to process the abort.
        with_tokio(|| tokio::time::sleep(Duration::from_millis(50))).await;
    });

    assert!(
        !completed.load(Ordering::Relaxed),
        "task should have been aborted on handle drop"
    );
}

// =============================================================================
// 5. Concurrent wake and poll — many tasks complete, task_count goes to 0
// =============================================================================

#[test]
fn cross_wake_concurrent_wake_and_poll() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let count = Rc::new(Cell::new(0u32));
    let count_ref = count.clone();

    rt.block_on(async move {
        // Spawn 20 tasks that complete at staggered times.
        for i in 0u32..20 {
            let c = count_ref.clone();
            let _ = spawn_boxed(async move {
                // Each task does a tokio sleep of varying duration.
                let delay = Duration::from_millis(1 + u64::from(i) * 2);
                with_tokio(|| tokio::time::sleep(delay)).await;
                c.set(c.get() + 1);
            });
        }

        // Poll until all 20 complete.
        for _ in 0..500 {
            nexus_async_rt::yield_now().await;
            if count.get() >= 20 {
                break;
            }
            with_tokio(|| tokio::time::sleep(Duration::from_millis(5))).await;
        }

        assert_eq!(count.get(), 20, "not all tasks completed");
    });

    assert_eq!(
        rt.task_count(),
        0,
        "task_count should be 0 after all tasks complete"
    );
}

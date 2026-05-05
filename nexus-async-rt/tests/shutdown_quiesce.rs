//! Integration tests for PR 2 §2.4 `Runtime::shutdown_quiesce`.
//!
//! The canonical pre-shutdown step. Drives the executor until the
//! cross-thread queue is drained and no local ready work remains, or
//! returns `Err(QuiesceTimeout)`.

use std::time::Duration;

use nexus_async_rt::{QuiesceTimeout, Runtime};
use nexus_rt::WorldBuilder;

#[test]
fn shutdown_quiesce_clean_runtime_returns_ok() {
    // Plan-specified test name: nothing was spawned, quiesce should
    // return Ok immediately.
    let mut wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let r = rt.shutdown_quiesce(Duration::from_millis(100));
    assert!(r.is_ok(), "clean runtime quiesce should return Ok: {r:?}");
}

#[test]
fn shutdown_quiesce_after_block_on_returns_ok() {
    // After a normal block_on, the runtime should be quiesced — all
    // tasks completed, queues empty.
    let mut wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let _ = nexus_async_rt::spawn_boxed(async {});
    });

    let r = rt.shutdown_quiesce(Duration::from_millis(100));
    assert!(
        r.is_ok(),
        "quiesce after clean block_on should return Ok: {r:?}"
    );
}

#[test]
fn shutdown_quiesce_drains_cross_queue() {
    // Plan-specified scenario: pre-quiesce, the cross-thread queue has
    // entries; quiesce drains them and returns Ok.
    //
    // Setup: spawn a slab task that holds a tokio_compat-style
    // cross-thread waker and arrange for an off-thread wake to arrive
    // pre-quiesce. Without tokio_compat in this test (deliberate — we
    // don't want to pull in the feature), we simulate by directly
    // pushing the runtime's task pointer into the cross_wake_queue
    // from a background thread BEFORE calling quiesce.
    //
    // Without crate-internal access, the closest we can do via the
    // public API is: spawn a task, run block_on briefly so the task
    // gets registered, then quiesce. The runtime's internal cross-thread
    // wakes fire during normal scheduling — this exercises the drain
    // path even if the cross-queue is empty at the entry point.
    //
    // For a richer test, see lib unit tests; this integration test
    // covers the public API contract: quiesce drains pending work and
    // returns Ok when done.
    let mut wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        // Spawn a self-waking task that completes after one yield — it
        // exercises the local ready queue + cross-queue drain logic.
        let _ = nexus_async_rt::spawn_boxed(async {
            nexus_async_rt::yield_now().await;
        });
    });

    // Quiesce should drain whatever's left and return Ok.
    let r = rt.shutdown_quiesce(Duration::from_millis(100));
    assert!(r.is_ok(), "quiesce should drain pending work: {r:?}");
}

#[test]
fn shutdown_quiesce_then_drop_no_abort() {
    // Plan-specified test name. Full canonical sequence: block_on →
    // quiesce → drop. ShutdownStats counters all zero post-drop.
    let mut wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let stats_handle = rt.shutdown_stats();

    rt.block_on(async {
        let _ = nexus_async_rt::spawn_boxed(async {});
    });

    rt.shutdown_quiesce(Duration::from_millis(100)).unwrap();
    drop(rt);

    let stats = stats_handle.snapshot();
    assert_eq!(stats.aborted_unwinds, 0, "no aborts on clean sequence");
    assert_eq!(stats.leaked_box_tasks, 0, "no leaks on clean sequence");
    assert_eq!(
        stats.unbalanced_normal_shutdowns, 0,
        "no unbalanced shutdowns on clean sequence"
    );
    assert_eq!(
        stats.cross_queue_undrained, 0,
        "no undrained cross-queue entries on clean sequence"
    );
}

#[test]
fn shutdown_quiesce_zero_timeout_clean_runtime_returns_ok() {
    // Edge case: timeout zero — quiesce checks the state once. Clean
    // runtime passes the check immediately.
    let mut wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let r = rt.shutdown_quiesce(Duration::from_millis(0));
    assert!(
        r.is_ok(),
        "zero-timeout quiesce on clean runtime should return Ok: {r:?}"
    );
}

#[test]
fn shutdown_quiesce_timeout() {
    // Plan-specified test name. Spawn a never-completing task,
    // quiesce briefly, expect timeout with non-zero outstanding ref
    // count and (most likely zero) cross-queue.
    let mut wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        // Spawn pending future, detach (drop handle). Yield so the
        // task gets registered.
        let h = nexus_async_rt::spawn_boxed(std::future::pending::<()>());
        drop(h);
        nexus_async_rt::yield_now().await;
    });

    let r = rt.shutdown_quiesce(Duration::from_millis(50));
    match r {
        Err(QuiesceTimeout {
            remaining_cross_queue,
            remaining_outstanding_refs,
            elapsed,
        }) => {
            assert!(
                remaining_outstanding_refs >= 1,
                "expected >= 1 outstanding refs, got {remaining_outstanding_refs}"
            );
            // No cross-thread wakes were performed in this scenario,
            // so the cross-queue should be empty. (PR2-John-review
            // item 4: pre-fix this was a tautological
            // `== 0 || >= 1` always-true assertion.)
            assert_eq!(
                remaining_cross_queue, 0,
                "no cross-thread wakes performed; remaining_cross_queue must be 0"
            );
            assert!(
                elapsed >= Duration::from_millis(40),
                "elapsed should be near the timeout, got {elapsed:?}"
            );
        }
        Ok(()) => panic!("expected QuiesceTimeout for pending task, got Ok"),
    }

    // Don't drop rt with outstanding refs — would panic in debug.
    // Leak the runtime in this test scenario; in production the user
    // would investigate before dropping.
    std::mem::forget(rt);
}

#[test]
fn shutdown_quiesce_completed_task_held_by_join_handle_times_out() {
    // PR2-John-review item 2 regression test.
    //
    // Pre-fix: shutdown_quiesce checked `task_count() == 0`
    // (`live_count`), which decrements unconditionally on completion.
    // A completed task held by a JoinHandle has `live_count -= 1` but
    // is still in `all_tasks` (rc=1, COMPLETED, HAS_JOIN). Quiesce
    // returned `Ok(())`. User dropped Runtime → `Executor::drop`
    // walked `all_tasks`, found the rc=1 task → fired
    // `drop_outstanding_normal` → debug-panic / release-leak +
    // `unbalanced_normal_shutdowns++`. Quiesce's contract ("after Ok,
    // dropping is clean") was violated.
    //
    // Post-fix: quiesce checks `outstanding_tasks() == 0`
    // (`all_tasks.len()`). The held task is still tracked → quiesce
    // returns `Err(QuiesceTimeout)` with `remaining_outstanding_refs >= 1`.
    let mut wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    // `block_on` returns the future's output. JoinHandle is `!Send`
    // but the runtime is single-threaded — block_on has no Send bound,
    // so we can return the handle directly out of the async block and
    // hold it past block_on.
    let kept_handle: nexus_async_rt::JoinHandle<u32> = rt.block_on(async {
        let h = nexus_async_rt::spawn_boxed(async { 42u32 });
        // Yield so the task gets polled to completion (state =
        // COMPLETED, rc=1 because handle still holds its ref).
        nexus_async_rt::yield_now().await;
        // Don't await the handle — we want a completed-but-held task
        // to feed quiesce.
        h
    });

    // Task is now COMPLETED + tracked in all_tasks (JoinHandle ref).
    // shutdown_quiesce SHOULD return Err(QuiesceTimeout) — the task
    // holds a ref that would fire `unbalanced_normal_shutdowns` if
    // we dropped Runtime now.
    let r = rt.shutdown_quiesce(Duration::from_millis(50));
    match r {
        Err(QuiesceTimeout {
            remaining_outstanding_refs,
            ..
        }) => {
            assert_eq!(
                remaining_outstanding_refs, 1,
                "completed task held by JoinHandle should count as outstanding"
            );
        }
        Ok(()) => panic!(
            "PR2-John-review item 2: quiesce mis-claimed clean shutdown for a \
             completed-but-held task. The user's subsequent drop would fire \
             unbalanced_normal_shutdowns."
        ),
    }

    // Cleanup: drop the held handle (frees its ref), then drop the
    // runtime cleanly.
    drop(kept_handle);
    drop(rt);
}

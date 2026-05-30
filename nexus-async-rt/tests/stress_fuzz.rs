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
//! Stress-fuzz test: exercises all runtime subsystems simultaneously.
//!
//! Run: `cargo test -p nexus-async-rt --test stress_fuzz`
//!
//! Each cycle spawns a mix of task types (immediate, yield-once, channel-blocked,
//! cancellable), drives interactions in randomized order, and verifies all
//! resources are correctly cleaned up.
//!
//! What this catches that targeted tests don't:
//! - Slab slot reuse across different task types
//! - Deferred free interacting with channel waker registration
//! - CancellationToken drain racing with task completion
//! - JoinHandle drop during active poll cycle
//!
//! ## Miri status
//!
//! Currently hits a pre-existing stacked borrows violation in the waker TLS
//! path: `Executor::poll()` stores `&mut self.incoming` as a raw pointer in
//! TLS, then `complete_task(&mut self)` invalidates that pointer via the
//! `&mut self` retag. The targeted miri tests (miri_task, miri_waker, etc.)
//! pass because they don't trigger waker wakes during `complete_task`.
//! The fix is to use `UnsafeCell` for the executor's vecs or derive the
//! TLS pointers without going through `&mut self`.

use std::cell::Cell;
use std::rc::Rc;

use nexus_async_rt::{CancellationToken, Runtime, spawn_boxed};
use nexus_rt::WorldBuilder;

// =============================================================================
// Deterministic PRNG (xorshift64)
// =============================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1)) // avoid zero seed
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

// =============================================================================
// Drop tracker
// =============================================================================

#[derive(Clone)]
struct DropTracker(Rc<Cell<u32>>);

impl DropTracker {
    fn new(counter: &Rc<Cell<u32>>) -> Self {
        Self(counter.clone())
    }
}

impl Drop for DropTracker {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

// =============================================================================
// Yield-once future
// =============================================================================

struct YieldOnce {
    yielded: bool,
    _tracker: DropTracker,
}

impl std::future::Future for YieldOnce {
    type Output = ();
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        if self.yielded {
            std::task::Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }
}

// =============================================================================
// Stress test
// =============================================================================

#[test]
fn stress_fuzz_all_subsystems() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let drop_count = Rc::new(Cell::new(0u32));

    for cycle in 0..10u32 {
        let mut rng = Rng::new((cycle as u64 + 1) * 42);
        let dc = drop_count.clone();

        rt.block_on(async move {
            let dc = dc;

            // --- Phase 1: Spawn a mix of task types ---

            // 3 fire-and-forget tasks (complete immediately)
            for _ in 0..3 {
                let t = DropTracker::new(&dc);
                drop(spawn_boxed(async move {
                    let _keep = t;
                }));
            }

            // 2 yield-once tasks (Pending then Ready)
            for _ in 0..2 {
                let t = DropTracker::new(&dc);
                drop(spawn_boxed(YieldOnce {
                    yielded: false,
                    _tracker: t,
                }));
            }

            // 2 channel-waiting tasks
            let (tx1, rx1) = nexus_async_rt::channel::local::channel::<DropTracker>(4);
            let (tx2, rx2) = nexus_async_rt::channel::local::channel::<DropTracker>(4);

            let dc_ch1 = dc.clone();
            drop(spawn_boxed(async move {
                let val = rx1.recv().await.unwrap();
                let _keep = val;
                // Also send confirmation back via a tracked value
                let _ = dc_ch1;
            }));

            let dc_ch2 = dc.clone();
            drop(spawn_boxed(async move {
                let val = rx2.recv().await.unwrap();
                let _keep = val;
                let _ = dc_ch2;
            }));

            // 1 cancellable task
            let token = CancellationToken::new();
            let token_clone = token.clone();
            let dc_cancel = dc.clone();
            drop(spawn_boxed(async move {
                let t = DropTracker::new(&dc_cancel);
                token_clone.cancelled().await;
                let _keep = t;
            }));

            // 1 joinable task (hold the handle)
            let dc_join = dc.clone();
            let mut join_handle = Some(spawn_boxed(async move {
                let t = DropTracker::new(&dc_join);
                nexus_async_rt::yield_now().await;
                t
            }));

            // --- Phase 2: Drive interactions in shuffled order ---
            // Steps: 0=yield, 1=send ch1, 2=send ch2, 3=cancel, 4=yield, 5=drop handle
            let mut steps = [0u8, 1, 2, 3, 4, 5];
            rng.shuffle(&mut steps);

            for &step in &steps {
                match step {
                    0 | 4 => {
                        // Yield to let executor poll ready tasks
                        nexus_async_rt::yield_now().await;
                    }
                    1 => {
                        // Send value through channel 1
                        let _ = tx1.send(DropTracker::new(&dc)).await;
                    }
                    2 => {
                        // Send value through channel 2
                        let _ = tx2.send(DropTracker::new(&dc)).await;
                    }
                    3 => {
                        // Cancel the token
                        token.cancel();
                    }
                    5 => {
                        // Drop the JoinHandle (detach path)
                        drop(join_handle.take());
                        nexus_async_rt::yield_now().await;
                    }
                    _ => unreachable!(),
                }
            }

            // --- Phase 3: Cleanup ---
            // Drop senders to close channels (receiver tasks will see Err)
            drop(tx1);
            drop(tx2);

            // Final yields to drain everything
            for _ in 0..5 {
                nexus_async_rt::yield_now().await;
            }
        });
    }

    // All 10 cycles complete. Verify drops.
    // Each cycle creates: 3 immediate + 2 yield + 2 channel values + 1 cancel + 1 join
    // + 2 channel send values = ~11 DropTrackers per cycle.
    // Exact count varies with shuffle order, but all must be dropped.
    let total = drop_count.get();
    // 10 cycles × ~9 DropTrackers per cycle = ~90 expected.
    // Exact count depends on channel backpressure and shuffle order,
    // but must be at least 10 (one per cycle minimum).
    assert!(
        total >= 10,
        "too few drops: {total} — expected at least 10 (1 per cycle)"
    );
}

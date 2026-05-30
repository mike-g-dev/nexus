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
//! Channel deadlock tests.
//!
//! Each test uses a timeout — if it doesn't complete within the limit,
//! it's a deadlock. Uses the real runtime with block_on_busy (spin mode)
//! so cross-thread wakes don't depend on eventfd.
//!
//! Run: `cargo test -p nexus-async-rt --test channel_deadlock`

use std::cell::Cell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use nexus_async_rt::{Runtime, spawn_boxed};
use nexus_rt::WorldBuilder;

fn runtime() -> (nexus_rt::World, Runtime) {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let rt = Runtime::new(&mut world);
    (world, rt)
}

/// Run a future with a timeout. Panics if it doesn't complete.
fn must_complete_within(
    rt: &mut Runtime,
    timeout: Duration,
    f: impl std::future::Future<Output = ()> + 'static,
) {
    let deadline = Instant::now() + timeout;
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on_busy(async move {
        spawn_boxed(async move {
            f.await;
            flag.set(true);
        });

        // Spin until done or deadline.
        loop {
            if done.get() {
                return;
            }
            assert!(
                Instant::now() <= deadline,
                "DEADLOCK: test did not complete within {timeout:?}"
            );
            nexus_async_rt::yield_now().await;
        }
    });
}

// =============================================================================
// Local channel deadlock tests
// =============================================================================

#[test]
fn local_backpressure_ping_pong() {
    let (mut _world, mut rt) = runtime();
    must_complete_within(&mut rt, Duration::from_secs(2), async {
        let (tx, rx) = nexus_async_rt::channel::local::channel::<u32>(2);

        spawn_boxed(async move {
            for i in 0..10_000 {
                tx.send(i).await.unwrap();
            }
        });

        for i in 0..10_000 {
            let val = rx.recv().await.unwrap();
            assert_eq!(val, i);
        }
    });
}

// =============================================================================
// MPSC typed channel deadlock tests
// =============================================================================

#[test]
fn mpsc_backpressure_single_sender() {
    let (mut _world, mut rt) = runtime();
    must_complete_within(&mut rt, Duration::from_secs(2), async {
        let (tx, rx) = nexus_async_rt::channel::mpsc::channel::<u32>(2);

        spawn_boxed(async move {
            for i in 0..10_000 {
                tx.send(i).await.unwrap();
            }
        });

        for i in 0..10_000 {
            let val = rx.recv().await.unwrap();
            assert_eq!(val, i);
        }
    });
}

#[test]
fn mpsc_backpressure_multiple_senders() {
    let (mut _world, mut rt) = runtime();
    must_complete_within(&mut rt, Duration::from_secs(2), async {
        let (tx, rx) = nexus_async_rt::channel::mpsc::channel::<u32>(4);

        for producer_id in 0u32..4 {
            let tx = tx.clone();
            spawn_boxed(async move {
                for i in 0..1000 {
                    tx.send(producer_id * 10_000 + i).await.unwrap();
                }
            });
        }
        drop(tx);

        let mut count = 0u32;
        loop {
            match rx.recv().await {
                Ok(_) => count += 1,
                Err(_) => break,
            }
        }
        assert_eq!(count, 4000);
    });
}

// =============================================================================
// SPSC typed channel deadlock tests
// =============================================================================

#[test]
fn spsc_backpressure_ping_pong() {
    let (mut _world, mut rt) = runtime();
    must_complete_within(&mut rt, Duration::from_secs(2), async {
        let (tx, rx) = nexus_async_rt::channel::spsc::channel::<u32>(2);

        spawn_boxed(async move {
            for i in 0..10_000 {
                tx.send(i).await.unwrap();
            }
        });

        for i in 0..10_000 {
            let val = rx.recv().await.unwrap();
            assert_eq!(val, i);
        }
    });
}

// =============================================================================
// SPSC bytes channel deadlock tests
// =============================================================================

#[test]
fn spsc_bytes_backpressure() {
    let (mut _world, mut rt) = runtime();
    must_complete_within(&mut rt, Duration::from_secs(2), async {
        // Small buffer — will fill quickly with 32-byte messages.
        let (mut tx, mut rx) = nexus_async_rt::channel::spsc_bytes::channel(256);

        spawn_boxed(async move {
            let data = [0xABu8; 32];
            for _ in 0..10_000 {
                let mut claim = tx.claim(32).await.unwrap();
                claim.copy_from_slice(&data);
                claim.commit();
            }
        });

        for _ in 0..10_000 {
            let msg = rx.recv().await.unwrap();
            assert_eq!(msg.len(), 32);
            // ReadClaim drop frees space → must wake sender
        }
    });
}

#[test]
fn spsc_bytes_alternating_small_buffer() {
    let (mut _world, mut rt) = runtime();
    must_complete_within(&mut rt, Duration::from_secs(2), async {
        // Buffer barely fits one message at a time.
        let (mut tx, mut rx) = nexus_async_rt::channel::spsc_bytes::channel(128);

        spawn_boxed(async move {
            for i in 0u64..5_000 {
                let mut claim = tx.claim(8).await.unwrap();
                claim.copy_from_slice(&i.to_le_bytes());
                claim.commit();
            }
        });

        for i in 0u64..5_000 {
            let msg = rx.recv().await.unwrap();
            assert_eq!(&*msg, &i.to_le_bytes());
        }
    });
}

// =============================================================================
// MPSC bytes channel deadlock tests
// =============================================================================

#[test]
fn mpsc_bytes_backpressure() {
    let (mut _world, mut rt) = runtime();
    must_complete_within(&mut rt, Duration::from_secs(2), async {
        let (mut tx, mut rx) = nexus_async_rt::channel::mpsc_bytes::channel(256);

        spawn_boxed(async move {
            let data = [0xCDu8; 32];
            for _ in 0..10_000 {
                let mut claim = tx.claim(32).await.unwrap();
                claim.copy_from_slice(&data);
                claim.commit();
            }
        });

        for _ in 0..10_000 {
            let msg = rx.recv().await.unwrap();
            assert_eq!(msg.len(), 32);
        }
    });
}

#[test]
fn mpsc_bytes_multiple_senders() {
    let (mut _world, mut rt) = runtime();
    must_complete_within(&mut rt, Duration::from_secs(2), async {
        let (tx, mut rx) = nexus_async_rt::channel::mpsc_bytes::channel(512);

        for producer_id in 0u32..4 {
            let mut tx = tx.clone();
            spawn_boxed(async move {
                for i in 0u32..500 {
                    let val = producer_id * 10_000 + i;
                    let mut claim = tx.claim(4).await.unwrap();
                    claim.copy_from_slice(&val.to_le_bytes());
                    claim.commit();
                }
            });
        }
        drop(tx);

        let mut count = 0u32;
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    assert_eq!(msg.len(), 4);
                    count += 1;
                }
                Err(_) => break,
            }
        }
        assert_eq!(count, 2000);
    });
}

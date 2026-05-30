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
//! Channel integration tests — run through the actual executor.

use std::cell::Cell;
use std::rc::Rc;

use nexus_async_rt::channel::local;
use nexus_async_rt::{Runtime, spawn_boxed};
use nexus_rt::WorldBuilder;

fn runtime() -> (nexus_rt::World, Runtime) {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let rt = Runtime::new(&mut world);
    (world, rt)
}

#[test]
fn single_producer_consumer() {
    let (_world, mut rt) = runtime();
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let (tx, rx) = local::channel::<u32>(16);

        spawn_boxed(async move {
            for i in 0..10 {
                tx.send(i).await.unwrap();
            }
        });

        for i in 0..10 {
            let val = rx.recv().await.unwrap();
            assert_eq!(val, i);
        }
        flag.set(true);
    });

    assert!(done.get());
}

#[test]
fn multiple_producers() {
    let (_world, mut rt) = runtime();
    let total = Rc::new(Cell::new(0u64));
    let total_clone = total.clone();

    rt.block_on(async move {
        let (tx, rx) = local::channel::<u64>(64);

        // Spawn 4 producers, each sending 100 values.
        for producer_id in 0u64..4 {
            let tx = tx.clone();
            spawn_boxed(async move {
                for i in 0..100 {
                    tx.send(producer_id * 1000 + i).await.unwrap();
                }
            });
        }
        drop(tx); // drop original sender

        // Consume all 400 values.
        let mut count = 0u64;
        loop {
            match rx.recv().await {
                Ok(_val) => count += 1,
                Err(_) => break,
            }
        }
        total_clone.set(count);
    });

    assert_eq!(total.get(), 400);
}

#[test]
fn backpressure_with_small_buffer() {
    let (_world, mut rt) = runtime();
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        // Buffer of 2 — producer will block frequently.
        let (tx, rx) = local::channel::<u32>(2);

        spawn_boxed(async move {
            for i in 0..1000 {
                tx.send(i).await.unwrap();
            }
        });

        for i in 0..1000 {
            let val = rx.recv().await.unwrap();
            assert_eq!(val, i);
        }
        flag.set(true);
    });

    assert!(done.get());
}

#[test]
fn sender_drop_closes_receiver() {
    let (_world, mut rt) = runtime();
    let closed = Rc::new(Cell::new(false));
    let flag = closed.clone();

    rt.block_on(async move {
        let (tx, rx) = local::channel::<u32>(8);

        spawn_boxed(async move {
            tx.send(1).await.unwrap();
            tx.send(2).await.unwrap();
            // sender dropped here
        });

        assert_eq!(rx.recv().await.unwrap(), 1);
        assert_eq!(rx.recv().await.unwrap(), 2);
        // Now sender is dropped — recv returns error.
        assert!(rx.recv().await.is_err());
        flag.set(true);
    });

    assert!(closed.get());
}

#[test]
fn receiver_drop_signals_senders() {
    let (_world, mut rt) = runtime();
    let got_error = Rc::new(Cell::new(false));
    let flag = got_error.clone();

    rt.block_on(async move {
        let (tx, rx) = local::channel::<u32>(2);

        spawn_boxed(async move {
            // Fill buffer.
            tx.send(1).await.unwrap();
            tx.send(2).await.unwrap();
            // This should fail when receiver drops.
            match tx.send(3).await {
                Err(_) => flag.set(true),
                Ok(()) => panic!("send should fail after receiver drop"),
            }
        });

        // Receive one, then drop receiver.
        let _ = rx.recv().await.unwrap();
        drop(rx);

        // Yield so the sender task gets re-polled and sees the closed error.
        nexus_async_rt::yield_now().await;
    });

    assert!(got_error.get());
}

#[test]
fn stress_high_throughput() {
    let (_world, mut rt) = runtime();
    let total = Rc::new(Cell::new(0u64));
    let total_clone = total.clone();

    rt.block_on(async move {
        let (tx, rx) = local::channel::<u64>(256);

        spawn_boxed(async move {
            for i in 0..100_000u64 {
                tx.send(i).await.unwrap();
            }
        });

        let mut sum = 0u64;
        for _ in 0..100_000 {
            sum += rx.recv().await.unwrap();
        }
        total_clone.set(sum);
    });

    // Sum of 0..100_000
    let expected: u64 = (0..100_000u64).sum();
    assert_eq!(total.get(), expected);
}

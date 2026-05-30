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
//! Channel latency measurement with HDR histogram.
//!
//! Measures per-operation latency in nanoseconds for both local and mpsc
//! channels. Reports p50, p99, p999, p9999, max.
//!
//! Run pinned: `taskset -c 0 cargo test -p nexus-async-rt --test channel_latency --release -- --nocapture`

use std::cell::Cell;
use std::rc::Rc;
use std::time::Instant;

use hdrhistogram::Histogram;
use nexus_async_rt::{Runtime, spawn_boxed};
use nexus_rt::WorldBuilder;

const WARMUP: u64 = 10_000;
const ITERS: u64 = 500_000;

fn print_histogram(name: &str, hist: &Histogram<u64>) {
    println!("\n=== {name} ({} samples) ===", hist.len());
    println!("  p50:    {:>6} ns", hist.value_at_quantile(0.50));
    println!("  p90:    {:>6} ns", hist.value_at_quantile(0.90));
    println!("  p99:    {:>6} ns", hist.value_at_quantile(0.99));
    println!("  p99.9:  {:>6} ns", hist.value_at_quantile(0.999));
    println!("  p99.99: {:>6} ns", hist.value_at_quantile(0.9999));
    println!("  max:    {:>6} ns", hist.max());
    println!("  mean:   {:>6.1} ns", hist.mean());
}

// =============================================================================
// Local channel: try_send + try_recv latency
// =============================================================================

#[test]
#[ignore]
fn local_try_send_recv_latency() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let (tx, rx) = nexus_async_rt::channel::local::channel::<u64>(1024);
        let mut hist = Histogram::<u64>::new(3).unwrap();

        // Warmup
        for i in 0..WARMUP {
            tx.try_send(i).unwrap();
            let _ = rx.try_recv().unwrap();
        }

        // Measure
        for i in 0..ITERS {
            let start = Instant::now();
            tx.try_send(i).unwrap();
            let _ = rx.try_recv().unwrap();
            let elapsed = start.elapsed().as_nanos() as u64;
            hist.record(elapsed).unwrap();
        }

        print_histogram("local try_send+try_recv", &hist);
    });
}

// =============================================================================
// MPSC channel: try_send + try_recv latency (same thread)
// =============================================================================

#[test]
#[ignore]
fn mpsc_try_send_recv_latency() {
    let (tx, rx) = nexus_queue::mpsc::ring_buffer::<u64>(1024);

    // We use the raw nexus_queue producer/consumer here to measure
    // the atomic data path without the channel wrapper overhead.
    // This gives us the baseline.
    let mut hist = Histogram::<u64>::new(3).unwrap();

    for i in 0..WARMUP {
        tx.push(i).unwrap();
        let _ = rx.pop().unwrap();
    }

    for i in 0..ITERS {
        let start = Instant::now();
        tx.push(i).unwrap();
        let _ = rx.pop().unwrap();
        let elapsed = start.elapsed().as_nanos() as u64;
        hist.record(elapsed).unwrap();
    }

    print_histogram("nexus_queue::mpsc raw push+pop", &hist);
}

#[test]
#[ignore]
fn mpsc_channel_try_send_recv_latency() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let (tx, rx) = nexus_async_rt::channel::mpsc::channel::<u64>(1024);
        let mut hist = Histogram::<u64>::new(3).unwrap();

        for i in 0..WARMUP {
            tx.try_send(i).unwrap();
            let _ = rx.try_recv().unwrap();
        }

        for i in 0..ITERS {
            let start = Instant::now();
            tx.try_send(i).unwrap();
            let _ = rx.try_recv().unwrap();
            let elapsed = start.elapsed().as_nanos() as u64;
            hist.record(elapsed).unwrap();
        }

        print_histogram("mpsc channel try_send+try_recv", &hist);
    });
}

// =============================================================================
// Async send+recv through the executor
// =============================================================================

#[test]
#[ignore]
fn local_async_send_recv_latency() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let hist_cell = Rc::new(Cell::new(None::<Histogram<u64>>));
    let hist_ref = hist_cell.clone();

    rt.block_on(async move {
        let (tx, rx) = nexus_async_rt::channel::local::channel::<u64>(64);
        let mut hist = Histogram::<u64>::new(3).unwrap();

        // Producer task: send values with timestamp
        spawn_boxed(async move {
            for i in 0..(WARMUP + ITERS) {
                tx.send(i).await.unwrap();
            }
        });

        // Consumer: measure receive latency (includes executor dispatch)
        for _ in 0..WARMUP {
            let _ = rx.recv().await.unwrap();
        }

        for _ in 0..ITERS {
            let start = Instant::now();
            let _ = rx.recv().await.unwrap();
            let elapsed = start.elapsed().as_nanos() as u64;
            hist.record(elapsed).unwrap();
        }

        hist_ref.set(Some(hist));
    });

    let hist = hist_cell.take().unwrap();
    print_histogram("local async recv (executor dispatch)", &hist);
}

#[test]
#[ignore]
fn mpsc_async_send_recv_latency() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let hist_cell = Rc::new(Cell::new(None::<Histogram<u64>>));
    let hist_ref = hist_cell.clone();

    rt.block_on(async move {
        let (tx, rx) = nexus_async_rt::channel::mpsc::channel::<u64>(64);
        let mut hist = Histogram::<u64>::new(3).unwrap();

        spawn_boxed(async move {
            for i in 0..(WARMUP + ITERS) {
                tx.send(i).await.unwrap();
            }
        });

        for _ in 0..WARMUP {
            let _ = rx.recv().await.unwrap();
        }

        for _ in 0..ITERS {
            let start = Instant::now();
            let _ = rx.recv().await.unwrap();
            let elapsed = start.elapsed().as_nanos() as u64;
            hist.record(elapsed).unwrap();
        }

        hist_ref.set(Some(hist));
    });

    let hist = hist_cell.take().unwrap();
    print_histogram("mpsc async recv (executor dispatch)", &hist);
}

// =============================================================================
// Cross-thread: sender on std::thread, receiver in runtime
// =============================================================================

#[test]
#[ignore]
fn mpsc_cross_thread_latency() {
    // --- Busy spin (hot path, no parking) ---
    {
        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let mut rt = Runtime::new(&mut world);

        let hist_cell = Rc::new(Cell::new(None::<Histogram<u64>>));
        let hist_ref = hist_cell.clone();

        rt.block_on_busy(async move {
            let (tx, rx) = nexus_async_rt::channel::mpsc::channel::<u64>(1024);
            let mut hist = Histogram::<u64>::new(3).unwrap();

            std::thread::spawn(move || {
                for i in 0..(WARMUP + ITERS) {
                    loop {
                        match tx.try_send(i) {
                            Ok(()) => break,
                            Err(e) if e.is_full() => {
                                std::hint::spin_loop();
                                continue;
                            }
                            Err(_) => panic!("channel closed"),
                        }
                    }
                }
            });

            for _ in 0..WARMUP {
                let _ = rx.recv().await.unwrap();
            }
            for _ in 0..ITERS {
                let start = Instant::now();
                let _ = rx.recv().await.unwrap();
                let elapsed = start.elapsed().as_nanos() as u64;
                hist.record(elapsed).unwrap();
            }
            hist_ref.set(Some(hist));
        });

        let hist = hist_cell.take().unwrap();
        print_histogram("mpsc cross-thread recv (busy spin)", &hist);
    }

    // --- Park mode (epoll, eventfd wake) ---
    {
        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let mut rt = Runtime::new(&mut world);

        let hist_cell = Rc::new(Cell::new(None::<Histogram<u64>>));
        let hist_ref = hist_cell.clone();

        rt.block_on(async move {
            let (tx, rx) = nexus_async_rt::channel::mpsc::channel::<u64>(1024);
            let mut hist = Histogram::<u64>::new(3).unwrap();

            std::thread::spawn(move || {
                for i in 0..(WARMUP + ITERS) {
                    loop {
                        match tx.try_send(i) {
                            Ok(()) => break,
                            Err(e) if e.is_full() => {
                                std::hint::spin_loop();
                                continue;
                            }
                            Err(_) => panic!("channel closed"),
                        }
                    }
                }
            });

            for _ in 0..WARMUP {
                let _ = rx.recv().await.unwrap();
            }
            for _ in 0..ITERS {
                let start = Instant::now();
                let _ = rx.recv().await.unwrap();
                let elapsed = start.elapsed().as_nanos() as u64;
                hist.record(elapsed).unwrap();
            }
            hist_ref.set(Some(hist));
        });

        let hist = hist_cell.take().unwrap();
        print_histogram("mpsc cross-thread recv (park, saturated)", &hist);
    }

    // --- Park mode with slow sender (forces eventfd wake) ---
    {
        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let mut rt = Runtime::new(&mut world);

        // Fewer iterations since we're sleeping between sends.
        const SLOW_WARMUP: u64 = 1_000;
        const SLOW_ITERS: u64 = 10_000;

        let hist_cell = Rc::new(Cell::new(None::<Histogram<u64>>));
        let hist_ref = hist_cell.clone();

        rt.block_on(async move {
            let (tx, rx) = nexus_async_rt::channel::mpsc::channel::<u64>(1024);
            let mut hist = Histogram::<u64>::new(3).unwrap();

            std::thread::spawn(move || {
                // Sleep 10µs between sends to force receiver into epoll park.
                for i in 0..(SLOW_WARMUP + SLOW_ITERS) {
                    tx.try_send(i).unwrap();
                    std::thread::sleep(std::time::Duration::from_micros(10));
                }
            });

            for _ in 0..SLOW_WARMUP {
                let _ = rx.recv().await.unwrap();
            }
            for _ in 0..SLOW_ITERS {
                let start = Instant::now();
                let _ = rx.recv().await.unwrap();
                let elapsed = start.elapsed().as_nanos() as u64;
                hist.record(elapsed).unwrap();
            }
            hist_ref.set(Some(hist));
        });

        let hist = hist_cell.take().unwrap();
        print_histogram("mpsc cross-thread recv (park, eventfd wake)", &hist);
    }
}

// =============================================================================
// SPSC channel
// =============================================================================

#[test]
#[ignore]
fn spsc_try_send_recv_latency() {
    // Raw nexus_queue::spsc baseline.
    let (tx, rx) = nexus_queue::spsc::ring_buffer::<u64>(1024);
    let mut hist = Histogram::<u64>::new(3).unwrap();

    for i in 0..WARMUP {
        tx.push(i).unwrap();
        let _ = rx.pop().unwrap();
    }
    for i in 0..ITERS {
        let start = Instant::now();
        tx.push(i).unwrap();
        let _ = rx.pop().unwrap();
        let elapsed = start.elapsed().as_nanos() as u64;
        hist.record(elapsed).unwrap();
    }
    print_histogram("nexus_queue::spsc raw push+pop", &hist);
}

#[test]
#[ignore]
fn spsc_channel_try_send_recv_latency() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let (tx, rx) = nexus_async_rt::channel::spsc::channel::<u64>(1024);
        let mut hist = Histogram::<u64>::new(3).unwrap();

        for i in 0..WARMUP {
            tx.try_send(i).unwrap();
            let _ = rx.try_recv().unwrap();
        }
        for i in 0..ITERS {
            let start = Instant::now();
            tx.try_send(i).unwrap();
            let _ = rx.try_recv().unwrap();
            let elapsed = start.elapsed().as_nanos() as u64;
            hist.record(elapsed).unwrap();
        }
        print_histogram("spsc channel try_send+try_recv", &hist);
    });
}

#[test]
#[ignore]
fn spsc_async_send_recv_latency() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let hist_cell = Rc::new(Cell::new(None::<Histogram<u64>>));
    let hist_ref = hist_cell.clone();

    rt.block_on(async move {
        let (tx, rx) = nexus_async_rt::channel::spsc::channel::<u64>(64);
        let mut hist = Histogram::<u64>::new(3).unwrap();

        spawn_boxed(async move {
            for i in 0..(WARMUP + ITERS) {
                tx.send(i).await.unwrap();
            }
        });

        for _ in 0..WARMUP {
            let _ = rx.recv().await.unwrap();
        }
        for _ in 0..ITERS {
            let start = Instant::now();
            let _ = rx.recv().await.unwrap();
            let elapsed = start.elapsed().as_nanos() as u64;
            hist.record(elapsed).unwrap();
        }
        hist_ref.set(Some(hist));
    });

    let hist = hist_cell.take().unwrap();
    print_histogram("spsc async recv (executor dispatch)", &hist);
}

#[test]
#[ignore]
fn spsc_cross_thread_latency() {
    // --- Busy spin ---
    {
        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let mut rt = Runtime::new(&mut world);

        let hist_cell = Rc::new(Cell::new(None::<Histogram<u64>>));
        let hist_ref = hist_cell.clone();

        rt.block_on_busy(async move {
            let (tx, rx) = nexus_async_rt::channel::spsc::channel::<u64>(1024);
            let mut hist = Histogram::<u64>::new(3).unwrap();

            std::thread::spawn(move || {
                for i in 0..(WARMUP + ITERS) {
                    loop {
                        match tx.try_send(i) {
                            Ok(()) => break,
                            Err(e) if e.is_full() => {
                                std::hint::spin_loop();
                                continue;
                            }
                            Err(_) => panic!("channel closed"),
                        }
                    }
                }
            });

            for _ in 0..WARMUP {
                let _ = rx.recv().await.unwrap();
            }
            for _ in 0..ITERS {
                let start = Instant::now();
                let _ = rx.recv().await.unwrap();
                let elapsed = start.elapsed().as_nanos() as u64;
                hist.record(elapsed).unwrap();
            }
            hist_ref.set(Some(hist));
        });

        let hist = hist_cell.take().unwrap();
        print_histogram("spsc cross-thread recv (busy spin)", &hist);
    }

    // --- Park, saturated ---
    {
        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let mut rt = Runtime::new(&mut world);

        let hist_cell = Rc::new(Cell::new(None::<Histogram<u64>>));
        let hist_ref = hist_cell.clone();

        rt.block_on(async move {
            let (tx, rx) = nexus_async_rt::channel::spsc::channel::<u64>(1024);
            let mut hist = Histogram::<u64>::new(3).unwrap();

            std::thread::spawn(move || {
                for i in 0..(WARMUP + ITERS) {
                    loop {
                        match tx.try_send(i) {
                            Ok(()) => break,
                            Err(e) if e.is_full() => {
                                std::hint::spin_loop();
                                continue;
                            }
                            Err(_) => panic!("channel closed"),
                        }
                    }
                }
            });

            for _ in 0..WARMUP {
                let _ = rx.recv().await.unwrap();
            }
            for _ in 0..ITERS {
                let start = Instant::now();
                let _ = rx.recv().await.unwrap();
                let elapsed = start.elapsed().as_nanos() as u64;
                hist.record(elapsed).unwrap();
            }
            hist_ref.set(Some(hist));
        });

        let hist = hist_cell.take().unwrap();
        print_histogram("spsc cross-thread recv (park, saturated)", &hist);
    }
}

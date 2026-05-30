//! EventQueue benchmark — cycle-accurate latency measurement.
//!
//! Measures notify(), poll(), and poll_limit() latency in CPU cycles
//! using rdtsc. Includes single-threaded hot-path benchmarks and
//! cross-thread round-trip latency.
//!
//! Usage:
//!   cargo build --release --example bench_event_queue -p nexus-notify
//!   taskset -c 0 ./target/release/examples/bench_event_queue
//!   taskset -c 0,2 ./target/release/examples/bench_event_queue roundtrip

#![allow(clippy::large_stack_frames)]

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use nexus_notify::{Events, Token, event_queue};

// ============================================================================
// Timing
// ============================================================================

#[inline(always)]
fn rdtsc_start() -> u64 {
    // SAFETY: x86_64 intrinsics for serialized timestamp counter read.
    unsafe {
        std::arch::x86_64::_mm_lfence();
        std::arch::x86_64::_rdtsc()
    }
}

#[inline(always)]
fn rdtsc_end() -> u64 {
    // SAFETY: rdtscp is serializing on the read side.
    unsafe {
        let tsc = std::arch::x86_64::__rdtscp(&mut 0u32 as *mut _);
        std::arch::x86_64::_mm_lfence();
        tsc
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    let idx = ((sorted.len() as f64) * p / 100.0) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_row(label: &str, samples: &mut [u64]) {
    samples.sort_unstable();
    println!(
        "  {:<20} {:>5} {:>5} {:>5} {:>6} {:>7} {:>7}",
        label,
        percentile(samples, 50.0),
        percentile(samples, 90.0),
        percentile(samples, 99.0),
        percentile(samples, 99.9),
        percentile(samples, 99.99),
        samples[samples.len() - 1],
    );
}

fn print_header() {
    println!(
        "  {:<20} {:>5} {:>5} {:>5} {:>6} {:>7} {:>7}",
        "", "p50", "p90", "p99", "p99.9", "p99.99", "max"
    );
}

const SAMPLES: usize = 50_000;
const WARMUP: usize = 5_000;

// ============================================================================
// Unroll
// ============================================================================

macro_rules! unroll_10 {
    ($op:expr) => {
        $op;
        $op;
        $op;
        $op;
        $op;
        $op;
        $op;
        $op;
        $op;
        $op;
    };
}

macro_rules! unroll_100 {
    ($op:expr) => {
        unroll_10!($op);
        unroll_10!($op);
        unroll_10!($op);
        unroll_10!($op);
        unroll_10!($op);
        unroll_10!($op);
        unroll_10!($op);
        unroll_10!($op);
        unroll_10!($op);
        unroll_10!($op);
    };
}

// ============================================================================
// NOTIFY
// ============================================================================

fn bench_notify() {
    println!("\nNOTIFY");
    print_header();

    // Unconflated: token is polled between each sample
    {
        let (notifier, poller) = event_queue(64);
        let token = Token::new(0);
        let mut events = Events::with_capacity(64);

        for _ in 0..WARMUP {
            notifier.notify(token).ok();
            poller.poll(&mut events);
        }

        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            poller.poll(&mut events);
            let start = rdtsc_start();
            unroll_100!(notifier.notify(black_box(token)).ok());
            let end = rdtsc_end();
            samples.push((end - start) / 100);
        }
        print_row("notify (new)", &mut samples);
    }

    // Conflated: same token notified 100x without poll
    {
        let (notifier, poller) = event_queue(64);
        let token = Token::new(0);
        let mut events = Events::with_capacity(64);

        notifier.notify(token).ok();
        for _ in 0..WARMUP {
            notifier.notify(token).ok();
        }

        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let start = rdtsc_start();
            unroll_100!(notifier.notify(black_box(token)).ok());
            let end = rdtsc_end();
            samples.push((end - start) / 100);
        }
        poller.poll(&mut events);
        print_row("notify (conflated)", &mut samples);
    }
}

// ============================================================================
// POLL EMPTY
// ============================================================================

fn bench_poll_empty() {
    println!("\nPOLL EMPTY (nothing ready)");
    print_header();

    let (_, poller) = event_queue(4096);
    let mut events = Events::with_capacity(4096);

    for _ in 0..WARMUP {
        poller.poll(&mut events);
    }

    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = rdtsc_start();
        unroll_100!(poller.poll(black_box(&mut events)));
        let end = rdtsc_end();
        samples.push((end - start) / 100);
    }
    print_row("cap=4096", &mut samples);
}

// ============================================================================
// POLL DENSITY
// ============================================================================

fn bench_poll_density() {
    println!("\nPOLL DENSITY (cap=4096, N tokens ready)");
    print_header();

    let cap = 4096usize;

    for n_ready in [1usize, 8, 32, 64, 128, 256, 512, 1024, 4096] {
        let (notifier, poller) = event_queue(cap);
        let mut events = Events::with_capacity(cap);

        let stride = cap / n_ready;
        let tokens: Vec<Token> = (0..n_ready).map(|i| Token::new(i * stride)).collect();

        // Warmup
        for _ in 0..WARMUP {
            for t in &tokens {
                notifier.notify(*t).ok();
            }
            poller.poll(&mut events);
        }

        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            for t in &tokens {
                notifier.notify(*t).ok();
            }

            let start = rdtsc_start();
            poller.poll(black_box(&mut events));
            let end = rdtsc_end();
            samples.push(end - start);
        }
        print_row(&format!("N={n_ready}"), &mut samples);
    }
}

// ============================================================================
// POLL_LIMIT
// ============================================================================

fn bench_poll_limit() {
    println!("\nPOLL_LIMIT (cap=4096, all ready, varying limit)");
    print_header();

    for limit in [32usize, 64, 128, 256, 512] {
        let (notifier, poller) = event_queue(4096);
        let mut events = Events::with_capacity(4096);
        let tokens: Vec<Token> = (0..4096).map(Token::new).collect();

        // Warmup
        for _ in 0..WARMUP {
            for t in &tokens {
                notifier.notify(*t).ok();
            }
            poller.poll(&mut events);
        }

        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            for t in &tokens {
                notifier.notify(*t).ok();
            }

            let start = rdtsc_start();
            poller.poll_limit(black_box(&mut events), limit);
            let end = rdtsc_end();
            samples.push(end - start);

            // Drain the rest so next iteration starts clean
            poller.poll(&mut events);
        }
        print_row(&format!("limit={limit}"), &mut samples);
    }
}

// ============================================================================
// ROUNDTRIP
// ============================================================================

const RT_WARMUP: u64 = 1_000;
const RT_SAMPLES: u64 = 50_000;

fn bench_roundtrip() {
    println!("\nROUNDTRIP (cross-thread notify → poll → ack, RTT/2)");
    print_header();

    let (notifier_fwd, poller_fwd) = event_queue(64);
    let (notifier_rev, poller_rev) = event_queue(64);

    let token_fwd = Token::new(0);
    let token_rev = Token::new(0);

    let total = RT_WARMUP + RT_SAMPLES;

    let worker = thread::spawn(move || {
        let mut events = Events::with_capacity(64);
        for _ in 0..total {
            loop {
                poller_fwd.poll(&mut events);
                if !events.is_empty() {
                    break;
                }
                std::hint::spin_loop();
            }
            notifier_rev.notify(token_rev).ok();
        }
    });

    let mut events = Events::with_capacity(64);
    let mut samples = Vec::with_capacity(RT_SAMPLES as usize);

    for i in 0..total {
        let start = rdtsc_start();

        notifier_fwd.notify(token_fwd).ok();

        loop {
            poller_rev.poll(&mut events);
            if !events.is_empty() {
                break;
            }
            std::hint::spin_loop();
        }

        let elapsed = rdtsc_end() - start;

        if i >= RT_WARMUP {
            samples.push(elapsed / 2);
        }
    }

    worker.join().unwrap();
    print_row("rtt/2", &mut samples);
}

// ============================================================================
// CONTENDED
// ============================================================================

fn bench_contended() {
    println!("\nCONTENDED NOTIFY (P producers, 1 consumer, cap=4096)");
    print_header();

    for num_producers in [1usize, 2, 4] {
        let (notifier, poller) = event_queue(4096);
        let mut events = Events::with_capacity(4096);
        let done = Arc::new(AtomicBool::new(false));

        let mut handles = Vec::new();
        for p in 0..num_producers {
            let n = notifier.clone();
            let token = Token::new(p);
            let done = Arc::clone(&done);
            handles.push(thread::spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    n.notify(token).ok();
                }
            }));
        }

        for _ in 0..WARMUP {
            poller.poll(&mut events);
        }

        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let start = rdtsc_start();
            poller.poll(black_box(&mut events));
            let end = rdtsc_end();
            samples.push(end - start);
        }

        done.store(true, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }

        print_row(&format!("P={num_producers}"), &mut samples);
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str);

    println!("nexus-notify ReadySet benchmark (MPSC queue + dedup flags)");
    println!("Samples: {SAMPLES}, warmup: {WARMUP}");
    println!("All times in CPU cycles (rdtsc)");

    match mode {
        Some("roundtrip") => {
            bench_roundtrip();
            bench_contended();
        }
        Some("single") | None => {
            bench_notify();
            bench_poll_empty();
            bench_poll_density();
            bench_poll_limit();
        }
        Some("all") => {
            bench_notify();
            bench_poll_empty();
            bench_poll_density();
            bench_poll_limit();
            bench_roundtrip();
            bench_contended();
        }
        Some(other) => {
            eprintln!("Unknown mode: {other}");
            eprintln!("Usage: bench_event_queue [single|roundtrip|all]");
            eprintln!("  single    — single-threaded ops (default)");
            eprintln!("  roundtrip — cross-thread latency + contention");
            eprintln!("  all       — everything");
            std::process::exit(1);
        }
    }
}

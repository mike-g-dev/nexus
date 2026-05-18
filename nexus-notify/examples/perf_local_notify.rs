//! LocalNotify performance benchmark.
//!
//! Measures cycle counts for mark, poll, and full frame operations.
//!
//! ```bash
//! taskset -c 0 cargo run --release -p nexus-notify --example perf_local_notify
//! ```

use std::hint::black_box;

use nexus_notify::{Events, LocalNotify, Token};

// =============================================================================
// Bench infrastructure
// =============================================================================

const ITERATIONS: usize = 100_000;
const WARMUP: usize = 10_000;
const BATCH: u64 = 100;

#[inline(always)]
#[cfg(target_arch = "x86_64")]
fn rdtsc_start() -> u64 {
    // SAFETY: x86_64 intrinsics for serialized timestamp counter read.
    // lfence ensures all prior instructions complete before reading rdtsc.
    unsafe {
        core::arch::x86_64::_mm_lfence();
        core::arch::x86_64::_rdtsc()
    }
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
fn rdtsc_end() -> u64 {
    // SAFETY: rdtscp serializes on the read side (waits for prior instructions).
    // Trailing lfence prevents subsequent instructions from reordering before the read.
    unsafe {
        let mut aux = 0u32;
        let tsc = core::arch::x86_64::__rdtscp(&raw mut aux);
        core::arch::x86_64::_mm_lfence();
        tsc
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    let idx = ((sorted.len() as f64) * p / 100.0) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn bench_batched<F: FnMut() -> u64>(name: &str, mut f: F) -> (u64, u64, u64) {
    for _ in 0..WARMUP {
        black_box(f());
    }
    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = rdtsc_start();
        for _ in 0..BATCH {
            black_box(f());
        }
        let end = rdtsc_end();
        samples.push(end.wrapping_sub(start) / BATCH);
    }
    samples.sort_unstable();
    let p50 = percentile(&samples, 50.0);
    let p99 = percentile(&samples, 99.0);
    let p999 = percentile(&samples, 99.9);
    println!("{:<52} {:>8} {:>8} {:>8}", name, p50, p99, p999);
    (p50, p99, p999)
}

fn print_header(title: &str) {
    println!("\n=== {} ===\n", title);
    println!(
        "{:<52} {:>8} {:>8} {:>8}",
        "Operation", "p50", "p99", "p999"
    );
    println!("{}", "-".repeat(80));
}

// =============================================================================
// Scenarios
// =============================================================================

fn scenario_mark_single() {
    print_header("mark() + poll — single token");

    let mut notify = LocalNotify::with_capacity(4);
    let mut events = Events::with_capacity(4);
    let t = notify.register();

    bench_batched("mark + poll (1 token)", || {
        notify.mark(t);
        notify.poll(&mut events);
        events.len() as u64
    });
}

fn scenario_mark_scale() {
    print_header("mark() + poll — varying token count");

    for &count in &[1, 5, 10, 50, 200] {
        let mut notify = LocalNotify::with_capacity(count);
        let mut events = Events::with_capacity(count);
        let tokens: Vec<Token> = (0..count).map(|_| notify.register()).collect();

        let label = format!("mark + poll ({} tokens)", count);
        bench_batched(&label, || {
            for &t in &tokens {
                notify.mark(t);
            }
            notify.poll(&mut events);
            events.len() as u64
        });
    }
}

fn scenario_dedup() {
    print_header("dedup — mark same token N times + poll");

    let mut notify = LocalNotify::with_capacity(4);
    let mut events = Events::with_capacity(4);
    let t = notify.register();

    for &marks in &[2, 10, 100] {
        let label = format!("mark {}x same token + poll", marks);
        bench_batched(&label, || {
            for _ in 0..marks {
                notify.mark(t);
            }
            notify.poll(&mut events);
            events.len() as u64
        });
    }
}

fn scenario_poll_only() {
    print_header("poll cost — tokens pre-marked");

    for &count in &[10, 50, 200] {
        let mut notify = LocalNotify::with_capacity(count);
        let mut events = Events::with_capacity(count);
        let tokens: Vec<Token> = (0..count).map(|_| notify.register()).collect();

        let label = format!("poll {} tokens", count);
        bench_batched(&label, || {
            for &t in &tokens {
                notify.mark(t);
            }
            notify.poll(&mut events);
            events.len() as u64
        });
    }
}

fn scenario_poll_limit() {
    print_header("poll_limit — partial drain");

    let mut notify = LocalNotify::with_capacity(200);
    let mut events = Events::with_capacity(200);
    let tokens: Vec<Token> = (0..200).map(|_| notify.register()).collect();

    bench_batched("poll_limit(32) from 200 marked", || {
        for &t in &tokens {
            notify.mark(t);
        }
        notify.poll_limit(&mut events, 32);
        // Clean up remainder
        notify.poll(&mut events);
        0
    });
}

fn scenario_bitset_scaling() {
    print_header("bitset scaling — mark 5 tokens out of N registered");

    for &total in &[64, 256, 1024, 4096] {
        let mut notify = LocalNotify::with_capacity(total);
        let mut events = Events::with_capacity(total);
        let mut tokens = Vec::new();
        for _ in 0..total {
            tokens.push(notify.register());
        }
        // Only mark first 5
        let hot: Vec<Token> = tokens[..5].to_vec();

        let label = format!("mark 5 + poll — {} total registered", total);
        bench_batched(&label, || {
            for &t in &hot {
                notify.mark(t);
            }
            notify.poll(&mut events);
            events.len() as u64
        });
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("LocalNotify Performance Benchmark");
    println!("Cycles per operation (batched, {} ops/sample)\n", BATCH);

    scenario_mark_single();
    scenario_mark_scale();
    scenario_dedup();
    scenario_poll_only();
    scenario_poll_limit();
    scenario_bitset_scaling();
}

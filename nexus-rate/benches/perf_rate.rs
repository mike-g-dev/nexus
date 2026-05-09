//! Cycles-per-operation benchmark for all nexus-rate primitives.
//!
//! Usage:
//!   cargo build --release --example perf_rate -p nexus-rate
//!   taskset -c 0 ./target/release/examples/perf_rate

use std::hint::black_box;
use std::time::{Duration, Instant};

use nexus_rate::{local, sync};

// ============================================================================
// Timing
// ============================================================================

#[inline(always)]
fn rdtsc_start() -> u64 {
    unsafe {
        std::arch::x86_64::_mm_lfence();
        std::arch::x86_64::_rdtsc()
    }
}

#[inline(always)]
fn rdtsc_end() -> u64 {
    unsafe {
        let mut aux = 0u32;
        let tsc = std::arch::x86_64::__rdtscp(&raw mut aux);
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
        "  {:<32} {:>6} {:>6} {:>6} {:>7} {:>7}",
        label,
        percentile(samples, 50.0),
        percentile(samples, 90.0),
        percentile(samples, 99.0),
        percentile(samples, 99.9),
        samples[samples.len() - 1],
    );
}

fn print_header() {
    println!(
        "  {:<32} {:>6} {:>6} {:>6} {:>7} {:>7}",
        "(cycles/op)", "p50", "p90", "p99", "p99.9", "max"
    );
}

const SAMPLES: usize = 100_000;
const WARMUP: usize = 10_000;
const BATCH: u64 = 64;

// ============================================================================
// Local GCRA
// ============================================================================

fn bench_local_gcra_allowed(samples: &mut [u64]) {
    let base = Instant::now();
    let mut g = local::Gcra::builder()
        .rate(1_000_000)
        .period(Duration::from_nanos(1_000_000))
        .burst(100)
        .build()
        .unwrap();
    let mut t = 0u64;

    for _ in 0..WARMUP {
        t += 1;
        let _ = g.try_acquire(1, base + Duration::from_nanos(t));
    }

    for s in samples.iter_mut() {
        let start = rdtsc_start();
        for _ in 0..BATCH {
            t += 1;
            black_box(g.try_acquire(1, base + Duration::from_nanos(t)));
        }
        let end = rdtsc_end();
        *s = (end - start) / BATCH;
    }
}

fn bench_local_gcra_rejected(samples: &mut [u64]) {
    let base = Instant::now();
    // Rate of 1 per 1M nanos, no burst — nearly all requests rejected
    let mut g = local::Gcra::builder()
        .rate(1)
        .period(Duration::from_nanos(1_000_000))
        .burst(0)
        .build()
        .unwrap();
    let _ = g.try_acquire(1, base); // consume the one allowed

    for s in samples.iter_mut() {
        let start = rdtsc_start();
        for _ in 0..BATCH {
            black_box(g.try_acquire(1, base + Duration::from_nanos(1)));
        }
        let end = rdtsc_end();
        *s = (end - start) / BATCH;
    }
}

// ============================================================================
// Sync GCRA
// ============================================================================

fn bench_sync_gcra_allowed(samples: &mut [u64]) {
    let base = Instant::now();
    let g = sync::Gcra::builder()
        .rate(1_000_000)
        .period(Duration::from_nanos(1_000_000))
        .burst(100)
        .now(base)
        .build()
        .unwrap();
    let mut t = 0u64;

    for _ in 0..WARMUP {
        t += 1;
        let _ = g.try_acquire(1, base + Duration::from_nanos(t));
    }

    for s in samples.iter_mut() {
        let tsc_start = rdtsc_start();
        for _ in 0..BATCH {
            t += 1;
            black_box(g.try_acquire(1, base + Duration::from_nanos(t)));
        }
        let tsc_end = rdtsc_end();
        *s = (tsc_end - tsc_start) / BATCH;
    }
}

fn bench_sync_gcra_rejected(samples: &mut [u64]) {
    let base = Instant::now();
    let g = sync::Gcra::builder()
        .rate(1)
        .period(Duration::from_nanos(1_000_000))
        .burst(0)
        .now(base)
        .build()
        .unwrap();
    let _ = g.try_acquire(1, base);

    for s in samples.iter_mut() {
        let tsc_start = rdtsc_start();
        for _ in 0..BATCH {
            black_box(g.try_acquire(1, base + Duration::from_nanos(1)));
        }
        let tsc_end = rdtsc_end();
        *s = (tsc_end - tsc_start) / BATCH;
    }
}

// ============================================================================
// Local TokenBucket
// ============================================================================

fn bench_local_tb_allowed(samples: &mut [u64]) {
    let base = Instant::now();
    let mut tb = local::TokenBucket::builder()
        .rate(1_000_000)
        .period(Duration::from_nanos(1_000_000))
        .burst(1_000_000)
        .now(base)
        .build()
        .unwrap();
    let mut t = 0u64;

    for _ in 0..WARMUP {
        t += 1;
        let _ = tb.try_acquire(1, base + Duration::from_nanos(t));
    }

    for s in samples.iter_mut() {
        let start = rdtsc_start();
        for _ in 0..BATCH {
            t += 1;
            black_box(tb.try_acquire(1, base + Duration::from_nanos(t)));
        }
        let end = rdtsc_end();
        *s = (end - start) / BATCH;
    }
}

fn bench_local_tb_rejected(samples: &mut [u64]) {
    let base = Instant::now();
    let mut tb = local::TokenBucket::builder()
        .rate(1)
        .period(Duration::from_nanos(1_000_000))
        .burst(1)
        .now(base)
        .build()
        .unwrap();
    let _ = tb.try_acquire(1, base + Duration::from_nanos(1)); // consume the one token

    for s in samples.iter_mut() {
        let start = rdtsc_start();
        for _ in 0..BATCH {
            black_box(tb.try_acquire(1, base + Duration::from_nanos(1)));
        }
        let end = rdtsc_end();
        *s = (end - start) / BATCH;
    }
}

// ============================================================================
// Sync TokenBucket
// ============================================================================

fn bench_sync_tb_allowed(samples: &mut [u64]) {
    let base = Instant::now();
    let tb = sync::TokenBucket::builder()
        .rate(1_000_000)
        .period(Duration::from_nanos(1_000_000))
        .burst(1_000_000)
        .now(base)
        .build()
        .unwrap();
    let mut t = 0u64;

    for _ in 0..WARMUP {
        t += 1;
        let _ = tb.try_acquire(1, base + Duration::from_nanos(t));
    }

    for s in samples.iter_mut() {
        let tsc_start = rdtsc_start();
        for _ in 0..BATCH {
            t += 1;
            black_box(tb.try_acquire(1, base + Duration::from_nanos(t)));
        }
        let tsc_end = rdtsc_end();
        *s = (tsc_end - tsc_start) / BATCH;
    }
}

fn bench_sync_tb_rejected(samples: &mut [u64]) {
    let base = Instant::now();
    let tb = sync::TokenBucket::builder()
        .rate(1)
        .period(Duration::from_nanos(1_000_000))
        .burst(1)
        .now(base)
        .build()
        .unwrap();
    let _ = tb.try_acquire(1, base + Duration::from_nanos(1));

    for s in samples.iter_mut() {
        let tsc_start = rdtsc_start();
        for _ in 0..BATCH {
            black_box(tb.try_acquire(1, base + Duration::from_nanos(1)));
        }
        let tsc_end = rdtsc_end();
        *s = (tsc_end - tsc_start) / BATCH;
    }
}

// ============================================================================
// Local SlidingWindow
// ============================================================================

fn bench_local_sw_allowed(samples: &mut [u64]) {
    let base = Instant::now();
    let mut sw = local::SlidingWindow::builder()
        .window(Duration::from_nanos(1_000_000))
        .sub_windows(10)
        .limit(10_000_000)
        .now(base)
        .build()
        .unwrap();
    let mut t = 0u64;

    for _ in 0..WARMUP {
        t += 1;
        let _ = sw.try_acquire(1, base + Duration::from_nanos(t));
    }

    for s in samples.iter_mut() {
        let start = rdtsc_start();
        for _ in 0..BATCH {
            t += 1;
            black_box(sw.try_acquire(1, base + Duration::from_nanos(t)));
        }
        let end = rdtsc_end();
        *s = (end - start) / BATCH;
    }
}

fn bench_local_sw_rejected(samples: &mut [u64]) {
    let base = Instant::now();
    let mut sw = local::SlidingWindow::builder()
        .window(Duration::from_nanos(1_000_000))
        .sub_windows(10)
        .limit(1)
        .now(base)
        .build()
        .unwrap();
    let _ = sw.try_acquire(1, base); // consume the one allowed

    for s in samples.iter_mut() {
        let start = rdtsc_start();
        for _ in 0..BATCH {
            black_box(sw.try_acquire(1, base));
        }
        let end = rdtsc_end();
        *s = (end - start) / BATCH;
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("\nnexus-rate benchmark — cycles per operation (batch={BATCH})");
    println!("==========================================================\n");

    let mut buf = vec![0u64; SAMPLES];

    println!("  --- GCRA ---");
    print_header();
    bench_local_gcra_allowed(&mut buf);
    print_row("local::Gcra (allowed)", &mut buf);
    bench_local_gcra_rejected(&mut buf);
    print_row("local::Gcra (rejected)", &mut buf);
    bench_sync_gcra_allowed(&mut buf);
    print_row("sync::Gcra (allowed)", &mut buf);
    bench_sync_gcra_rejected(&mut buf);
    print_row("sync::Gcra (rejected)", &mut buf);

    println!("\n  --- Token Bucket ---");
    print_header();
    bench_local_tb_allowed(&mut buf);
    print_row("local::TokenBucket (allowed)", &mut buf);
    bench_local_tb_rejected(&mut buf);
    print_row("local::TokenBucket (rejected)", &mut buf);
    bench_sync_tb_allowed(&mut buf);
    print_row("sync::TokenBucket (allowed)", &mut buf);
    bench_sync_tb_rejected(&mut buf);
    print_row("sync::TokenBucket (rejected)", &mut buf);

    println!("\n  --- Sliding Window ---");
    print_header();
    bench_local_sw_allowed(&mut buf);
    print_row("local::SlidingWindow (allowed)", &mut buf);
    bench_local_sw_rejected(&mut buf);
    print_row("local::SlidingWindow (rejected)", &mut buf);

    println!();
}

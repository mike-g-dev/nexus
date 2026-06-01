//! Cycle-level benchmarks for FIX delimiter scanning.
//!
//! Tests `find_soh` and `find_eq` at various buffer lengths to show
//! the SIMD dispatch cost and cycles/byte throughput.
//!
//! Run with:
//! ```bash
//! # SSE2 (default on x86_64)
//! cargo build --release --example perf_scan -p nexus-fix-codec
//! taskset -c 0 ./target/release/examples/perf_scan
//!
//! # AVX2
//! RUSTFLAGS="-C target-feature=+avx2" cargo build --release --example perf_scan -p nexus-fix-codec
//! taskset -c 0 ./target/release/examples/perf_scan
//! ```

#[path = "_bench_utils.rs"]
mod _bench_utils;

use _bench_utils::{BATCH, WARMUP, percentile, print_intro, rdtsc_fenced_end, rdtsc_fenced_start};
use nexus_fix_codec::reader::{FieldReader, checksum};
use nexus_fix_codec::scan;
use nexus_fix_codec::writer::FieldWriter;
use std::hint::black_box;

/// Number of batches sampled per benchmark. Each batch averages `BATCH`
/// calls, so this is the number of percentile data points.
const SAMPLES: usize = 50_000;

/// Cycle cost per call, reported as percentiles over `SAMPLES` batches.
///
/// Each sample times `BATCH` back-to-back calls between a single
/// `lfence`-serialized `rdtsc`/`rdtscp` pair and divides by `BATCH`. This
/// amortizes the ~20-30 cycle timestamp+fence overhead that a single-shot
/// `rdtsc(); f(); rdtsc()` cannot resolve below — so sub-30-cycle kernels
/// (scan, checksum) report their real per-call throughput instead of the
/// measurement floor. Numbers are sustained per-call cost (back-to-back),
/// not isolated single-call latency.
fn benchmark<T, F: FnMut() -> T>(mut f: F) -> (u64, u64, u64) {
    for _ in 0..WARMUP {
        black_box(f());
    }

    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = rdtsc_fenced_start();
        for _ in 0..BATCH {
            black_box(f());
        }
        let end = rdtsc_fenced_end();
        samples.push(end.wrapping_sub(start) / BATCH);
    }

    samples.sort_unstable();
    (
        percentile(&samples, 50.0),
        percentile(&samples, 99.0),
        percentile(&samples, 99.9),
    )
}

fn main() {
    print_intro("FIX SCAN CYCLE BENCHMARK");

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    println!("SIMD: AVX2 (32 bytes/iteration)\n");
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    println!("SIMD: SSE2 (16 bytes/iteration)\n");
    #[cfg(not(target_arch = "x86_64"))]
    println!("SIMD: Scalar SWAR (8 bytes/iteration)\n");

    // =========================================================================
    // find_soh: target at end (worst case — full scan)
    // =========================================================================

    let lengths = [
        4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
    ];

    println!("=== find_soh: target at end (worst case full scan) ===\n");
    println!(
        "{:<30} {:>8} {:>8} {:>8} {:>10}",
        "Length", "p50", "p99", "p999", "cyc/byte"
    );
    println!("{}", "-".repeat(70));

    for &len in &lengths {
        let mut buf = vec![b'A'; len];
        *buf.last_mut().unwrap() = 0x01;

        let (p50, p99, p999) = benchmark(|| scan::find_soh(black_box(&buf), 0));

        println!(
            "{:<30} {:>8} {:>8} {:>8} {:>10.2}",
            format!("{}B", len),
            p50,
            p99,
            p999,
            p50 as f64 / len as f64
        );
    }

    // =========================================================================
    // find_soh: no match (scan entire buffer, return None)
    // =========================================================================

    println!("\n=== find_soh: no match (full scan, return None) ===\n");
    println!(
        "{:<30} {:>8} {:>8} {:>8} {:>10}",
        "Length", "p50", "p99", "p999", "cyc/byte"
    );
    println!("{}", "-".repeat(70));

    for &len in &lengths {
        let buf = vec![b'A'; len];

        let (p50, p99, p999) = benchmark(|| scan::find_soh(black_box(&buf), 0));

        println!(
            "{:<30} {:>8} {:>8} {:>8} {:>10.2}",
            format!("{}B", len),
            p50,
            p99,
            p999,
            p50 as f64 / len as f64
        );
    }

    // =========================================================================
    // find_eq: tag=value separation (typical short scan)
    // =========================================================================

    println!("\n=== find_eq: typical tag=value (target near start) ===\n");
    println!("{:<30} {:>8} {:>8} {:>8}", "Scenario", "p50", "p99", "p999");
    println!("{}", "-".repeat(58));

    // 1-digit tag: "8=..."
    let field_1d = b"8=FIX.4.4\x01";
    let (p50, p99, p999) = benchmark(|| scan::find_eq(black_box(field_1d.as_slice()), 0));
    println!(
        "{:<30} {:>8} {:>8} {:>8}",
        "1-digit tag (8=)", p50, p99, p999
    );

    // 2-digit tag: "35=..."
    let field_2d = b"35=D\x01";
    let (p50, p99, p999) = benchmark(|| scan::find_eq(black_box(field_2d.as_slice()), 0));
    println!(
        "{:<30} {:>8} {:>8} {:>8}",
        "2-digit tag (35=)", p50, p99, p999
    );

    // 3-digit tag: "150=..."
    let field_3d = b"150=2\x01";
    let (p50, p99, p999) = benchmark(|| scan::find_eq(black_box(field_3d.as_slice()), 0));
    println!(
        "{:<30} {:>8} {:>8} {:>8}",
        "3-digit tag (150=)", p50, p99, p999
    );

    // 4-digit tag: "5592=..."
    let field_4d = b"5592=CUSTOM\x01";
    let (p50, p99, p999) = benchmark(|| scan::find_eq(black_box(field_4d.as_slice()), 0));
    println!(
        "{:<30} {:>8} {:>8} {:>8}",
        "4-digit tag (5592=)", p50, p99, p999
    );

    // =========================================================================
    // Realistic: scan all SOH delimiters in a FIX NewOrderSingle
    // =========================================================================

    println!("\n=== Realistic: scan all SOH in NewOrderSingle ===\n");

    let msg = b"8=FIX.4.4\x019=120\x0135=D\x0149=SENDER\x0156=TARGET\x01\
                34=42\x0152=20260530-12:00:00.000\x0111=order-001\x01\
                55=BTC-USD\x0154=1\x0138=1.50000000\x0140=2\x01\
                44=67500.00\x0159=0\x0110=178\x01";

    let msg_len = msg.len();
    let (p50, p99, p999) = benchmark(|| {
        let buf = black_box(msg.as_slice());
        let mut pos = 0;
        let mut count = 0u64;
        while let Some(soh) = scan::find_soh(buf, pos) {
            count += 1;
            pos = soh + 1;
        }
        count
    });

    println!("  Message length: {} bytes, 15 fields", msg_len);

    println!("\n  find_soh loop (re-scan per call):");
    println!("    p50={} p99={} p999={} cycles", p50, p99, p999);
    println!(
        "    {:.1} cycles/field, {:.2} cycles/byte",
        p50 as f64 / 15.0,
        p50 as f64 / msg_len as f64
    );

    let (p50_iter, p99_iter, p999_iter) = benchmark(|| {
        let buf = black_box(msg.as_slice());
        scan::soh_iter(buf, 0).count() as u64
    });

    println!("\n  soh_iter (mask-cached):");
    println!(
        "    p50={} p99={} p999={} cycles",
        p50_iter, p99_iter, p999_iter
    );
    println!(
        "    {:.1} cycles/field, {:.2} cycles/byte",
        p50_iter as f64 / 15.0,
        p50_iter as f64 / msg_len as f64
    );

    // =========================================================================
    // FieldReader: fused scan + tag parse + checksum
    // =========================================================================

    println!("\n=== FieldReader: fused scan + tag + checksum ===\n");

    let (p50_parse, p99_parse, p999_parse) = benchmark(|| {
        let buf = black_box(msg.as_slice());
        let mut parser = FieldReader::new(buf, 0);
        let mut count = 0u64;
        while parser.next_field().is_some() {
            count += 1;
        }
        black_box(parser.checksum());
        count
    });

    println!("  FieldReader (scan + tag parse + checksum):");
    println!(
        "    p50={} p99={} p999={} cycles",
        p50_parse, p99_parse, p999_parse
    );
    println!(
        "    {:.1} cycles/field, {:.2} cycles/byte",
        p50_parse as f64 / 15.0,
        p50_parse as f64 / msg_len as f64
    );

    println!(
        "\n  Overhead vs soh_iter: {} cycles ({:.1}%)",
        p50_parse.saturating_sub(p50_iter),
        if p50_iter > 0 {
            (p50_parse.saturating_sub(p50_iter)) as f64 / p50_iter as f64 * 100.0
        } else {
            0.0
        }
    );

    // =========================================================================
    // FieldWriter: encode a full NewOrderSingle (write hot path)
    // =========================================================================

    println!("\n=== FieldWriter: encode NewOrderSingle ===\n");

    // Same 15-field NewOrderSingle the reader decodes above.
    let fields: &[(u32, &[u8])] = &[
        (8, b"FIX.4.4"),
        (9, b"120"),
        (35, b"D"),
        (49, b"SENDER"),
        (56, b"TARGET"),
        (34, b"42"),
        (52, b"20260530-12:00:00.000"),
        (11, b"order-001"),
        (55, b"BTC-USD"),
        (54, b"1"),
        (38, b"1.50000000"),
        (40, b"2"),
        (44, b"67500.00"),
        (59, b"0"),
        (10, b"178"),
    ];
    let n_fields = fields.len();

    let mut out = [0u8; 256];
    let (p50_enc, p99_enc, p999_enc) = benchmark(|| {
        let mut w = FieldWriter::wrap(&mut out);
        for &(tag, val) in black_box(fields) {
            w.field(tag, val);
        }
        // black_box the written bytes (not just the cursor) so the stores
        // are observably live and cannot be dead-code-eliminated.
        black_box(w.data()).len()
    });
    let enc_len = {
        let mut w = FieldWriter::wrap(&mut out);
        for &(tag, val) in fields {
            w.field(tag, val);
        }
        w.pos()
    };

    println!("  FieldWriter (encode {} fields):", n_fields);
    println!(
        "    p50={} p99={} p999={} cycles",
        p50_enc, p99_enc, p999_enc
    );
    println!(
        "    {:.1} cycles/field, {:.2} cycles/byte",
        p50_enc as f64 / n_fields as f64,
        p50_enc as f64 / enc_len as f64
    );

    // =========================================================================
    // checksum(): standalone byte-sum over message body (encode-side tag 10)
    // =========================================================================

    println!("\n=== checksum(): standalone body sum ===\n");
    println!(
        "{:<30} {:>8} {:>8} {:>8} {:>10}",
        "Length", "p50", "p99", "p999", "cyc/byte"
    );
    println!("{}", "-".repeat(70));

    for &len in &[16usize, 32, 64, 128, 143, 256, 512] {
        let buf = vec![b'5'; len];
        let (p50, p99, p999) = benchmark(|| checksum(black_box(&buf)));
        println!(
            "{:<30} {:>8} {:>8} {:>8} {:>10.2}",
            format!("{}B", len),
            p50,
            p99,
            p999,
            p50 as f64 / len as f64
        );
    }
}

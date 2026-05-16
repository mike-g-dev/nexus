use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use nexus_decimal::Decimal;

type D32 = Decimal<i32, 4>;
type D64 = Decimal<i64, 8>;
type D96 = Decimal<i128, 12>;
type D128 = Decimal<i128, 18>;

// ============================================================================
// Addition
// ============================================================================

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    group.throughput(Throughput::Elements(1));

    group.bench_function("D32", |b| {
        let a = D32::new(123, 4567);
        let b_val = D32::new(89, 1234);
        b.iter(|| std::hint::black_box(a.checked_add(b_val)));
    });

    group.bench_function("D64", |b| {
        let a = D64::new(123, 45_678_900);
        let b_val = D64::new(89, 12_345_600);
        b.iter(|| std::hint::black_box(a.checked_add(b_val)));
    });

    group.bench_function("D96", |b| {
        let a = D96::new(123, 456_789_000_000);
        let b_val = D96::new(89, 123_456_000_000);
        b.iter(|| std::hint::black_box(a.checked_add(b_val)));
    });

    group.bench_function("D128", |b| {
        let a = D128::new(123, 456_789_000_000_000_000);
        let b_val = D128::new(89, 123_456_000_000_000_000);
        b.iter(|| std::hint::black_box(a.checked_add(b_val)));
    });

    group.finish();
}

// ============================================================================
// Multiplication
// ============================================================================

fn bench_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul");
    group.throughput(Throughput::Elements(1));

    group.bench_function("D32", |b| {
        let a = D32::new(12, 3456);
        let b_val = D32::new(7, 8901);
        b.iter(|| std::hint::black_box(a.checked_mul(b_val)));
    });

    group.bench_function("D64", |b| {
        let a = D64::new(123, 45_678_900);
        let b_val = D64::new(7, 89_012_300);
        b.iter(|| std::hint::black_box(a.checked_mul(b_val)));
    });

    group.bench_function("D96", |b| {
        let a = D96::new(123, 456_789_000_000);
        let b_val = D96::new(7, 890_123_000_000);
        b.iter(|| std::hint::black_box(a.checked_mul(b_val)));
    });

    group.bench_function("D128", |b| {
        let a = D128::new(123, 456_789_000_000_000_000);
        let b_val = D128::new(7, 890_123_000_000_000_000);
        b.iter(|| std::hint::black_box(a.checked_mul(b_val)));
    });

    group.finish();
}

// ============================================================================
// Division
// ============================================================================

fn bench_div(c: &mut Criterion) {
    let mut group = c.benchmark_group("div");
    group.throughput(Throughput::Elements(1));

    group.bench_function("D32", |b| {
        let a = D32::new(123, 4567);
        let b_val = D32::new(7, 8901);
        b.iter(|| std::hint::black_box(a.checked_div(b_val)));
    });

    group.bench_function("D64", |b| {
        let a = D64::new(123, 45_678_900);
        let b_val = D64::new(7, 89_012_300);
        b.iter(|| std::hint::black_box(a.checked_div(b_val)));
    });

    group.bench_function("D96", |b| {
        let a = D96::new(123, 456_789_000_000);
        let b_val = D96::new(7, 890_123_000_000);
        b.iter(|| std::hint::black_box(a.checked_div(b_val)));
    });

    group.bench_function("D128", |b| {
        let a = D128::new(123, 456_789_000_000_000_000);
        let b_val = D128::new(7, 890_123_000_000_000_000);
        b.iter(|| std::hint::black_box(a.checked_div(b_val)));
    });

    group.finish();
}

// ============================================================================
// mul_int (price × quantity)
// ============================================================================

fn bench_mul_int(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_int");
    group.throughput(Throughput::Elements(1));

    group.bench_function("D64", |b| {
        let price = D64::new(123, 45_000_000);
        b.iter(|| std::hint::black_box(price.mul_int(100)));
    });

    group.finish();
}

// ============================================================================
// Financial: midpoint
// ============================================================================

fn bench_midpoint(c: &mut Criterion) {
    let mut group = c.benchmark_group("midpoint");
    group.throughput(Throughput::Elements(1));

    group.bench_function("D64", |b| {
        let bid = D64::new(100, 25_000_000);
        let ask = D64::new(100, 75_000_000);
        b.iter(|| std::hint::black_box(bid.midpoint(ask)));
    });

    group.finish();
}

// ============================================================================
// Financial: round_to_tick
// ============================================================================

fn bench_round_to_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("round_to_tick");
    group.throughput(Throughput::Elements(1));

    group.bench_function("D64", |b| {
        let price = D64::new(100, 23_700_000);
        let tick = D64::new(0, 1_000_000); // 0.01
        b.iter(|| std::hint::black_box(price.round_to_tick(tick)));
    });

    group.finish();
}

// ============================================================================
// mul_pow2 — constant n (LLVM should fold to single shift)
// ============================================================================

fn bench_mul_pow2_const(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_pow2_const_n");
    group.throughput(Throughput::Elements(1));

    group.bench_function("D32", |b| {
        let a = D32::new(12, 3456);
        b.iter(|| std::hint::black_box(std::hint::black_box(a).mul_pow2(3)));
    });

    group.bench_function("D64", |b| {
        let a = D64::new(123, 45_678_900);
        b.iter(|| std::hint::black_box(std::hint::black_box(a).mul_pow2(3)));
    });

    group.bench_function("D128", |b| {
        let a = D128::new(123, 456_789_000_000_000_000);
        b.iter(|| std::hint::black_box(std::hint::black_box(a).mul_pow2(3)));
    });

    group.finish();
}

// ============================================================================
// mul_pow2 — variable n (black_box defeats const-folding)
// ============================================================================

fn bench_mul_pow2_var(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_pow2_var_n");
    group.throughput(Throughput::Elements(1));

    group.bench_function("D64", |b| {
        let a = D64::new(123, 45_678_900);
        b.iter(|| {
            let n = std::hint::black_box(3u32);
            std::hint::black_box(std::hint::black_box(a).mul_pow2(n))
        });
    });

    group.finish();
}

// ============================================================================
// checked_mul_pow2 — constant n vs variable n
// ============================================================================

fn bench_checked_mul_pow2(c: &mut Criterion) {
    let mut group = c.benchmark_group("checked_mul_pow2");
    group.throughput(Throughput::Elements(1));

    group.bench_function("D64_const_n", |b| {
        let a = D64::new(123, 45_678_900);
        b.iter(|| std::hint::black_box(std::hint::black_box(a).checked_mul_pow2(3)));
    });

    group.bench_function("D64_var_n", |b| {
        let a = D64::new(123, 45_678_900);
        b.iter(|| {
            let n = std::hint::black_box(3u32);
            std::hint::black_box(std::hint::black_box(a).checked_mul_pow2(n))
        });
    });

    group.finish();
}

// ============================================================================
// div_pow2 — constant n folds to shift + sign-correct; variable n is real div
// ============================================================================

fn bench_div_pow2_const(c: &mut Criterion) {
    let mut group = c.benchmark_group("div_pow2_const_n");
    group.throughput(Throughput::Elements(1));

    group.bench_function("D32", |b| {
        let a = D32::new(12, 3456);
        b.iter(|| std::hint::black_box(std::hint::black_box(a).div_pow2(3)));
    });

    group.bench_function("D64", |b| {
        let a = D64::new(123, 45_678_900);
        b.iter(|| std::hint::black_box(std::hint::black_box(a).div_pow2(3)));
    });

    group.bench_function("D128", |b| {
        let a = D128::new(123, 456_789_000_000_000_000);
        b.iter(|| std::hint::black_box(std::hint::black_box(a).div_pow2(3)));
    });

    group.finish();
}

fn bench_div_pow2_var(c: &mut Criterion) {
    let mut group = c.benchmark_group("div_pow2_var_n");
    group.throughput(Throughput::Elements(1));

    group.bench_function("D64", |b| {
        let a = D64::new(123, 45_678_900);
        b.iter(|| {
            let n = std::hint::black_box(3u32);
            std::hint::black_box(std::hint::black_box(a).div_pow2(n))
        });
    });

    group.finish();
}

// ============================================================================
// checked_abs_diff — overflow-safe |a - b|. Baseline = naive (a-b).abs().
// ============================================================================

fn bench_checked_abs_diff(c: &mut Criterion) {
    let mut group = c.benchmark_group("checked_abs_diff");
    group.throughput(Throughput::Elements(1));

    group.bench_function("D64", |b| {
        let a = D64::new(123, 45_678_900);
        let b_val = D64::new(89, 12_345_600);
        b.iter(|| std::hint::black_box(std::hint::black_box(a).checked_abs_diff(b_val)));
    });

    // Baseline: naive (a - b).abs() — overflows on opposite signs near rails.
    group.bench_function("D64_baseline_naive", |b| {
        let a = D64::new(123, 45_678_900);
        let b_val = D64::new(89, 12_345_600);
        b.iter(|| {
            let diff = std::hint::black_box(a).checked_sub(b_val);
            std::hint::black_box(diff.and_then(D64::checked_abs))
        });
    });

    group.finish();
}

// ============================================================================
// Main
// ============================================================================

criterion_group!(
    benches,
    bench_add,
    bench_mul,
    bench_div,
    bench_mul_int,
    bench_midpoint,
    bench_round_to_tick,
    bench_mul_pow2_const,
    bench_mul_pow2_var,
    bench_checked_mul_pow2,
    bench_div_pow2_const,
    bench_div_pow2_var,
    bench_checked_abs_diff,
);
criterion_main!(benches);

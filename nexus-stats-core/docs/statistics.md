# Statistics

Streaming, fixed-memory, O(1)-per-update statistical primitives. Module path: `nexus_stats_core::statistics`.

## At a glance

| Type | What it computes | Memory |
|------|------------------|--------|
| `WelfordF64` / `WelfordF32` | Mean, variance, stddev | 3 words |
| `MomentsF64` / `MomentsF32` | Mean, variance, skewness, kurtosis (Pebay 2008) | 5 words |
| `EwmaVarF64` / `EwmaVarF32` | Exponentially-weighted mean & variance | small, fixed |
| `CovarianceF64` / `CovarianceF32` | Online covariance + Pearson correlation | 5 words |
| `OnlineCovarianceF64` | N-dimensional covariance matrix | O(N²) |
| `HarmonicMeanF64` / `HarmonicMeanF32` | Harmonic mean | 2 words |
| `PercentileF64` / `PercentileF32` | Online percentile (P² algorithm) | small, fixed |
| `BucketAccumulator` | Bar-style OHLC / VWAP aggregation | small, fixed |
| `AmihudF64` | Amihud illiquidity measure (EW) | small |
| `HitRateF64` | Rolling success rate over labels | small |
| `HalfLifeF64` | Mean-reversion half-life estimator | small |
| `HurstF64` | Hurst exponent (R/S estimator) | alloc, windowed |
| `VarianceRatioF64` | Lo-MacKinlay variance ratio (mean-reversion test) | small |
| `LpmF64` / `LpmF32` | Lower partial moments (downside risk, semivariance) | 5 words |
| `CvarF64` / `CvarF32` | Conditional Value at Risk (expected shortfall) | small, fixed |

---

## WelfordF64 — Online mean / variance / stddev

Numerically stable single-pass computation of sample mean and variance using Welford's algorithm. No catastrophic cancellation. Supports merging partial results via Chan's algorithm for parallel aggregation.

```rust
use nexus_stats_core::statistics::WelfordF64;

let mut w = WelfordF64::new();
for latency_us in [120.0, 118.0, 125.0, 130.0, 119.0] {
    w.update(latency_us).unwrap();
}

let mean = w.mean().unwrap();        // ~ 122.4
let var  = w.variance().unwrap();    // sample variance (n-1)
let std  = w.std_dev().unwrap();
let n    = w.count();
```

**Use for:** running mean/variance/stddev on any float stream. The default first choice when you need "how noisy is this?".

**Caveats:** sample variance (divisor `n-1`), not population. If you need population variance, compute `var * (n-1)/n` yourself. Primed after 2 samples.

---

## MomentsF64 — Skewness and kurtosis

Extends Welford to third and fourth central moments using Pébay's online formulas. Numerically stable, single pass.

```rust
use nexus_stats_core::statistics::MomentsF64;

let mut m = MomentsF64::new();
for r in returns { m.update(r).unwrap(); }

let skew      = m.skewness().unwrap();         // asymmetry, 0 = symmetric
let kurt      = m.kurtosis().unwrap();         // classical (normal = 3)
let excess    = m.excess_kurtosis().unwrap();  // normal = 0
```

**Use for:** return distributions, latency distribution shape, detecting fat tails. `excess_kurtosis > 3` is a serious tail warning.

**Caveats:** higher moments need more samples to stabilize. Don't trust skewness below ~30 samples, kurtosis below ~100.

---

## EwmaVarF64 — Exponentially-weighted variance

Mean and variance with an EMA-style forgetting factor. Adapts to drift unlike `WelfordF64`.

```rust
use nexus_stats_core::statistics::EwmaVarF64;

let mut ew = EwmaVarF64::builder()
    .halflife(100.0)
    .build()
    .unwrap();

for x in stream { ew.update(x).unwrap(); }
let (mean, var) = (ew.mean().unwrap(), ew.variance().unwrap());
```

**Use for:** adaptive z-scores, volatility bands on a drifting mean, anywhere you want `Welford` but with decay.

**Caveats:** `variance()` is the EW sample variance, which is biased low under finite memory — corrections exist but are rarely worth the cycles.

---

## CovarianceF64 — Online covariance and correlation

Tracks `cov(X, Y)` and `corr(X, Y)` in one pass.

```rust
use nexus_stats_core::statistics::CovarianceF64;

let mut c = CovarianceF64::new();
for (bid_change, ask_change) in book_updates {
    c.update(bid_change, ask_change).unwrap();
}

let cov  = c.covariance().unwrap();
let corr = c.correlation().unwrap();
```

**Use for:** pairs trading, book side cohesion, any two-stream relationship measurement.

---

## OnlineCovarianceF64 — N-dim covariance matrix

Builder-configured dimension `N`, streaming observation vectors of length `N`.

```rust
use nexus_stats_core::statistics::OnlineCovarianceF64;

let mut cov = OnlineCovarianceF64::builder().dimensions(3).build().unwrap();
for observation in observations {
    cov.update(&observation).unwrap();  // &[f64] of length 3
}
// query cov.covariance_matrix() / cov.correlation_matrix() etc.
```

**Use for:** multi-asset portfolios, multi-factor models, PCA inputs.

**Caveats:** O(N²) memory and O(N²) per update. Fine for small N (2-20), overkill for N > 50. Requires `alloc`.

---

## HarmonicMeanF64 — Correct average of rates

`H = n / sum(1/x_i)`. The right mean for ratios, rates, and latencies when you care about throughput semantics.

```rust
use nexus_stats_core::statistics::HarmonicMeanF64;

let mut h = HarmonicMeanF64::new();
for rate_mbps in throughputs { h.update(rate_mbps).unwrap(); }
let effective = h.value().unwrap();
```

**Use for:** "what's the effective throughput across these links?" (classical HM use case). Rarely the right answer for latency — arithmetic mean over latency is usually what you want.

**Caveats:** requires all inputs strictly positive. Rejects zero and negatives.

---

## PercentileF64 — Online percentile (P²)

The P² algorithm: maintains 5 markers that track a target percentile in O(1) memory. No window, no allocation, adapts over time.

```rust
use nexus_stats_core::statistics::PercentileF64;

let mut p99 = PercentileF64::new(0.99).unwrap();
for latency in latencies { p99.update(latency).unwrap(); }

if p99.is_primed() {
    let estimate = p99.percentile().unwrap();
}
```

**Use for:** p50/p95/p99/p999 estimation on a continuous stream. No storage costs.

**Caveats:** approximate. Accuracy depends on how many samples you've fed it — p50 primes at ~5 samples, p99 at ~100, p999 at ~1000. See `is_primed()`. For exact percentiles on bounded data, use histograms.

---

## BucketAccumulator — Bar-style aggregation

Accumulates samples into time-bucketed bars (OHLC, VWAP, count, sum). Emits a `BucketSummary` when the bucket rolls.

```rust
use nexus_stats_core::statistics::BucketAccumulator;

let mut bucket = BucketAccumulator::builder()
    // parameters here, see source
    .build()
    .unwrap();

for tick in ticks {
    if let Some(Some(summary)) = Some(bucket.update_volume(tick.price, tick.volume).ok()) {
        // bucket rolled, summary is the finished bar
    }
}
```

**Use for:** building minute bars, hour bars, volume bars from a tick stream without allocating.

---

## AmihudF64 — Illiquidity estimator

Streaming (EW-smoothed) Amihud illiquidity: average of `|return| / dollar_volume`. Higher = less liquid.

```rust
use nexus_stats_core::statistics::AmihudF64;
let mut a = AmihudF64::builder().halflife(500.0).build().unwrap();
a.update(abs_return, dollar_volume).unwrap();
let score = a.value().unwrap();
```

**Use for:** liquidity monitoring on crypto venues. Cross-venue comparison. Signal input to smart order routing.

---

## HitRateF64 — Rolling success rate

EW rolling fraction-of-labels-true. Like `ErrorRate` but generalized.

```rust
use nexus_stats_core::statistics::HitRateF64;
let mut hr = HitRateF64::builder().halflife(200.0).build().unwrap();
hr.update(fill_happened, target_shares).unwrap();
let rate = hr.value().unwrap();
```

---

## HalfLifeF64 — Mean-reversion half-life

Fits a linear regression `Δx_t = α + β*(x_{t-1} - μ)` in a streaming way. If `β < 0`, the series is mean-reverting and the half-life is `ln(2) / -β`.

```rust
use nexus_stats_core::statistics::HalfLifeF64;
let mut hl = HalfLifeF64::builder().build();
for spread in spreads { hl.update(spread).unwrap(); }

if let Some(h) = hl.half_life() {
    println!("half life = {h:.1} samples");
}
```

**Use for:** stat arb, pairs trading calibration — if half-life is longer than your holding period, the trade isn't mean-reverting fast enough.

---

## HurstF64 — Hurst exponent

Rescaled-range estimator. Classifies a series:

- `H = 0.5` → random walk.
- `H < 0.5` → mean-reverting.
- `H > 0.5` → persistent / trending.

```rust
use nexus_stats_core::statistics::HurstF64;
let mut h = HurstF64::builder().build().unwrap();
for x in series { h.update(x).unwrap(); }
let estimate = h.hurst().unwrap();
```

**Use for:** regime detection, signal-vs-noise classification on slow timescales.

**Caveats:** R/S estimator is noisy on small samples. Requires `alloc`.

---

## VarianceRatioF64 — Lo-MacKinlay variance ratio test

`VR(k) = Var(k-period return) / (k * Var(1-period return))`. Under a random walk VR = 1; VR < 1 indicates mean-reversion, VR > 1 indicates momentum/persistence.

```rust
use nexus_stats_core::statistics::VarianceRatioF64;
let mut vr = VarianceRatioF64::builder().build().unwrap();
for price in prices { vr.update(price).unwrap(); }
let ratio = vr.ratio().unwrap();
```

**Use for:** quantitative mean-reversion testing, complement to `HalfLifeF64` and `HurstF64`.

---

## LpmF64 — Lower Partial Moments (downside risk)

Measures deviations below a target threshold, raised to a configurable integer order. Fishburn (1977), Sortino & van der Meer (1991).

- **Order 0**: shortfall probability (fraction of samples below target)
- **Order 1**: expected shortfall (mean distance below target)
- **Order 2**: semivariance (variance of downside deviations only)

```rust
use nexus_stats_core::statistics::LpmF64;

// Semivariance (order 2) with target = 0.0
let mut lpm = LpmF64::semivariance(0.0).unwrap();
for &v in &[-3.0, -1.0, 0.0, 2.0, 5.0] {
    lpm.update(v).unwrap();
}
let sv = lpm.lpm().unwrap();  // (9 + 1 + 0 + 0 + 0) / 5 = 2.0

// Shortfall probability (order 0)
let mut sp = LpmF64::builder().target(50.0).order(0).build().unwrap();
for i in 0..100 { sp.update(i as f64).unwrap(); }
let prob = sp.lpm().unwrap();  // ~0.5
```

**Use for:** downside risk measurement, Sortino ratio inputs, tail-sensitive risk metrics. Semivariance is the most common application — use the `semivariance(target)` convenience constructor.

**Caveats:** orders >= 3 use a manual multiply loop (`powi` unavailable in no_std). Fine in practice — nobody runs order > 3.

---

## CvarF64 — Conditional Value at Risk (Expected Shortfall)

Streaming CVaR: the average loss in the worst α-tail, estimated online using a P² percentile estimator for VaR.

```rust
use nexus_stats_core::statistics::CvarF64;

let mut cvar = CvarF64::builder().alpha(0.05).build().unwrap();
for i in 0..2000 {
    cvar.update((i % 1000 + 1) as f64).unwrap();
}
if cvar.is_primed() {
    let var  = cvar.var().unwrap();   // 5th percentile estimate
    let es   = cvar.cvar().unwrap();  // mean of samples <= VaR
}
```

**Use for:** tail risk measurement, position sizing, drawdown budgeting. More informative than VaR alone — tells you how bad the bad cases are, not just where the threshold is.

**Caveats:** tail tracking starts after P² primes (~5 samples). During P² convergence the VaR estimate is evolving, so early tail classifications may be slightly off. Not windowed — streaming with a priming phase. For windowed tail risk, compose with a ring buffer upstream.

---

## Cross-references

- Smoothers on top of statistics: [`smoothing.md`](smoothing.md), [`nexus-stats-smoothing`](../../nexus-stats-smoothing/docs/INDEX.md).
- Change detection built on statistics: [`detection.md`](detection.md), [`nexus-stats-detection`](../../nexus-stats-detection/docs/INDEX.md).
- Regression and Kalman estimators: [`nexus-stats-regression`](../../nexus-stats-regression/docs/INDEX.md).

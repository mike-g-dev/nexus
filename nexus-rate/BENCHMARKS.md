# Benchmarks

Performance measurements for nexus-rate (GCRA, Token Bucket, Sliding
Window — local + sync variants).

## Running Benchmarks

```bash
# Build release benches
cargo build -p nexus-rate --benches --release

# Run with CPU pinning (single physical core for these per-call benches)
# Check your topology with: lscpu -e
taskset -c 0 ./target/release/deps/perf_rate-*

# For more stable results, disable turbo boost:
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# Re-enable after:
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Hardware counters (requires perf):
sudo perf stat -r 5 taskset -c 0 ./target/release/deps/perf_rate-*
```

## Methodology

- **Latency**: Per-call latency for `try_acquire(cost, now)` on the hot
  path (token-available path; the rejection path is also measured).
- **Timing**: rdtscp instruction for cycle-accurate measurement.
- **Histogram**: hdrhistogram for percentile distribution.
- **Warmup**: 10,000 iterations before measurement.
- **Samples**: 100,000 latency samples.

The rate primitives are designed for the success-case fast path. We
don't measure throughput here — these aren't queue primitives. The
relevant question is "how many cycles does a single `try_acquire` cost
when the limiter has tokens available?" Sub-bucket arithmetic (GCRA's
single timestamp update, token bucket's lazy refill) should keep this
in the low-single-digit cycle range.

## Baseline Results

Intel Core Ultra 7 165U (hybrid P+E cores), 2.69 GHz base clock, Linux 6.18.
Pinned to physical P-cores via `taskset -c 0,2`. Best-of-5 floor per
percentile, batch=64 amortization. **Turbo on**, no other manipulation.

The bench measures realistic per-call cost — `try_acquire(cost, now)` with
`now` constructed inside the timed window via `Instant + Duration::from_nanos(t)`.
This adds a few cycles of timestamp-construction overhead to the
algorithm-only cost, but reflects what users actually pay per call.

### `try_acquire` (cycles, success path)

| Variant | p50 | p99 | p999 |
|---------|-----|-----|------|
| **local::Gcra** | 13 | 21 | 32 |
| **local::TokenBucket** | 12 | 13 | 24 |
| **local::SlidingWindow** | 15 | 18 | 37 |
| **sync::Gcra** | 24 | 26 | 79 |
| **sync::TokenBucket** | 29 | 33 | 88 |
| **sync::SlidingWindow** | (not exercised by bench) | | |

### `try_acquire` (cycles, rejection path)

| Variant | p50 | p99 | p999 |
|---------|-----|-----|------|
| **local::Gcra** | 16 | 21 | 40 |
| **local::TokenBucket** | 11 | 12 | 25 |
| **local::SlidingWindow** | 11 | 14 | 26 |
| **sync::Gcra** | 12 | 19 | 34 |
| **sync::TokenBucket** | 13 | 16 | 30 |

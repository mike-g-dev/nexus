# Benchmarks

Performance measurements for nexus-timer (hierarchical timer wheel).

## Running Benchmarks

```bash
# Build release benches
cargo build -p nexus-timer --benches --release

# Run with CPU pinning
# Check your topology with: lscpu -e
taskset -c 0 ./target/release/deps/perf_timer-*

# For more stable results, disable turbo boost:
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# Re-enable after:
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Hardware counters (requires perf):
sudo perf stat -r 5 taskset -c 0 ./target/release/deps/perf_timer-*
```

## Methodology

- **Latency**: Per-operation latency for the timer-wheel hot paths:
  - `schedule(deadline)` — insert a timer
  - `cancel(handle)` — cancel an in-flight timer
  - `poll(now)` — drain expired timers
  - `schedule + cancel` paired round trip (the common short-lived case)
- **Timing**: `Instant::now()` deltas for the schedule/cancel/poll
  per-op measurements (sub-microsecond resolution from the kernel
  `clock_gettime` fast path); sample sizes large enough that small
  per-op overhead amortizes well.
- **Warmup**: 5,000 iterations before measurement.
- **Samples**: 50,000 per operation. Steady-state population: 100,000
  pending timers.
- **Poll batch**: 100 expirations per poll measurement.

## Baseline Results

Intel Core Ultra 7 165U (hybrid P+E cores), 2.69 GHz base clock, Linux 6.18.
Pinned to physical P-cores via `taskset -c 0,2`. Best-of-5 floor per
percentile. **Turbo on**, no other manipulation.

### Timer wheel hot paths (cycles)

| Operation | p50 | p99 | p999 | Notes |
|-----------|-----|-----|------|-------|
| **schedule** (`schedule_forget`) | 40 | 46 | 74 | insert into level-0 slot |
| **cancel** (pre-scheduled) | 26 | 34 | 62 | O(1) handle-based |
| **poll** (per expired, 100 batch) | 7 | 13 | 24 | amortized over batch |
| **schedule + cancel** (paired) | 50 | 68 | 78 | paired short-lived |
| **poll** (empty wheel) | 44 | 48 | 50 | level-mask scan, no expirations |

### Population sensitivity

Schedule/cancel cost is independent of pending population (slab-backed,
intrusive list per slot). Confirmed empirically: `schedule + cancel`
at @100k active timers measured **p50=50** — identical to the empty-wheel
case, within noise. Poll cost is proportional to expirations returned,
not to total population.

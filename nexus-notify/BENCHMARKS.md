# nexus-notify Benchmarks

All cycle measurements use `rdtsc` with `lfence`/`rdtscp` serialization.
Criterion benchmarks provide detailed statistical analysis of performance.

## System

- CPU: AMD (hybrid, P-cores + E-cores)
- OS: Arch Linux 6.18.2
- Rust: stable 1.90+
- Profile: `--release`

## Event Queue (non-blocking)

### Notify latency (p50 cycles)

| Operation | Cycles | Notes |
|-----------|--------|-------|
| notify (new) | 16 | flag swap + CAS push |
| notify (conflated) | 16 | flag swap only |

### Poll latency (p50 cycles, cap=4096)

| N ready | Cycles | cy/token |
|---------|--------|----------|
| 0 (empty) | 2 | — |
| 1 | 28 | 28.0 |
| 8 | 48 | 6.0 |
| 32 | 178 | 5.6 |
| 64 | 344 | 5.4 |
| 128 | 684 | 5.3 |
| 256 | 1336 | 5.2 |
| 512 | 2980 | 5.8 |
| 1024 | 5028 | 4.9 |
| 4096 | 17236 | 4.2 |

Per-token cost converges to ~5 cycles. Sequential array access (ring
buffer slots + flag array) keeps the prefetcher happy.

### Poll limit (p50 cycles, cap=4096, all 4096 ready)

| Limit | Cycles | cy/token |
|-------|--------|----------|
| 32 | 162 | 5.1 |
| 64 | 278 | 4.3 |
| 128 | 520 | 4.1 |
| 256 | 1016 | 4.0 |
| 512 | 2228 | 4.4 |

O(limit) — cost scales with the limit, not capacity.

### Cross-thread (p50 cycles)

| Operation | Cycles | Notes |
|-----------|--------|-------|
| roundtrip/2 | 362 | ~100ns @ 3.5GHz |
| contended P=1 | 124 | single producer |
| contended P=2 | 108 | two producers |
| contended P=4 | 136 | four producers |

### Memory

For `max_tokens = 4096`:
- Flags: 4 KB (one `AtomicBool` per token)
- MPSC queue: 64 KB (4096 slots × 16 bytes, rounded to power-of-two)
- Total: ~68 KB

## Design Evolution

Four implementations were benchmarked before arriving at the final design:

### Comparison (p50 cycles, cap=4096)

| Metric | Bitmap | Treiber LIFO | Treiber FIFO | **MPSC Queue** |
|--------|--------|-------------|-------------|----------------|
| notify | 14 | 13 | 13 | **16** |
| poll empty | 110 | 14 | 14 | **2** |
| poll N=32 | 414 | 188 | 434 | **178** |
| poll N=128 | 840 | 446 | 830 | **684** |
| poll N=4096 | 5724 | 13268 | 25340 | **17236** |
| poll_limit=32 | n/a | ~11k | ~12k | **162** |
| FIFO | no | no | yes (2x cost) | **yes (native)** |
| O(limit) | no | no | no | **yes** |

### Why MPSC queue won

1. **Bitmap** — O(capacity/64) poll regardless of readiness. Bad for
   poll_limit. No FIFO.

2. **Treiber LIFO** — O(ready) but wrong ordering. Hot symbols starve
   cold ones under poll_limit.

3. **Treiber FIFO** — Correct ordering via LIFO→FIFO reversal, but
   the reversal doubles the walk cost (2x regression at all N).

4. **MPSC queue + dedup flags** — FIFO native, O(limit) poll_limit,
   sequential array access. The Akka pattern (separate dedup from
   delivery) is the right decomposition.

## Methodology

### rdtsc benchmarks (`examples/bench_ready_set.rs`)

- 50,000 samples per benchmark, 5,000 warmup
- `rdtsc_start()`: `lfence` + `rdtsc` (serialized read)
- `rdtsc_end()`: `rdtscp` + `lfence` (serialized write)
- Percentiles: p50, p90, p99, p99.9, p99.99, max
- Unrolled 100x for single-op measurements (notify, poll_empty)
- Pin to physical cores with `taskset -c 0,2` for cross-thread

### Criterion benchmarks (`benches/event_queue.rs`)

- Standard criterion statistical analysis
- Shuffled notify order for realistic cache patterns
- Deterministic seed for reproducibility

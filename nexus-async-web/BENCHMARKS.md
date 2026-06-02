# Benchmarks

Performance measurements for nexus-async-web — async WebSocket and REST
adapters over both `tokio` and `nexus-async-rt` runtimes.

## Running Benchmarks

The crate ships **two mutually-exclusive runtime backends** (`tokio-rt`,
`nexus`). Each bench requires the corresponding feature; build the
backends separately.

```bash
# Build release benches (tokio backend, default)
cargo build -p nexus-async-web --benches --release

# Build release benches (nexus-async-rt backend)
cargo build -p nexus-async-web --benches --release \
    --no-default-features --features nexus,tls

# Run with CPU pinning (separate physical cores for client/server)
# Check your topology with: lscpu -e
taskset -c 0,2 ./target/release/deps/perf_ws_cycles_tokio-*
taskset -c 0,2 ./target/release/deps/perf_ws_cycles_nexus-*
taskset -c 0,2 ./target/release/deps/perf_async_ws-*
taskset -c 0,2 ./target/release/deps/perf_async_ws_nexus-*
taskset -c 0,2 ./target/release/deps/bench_pool_throughput-*

# For more stable results, disable turbo boost:
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# Re-enable after:
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

**Important**: Use separate physical cores (e.g., 0,2) not hyperthreads
(e.g., 0,1). Hyperthreads share L1/L2 cache and give artificially low
latency numbers.

## Methodology

- **Latency** (`perf_ws_cycles_*`): Per-message rdtscp-measured latency
  on a loopback WebSocket connection. Captures parser dispatch + waker
  plumbing + socket round-trip cost.
- **Throughput** (`perf_async_ws*`, `bench_pool_throughput`):
  Wall-clock messages/sec under sustained load. Measures the steady-
  state cost amortized across batched syscalls.
- **Histogram**: hdrhistogram for percentile distribution where
  applicable.
- **Warmup**: 10,000 iterations before measurement on cycle benches.
- **Samples**: 100,000 cycle samples; ≥1M messages for throughput
  benches.

The two backends measure the same protocol-layer work; the differences
that surface are runtime cost (waker, scheduler dispatch, syscall
batching) under each model.

## Baseline Results

Intel Core Ultra 7 165U (hybrid P+E cores), 2.69 GHz base clock, Linux 6.18.
Pinned to separate physical P-cores via `taskset -c 0,2`. Best-of-5
floor per percentile. **Turbo on**, no other manipulation.

### WebSocket recv per-message latency (cycles, 128B binary, in-memory)

Source: `perf_ws_cycles_tokio` and `perf_ws_cycles_nexus`. In-memory
mock transport — no kernel, no TLS. Pure protocol-layer cost.

| Backend | p50 | p99 | p999 |
|---------|-----|-----|------|
| **nexus-async-web (tokio)** | 64 | 188 | 224 |
| **nexus-async-web (nexus-async-rt)** | 64 | 184 | 206 |
| tokio-tungstenite | (not measured by perf_ws_cycles — see throughput table below) | | |

### WebSocket send per-message latency (cycles, 128B text, in-memory)

| Backend | p50 | p99 | p999 |
|---------|-----|-----|------|
| **nexus-async-web (tokio)** | 96 | 228 | 266 |
| **nexus-async-web (nexus-async-rt)** | 84 | 218 | 246 |

### WebSocket recv per-message latency, batched x64 (amortized cycles/msg)

| Backend | Payload | p50 | p99 | p999 |
|---------|---------|-----|-----|------|
| nexus-async-web (tokio) | 40B | 48 | 72 | 146 |
| nexus-async-web (tokio) | 128B | 55 | 68 | 188 |
| nexus-async-web (nexus-async-rt) | 40B | 43 | (varies) | (varies) |
| nexus-async-web (nexus-async-rt) | 128B | 52 | (varies) | (varies) |

### Throughput — TLS loopback (ns/msg, real syscalls + TLS)

Source: `perf_async_ws` TLS Loopback section. Median of 3 runs,
`tls-0.7-refactor` branch state.

| Path | 40B | 128B |
|------|-----|------|
| nexus-async-web (tokio + TLS) | 54 ns/msg | 134 ns/msg |
| nexus-net (blocking + TLS, sync `TlsStream`) | 37 ns/msg | 86 ns/msg |
| tokio-tungstenite (+ TLS) | 177 ns/msg | 280 ns/msg |

The blocking sync path is 1.5–2× faster than the async path on TLS
loopback because it avoids the runtime's task-dispatch overhead on
every recv. Both nexus paths are 3–5× faster than tokio-tungstenite.

### REST connection-pool throughput

Not yet measured — `bench_pool_throughput` is a tokio-against-remote-endpoint
demo, not a pinned cycle bench. To be measured in a follow-up pass with
a local mock REST endpoint.

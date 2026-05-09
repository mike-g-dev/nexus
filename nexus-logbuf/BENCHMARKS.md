# nexus-logbuf Benchmarks

All measurements on AMD Ryzen / Intel Core (adjust for your hardware).
Run in release mode with turbo boost disabled for stable results.

## Running Benchmarks

```bash
# Build
cargo build --release -p nexus-logbuf --bench perf_spsc_latency
cargo build --release -p nexus-logbuf --bench perf_mpsc_latency
cargo build --release -p nexus-logbuf --example throughput_bench

# Run (pin to physical cores, disable turbo for consistency)
./target/release/deps/perf_spsc_latency-*
./target/release/deps/perf_mpsc_latency-*
./target/release/examples/throughput_bench
```

## Queue Layer Latency

### SPSC (queue::spsc)

Single-thread, no contention. Measures raw primitive performance.

| Payload | Producer p50 | Producer p99 | Consumer p50 | Consumer p99 |
|---------|--------------|--------------|--------------|--------------|
| 8 bytes | 38 cycles | 40 cycles | 42 cycles | 44 cycles |
| 64 bytes | 40 cycles | 64 cycles | 26 cycles | 50 cycles |
| 256 bytes | 24 cycles | 52 cycles | 30 cycles | 46 cycles |
| 1024 bytes | 44 cycles | 64 cycles | 40 cycles | 70 cycles |

**Round-trip latency** (ping-pong between threads, one-way estimate):

| Payload | p50 | p99 | p999 |
|---------|-----|-----|------|
| 8 bytes | 169 cycles | 283 cycles | 392 cycles |
| 64 bytes | 191 cycles | 357 cycles | 523 cycles |
| 256 bytes | 269 cycles | 484 cycles | 688 cycles |
| 1024 bytes | 576 cycles | 942 cycles | 1366 cycles |

### MPSC (queue::mpsc)

**Single producer (no contention baseline):**

| Payload | Producer p50 | Producer p99 | Consumer p50 | Consumer p99 |
|---------|--------------|--------------|--------------|--------------|
| 8 bytes | 40 cycles | 62 cycles | 26 cycles | 36 cycles |
| 64 bytes | 42 cycles | 58 cycles | 28 cycles | 42 cycles |
| 256 bytes | 46 cycles | 80 cycles | 34 cycles | 70 cycles |

**Under contention:**

| Payload | 2 Producers p50 | 2 Producers p99 | 4 Producers p50 | 4 Producers p99 |
|---------|-----------------|-----------------|-----------------|-----------------|
| 8 bytes | 148 cycles | 1248 cycles | 314 cycles | 1750 cycles |
| 64 bytes | 144 cycles | 1162 cycles | 340 cycles | 2819 cycles |
| 256 bytes | 166 cycles | 1280 cycles | 254 cycles | 2227 cycles |

Contention increases tail latency significantly. This is expected for CAS-based
multi-producer coordination. If multiple threads are on the hot path, consider
per-thread SPSC buffers instead.

## Throughput

Cross-thread throughput with random message sizes (8-1024 bytes), 2MB buffer.

### SPSC

| Metric | Value |
|--------|-------|
| Throughput | **20.7 GB/s** |
| Message rate | 40.2M msgs/sec |

### MPSC

| Producers | Mode | Throughput | Message Rate |
|-----------|------|------------|--------------|
| 2 | Tight spin | 7.27 GB/s | 14.1M msgs/sec |
| 2 | Backoff | 7.16 GB/s | 13.9M msgs/sec |
| 4 | Tight spin | 6.29 GB/s | 12.2M msgs/sec |
| 4 | Backoff | 5.38 GB/s | 10.4M msgs/sec |

Tight spin maximizes throughput at the cost of CPU. Backoff reduces CPU usage
with minimal throughput impact.

## Message Rate (Fixed Payload)

From MPSC latency benchmark, 8-byte payload:

| Producers | Messages/sec |
|-----------|--------------|
| 1 | 60.6M |
| 2 | 19.9M |
| 4 | 16.5M |

## Channel Layer

The channel layer wraps queues with backoff and parking. On the producer hot
path, `try_send()` adds only a disconnection check (~1 atomic load) over the
raw queue `try_claim()`.

The `send()` method spins with backoff when full, never syscalls.

Receiver blocking uses `park_timeout` which does syscall, but receivers are
assumed to be background threads where this is acceptable.

## Methodology

- **Latency**: Measured in CPU cycles using `rdtscp`
- **Warmup**: 10,000 iterations discarded before measurement
- **Samples**: 100,000 iterations per benchmark
- **Histograms**: HDR histogram with 3 significant digits

For accurate results:
1. Disable turbo boost: `echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo`
2. Pin to physical cores: `taskset -c 0,2,4,6 ./benchmark`
3. Avoid hyperthreading siblings (check `lscpu -e`)

## Comparison Notes

These numbers are for the **queue layer** (raw primitives). The channel layer
adds minimal overhead on the producer side. Consumer blocking adds syscall
latency when parking, but this is by design—consumers are background threads.

For context, typical latencies in trading systems:
- Network RTT (same datacenter): 50-100us
- Kernel network stack: 5-20us
- This crate's round-trip: ~100-600 cycles = 30-200ns

The buffer is not the bottleneck.

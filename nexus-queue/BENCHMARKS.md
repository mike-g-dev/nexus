# Benchmarks

Performance measurements for nexus-queue SPSC, MPSC, and SPMC ring buffers.

## Running Benchmarks

```bash
# Build release benches
cargo build -p nexus-queue --benches --release

# Run with CPU pinning (separate physical cores, not hyperthreads)
# Check your topology with: lscpu -e
taskset -c 0,2 ./target/release/deps/bench_spsc-*
taskset -c 0,2 ./target/release/deps/bench_mpsc_pingpong-*
taskset -c 0,2 ./target/release/deps/bench_spmc-*

# For more stable results, disable turbo boost:
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# Re-enable after:
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Hardware counters (requires perf):
sudo perf stat -r 5 taskset -c 0,2 ./target/release/deps/bench_spsc-*
```

**Important**: Use separate physical cores (e.g., 0,2) not hyperthreads (e.g., 0,1).
Hyperthreads share L1/L2 cache and give artificially low latency numbers.

## Methodology

- **Latency**: Ping-pong round-trip time divided by 2 for one-way estimate
- **Timing**: rdtscp instruction for cycle-accurate measurement
- **Histogram**: hdrhistogram for percentile distribution
- **Warmup**: 10,000 iterations before measurement
- **Samples**: 100,000 latency samples
- **Throughput**: 1M messages for SPSC/MPSC, 10M for SPMC fan-out

## Baseline Results

Intel Core Ultra 7 165U (hybrid P+E cores), 2.69 GHz base clock, Linux 6.18.
**Single-socket, single NUMA node.** Pinned to separate physical P-cores (0,2).

### SPSC Latency (one-way, cycles)

| Queue | min | p50 | p99 | p999 | max |
|-------|-----|-----|-----|------|-----|
| **nexus-queue SPSC** | ~180 | 200 | 210 | 266 | ~4k |
| rtrb | ~180 | ~210 | ~240 | ~400 | ~4k |
| crossbeam (MPMC) | ~450 | 520 | 580 | 820 | ~8k |

### SPSC Throughput

| Queue | M msgs/sec | ns/msg |
|-------|------------|--------|
| **nexus-queue SPSC** | 113 | ~9 |

### MPSC Latency (one-way, cycles)

| Queue | p50 | p99 | p999 | Notes |
|-------|-----|-----|------|-------|
| **nexus-queue MPSC** | 180 | 304 | 414 | CAS + turn counter |
| crossbeam ArrayQueue | 522-532 | 574-584 | 817-876 | MPMC overhead |

### MPSC Latency (one-way, nanoseconds)

| Queue | p50 | p99 | p999 |
|-------|-----|-----|------|
| **nexus-queue MPSC** | 67 ns | 113 ns | 154 ns |
| crossbeam ArrayQueue | 195 ns | 213-217 ns | 304-326 ns |

### SPMC Latency (one-way, cycles)

| Queue | p50 | p99 | p999 |
|-------|-----|-----|------|
| **nexus-queue SPMC** | 169 | 325 | 462 |
| crossbeam ArrayQueue | 505 | 567 | 775 |

### SPMC Latency (one-way, nanoseconds)

| Queue | p50 | p99 | p999 |
|-------|-----|-----|------|
| **nexus-queue SPMC** | 63 ns | 121 ns | 172 ns |
| crossbeam ArrayQueue | 187.9 ns | 210.9 ns | 288.3 ns |

### SPMC Throughput

| Consumers | nexus-queue SPMC | crossbeam ArrayQueue | Delta |
|-----------|-----------------|---------------------|-------|
| 1 | 47 M/s | 85 M/s | -45% |
| 2 | 27 M/s | 57 M/s | -53% |
| 4 | 17 M/s | 47 M/s | -63% |

Note: crossbeam wins on sustained-saturation fan-out throughput. See Analysis section for why this tradeoff is acceptable.

## Analysis

### SPSC

**nexus-queue vs rtrb**: Nearly identical performance. Both use the same
cached index design with separate cache lines for head/tail.

**vs crossbeam**: crossbeam's ArrayQueue is MPMC (multi-producer multi-consumer),
requiring CAS operations on every push/pop. SPSC queues avoid this overhead entirely.

### MPSC

**nexus-queue MPSC is 2.6x faster than crossbeam ArrayQueue at p50.**

Key optimizations:
1. **Cached head in Producer** - avoids atomic load when queue not full
2. **Cached slots/mask/shift** - avoids Arc indirection on hot path
3. **Single consumer** - no CAS contention on consumer side (vs crossbeam's MPMC)
4. **Division-free turn calculation** - `>> shift` instead of `/ capacity`
5. **`#[repr(C)]` layout** - hot fields at struct base

The MPSC is ~3% slower than SPSC (202 vs 208 cycles) for the producer side due to
CAS on tail, but this is acceptable for the "producers NOT on hot path" use case.

### SPMC

**nexus-queue SPMC is 2.7x faster than crossbeam ArrayQueue at p50 latency.**

Key optimizations:
1. **Single-writer producer** - no CAS on tail (vs crossbeam's MPMC CAS)
2. **Cached slots/mask/shift** - avoids Arc indirection on hot path
3. **Division-free turn calculation** - `>> shift` instead of `/ capacity`
4. **`#[repr(C)]` layout** - hot fields at struct base

**Throughput tradeoff**: crossbeam is faster in sustained-saturation flooding
benchmarks (queue persistently 100% full). This is a fundamental consequence of
our faster push: when the queue is full, the producer retries faster, keeping
producer and consumer in lock-step on the same cache lines. Crossbeam's MPMC
overhead (SeqCst fence, tail CAS) acts as implicit backoff, reducing cache
coherency traffic during sustained contention.

This tradeoff is acceptable for the intended use case (1 IO thread fanning out
to N parser threads) because:
- The producer has other work between pushes (epoll, reads, frame parsing)
- Sustained 100% saturation means parsers can't keep up — a capacity problem
- Per-message latency is the metric that matters for fan-out delivery time
- For uniform fan-out, N SPSC channels with round-robin assignment is faster
  than a single SPMC queue (zero contention on both sides)

## Notes

- Results vary significantly with core topology - always use separate physical cores
- Hyperthreaded cores (same physical core) share cache and give misleading results
- For accurate comparisons, benchmark on your target production hardware
- Tail latency (p9999+) dominated by OS scheduling and interrupts

## Multi-Socket NUMA Considerations

These benchmarks were run on a **single-socket** system. On multi-socket NUMA
architectures (common in production servers), the benefits of nexus-queue's
design should be **even more pronounced**:

1. **Cached head/slots/mask** - Avoids cross-socket memory accesses on the hot path.
   Reading from a remote NUMA node can cost 100-300ns additional latency.

2. **Separate cache lines for head/tail** - Prevents false sharing across sockets,
   which is particularly expensive when cache coherency traffic must traverse
   the interconnect (QPI/UPI on Intel, Infinity Fabric on AMD).

3. **Local producer state** - Each producer's cached_head stays socket-local,
   only refreshing from shared memory when the cache indicates the queue is full.

For latency-critical production deployments on multi-socket servers, pin producers
and consumers to cores on the same socket when possible, or ensure the queue's
shared memory is allocated on the consumer's local NUMA node.

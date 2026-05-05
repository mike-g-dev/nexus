# nexus-async-rt Benchmarks

## Summary

| Metric | nexus-async-rt | tokio LocalSet | Ratio |
|---|---|---|---|
| Dispatch p50 | 62-68 cy | 146-187 cy | **2.2x faster** |
| Dispatch p99 | 126-138 cy | 213-263 cy | **1.6x faster** |
| TCP echo p50 | ~8,800 cy (~2.5 μs) | ~10,500 cy (~3.0 μs) | 1.2x faster |
| TCP throughput | 8.8 GB/s | — | loopback, 64KB chunks |

## Dispatch Overhead: Async vs Sync

**Question:** What does Rust's async machinery cost compared to manually
polling a handler in a loop?

**Answer:** 3-8 cycles at p50. The async executor adds negligible overhead
over a direct `Handler::run` call.

### Methodology

All measurements use `rdtsc`/`rdtscp` with `lfence` serialization.
Batched: one rdtsc pair wraps 100 iterations, total divided by 100.
This amortizes the ~20 cycle rdtsc floor to ~0.2 cy per sample.

Run with:
```bash
cargo test -p nexus-async-rt --release -- --ignored --nocapture dispatch_latency
```

### Sync vs Async (batched, 100K samples)

All values in **cycles**.

| Path | p50 | p99 | p999 | Description |
|---|---|---|---|---|
| sync `Box<dyn Handler>` (0 params) | 1 | 3 | 5-7 | Pure dispatch — vtable call, no param resolution |
| sync `Box<dyn Handler>` (1 param) | 7-8 | 8-9 | 10-12 | + pointer deref for `ResMut<T>` |
| async poll (IO-woken) | 3 | 5-8 | 8-13 | Task pre-queued by IO driver, executor polls it |
| async poll (self-waking) | 12-16 | 18-25 | 45-66 | Task re-queues itself via `wake_by_ref` |

### Analysis

**Sync 0 params (1 cy)** is the measurement floor — LLVM devirtualizes
when there's a single concrete type behind `Box<dyn Handler>`.

**Async IO-woken (3 cy)** is the production async hot path: a task was
idle, the IO driver queued it, the executor polls it. The 3 cycles cover:
Vec iterate, `is_completed` check, `is_queued` flag clear, data pointer
update on the reusable waker, and the indirect `poll_fn` call.

**Async self-waking (12-16 cy)** adds the cost of `wake_by_ref` inside
the future: TLS read for the ready queue, `is_queued` check, Vec push,
plus waker refcount operations.

## nexus-async-rt vs tokio LocalSet

**Question:** How does our single-threaded runtime compare to tokio's
`LocalSet` on equivalent workloads?

### Pure Dispatch (no IO, no syscalls)

A self-waking task measured with batched rdtsc. Isolates only the
userspace async machinery — no kernel involvement.

```bash
cargo test -p nexus-async-rt --release --test vs_tokio_dispatch -- --ignored --nocapture --test-threads=1
```

| Percentile | nexus-async-rt | tokio LocalSet | Ratio |
|---|---|---|---|
| **p50** | 62-68 cy | 146-187 cy | **2.2x faster** |
| **p90** | 88-99 cy | 174-213 cy | **1.9x faster** |
| **p99** | 126-138 cy | 213-263 cy | **1.6x faster** |
| **p999** | 192-256 cy | 276-395 cy | **1.4x faster** |

The ~80 cycle advantage at p50 comes from:
- No cooperative scheduling budget (tokio checks `coop::budget` per poll)
- No task harness state machine (tokio manages notification/ref-count)
- Double-buffer Vec swap vs VecDeque (eliminates modular arithmetic)
- ReusableWaker hoists Context construction out of the loop
- Zero-alloc wakers (raw pointer, no Arc)

### TCP Echo (with IO, real syscalls)

64-byte messages over loopback TCP. Both runtimes do real `epoll_wait`
and socket read/write.

```bash
cargo test -p nexus-async-rt --release --test vs_tokio -- --ignored --nocapture --test-threads=1
```

| Percentile | nexus-async-rt | tokio LocalSet | Ratio |
|---|---|---|---|
| **p50** | ~8,800 cy (~2.5 μs) | ~10,500 cy (~3.0 μs) | **1.2x faster** |
| **p99** | ~16,000 cy | ~19,000 cy | **1.2x faster** |
| **p999** | ~25,000 cy | ~35,000 cy | **1.4x faster** |

The kernel dominates (~95% of latency). The dispatch advantage is
diluted by the syscall cost, but still measurable. The tail (p999)
shows a larger gap — tokio's cooperative scheduling adds variance.

### TCP Throughput

100 MB over loopback, 64KB write chunks:

| Metric | Value |
|---|---|
| Throughput | 8.8 GB/s |
| Elapsed | ~11 ms |

Close to memory bandwidth — the kernel is essentially memcpy between
send and receive buffers.

## Dispatch Distribution (per-poll histogram)

Detailed per-poll latency distribution from 500K samples. Shows where
cycles are spent.

```bash
cargo test -p nexus-async-rt --release --test dispatch_histo -- --ignored --nocapture
```

### Raw (includes OS noise)

| Percentile | Cycles |
|---|---|
| p50 | 64 cy |
| p90 | 68 cy |
| p99 | 106-116 cy |
| p999 | 2,700 cy |
| p9999 | 5,400 cy |

### Filtered (kernel noise removed, ≤500 cy only)

| Percentile | Cycles | Cause |
|---|---|---|
| p50 | 64 cy | Clean dispatch |
| p90 | 68 cy | Clean dispatch |
| p99 | 106 cy | L1/L2 cache miss (1-2 lines) |
| p999 | 246 cy | Larger cache miss or minor OS noise |
| max | ~500 cy | Userspace ceiling |

**98%+ of polls are 50-70 cycles.** The remaining ~2% are cache
misses from the benchmark's measurement overhead, not the executor.

The p999 raw (2,700 cy) is entirely kernel timer interrupts (~1ms
tick at 3.5 GHz). The p9999 (5,400 cy) is longer preemptions.
These are eliminated with `isolcpus` + `nohz_full` kernel tuning
on production trading boxes.

## What the Executor Does Per Poll

For a single IO-woken task, the hot path:

```
Vec iterate            — load task pointer (sequential, prefetch-friendly)
is_completed check     — 1 byte read at offset 17 (skip completed tasks)
clear is_queued        — 1 byte store at offset 16
update waker data      — 1 pointer store (reusable waker, hoisted setup)
call poll_fn           — indirect call through task header vtable
```

No heap allocation. No HashMap lookup. No VecDeque index math.
The waker vtable and Context are pre-built once per `poll()` call
and reused across all tasks.

## Spawn + Free Overhead

Task spawn allocates into a nexus-slab byte slab (placement new, O(1)).
Task completion frees the slot (O(1) freelist push + O(1) slab tracker
removal). This is a separate concern from dispatch — spawn/free happens
once per task lifetime, not per poll.

| Operation | p50 | Description |
|---|---|---|
| Slab alloc + free (256B) | 8 cy | nexus-slab byte slab, placement new |
| Spawn + poll + free (0B future) | ~30 cy | Full lifecycle for immediate task |
| Spawn + poll + free (64B future) | ~35 cy | Includes 64-byte copy into slab |

## Running Benchmarks

```bash
# Full suite (ignored tests, release mode)
cargo test -p nexus-async-rt --release -- --ignored --nocapture --test-threads=1

# Individual benchmarks
cargo test -p nexus-async-rt --release --test dispatch_histo -- --ignored --nocapture
cargo test -p nexus-async-rt --release --test vs_tokio_dispatch -- --ignored --nocapture --test-threads=1
cargo test -p nexus-async-rt --release --test vs_tokio -- --ignored --nocapture --test-threads=1
cargo test -p nexus-async-rt --release --test net_perf -- --ignored --nocapture --test-threads=1
cargo test -p nexus-async-rt --release --lib -- --ignored --nocapture dispatch_latency
```

All benchmarks require x86_64 (rdtsc). Results vary by CPU —
numbers above measured on a desktop-class processor at ~3.5 GHz.
For controlled measurements, disable turbo boost and pin to a
physical core:

```bash
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
sudo taskset -c 0 cargo test ...
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

---

## tokio_compat: cross-task waker clone (PR 2 §2.1)

PR 2 §2.1 rewrote the tokio bridge's `cross_task_clone` from per-clone
`Box<CrossTaskWakerData>` allocation to `Arc::clone` on a shared
`Arc<CrossTaskWakerInner>`. The old shape allocated on every waker
clone (tokio clones wakers on every IO register and freely during
scheduling). The new shape is one heap allocation at construction
plus atomic refcount bumps per clone.

Measured on the same desktop-class processor:

| Operation              | Pre-§2.1 (Box)     | Post-§2.1 (Arc)    |
|------------------------|--------------------|--------------------|
| `Waker::clone`         | ~50 ns (glibc malloc, plan estimate) | **9 ns** (Arc::clone — measured) |
| `Waker::drop`          | ~25 ns (glibc free, plan estimate)   | **5 ns** (atomic decrement — measured) |

The pre-§2.1 numbers are from the plan's stated estimates (glibc
malloc latency); a controlled before/after measurement requires
running the bench against the pre-§2.1 commit. The post-§2.1
numbers were measured by:

```bash
cargo test -p nexus-async-rt --features tokio-compat --release \
    --lib tokio_compat::arc_tests::bench_cross_task_clone \
    -- --ignored --nocapture
```

The bench measures 1,000,000 iterations of `waker.clone()` followed
by 1,000,000 drops, after a 10K-iteration warmup. The runtime
allocator is glibc's default (no jemalloc/tcmalloc).

Note: the `benches/cross_thread_wake.rs` criterion target referenced
by the PR 2 plan was not added — the existing `benches/` files
(`executor.rs`, `channel.rs`) don't build against the current crate
API (pre-existing breakage, out of PR 2 scope). The §2.1 perf
measurement uses a `#[test] #[ignore]` bench instead, matching the
convention of `lib.rs::tests::dispatch_latency`.

# Contributing to Nexus

Thanks for your interest in contributing. Before diving in, please read this document carefully — it explains the philosophy behind the project and the standards we hold contributions to.

## Philosophy

### It's a Different Problem

This project takes inspiration from Mike Acton's data-oriented design philosophy: **if you have different constraints, you have a different problem, and a different problem deserves a different solution.**

A single-producer single-consumer queue is not a "special case" of a multi-producer multi-consumer queue. It's a fundamentally different problem with a fundamentally different solution. The constraints aren't limitations to work around — they're what enable the performance.

### Specialization Over Generalization

We are not building general-purpose data structures. We are building **tuned primitives for specific access patterns**.

The standard library and crates like `crossbeam` already provide excellent general-purpose solutions. They handle every case reasonably well. Nexus exists for when "reasonably well" isn't good enough — when you've profiled your system, identified the bottleneck, and need something purpose-built.

This means:

- **We reject features that compromise the core use case.** A bounded queue doesn't need to grow. An SPSC channel doesn't need a sender that's `Clone`.
- **We let users pick and choose.** Small, focused crates that do one thing extremely well. Compose them yourself rather than getting a monolith that's decent at everything.
- **We are honest about constraints.** Every crate documents exactly what it's for and what it's not for.

### Optimize the Common Case

Know your access patterns. If `push` is called a million times for every `len` check, optimizing `len` at the expense of `push` is not a win.

**Tune for the distribution**. We optimize for the lowest expected cost across realistic call patterns — not individual operations in isolation. Profile real workloads, weight by frequency, and optimize the aggregate.

### No Kitchen Sinks

If you're proposing a new feature, ask yourself:

1. Does this serve the core use case, or is it "nice to have"?
2. Does this add complexity to the hot path?
3. Would this be better as a separate crate?

The answer to (3) is usually yes.

## Benchmarking Standards

Performance claims require evidence. We have specific standards for how benchmarks should be conducted to produce reproducible, meaningful results.

### Use Cycles, Not Time

Wall clock time is noisy. We measure CPU cycles using the TSC (Time Stamp Counter):

```rust
fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}
```

Cycles give you a direct measure of work done, independent of clock speed variations.

### Eliminate Jitter

Before running benchmarks:

```bash
# Disable turbo boost (Intel)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Or for AMD
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost

# Pin to physical cores (avoid hyperthreading)
# Core 0 and 2 are often physical siblings — check your topology
sudo taskset -c 0,2 ./target/release/bench
```

### What to Measure

- **Latency**: Ping-pong between two threads. Measure round-trip cycles, divide by 2.
- **Tail latency**: Report p50, p99, p999, max. Median is not enough.
- **Throughput**: Messages per second under sustained load.
- **Jitter**: Standard deviation of latency. Low mean with high variance is often worse than higher mean with low variance.

### Warmup and Sample Size

- Warmup: At least 10,000 iterations before measurement
- Samples: At least 100,000 measured iterations
- Report methodology alongside results

### Linux Perf

For deeper analysis, we expect `perf stat` and `perf record` results:

```bash
sudo perf stat -e cycles,instructions,cache-misses,branch-misses \
    taskset -c 0,2 ./target/release/bench

sudo perf record -e cycles -g \
    taskset -c 0,2 ./target/release/bench
sudo perf report
```

## Architecture Support

### Current State

The primary development and tuning has been done on **Intel x86-64 on Arch Linux**. This is the baseline architecture where we have the most confidence in performance characteristics.

### Contributing for Other Architectures

We actively welcome contributions to improve performance on other architectures:

- AMD x86-64
- Apple Silicon (ARM64)
- Other ARM64 (AWS Graviton, Ampere, etc.)

When contributing architecture-specific optimizations:

1. **Benchmark on the target architecture** using the methodology above
2. **Compare against the baseline** implementation
3. **Document the improvement** with concrete numbers

### Architecture-Specific Code

If an optimization performs significantly better on one architecture but the same or worse on others, gate it behind `cfg(target_arch)`:

```rust
#[cfg(target_arch = "x86_64")]
fn hot_path() {
    // x86-specific implementation
}

#[cfg(target_arch = "aarch64")]
fn hot_path() {
    // ARM-specific implementation
}
```

This way the right code is selected automatically at compile time — no feature flags to remember, cross-compilation just works.

**Example: Memory Ordering**

A real case where this matters is atomic operations vs explicit fences:

```rust
// x86: Strong memory model. Acquire/Release on the atomic itself
// compiles to plain mov instructions — the hardware guarantees ordering.
#[cfg(target_arch = "x86_64")]
fn publish(slot: &AtomicUsize, value: usize) {
    slot.store(value, Ordering::Release);
}

// ARM: Weak memory model. Explicit fences before relaxed stores
// have shown better performance than Release stores in some scenarios.
#[cfg(target_arch = "aarch64")]
fn publish(slot: &AtomicUsize, value: usize) {
    fence(Ordering::Release);
    slot.store(value, Ordering::Relaxed);
}
```

The principle: **optimize for the architecture you're deploying to, don't pessimize others.**

We are not trying to make one architecture "win." We're trying to give users the best performance on whatever hardware they have.

### Memory Ordering

Be especially careful with memory ordering across architectures:

- x86-64 has a strong memory model — many fences compile to nothing
- ARM64 has a weak memory model — incorrect ordering will break

Code must be **correct on all architectures**. Performance can vary, correctness cannot.

### Manual Prefetching

**Default: don't.** Trust the hardware prefetcher. It handles the cases that matter (linear access, predictable strides, cache-resident working sets) without our help. Explicit `_mm_prefetch` calls are a tax on every invocation — extra µops issued, execution-port pressure, and potential conflict with the CPU's own speculation.

Manual prefetching can hurt — sometimes substantially — when:

- The working set already fits in cache after warmup (no DRAM latency to hide)
- Both branches are prefetched but only one is followed (wasted bandwidth)
- The access pattern is already predictable to the hardware prefetcher
- Tree depth or call rate causes the per-call hint cost to compound

**If you genuinely need a prefetch:**

1. Bench the proposed site at realistic populations (small / fits-cache / exceeds-cache).
2. Measure prefetch ON vs OFF at each population.
3. Only ship the prefetch if it shows a meaningful improvement at *some* population AND no regression at *any*.
4. Land the bench evidence in the PR body so future audits don't re-litigate the same question.

Same discipline as `#[inline]`: measure first, only add when you can prove it earns its place.

## Code Standards

### Documentation

Every public item needs documentation that explains:

- What it does
- When to use it (and when not to)
- Performance characteristics
- Example usage

### Testing

- Unit tests for correctness
- Cross-thread tests for concurrent structures  
- Miri for undefined behavior detection where applicable
- Stress tests that run longer than you think necessary

### No Unsafe Without Justification

Unsafe code requires a `// SAFETY:` comment explaining why it's sound. "It's faster" is not sufficient — explain the invariants being upheld.

### Builder and Setter Naming

Workspace convention for naming methods that configure or mutate
state. Matches Rust ecosystem patterns (`Vec`, `String`, `BufReader`,
`tokio`, `hyper`).

**Builder methods** take `mut self` and return `Self`:

- `*_capacity` — application-level allocated buffer (matches
  `Vec::with_capacity`, `String::with_capacity`,
  `BufReader::with_capacity`).
- `*_size` (in OS-option setters) — kernel-side buffer size for
  socket options (matches `tokio::TcpSocket::set_recv_buffer_size`,
  `socket2::Socket::set_*_buffer_size`).
- `max_*_size` — application-level upper limit (matches
  `hyper::Builder::max_buf_size`, `hyper::Body::limit_max_size`).
- No `set_` prefix on builder methods. Builders compose; setters
  mutate.

**Runtime setters** on already-constructed types take `&mut self`,
return `()`, and use the `set_*` prefix (matches `Vec::set_len`).
This is the standard distinction between "configure during
construction" and "mutate after construction."

Example:

```rust
use nexus_async_net::ws::WsStreamBuilder;

// Builder — no `set_` prefix; chains `mut self -> Self`.
let mut stream = WsStreamBuilder::new()
    .buffer_capacity(8192)        // app buffer
    .recv_buffer_size(64 * 1024)  // SO_RCVBUF
    .max_message_size(1 << 20)    // app limit
    .connect("wss://exchange.com/ws")
    .await?;

// Runtime mutator — `set_` prefix; takes `&mut self`.
stream.set_max_read_size(16 * 1024);  // adjust after construction
```

When in doubt, look at how `std`, `tokio`, or `hyper` names a similar
concept.

## Submitting Changes

1. **Open an issue first** for non-trivial changes. Let's discuss the approach before you write code.
2. **Include benchmarks** for performance-related changes.
3. **Keep PRs focused.** One crate, one feature, one fix.
4. **Update documentation** if behavior changes.

## Questions?

Open an issue. We'd rather answer questions than review PRs that don't fit the project's direction.

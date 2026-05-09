# nexus-rate

Fixed-memory, zero-allocation rate limiting for real-time systems.

Three algorithms, two threading models, weighted requests. Every check is
O(1) with single-digit cycle overhead. `no_std` compatible.

## Quick Start

```rust
use nexus_rate::local::Gcra;

// 100 requests per second, burst of 10
let mut limiter = Gcra::builder()
    .rate(100)
    .period(1_000_000_000)  // 1 second in nanoseconds
    .burst(10)
    .build()
    .unwrap();

// On each request:
if limiter.try_acquire(1, now_ns) {
    process_request();
} else {
    reject_or_backoff();
}
```

## Algorithms

| Algorithm | What It Does | State | Allowed (p50) |
|-----------|-------------|-------|--------------|
| **GCRA** | Virtual scheduling. One multiply with precomputed interval. No division. | 8 bytes | 13 cycles |
| **TokenBucket** | Lazy token computation. Burst-tolerant. | 8 bytes + config | 12 cycles |
| **SlidingWindow** | Exact count over rolling time window. | N×8 bytes | 15 cycles |

Numbers are `local::*` `try_acquire(cost, now)` p50 floors — best-of-5
on Intel Core Ultra 7 165U P-cores, taskset-pinned, turbo on. The
algorithm body itself is 2-4 cycles; the rest is `Instant + Duration`
construction inside the timed window (matches realistic per-call cost
in user code).

All three share the same primary API: `try_acquire(cost, now) -> bool`.

### When to Use Which

- **GCRA** — simplest, fastest. Steady-rate limiting with burst tolerance.
  No multiplication on the check path.
- **TokenBucket** — same capability as GCRA but uses the "available tokens"
  mental model. Has `available(now)` query.
- **SlidingWindow** — when you need an exact event count over a time window.
  Use this to mirror exchange rate limit logic (e.g., "1200 orders per minute").

## Threading Models

```rust
// Single-threaded — &mut self, no atomics
use nexus_rate::local::Gcra;
let mut limiter = Gcra::builder()
    .rate(100).period(1_000_000_000).burst(10)
    .build().unwrap();
limiter.try_acquire(1, now);  // &mut self

// Multi-threaded — &self, CAS loop on atomics
use nexus_rate::sync::Gcra;
let limiter = Gcra::builder()
    .rate(100).period(1_000_000_000).burst(10)
    .build().unwrap();
limiter.try_acquire(1, now);  // &self — safe to share via Arc
```

| Module | `try_acquire` | Sync | Cost |
|--------|--------------|------|------|
| `local` | `&mut self` | Single-threaded | 12-15 cycles |
| `sync` | `&self` | Thread-safe (CAS) | 24-29 cycles |

## Weighted Requests

Some systems weight operations differently. For example, an exchange
might count cancel=1, new_order=2, amend=3:

```rust
// cost parameter controls the weight
limiter.try_acquire(1, now);  // cancel — weight 1
limiter.try_acquire(2, now);  // new order — weight 2
limiter.try_acquire(3, now);  // amend — weight 3
```

Applies uniformly: GCRA advances TAT by `cost × emission_interval`,
TokenBucket consumes `cost` tokens, SlidingWindow adds `cost` to the count.

## Runtime Reconfiguration

Rate limits change — exchange adjusts limits, admin command, config reload:

```rust
// Change rate/burst at runtime without rebuilding
limiter.reconfigure(200, 1_000_000_000, 20);
```

No allocation, no state reset. Takes effect on the next `try_acquire`.

## Multi-Rate Composition

Exchanges often enforce multiple limits (e.g., 10/s AND 1200/min).
Compose by checking multiple limiters:

```rust
let mut per_second = Gcra::builder()
    .rate(10).period(1_000_000_000).burst(3)
    .build().unwrap();
let mut per_minute = Gcra::builder()
    .rate(1200).period(60_000_000_000).burst(50)
    .build().unwrap();

// Both must allow:
if per_second.try_acquire(1, now) && per_minute.try_acquire(1, now) {
    send_order();
}
```

## API Summary

| Method | What |
|--------|------|
| `try_acquire(cost, now) -> bool` | Can I proceed? |
| `time_until_allowed(cost, now) -> u64` | How long to wait? (GCRA) |
| `available(now) -> u64` | Tokens remaining (TokenBucket) |
| `count() -> u64` | Current window count (SlidingWindow) |
| `remaining() -> u64` | Capacity left (SlidingWindow) |
| `reconfigure(...)` | Change limits at runtime |
| `reset(...)` | Clear state |

## Performance

All measurements in CPU cycles (`rdtsc`), batch of 64 checks, pinned core
(taskset-pinned P-cores, turbo on, best-of-5 floor).

| Type | Allowed (p50) | Rejected (p50) |
|------|--------------|----------------|
| `local::Gcra` | 13 | 16 |
| `sync::Gcra` | 24 | 12 |
| `local::TokenBucket` | 12 | 11 |
| `sync::TokenBucket` | 29 | 13 |
| `local::SlidingWindow` | 15 | 11 |

Numbers reflect realistic per-call cost — `try_acquire(cost, now)` with
`now` constructed from `Instant + Duration::from_nanos` inside the timed
window. The algorithm body itself is 2-4 cycles for local variants; the
remainder is timestamp construction overhead user code pays per call.

Rejected paths are typically the same cost or cheaper than allowed —
the branch predictor handles rejection well, and rejection skips the
TAT/token-state update.

See `BENCHMARKS.md` for full p50/p99/p999 tables.

```bash
cargo build --release --example perf_rate -p nexus-rate
taskset -c 0 ./target/release/examples/perf_rate
```

## Features

| Feature | Default | What |
|---------|---------|------|
| `std` | yes | Implies `alloc`. `Error` trait on `ConfigError`. |
| `alloc` | no | Enables `SlidingWindow` (heap-allocated ring buffer). |

GCRA and TokenBucket work without any features (`no_std`, no `alloc`).
SlidingWindow requires `alloc`.

## Hot Path Internals

GCRA and TokenBucket precompute `nanos_per_token` at construction and reconfiguration time, avoiding u128 divisions on the hot path. Token computation uses ceil-division to guarantee that fractional tokens are never silently lost -- `try_acquire(1, now)` always consumes at least one token's worth of time.

## Timestamps

All timestamps are `u64`. The caller defines what the units mean —
nanoseconds, rdtsc cycles, milliseconds, etc. The algorithms don't
read clocks internally, making them deterministic and testable.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.

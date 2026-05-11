# nexus-async-rt

> **Status: Experimental — not under active development.**
>
> This crate is a reference implementation of single-threaded busy-poll
> async patterns on mio. **For production async work, use [tokio](https://tokio.rs).**
> The rest of the nexus workspace is runtime-agnostic and composes
> cleanly with tokio.
>
> nexus-async-rt continues to compile, pass tests, and remains usable
> for the workloads it already supports. Bug-fix PRs are welcome. There
> is no commitment to optimize, extend, or maintain feature parity with
> tokio. If you're starting a new project that needs an async runtime,
> reach for tokio first.

Single-threaded async runtime for latency-sensitive systems. Built on mio.

## Why this exists (and why tokio is still the right default)

Tokio's design parks the executor thread when there's no work. For most
workloads that's correct — it gives the OS scheduler back to other
processes. For ultra-low-latency workloads where the thread should never
yield (HFT, real-time control), parking is the wrong policy. nexus-async-rt
was a reference exploration of an executor that doesn't park.

In practice, **tokio can be coerced into busy-poll behavior with known
workarounds**, which makes the structural advantage of nexus-async-rt
small enough that the maintenance cost isn't justified. Hence the
experimental status.

## Quick start

```rust
use nexus_async_rt::*;
use nexus_rt::WorldBuilder;

let mut world = WorldBuilder::new().build();
let mut rt = Runtime::new(&mut world);

rt.block_on(async {
    let handle = spawn_boxed(async { 42 });
    let result = handle.await;
    assert_eq!(result, 42);
});
```

## What you get

### Task spawning

Two strategies, same API:

```rust
// Box-allocated — default, no setup needed
let handle = spawn_boxed(async { compute() });

// Slab-allocated — pre-allocated, zero-alloc hot path
let handle = spawn_slab(async { compute() });
```

Both return `JoinHandle<T>` — await for the result, drop to detach,
or call `abort()` to cancel (consumes the handle).

### Slab allocation (zero-alloc spawn)

For hot-path tasks where allocation jitter is unacceptable:

```rust
// SAFETY: single-threaded runtime owns the slab.
let slab = unsafe { Slab::<256>::with_chunk_capacity(64) };
let mut rt = Runtime::builder(&mut world)
    .slab_unbounded(slab)
    .build();

rt.block_on(async {
    // Pre-allocated — no Box, no allocator, zero syscalls
    let handle = spawn_slab(async { fast_path() });
    handle.await
});
```

Or claim a slot first, spawn later:

```rust
if let Some(claim) = try_claim_slab() {
    let handle = claim.spawn(async { work() });
    // ...
}
```

### Timers

```rust
use std::time::Duration;

// Sleep
sleep(Duration::from_millis(100)).await;

// Timeout
let result = timeout(Duration::from_secs(5), some_future).await;

// Interval
let mut tick = interval(Duration::from_millis(10));
loop {
    tick.tick().await;
    poll_market_data();
}
```

### I/O (mio-based)

```rust
use nexus_async_rt::{TcpStream, TcpListener};

// Client
let stream = TcpStream::connect(addr)?;

// Server
let listener = TcpListener::bind(addr)?;
let (stream, peer) = listener.accept().await?;
```

The constructors fetch the runtime's `IoHandle` internally via
`IoHandle::current()` — mirrors `tokio::net::TcpListener::bind` /
`tokio::net::TcpStream::connect`. They panic if called outside
[`Runtime::block_on`]. Library authors who need the handle directly
can call `IoHandle::current()` themselves.

### Channels

Three flavors for different use cases:

```rust
use nexus_async_rt::channel;

// Local MPSC — !Send, zero atomics, single-threaded
let (tx, rx) = channel::local::channel(64);

// Cross-thread MPSC — Sender: Clone + Send
let (tx, rx) = channel::mpsc::channel(64);

// Cross-thread SPSC — fastest cross-thread path
let (tx, rx) = channel::spsc::channel(64);
```

### World access

Access nexus-rt `World` resources from async tasks:

```rust
WorldCtx::current().with_world(|world| {
    let config = world.resource::<Config>();
    // ...
});
```

`WorldCtx::current()` returns the `World` handle for the active runtime
(panics outside `block_on`). Use `WorldCtx::new(&mut world)` instead
when constructing the handle outside the runtime context (e.g.,
capturing into a task before `block_on`).

### Graceful shutdown

```rust
rt.block_on(async {
    // ... spawn tasks ...
    ShutdownSignal::current().await; // waits for Ctrl+C
});
```

### Cancellation

```rust
let token = CancellationToken::new();
let child = token.child_token();

spawn_boxed(async move {
    while !child.is_cancelled() {
        do_work().await;
    }
    // cleanup
});

token.cancel(); // cancels all children
```

## JoinHandle

`spawn_boxed` and `spawn_slab` return `JoinHandle<T>`:

- **Await** — get the result: `let val = handle.await;`
- **Detach** — drop the handle, task continues, output dropped on completion
- **Abort** — `handle.abort()` consumes the handle, future dropped on next poll
- **Check** — `handle.is_finished()` for non-blocking status

`JoinHandle` is `!Send` and `!Sync` — stays on the executor thread.

## Performance

Measured on Intel Core Ultra 7 165U P-cores, taskset-pinned, turbo on,
best-of-5 floor. See `BENCHMARKS.md` for methodology.

### Dispatch and runtime machinery

| Path | p50 |
|------|-----|
| Task dispatch (poll cycle, no wake) | 55-64 cycles |
| Per-task lifecycle (spawn + 1 poll → Ready + complete + join, amortized) | 228 cycles / ~85 ns |
| Per-poll cycle (steady-state, includes `wake_by_ref` re-arm) | 485 cycles / ~180 ns |

**Dispatch** is the pure poll step — pop ready task, build Context,
call `Future::poll`, handle result. No wake/reschedule. The 55-64cy
figure measures this path in isolation (requires an executor-internal
entrypoint not currently exposed; carried forward from prior baselines).

**Per-task lifecycle** is the realistic spawn-callback pattern: birth
a task, poll it once to completion, retire it. Includes allocation +
spawn + dispatch + complete + handle resolve + free.

**Per-poll cycle (steady-state)** measures a self-rewoken future that
returns Pending and re-arms via `wake_by_ref` each poll — so the cycle
includes wake plumbing (pop + Context + poll + wake_by_ref +
re-push). The ~257cy delta vs per-task is roughly the cost of the
same-thread wake path (atomic queue op, possibly eventfd write).
Investigation of a same-thread wake fast-path is open as a follow-up.

### Channels

| Path | p50 |
|------|-----|
| Local channel try_send+try_recv | 13 ns |
| MPSC channel try_send+try_recv | 22 ns |
| SPSC channel try_send+try_recv | 15 ns |
| Cross-thread channel (busy spin) | 15 ns |
| Cross-thread channel (park/epoll) | 1.7 us |
| Tokio-compat waker bridge | 76 ns |

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `tokio-compat` | No | Adapters for bridging tokio and nexus-async-rt in the same process |

## Dependencies

- **mio** — I/O event loop (epoll/kqueue)
- **nexus-rt** — World/WorldBuilder for typed resource storage
- **nexus-slab** — Optional pre-allocated task storage
- **nexus-timer** — Hierarchical timer wheel
- **nexus-queue** / **nexus-logbuf** — Lock-free internal queues

## Design Notes

`Runtime::block_on` is the only entry point for driving the executor. The `drain()` method was removed -- all task completion is handled within `block_on`'s poll loop.

Cross-thread wakes use a deferred-free strategy: tasks woken from another thread are queued via an intrusive Vyukov MPSC queue and processed on the next executor poll. Task memory is freed on the executor thread, not the waking thread, to avoid cross-thread deallocation.

## Platform support

Unix only (`#![cfg(unix)]`). Linux is the primary target, macOS supported.

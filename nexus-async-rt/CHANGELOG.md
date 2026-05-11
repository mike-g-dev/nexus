# Changelog

All notable changes to nexus-async-rt are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **Crate marked experimental.** README updated with status banner;
  Cargo.toml description clarifies tokio is the supported async
  runtime for production use. nexus-async-rt continues to compile,
  pass tests, and remains usable for current consumers, but is not
  under active development. Bug-fix PRs welcome; no commitment to
  optimize or extend.

### Changed (breaking)

- **Bare-noun context-fetcher free functions converted to
  `Type::current()` pattern.** The four functions that returned a handle
  or future (not a future factory) now live as inherent `current()`
  methods on the relevant type. Mirrors `tokio::runtime::Handle::current()`
  convention; makes call sites self-documenting and discourages threading
  handles through APIs.

  Migration:

  | 0.6.x | 0.7.0 |
  |---|---|
  | `nexus_async_rt::io()` | `IoHandle::current()` |
  | `nexus_async_rt::with_world(\|w\| ...)` | `WorldCtx::current().with_world(\|w\| ...)` |
  | `nexus_async_rt::with_world_ref(\|w\| ...)` | `WorldCtx::current().with_world_ref(\|w\| ...)` |
  | `nexus_async_rt::shutdown_signal()` | `ShutdownSignal::current()` |

  Future factories (`sleep`, `sleep_until`, `interval`, `interval_at`,
  `after`, `after_delay`, `timeout`, `timeout_at`, `yield_now`) and the
  pure value getter (`event_time`) stay as free functions — idiomatic for
  the Rust async ecosystem. The future *is* the API; there's no enclosing
  handle to fetch.

- **Constructor signatures hide `IoHandle`.** Eight public constructors
  on `TcpListener`, `TcpStream`, `TcpSocket`, and `UdpSocket` no longer
  take an explicit `io: IoHandle` parameter — they fetch
  `IoHandle::current()` internally. Mirrors `tokio::net::TcpListener::bind`
  / `tokio::net::TcpStream::connect` ergonomics and demotes the runtime
  internal from end-user APIs (it's now a library-author primitive that
  end users rarely need to reference).

  Migration:

  | 0.6.x | 0.7.0 |
  |---|---|
  | `TcpListener::bind(addr, io)` | `TcpListener::bind(addr)` |
  | `TcpListener::from_std(listener, io)` | `TcpListener::from_std(listener)` |
  | `TcpStream::connect(addr, io)` | `TcpStream::connect(addr)` |
  | `TcpStream::from_std(stream, io)` | `TcpStream::from_std(stream)` |
  | `UdpSocket::bind(addr, io)` | `UdpSocket::bind(addr)` |
  | `UdpSocket::from_std(socket, io)` | `UdpSocket::from_std(socket)` |
  | `TcpSocket::connect(self, addr, io)` | `TcpSocket::connect(self, addr)` |
  | `TcpSocket::listen(self, backlog, io)` | `TcpSocket::listen(self, backlog)` |

  All constructors now panic if called outside `Runtime::block_on`
  (because the internal `IoHandle::current()` does). This is the same
  semantic tokio enforces — bind/connect inside the runtime context.

### Added

- `IoHandle::current()`, `WorldCtx::current()`, `ShutdownSignal::current()`
  — TLS-based fetchers for the active runtime context. All three panic
  outside [`Runtime::block_on`]; all three are `#[must_use]`.

## [0.6.0] — 2026-05-08

The "byte-channel error contract cleanup" release. Companion to
[nexus-logbuf 2.2.0](../nexus-logbuf/CHANGELOG.md). The `ZeroLength`
variant in `nexus-logbuf::{TryClaimError, SendError, TrySendError}`
was a programmer-bug-as-error; it's now a panic at the queue layer.
The cascade reaches `nexus-async-rt::channel::{spsc_bytes,
mpsc_bytes}::ClaimError`, which had its own redundant `ZeroLength`
variant — removed. `Sender::try_claim` signature changes to reflect
the new `nexus_logbuf::BufferFull` unit struct.

### Breaking changes

- **`ClaimError::ZeroLength` removed** from both `channel::spsc_bytes`
  and `channel::mpsc_bytes`. The variant is unreachable now that
  `nexus-logbuf` panics on `len == 0`. `ClaimError` retains
  `Closed` and `TooLarge` (still `#[non_exhaustive]`).
- **`Sender::try_claim` error type changed** in both `spsc_bytes` and
  `mpsc_bytes`:
  - Before: `try_claim(len) -> Result<WriteClaim<'_>, nexus_logbuf::TryClaimError>`
  - After:  `try_claim(len) -> Result<WriteClaim<'_>, nexus_logbuf::BufferFull>`
- **`Sender::claim(len).await` and `Sender::try_claim(len)` now panic on
  `len == 0`.** Channel-layer `assert!(len > 0)` runs before any state
  inspection, so the panic contract is unconditional regardless of
  whether the receiver has been dropped.

### Changed

- Dependency declaration: `nexus-logbuf` `2.1.3` → `2.2.0`. Pulls in
  the new `BufferFull` / `ChannelClosed` types and the structural
  `assert!(len > 0)` at the queue layer.

### Migration

| 0.5.0 | 0.6.0 |
|---|---|
| `Err(ClaimError::ZeroLength)` (in match) | remove arm; assert it can't happen, or fix the bug |
| `Result<_, nexus_logbuf::TryClaimError>` (function sig) | `Result<_, nexus_logbuf::BufferFull>` |
| `Err(nexus_logbuf::TryClaimError::Full)` (in match) | `Err(nexus_logbuf::BufferFull)` |
| `Err(nexus_logbuf::TryClaimError::ZeroLength)` (in match) | remove arm |

If your code called `claim(0).await` / `try_claim(0)` and handled
the `ClaimError::ZeroLength` variant, that path was already a
programmer bug surfaced as a soft error — the fix is to ensure
`len > 0` upstream, not to handle the panic.

## [0.5.0] — 2026-05-05

The "production hardening" release. Five PRs over the hardening
sequence (#198, #199, #201, #202, plus the prior #197) close every
known correctness bug in the runtime — BUG-1, BUG-2, BUG-2 follow-up,
BUG-3, BUG-4. Public API grows to expose shutdown observability and
deterministic clean-shutdown. Two technical breaking changes — both
enforce correct usage at the type level rather than via runtime
checks.

### Breaking changes

- **`TASK_HEADER_SIZE` value: `64` → `72`.** The task header gained
  a `cross_wake_ctx: *const CrossWakeContext` pointer at offset 64
  to support the new terminal-drop routing in `dispose_terminal`.
  Hot-path reads (state, drop_fn, free_fn, poll_fn) stay at low
  offsets; the new field is cold-path-only. Callers using slab
  storage with `Slab<256>` see payload room drop from 192 to 184
  bytes — most async futures fit comfortably. Callers using
  `Slab<128>` see 64 → 56 bytes; non-trivial state machines may
  need to bump slot size.
- **`Cancelled` is now `!Unpin`** (via `PhantomPinned`). `.await`
  auto-pins futures created in async blocks (no caller change for
  the common case). Code calling `Cancelled::poll` directly without
  pinning, or putting `Cancelled` in containers requiring `Unpin`
  (e.g., `Vec<Cancelled>` then iterating with `Pin::new(&mut v[i])`),
  fails to compile. Hot loops re-polling the same `Cancelled` should
  use `pin!()` once outside the loop:
  ```rust
  let cancelled = token.cancelled();
  let mut cancelled = pin!(cancelled);
  loop { /* poll cancelled.as_mut() */ }
  ```

### Added

- `Runtime::shutdown_quiesce(timeout) -> Result<(), QuiesceTimeout>`
  — drives the executor until the cross-thread queue is drained
  and `all_tasks` is empty. After `Ok`, dropping the Runtime is
  guaranteed clean (the abnormal-shutdown branches in
  `Executor::drop` are unreachable in normal operation).
  Documented as the canonical shutdown path in
  [`docs/SHUTDOWN.md`](docs/SHUTDOWN.md).
- `Runtime::shutdown_stats() -> Arc<ShutdownStatsAtomics>` — Arc
  handle that survives `Runtime::drop` so users can read the
  abnormal-shutdown counters post-mortem. Counters fire DURING
  drop; pre-drop snapshots always read zero. Read final values
  via `.snapshot()` on the handle.
- New public types: `ShutdownStats` (plain Copy snapshot users
  match on), `ShutdownStatsAtomics` (Arc-shared inner with
  `.snapshot()`), `QuiesceTimeout` (error type with diagnostic
  fields `remaining_cross_queue`, `remaining_outstanding_refs`,
  `elapsed`).
- `docs/SHUTDOWN.md` — operational guide for the canonical
  shutdown sequence (stop producers → quiesce → drop), including
  the "trigger shutdown from outside `block_on`" pattern.

### Fixed

- **BUG-1** (#167) — slab-allocated tasks surviving past `block_on`
  could panic at `Runtime::drop` with "slab free called without a
  slab configured." Fixed via field-order RAII: slab TLS is
  installed at Runtime construction (not `block_on`) and restored
  on Runtime drop, with a `const _:` assert enforcing field order
  (`_slab_guard` after `executor`).
- **BUG-2** (#168) — channel cross-thread waker UAF. A sender
  capturing the receiver's task pointer mid-`wake()` could race
  the receiver's slot drop, dereferencing freed memory. Fixed via
  `TaskWakerSlot` consolidation: the slot holds a refcount on the
  registered task for the lifetime of the registration.
- **BUG-2 follow-up** — wake/register race in
  `RxWakerSlot::register`. The original ref-release gate was
  conditional on observed slot state, which could miss a
  prev-pointer release if a concurrent `wake()` had transitioned
  state STORED→EMPTY without yet swapping the task pointer. Fixed
  by always releasing prev_ptr if non-null, regardless of state.
- **BUG-3** (#169) — `Cancelled` waker leak on long-running tokens
  with waker churn (e.g., `select!`/`Timeout` patterns). Each poll
  with a changed waker registered a `Box<WaiterNode>` on a Treiber
  stack; old nodes weren't removed. PR #197 reduced trigger
  conditions via `last_waker` gating; this release closes the
  class via intrusive `WaiterNode` embedded in the `Cancelled`
  future itself (no Box per registration). Verified by the new
  `cancel_alloc_count.rs` test: 99 allocs in 100 cycling-waker
  re-polls pre-fix → 0 post-fix.
- **BUG-4** (#196) — busy-spin starvation + slab UAF on unwind.
  Two issues fixed together: tokio-bridge tests using `yield_now`
  to wait on cross-thread state could starve tokio's worker
  thread; `Executor::drop` during a panic could abort or UAF when
  outstanding cross-thread refs raced shutdown. Fixed via
  `dispose_terminal` routing through DEFERRED_FREE TLS (preserves
  `all_tasks` bookkeeping) and the new `shutdown_quiesce` API as
  the deterministic alternative to relying on the 100ms unwind
  defense.

### Internal improvements

- Cross-thread refcount discipline consolidated via `TaskRef`
  smart pointer. Manual `unsafe fn ref_inc(*mut u8)` /
  `unsafe fn ref_dec(*mut u8)` discipline collapses to RAII
  acquire/Drop. Compile-time inc/dec pairing.
- `dispose_terminal` is the single helper for terminal-drop
  routing (replaced six call sites with subtle variations).
  Reads owning `CrossWakeContext` from the task header; on the
  executor's thread, defers via DEFERRED_FREE; off-thread, queues
  via cross-wake queue.
- `TaskWakerSlot` consolidates four duplicate `RxWakerSlot` types
  (mpsc, spsc, mpsc_bytes, spsc_bytes) to one definition.
  `FallbackWaker` and `TxWakerSlot` similarly consolidated.
- `Executor::drop` 4-branch logic refactored into seven named
  helpers, each with documented invariants. No behavior change.
- Per-clone `Box<CrossTaskWakerData>` in the tokio bridge
  replaced with `Arc<CrossTaskWakerInner>` (~50ns malloc → ~9ns
  atomic per clone). Hot-path malloc eliminated for high-rate
  tokio waker traffic.
- Runtime field-ordering invariants documented in a single
  doc-block in `runtime.rs`. Two `const _:` asserts enforce the
  load-bearing invariants (`_slab_guard` after `executor`;
  `_cross_wake_tls_guard` after `executor`); two convention-only
  invariants (`_runtime_presence` last; `cross_wake` Arc lifetime
  via off-thread holders) documented.
- Cancellation token rewrite uses an intrusive doubly-linked list
  protected by a per-token spinlock. Hot poll path takes the lock
  for ~30ns under typical load; lock-free fast-out for already-
  cancelled tokens stays at ~3 cycles. Wake calls happen AFTER
  releasing the lock (collected into a local `Vec<Waker>` first)
  to defend against re-entrance and long-running waker
  implementations.

### Migration notes

For most users, `cargo update -p nexus-async-rt` is the only
change required. `.await` on `Cancelled` continues to work
unchanged. The `TASK_HEADER_SIZE` change is internal unless your
code matched on the const value directly (rare).

For users adopting the new shutdown observability:
1. Clone `runtime.shutdown_handle()` and `runtime.shutdown_stats()`
   BEFORE entering `block_on` (both are `&self` methods).
2. Hand the `ShutdownHandle` to wherever can trigger shutdown
   (signal handler, RPC, supervised parent).
3. Inside the future passed to `block_on`, await
   `shutdown.signal()` to exit on trigger.
4. After `block_on` returns, call
   `runtime.shutdown_quiesce(Duration::from_millis(N))?` to drain
   producers.
5. Drop the runtime. Inspect `stats.snapshot()` for any non-zero
   abnormal-shutdown counters.

See [`docs/SHUTDOWN.md`](docs/SHUTDOWN.md) for the full pattern.

For users in hot loops re-polling `Cancelled`: replace
`token.cancelled().await` (no change needed) with
`pin!(token.cancelled())` once outside the loop, then poll
`cancelled.as_mut()` repeatedly.

---

Versions prior to 0.5.0 are not documented in this CHANGELOG. See
the git history and GitHub release notes for 0.4.x and earlier.

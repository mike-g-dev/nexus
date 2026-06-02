# Changelog

All notable changes to nexus-net are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Breaking

- **`ws`, `http`, `rest` modules extracted to `nexus-web` 0.8.0.**
  nexus-net is now a primitives-only crate: `buf`, `tls`, `wire`,
  `maybe_tls`. Protocol implementations (WebSocket RFC 6455, HTTP/1.1
  parsing, REST client) live in
  [`nexus-web`](https://crates.io/crates/nexus-web), which depends on
  nexus-net for buffer, TLS, and wire abstractions.
  ([#413](https://github.com/Abso1ut3Zer0/nexus/issues/413))

### Migration

| 0.7.x | 0.8.0 |
|---|---|
| `nexus_net::ws::*` | `nexus_web::ws::*` |
| `nexus_net::http::*` | `nexus_web::http::*` |
| `nexus_net::rest::*` | `nexus_web::rest::*` |
| `nexus_net::buf::*` | unchanged |
| `nexus_net::tls::*` | unchanged |
| `nexus_net::wire::*` | unchanged |
| `nexus_net::MaybeTls` | unchanged |
| `nexus_net::WireStream` | unchanged |
| `nexus_net::ParserSink` | unchanged |

Add `nexus-web = "0.8"` to your `Cargo.toml` and update `use` paths
for ws/http/rest types. All primitives (`buf`, `tls`, `wire`,
`MaybeTls`, `WireStream`, `ParserSink`) remain in nexus-net.

## [0.7.1] — 2026-05-10

Doc + small additive release.

### Added

- **`pub const HTTP_HANDSHAKE_BUFFER: usize = 4096`** in
  [`nexus_net::http`] — names the default capacity used for HTTP
  request/response head buffers, REST body / wire / decode scratch
  buffers, and the WebSocket upgrade handshake. Previously inlined
  as a magic literal in ~25 sites across `nexus-async-net`; now
  importable for downstream consumers that want to size their own
  scratch space the same way.
- Module-level doc paragraph on [`rest::error`] explaining the
  partial-preservation behavior of `From<TlsError> for RestError`
  (sync paths surface `Tls` directly, async paths route TLS-IO via
  `Io` so the existing `WireStream` `io::Result` contract stays
  intact). Same wording on [`ws::Error::Tls`] for symmetry.
- Missing-doc additions across `http`, `rest`, `ws`, `tls`
  modules.

### Internal

- Module re-exports unchanged; no behavior change.

## [0.7.0] — 2026-05-08

The "TLS adapter architectural refactor" release. Five rounds of
patches across PR #205 + PR #206 closed real bugs but left iteration
scars: deprecated primitives shipping alongside replacements, a
const_assert holding a latent bug at bay, three-way duplicated
handshake drivers, decision-matrix module docs. 0.7.0 collapses the
TLS surface to a small set of correct-by-construction primitives.

### Breaking

- **`TlsCodec::read_tls`** removed (was deprecated in 0.6.2). The
  former `read_tls_step` is renamed to `read_tls` — same single-packet
  semantics. No-progress now returns `Ok(0)` (was
  `Err(InvalidData)`); matches the `Read::read` idiom and lets caller
  loops detect stuck state.
- **`TlsCodec::encrypt`** all-or-nothing variant removed (was
  deprecated in 0.6.2). The former `try_encrypt` is renamed to
  `encrypt` — chunked, returns bytes accepted.
- **`TlsCodec::process_new_packets`** removed from the public API.
  Folded into `read_tls` and `read_tls_from`.
- **`TlsCodec::process_into`** renamed to
  `drain_plaintext_into<P: ParserSink>`. Generalized over the new
  `ParserSink` trait — works with `FrameReader`, `ResponseReader`,
  `RequestReader`, and any third-party parser implementing `spare` /
  `filled`. (Phase 2: composition seam between TLS and parsers.)
- **`TlsCodec::read_tls_from`**: behavior change. Now does
  read + process internally (was read-only; caller had to call
  `process_new_packets` separately). Return type is
  `Result<usize, TlsError>` (was `io::Result<usize>`).
- **`TlsStream<S>` tokio impl removed.** `TlsStream` is now sync
  only. Async TLS lives in `nexus-async-net::maybe_tls::TlsInner`,
  which is the canonical sans-IO async TLS adapter for the workspace.
- **The `tokio` feature is gone.** nexus-net is sans-IO + sync only.
  The async impls on `Client` (in the deleted `ws/async_tokio.rs` and
  `rest/async_tokio.rs`) were duplicates of
  `nexus_async_net::ws::WsStream` / `nexus_async_net::rest::HttpConnection`
  — every method on the deleted impls (`connect_with`, `accept`,
  `recv`, `send_*`, `close`, async `send`) is provided by the
  `nexus-async-net` types. Layer cleanup: async lives in
  `nexus-async-net`, period.
- **`TlsStream::new` / `TlsStream::handshake` / `TlsStream::with_capacities`**
  removed from the public API. The new entry point is
  `TlsStream::connect(stream, codec)` — it constructs and drives the
  handshake atomically. Fewer half-states, no two-step ceremony.
- **`TlsStream::TMP_SIZE` and `TlsStream::DEFAULT_PENDING_WRITE_CAPACITY`
  consts** removed (no longer relevant; the read chunk size is owned
  by the adapter, not the stream type).

### Added

- **`TlsBufferCapacities`** + builder. Per-connection TLS buffer
  sizing for adapters that need to tune memory footprint. Construct
  via `TlsBufferCapacities::builder().pending_write(64 * 1024).build()`
  or `TlsBufferCapacities::default()` for the standard 16 KiB on
  both sides (~33 KiB resident per connection including rustls
  state). 16 KiB inbound matches rustls's max plaintext record so a
  single record fits in one transport read. Builder gives
  forward-compat headroom for future axes.
- **`WireStream`** trait — composition seam consumed by
  `WsStream`/`HttpConnection`. Bidirectional byte stream with a
  `poll_fill_into<P: ParserSink>` method that delivers bytes directly
  into the parser's spare region. Implemented by `MaybeTls` (in
  `nexus-async-net`) and by user-provided transports via the
  `AsyncReadAdapter` types (also in `nexus-async-net`, one per
  runtime). The nexus-async-rt TLS variant uses the trait to skip the
  `&mut [u8]` intermediate that `AsyncRead` requires — one fewer
  memcpy per recv from rustls's plaintext queue.
- **`ParserSink`** trait — `spare(&mut self) -> &mut [u8]` +
  `filled(&mut self, n: usize)`. Implemented by `FrameReader`,
  `ResponseReader`, and `RequestReader`. Any parser following the
  `spare`/`filled` discipline can plug into the WireStream path.
- **`WireStream::poll_fill_into` precondition contract.** Caller
  must pass `max > 0` and a sink with non-empty
  `spare()`. Implementations return `Err(InvalidInput)` if either
  precondition is violated; with the preconditions met, `Ok(0)`
  unambiguously signals EOF. This removes the ambiguity where
  `Ok(0)` previously could mean either EOF or "no buffer space."
- `TlsCodec::drain_plaintext_into(&mut P: ParserSink)` — renamed
  from `process_into` and generalized. Direct-feed path, one fewer
  copy than reading into an intermediate slice.

### Changed

- Module-level `nexus_net::tls` doc simplified — the decision-matrix
  for input primitives is gone (two clearly-named primitives don't
  need a triage table).
- `TlsCodec::read_and_process_tls` — kept but its docs now describe
  the bounded-input contract clearly. Use for handshake bytes /
  in-memory tests; **do not** use for streaming app-data.

### Removed

- The latent-bug `const_assert!(TMP_SIZE <= 16 KiB)` is gone — the
  handshake-piggyback fix is structural now (the handshake driver
  reads directly into `pending_read.spare()` and stops stepping at
  the handshake transition).
- All deprecated 0.6.2 primitives (see Breaking above).

### Migration

| 0.6.2 | 0.7.0 |
|---|---|
| `read_tls(&[u8])` (deprecated) | `read_tls(&[u8])` (new safe step semantics) |
| `read_tls_step(&[u8])` | `read_tls(&[u8])` |
| `read_and_process_tls(&[u8])` | unchanged (kept as bounded-input helper) |
| `process_new_packets()` | removed; folded into `read_tls`/`read_tls_from` |
| `read_tls_from<R>(&mut R)` (read-only) | `read_tls_from<R>(&mut R)` (read + process) |
| `process_into(&mut FrameReader)` | `drain_plaintext_into(&mut sink)` (any `ParserSink`) |
| `encrypt(&[u8])` (deprecated, all-or-nothing) | use `encrypt` (chunked) |
| `try_encrypt(&[u8])` | `encrypt(&[u8])` |
| `TlsStream::new + handshake` | `TlsStream::connect` |
| `TlsStream::with_capacities` (tokio) | removed; use `nexus-async-net::TlsInner::connect` for async |
| `nexus_net::ws::Client::recv().await` (under `--features tokio`) | `nexus_async_net::ws::WsStream::recv().await` |
| `nexus_net::ws::Client::send_text(...).await` (under `--features tokio`) | `nexus_async_net::ws::WsStream::send_text(...).await` |
| `nexus_net::ws::ClientBuilder::connect_with(s, url).await` (under `--features tokio`) | `nexus_async_net::ws::WsStreamBuilder::connect_with(...).await` |
| `nexus_net::rest::Client::send(req, &mut reader).await` (under `--features tokio`) | `nexus_async_net::rest::HttpConnection::send(req, &mut reader).await` |
| `--features tokio` on `nexus-net` | removed; use `nexus-async-net` for async |

## [0.6.2] — 2026-05-07

The "TLS plaintext-backpressure + steady-state hardening" release.
Closes [#205](https://github.com/Abso1ut3Zer0/nexus/pull/205) — birch
diagnosed that even after the 0.6.1 handshake byte-loss fix, steady-
state TLS app-data >16 KiB could still overflow rustls's internal
plaintext buffer (`received plaintext buffer full`) because the
helper kept feeding ciphertext without giving the caller a chance to
drain plaintext. This release closes that bug surface plus a chain of
related issues surfaced across three audit passes (code review +
hot-path + deep audit).

### Added

- `TlsCodec::read_tls_step(&[u8]) -> Result<usize, TlsError>` — single
  packet-step primitive (`read_tls` + `process_new_packets` once,
  returns bytes consumed). Use for streaming app-data adapters where
  the caller alternates ciphertext input with plaintext output. Avoids
  overflowing rustls's plaintext queue.
- `TlsCodec::try_encrypt(&[u8]) -> Result<usize, TlsError>` — chunked
  variant of `encrypt`. Returns the number of plaintext bytes accepted
  (which may be less than `plaintext.len()`). Implements the proper
  `AsyncWrite::poll_write` contract for plaintexts that exceed
  rustls's outbound queue cap.
- `TlsCodec::set_buffer_limit(Option<usize>)` — pass-through to
  rustls's outbound plaintext queue limit (`DEFAULT_BUFFER_LIMIT =
  64 KiB`). `None` for unlimited.
- `TlsCodec::send_close_notify()` — idempotent rustls wrapper. Used
  by `TlsStream::poll_shutdown` to send a TLS close_notify alert
  before TCP FIN.
- `WriteBuf::spare(&mut self) -> &mut [u8]` and
  `WriteBuf::filled(&mut self, n: usize)` — symmetric with
  `ReadBuf::spare`/`filled`. Enables cursor-FIFO usage where a sans-IO
  codec writes directly into the buffer's tail and commits with
  `filled(n)`.
- `TlsStream::with_capacities(stream, codec, pending_read_cap,
  pending_write_cap)` — explicit buffer capacity tuning. Default
  `new()` uses 8 KiB / 64 KiB.
- `TlsStream::TMP_SIZE` (8 KiB) and
  `TlsStream::DEFAULT_PENDING_WRITE_CAPACITY` (64 KiB) — public
  constants documenting the default sizing and the lower bound for
  `pending_read_cap` (must be ≥ `TMP_SIZE` else `with_capacities`
  panics).
- `TlsStream::set_buffer_limit(Option<usize>)` — convenience
  pass-through to the inner codec.
- Module-level "Choosing an input primitive" decision matrix in
  `tls/mod.rs` covering when to use `read_tls_step` vs
  `read_and_process_tls` vs (deprecated) `read_tls`.

### Changed

- `WriteBuf::advance(n)` now auto-resets `head`/`tail` to
  `reset_offset` when the buffer becomes empty post-advance — matches
  `ReadBuf::advance` semantics. Backwards-compatible: existing callers
  that follow `advance` with `clear()` continue to work; the `clear()`
  is now redundant.

### Deprecated

- `TlsCodec::read_tls(&[u8]) -> Result<usize, TlsError>` — direct
  rustls wrapper that doesn't encode partial-consumption semantics.
  Migrate to `read_tls_step` (streaming) or `read_and_process_tls`
  (bounded handshake input). The bare primitive remains for advanced
  use; its docs now warn against direct use.
- `TlsCodec::encrypt(&[u8]) -> Result<(), TlsError>` — all-or-nothing
  shape that errors with `WriteZero` when plaintext exceeds rustls's
  outbound queue cap. Migrate to `try_encrypt` for chunked semantics.

### Fixed

- `TlsStream::poll_read` (`tls/stream.rs:207`) — steady-state app-data
  bursts ≥16 KiB no longer error with `received plaintext buffer
  full`. Adapter now uses `read_tls_step` + a `pending_read: ReadBuf`
  spillover, drains plaintext between packet steps. Symmetric fix
  for `nexus-async-net::MaybeTls::poll_read` shipped in
  nexus-async-net 0.6.2.
- `TlsStream::poll_write` correctly chunks plaintexts larger than
  rustls's outbound queue cap (default 64 KiB). Previously, a single
  `write_all(&[u8; 100_000])` would surface a confusing `WriteZero`
  error from rustls's writer; now the adapter uses `try_encrypt` and
  returns `Ok(N)` where N may be less than the input length, deferring
  to the standard `AsyncWrite` retry contract.
- `TlsStream::poll_shutdown` now queues a TLS `close_notify` alert,
  flushes the resulting ciphertext, then closes the transport.
  Pre-fix: only TCP FIN was sent, peer treated EOF as a truncation
  signal and errored mid-stream when reading the last bytes (matches
  rustls's defensive behavior). Doc-comment updated to reflect actual
  semantics.

### Internal

- `pending_read: ReadBuf` and `pending_write: WriteBuf` migrated from
  `Vec<u8>` (which used `drain(..n)` — an O(n) memmove on every TLS
  packet step under partial socket reads/writes). Cursor-based buffers
  give O(1) advance with auto-reset to start when fully drained.
- Per-poll `tmp: Box<[u8; 8192]>` hoisted into the struct from a
  per-poll stack alloca + memset. Eliminates ~256 cycles + L1
  pollution per `poll_read` (Casey-audit confirmed via cargo-asm).
- `#[inline]` on `TlsCodec::{read_tls_step, read_plaintext, encrypt,
  is_handshaking, wants_read, wants_write}` — eliminates cross-crate
  function calls per packet step under default codegen-units=16.
- New tokio integration tests in
  `tests/tls_stream_async_backpressure.rs`: oversize app-data burst
  (256 KiB), large write chunking (256 KiB), tiny pending_write
  capacity drain-and-refill, drop-mid-poll regression. Side-by-side
  codec-level demonstrators
  (`adapter_pattern_with_read_and_process_tls_overflows_on_oversize_chunks`
  + `adapter_pattern_with_read_tls_step_handles_oversize_chunks`) pin
  the bug at 32 KiB chunks: identical adapter loop, identical input,
  only the helper differs.
- `const_assert!(TMP_SIZE <= 16 * 1024)` guards the latent
  handshake-piggyback overflow (TLS 1.3 servers can piggyback app-data
  in the same TCP segment as ServerFinished). The architectural fix
  for this — hoisting handshake into `TlsInner` so `pending_read` is
  reachable for direct stash without an intermediate allocation — is
  filed as a 0.7.0 follow-up.

### Migration notes

For most users, `cargo update -p nexus-net` is sufficient. Public API
behavior of high-level types (`TlsStream`, builders) is unchanged.

Direct callers of `TlsCodec::read_tls` or `TlsCodec::encrypt` will see
deprecation warnings — migrate to `read_tls_step` (streaming) or
`try_encrypt` (chunked) per the module-level decision matrix in
`tls/mod.rs`. The deprecated primitives continue to work in 0.6.x;
removal is planned for 0.7.0 alongside the architectural refactor.

## [0.6.1] — 2026-05-05

The "TLS handshake byte-loss" release. Closes
[#200](https://github.com/Abso1ut3Zer0/nexus/issues/200) — TLS
handshakes against servers that emit multi-record handshake bursts
exceeding rustls's per-call read cap (e.g., long cert chains, OCSP
stapling, large certs) could silently drop unconsumed bytes,
stalling the handshake and producing
`Io(UnexpectedEof, "closed during TLS handshake")` after the
server's timeout.

### Added

- `TlsCodec::read_and_process_tls(&[u8]) -> Result<usize, TlsError>` —
  helper that loops `read_tls` + `process_new_packets` until the
  entire input slice is consumed. Use anywhere code reads ciphertext
  bytes into a buffer first (async paths, sans-IO pipelines, IO
  drivers without a `Read` trait) and then needs to push them into
  the codec. Returns `Ok(src.len())` on success or
  `TlsError::Io(InvalidData)` if rustls's deframer can't make
  progress despite intervening `process_new_packets` calls.

### Fixed

- `TlsStream::handshake_async` (line 166) and `poll_read` (line 207)
  now use `read_and_process_tls` instead of the bare
  `read_tls` + `process_new_packets` pair. Pre-fix,
  `rustls::Connection::read_tls(&mut Cursor)` could consume only part
  of the slice and the unconsumed tail was silently lost when `tmp`
  got overwritten on the next loop iteration. This was the bug birch
  reported in #200 against `wss://ws-subscriptions-clob.polymarket.com`.

### Internal

- New regression tests: `tls::codec::tests::read_and_process_tls_handles_oversize_burst`
  (in-process, deterministic; uses 10-cert ECDSA-P256 chain to push
  the burst past rustls's `READ_SIZE = 4096`) and
  `wss_echo::local_wss_echo_with_oversize_handshake_burst` (hermetic
  localhost TLS+WS+frame-echo). Plus
  `tls::codec::tests::bare_read_tls_partially_consumes_large_slice`
  documents rustls's contract directly.
- `read_tls`'s doc-comment now points readers at
  `read_and_process_tls` for the common case (callers who already
  have bytes in a buffer).

### Migration notes

`cargo update -p nexus-net` is the only change required for most
users. The bare `read_tls` API is unchanged — anyone implementing
their own consume loop on top of it continues to work. New code
that pre-buffers bytes should reach for `read_and_process_tls`
instead.

## [0.6.0] — 2026-04-15

Earlier 0.6.x and prior versions are not documented in this
CHANGELOG. See git history and GitHub release notes for details.

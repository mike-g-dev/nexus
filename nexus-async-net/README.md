# nexus-async-net

> **Deprecation notice:** This crate is being renamed to `nexus-async-web`
> (starting at 0.10.0) as part of the networking crate restructure
> ([#413](https://github.com/Abso1ut3Zer0/nexus/issues/413)). The underlying
> protocol code is moving to `nexus-web` (extracted from `nexus-net`), and
> this async adapter will follow under the new name. No further updates will
> be published to `nexus-async-net`.

Async adapters for [nexus-net](../nexus-net). Tokio-compatible.

Same sans-IO primitives, same performance — just `.await` on socket I/O.

- **WebSocket** — `WsStream<S>` wrapping nexus-net's FrameReader/FrameWriter
- **REST HTTP/1.1** — `HttpConnection<S>` wrapping nexus-net's RequestWriter/ResponseReader
- **Client Pool** — `ClientPool` (single-threaded) and `AtomicClientPool` (thread-safe) for connection reuse with LIFO acquire, inline reconnect, and RAII guards

## Backends

This crate has two runtime backends, exposed via Cargo features:

- **`tokio-rt` (default, supported)** — production-ready, the recommended path
- **`nexus` (experimental)** — backed by [nexus-async-rt](../nexus-async-rt/),
  a reference implementation. The `nexus-async-rt` crate itself is
  experimental and not under active development; this feature exists for
  parity but tokio is the supported path for production use.

### Feature matrix

| Feature | What you get | Notes |
|---|---|---|
| (default) | `tokio-tls` | tokio runtime + TLS (rustls via tokio-rustls) |
| `tokio-rt` | tokio runtime alone | no TLS |
| `tokio-tls` | tokio runtime + TLS | composite |
| `tokio-full` | tokio-tls + socket-opts + bytes | recommended for most users |
| `full` | alias for `tokio-full` | backward-compat default bundle |
| `nexus` | nexus-async-rt runtime alone | no TLS, **experimental** |
| `nexus-tls` | nexus runtime + TLS | composite, **experimental** |
| `nexus-full` | nexus-tls + socket-opts + bytes | **experimental** |

Backends are mutually exclusive — pick one runtime feature. To use the
nexus backend, set `default-features = false` and pick a `nexus-*`
composite explicitly.

## Quick Start

```rust
use nexus_async_net::ws::WsStreamBuilder;
use nexus_net::ws::Message;
use nexus_net::tls::TlsConfig;

let tls = TlsConfig::new()?;
let mut ws = WsStreamBuilder::new().tls(&tls).connect("wss://exchange.com/ws").await?;

ws.send_text("subscribe").await?;

while let Some(msg) = ws.recv().await? {
    match msg {
        Message::Text(s) => println!("{s}"),     // zero-copy — borrows from internal buffer
        Message::Binary(b) => process(b),        // zero-copy
        Message::Ping(p) => ws.send_pong(p).await?,
        Message::Close(_) => break,
        _ => {}
    }
}
```

### REST Client (async)

```rust
use nexus_net::rest::RequestWriter;
use nexus_net::http::ResponseReader;
use nexus_async_net::rest::HttpConnectionBuilder;

// Same sans-IO primitives as blocking nexus-net
let mut writer = RequestWriter::new("httpbin.org")?;
writer.default_header("Accept", "application/json")?;
let mut reader = ResponseReader::new(32 * 1024).max_body_size(32 * 1024);

// Async transport — TLS config created once at startup
let tls = nexus_net::tls::TlsConfig::new()?;
let mut conn = HttpConnectionBuilder::new()
    .tls(&tls)
    .connect("https://httpbin.org")
    .await?;

// GET with query params
let req = writer.get("/get")
    .query("symbol", "BTC-USD")
    .finish()?;
let resp = conn.send(req, &mut reader).await?;
println!("{}", resp.body_str()?);
drop(resp);

// POST with body
let req = writer.post("/post")
    .header("Content-Type", "application/json")
    .body(br#"{"action":"buy"}"#)
    .finish()?;
let resp = conn.send(req, &mut reader).await?;
```

The `RequestWriter` and `ResponseReader` are the same types used by
blocking `nexus-net`. The only difference is `.await` on the transport.

### REST Builder (connect timeout, TLS, socket options)

```rust
use std::time::Duration;
use nexus_async_net::rest::HttpConnectionBuilder;

let mut conn = HttpConnectionBuilder::new()
    .connect_timeout(Duration::from_secs(5))
    .disable_nagle()
    .connect("https://api.binance.com")
    .await?;
```

### Server-Side WebSocket (accept)

```rust
use nexus_async_net::ws::WsStream;
use tokio::net::TcpListener;

let listener = TcpListener::bind("127.0.0.1:8080").await?;
let (tcp, _addr) = listener.accept().await?;
let mut ws = WsStream::accept(tcp).await?;

while let Some(msg) = ws.recv().await? {
    // handle messages
}
```

### Client Pool (connection reuse)

```rust
use nexus_async_net::rest::ClientPool;

// Build pool — connects all slots at startup
let pool = ClientPool::builder()
    .url("https://api.binance.com")
    .base_path("/api/v3")
    .default_header("X-API-KEY", &key)?
    .default_header("Content-Type", "application/json")?
    .connections(4)
    .tls(&tls)           // requires "tls" feature (enabled by default)
    .disable_nagle()
    .build()
    .await?;

// Fast path (trading) — no reconnect, no wait, no I/O
// let slot = pool.try_acquire().unwrap();

// Patient path (background) — waits, reconnects with backoff
let mut slot = pool.acquire().await?;

// Build request using the slot's writer
let req = slot.writer.post("/order")
    .header("X-Timestamp", &ts)
    .body(order_json)
    .finish()?;

// Send using the slot's connection + reader (split borrow)
let (conn, reader) = slot.conn_and_reader()?;
let resp = conn.send(req, reader).await?;
println!("{}", resp.body_str()?);

// drop(slot) — returns to pool. If poisoned, reconnects on next acquire.
```

Each slot owns a complete pipeline: `RequestWriter` + `ResponseReader` +
`HttpConnection`. No shared state between slots.

## Client Pool Performance

| Config | Throughput | Pool Overhead |
|--------|-----------|---------------|
| Single connection (no pool) | 255K req/sec | — |
| Pool (1 conn, sequential) | 248K req/sec | ~0% |
| Pool (4 conn, 4 concurrent tasks) | 279K req/sec | **+9.5%** throughput |
| Pool (8 conn, 8 concurrent tasks) | 289K req/sec | **+13.3%** throughput |

Pool acquire/release: **26 cycles** (local), **42 cycles** (atomic).

Measured on localhost TCP where the round-trip is ~9μs. On a real
network (1-10ms round-trip), the concurrency benefit is dramatically
larger — overlapping I/O wait across N connections gives close to Nx
throughput. The localhost benchmark is bottlenecked by the echo server,
not the client.

## Client Pool Design

### Two variants

**`ClientPool`** — single-threaded (`!Send`). Uses `Rc`-based
`nexus_pool::local::Pool`. For `current_thread` runtime + `LocalSet`.
26-cycle acquire/release. This is the primary variant for trading
systems where the hot path runs on a dedicated thread.

**`AtomicClientPool`** — thread-safe (`Send`). Uses atomic CAS-based
`nexus_pool::sync::Pool`. 42-cycle acquire/release. **Single acquirer,
any returner** — one task dispatches requests, guards can be dropped
from any thread.

The `AtomicClientPool` is designed for an architecture where a single
task owns the pool and dispatches requests. It is NOT a global pool
that arbitrary tasks acquire from concurrently — `sync::Pool` is
`Send` but not `Sync`. If you need shared acquire, wrap in a `Mutex`,
but consider whether a single dispatcher task is the better design.

### Failure model

1. **Request fails** — caller gets the error. No retry. The request is
   late (stale timestamp, wrong nonce). Caller decides: log, resubmit
   with fresh params, escalate to another venue.

2. **Connection dies** — slot is poisoned. On drop, the guard returns
   the slot to the pool and the reset closure clears the dead connection
   and response buffer. Next `acquire()` reconnects inline.

3. **Reconnect fails** — `acquire()` returns the error. Slot stays
   disconnected in the pool. Next `acquire()` tries again. When the
   server comes back, the first successful acquire recovers.

4. **All connections dead** — every `acquire()` attempts reconnect.
   Natural recovery when the server returns. No circuit breaker state
   machine — the reconnect-on-acquire pattern IS the recovery.

### Invariants

- The pool **never hands out a poisoned connection**. Every `acquire()`
  checks `needs_reconnect()` and reconnects inline if needed.
- The **reset closure clears stale state** — dead connections are
  dropped and the response reader buffer is reset on return.
- **Writer + reader survive reconnect** — only the transport is replaced.
  Host headers, default headers, base path, buffer capacity are preserved.
- **LIFO acquire** — the most recently used (warmest cache lines)
  connection is acquired first.
- Slots have **public fields** for split borrows through `Pooled<T>`'s
  `DerefMut`. Use `conn_and_reader()` for the common pattern, or
  `let s: &mut ClientSlot = &mut slot;` for direct field access.

## Two API Paths (WebSocket)

### Zero-copy `recv()` (recommended)

`recv()` returns `Message<'_>` borrowing directly from the internal buffer. No allocation per message. Use this for latency-sensitive code — trading systems, market data feeds, high-throughput pipelines.

```rust
while let Some(msg) = ws.recv().await? {
    match msg {
        Message::Text(s) => handle(s),  // s: &str, borrows from ReadBuf
        _ => {}
    }
}
```

### Stream/Sink (ergonomic, tokio-rt only)

`Stream<Item = Result<OwnedMessage, WsError>>` allocates per message but
enables the full `StreamExt`/`SinkExt` combinator API. Use this when
ergonomics matter more than nanoseconds. Only available on the `tokio-rt`
backend — the `nexus` backend uses direct `recv()`/`send_*()` methods
for explicit poll-loop control.

```rust
use futures::StreamExt;
use nexus_net::ws::OwnedMessage;

while let Some(msg) = ws.next().await {
    match msg? {
        OwnedMessage::Text(s) => handle(&s),  // s: String, owned
        _ => {}
    }
}
```

## Performance

### Throughput (in-memory, 1M messages)

| Payload | nexus-rt async | tokio-rt async | blocking | tungstenite |
|---------|---------------|---------------|----------|-------------|
| 40B     | **80.9M** (12ns) | 51.3M (19ns) | 66.3M (15ns) | 9.7M (104ns) |
| 128B    | **55.6M** (18ns) | 38.2M (26ns) | 48.7M (21ns) | 8.6M (117ns) |
| 512B    | **23.1M** (43ns) | 20.3M (49ns) | 21.2M (47ns) | 6.6M (151ns) |

### TCP loopback (tokio, single-threaded)

| Payload | nexus-async-net | tungstenite | Speedup |
|---------|----------------|-------------|---------|
| 40B     | 51.5M (19ns)   | 7.9M (126ns) | **6.5x** |
| 128B    | 31.5M (32ns)   | 6.9M (145ns) | **4.6x** |

### TLS loopback (tokio)

| Payload | nexus-async-net | tungstenite | Speedup |
|---------|----------------|-------------|---------|
| 40B     | 29.4M (34ns)   | 8.3M (120ns) | **3.5x** |
| 128B    | 12.3M (81ns)   | 5.5M (180ns) | **2.2x** |

### Cycle-level latency (nexus-rt, realistic TCP mock)

Per-message (256KB buffer, compact@50%):

| Payload | p50 | p90 | p99 | p99.9 | p99/p50 |
|---------|-----|-----|-----|-------|---------|
| 40B     | 62  | 126 | 212 | 420   | 3.4x    |
| 128B    | 56  | 86  | 182 | 214   | 3.3x    |
| 1024B   | 136 | 146 | 174 | 292   | 1.3x    |

Batched x64 (amortized cycles/msg):

| Payload | p50 | p90 | p99 | p99.9 |
|---------|-----|-----|-----|-------|
| 40B     | 40  | 43  | 68  | 155   |
| 128B    | 52  | 56  | 87  | 168   |

The nexus-rt async path is faster than blocking (80.9M vs 66.3M at 40B).
The noop-waker executor + monomorphized async state machine lets LLVM
optimize better than `std::io::Read` trait dispatch in the blocking path.

Teams already on tokio should use nexus-async-net directly. There is no
performance reason to avoid async — the tokio path matches or beats blocking.

## Builder

```rust
use std::time::Duration;
use nexus_async_net::ws::WsStreamBuilder;
use nexus_net::tls::TlsConfig;

let tls = TlsConfig::new()?;
let mut ws = WsStreamBuilder::new()
    .tls(&tls)                              // requires "tls" feature (default)
    .disable_nagle()
    .buffer_capacity(2 * 1024 * 1024)
    .connect_timeout(Duration::from_secs(5))
    .connect("wss://exchange.com/ws")
    .await?;
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `tokio-rt` | **Yes** (via `tokio-tls`) | Tokio-based async adapters. |
| `nexus` | No | nexus-async-rt-based adapters (single-threaded, pre-allocated). Mutually exclusive with `tokio-rt`. |
| `tls` | **Yes** | TLS support via tokio-rustls + aws-lc-rs. `wss://` and `https://` URLs auto-detected. |
| `socket-opts` | No | Socket options (`SO_RCVBUF`, `SO_SNDBUF`, TCP keepalive) via socket2 on all builders. |
| `bytes` | No | Pass-through — enables `bytes::Bytes` conversion on nexus-net types. |
| `full` | No | All non-runtime features (`tls`, `socket-opts`, `bytes`). |

Disable TLS with `default-features = false` for TLS-free builds.

- **Zero-copy WebSocket** — `Message<'_>` borrows from the internal buffer via `recv()`
- **Stream/Sink** — `OwnedMessage` for `StreamExt`/`SinkExt` ergonomics
- **Zero-alloc REST** — same `RequestWriter`/`ResponseReader` as blocking, just `.await` on I/O
- **Automatic TLS** — `wss://` and `https://` URLs handled transparently via tokio-rustls
- **Connect timeout** — `WsStreamBuilder::connect_timeout()` and `HttpConnectionBuilder::connect_timeout()`
- **Server-side WebSocket** — `WsStream::accept(stream)` for incoming connections
- **Chunked transfer encoding** — decoded transparently for REST responses
- **Same sans-IO primitives** — identical parse path as blocking nexus-net
- **Single-threaded friendly** — works with `current_thread` runtime + `LocalSet`

## Dependencies

- `nexus-net` — sans-IO WebSocket + HTTP primitives
- `tokio` — async runtime (io-util, net, rt)
- `tokio-rustls` — async TLS
- `futures-core` / `futures-sink` — Stream + Sink traits

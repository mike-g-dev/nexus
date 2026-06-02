# nexus-async-web

Async adapters for [nexus-web](../nexus-web). Tokio-compatible.

Same sans-IO primitives, same performance — just `.await` on socket I/O.

- **WebSocket** — `WsReader`/`WsWriter` wrapping nexus-web's FrameReader/FrameWriter
- **REST HTTP/1.1** — `HttpConnection<S>` wrapping nexus-web's RequestWriter/ResponseReader
- **Client Pool** — `ClientPool` (single-threaded) and `AtomicClientPool` (thread-safe) for connection reuse with LIFO acquire, inline reconnect, and RAII guards

> Previously published as `nexus-async-net`. Renamed as part of the
> nexus-net/nexus-web crate restructure
> ([#413](https://github.com/Abso1ut3Zer0/nexus/issues/413)).

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
use nexus_async_web::ws::WsStreamBuilder;
use nexus_web::ws::Message;
use nexus_web::tls::TlsConfig;

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
use nexus_web::rest::RequestWriter;
use nexus_web::http::ResponseReader;
use nexus_async_web::rest::HttpConnectionBuilder;

let mut writer = RequestWriter::new("httpbin.org")?;
writer.default_header("Accept", "application/json")?;
let mut reader = ResponseReader::new(32 * 1024).max_body_size(32 * 1024);

let tls = nexus_web::tls::TlsConfig::new()?;
let mut conn = HttpConnectionBuilder::new()
    .tls(&tls)
    .connect("https://httpbin.org")
    .await?;

let req = writer.get("/get")
    .query("symbol", "BTC-USD")
    .finish()?;
let resp = conn.send(req, &mut reader).await?;
println!("{}", resp.body_str()?);
drop(resp);

let req = writer.post("/post")
    .header("Content-Type", "application/json")
    .body(br#"{"action":"buy"}"#)
    .finish()?;
let resp = conn.send(req, &mut reader).await?;
```

The `RequestWriter` and `ResponseReader` are the same types used by
blocking `nexus-web`. The only difference is `.await` on the transport.

### Client Pool (connection reuse)

```rust
use nexus_async_web::rest::ClientPool;

let pool = ClientPool::builder()
    .url("https://api.binance.com")
    .base_path("/api/v3")
    .default_header("X-API-KEY", &key)?
    .default_header("Content-Type", "application/json")?
    .connections(4)
    .tls(&tls)
    .disable_nagle()
    .build()
    .await?;

let mut slot = pool.acquire().await?;

let req = slot.writer.post("/order")
    .header("X-Timestamp", &ts)
    .body(order_json)
    .finish()?;

let (conn, reader) = slot.conn_and_reader()?;
let resp = conn.send(req, reader).await?;
```

## Performance

### Throughput (in-memory, 1M messages)

| Payload | nexus-rt async | tokio-rt async | blocking | tungstenite |
|---------|---------------|---------------|----------|-------------|
| 40B     | **80.9M** (12ns) | 51.3M (19ns) | 66.3M (15ns) | 9.7M (104ns) |
| 128B    | **55.6M** (18ns) | 38.2M (26ns) | 48.7M (21ns) | 8.6M (117ns) |
| 512B    | **23.1M** (43ns) | 20.3M (49ns) | 21.2M (47ns) | 6.6M (151ns) |

### TLS loopback (tokio)

| Payload | nexus-async-web | tungstenite | Speedup |
|---------|-----------------|-------------|---------|
| 40B     | 29.4M (34ns)    | 8.3M (120ns) | **3.5x** |
| 128B    | 12.3M (81ns)    | 5.5M (180ns) | **2.2x** |

### Client Pool

| Config | Throughput | Pool Overhead |
|--------|-----------|---------------|
| Single connection (no pool) | 255K req/sec | — |
| Pool (4 conn, 4 concurrent) | 279K req/sec | **+9.5%** throughput |

Pool acquire/release: **26 cycles** (local), **42 cycles** (atomic).

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `tokio-rt` | **Yes** (via `tokio-tls`) | Tokio-based async adapters. |
| `nexus` | No | nexus-async-rt-based adapters. Mutually exclusive with `tokio-rt`. |
| `tls` | **Yes** | TLS support via tokio-rustls + aws-lc-rs. |
| `socket-opts` | No | Socket options via socket2. |
| `bytes` | No | `bytes::Bytes` conversion on nexus-web types. |
| `full` | No | All non-runtime features. |

## Dependencies

- `nexus-web` — sans-IO WebSocket + HTTP + REST protocol primitives
- `nexus-net` — buffer, TLS, and wire abstractions
- `tokio` — async runtime (io-util, net, rt)
- `tokio-rustls` — async TLS
- `futures-core` / `futures-sink` — Stream + Sink traits

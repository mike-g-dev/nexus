# nexus-web

Low-latency WebSocket, HTTP/1.1, and REST primitives. Sans-IO, zero-copy,
SIMD-accelerated. Framework-agnostic — works with mio, io_uring, tokio,
or raw syscalls.

Extracted from [nexus-net](../nexus-net) 0.7.x. Protocol state machines
operate on byte slices — no async runtime, no I/O layer.

## Performance

### vs tungstenite (in-memory parse, pinned to cores 0,2)

| Payload | Type | nexus-web | tungstenite | Speedup |
|---------|------|-----------|-------------|---------|
| 40B | binary parse | 19ns (52M/s) | 61ns (16M/s) | **3.2x** |
| 128B | binary parse | 24ns (42M/s) | 75ns (13M/s) | **3.1x** |
| 512B | binary parse | 49ns (20M/s) | 105ns (10M/s) | **2.1x** |
| 77B | JSON quote parse+deser | 146ns (6.9M/s) | 205ns (4.9M/s) | **1.4x** |
| 40B | binary TCP loopback | 30ns (33M/s) | 66ns (15M/s) | **2.2x** |

### rdtsc cycle distribution (pinned to core 0, batch=64)

| Path | p50 | p90 | p99 | p99.9 |
|------|-----|-----|-----|-------|
| text unmasked 128B | 39 | 39 | 43 | 65 |
| binary unmasked 128B | 35 | 36 | 44 | 129 |
| text masked 128B | 52 | 53 | 58 | 124 |
| apply_mask 128B | 12 | 12 | 16 | 31 |
| encode_text 128B server | 10 | 11 | 22 | 39 |
| throughput 100x128B /msg | 28 | 28 | 44 | 91 |

At 3GHz: 39 cycles = 13ns. In-memory throughput: 107M msg/sec (28 cycles/msg).

### vs reqwest (REST HTTP/1.1 client)

| Benchmark | nexus-web | reqwest | Speedup |
|-----------|-----------|---------|---------|
| POST build+write+parse (mock) p50 | 494 cycles (165ns) | 1,549 cycles (516ns) build-only | **3.1x** |
| TCP loopback round-trip p50 | 22,924 cycles (7.6us) | 62,802 cycles (20.9us) | **2.7x** |
| TCP loopback throughput | 114K req/sec | 39K req/sec | **2.9x** |

517/517 Autobahn WebSocket conformance. 16/16 httpbin.org REST conformance.

## Architecture

```
Application
    |
    +-- WebSocket                    REST HTTP/1.1
    |   ^ Message<'a>               ^ Request<'a> / RestResponse<'a>
    |   FrameReader / FrameWriter   RequestWriter / ResponseReader    (sans-IO)
    |   ^ plaintext bytes           ^ plaintext bytes
    |   +----------+----------------+
    |              TlsCodec                     (optional, nexus-net)
    |              ^ encrypted bytes
    +--------------+
                   I/O                          (your choice)
```

Each layer is a pure state machine. No syscalls, no sockets, no async.
Bytes in, messages out.

## Async Adapters

For async usage, see [nexus-async-web](https://crates.io/crates/nexus-async-web) —
thin async wrappers over the same sans-IO primitives. Supports both tokio and
nexus-async-rt backends.

## Quick Start

```toml
[dependencies]
# WebSocket + HTTP, no TLS
nexus-web = "0.8"

# With TLS (rustls + aws-lc-rs via nexus-net)
nexus-web = { version = "0.8", features = ["tls"] }

# Everything (TLS + socket options + bytes)
nexus-web = { version = "0.8", features = ["full"] }
```

### WebSocket Client (ws://)

```rust
use nexus_web::ws::{Client, Message, CloseCode};

let mut ws = Client::builder().connect("ws://exchange.com:80/ws/v1")?;

ws.send_text(r#"{"subscribe":"trades.BTC-USD"}"#)?;

loop {
    match ws.recv()? {
        Some(Message::Text(json)) => process(json),     // &str, zero-copy
        Some(Message::Binary(data)) => process(data),   // &[u8], zero-copy
        Some(Message::Ping(p)) => ws.send_pong(p)?,
        Some(Message::Close(frame)) => {
            ws.close(CloseCode::Normal, "")?;
            break;
        }
        Some(Message::Pong(_)) => {}
        None => break,
    }
}
```

### WebSocket Client (wss://)

```rust
use nexus_web::ws::Client;
use nexus_web::tls::TlsConfig;  // re-exported from nexus-net

let tls = TlsConfig::new()?;
let mut ws = Client::builder().tls(&tls).connect("wss://exchange.com/ws/v1")?;
```

### REST Client (HTTP/1.1, blocking)

```rust
use nexus_web::rest::{Client, RequestWriter};
use nexus_web::http::ResponseReader;

let mut writer = RequestWriter::new("httpbin.org")?;
writer.default_header("Accept", "application/json")?;

let mut reader = ResponseReader::new(32 * 1024).max_body_size(32 * 1024);

let tls = nexus_web::tls::TlsConfig::new()?;
let mut conn = Client::builder().tls(&tls).connect("https://httpbin.org")?;

let req = writer.get("/get")
    .query("symbol", "BTC-USD")
    .query("limit", "100")
    .finish()?;
let resp = conn.send(req, &mut reader)?;
println!("status: {}", resp.status());
```

### Sans-IO (decoupled from sockets)

```rust
use nexus_web::ws::{self, Message, Role};

let (mut reader, writer) = ws::pair(Role::Client);

reader.read_from(&mut socket)?;

for _ in 0..8 {
    match reader.next()? {
        Some(Message::Text(s)) => handle(s),
        Some(Message::Ping(p)) => {
            let mut dst = [0u8; 131];
            let n = writer.encode_pong(p, &mut dst)?;
            socket.write_all(&dst[..n])?;
        }
        None => break,
        _ => {}
    }
}
```

## Modules

### `ws` — WebSocket (RFC 6455)

- **`FrameReader`** — sans-IO inbound parser. Frame parsing, fragment
  assembly, control frame interleaving, SIMD masking, UTF-8 validation.
- **`FrameWriter`** — sans-IO outbound encoder.
- **`Client<S>`** — convenience I/O wrapper with HTTP upgrade handshake.
- **`Message<'a>`** — `Text(&str)`, `Binary(&[u8])`, `Ping`, `Pong`, `Close`.

### `http` — HTTP/1.1 Primitives

- **`RequestReader`** / **`ResponseReader`** — sans-IO HTTP parsers
  backed by httparse (SIMD-accelerated).
- **`ChunkedDecoder`** — sans-IO chunked transfer encoding decoder.
- **`write_request`** / **`write_response`** — zero-alloc HTTP construction.

### `rest` — HTTP/1.1 REST Client

- **`RequestWriter`** — sans-IO request encoder with typestate builder.
- **`Client<S>`** — pure transport. TLS handled at the stream level.
- **`RestResponse<'a>`** — borrows from `ResponseReader`.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `tls` | No | TLS support via nexus-net (rustls + aws-lc-rs) |
| `socket-opts` | No | Socket options via socket2 |
| `bytes` | No | `bytes::Bytes` conversion on `OwnedMessage` and `RestResponse` |
| `full` | No | `tls` + `socket-opts` + `bytes` |

## Design Decisions

**Zero-copy inbound.** `Message::Text(&str)` borrows from the reader's
internal buffer. No heap allocation per message.

**Sans-IO.** Protocol logic is a pure state machine. The same
`FrameReader` works with blocking sockets, mio, io_uring, tokio, or
kernel bypass.

**SIMD-accelerated.** XOR masking uses SSE2/AVX2. UTF-8 validation
uses simdutf8. HTTP header parsing uses httparse (SIMD vectorized).

**No permessage-deflate.** Exchanges that compress use application-level
gzip, not WebSocket compression.

## Testing

```bash
cargo test -p nexus-web
cargo test -p nexus-web --features tls

# Autobahn conformance (requires Podman)
podman run --rm -d --network=host \
    -v "${PWD}/nexus-web/tests/autobahn:/config:Z" \
    -v "${PWD}/target/autobahn-reports:/reports:Z" \
    docker.io/crossbario/autobahn-testsuite \
    wstest -m fuzzingserver -s /config/fuzzingserver.json
cargo test -p nexus-web --test autobahn -- --ignored --nocapture

# httpbin.org conformance (requires network)
cargo test -p nexus-web --all-features --test httpbin -- --ignored --test-threads=1

# Benchmarks
cargo run --release -p nexus-web --example perf_ws
cargo run --release -p nexus-web --example perf_vs_tungstenite
cargo run --release -p nexus-web --example perf_rest
cargo run --release -p nexus-web --features tls --example perf_tls
```

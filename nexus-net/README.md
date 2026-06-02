# nexus-net

Low-latency networking primitives: buffers, TLS codec, wire abstractions.
Framework-agnostic — works with mio, io_uring, tokio, or raw syscalls.

## What's Here

nexus-net provides the foundation layer that protocol crates build on:

- **`buf`** — `ReadBuf` and `WriteBuf` for zero-copy inbound parsing and
  outbound framing. Pre/post padding, cursor advancement, auto-reset.
- **`tls`** — `TlsCodec` (sans-IO encrypt/decrypt via rustls), `TlsConfig`
  (shared config with system roots), `TlsStream` (sync adapter).
  Feature-gated behind `tls`.
- **`wire`** — `WireStream` and `ParserSink` traits for composing transport
  and parser layers.
- **`maybe_tls`** — `MaybeTls<S>` enum for transparent plaintext/TLS streams.

## Protocol Crates

WebSocket, HTTP/1.1, and REST protocol implementations have moved to
[`nexus-web`](https://crates.io/crates/nexus-web) (sans-IO) and
[`nexus-async-web`](https://crates.io/crates/nexus-async-web) (async
adapters). Both depend on nexus-net for primitives.

```
nexus-web ──────► nexus-net (buf, tls, wire, maybe_tls)
nexus-async-web ► nexus-web + nexus-net
```

## Quick Start

```toml
[dependencies]
# Primitives only (buf, wire)
nexus-net = "0.8"

# With TLS (rustls + aws-lc-rs)
nexus-net = { version = "0.8", features = ["tls"] }
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `tls` | No | TLS support via rustls + aws-lc-rs |
| `bytes` | No | `bytes::Bytes` integration for buffer types |
| `full` | No | `tls` + `bytes` |

## Testing

```bash
cargo test -p nexus-net
cargo test -p nexus-net --features tls
```

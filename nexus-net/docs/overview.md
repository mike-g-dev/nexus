# Overview

nexus-net provides low-latency networking **primitives**: buffers, a TLS
codec, and wire-level trait abstractions. Protocol implementations
(WebSocket, HTTP/1.1, REST) have been extracted to
[nexus-web](../../nexus-web/).

## What you get

**Buffers (`buf`):**
- `ReadBuf` — flat inbound buffer with pre/post padding, zero-copy
  parsing surface, compaction support.
- `WriteBuf` — outbound buffer with **prepend headroom** so frame
  headers can be written in-place after the payload is known.

**TLS (`tls`, feature-gated):**
- `TlsConfig` / `TlsConfigBuilder` — rustls configuration wrapper.
- `TlsCodec` — sans-IO encrypt/decrypt adapter around rustls.
- `TlsStream<S>` — blocking adapter implementing `Read + Write`.

**Wire abstractions (`wire`):**
- `WireStream` — composition seam for bidirectional byte streams.
- `ParserSink` — `spare`/`filled` discipline for zero-copy parser feeding.

**Transport (`maybe_tls`):**
- `MaybeTls<S>` — transparent plaintext/TLS enum stream.

## Dependency shape

```
nexus-web ──────► nexus-net (buf, tls, wire, maybe_tls)
nexus-async-web ► nexus-web + nexus-net
```

## Design principles

- **No allocation on the hot path.** Buffers are sized at construction.
- **Zero-copy where the wire format allows.** Parsers read directly from
  `ReadBuf::spare()`.
- **Sans-IO first.** `TlsCodec` operates on byte slices, not sockets.
- **Honest failure modes.** IO errors poison the connection.
  The caller decides whether to reconnect.

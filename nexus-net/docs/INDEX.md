# nexus-net Documentation

Low-latency networking primitives: buffers, TLS codec, wire abstractions.

Protocol implementations (WebSocket, HTTP/1.1, REST) have moved to
[nexus-web](../../nexus-web/docs/).

## Contents

1. [overview.md](./overview.md) — What nexus-net is, architecture, when to use it
2. [tls.md](./tls.md) — Rustls integration, certificate handling, ALPN
3. [buffers.md](./buffers.md) — ReadBuf and WriteBuf semantics

## Source tour

```
src/
  buf/         ReadBuf / WriteBuf — byte buffers with prepend headroom
  tls/         Rustls codec + TlsStream (feature: tls)
  wire.rs      WireStream / ParserSink traits
  maybe_tls.rs MaybeTls<S> — transparent plaintext/TLS
```

## See also

- [nexus-web](../../nexus-web/) — WebSocket, HTTP/1.1, REST protocol primitives
- [nexus-async-web](../../nexus-async-web/) — async adapters for nexus-web

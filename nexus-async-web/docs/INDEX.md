# nexus-async-web Documentation

Async adapter for [nexus-net](../../nexus-net/docs/INDEX.md). Same
zero-copy parsing, same SIMD codecs — just `.await` on I/O.

## Contents

1. [overview.md](./overview.md) — Relationship to nexus-net, when to use async
2. [websocket.md](./websocket.md) — `WsStream`, async `recv`/`send`, `Stream`/`Sink`
3. [http.md](./http.md) — `HttpConnection`, async REST patterns
4. [client-pool.md](./client-pool.md) — `ClientPool`: single-threaded pool with self-healing
5. [atomic-client-pool.md](./atomic-client-pool.md) — `AtomicClientPool`: thread-safe variant
6. [reconnect.md](./reconnect.md) — Self-healing reconnect, backoff, retry semantics
7. [tuning.md](./tuning.md) — Performance tuning knobs (buffer, compact_at, max_read_size)
8. [patterns.md](./patterns.md) — Cookbook: exchange client, pooled REST, accepting connections
9. [performance.md](./performance.md) — Numbers vs tokio-tungstenite, current_thread optimization

## Features

Exactly one runtime must be enabled (mutually exclusive):

- **`tokio-rt`** *(default)* — tokio-based adapters. `Stream`/`Sink`
  trait support for the WebSocket types.
- **`nexus`** — nexus-async-rt based adapters. Single-threaded,
  pre-allocated tasks, no work-stealing. Faster but the ecosystem
  is smaller.

Plus:

- **`tls`** — rustls via nexus-net's TLS layer.
- **`socket-opts`** — tcp keepalive, SO_RCVBUF / SO_SNDBUF tuning.

## Quick pointers

- Exchange connection with self-healing reconnect:
  [patterns.md — Exchange client](./patterns.md#exchange-client-with-reconnect)
- Pooled REST client for order entry:
  [client-pool.md](./client-pool.md)
- Thread-safe pool for a multi-threaded tokio runtime:
  [atomic-client-pool.md](./atomic-client-pool.md)
- When to use sync nexus-net instead:
  [overview.md](./overview.md#when-to-use-which)

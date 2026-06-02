# ClientPool

`rest::ClientPool` is a single-threaded pool of HTTP keep-alive
connections with **inline self-healing reconnect**. It's the piece
you'll use for production trading REST paths on a `current_thread`
tokio runtime + `LocalSet`.

It is intentionally `!Send`. If you need `Send`, use
[`AtomicClientPool`](./atomic-client-pool.md).

## What it gives you

- Pre-allocated slots (one `HttpConnection` + `RequestWriter` +
  `ResponseReader` per slot)
- LIFO acquire for hot-cache behavior
- `try_acquire()` — non-blocking fast path for trading
- `acquire().await` — patient path with backoff for background tasks
- Automatic reconnect on a dead slot via `spawn_local` task — slot
  rejoins the pool when healed
- Shared default headers (auth keys, `User-Agent`, etc.) applied to
  every request via `RequestWriter::default_header`

## Building a pool

```rust
use nexus_async_web::rest::ClientPool;
use nexus_net::tls::TlsConfig;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let local = tokio::task::LocalSet::new();
    local.run_until(async move {
        let tls = TlsConfig::new()?;

        let pool = ClientPool::builder()
            .url("https://api.binance.com")
            .base_path("/api/v3")
            .default_header("X-MBX-APIKEY", &api_key)?
            .connections(4)
            .tls(&tls)
            .disable_nagle()
            .tcp_keepalive(std::time::Duration::from_secs(60))
            .write_buffer_capacity(32 * 1024)
            .response_buffer_capacity(64 * 1024)
            .max_body_size(1 << 20)
            .build()
            .await?;

        // ... use the pool ...
        Ok::<_, anyhow::Error>(())
    }).await?;
    Ok(())
}
```

`build()` is async because it eagerly opens every connection. If
any of the initial connects fail, the builder returns the error
and no pool is created.

The pool is **single-threaded**: it uses `tokio::task::spawn_local`
for reconnect tasks, which requires a `LocalSet`. Running under a
multi-thread runtime without a `LocalSet` will panic when the first
reconnect fires. On `current_thread`, the default task executor is
local, so you can `spawn_local` freely from within any task driven
by that runtime.

## Acquiring a slot

Two paths, same underlying pool:

### Fast path — `try_acquire()`

```rust
if let Some(mut slot) = pool.try_acquire() {
    let s = &mut *slot;    // deref to ClientSlot
    let req = s.writer.post("/order")
        .header("Content-Type", "application/json")
        .body(order_json)
        .finish()?;
    let (conn, reader) = s.conn_and_reader()?;
    let resp = conn.send(req, reader).await?;
    // drop(slot) returns it to the pool
} else {
    // All slots busy or reconnecting. Fall through to an alternate
    // strategy — reject the order, fail fast, or log and drop.
}
```

`try_acquire()` returns:

- `Some(Pooled<ClientSlot>)` — healthy slot, yours until drop
- `None` — every slot is either in use OR currently reconnecting

On every call, `try_acquire()` walks the pool's LIFO list until it
finds a healthy slot. Any dead slot (`needs_reconnect() == true`)
encountered on the way is **ejected and its reconnect task is
spawned** before `try_acquire()` moves on. This is the
self-healing step — a dead slot on top of the stack doesn't block
access to healthy slots underneath.

### Patient path — `acquire().await`

```rust
let mut slot = pool.acquire().await?;
let s = &mut *slot;
let req = s.writer.get("/klines").query("symbol", "BTCUSDT").finish()?;
let (conn, reader) = s.conn_and_reader()?;
let resp = conn.send(req, reader).await?;
```

`acquire()` calls `try_acquire()` in a loop with exponential
backoff (1ms, 2ms, 4ms, ..., capped at 1s) for up to ~20 attempts.
If no slot becomes available, it returns
`RestError::ConnectionClosed("pool acquire timed out...")`.

This is the path for background tasks — account sync, risk
checks, anything that can afford to wait for a connection to come
back online.

## The slot

`ClientSlot` exposes:

```rust
pub struct ClientSlot {
    pub writer: RequestWriter,        // per-slot WriteBuf
    pub reader: ResponseReader,       // per-slot ReadBuf
    pub conn: Option<HttpConnection<MaybeTls>>,  // None = dead
}

impl ClientSlot {
    pub fn needs_reconnect(&self) -> bool;
    pub fn conn_and_reader(&mut self) -> Result<(&mut HttpConnection<MaybeTls>, &mut ResponseReader), RestError>;
}
```

`conn_and_reader()` returns `RestError::ConnectionPoisoned` if the
slot is dead — in practice you'll never see this, because
`try_acquire` already filters dead slots out.

The writer's default headers (set via
`ClientPoolBuilder::default_header`) are applied every time you
call `writer.get/post/...` — you don't need to re-add auth on each
request.

## Self-healing: what happens when a connection dies

1. Your code calls `conn.send(req, reader).await`.
2. The socket is dead (RST, timeout, black hole). `send` returns
   `Err(RestError::Io(_))`. The `HttpConnection` marks itself
   poisoned.
3. You propagate the error. Drop the slot.
4. Next `try_acquire()` sees `slot.needs_reconnect() == true`,
   calls `spawn_reconnect(slot)`.
5. That task owns the slot guard. It enters a reconnect loop with
   100ms → 5s exponential backoff. On success, it writes a fresh
   `HttpConnection` into the slot and drops the guard — the slot
   rejoins the pool automatically.
6. Meanwhile, `try_acquire()` keeps scanning and returns the next
   healthy slot. Callers of `acquire().await` who hit zero healthy
   slots wait for the reconnect task to finish.

See [reconnect.md](./reconnect.md) for full semantics.

## Capacity sizing

- **`connections(n)`** — number of slots. At-most-`n` concurrent
  in-flight requests. Size for your burst rate, not steady-state:
  trading APIs often hit per-endpoint rate limits that matter more
  than server concurrency.
- **`write_buffer_capacity`** — per-slot outbound buffer. Must hold
  the largest request you emit.
- **`response_buffer_capacity`** — per-slot inbound buffer. Must
  hold your largest response body plus headers.
- **`max_body_size`** — hard cap on inbound body length. Defends
  against a buggy server streaming multi-gigabyte chunked responses.
  `0` means unlimited.

## Limitations

- **Single-threaded.** `ClientPool` is `!Send`. Use it on
  `current_thread` runtimes or within a `LocalSet`.
- **One URL per pool.** All slots target the same host + base path.
  Use one pool per venue.
- **No request-level retry.** The pool heals connections; retrying
  the *request* (idempotency, backoff) is the caller's job.
- **No HTTP/2.** All slots are HTTP/1.1 keep-alive.

# AtomicClientPool

`rest::AtomicClientPool` is the thread-safe cousin of
[`ClientPool`](./client-pool.md). Same slot type, same self-healing
reconnect, same API — but built on `nexus_pool::sync::Pool`, so it
supports cross-thread slot return.

Use it when you're on a **multi-threaded tokio runtime** and the
pool is shared across workers.

## What changes vs `ClientPool`

| Aspect | `ClientPool` | `AtomicClientPool` |
|--------|--------------|--------------------|
| Underlying pool | `nexus_pool::local::Pool` | `nexus_pool::sync::Pool` |
| `Send` | **No** (`!Send`) | **Yes** |
| `Sync` | No | **No** (single acquirer) |
| Reconnect task | `tokio::task::spawn_local` | `tokio::spawn` |
| Slot return | Same thread | Any thread |
| Typical runtime | `current_thread` + `LocalSet` | Multi-threaded tokio |

Everything else is identical — same `try_acquire`/`acquire`
semantics, same builder, same `ClientSlot` (literally the same type
— `AtomicClientSlot = ClientSlot`).

## `Send` but not `Sync`

`AtomicClientPool` is `Send` — you can move it between tasks —
but it is **not `Sync`**. That means you can't share `&pool` across
multiple tasks running on different threads simultaneously.

To share acquire across threads, wrap the pool in a `Mutex`:

```rust
use std::sync::Arc;
use tokio::sync::Mutex;
use nexus_async_web::rest::AtomicClientPool;

let pool = Arc::new(Mutex::new(AtomicClientPool::builder()
    .url("https://api.exchange.com")
    .connections(8)
    .build()
    .await?));

let handle1 = tokio::spawn({
    let pool = pool.clone();
    async move {
        let mut slot = pool.lock().await.try_acquire().unwrap();
        // ... use slot ...
    }
});
```

Slot **return** is lock-free regardless — a `Pooled<AtomicClientSlot>`
guard can be dropped from any thread without going through the
mutex. Only `try_acquire` / `acquire` need to be serialized.

For many workloads the mutex is not a bottleneck: acquire happens
once per request, the critical section is O(1), and you're already
in an async context where the mutex is fine. Measure if in doubt.

## Example

```rust
use nexus_async_web::rest::AtomicClientPool;
use nexus_net::tls::TlsConfig;

#[tokio::main]  // multi-thread by default
async fn main() -> anyhow::Result<()> {
    let tls = TlsConfig::new()?;

    let pool = AtomicClientPool::builder()
        .url("https://api.exchange.com")
        .default_header("X-API-KEY", &key)?
        .connections(8)
        .tls(&tls)
        .build()
        .await?;

    // Single-task acquire is the simple case:
    let mut slot = pool.try_acquire().unwrap();
    let s = &mut *slot;
    let req = s.writer.get("/ticker").finish()?;
    let (conn, reader) = s.conn_and_reader()?;
    let resp = conn.send(req, reader).await?;
    println!("{}", resp.status());
    Ok(())
}
```

## When to use which pool

Default to `ClientPool` if you can. Single-threaded tokio is
faster for latency-sensitive code — no work-stealing overhead, no
cross-core cache bouncing, predictable task ordering.

Switch to `AtomicClientPool` when:

- You're running on a multi-thread runtime (usually because some
  dependency requires it — hyper servers, axum, etc.)
- You need to share a single pool across tokio worker threads

If you don't have a forcing constraint, run your trading code on
a dedicated `current_thread` runtime pinned to a core and use
`ClientPool`. That's the configuration benchmarked in
[performance.md](./performance.md).

## Reconnect semantics

Identical to `ClientPool`. See [reconnect.md](./reconnect.md) for
the full write-up. The only difference is that reconnect tasks are
spawned with `tokio::spawn` instead of `spawn_local`, so they can
run on any worker thread.

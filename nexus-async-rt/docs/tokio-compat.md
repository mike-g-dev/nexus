# Tokio Compatibility

The `tokio-compat` feature bridges nexus-async-rt to tokio's ecosystem.
Use it for **cold-path code** that needs to run on tokio: `reqwest`,
`tokio-rustls`, `sqlx`, any other crate that only works inside a tokio
runtime.

The hot path stays on nexus-async-rt. Tokio is a passenger, not the driver.

```toml
[dependencies]
nexus-async-rt = { version = "*", features = ["tokio-compat"] }
```

## Two Entry Points

| API                 | Runs on                       | Use when                            |
|---------------------|-------------------------------|-------------------------------------|
| `with_tokio(fn)`    | our executor (borrowed tokio) | cold-path future needs tokio ctx    |
| `spawn_on_tokio(f)` | tokio's thread pool           | blocking/CPU work off our thread    |

`with_tokio` drives the future on our thread, inside a tokio runtime
context (so `tokio::spawn`, timers, and I/O primitives all see a runtime).
`spawn_on_tokio` hands the future to tokio's multi-thread pool and wakes
us cross-thread when it completes — useful for work that must not block
our event loop.

## `with_tokio` — Run a Tokio Future Here

```rust
pub fn with_tokio<F, Fut>(f: F) -> TokioCompat<Fut>
where
    F: FnOnce() -> Fut,
    Fut: Future;
```

```rust
use nexus_async_rt::{Runtime, spawn_boxed, tokio_compat::with_tokio};
use nexus_rt::WorldBuilder;

fn main() {
    let mut world = WorldBuilder::new().build();
    let mut rt = Runtime::new(&mut world);
    rt.block_on(async {
        spawn_boxed(async {
            // reqwest needs a tokio runtime context.
            let body: String = with_tokio(|| async {
                reqwest::get("https://example.com/api/ref-prices")
                    .await.unwrap()
                    .text().await.unwrap()
            }).await;

            nexus_async_rt::with_world(|_w| {
                // parse body, update World state
                let _ = body;
            });
        }).await;
    });
}
```

The future runs on **our** thread — no cross-thread hop. It's a good fit
for HTTP calls, TLS handshakes, DNS, etc., where you're spending most of
your time waiting for IO and the ecosystem demands tokio.

## `spawn_on_tokio` — Off-Thread Execution

```rust
pub fn spawn_on_tokio<F, T>(future: F) -> TokioJoinHandle<T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static;
```

Spawns the future on tokio's multi-thread runtime (lazy global, initialized
on first use via `OnceLock`). Returns a `TokioJoinHandle<T>` that implements
`Future<Output = Result<T, TokioJoinError>>`. When the task completes, it
wakes us via our cross-thread waker — ~76ns per hop, measured.

```rust
use nexus_async_rt::{Runtime, spawn_boxed, spawn_on_tokio};
use nexus_rt::WorldBuilder;

fn expensive_compute(input: Vec<f64>) -> f64 {
    // pretend this takes 200ms
    input.iter().sum()
}

fn main() {
    let mut world = WorldBuilder::new().build();
    let mut rt = Runtime::new(&mut world);
    rt.block_on(async {
        spawn_boxed(async {
            let data = vec![1.0; 1_000_000];

            // Run on tokio's thread pool — our event loop stays responsive.
            let handle = spawn_on_tokio(async move {
                expensive_compute(data)
            });

            match handle.await {
                Ok(sum) => println!("sum = {sum}"),
                Err(e) if e.is_cancelled() => eprintln!("cancelled"),
                Err(e) if e.is_panic() => eprintln!("panicked"),
                Err(_) => eprintln!("unknown error"),
            }
        }).await;
    });
}
```

**When to use `spawn_on_tokio` instead of `with_tokio`:**

- The work is CPU-bound and would block our single-threaded loop.
- The work calls blocking APIs wrapped in `tokio::task::spawn_blocking`.
- You want the work to continue even if our event loop is momentarily
  backed up.

**When to stick with `with_tokio`:**

- You're waiting on IO (HTTP, TLS). The tokio thread pool adds latency
  for no benefit — our epoll is already good.
- You need the result quickly (sub-ms). Cross-thread hops add ~76ns each.

## `TokioJoinHandle` and `TokioJoinError`

```rust
pub struct TokioJoinHandle<T>;
impl TokioJoinHandle<T> {
    pub fn is_finished(&self) -> bool;
    pub fn abort(&self);
}

pub struct TokioJoinError(tokio::task::JoinError);
impl TokioJoinError {
    pub fn is_cancelled(&self) -> bool;
    pub fn is_panic(&self) -> bool;
}
```

Unlike `nexus_async_rt::JoinHandle<T>` (which returns `T`),
`TokioJoinHandle<T>` returns `Result<T, TokioJoinError>`. This is because
tokio can report panic isolation and cancellation, which nexus-async-rt
cannot (tokio's runtime catches panics; ours lets them unwind).

## The Lazy Global Tokio Runtime

On first use of `spawn_on_tokio`, a multi-thread tokio runtime is
constructed inside a `OnceLock`. It stays alive for the process lifetime.
You do not create or manage it — just call `spawn_on_tokio`.

This means:

- The first call pays the runtime startup cost (~1ms).
- All subsequent calls are cheap.
- The tokio runtime uses whatever number of worker threads tokio's
  default config picks (typically `num_cpus`).

If you need a custom tokio runtime, don't use `spawn_on_tokio` — build
your own `tokio::runtime::Runtime` and use it explicitly.

## Cross-Thread Waker Fix

`spawn_on_tokio` relies on a correct cross-thread waker path: when the
tokio task completes, it wakes an async task on our executor. The waker
path has a subtle invariant — waking a task that's already been freed
must be sound. This is fixed in ARCHITECTURE.md under "Cross-Thread Wake
Queue". Summary: we use an intrusive Vyukov MPSC queue with refcounted
task pointers, and completed tasks reach a TERMINAL state that routes
them to `deferred_free` rather than re-polling.

You don't need to think about this — it's a runtime invariant — but if
you hit a cross-thread waker bug, that's the first place to look.

## Example: `reqwest` for REST Cold Path

```rust
use nexus_async_rt::{Runtime, spawn_boxed, with_world, tokio_compat::with_tokio};
use nexus_rt::{Resource, WorldBuilder};
use std::time::Duration;

#[derive(Resource, Default)]
struct RefPrices { btc_usd: f64 }

async fn refresh_prices() {
    // This is a cold path — we refresh every 60 seconds, not per tick.
    let body: String = with_tokio(|| async {
        reqwest::Client::new()
            .get("https://api.example.com/spot/BTC-USD")
            .timeout(Duration::from_secs(5))
            .send().await.unwrap()
            .text().await.unwrap()
    }).await;

    // Parse & apply under a scoped World borrow.
    let price: f64 = body.parse().unwrap_or(0.0);
    with_world(|w| {
        w.resource_mut::<RefPrices>().btc_usd = price;
    });
}

fn main() {
    let mut world = WorldBuilder::new()
        .with_resource(RefPrices::default())
        .build();
    let mut rt = Runtime::new(&mut world);
    rt.block_on(async {
        spawn_boxed(async {
            let mut interval = nexus_async_rt::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                refresh_prices().await;
            }
        }).await;
    });
}
```

## Example: TLS Handshake Bridge

`tokio-rustls` handshakes can be done inline via `with_tokio`:

```rust
use nexus_async_rt::tokio_compat::with_tokio;

async fn tls_handshake(tcp: tokio::net::TcpStream) -> std::io::Result<tokio_rustls::client::TlsStream<tokio::net::TcpStream>> {
    with_tokio(|| async move {
        let cfg = std::sync::Arc::new(make_client_config());
        let connector = tokio_rustls::TlsConnector::from(cfg);
        let server_name = rustls::pki_types::ServerName::try_from("api.example.com").unwrap();
        connector.connect(server_name, tcp).await
    }).await
}

fn make_client_config() -> rustls::ClientConfig { todo!() }
```

For the hot path — market data over TLS — use `nexus-net` + `nexus-async-net`
directly. `with_tokio` is fine for one-off REST calls, order cancels on
shutdown, etc.

## See Also

- [Architecture](ARCHITECTURE.md) — cross-thread waker details
- [Channels](channels.md) — feeding results from tokio tasks into the
  hot-path loop
- [Integration with nexus-rt](integration-with-nexus-rt.md) — updating
  the World from tokio-bridged tasks
- [Shutdown](SHUTDOWN.md) — canonical shutdown sequence when mixing
  tokio + nexus runtimes; covers `shutdown_quiesce` and `ShutdownStats`
  for observability

# Cookbook: Exchange Connection End to End

**Goal:** connect to an exchange with TLS, self-heal on disconnect,
enforce the exchange's rate limit on outbound orders, mint unique
client order IDs, and archive every byte for compliance / replay.

**Crates used:**
`nexus-async-web` (WebSocket + TLS), `nexus-async-rt` (executor),
`nexus-rate` (GCRA / token bucket), `nexus-id` (Snowflake), `nexus-logbuf`
(archival), `nexus-ascii` (exchange symbol).

Coinbase-style endpoints are used for flavor. Every exchange has
different quirks — the building blocks are the same.

---

## 1. The connection lifecycle

```
        ┌────────────┐
        │   closed   │
        └─────┬──────┘
              │ connect()
              ▼
        ┌────────────┐       fail
        │ connecting │──────────────┐
        └─────┬──────┘               │
              │ ok                  backoff
              ▼                      │
        ┌────────────┐               │
        │  open      │◀──────────────┘
        └─────┬──────┘
              │ disconnect / error
              ▼
        ┌────────────┐
        │  draining  │ (flush archive, rebuild state)
        └─────┬──────┘
              │
              ▼
        ┌────────────┐
        │ reconnect  │ → back to connecting
        └────────────┘
```

- **Connecting** is a bounded-retry state. Back off with jitter.
  Don't hammer the exchange.
- **Open** is the normal state. Send / receive.
- **Draining** runs on every disconnect, even "clean" ones. Flush
  the archive, emit a "connection lost" event so downstream can
  invalidate its book, resubscribe on reconnect.

---

## 2. Outbound rate limiting

Every exchange publishes a rate limit. Violating it gets you a ban.
`nexus-rate` gives you three algorithms — `Gcra`, `TokenBucket`,
`SlidingWindow` — in `local` (single-threaded) and `sync` (atomic)
flavors.

Pick based on the exchange's published policy:

- **"500 requests per 10 seconds, burst up to 50"** → token bucket
  with capacity 50, refill 50/s.
- **"5 orders per 200ms, smoothed"** → GCRA, 25/s rate, burst 5.
- **"100 orders per rolling 60-second window, hard cap"** → sliding
  window.

For a single-threaded connection (one writer = the exchange task),
use the `local` variant — it's `&mut self` and cheaper.

```rust
use nexus_rate::local::Gcra;
use std::time::{Duration, Instant};

pub struct OrderRateLimiter {
    gcra: Gcra,
}

impl OrderRateLimiter {
    pub fn coinbase_style() -> Self {
        // fixed: Gcra uses a builder. "5 orders per second, burst 10"
        // → rate=5, period=1s, burst=10.
        Self {
            gcra: Gcra::builder()
                .rate(5)
                .period(Duration::from_secs(1))
                .burst(10)
                .build()
                .unwrap(),
        }
    }

    // fixed: `try_acquire(cost, now) -> bool`. Use `time_until_allowed`
    // for the wait-hint.
    pub fn try_send_one(&mut self, now: Instant) -> Result<(), Duration> {
        if self.gcra.try_acquire(1, now) {
            Ok(())
        } else {
            Err(self.gcra.time_until_allowed(1, now))
        }
    }

    /// If the exchange rejected the request, rebate the token so a
    /// retry doesn't double-count.
    pub fn rebate(&mut self, now: Instant) {
        self.gcra.release(1, now);
    }
}
```

The wrapper above returns `Err(Duration)` telling you how long to wait
before the next attempt will succeed. Don't `sleep(d)` — instead,
park the request on a queue and try again from the event loop.

---

## 3. Client order IDs

Every outbound order needs an ID **you** control, so you can
correlate the ack and handle duplicates on reconnect. Snowflake is
the fastest path: monotonic, 64-bit, p50 ~22 cycles.

```rust
// fixed: the generator type is `Snowflake64<TS, WK, SQ>` with
// const-generic layout, and `next_id(tick)` returns
// `Result<SnowflakeId64<...>, SequenceExhausted>`.
use nexus_id::Snowflake64;
use std::time::Instant;

pub type ClientIdLayout = Snowflake64<42, 6, 16>;

pub struct ClientOrderIdFactory {
    gen: ClientIdLayout,
    epoch: Instant,
}

impl ClientOrderIdFactory {
    pub fn new(worker_id: u16) -> Self {
        Self {
            gen: ClientIdLayout::new(worker_id as u64),
            epoch: Instant::now(),
        }
    }

    pub fn next(&mut self) -> u64 {
        let tick = (Instant::now() - self.epoch).as_millis() as u64;
        u64::from(self.gen.next_id(tick).expect("sequence exhausted"))
    }
}
```

**Idempotency matters on reconnect.** If the connection dropped
mid-send, you don't know whether the exchange saw the order. If the
exchange accepts a client-supplied ID, reuse the same ID on retry:
the exchange will either ack the existing order or reject as
duplicate — both are recoverable. If you mint a fresh ID on retry
you risk double-fills.

Store the last pending id in a resource so the reconnect handler
can read it.

---

## 4. TLS WebSocket with `nexus-async-web`

```rust
// fixed: nexus-async-web exposes `WsStream` + `WsStreamBuilder`, not
// `WebSocketClient`. `MaybeTls` is re-exported from the same module
// and is the transport produced by `WsStreamBuilder::connect`.
use nexus_async_web::ws::{MaybeTls, Message, WsStream, WsStreamBuilder};

async fn connect()
    -> Result<WsStream<MaybeTls>, nexus_async_web::ws::WsError>
{
    let url = "wss://ws-feed.exchange.coinbase.com";
    WsStreamBuilder::new().connect(url).await
}
```

The same `WsStream` is used for both sides — `ws.recv()` for
inbound, `ws.send_text()`/`ws.send_binary()` for outbound.

---

## 5. Archival of everything

Every byte in, every byte out, to a dedicated SPSC `nexus-logbuf`.
Dedicated cold thread drains to disk.

```rust
// fixed: logbuf exports `spsc::new` (not `log_buf`) and the Producer
// uses `try_claim(len) -> Result<WriteClaim, _>`; WriteClaim derefs to
// `&mut [u8]`.
use nexus_logbuf::spsc::Producer;

pub struct ArchiveWriter {
    tx: Producer,
}

impl ArchiveWriter {
    pub fn record(&mut self, dir: Direction, bytes: &[u8], ts_ns: u64) {
        // record layout: [dir:1][pad:7][ts:8][len:4][bytes:len]
        let total = 1 + 7 + 8 + 4 + bytes.len();
        let Ok(mut claim) = self.tx.try_claim(total) else { return };
        claim[0] = dir as u8;
        claim[8..16].copy_from_slice(&ts_ns.to_le_bytes());
        claim[16..20].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
        claim[20..20 + bytes.len()].copy_from_slice(bytes);
        claim.commit();
    }
}

#[repr(u8)]
pub enum Direction { Inbound = 0, Outbound = 1 }
```

**Archive on both sides.** On reconnect, a replay pass over the
archive tells you exactly what state the exchange saw — which is
the only reliable way to reconcile after a disconnect.

---

## 6. Putting it together — the exchange task

```rust
pub struct Exchange {
    client: Option<WsStream<MaybeTls>>,
    limiter: OrderRateLimiter,
    ids: ClientOrderIdFactory,
    archive: ArchiveWriter,
    pending: std::collections::VecDeque<OrderRequest>,
    last_pending_id: Option<u64>,
}

pub struct OrderRequest {
    pub id: u64,
    pub symbol: nexus_ascii::AsciiString16,
    pub side: Side,
    pub price: i64,
    pub qty: i64,
}

#[derive(Clone, Copy)] pub enum Side { Buy, Sell }

impl Exchange {
    pub async fn run(&mut self) {
        'outer: loop {
            // 1. Connect (with backoff).
            let mut backoff_ms = 100u64;
            self.client = loop {
                match connect().await {
                    Ok(c) => break Some(c),
                    Err(e) => {
                        tracing::warn!(?e, backoff_ms, "connect failed");
                        tokio::time::sleep(
                            std::time::Duration::from_millis(backoff_ms)
                        ).await;
                        backoff_ms = (backoff_ms * 2).min(30_000);
                    }
                }
            };
            let ws = self.client.as_mut().unwrap();

            // 2. Resubscribe / replay.
            self.resubscribe(ws).await.ok();
            // If we had a pending id mid-flight, resend with SAME id.
            if let Some(id) = self.last_pending_id.take() {
                self.requeue_pending(id);
            }

            // 3. Main poll loop.
            loop {
                tokio::select! {
                    // fixed: ws.recv() -> Result<Option<Message<'_>>, WsError>.
                    incoming = ws.recv() => {
                        match incoming {
                            Ok(Some(Message::Text(s))) => {
                                self.archive.record(
                                    Direction::Inbound, s.as_bytes(), now_ns());
                                self.handle_incoming(s).await;
                            }
                            Ok(Some(Message::Close(_))) | Ok(None) | Err(_) => {
                                tracing::warn!("exchange disconnected");
                                break; // back to connect
                            }
                            _ => {}
                        }
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(1)) => {
                        self.drain_pending(ws).await;
                    }
                }

                if self.should_shutdown() { break 'outer; }
            }
        }
    }

    async fn drain_pending(&mut self, ws: &mut WsStream<MaybeTls>) {
        let now = Instant::now();
        while let Some(req) = self.pending.front().cloned() {
            match self.limiter.try_send_one(now) {
                Ok(()) => {
                    let text: String = serialize(&req);
                    self.last_pending_id = Some(req.id);
                    if let Err(e) = ws.send_text(&text).await {
                        self.limiter.rebate(now);
                        tracing::warn!(?e, "send failed, will reconnect");
                        return;
                    }
                    self.archive.record(
                        Direction::Outbound, text.as_bytes(), now_ns());
                    self.pending.pop_front();
                }
                Err(_wait) => break, // rate limited; try next tick
            }
        }
    }
}
```

Key patterns:

- **`tokio::select!` with a 1ms timer** lets us both drain pending
  outbound and receive inbound. The 1ms gives the rate limiter a
  chance to unblock pending sends.
- **Rebate on send failure.** If the WS errored after we consumed
  a token but before the bytes actually left, the token gets
  refunded so we don't artificially throttle ourselves.
- **`last_pending_id` captures the mid-flight state.** On reconnect
  we re-queue it with the same id for idempotency.

---

## 7. Heartbeat / keepalive

Most exchanges require you to send a ping every 20-30 seconds or
they'll drop you. `WsStream` handles low-level `Ping` frames
automatically, but some exchanges want an application-level heartbeat
too.

```rust
async fn heartbeat(ws: &mut WsStream<MaybeTls>)
    -> Result<(), nexus_async_web::ws::WsError>
{
    let msg = r#"{"type":"ping"}"#;
    ws.send_text(msg).await
}
```

Run this on a separate timer branch in the `select!`. If the exchange
misses a heartbeat reply, trigger a reconnect even though the socket
still looks alive — it's wedged.

---

## 8. Multiple venues

If you're trading on multiple venues, don't write N connection
state machines inline — define one `Exchange` struct (as above) and
spawn one per venue on the executor. Each venue gets its own rate
limiter, archive writer, and id factory.

> NOTE: `nexus-async-web` does not ship a `ClientPool` abstraction;
> venue multiplexing is user code. The pattern is: hold a
> `HashMap<VenueId, Exchange>` in your supervisor task, spawn each
> `Exchange::run()` as a `tokio::task::spawn_local` future, and merge
> their inbound channels into a single handler.

---

## 9. Gotchas

- **Don't reuse client order IDs across deployments.** Snowflake
  embeds worker id + timestamp, so this is handled if you set worker
  id per process. Verify on deploy.
- **Archive is not durable until it's on disk.** The log buffer is
  in-memory. If the process dies, the tail of the log is lost.
  Budget for `fsync` frequency — every second is usually fine for
  replay, but compliance requirements may demand more.
- **Rate limiters are stateful across reconnects.** Do NOT reset
  the limiter on reconnect — the exchange's counter didn't reset.
- **Heartbeat != connection liveness.** A socket can be open, the
  kernel TCP keepalive can be happy, and the exchange endpoint can
  be wedged. App-level heartbeats are the only reliable signal.
- **Binary JSON decoders** (sonic-rs, simd-json) are worth it for
  order-entry feedback. The parse latency is on the critical path.
- **Don't send one large message per token.** Batch is a tempting
  optimization, but exchanges often count individual orders, not
  frames. Check the published policy.

---

## Further reading

- `nexus-async-web/docs/` — `WsStream`, `WsStreamBuilder`, TLS config
- `nexus-net/docs/` — sans-IO layer (non-tokio use cases)
- `nexus-rate/docs/` — algorithm selection, checked Duration,
  token rebate semantics
- `nexus-id/docs/` — Snowflake layout, worker id allocation
- `nexus-logbuf/docs/` — claim API, skip markers, reader semantics
- [cookbook-market-data-gateway.md](./cookbook-market-data-gateway.md)
  — the inbound counterpart to this cookbook
- [cookbook-latency-monitoring.md](./cookbook-latency-monitoring.md)
  — measuring RTT on this connection

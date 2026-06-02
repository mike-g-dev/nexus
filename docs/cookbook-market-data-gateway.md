# Cookbook: Market Data Gateway

**Goal:** build a gateway that connects to an exchange WebSocket feed,
parses frames, archives raw bytes, publishes the latest price, fans
out to multiple consumers, and monitors feed health.

**Crates used:**
`nexus-async-web` (WebSocket), `nexus-async-rt` (executor),
`nexus-queue` (SPMC fan-out), `nexus-slot` (latest-price conflation),
`nexus-logbuf` (archival), `nexus-stats` (feed health),
`nexus-id` (message IDs), `nexus-ascii` (symbol).

This example uses Binance-style message shapes but everything here
applies to Coinbase, OKX, Bybit, and friends. It is illustrative, not
a complete library.

---

## Architecture

```
   ┌──────────────┐      ┌──────────────┐
   │ Binance WS   │──────▶│ Reader task  │
   │  (TLS)       │      │ (nexus-web)  │
   └──────────────┘      └──────┬───────┘
                                │
          ┌─────────────┬───────┼────────┬────────────┐
          ▼             ▼       ▼        ▼            ▼
   ┌────────────┐ ┌──────────┐ ┌────┐ ┌──────┐ ┌────────┐
   │ logbuf     │ │ slot     │ │SPMC│ │stats │ │trace   │
   │ (archive)  │ │ (latest) │ │fan │ │health│ │counters│
   └────────────┘ └──────────┘ │out │ └──────┘ └────────┘
                               └────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
                 book          signal       risk
                 builder       calc         monitor
```

Reader task is the **only writer** on the queues and slot. That's
the single-writer principle doing its job.

---

## Step 1. Shared types

```rust
use nexus_ascii::AsciiString16;
use nexus_id::SnowflakeId64;

// Our chosen Snowflake layout: 42 bits timestamp, 6 bits worker, 16 bits sequence.
// Users pick the layout per-deployment; nexus-id does not ship a default.
pub type TradeIdLayout = nexus_id::Snowflake64<42, 6, 16>;
pub type TradeId = SnowflakeId64<42, 6, 16>;

/// A decoded trade from the wire.
#[derive(Clone, Copy)]
pub struct Trade {
    pub id: TradeId,
    pub symbol: AsciiString16,
    pub price_bits: i64,   // nexus-decimal scaled
    pub qty_bits: i64,
    pub ts_exchange_ns: u64,
    pub ts_local_ns: u64,
}

/// Latest top-of-book for a symbol.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Tob {
    pub bid_bits: i64,
    pub ask_bits: i64,
    pub ts_exchange_ns: u64,
    pub ts_local_ns: u64,
}
```

`nexus-slot` requires `Pod` (no heap, no padding UB), so `Tob` is
`#[repr(C)]` with only integer fields. `nexus-ascii`'s
`AsciiString16` is `Copy` and precomputes an XXH3 hash, so it makes
a good symbol key.

---

## Step 2. Wire the primitives up at startup

All allocation happens here. After this point no hot-path code calls
`malloc`.

```rust
use nexus_logbuf::spsc as logbuf_spsc;
use nexus_queue::spmc;
use nexus_slot::spsc as slot_spsc;
use nexus_stats::{monitoring::EventRateF64, statistics::PercentileF64};

pub struct Gateway {
    // Archival — raw WS frames, off the hot path, for replay/compliance.
    archive_tx: logbuf_spsc::Producer,

    // Latest top-of-book. SPSC: reader task writes, book snapshots
    // are conflated so slow consumers never block us.
    tob_writer: slot_spsc::Writer<Tob>,
    tob_reader: slot_spsc::Reader<Tob>,

    // Fan-out of trades to downstream consumers (book, signal, risk).
    // SPMC: one writer, many readers. Bounded — if a reader falls
    // behind, they lose samples (intentional, we don't pay for MPMC).
    trade_fanout: spmc::Producer<Trade>,
    trade_rx: spmc::Consumer<Trade>, // clone() this per consumer

    // Health metrics. Updated by reader task.
    inter_arrival: PercentileF64,
    msg_rate: EventRateF64,
}

impl Gateway {
    pub fn new() -> Self {
        // fixed: logbuf uses `spsc::new`, not `log_buf`
        let (archive_tx, _archive_rx) = logbuf_spsc::new(1 << 20);
        // Spawn _archive_rx on a cold thread — see step 4.

        // fixed: slot ctor is `spsc::slot::<T>() -> (Writer, Reader)`
        let (tob_writer, tob_reader) = slot_spsc::slot::<Tob>();

        // fixed: `spmc::ring_buffer` returns (Producer, Consumer); Consumer is Clone.
        let (trade_fanout, trade_rx) = spmc::ring_buffer::<Trade>(4096);

        Self {
            archive_tx,
            tob_writer,
            tob_reader,
            trade_fanout,
            trade_rx,
            inter_arrival: PercentileF64::new(0.999).unwrap(),
            // fixed: EventRateF64 uses a builder and takes an alpha smoothing
            // factor; it does not accept a Duration.
            msg_rate: EventRateF64::builder().alpha(0.1).build().unwrap(),
        }
    }
}
```

Things to notice:

- **SPMC producer, not MPMC.** Reader task is the only writer. The
  consumers each hold a `spmc::Consumer<Trade>`.
- **Slot is SPSC.** Only the reader writes it. A book-builder,
  position monitor, etc. all share a single reader (or copy it per
  consumer since it's `Copy`).
- **Logbuf is SPSC.** The reader task is the only writer, a dedicated
  archive thread is the only consumer. No MPMC needed.
- **`Percentile` targets p999.** Because `is_primed()` is target-aware,
  it won't report until it's seen at least 1000 samples.

---

## Step 3. The reader task

This is the only place IO happens. It's async because we're on
`nexus-async-rt`, which gives us `.await` without losing control
over threading.

```rust
// fixed: nexus-async-web exports `WsStreamBuilder`, not `WebSocketClient`.
// `WsStreamBuilder::new().connect(url).await` returns `WsStream<MaybeTls>`.
use nexus_async_web::ws::{WsStreamBuilder, Message};

async fn reader_task(
    gw: &mut Gateway,
    // The user supplies their own Snowflake layout — see Step 1.
    ids: &mut TradeIdLayout,
    epoch: std::time::Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    let url = "wss://stream.binance.com:9443/ws/btcusdt@trade";
    let mut ws = WsStreamBuilder::new().connect(url).await?;

    // fixed: `recv()` returns Result<Option<Message<'_>>, WsError>.
    while let Some(msg) = ws.recv().await? {
        let (text, _is_text) = match msg {
            Message::Text(s) => (s, true),
            Message::Binary(_) => continue, // skip
            // WsStream auto-responds to pings internally; this arm is
            // defensive.
            Message::Ping(p) => { ws.send_pong(p).await?; continue; }
            Message::Close(_) => break,
            _ => continue,
        };

        let now_ns = now_nanos();

        // 1. Archive first. If anything downstream panics, raw bytes
        //    are already persisted — replay is possible.
        {
            // fixed: logbuf Producer uses `try_claim(len) -> Result<WriteClaim, _>`.
            let mut claim = gw.archive_tx.try_claim(text.len())
                .expect("archive logbuf sized too small");
            // WriteClaim derefs to &mut [u8].
            claim[..text.len()].copy_from_slice(text.as_bytes());
            claim.commit();
        }

        // 2. Parse. Use sonic-rs or simd-json for real code; this is
        //    a stand-in.
        let Some((symbol, price_bits, qty_bits, ts_exch)) = parse_trade(text) else {
            continue;
        };

        // fixed: Snowflake64::next_id takes a `tick: u64` and returns
        // Result<SnowflakeId64<...>, SequenceExhausted>.
        let tick = (std::time::Instant::now() - epoch).as_millis() as u64;
        let id = ids.next_id(tick).expect("sequence exhausted");

        let trade = Trade {
            id,
            symbol,
            price_bits,
            qty_bits,
            ts_exchange_ns: ts_exch,
            ts_local_ns: now_ns,
        };

        // 3. Update health metrics — feed them the local arrival time.
        //    p999 of inter-arrival catches stalls before the feed
        //    stops entirely.
        gw.inter_arrival.update((now_ns as f64) / 1e9).ok();
        // fixed: EventRateF64::update takes an f64 timestamp.
        gw.msg_rate.update(now_ns as f64 / 1e9).ok();

        // 4. Publish to SPMC. If the queue is full, we drop — a slow
        //    downstream does not block the reader.
        // fixed: SPMC Producer uses `push(&self, T) -> Result<(), Full<T>>`.
        let _ = gw.trade_fanout.push(trade);

        // 5. Update the conflation slot with the new top-of-book.
        //    (In a real gateway you'd parse depth updates here too.)
        let tob = Tob {
            bid_bits: price_bits - 1, // placeholder
            ask_bits: price_bits + 1,
            ts_exchange_ns: ts_exch,
            ts_local_ns: now_ns,
        };
        // fixed: slot::spsc::Writer::write(&mut self, T).
        gw.tob_writer.write(tob);
    }
    Ok(())
}
```

Gotchas:

- **Archive first.** If anything after this panics, you still have
  the bytes. Replay from archive is the recovery story.
- **`push` is non-blocking and fallible.** A full SPMC means a downstream is
  slow. You *will not* block the reader on that. Drop samples,
  bump a counter, alert on it. Blocking the reader means no market
  data — stale data is recoverable, dead is not.
- **Slot has no backpressure by design.** Writer always wins,
  reader gets the latest. If a consumer needs every sample it
  reads from the SPMC fan-out, not from the slot.
- **Monitor inter-arrival, not just rate.** `EventRate` tells you
  average throughput. `PercentileF64` on inter-arrival times tells
  you the p999 gap — the gap is where stalls hide.

---

## Step 4. The archive consumer (cold thread)

A dedicated OS thread drains the logbuf to disk. It's not on the
hot path so we use blocking IO.

```rust
use std::io::Write;

// fixed: `read_blocking` does not exist on the raw queue Consumer.
// Use the blocking `channel::spsc::Receiver` if you want a timeout, or
// the raw queue `Consumer::try_claim()` spin below. Here we use the
// raw queue variant paired with the Producer from Step 2.
fn archive_thread(
    mut rx: nexus_logbuf::spsc::Consumer,
    mut file: std::fs::File,
) {
    loop {
        match rx.try_claim() {
            Some(claim) => {
                // ReadClaim derefs to &[u8].
                file.write_all(&claim).expect("archive write failed");
                // claim is released on drop.
            }
            None => {
                if rx.is_disconnected() { break; }
                std::thread::yield_now();
            }
        }
    }
}
```

Dedicated thread, not a tokio task — you don't want the archive to
share a core with the reader. `taskset` pins the reader to core 2
and the archive thread to core 6 (or wherever your topology puts a
non-sibling physical core).

---

## Step 5. Consumer example — book builder

Downstream consumers look like any ordinary loop. They take their
own `spmc::Consumer<Trade>` and drain.

```rust
use nexus_queue::spmc;
use nexus_slot::spsc as slot_spsc;

fn book_builder(
    trades: spmc::Consumer<Trade>,
    mut tob_snapshot: slot_spsc::Reader<Tob>,
) {
    loop {
        // fixed: SPMC Consumer::pop returns Option<T> (no `try_pop`).
        if let Some(trade) = trades.pop() {
            apply_trade_to_book(trade);
        } else if let Some(tob) = tob_snapshot.read() {
            // Periodic reconcile against the latest TOB.
            reconcile_book(tob);
        } else {
            std::hint::spin_loop();
        }
    }
}
```

If `trades` fills up (the reader is overwhelming us), we lose
samples. That's a visible failure — you monitor
`trades.pop()` returning `None` alongside a non-empty reader,
or you count missed sequence numbers.

---

## Step 6. Health monitoring

Expose the stats you collected in step 3 so an ops dashboard can
scrape them.

```rust
pub struct GatewayHealth {
    pub msg_rate_per_sec: f64,
    pub inter_arrival_p999_ms: f64,
    pub archive_lag_bytes: usize,
    pub trades_dropped: u64,
}

impl Gateway {
    pub fn health(&self) -> GatewayHealth {
        GatewayHealth {
            // fixed: EventRateF64::rate() returns Option<f64>.
            msg_rate_per_sec: self.msg_rate.rate().unwrap_or(0.0),
            // fixed: PercentileF64 query is `.percentile()`, not `.value()`.
            inter_arrival_p999_ms: self
                .inter_arrival
                .percentile()
                .unwrap_or(0.0) * 1000.0,
            // NOTE: raw queue Producer does not expose an `available_to_read`
            // helper; you would track archive lag by sampling the Consumer
            // side or maintaining a counter on the Producer side.
            archive_lag_bytes: 0,
            trades_dropped: /* counter from push failures */ 0,
        }
    }
}
```

A realistic alert is "p999 inter-arrival > 500ms" — that's a
feed that's stuck. An alert on "msg rate == 0" fires too late.

---

## Reconnect and sequence numbers

Real exchange feeds have **sequence numbers** on every message. When
you reconnect, you must:

1. Resume from the last sequence number (or re-snapshot the book
   and resume live).
2. Detect gaps — if the seq number jumps, the book is stale.
3. Re-emit a "book invalidated" event so downstream consumers
   drop their local state.

`nexus-rt` and `nexus-async-rt` have nothing to say about this — it's
a protocol-level concern. The cookbook here isn't complete without
a gap-handling state machine, but the gap-handling state machine is
**per-exchange**, so it doesn't belong in a reusable cookbook.

The pattern: wrap the reader task in a supervising loop that catches
disconnects, emits a `BookInvalidated` event on the SPMC fan-out,
and reconnects. Downstream consumers watch for that event and reset.

---

## Further reading

- `nexus-async-web/docs/` — full WebSocket API
- `nexus-net/docs/` — sans-IO layer if you aren't on tokio
- `nexus-logbuf/docs/` — claim-based write API details
- `nexus-slot/docs/` — Pod requirements, ordering guarantees
- `nexus-queue/docs/` — SPMC vs MPSC vs SPSC tradeoffs
- [cookbook-exchange-connection.md](./cookbook-exchange-connection.md)
  — reconnect, TLS, rate limiting
- [cookbook-latency-monitoring.md](./cookbook-latency-monitoring.md)
  — instrumenting feed latency end to end

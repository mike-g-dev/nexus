# WebSocket Performance Tuning

nexus-async-web exposes three knobs that control the latency/throughput
tradeoff on the recv path. This guide explains what each does, how they
interact, and how to tune them for common workloads.

## The three knobs

### `buffer_capacity` (FrameReaderBuilder)

Total size of the ReadBuf — a flat byte slab that holds inbound wire
data. Frames are parsed in-place (zero-copy for single-frame messages).

- **Default:** 1 MB
- **Set via:** `WsStreamBuilder::buffer_capacity(n)` or
  `FrameReader::builder().buffer_capacity(n)`

**Larger buffer:**
- Fewer reads from the transport (more frames buffered per read)
- Higher memory footprint per connection
- Compaction moves more data when triggered

**Smaller buffer:**
- Lower memory per connection
- More frequent reads (each smaller)
- Compaction is cheaper (less data to move)

**Guidance:** Size the buffer to hold several seconds of peak message
throughput. For a market data feed delivering 10K msgs/sec at 128B:
~1.3 MB/sec. A 256 KB buffer holds ~200ms of data — plenty.

### `compact_at` (FrameReaderBuilder)

Fraction of buffer capacity consumed before proactive compaction
triggers. As messages are parsed, the read cursor advances through
the buffer. Eventually the writable region (spare) at the tail
runs out. Compaction reclaims space by moving unconsumed data
(typically a partial frame — a few bytes) to the front.

- **Default:** 0.5 (compact when half the buffer is consumed)
- **Set via:** `FrameReader::builder().compact_at(fraction)`
- **Range:** 0.0 to 1.0

At 0.5, compaction triggers after the cursor has passed the halfway
mark. The memmove is cheap — it only moves the unconsumed bytes
(partial frame data, not the whole buffer).

**Lower values (e.g., 0.25):**
- Compaction happens more often
- Each compaction moves less data
- Spare region is always large

**Higher values (e.g., 1.0):**
- Compaction only when spare is completely empty
- Fewer compactions, but each is larger
- Risk of a single expensive compaction stalling a message

**Guidance:** 0.5 is well-validated (Databento uses the same strategy).
Leave it unless you have a specific reason to change.

### `max_read_size` (WsStreamBuilder)

Maximum bytes read from the transport per recv() call. Caps the slice
passed to the underlying socket read, bounding the worst-case memcpy
per message.

- **Default:** `buffer_capacity / 8`
- **Set via:** `WsStreamBuilder::max_read_size(n)`
- **Clamped to:** `[1, buffer_capacity]`

This is the primary latency tuning knob. Without a cap, a single
read could fill the entire buffer — a large memcpy that stalls one
message with all the cost.

**Smaller values (e.g., 8 KB):**
- Each read is fast and predictable
- More reads per buffer fill
- Better tail latency (bounded worst-case)
- Slightly more overhead from frequent reads

**Larger values (e.g., 128 KB):**
- Fewer reads (better syscall amortization)
- Higher throughput for bulk transfers
- Worse tail latency (occasional large memcpy)

**Guidance:** Set to at least your largest expected message size.
For market data with book snapshots up to 16 KB, use 32 KB.
For pure tick data (< 200B), 8-16 KB is fine.

## Workload profiles

### Low-latency market data (ticks, deltas)

Messages: 40-200 bytes, 1K-100K/sec. Tail latency matters more
than throughput. No large snapshots.

```rust
WsStreamBuilder::new()
    .buffer_capacity(64 * 1024)    // 64 KB — fits in L2
    .max_read_size(8 * 1024)       // 8 KB — bounded reads
    // compact_at defaults to 0.5
    .connect("ws://exchange/ws")
    .await?;
```

Expected: p50 ~60 cycles, p99 ~200 cycles, p99.9 ~400 cycles.

### Market data with book snapshots

Messages: 40-200 bytes (ticks) + 4-64 KB (snapshots). Need the
buffer large enough for snapshots, reads large enough to receive
them in one shot.

```rust
WsStreamBuilder::new()
    .buffer_capacity(256 * 1024)   // 256 KB
    .max_read_size(32 * 1024)      // 32 KB — covers any snapshot
    .connect("ws://exchange/ws")
    .await?;
```

### High-throughput bulk transfer

Maximizing msg/sec for data replay, backtesting, or archival.
Tail latency is acceptable if throughput is high.

```rust
WsStreamBuilder::new()
    .buffer_capacity(1024 * 1024)  // 1 MB
    .max_read_size(128 * 1024)     // 128 KB — large reads, fewer syscalls
    .connect("ws://source/stream")
    .await?;
```

### Memory-constrained (many connections)

Hundreds or thousands of WS connections. Minimize per-connection
memory. Accept more frequent reads.

```rust
WsStreamBuilder::new()
    .buffer_capacity(16 * 1024)    // 16 KB per connection
    .max_read_size(4 * 1024)       // 4 KB reads
    .connect("ws://source/stream")
    .await?;
```

## How recv() uses these knobs

Each call to `recv()`:

1. **Parse buffered data.** If a complete message is already in the
   ReadBuf, return it immediately. No read, no copy — this is the
   fast path (~40-60 cycles).

2. **Proactive compaction.** If consumed bytes exceed `compact_at *
   buffer_capacity`, move unconsumed data (partial frame) to the
   front. Cost: O(partial_frame_bytes), typically < 100 cycles.

3. **Fallback compaction.** If spare is empty after proactive check
   (buffer genuinely full of unconsumed data), compact and retry.

4. **Read from transport.** Read up to `min(spare_len, max_read_size)`
   bytes from the socket into the ReadBuf's spare region.

5. **Loop.** Go back to step 1 — the new data may contain one or
   many complete frames.

Most messages hit step 1 only. The read (step 4) happens every
`max_read_size / frame_size` messages. Compaction (step 2) happens
every `buffer_capacity * compact_at / frame_size` messages, but
the cost is trivial since it only moves partial-frame bytes.

## Measuring

The crate includes cycle-level benchmarks with realistic TCP mocks:

```bash
# Build
cargo build --release -p nexus-async-web --example perf_ws_cycles_tokio
cargo build --release -p nexus-async-web --no-default-features \
    --features nexus --example perf_ws_cycles_nexus

# Run (pin to one core for stable results)
taskset -c 0 ./target/release/examples/perf_ws_cycles_tokio
taskset -c 0 ./target/release/examples/perf_ws_cycles_nexus

# Throughput (msg/sec)
cargo run --release -p nexus-async-web --example perf_async_ws
cargo run --release -p nexus-async-web --no-default-features \
    --features nexus --example perf_async_ws_nexus
```

The cycle benchmarks report p50/p90/p99/p99.9/max for both
per-message and batched (x64 amortized) measurements. Tune
the knobs in `make_ws()` to test different configurations.

## Benchmark results

All results measured on pinned core, turbo disabled, cycle-level
rdtsc with serializing fences. Throughput measured with wall-clock
over 1M messages.

### Throughput (in-memory, zero-copy recv)

| payload | nexus-rt async | tokio-rt async | blocking | tungstenite |
|---------|---------------|---------------|----------|-------------|
| 40B     | **80.9M msg/sec** (12ns) | 51.3M (19ns) | 66.3M (15ns) | 9.7M (104ns) |
| 128B    | **55.6M msg/sec** (18ns) | 38.2M (26ns) | 48.7M (21ns) | 8.6M (117ns) |
| 512B    | **23.1M msg/sec** (43ns) | 20.3M (49ns) | 21.2M (47ns) | 6.6M (151ns) |

### Throughput over TCP loopback (tokio, single-threaded)

| payload | nexus-async-web | tungstenite | speedup |
|---------|----------------|-------------|---------|
| 40B     | 51.5M msg/sec (19ns) | 7.9M (126ns) | **6.5x** |
| 128B    | 31.5M msg/sec (32ns) | 6.9M (145ns) | **4.6x** |

### Throughput over TLS loopback (tokio)

| payload | nexus-async-web | tungstenite | speedup |
|---------|----------------|-------------|---------|
| 40B     | 29.4M msg/sec (34ns) | 8.3M (120ns) | **3.5x** |
| 128B    | 12.3M msg/sec (81ns) | 5.5M (180ns) | **2.2x** |

### Latency distribution (nexus-rt, per-message, cycles)

Measured with realistic TCP mock (1460-byte segments, L1-hot wire
data, 256 KB buffer, compact@50%).

| payload | p50 | p90 | p99 | p99.9 | p99/p50 |
|---------|-----|-----|-----|-------|---------|
| 40B     | 62  | 126 | 212 | 420   | 3.4x    |
| 128B    | 56  | 86  | 182 | 214   | 3.3x    |
| 1024B   | 136 | 146 | 174 | 292   | 1.3x    |

### Latency distribution (nexus-rt, batched x64, amortized cycles/msg)

| payload | p50 | p90 | p99 | p99.9 |
|---------|-----|-----|-----|-------|
| 40B     | 40  | 43  | 68  | 155   |
| 128B    | 52  | 56  | 87  | 168   |

### Latency distribution (tokio-rt, per-message, cycles)

| payload | p50 | p90 | p99 | p99.9 | p99/p50 |
|---------|-----|-----|-----|-------|---------|
| 40B     | 74  | 78  | 206 | 270   | 2.8x    |
| 128B    | 74  | 118 | 218 | 278   | 2.9x    |
| 1024B   | 164 | 178 | 206 | 312   | 1.3x    |

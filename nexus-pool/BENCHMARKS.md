# nexus-pool Benchmarks

Cycle-accurate latency on Intel Core Ultra 7 165U, pinned to physical core.
All values in cycles per operation.

Pool objects are `Vec<u8>` with `capacity(64)`, reset via `Vec::clear`.
Release timings include the reset call.

## local::BoundedPool (fixed capacity, pre-initialized)

| Operation | p50 | p99 | p999 |
|-----------|-----|-----|------|
| try_acquire (empty pool) | 26 | 30 | 44 |
| release (empty pool) | 26 | 56 | 60 |
| try_acquire (50% held) | 26 | 30 | 42 |
| release (50% held) | 26 | 56 | 62 |

## local::Pool (growable, RAII)

| Operation | p50 | p99 | p999 |
|-----------|-----|-----|------|
| try_acquire (reuse) | 26 | 28 | 36 |
| release (reuse) | 26 | 56 | 60 |
| acquire (factory) | 32 | 38 | 48 |
| release (factory) | 26 | 32 | 52 |

`take()`/`put()` bypass the Rc/Weak bookkeeping in the RAII path. Expected
to be ~15-25% faster per round trip (eliminates 3 Cell read-modify-writes,
2 branches, and a `ManuallyDrop::take` copy per cycle).

## sync::Pool (one acquirer, any returner)

### Sequential access (channel-based)

| Scenario | Acquire p50 | Release p50 | Release p99 | Release p999 |
|----------|-------------|-------------|-------------|--------------|
| same thread | 50 | 62 | 170 | 174 |
| 1 returner | 52 | 64 | 80 | 134 |
| 2 returners | 52 | 66 | 86 | 120 |
| 4 returners | 52 | 66 | 88 | 142 |

### Concurrent return (CAS contention)

| Threads | Release p50 | Release p99 | Release p999 |
|---------|-------------|-------------|--------------|
| 1 | 68 | 88 | 104 |
| 2 | 66 | 74 | 98 |
| 4 | 66 | 82 | 110 |

Acquire is always single-threaded (42-52 cycles p50). Release scales
well — CAS contention does not degrade p50 even at 4 concurrent returners.

## Running Benchmarks

```bash
cargo build --release --benches -p nexus-pool
taskset -c 0 ./target/release/deps/perf_local_pool-*
taskset -c 0 ./target/release/deps/perf_sync_pool-*
```

# nexus-id Performance Benchmarks

CPU cycles measured via `rdtscp` on x86_64. Pinned to a single core via
`taskset -c 0` for stable measurements.

**System:** Intel Core Ultra 7 165U (Meteor Lake). 1M iterations, 10k warmup.

Run benchmarks yourself:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release -p nexus-id --benches
taskset -c 0 ./target/release/deps/perf_benchmark-*
taskset -c 0 ./target/release/deps/perf_snowflake-*
taskset -c 0 ./target/release/deps/perf_uuid-*
taskset -c 0 ./target/release/deps/perf_id_hashing-*
```

Numbers below reflect SSSE3-enabled builds (`target-cpu=native` or
`target-feature=+ssse3`). Without SSSE3, hex encode falls back to scalar;
SSE2 decode is always available on x86_64.

---

## ID Generation

### Snowflake

| Operation | p50 | p99 | p999 | Notes |
|-----------|-----|-----|------|-------|
| `next()` new timestamp | 22 | 48 | 256 | Sequence reset path |
| `next()` same timestamp | 22 | 24 | 28 | Sequence increment (burst) |

All layouts (`<42,6,16>`, `<41,10,12>`, `<20,4,8>`) perform identically at p50.

### UUID

| Operation | p50 | p99 | p999 | Notes |
|-----------|-----|-----|------|-------|
| `UuidV4::next_raw()` | 22 | 38 | 54 | Returns `(u64, u64)` |
| `UuidV4::next()` | 48 | 106 | 304 | Returns `Uuid` (36-char, SSSE3 encode) |
| `UuidV4::next_compact()` | 32 | 72 | 94 | Returns `UuidCompact` (32-char) |
| `UuidV7::next_raw()` same ts | 30 | 34 | 34 | Monotonic sequence path |
| `UuidV7::next()` same ts | 60 | 68 | 94 | Returns `Uuid` |
| `UuidV7::next()` new ts | 62 | 74 | 144 | Timestamp advanced |

### ULID

| Operation | p50 | p99 | p999 | Notes |
|-----------|-----|-----|------|-------|
| `next()` same timestamp | 80 | 102 | 138 | Monotonic increment |
| `next()` new timestamp | 80 | 112 | 144 | Timestamp advanced |

ULID is slower than UUID due to Crockford Base32 encoding (26 chars, 5-bit
groups) vs hex encoding (32/36 chars, SIMD-accelerated).

---

## Newtype Operations

| Operation | p50 | p99 | p999 | Notes |
|-----------|-----|-----|------|-------|
| `mixed()` | 22 | 46 | 220 | Fibonacci multiply (~1 cycle, measurement floor) |
| `unmix()` | 20 | 28 | 40 | Inverse multiply |
| `unpack()` | 20 | 24 | 36 | 3 shifts + masks |

These are at the `rdtscp` measurement floor (~20 cycles). The actual operation
cost is 1-3 cycles; the rest is measurement overhead.

---

## String Encoding

| Operation | p50 | p99 | p999 | Notes |
|-----------|-----|-----|------|-------|
| `HexId64::encode(u64)` | 26 | 34 | 72 | SSSE3 pshufb, 16 chars |
| `SnowflakeId64::to_hex()` | 20 | 28 | 32 | Same (at measurement floor) |
| `Base62Id::encode(u64)` | 62 | 70 | 92 | Digit-pair decomposition |
| `SnowflakeId64::to_base62()` | 42 | 66 | 72 | Same as above (inlined) |
| `Base36Id::encode(u64)` | 66 | 70 | 88 | Digit-pair decomposition |

Hex encoding uses SSSE3 `pshufb` as a 16-entry LUT in an XMM register (falls
back to scalar lookup table on non-SSSE3 targets). Base62/36 use digit-pair
decomposition that reduces division count (5 divmod ops for base62, 6 for base36).

---

## String Parsing

| Operation | p50 | p99 | p999 | Notes |
|-----------|-----|-----|------|-------|
| `HexId64::parse()` | 42 | 46 | 58 | SSE2 parallel range classify |
| `UuidCompact::parse()` | 48 | 58 | 142 | SSE2 (2x 16-char decode) |
| `Uuid::parse()` | 70 | 76 | 108 | SSSE3 dash compaction + SSE2 decode |
| `Ulid::parse()` | 90 | 110 | 212 | 26-char Crockford (256-byte LUT) |
| `Base62Id::parse()` | 86 | 106 | 246 | 11-char with multiply-accumulate |
| `Base36Id::parse()` | 108 | 120 | 296 | 13-char |
| `TypeId::parse()` | 136 | 198 | 316 | Prefix + ULID suffix |

Hex-based parsing uses SSE2 parallel range classification (x86_64 baseline).
UUID dashed parsing uses SSSE3 `pshufb` to compact dashes in-register before
SSE2 decode. All parsing is single-pass with no allocation.

---

## TypeId

| Operation | p50 | p99 | p999 | Notes |
|-----------|-----|-----|------|-------|
| `TypeId::new("user", ulid)` | 56 | 78 | 88 | Construct from prefix + ULID |
| `TypeId::parse("user_...")` | 136 | 198 | 316 | Full string parse |
| `TypeId::prefix()` | 22 | 26 | 40 | Slice into stored string |

---

## Combined Operations

| Operation | p50 | p99 | p999 | Notes |
|-----------|-----|-----|------|-------|
| `next_id() + to_hex()` | 28 | 34 | 54 | Generate + SSSE3 hex format |
| `next_id() + to_base62()` | 64 | 130 | 140 | Generate + base62 format |
| `next_id() + mixed()` | 22 | 24 | 34 | Generate + hash-ready |

The common hot-path pattern — generate a snowflake and mix it for HashMap
lookup — costs 22 cycles (p50). This is the cost of two multiplies.

---

## HashMap Performance

Demonstrates why bit distribution matters for hash table performance.

**Setup:** 100k IDs inserted, 1M random lookups measured.

### Lookup Latency (cycles)

| ID Pattern | Identity | FxHash | AHash |
|------------|----------|--------|-------|
| Snowflake (sequential bits) | 3535 | 52 | 60 |
| Sequential u64 | 30 | 64 | 60 |

Snowflake IDs with identity hashers are **catastrophic** — 3535 cycles/lookup
due to clustering in power-of-2 bucket tables. Use either:
1. A real hasher (FxHash, AHash) — 52-64 cycles
2. `MixedId64` with identity hasher — distributes bits uniformly

### Insert Throughput (cycles/insert)

| ID Pattern | Identity | FxHash |
|------------|----------|--------|
| Snowflake (sequential bits) | 3438 | 16 |
| Sequential u64 | 16 | 14 |

---

## Cost Summary

| What you're doing | p50 cycles | Recommendation |
|-------------------|-----------|----------------|
| Generate a numeric ID | 22 | `Snowflake64::next_id()` |
| Generate + hash-ready | 22 | `next_id() + mixed()` |
| Generate + hex string | 28 | `next_id() + to_hex()` |
| Generate a UUID v4 | 48 | `UuidV4::next()` |
| Generate a UUID v7 | 62 | `UuidV7::next()` |
| Generate a ULID | 80 | `UlidGenerator::next()` |
| Parse a hex ID | 42 | `HexId64::parse()` |
| Parse a UUID string | 70 | `Uuid::parse()` |
| Parse a ULID string | 90 | `Ulid::parse()` |
| Mix/unmix for hashing | 20 | At measurement floor |

All operations are allocation-free, stack-only, and syscall-free.
SIMD acceleration (SSE2 decode, SSSE3 encode) is compile-time dispatched on x86_64.

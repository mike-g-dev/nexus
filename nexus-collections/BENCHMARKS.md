# nexus-collections Benchmarks

Batched timing (100 ops per rdtsc pair via `seq!` unroll), pinned to core 0.
All values in CPU cycles per operation.

## List (doubly-linked list, RcSlot handles)

| Operation | p50 | p90 | p99 | p999 |
|-----------|-----|-----|-----|------|
| link_back (empty) | 2 | 2 | 3 | 5 |
| link_back (@1000) | 3 | 4 | 8 | 9 |
| link_front (empty) | 2 | 2 | 3 | 7 |
| pop_front | 3 | 3 | 3 | 4 |
| pop_back | 3 | 3 | 3 | 5 |
| unlink (from front) | 3 | 4 | 6 | 13 |
| unlink (@1000 steady) | 5 | 7 | 11 | 15 |
| try_push_back (alloc+link) | 4 | 4 | 5 | 8 |
| front / back (peek) | <1 | <1 | <1 | 1 |

## Heap (pairing heap, RcSlot handles)

| Operation | p50 | p90 | p99 | p999 |
|-----------|-----|-----|-----|------|
| link (empty) | 6 | 7 | 11 | 17 |
| link (@1000) | 6 | 7 | 10 | 17 |
| pop (from 100) | 110 | 117 | 145 | 207 |
| pop (@1000 steady) | 111 | 118 | 149 | 220 |
| unlink (from 100) | 42 | 46 | 54 | 109 |
| unlink (@1000 steady) | 6 | 17 | 30 | 53 |
| try_push (alloc+link) | 9 | 10 | 20 | 33 |
| peek | <1 | <1 | <1 | <1 |

## Sorted Maps — Full Comparison

Three sorted map implementations measured with identical methodology.
Population is 10,000 entries unless noted.

### nexus RbTree (red-black tree, slab-backed, @10k)

| Operation | p50 | p90 | p99 | p999 | max |
|-----------|-----|-----|-----|------|-----|
| get (hit, @100) | 13 | 14 | 22 | 68 | 2244 |
| get (hit, @10k) | 16 | 17 | 26 | 71 | 190 |
| get (miss, @10k) | 35 | 36 | 65 | 116 | 427 |
| get (cold rand, @10k) | 209 | 222 | 278 | 336 | 4218 |
| contains_key (hit) | 16 | 17 | 32 | 75 | 9816 |
| insert (growing, per-op) | 358 | 462 | 562 | 684 | 7404 |
| insert (steady) | 337 | 359 | 429 | 525 | 11213 |
| insert (duplicate) | 23 | 24 | 47 | 81 | 242 |
| remove | 342 | 361 | 433 | 508 | 4434 |
| pop_first | 24 | 26 | 35 | 80 | 415 |
| pop_last | 24 | 26 | 35 | 87 | 710 |
| first_key_value | <1 | <1 | 1 | 1 | 87 |
| churn (remove+insert) | 655 | 699 | 807 | 915 | 13896 |
| entry (occupied) | 22 | 24 | 36 | 77 | 201 |
| entry (vacant+insert) | 335 | 357 | 423 | 495 | 1098 |

### nexus BTree (B=8, slab-backed, @10k)

| Operation | p50 | p90 | p99 | p999 | max |
|-----------|-----|-----|-----|------|-----|
| get (hit, @100) | 24 | 26 | 43 | 105 | 383 |
| get (hit, @10k) | 40 | 65 | 81 | 173 | 814 |
| get (miss, @10k) | 48 | 50 | 86 | 136 | 718 |
| get (cold rand, @10k) | 177 | 196 | 274 | 347 | 4397 |
| contains_key (hit) | 44 | 47 | 79 | 164 | 408 |
| insert (growing, per-op) | 302 | 364 | 666 | 808 | 7994 |
| insert (steady) | 252 | 277 | 365 | 464 | 4233 |
| insert (duplicate) | 48 | 51 | 82 | 135 | 5164 |
| remove | 241 | 257 | 334 | 455 | 4125 |
| pop_first | 49 | 51 | 82 | 167 | 375 |
| pop_last | 50 | 58 | 88 | 164 | 5418 |
| first_key_value | 7 | 7 | 9 | 14 | 207 |
| churn (remove+insert) | 520 | 575 | 714 | 945 | 9203 |
| entry (occupied) | 43 | 45 | 64 | 118 | 284 |
| entry (vacant+insert) | 454 | 488 | 632 | 864 | 11920 |

### std::collections::BTreeMap (baseline, @10k)

| Operation | p50 | p90 | p99 | p999 | max |
|-----------|-----|-----|-----|------|-----|
| get (hit, @100) | 22 | 23 | 31 | 85 | 1808 |
| get (hit, @10k) | 38 | 39 | 44 | 103 | 401 |
| get (miss, @10k) | 47 | 48 | 73 | 153 | 540 |
| get (cold rand, @10k) | 158 | 165 | 220 | 302 | 39680 |
| contains_key (hit) | 37 | 39 | 53 | 115 | 472 |
| insert (growing, per-op) | 238 | 342 | 720 | 5126 | 40916 |
| insert (steady) | 195 | 210 | 274 | 350 | 710 |
| insert (duplicate) | 38 | 41 | 45 | 105 | 11695 |
| remove | 203 | 220 | 284 | 360 | 945 |
| pop_first | 56 | 61 | 80 | 146 | 574 |
| churn (remove+insert) | — | — | — | — | — |
| entry (occupied) | 36 | 40 | 79 | 130 | 3933 |
| entry (vacant+insert) | 225 | 242 | 308 | 408 | 6293 |

### p50 Comparison Matrix

| Operation | nexus RbTree | nexus BTree | std BTreeMap | Best |
|---|---|---|---|---|
| get (hit, @100) | **13** | 24 | 22 | RbTree |
| get (hit, @10k) | **16** | 40 | 38 | RbTree |
| get (miss, @10k) | **35** | 48 | 47 | RbTree |
| contains_key (hit) | **16** | 44 | 37 | RbTree |
| insert (growing) | 358 | **302** | 238 | std |
| insert (steady) | 337 | **252** | 195 | std |
| remove | 342 | **241** | 203 | std |
| pop_first | **24** | 49 | 56 | RbTree |
| churn | 655 | **520** | — | BTree |
| entry (occupied) | **22** | 43 | 36 | RbTree |
| entry (vacant+insert) | **335** | 454 | 225 | std |

### p999 Tail Latency Comparison

| Operation | nexus RbTree | nexus BTree | std BTreeMap | Best |
|---|---|---|---|---|
| get (hit, @100) | **68** | 105 | 85 | RbTree |
| get (hit, @10k) | **71** | 173 | 103 | RbTree |
| insert (growing) | **684** | 808 | 5126 | RbTree |
| insert (steady) | 525 | **464** | 350 | std |
| remove | 508 | **455** | 360 | std |
| pop_first | **80** | 167 | 146 | RbTree |
| entry (occupied) | **77** | 118 | 130 | RbTree |

### Analysis

**Tail latency is the differentiator.** std BTreeMap wins p50 on insert/remove
(195/203 vs nexus's 252/241), but at p999 on growing insert, std explodes to
5126 cycles (global allocator pressure from node splits). nexus RbTree stays at
684. The slab allocator eliminates allocation jitter.

**nexus RbTree strengths**: Lookups (16 cycles at 10k — 40B nodes fit one
cache line), pop operations (24 cycles — cached extremes), entry API,
and growing-insert tail latency.

**nexus BTree strengths**: Remove (241 vs RbTree's 342), churn (520 vs 655),
and steady-state insert (252 vs 337). Contiguous key layout helps.

**std wins p50 on mutation** because it doesn't pay the slab indirection cost.
But it pays at p999 — the global allocator is a shared resource.

### When to Choose Which

**RbTree**: Lookup-heavy workloads (order books), pop-heavy workloads (timer
wheels), entry API patterns, anything where tail latency matters more than
median.

**BTree**: Remove-heavy workloads, high-churn streaming data, range scans.
Tunable branching factor via const generic `B`.

**std BTreeMap**: When you don't need stable handles, don't need O(1) pop,
and median latency matters more than tail.

## Running Benchmarks

```bash
# Disable turbo boost (Intel)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Build
cargo build --release --benches -p nexus-collections

# Run pinned to a physical core
taskset -c 0 ./target/release/deps/perf_list_cycles-*
taskset -c 0 ./target/release/deps/perf_heap_cycles-*
taskset -c 0 ./target/release/deps/perf_rbtree-*
taskset -c 0 ./target/release/deps/perf_btree-*
taskset -c 0 ./target/release/deps/perf_std_btreemap-*

# Re-enable turbo boost
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

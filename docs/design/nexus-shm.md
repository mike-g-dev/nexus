# nexus-shm — Shared Memory Primitives

## Overview

Cross-process communication and durable storage over memory-mapped
files. Same design principles as the existing single-process
primitives (bounded, pre-allocated, cache-line aware, honest about
constraints) extended across the process boundary.

### What's different from single-process

The hard problem isn't the ring buffer — nexus-queue already solves
that. The hard problems are:

1. **Liveness detection.** Telling a dead process from a busy one
   without blocking your own side.
2. **Crash recovery.** A SIGKILL leaves a half-written slot in shared
   memory. No unwinding, no RAII cleanup on the far side.
3. **Memory lifecycle.** Who creates the mapping, who opens it, what
   happens when one side restarts.

These problems shape the memory layout. They must be designed first,
not bolted on.

### Philosophy

**Mechanism, not policy.** The crate provides the data structures,
sequencing, and recovery invariants. It has no opinion on:

- Sync vs async
- Who drives the event loop
- How you poll for new data
- Whether you use epoll, io_uring, or kernel bypass

The core is a passive data structure plus a recovery routine. No
runtime, no threads of its own, no I/O assumptions.

---

## Foundation Layer

All shm primitives sit on a common mmap foundation. Get this right
once, everything above inherits it.

### Design question: separate crate?

**Path A — `nexus-memmap` as its own crate:**
A standalone opinionated wrapper around `memmap2`. Provides
`MappedBuffer` / `MappedBufferMut` with `Pod`-based type-safe
read/write, builder with `huge_pages` and `populate` flags. Single
`unsafe` boundary at `map()`, fully safe API after.

Useful beyond shm — any mmap use case in the ecosystem can depend
on it. Clean separation of concerns.

**Path B — foundation internal to `nexus-shm`:**
The mmap setup, alignment, and lifecycle management is an internal
module of `nexus-shm`. Simpler dependency graph, fewer crates to
publish. The argument: nobody outside shm needs raw mmap primitives,
and if they do, we extract later.

**Tradeoff:** Path A is more general but adds a crate. Path B is
simpler but couples the foundation to one consumer.

### Foundation responsibilities

Regardless of crate structure, the foundation layer handles:

- **mmap lifecycle:** Create, open, resize, unmap. Pre-faulting
  (`MAP_POPULATE`), huge pages (`MAP_HUGETLB`) as opt-in flags.
- **Cache-line alignment:** All shared structures aligned to 64-byte
  boundaries. False sharing prevention between producer and consumer
  metadata.
- **Crash/stale peer detection:** Heartbeat or epoch mechanism to
  distinguish dead peers from slow ones. This is the core invariant
  the upper layers depend on.
- **Recovery routine:** Given a mapping that may contain half-written
  data from a crashed writer, restore to a consistent state.

---

## Primitive Inventory

### 1. ShmRingBuffer — Cross-process ring buffer

The cross-process version of nexus-queue. SPSC first, MPSC if
there's a caller.

**Key differences from nexus-queue:**
- Backing memory is mmap'd, not heap-allocated
- Sequence-based so a dead reader or writer is detectable
- Recovery path for half-written slots when writer dies mid-claim

**API sketch:**
```rust
let (tx, rx) = shm_ring_buffer::<Order>(path, capacity)?;

// Producer side (one process)
if let Some(mut claim) = tx.try_claim() {
    claim.write(order);
    claim.commit();
}

// Consumer side (another process)
if let Some(msg) = rx.try_read() {
    process(msg);
}
```

**Design questions:**
- Lap counters (nexus-queue style) vs sequence numbers (Aeron style)?
- What happens when the consumer is too slow — overwrite (lossy) or
  block (backpressure)? Probably both variants, caller chooses.

### 2. ShmJournal — Append-only segmented log

Durable append-only log for protocol archival, replay, and crash
recovery. Primary caller: FIX message journaling.

**Key properties:**
- Segmented file layout (configurable segment size, default 64MB)
- Writing to the mmap IS the persistence — no separate archival
  thread, no extra copies
- Crash recovery: scan last segment on startup, find the last
  committed record, discard partial writes
- Cross-process read: other processes mmap segments read-only

**Design question: sequence awareness**

**Path A — Position-only journal (Aeron Archive model):**
The journal stores `[len][type_id][payload]` records. It knows
nothing about sequence numbers. Position (byte offset) is the only
addressing mechanism. A separate companion crate
(`nexus-journal-index` or the FIX engine itself) builds the
`sequence -> position` mapping.

Pros: maximally general, zero overhead from indexing on write path,
clean separation of concerns.

Cons: requires a separate indexer for any sequence-based lookup.
Two-crate solution for FIX resend.

**Path B — Sequence-aware journal:**
The journal stores `[len][seq][timestamp][payload]` per record.
Built-in `read_range(start_seq, end_seq)` for replay. Optional
fixed-size ring index for O(1) recent sequence lookup.

Pros: self-contained for FIX use case, simpler integration. One
crate does storage + lookup for recent sequences.

Cons: bakes FIX-like semantics into a general primitive. Sequence
field is overhead for non-FIX callers.

**Path C — Framing is pluggable:**
The journal provides `[len][payload]` framing. A header trait or
generic parameter lets the caller define additional per-record
metadata (sequence, timestamp, type tag, nothing). The journal
reads/writes the header but doesn't interpret it.

Pros: general without being wasteful. FIX puts `(seq, ts)` in the
header, other callers put nothing.

Cons: more complex API surface. Generic parameter may complicate
the type signatures.

**API sketch (Path A):**
```rust
let (writer, reader) = Journal::open(path, config)?;

// Write (hot path, ~memcpy cost)
let mut claim = writer.try_claim(type_id, len)?;
claim.as_mut_slice().copy_from_slice(payload);
claim.commit();

// Read (sequential)
while let Some(record) = reader.next_record() {
    process(record.as_slice());
}

// Seek (cold path, for replay)
reader.seek_to_position(saved_position)?;
```

**API sketch (Path B):**
```rust
let journal = ShmJournal::open(path, config)?;

// Write
journal.append(seq, timestamp, payload)?;

// Read by sequence range (for FIX resend)
let messages = journal.read_range(start_seq, end_seq)?;
```

### 3. ShmSlot — Versioned shared slot (seqlock-style)

Cross-process version of nexus-slot. Latest-wins semantics: writer
always overwrites, reader gets the most recent consistent value.
Lock-free readers, no torn reads.

**Use cases:**
- Config propagation (admin process -> trading process)
- Book snapshots (gateway -> strategy)
- Any "latest value wins" cross-process state

**Key properties:**
- Seqlock pattern: writer increments version before and after write.
  Reader retries if version changed or is odd (mid-write).
- Requires `Pod` types (plain old data, no heap pointers)
- No blocking — reader never waits on writer

**Design question:** How to handle a writer that dies mid-write
(version left odd)? Options: timeout-based staleness detection, or
epoch/heartbeat in a separate field so readers know the writer is
alive.

### 4. ShmMap — Shared memory map (Chronicle-Map style)

Shared key-value store for large shared state across processes.

**Use cases:**
- Covariance matrix shared between risk calc and strategy
- Instrument reference data shared across processes
- Any large structured state where you need random access to
  specific slots

**Design questions:**
- Fixed-key vs dynamic-key? Fixed-key (pre-registered slots) is
  simpler and avoids hash table complexity in shared memory. Dynamic
  key requires a concurrent hash map over mmap.
- Chronicle-Map uses memory-mapped off-heap storage with lock-free
  reads. How much of that complexity do we need vs a simpler
  fixed-slot approach?
- Is this actually a "map" or is it a typed slab in shared memory
  where slots are pre-assigned? The simpler version might be
  sufficient.

---

## Implementation Order

Based on dependency chain:

1. **Foundation** — mmap lifecycle, alignment, crash/stale detection
2. **ShmJournal** — first consumer of foundation, primary caller
   is FIX journaling
3. **ShmRingBuffer** — second consumer, cross-process messaging
4. **ShmSlot** — third consumer, latest-value sharing
5. **ShmMap** — if a concrete caller materializes

## References

- **Aeron** — Triple buffering, log buffer design, archive model.
  Transport abstraction pattern. Java + C implementations.
- **Chronicle-Map** — Lock-free shared memory map. Off-heap storage.
- **nexus-queue** — Existing SPSC ring buffer. Design patterns to
  mirror (lap counters, manual fencing, cache-line padding).
- **nexus-logbuf** — Existing byte ring buffer. Claim-based API
  pattern (WriteClaim/ReadClaim with RAII commit).
- **nexus-slot** — Existing conflation slot. Pod trait, seqlock
  pattern.

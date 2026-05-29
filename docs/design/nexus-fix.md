# nexus-fix — FIX Protocol Toolkit

## Overview

A toolkit for building FIX engines, not a monolithic engine.
Composable primitives that users assemble to match their
architecture — single-process, custom IPC, etc.

Sans-IO throughout: pure state machines, no sockets, no async, no
runtime dependency. Time is injected. Works with mio, tokio,
io_uring, kernel bypass, or a plain blocking thread.

### Crate structure

```
nexus-fix/           -- FIX codec autogen from XML dictionaries
  nexus-fix-macros/  -- proc macro crate (if needed)
nexus-fix-engine/    -- Sans-IO session state machine, framing
```

`nexus-fix` generates the flyweight codecs. `nexus-fix-engine` is
the session layer that uses them.

**Depends on:** `nexus-shm` (ShmJournal for message persistence).

---

## nexus-fix: Codec Generation

### Approach

QuickFIX XML data dictionaries -> Rust flyweight codecs at compile
time.

```rust
// build.rs
fn main() {
    nexus_fix::compile(&[
        ("coinbase", "dictionaries/FIX44_coinbase.xml"),
        ("deribit", "dictionaries/FIX44_deribit.xml"),
    ]).unwrap();
}
```

**Generated per venue:**
- Flyweight decoders: zero-copy, single-pass. Hot fields pre-indexed
  (O(1) access), cold fields via linear scan.
- Encoders: direct buffer write, builder pattern. Caller provides
  the buffer.
- Enum types: `repr(u8)`, exhaustive matching, per-dictionary (no
  cross-venue sharing — if users need unified types, they build their
  own abstraction).

### Design question: codegen vs proc macro

**Path A — build.rs codegen (prost model):**
Generate to `OUT_DIR`, include via `include!`. Standard pattern.
Don't commit generated code.

**Path B — proc macro:**
```rust
#[derive(FixCodec)]
#[fix(dictionary = "FIX44_coinbase.xml")]
struct CoinbaseFix;
```

Pros: better IDE support, no build.rs complexity.
Cons: proc macros reading XML at compile time is unusual and may
have tooling issues.

**Recommendation:** Path A is proven and matches the ecosystem
convention. Path B can be explored later if there's demand.

### Flyweight decoder (zero allocation)

```rust
pub struct NewOrderSingleDecoder<'buf> {
    buffer: &'buf [u8],
    // Pre-indexed hot fields (O(1))
    cl_ord_id: FieldSpan,
    symbol: FieldSpan,
    side: u8,
    price: FieldSpan,
    // Cold fields (linear scan)
    offsets: [FieldOffset; MAX_FIELDS],
    count: u8,
}
```

The flyweight pattern provides zero-copy access over a buffer
segment. The decoder borrows the underlying bytes and provides
typed accessors that interpret field spans without copying.

### Hot-path concern: repeating groups

Most FIX engines fall back to HashMap for repeating groups and give
back the latency earned everywhere else. This needs a purpose-built
solution — likely a small inline array with overflow to a pre-
allocated buffer.

### Shared vs generated types

A key design concern with codegen is the boundary between shared
types and generated types. SBE's codegen, for example, generates
`ReadBuf`/`WriteBuf` wrappers per schema that really should be
shared infrastructure rather than regenerated each time. The codec
generation should keep the buffer access and field encoding
primitives in a shared library crate, generating only the
message-specific flyweight types per dictionary.

---

## nexus-fix-engine: Session Layer

### Sans-IO session state machine

Same pattern as nexus-net's WebSocket codec. Pure state machine
with poll-based event dispatch and caller-owned encode buffer.

```
handle_message(msg, now) -> pushes events to internal buffer
handle_timeout(now)      -> pushes timer events
poll_event()             -> drains events, caller processes
encode_pending(kind, buf)-> encodes into caller's buffer
```

**Key design:** Session never allocates. The caller provides a
"workhorse buffer" for encoding. Events are `Copy` enum variants
(no borrowed data).

### Session state machine

```
Disconnected -> LogonSent -> Active <-> Resending -> LogoutPending -> Disconnected
```

All admin messages (Logon, Logout, Heartbeat, TestRequest,
ResendRequest, SequenceReset, Reject) handled internally by the
state machine. Application messages emitted as events for user code.

### Persistence integration

Uses `nexus-shm`'s ShmJournal for durable message storage.

```
TCP read -> raw bytes
  -> journal.append(...)     <- mmap write, ~ns
  -> fix_codec.parse(bytes)  <- zero-copy from read buffer
  -> session.handle_message  <- state machine
  -> poll_event loop         <- dispatch

Resend request:
  -> journal.read_range(...) <- mmap read, zero-copy
  -> TCP write
```

### Trait-based persistence

Pluggable via traits so users can swap implementations:

```rust
pub trait MessageStore {
    type Error;
    fn store(&mut self, session_id: SessionId, direction: Direction,
             seq_num: u32, msg: &[u8]) -> Result<(), Self::Error>;
    fn retrieve(&self, session_id: SessionId, range: SeqRange)
             -> impl Iterator<Item = Result<StoredMessage<'_>, Self::Error>>;
}

pub trait SessionStore {
    type Error;
    fn load(&self, id: SessionId) -> Result<Option<SessionState>, Self::Error>;
    fn save(&mut self, id: SessionId, state: &SessionState) -> Result<(), Self::Error>;
}
```

Default implementations backed by ShmJournal. Users can implement
these traits for their own storage.

### Performance targets

| Operation | Target |
|-----------|--------|
| SOH scan + checksum | < 200ns |
| Session logic | < 100ns |
| Message store write | < 500ns |
| Full inbound path | < 1us |
| Outbound encode | < 300ns |

### Implementation order

1. **nexus-fix** — Codec generation (XML -> flyweight codecs)
2. **nexus-fix-engine** — Session state machine + framing
3. **Persistence integration** — Wire up ShmJournal via traits

nexus-shm is a prerequisite for step 3. Steps 1-2 can proceed
in parallel with shm work.

---

## FIX versions

Start with FIX 4.2 and 4.4 (dominant in crypto). Design the codegen
to be version-generic so 5.0/FIXT can be added later without
restructuring.

## References

- **SBE (Simple Binary Encoding)** — Flyweight codec generation
  over buffer segments. Good pattern for zero-copy field access.
  Design note: shared buffer primitives (ReadBuf/WriteBuf) should
  live in a common crate rather than being regenerated per schema.
- **prost** — build.rs codegen from schema files (protobuf). The
  `compile()` API pattern and `OUT_DIR` generation model.
- **Artio** — Adaptive's FIX engine. Engine/library separation.
  Session-layer design inspiration.
- **QuickFIX** — Canonical FIX engine. Dictionary XML format is
  the de facto standard for FIX message schemas.
- **nexus-net** — Existing sans-IO WebSocket implementation. Same
  architectural pattern for the session state machine.

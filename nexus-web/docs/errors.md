# Errors and Poisoning

nexus-web error types follow a single convention: one top-level
error per subsystem, with an `Io` variant for transport failures and
a protocol variant for malformed wire data. Every error implements
`std::error::Error` and `Display`.

## WebSocket: `ws::Error`

```rust
use nexus_web::ws::Error;
```

| Variant | Meaning |
|---------|---------|
| `Io(io::Error)` | Transport read/write failure |
| `Protocol(ProtocolError)` | Inbound frame violates RFC 6455 |
| `Encode(EncodeError)` | Outbound buffer too small / oversized control frame |
| `Handshake(HandshakeError)` | HTTP Upgrade failure |
| `Tls(TlsError)` | TLS negotiation or record failure (feature `tls`) |
| `TlsNotEnabled` | `wss://` URL passed without the `tls` feature |

### `ProtocolError`

Fires on bad inbound frames:

- `ProtocolError::MaskViolation` — server-role got an unmasked frame
  (or client-role got a masked one)
- `ProtocolError::BadOpcode` — reserved or unknown opcode
- `ProtocolError::ControlFrameTooLarge` — control frame > 125 bytes
- `ProtocolError::FragmentedControlFrame`
- `ProtocolError::InvalidUtf8` — text frame failed UTF-8 validation
- `ProtocolError::InvalidCloseCode`
- `ProtocolError::MessageTooLarge` — assembled message > `max_message_size`
- `ProtocolError::FrameTooLarge` — single frame > `max_frame_size`
- `ProtocolError::ReservedBits` — RSV1/2/3 set without extension
- `ProtocolError::UnexpectedContinuation` — CONT frame with no opener

These are all terminal — once the reader yields a `ProtocolError`,
the peer has violated the spec and you must close the connection.

### `EncodeError`

Fires on bad outbound frames:

- `EncodeError::Overflow` — payload too large for the destination
  slice (fixed-size encoders) or for the `WriteBuf`
- `EncodeError::ControlFrameTooLarge` — control payload > 125 bytes

## REST: `rest::RestError`

| Variant | Meaning |
|---------|---------|
| `Io(io::Error)` | Transport failure |
| `Http(HttpError)` | Response parse error |
| `BodyTooLarge { size, max }` | Response body exceeded `max_body_size` |
| `RequestTooLarge { capacity }` | Serialized request exceeded `WriteBuf` capacity |
| `CrlfInjection` | Header or query contained CR/LF |
| `ConnectionPoisoned` | Previous send failed; connection unusable |
| `ReadTimeout` | No response within `read_timeout` |
| `ConnectionStale` | Dead socket detected (pooled client heal signal) |
| `ConnectionClosed(&'static str)` | Remote closed before response complete |
| `InvalidUrl(String)` | URL parse failure |
| `TlsNotEnabled` | `https://` without the `tls` feature |
| `Tls(TlsError)` | TLS error (feature `tls`) |

## TLS: `tls::TlsError`

| Variant | Meaning |
|---------|---------|
| `Io(io::Error)` | Underlying transport |
| `InvalidDnsName(String)` | Hostname is not a valid DNS name per RFC 5890 |
| `NoRootCerts` | System trust store couldn't be loaded |
| `Rustls(rustls::Error)` | Handshake, certificate, or record error |

## HTTP: `http::HttpError`

Low-level parse errors, bubbled up through `RestError::Http`:

- `HttpError::InvalidHeader`
- `HttpError::InvalidStatusLine`
- `HttpError::HeaderBufferFull`
- `HttpError::ChunkedEncodingError`
- `HttpError::InvalidChunkSize`
- `HttpError::Incomplete` (not terminal — more bytes needed)

## Connection poisoning

Both `ws::Client` and `rest::Client` maintain a `poisoned` bool. It
is set when an IO error fires **after a partial write has begun** —
i.e. the framed bytes may be half on the wire, leaving the peer in
an indeterminate state.

**Once poisoned, a connection is unusable.** You must construct a
new client. `is_poisoned()` returns the flag; the next send attempt
returns `RestError::ConnectionPoisoned` / `ws::Error::Io` with a
clear inner error.

### Why poisoning (vs. auto-reconnect)?

The library doesn't know what you want. For market data, you
typically *do* want to reconnect — but with backoff, with a fresh
subscription, with your own circuit breaker. For order entry, a
blind reconnect after a partial write is dangerous: the exchange may
have received an incomplete `"NEW_ORDER"` and is about to return a
parse error (or worse, succeed on a truncated field).

The caller owns the failure policy. See
[patterns.md — REST client with retry](./patterns.md#rest-client-with-retry)
for a concrete reconnect strategy.

### When poison fires

- `send_*()` IO error on a WebSocket `Client`
- `send()` IO error on a REST `Client`
- `send()` IO error during chunked-decoded response read

Reads on a healthy connection that simply return `Ok(None)` (EOF,
WouldBlock) do **not** poison — the peer closed cleanly, or the
socket is non-blocking and has no more bytes. You choose whether to
reconnect.

## Recovery

On poison:

1. Drop the `Client`.
2. Reconnect with the same `ClientBuilder`.
3. Re-send your auth / subscription / session-level state.

On `ConnectionStale` (REST pool):

1. The pool slot handles this automatically — `needs_reconnect()`
   is set and a reconnect task is spawned. See the async
   `ClientPool` docs.

On `ReadTimeout`:

- Your read_timeout elapsed with no bytes. The connection may still
  be alive (slow peer) or dead (TCP black hole). Decide based on
  your SLA. Most trading systems treat a read timeout as "reconnect
  now" because a slow peer is indistinguishable from a dead one at
  the protocol layer.

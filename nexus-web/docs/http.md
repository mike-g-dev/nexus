# HTTP / REST

nexus-web provides HTTP/1.1 primitives in two modules:

- `http` — low-level parsers and writers (`ResponseReader`,
  `ChunkedDecoder`, `write_request`, `write_response`)
- `rest` — a high-level REST client built on top (`Client`,
  `RequestWriter`, `RequestBuilder`, `RestResponse`)

Most users want `rest`. Drop to `http` if you're writing a server or
using a custom transport.

## REST client quickstart

```rust
use nexus_web::rest::{Client, RequestWriter};
use nexus_web::http::ResponseReader;
use nexus_net::tls::TlsConfig;
use std::time::Duration;

let tls = TlsConfig::new()?;
let mut conn = Client::builder()
    .tls(&tls)
    .disable_nagle()
    .connect_timeout(Duration::from_secs(3))
    .read_timeout(Duration::from_secs(5))
    .connect("https://api.binance.com")?;

// RequestWriter owns the WriteBuf used to serialize requests.
let mut writer = RequestWriter::new("api.binance.com")?;
writer.default_header("X-MBX-APIKEY", &api_key)?;
writer.set_base_path("/api/v3")?;

// Response parser — owns a ReadBuf and decoded response state.
let mut reader = ResponseReader::new(32 * 1024);

// Build and send a request.
let req = writer
    .get("/ticker/price")
    .query("symbol", "BTCUSDT")
    .finish()?;

let resp = conn.send(req, &mut reader)?;
println!("status: {}", resp.status());
for (name, value) in resp.headers() {
    println!("  {name}: {value:?}");
}
let body: &[u8] = resp.body();
```

The connection is keep-alive by default. Subsequent `conn.send(...)`
calls reuse the TCP (and TLS) session.

## RequestWriter — typestate builder

`RequestWriter` owns a shared `WriteBuf` and a set of default headers
that are applied to every request. The `get`/`post`/`put`/`delete`
methods return a `RequestBuilder<'_, Query>`; transitioning through
phases via `.query()`, `.header()`, `.body()`, `.finish()`:

```text
Query ──query()──▶ Query ──header()──▶ Headers ──body()──▶ Ready ──finish()──▶ Request<'_>
        │                  │
        └─body()──▶ Ready ──┘
```

- **`Query`** phase: you can add query params (`.query(k, v)`) or
  start adding headers / body.
- **`Headers`** phase: you can add more headers or set a body.
- **`Ready`** phase: only `.finish()` is left.

Each phase is a compile-time marker type, so `finish()` is only
callable when the request is fully specified. Adding a query after
setting a body is a compile error.

```rust
let req = writer
    .post("/order")
    .query("timestamp", &ts)
    .header("Content-Type", "application/json")
    .body(body_bytes)
    .finish()?;
```

### Body variants

- `.body(&[u8])` — copy the body into the WriteBuf (Content-Length set
  automatically)
- `.body_writer(|w| -> Result<(), E>)` — write directly into the
  WriteBuf via a closure, no intermediate copy. Content-Length is
  back-filled from the actual bytes written.
- `.body_fixed(len, |dst| -> Result<(), E>)` — pre-size the body for
  fastest encoding when you know the length up front.

### Raw builders

`writer.get_raw(path)` starts from the `Headers` phase — use when you
don't want automatic query encoding or need wire-level control.

## Response

`RestResponse<'_>` borrows from the `ResponseReader` buffer:

- `.status()` — `u16` status code
- `.headers()` — iterator of `(&str, &[u8])`
- `.header(name)` — lookup by case-insensitive name
- `.body()` — `&[u8]` pointing into the reader's buffer

The lifetime ties the response to `reader`; hold onto it only until
the next `conn.send(..., &mut reader)`.

### Chunked transfer encoding

`ResponseReader` handles chunked transfer encoding transparently via
`ChunkedDecoder`. By the time `send()` returns, the full body has
been collected and `resp.body()` gives you the reassembled bytes.
There is no streaming-body API at this layer — use the sans-IO
`ChunkedDecoder` directly if you need to stream.

```rust
use nexus_web::http::ChunkedDecoder;

let mut dec = ChunkedDecoder::new();
let consumed = dec.decode(wire_bytes, &mut output_buf)?;
if dec.is_complete() { /* trailers, done */ }
```

## Keep-alive, redirects, poisoning

- **Keep-alive** is the default. `Client` reuses the TCP session for
  every `send()`. If the remote sends `Connection: close`, the next
  `send()` returns `RestError::ConnectionClosed`.
- **Redirects** are **not** followed automatically. You see the 3xx
  status in the response and decide what to do. (Rationale: trading
  APIs don't redirect; silently following redirects can mask bugs.)
- **Poisoning:** if `send()` returns an IO error mid-request, the
  `Client` is marked poisoned and all subsequent sends return
  `RestError::ConnectionPoisoned`. Construct a new `Client`.

## Low-level `http` module

- `write_request(&mut WriteBuf, method, path, headers, body)` — emit
  a raw HTTP/1.1 request into a WriteBuf.
- `write_response(&mut WriteBuf, status, headers, body)` — emit a
  raw response (used by servers / the WebSocket accept path).
- `ResponseReader::new(capacity)` — parser for inbound responses.
- `ChunkedDecoder::new()` — streaming decoder for chunked bodies.
- `RequestReader` — parser for inbound requests. Used internally by
  the WebSocket server-side handshake; exposed in case you need it.

See [patterns.md](./patterns.md) for a full REST-with-retry cookbook
and [errors.md](./errors.md) for the error taxonomy.

# Patterns and Recipes

This is a cookbook for common production patterns. Every example
compiles against the public API as of nexus-web v0.x.

## Exchange WebSocket client

Connect to an exchange market-data feed, subscribe, handle messages,
reply to pings, and reconnect with exponential backoff on failure.

```rust
use nexus_web::ws::{Client, Message, CloseCode, Error};
use nexus_net::tls::TlsConfig;
use std::time::Duration;
use std::thread;

fn run_feed(tls: &TlsConfig) -> Result<(), Error> {
    let mut backoff = Duration::from_millis(100);
    let max_backoff = Duration::from_secs(30);

    loop {
        match connect_and_stream(tls) {
            Ok(()) => {
                tracing::info!("clean close, reconnecting");
                backoff = Duration::from_millis(100);
            }
            Err(e) => {
                tracing::warn!(?e, "feed disconnected");
                thread::sleep(backoff);
                backoff = (backoff * 2).min(max_backoff);
            }
        }
    }
}

fn connect_and_stream(tls: &TlsConfig) -> Result<(), Error> {
    let mut ws = Client::builder()
        .tls(tls)
        .disable_nagle()
        .buffer_capacity(1 << 20)
        .max_message_size(16 << 20)
        .connect_timeout(Duration::from_secs(3))
        .read_timeout(Duration::from_secs(30))
        .connect("wss://stream.binance.com:9443/ws")?;

    // Subscribe.
    ws.send_text(r#"{
        "method":"SUBSCRIBE",
        "params":["btcusdt@trade","btcusdt@depth20@100ms"],
        "id":1
    }"#)?;

    // Event loop. Message borrows from ws.reader internally.
    loop {
        match ws.recv()? {
            Some(Message::Text(json)) => on_json(json),
            Some(Message::Binary(bytes)) => on_binary(bytes),
            Some(Message::Ping(data)) => ws.send_pong(data)?,
            Some(Message::Pong(_)) => {}
            Some(Message::Close(_)) => {
                let _ = ws.close(CloseCode::Normal, "");
                return Ok(());
            }
            None => return Ok(()),  // EOF
        }
    }
}

fn on_json(_: &str) { /* parse + dispatch */ }
fn on_binary(_: &[u8]) { /* ... */ }
```

Key points:

- **Poisoning is caught by the outer loop.** Any `send_*` IO error
  propagates out of `connect_and_stream`, the outer loop sleeps and
  reconnects.
- **`Message` borrows from `ws`.** Don't hold a `Message::Text(s)`
  across another `recv()`. Copy (`s.to_owned()`) or parse immediately.
- **Disable Nagle.** Trading messages are small and latency-sensitive.
- **Subscribe after connect.** The subscription message is part of
  session state that you re-send on every reconnect.

## REST client with retry

Paginated request pattern with idempotent retry on transport errors:

```rust
use nexus_web::rest::{Client, RequestWriter, RestError};
use nexus_web::http::ResponseReader;
use nexus_net::tls::TlsConfig;
use std::time::Duration;
use std::thread;

struct Api {
    conn: Client<nexus_net::MaybeTls<std::net::TcpStream>>,
    writer: RequestWriter,
    reader: ResponseReader,
    tls: TlsConfig,
}

impl Api {
    fn new(tls: TlsConfig, api_key: &str) -> Result<Self, RestError> {
        let conn = Client::builder()
            .tls(&tls)
            .disable_nagle()
            .connect_timeout(Duration::from_secs(3))
            .read_timeout(Duration::from_secs(5))
            .connect("https://api.exchange.com")?;

        let mut writer = RequestWriter::new("api.exchange.com")?;
        writer.default_header("X-API-KEY", api_key)?;
        writer.set_base_path("/v1")?;

        Ok(Self {
            conn,
            writer,
            reader: ResponseReader::new(64 * 1024),
            tls,
        })
    }

    fn reconnect(&mut self) -> Result<(), RestError> {
        self.conn = Client::builder()
            .tls(&self.tls)
            .disable_nagle()
            .connect_timeout(Duration::from_secs(3))
            .read_timeout(Duration::from_secs(5))
            .connect("https://api.exchange.com")?;
        Ok(())
    }

    /// GET a resource, retrying on transport errors.
    fn get(&mut self, path: &str) -> Result<Vec<u8>, RestError> {
        let mut attempts = 0;
        loop {
            let req = self.writer.get(path).finish()?;
            match self.conn.send(req, &mut self.reader) {
                Ok(resp) if resp.status() < 500 => {
                    return Ok(resp.body().to_vec());
                }
                Ok(resp) => {
                    // 5xx: server-side; retry after backoff.
                    tracing::warn!(status = resp.status(), "5xx from {}", path);
                }
                Err(RestError::ConnectionPoisoned)
                | Err(RestError::ConnectionClosed(_))
                | Err(RestError::ConnectionStale)
                | Err(RestError::Io(_))
                | Err(RestError::ReadTimeout) => {
                    self.reconnect()?;
                }
                Err(e) => return Err(e),
            }

            attempts += 1;
            if attempts >= 5 { return Err(RestError::ReadTimeout); }
            thread::sleep(Duration::from_millis(100 * (1 << attempts)));
        }
    }
}
```

**Note:** retry is only safe for **idempotent** requests (GET, PUT,
DELETE). Never blind-retry a POST that creates an order — you may
duplicate it. Use exchange-provided idempotency keys (`clientOrderId`)
and check whether the order already exists before retry.

## Server accepting connections

Use the `accept()` path to run WebSocket server-side:

```rust
use nexus_web::ws::{Client, Message};
use std::net::TcpListener;
use std::thread;

fn main() -> std::io::Result<()> {
    let listener = TcpListener::bind("0.0.0.0:9001")?;
    for tcp in listener.incoming() {
        let tcp = tcp?;
        tcp.set_nodelay(true)?;
        thread::spawn(move || {
            let mut ws = match Client::builder().accept(tcp) {
                Ok(ws) => ws,
                Err(e) => { eprintln!("handshake failed: {e}"); return; }
            };
            while let Ok(Some(msg)) = ws.recv() {
                match msg {
                    Message::Text(s) => { let _ = ws.send_text(s); }
                    Message::Binary(b) => { let _ = ws.send_binary(b); }
                    Message::Ping(b) => { let _ = ws.send_pong(b); }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        });
    }
    Ok(())
}
```

## Integrating with mio

Sans-IO shines when you already own the event loop. `FrameReader`
and `FrameWriter` don't own a socket — they just consume and
produce bytes:

```rust
use mio::{Events, Interest, Poll, Token};
use mio::net::TcpStream;
use nexus_web::ws::{FrameReader, FrameWriter, Role, Message};
use nexus_net::buf::WriteBuf;
use std::io::{Read, Write};

const WS: Token = Token(0);

fn run(addr: &str) -> std::io::Result<()> {
    let mut poll = Poll::new()?;
    let mut events = Events::with_capacity(128);

    let mut tcp = TcpStream::connect(addr.parse().unwrap())?;
    poll.registry()
        .register(&mut tcp, WS, Interest::READABLE | Interest::WRITABLE)?;

    // Skipping the handshake for brevity — in real code,
    // perform the Upgrade handshake first.
    let mut reader = FrameReader::builder().role(Role::Client).build();
    let mut writer = FrameWriter::new(Role::Client);
    let mut out = WriteBuf::new(64 * 1024, 14);

    writer.encode_text_into(br#"{"subscribe":"trades"}"#, &mut out);

    loop {
        poll.poll(&mut events, None)?;
        for event in &events {
            if event.token() != WS { continue; }

            if event.is_writable() && !out.data().is_empty() {
                match tcp.write(out.data()) {
                    Ok(0) => return Ok(()),
                    Ok(n) => out.advance(n),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(e) => return Err(e),
                }
            }

            if event.is_readable() {
                loop {
                    let dst = reader.spare();
                    if dst.is_empty() {
                        reader.compact();
                        continue;
                    }
                    match tcp.read(dst) {
                        Ok(0) => return Ok(()),
                        Ok(n) => reader.filled(n),
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                        Err(e) => return Err(e),
                    }
                }
                while reader.poll().unwrap_or(false) {
                    match reader.next() {
                        Ok(Some(Message::Text(s))) => handle(s),
                        Ok(Some(Message::Ping(data))) => {
                            writer.encode_pong_into(data, &mut out)
                                .expect("pong fits");
                        }
                        Ok(Some(_)) | Ok(None) => {}
                        Err(e) => { eprintln!("proto: {e}"); return Ok(()); }
                    }
                }
            }
        }
    }
}

fn handle(_: &str) {}
```

The codec doesn't know or care that you're on mio. You could swap
mio for `io_uring` or DPDK and the protocol code wouldn't change.

## See also

- Async equivalents of these patterns live in
  [nexus-async-web/docs/patterns.md](../../nexus-async-web/docs/patterns.md).

//! Async WebSocket — nexus-async-rt backend.
//!
//! Builder + handshake logic. `WsReader`/`WsWriter` (the primary API)
//! live in the shared `ws::parts` module.

use std::io;
use std::net::ToSocketAddrs;

use nexus_async_rt::TcpStream;
use nexus_net::WireStream;
use nexus_net::buf::WriteBuf;
#[cfg(feature = "tls")]
use nexus_net::tls::TlsConfig;
use nexus_web::http::HTTP_HANDSHAKE_BUFFER;
use nexus_web::ws::{
    Error as WsError, FrameReader, FrameReaderBuilder, FrameWriter, HandshakeError, Role,
    parse_ws_url,
};

use crate::maybe_tls::MaybeTls;
use crate::ws::parts::{WsReader, WsWriter, fill_async, write_all_async};

// =============================================================================
// Handshake — standalone async functions
// =============================================================================

async fn connect_handshake<S: WireStream + Unpin>(
    mut stream: S,
    url: &str,
    reader_builder: FrameReaderBuilder,
    write_cap: usize,
    max_read_size: usize,
) -> Result<(WsReader, WsWriter, S), WsError> {
    let parsed = parse_ws_url(url)?;
    let host_header = parsed.host_header();

    let key = nexus_web::ws::handshake::generate_key();
    let key_str =
        std::str::from_utf8(&key).expect("base64-encoded key is always valid ASCII/UTF-8");

    let headers: [(&str, &str); 5] = [
        ("Host", &host_header),
        ("Upgrade", "websocket"),
        ("Connection", "Upgrade"),
        ("Sec-WebSocket-Key", key_str),
        ("Sec-WebSocket-Version", "13"),
    ];
    let req_size = nexus_web::http::request_size("GET", parsed.path, &headers);
    let mut req_buf = vec![0u8; req_size];
    let n = nexus_web::http::write_request("GET", parsed.path, &headers, &mut req_buf)
        .map_err(|_| HandshakeError::MalformedHttp)?;

    write_all_async(&mut stream, &req_buf[..n]).await?;

    let mut resp_reader = nexus_web::http::ResponseReader::new(HTTP_HANDSHAKE_BUFFER);
    loop {
        if resp_reader.spare().is_empty() {
            return Err(HandshakeError::MalformedHttp.into());
        }
        let n = fill_async(&mut stream, &mut resp_reader, HTTP_HANDSHAKE_BUFFER).await?;
        if n == 0 {
            return Err(HandshakeError::MalformedHttp.into());
        }
        match resp_reader.next() {
            Ok(Some(resp)) => {
                if resp.status != 101 {
                    return Err(HandshakeError::UnexpectedStatus(resp.status).into());
                }
                let upgrade = resp
                    .header("Upgrade")
                    .ok_or(HandshakeError::MissingUpgrade)?;
                if !upgrade.eq_ignore_ascii_case("websocket") {
                    return Err(HandshakeError::MissingUpgrade.into());
                }
                let conn = resp
                    .header("Connection")
                    .ok_or(HandshakeError::MissingConnection)?;
                if !conn
                    .as_bytes()
                    .windows(7)
                    .any(|w| w.eq_ignore_ascii_case(b"upgrade"))
                {
                    return Err(HandshakeError::MissingConnection.into());
                }
                let accept = resp
                    .header("Sec-WebSocket-Accept")
                    .ok_or(HandshakeError::InvalidAcceptKey)?;
                if !nexus_web::ws::handshake::validate_accept(key_str, accept) {
                    return Err(HandshakeError::InvalidAcceptKey.into());
                }

                let mut reader = reader_builder.role(Role::Client).build();
                let remainder = resp_reader.remainder();
                if !remainder.is_empty() {
                    reader
                        .read(remainder)
                        .map_err(|_| HandshakeError::MalformedHttp)?;
                }

                return Ok((
                    WsReader {
                        reader,
                        max_read_size,
                    },
                    WsWriter {
                        writer: FrameWriter::new(Role::Client),
                        write_buf: WriteBuf::new(write_cap, 14),
                    },
                    stream,
                ));
            }
            Ok(None) => {}
            Err(_) => return Err(HandshakeError::MalformedHttp.into()),
        }
    }
}

async fn accept_handshake<S: WireStream + Unpin>(
    mut stream: S,
    reader_builder: FrameReaderBuilder,
    write_cap: usize,
    max_read_size: usize,
) -> Result<(WsReader, WsWriter, S), WsError> {
    let mut req_reader = nexus_web::http::RequestReader::new(HTTP_HANDSHAKE_BUFFER);

    let ws_key;
    loop {
        if req_reader.spare().is_empty() {
            return Err(HandshakeError::MalformedHttp.into());
        }
        let n = fill_async(&mut stream, &mut req_reader, HTTP_HANDSHAKE_BUFFER).await?;
        if n == 0 {
            return Err(HandshakeError::MalformedHttp.into());
        }
        match req_reader.next() {
            Ok(Some(req)) => {
                if req.method != "GET" {
                    return Err(HandshakeError::MalformedHttp.into());
                }
                let upgrade = req
                    .header("Upgrade")
                    .ok_or(HandshakeError::MissingUpgrade)?;
                if !upgrade.eq_ignore_ascii_case("websocket") {
                    return Err(HandshakeError::MissingUpgrade.into());
                }
                let conn = req
                    .header("Connection")
                    .ok_or(HandshakeError::MissingConnection)?;
                if !conn
                    .as_bytes()
                    .windows(7)
                    .any(|w| w.eq_ignore_ascii_case(b"upgrade"))
                {
                    return Err(HandshakeError::MissingConnection.into());
                }
                let version = req
                    .header("Sec-WebSocket-Version")
                    .ok_or(HandshakeError::UnsupportedVersion)?;
                if version != "13" {
                    return Err(HandshakeError::UnsupportedVersion.into());
                }
                let key = req
                    .header("Sec-WebSocket-Key")
                    .ok_or(HandshakeError::MissingKey)?;
                ws_key = key.to_owned();
                break;
            }
            Ok(None) => {}
            Err(_) => return Err(HandshakeError::MalformedHttp.into()),
        }
    }

    let accept = nexus_web::ws::handshake::compute_accept_key(&ws_key);
    let accept_str = std::str::from_utf8(&accept).expect("base64 output is valid ASCII");

    let resp_headers = [
        ("Upgrade", "websocket"),
        ("Connection", "Upgrade"),
        ("Sec-WebSocket-Accept", accept_str),
    ];
    let resp_size = nexus_web::http::response_size("Switching Protocols", &resp_headers);
    let mut resp_buf = vec![0u8; resp_size];
    let n =
        nexus_web::http::write_response(101, "Switching Protocols", &resp_headers, &mut resp_buf)
            .map_err(|_| HandshakeError::MalformedHttp)?;
    write_all_async(&mut stream, &resp_buf[..n]).await?;

    let mut reader = reader_builder.role(Role::Server).build();
    let remainder = req_reader.remainder();
    if !remainder.is_empty() {
        reader
            .read(remainder)
            .map_err(|_| HandshakeError::MalformedHttp)?;
    }

    Ok((
        WsReader {
            reader,
            max_read_size,
        },
        WsWriter {
            writer: FrameWriter::new(Role::Server),
            write_buf: WriteBuf::new(write_cap, 14),
        },
        stream,
    ))
}

// =============================================================================
// Builder
// =============================================================================

/// Builder for WebSocket connections (nexus-async-rt backend).
///
/// Returns `(WsReader, WsWriter, S)` — the decomposed sans-IO types.
///
/// # Example
///
/// ```ignore
/// let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
///     .disable_nagle()
///     .connect("ws://localhost:8080/ws")
///     .await?;
/// ```
pub struct WsStreamBuilder {
    reader_builder: FrameReaderBuilder,
    write_buf_capacity: usize,
    buffer_capacity: usize,
    max_read_size: Option<usize>,
    #[cfg(feature = "tls")]
    tls_config: Option<TlsConfig>,
    #[cfg(feature = "tls")]
    tls_capacities: Option<nexus_net::tls::TlsBufferCapacities>,
    nodelay: bool,
    connect_timeout: Option<std::time::Duration>,
    #[cfg(feature = "socket-opts")]
    tcp_keepalive: Option<std::time::Duration>,
    #[cfg(feature = "socket-opts")]
    recv_buf_size: Option<usize>,
    #[cfg(feature = "socket-opts")]
    send_buf_size: Option<usize>,
}

const DEFAULT_BUFFER_CAPACITY: usize = 1024 * 1024;

impl WsStreamBuilder {
    /// Create a new builder with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            reader_builder: FrameReader::builder(),
            write_buf_capacity: 65_536,
            buffer_capacity: DEFAULT_BUFFER_CAPACITY,
            max_read_size: None,
            #[cfg(feature = "tls")]
            tls_config: None,
            #[cfg(feature = "tls")]
            tls_capacities: None,
            nodelay: false,
            connect_timeout: None,
            #[cfg(feature = "socket-opts")]
            tcp_keepalive: None,
            #[cfg(feature = "socket-opts")]
            recv_buf_size: None,
            #[cfg(feature = "socket-opts")]
            send_buf_size: None,
        }
    }

    fn resolved_max_read_size(&self) -> usize {
        self.max_read_size.map_or_else(
            || (self.buffer_capacity / 8).max(1),
            |n| n.min(self.buffer_capacity).max(1),
        )
    }

    /// ReadBuf capacity. Default: 1MB.
    #[must_use]
    pub fn buffer_capacity(mut self, n: usize) -> Self {
        self.buffer_capacity = n;
        self.reader_builder = self.reader_builder.buffer_capacity(n);
        self
    }

    /// Maximum bytes to read from the transport per recv call.
    ///
    /// Caps the slice passed to the underlying read, bounding the worst-case
    /// memcpy per message. Lower values reduce tail latency at the cost of
    /// more frequent reads.
    ///
    /// Default: 1/8 of buffer capacity. Clamped to `[1, buffer_capacity]`.
    #[must_use]
    pub fn max_read_size(mut self, n: usize) -> Self {
        self.max_read_size = Some(n);
        self
    }

    /// Fraction of buffer capacity consumed before proactive compaction.
    ///
    /// See [`FrameReaderBuilder::compact_at`](nexus_web::ws::FrameReaderBuilder::compact_at)
    /// for details. Default: 0.5.
    #[must_use]
    pub fn compact_at(mut self, fraction: f64) -> Self {
        self.reader_builder = self.reader_builder.compact_at(fraction);
        self
    }

    /// Maximum single frame payload. Default: 16MB.
    #[must_use]
    pub fn max_frame_size(mut self, n: u64) -> Self {
        self.reader_builder = self.reader_builder.max_frame_size(n);
        self
    }

    /// Maximum assembled message size. Default: 16MB.
    #[must_use]
    pub fn max_message_size(mut self, n: usize) -> Self {
        self.reader_builder = self.reader_builder.max_message_size(n);
        self
    }

    /// Write buffer capacity. Default: 64KB.
    #[must_use]
    pub fn write_buffer_capacity(mut self, n: usize) -> Self {
        self.write_buf_capacity = n;
        self
    }

    /// Custom TLS configuration.
    #[cfg(feature = "tls")]
    #[must_use]
    pub fn tls(mut self, config: &TlsConfig) -> Self {
        self.tls_config = Some(config.clone());
        self
    }

    /// Override the TLS adapter's per-connection buffer capacities.
    /// Only applies when the connection is `wss://`.
    ///
    /// Defaults: 8 KiB read chunk + 64 KiB pending_write. Trading
    /// workloads with small messages can drop the pending_write
    /// capacity to 8–16 KiB to reduce per-connection footprint.
    /// See [`TlsBufferCapacities`](nexus_net::tls::TlsBufferCapacities).
    #[cfg(feature = "tls")]
    #[must_use]
    pub fn tls_buffer_capacities(
        mut self,
        capacities: nexus_net::tls::TlsBufferCapacities,
    ) -> Self {
        self.tls_capacities = Some(capacities);
        self
    }

    /// Set TCP_NODELAY.
    #[must_use]
    pub fn disable_nagle(mut self) -> Self {
        self.nodelay = true;
        self
    }

    /// TCP connect timeout.
    #[must_use]
    pub fn connect_timeout(mut self, d: std::time::Duration) -> Self {
        self.connect_timeout = Some(d);
        self
    }

    /// Set TCP keepalive idle time.
    ///
    /// Enables OS-level dead connection detection. The kernel sends
    /// probes after `idle` of inactivity.
    #[cfg(feature = "socket-opts")]
    #[must_use]
    pub fn tcp_keepalive(mut self, idle: std::time::Duration) -> Self {
        self.tcp_keepalive = Some(idle);
        self
    }

    /// Set `SO_RCVBUF` (socket receive buffer size).
    #[cfg(feature = "socket-opts")]
    #[must_use]
    pub fn recv_buffer_size(mut self, n: usize) -> Self {
        self.recv_buf_size = Some(n);
        self
    }

    /// Set `SO_SNDBUF` (socket send buffer size).
    #[cfg(feature = "socket-opts")]
    #[must_use]
    pub fn send_buffer_size(mut self, n: usize) -> Self {
        self.send_buf_size = Some(n);
        self
    }

    /// Connect to a WebSocket server. Creates TCP socket, handles TLS.
    ///
    /// DNS resolution uses blocking `ToSocketAddrs` (cold path).
    /// TCP connect uses `nexus_async_rt::TcpStream::connect` (mio, non-blocking).
    #[allow(clippy::future_not_send)]
    pub async fn connect(self, url: &str) -> Result<(WsReader, WsWriter, MaybeTls), WsError> {
        let parsed = parse_ws_url(url)?;
        let addr_str = format!("{}:{}", parsed.host, parsed.port);
        let addr = addr_str
            .to_socket_addrs()
            .map_err(WsError::Io)?
            .next()
            .ok_or_else(|| {
                WsError::Io(io::Error::new(
                    io::ErrorKind::AddrNotAvailable,
                    format!("DNS resolution failed: {addr_str}"),
                ))
            })?;

        let connect_fn = async {
            let tcp = TcpStream::connect(addr)?;
            Ok::<TcpStream, WsError>(tcp)
        };

        #[allow(unused_mut)]
        let mut tcp = match self.connect_timeout {
            Some(dur) => nexus_async_rt::timeout(dur, connect_fn)
                .await
                .map_err(|_| {
                    WsError::Io(io::Error::new(io::ErrorKind::TimedOut, "connect timeout"))
                })??,
            None => connect_fn.await?,
        };

        if self.nodelay {
            tcp.set_nodelay(true)?;
        }
        #[cfg(feature = "socket-opts")]
        self.apply_socket_opts(&tcp)?;

        let stream = if parsed.tls {
            #[cfg(feature = "tls")]
            {
                let tls_config = match &self.tls_config {
                    Some(c) => c.clone(),
                    None => TlsConfig::new().map_err(WsError::Tls)?,
                };

                let codec = nexus_net::tls::TlsCodec::new(&tls_config, parsed.host)?;
                let capacities = self.tls_capacities.unwrap_or_default();
                let tls_inner = crate::maybe_tls::TlsInner::connect(tcp, codec, capacities).await?;
                MaybeTls::Tls(Box::new(tls_inner))
            }
            #[cfg(not(feature = "tls"))]
            {
                return Err(WsError::TlsNotEnabled);
            }
        } else {
            MaybeTls::Plain(tcp)
        };

        let max_read_size = self.resolved_max_read_size();
        connect_handshake(
            stream,
            url,
            self.reader_builder,
            self.write_buf_capacity,
            max_read_size,
        )
        .await
    }

    /// Connect with a pre-connected async stream.
    pub async fn connect_with<S: WireStream + Unpin>(
        self,
        stream: S,
        url: &str,
    ) -> Result<(WsReader, WsWriter, S), WsError> {
        let max_read_size = self.resolved_max_read_size();
        connect_handshake(
            stream,
            url,
            self.reader_builder,
            self.write_buf_capacity,
            max_read_size,
        )
        .await
    }

    /// Accept an incoming WebSocket connection (server-side).
    pub async fn accept<S: WireStream + Unpin>(
        self,
        stream: S,
    ) -> Result<(WsReader, WsWriter, S), WsError> {
        let max_read_size = self.resolved_max_read_size();
        accept_handshake(
            stream,
            self.reader_builder,
            self.write_buf_capacity,
            max_read_size,
        )
        .await
    }
}

#[cfg(feature = "socket-opts")]
impl WsStreamBuilder {
    fn apply_socket_opts(&self, tcp: &TcpStream) -> Result<(), WsError> {
        use std::os::fd::AsFd;
        let fd = tcp.as_fd();
        let sock = socket2::SockRef::from(&fd);
        if let Some(idle) = self.tcp_keepalive {
            let keepalive = socket2::TcpKeepalive::new().with_time(idle);
            sock.set_tcp_keepalive(&keepalive)?;
        }
        if let Some(size) = self.recv_buf_size {
            sock.set_recv_buffer_size(size)?;
        }
        if let Some(size) = self.send_buf_size {
            sock.set_send_buffer_size(size)?;
        }
        Ok(())
    }
}

impl Default for WsStreamBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NexusAsyncReadAdapter;
    use nexus_web::ws::{CloseCode, Message};
    use std::io::Cursor;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use nexus_async_rt::{AsyncRead, AsyncWrite};

    struct MockStream {
        read: Cursor<Vec<u8>>,
        written: Vec<u8>,
    }

    impl MockStream {
        fn from_bytes(data: Vec<u8>) -> Self {
            Self {
                read: Cursor::new(data),
                written: Vec::new(),
            }
        }
    }

    impl AsyncRead for MockStream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            let n = std::io::Read::read(&mut self.read, buf)?;
            Poll::Ready(Ok(n))
        }
    }

    impl AsyncWrite for MockStream {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            self.written.extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }
        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        use std::task::{RawWaker, RawWakerVTable, Waker};

        fn noop(_: *const ()) {}
        fn noop_clone(p: *const ()) -> RawWaker {
            RawWaker::new(p, &VTABLE)
        }
        const VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);

        // SAFETY: The vtable functions (clone/wake/wake_by_ref/drop) are all no-ops
        // that never dereference the data pointer, so the null data pointer is sound.
        // The vtable is 'static (const) and correctly returns a valid RawWaker on clone.
        let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);

        for _ in 0..1000 {
            if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
                return v;
            }
        }
        panic!("mock future did not resolve within 1000 polls");
    }

    fn make_frame(fin: bool, opcode: u8, payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::new();
        let byte0 = if fin { 0x80 } else { 0x00 } | opcode;
        frame.push(byte0);
        if payload.len() <= 125 {
            frame.push(payload.len() as u8);
        } else if payload.len() <= 65535 {
            frame.push(126);
            frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        } else {
            frame.push(127);
            frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
        }
        frame.extend_from_slice(payload);
        frame
    }

    fn parts_from_bytes(data: Vec<u8>) -> (WsReader, WsWriter, NexusAsyncReadAdapter<MockStream>) {
        let mock = NexusAsyncReadAdapter::new(MockStream::from_bytes(data));
        let reader = FrameReader::builder().role(Role::Client).build();
        let writer = FrameWriter::new(Role::Client);
        (
            WsReader {
                reader,
                max_read_size: usize::MAX,
            },
            WsWriter {
                writer,
                write_buf: WriteBuf::new(65_536, 14),
            },
            mock,
        )
    }

    #[test]
    fn recv_text() {
        let frame = make_frame(true, 0x1, b"Hello");
        let (mut reader, _writer, mut conn) = parts_from_bytes(frame);
        block_on(async {
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "Hello"),
                other => panic!("expected Text, got {other:?}"),
            }
        });
    }

    #[test]
    fn recv_binary() {
        let frame = make_frame(true, 0x2, &[0x42; 100]);
        let (mut reader, _writer, mut conn) = parts_from_bytes(frame);
        block_on(async {
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Binary(b) => assert_eq!(b.len(), 100),
                other => panic!("expected Binary, got {other:?}"),
            }
        });
    }

    #[test]
    fn recv_ping() {
        let frame = make_frame(true, 0x9, b"ping");
        let (mut reader, _writer, mut conn) = parts_from_bytes(frame);
        block_on(async {
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Ping(p) => assert_eq!(p, b"ping"),
                other => panic!("expected Ping, got {other:?}"),
            }
        });
    }

    #[test]
    fn recv_fragmented_text() {
        let mut data = make_frame(false, 0x1, b"Hel");
        data.extend_from_slice(&make_frame(true, 0x0, b"lo"));
        let (mut reader, _writer, mut conn) = parts_from_bytes(data);
        block_on(async {
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "Hello"),
                other => panic!("expected Text, got {other:?}"),
            }
        });
    }

    #[test]
    fn recv_fragment_with_control() {
        let mut data = make_frame(false, 0x1, b"Hel");
        data.extend_from_slice(&make_frame(true, 0x9, b"ping"));
        data.extend_from_slice(&make_frame(true, 0x0, b"lo"));
        let (mut reader, _writer, mut conn) = parts_from_bytes(data);
        block_on(async {
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Ping(p) => assert_eq!(p, b"ping"),
                other => panic!("expected Ping, got {other:?}"),
            }
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "Hello"),
                other => panic!("expected Text, got {other:?}"),
            }
        });
    }

    #[test]
    fn recv_close() {
        let mut payload = vec![];
        payload.extend_from_slice(&1000u16.to_be_bytes());
        payload.extend_from_slice(b"bye");
        let frame = make_frame(true, 0x8, &payload);
        let (mut reader, _writer, mut conn) = parts_from_bytes(frame);
        block_on(async {
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Close(cf) => {
                    assert_eq!(cf.code, CloseCode::Normal);
                    assert_eq!(cf.reason, "bye");
                }
                other => panic!("expected Close, got {other:?}"),
            }
        });
    }

    #[test]
    fn eof_returns_none() {
        let (mut reader, _writer, mut conn) = parts_from_bytes(Vec::new());
        block_on(async {
            assert!(reader.recv(&mut conn).await.unwrap().is_none());
        });
    }

    #[test]
    fn fifo_three_messages() {
        let mut data = make_frame(true, 0x1, b"first");
        data.extend_from_slice(&make_frame(true, 0x1, b"second"));
        data.extend_from_slice(&make_frame(true, 0x1, b"third"));
        let (mut reader, _writer, mut conn) = parts_from_bytes(data);

        block_on(async {
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "first"),
                other => panic!("expected first, got {other:?}"),
            }
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "second"),
                other => panic!("expected second, got {other:?}"),
            }
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "third"),
                other => panic!("expected third, got {other:?}"),
            }
        });
    }

    #[test]
    fn send_text_writes_frame() {
        let (_, mut writer, mut conn) = parts_from_bytes(Vec::new());
        block_on(async {
            writer.send_text(&mut conn, "hello").await.unwrap();
        });
        assert!(!conn.get_ref().written.is_empty());
    }

    #[test]
    fn send_binary_writes_frame() {
        let (_, mut writer, mut conn) = parts_from_bytes(Vec::new());
        block_on(async {
            writer.send_binary(&mut conn, &[1, 2, 3]).await.unwrap();
        });
        assert!(!conn.get_ref().written.is_empty());
    }

    #[test]
    fn ping_echo_split_borrow() {
        let mut data = make_frame(true, 0x9, b"ping-data");
        data.extend_from_slice(&make_frame(true, 0x1, b"hello"));
        let (mut reader, mut writer, mut conn) = parts_from_bytes(data);

        block_on(async {
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Ping(payload) => {
                    writer.send_pong(&mut conn, payload).await.unwrap();
                }
                other => panic!("expected Ping, got {other:?}"),
            }

            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "hello"),
                other => panic!("expected Text, got {other:?}"),
            }
        });
    }

    #[test]
    fn text_response_while_holding_message() {
        let data = make_frame(true, 0x1, b"request");
        let (mut reader, mut writer, mut conn) = parts_from_bytes(data);

        block_on(async {
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Text(req) => {
                    assert_eq!(req, "request");
                    let response = format!("echo: {req}");
                    writer.send_text(&mut conn, &response).await.unwrap();
                }
                other => panic!("expected Text, got {other:?}"),
            }
        });
    }

    struct BrokenWriteStream(Cursor<Vec<u8>>);

    impl AsyncRead for BrokenWriteStream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            let n = std::io::Read::read(&mut self.0, buf)?;
            Poll::Ready(Ok(n))
        }
    }

    impl AsyncWrite for BrokenWriteStream {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "connection lost",
            )))
        }
        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[test]
    fn send_on_broken_stream_returns_error() {
        let mock = NexusAsyncReadAdapter::new(BrokenWriteStream(Cursor::new(Vec::new())));
        let mut writer = WsWriter {
            writer: FrameWriter::new(Role::Client),
            write_buf: WriteBuf::new(65_536, 14),
        };
        let mut conn = mock;

        block_on(async {
            let result = writer.send_text(&mut conn, "hello").await;
            assert!(result.is_err(), "send on broken stream should return error");

            let result = writer.send_binary(&mut conn, &[1, 2, 3]).await;
            assert!(result.is_err(), "subsequent send should also fail");
        });
    }
}

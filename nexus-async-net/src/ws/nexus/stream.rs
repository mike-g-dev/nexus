//! Async WebSocket stream — nexus-async-rt backend.

use std::io;
use std::net::ToSocketAddrs;
use std::pin::Pin;

use nexus_async_rt::TcpStream;
use nexus_net::buf::WriteBuf;
use nexus_net::http::HTTP_HANDSHAKE_BUFFER;
#[cfg(feature = "tls")]
use nexus_net::tls::TlsConfig;
use nexus_net::ws::{
    CloseCode, Error as WsError, FrameReader, FrameReaderBuilder, FrameWriter, HandshakeError,
    Message, Role, parse_ws_url,
};
use nexus_net::{ParserSink, WireStream};

use crate::maybe_tls::MaybeTls;

// =============================================================================
// Async I/O helpers (poll_fn wrappers over WireStream)
// =============================================================================

/// Drive a single `poll_fill_into` call on the stream.
async fn fill_async<W: WireStream + Unpin, P: ParserSink>(
    s: &mut W,
    sink: &mut P,
    max: usize,
) -> io::Result<usize> {
    std::future::poll_fn(|cx| Pin::new(&mut *s).poll_fill_into(cx, sink, max)).await
}

async fn write_all_async<W: WireStream + Unpin>(s: &mut W, mut buf: &[u8]) -> io::Result<()> {
    while !buf.is_empty() {
        let n = std::future::poll_fn(|cx| Pin::new(&mut *s).poll_write(cx, buf)).await?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "write returned 0"));
        }
        buf = &buf[n..];
    }
    Ok(())
}

// =============================================================================
// WsStream
// =============================================================================

/// Async WebSocket stream (nexus-async-rt backend).
///
/// Wraps nexus-net's synchronous FrameReader/FrameWriter with async I/O.
/// Same zero-copy parsing, same Message type — just `.await` on socket ops.
///
/// # Usage
///
/// ```ignore
/// use nexus_async_net::ws::WsStreamBuilder;
/// use nexus_net::tls::TlsConfig;
///
/// // Plain WebSocket
/// let mut ws = WsStreamBuilder::new().connect("ws://localhost:8080/ws").await?;
///
/// // TLS WebSocket
/// let tls = TlsConfig::new()?;
/// let mut ws = WsStreamBuilder::new().tls(&tls).connect("wss://exchange.com/ws").await?;
///
/// ws.send_text("Hello!").await?;
/// while let Some(msg) = ws.recv().await? {
///     // msg is nexus_net::ws::Message<'_>
/// }
/// ```
pub struct WsStream<S> {
    stream: S,
    reader: FrameReader,
    writer: FrameWriter,
    write_buf: WriteBuf,
    max_read_size: usize,
}

// -- Generic impl for any WireStream-bearing transport ----------------------

impl<S: WireStream + Unpin> WsStream<S> {
    /// Connect with a pre-connected async stream.
    pub async fn connect_with(stream: S, url: &str) -> Result<Self, WsError> {
        WsStreamBuilder::new().connect_with(stream, url).await
    }

    /// Accept an incoming WebSocket connection (server-side).
    ///
    /// Reads the client's HTTP upgrade request, validates it,
    /// sends back 101 Switching Protocols, and returns a server-role
    /// WsStream ready for recv/send.
    pub async fn accept(stream: S) -> Result<Self, WsError> {
        WsStreamBuilder::new().accept(stream).await
    }

    /// Create from pre-existing parts. For testing or custom handshakes.
    ///
    /// `max_read_size` defaults to unlimited. Call [`set_max_read_size`](Self::set_max_read_size)
    /// after construction to cap per-recv read size for better tail latency.
    pub fn from_parts(stream: S, reader: FrameReader, writer: FrameWriter) -> Self {
        Self {
            stream,
            reader,
            writer,
            write_buf: WriteBuf::new(65_536, 14),
            max_read_size: usize::MAX,
        }
    }

    /// Receive the next message.
    pub async fn recv(&mut self) -> Result<Option<Message<'_>>, WsError> {
        loop {
            if self.reader.poll()? {
                return Ok(self.reader.next()?);
            }

            if self.reader.should_compact() {
                self.reader.compact();
            }
            if self.reader.spare().is_empty() {
                self.reader.compact();
                if self.reader.spare().is_empty() {
                    return Ok(None); // buffer genuinely full
                }
            }

            // poll_fill_into delivers bytes directly into reader.spare()
            // and commits via reader.filled(n) — zero-copy on adapters
            // that support it (nexus-async-rt TLS).
            let n = fill_async(&mut self.stream, &mut self.reader, self.max_read_size).await?;
            if n == 0 {
                return Ok(None); // EOF
            }
        }
    }

    /// Send a text message.
    pub async fn send_text(&mut self, text: &str) -> Result<(), WsError> {
        self.writer
            .encode_text_into(text.as_bytes(), &mut self.write_buf);
        write_all_async(&mut self.stream, self.write_buf.data()).await?;
        Ok(())
    }

    /// Send a binary message.
    pub async fn send_binary(&mut self, data: &[u8]) -> Result<(), WsError> {
        self.writer.encode_binary_into(data, &mut self.write_buf);
        write_all_async(&mut self.stream, self.write_buf.data()).await?;
        Ok(())
    }

    /// Send a ping.
    pub async fn send_ping(&mut self, data: &[u8]) -> Result<(), WsError> {
        self.writer
            .encode_ping_into(data, &mut self.write_buf)
            .map_err(WsError::Encode)?;
        write_all_async(&mut self.stream, self.write_buf.data()).await?;
        Ok(())
    }

    /// Send a pong.
    pub async fn send_pong(&mut self, data: &[u8]) -> Result<(), WsError> {
        self.writer
            .encode_pong_into(data, &mut self.write_buf)
            .map_err(WsError::Encode)?;
        write_all_async(&mut self.stream, self.write_buf.data()).await?;
        Ok(())
    }

    /// Initiate close handshake.
    pub async fn close(&mut self, code: CloseCode, reason: &str) -> Result<(), WsError> {
        if code == CloseCode::NoStatus {
            let mut dst = [0u8; 14];
            let n = self.writer.encode_empty_close(&mut dst);
            write_all_async(&mut self.stream, &dst[..n]).await?;
        } else {
            self.writer
                .encode_close_into(code.as_u16(), reason.as_bytes(), &mut self.write_buf)
                .map_err(WsError::Encode)?;
            write_all_async(&mut self.stream, self.write_buf.data()).await?;
        }
        Ok(())
    }

    /// Access the underlying stream.
    pub fn stream(&self) -> &S {
        &self.stream
    }

    /// Mutable access to the underlying stream.
    pub fn stream_mut(&mut self) -> &mut S {
        &mut self.stream
    }

    /// Access the FrameReader.
    pub fn reader(&self) -> &FrameReader {
        &self.reader
    }

    /// Access the FrameWriter.
    pub fn frame_writer(&self) -> &FrameWriter {
        &self.writer
    }

    /// Override max bytes read per recv call.
    pub fn set_max_read_size(&mut self, n: usize) {
        self.max_read_size = n.max(1);
    }

    // =========================================================================
    // Internal — async handshake (client connect)
    // =========================================================================

    async fn connect_impl(
        mut stream: S,
        url: &str,
        reader_builder: FrameReaderBuilder,
        write_cap: usize,
        max_read_size: usize,
    ) -> Result<Self, WsError> {
        let parsed = parse_ws_url(url)?;
        let host_header = parsed.host_header();

        let key = nexus_net::ws::handshake::generate_key();
        let key_str =
            std::str::from_utf8(&key).expect("base64-encoded key is always valid ASCII/UTF-8");

        let headers: [(&str, &str); 5] = [
            ("Host", &host_header),
            ("Upgrade", "websocket"),
            ("Connection", "Upgrade"),
            ("Sec-WebSocket-Key", key_str),
            ("Sec-WebSocket-Version", "13"),
        ];
        let req_size = nexus_net::http::request_size("GET", parsed.path, &headers);
        let mut req_buf = vec![0u8; req_size];
        let n = nexus_net::http::write_request("GET", parsed.path, &headers, &mut req_buf)
            .map_err(|_| HandshakeError::MalformedHttp)?;

        write_all_async(&mut stream, &req_buf[..n]).await?;

        // Feed bytes directly into resp_reader's spare region via
        // WireStream — one fewer copy than reading into a tmp slice
        // and pushing through resp_reader.read().
        let mut resp_reader = nexus_net::http::ResponseReader::new(HTTP_HANDSHAKE_BUFFER);
        loop {
            // Pre-check the WireStream::poll_fill_into precondition
            // (sink.spare() non-empty). If full without a parsed
            // response, the head exceeds capacity — treat as malformed.
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
                    if !nexus_net::ws::handshake::validate_accept(key_str, accept) {
                        return Err(HandshakeError::InvalidAcceptKey.into());
                    }

                    let mut reader = reader_builder.role(Role::Client).build();
                    let remainder = resp_reader.remainder();
                    if !remainder.is_empty() {
                        reader
                            .read(remainder)
                            .map_err(|_| HandshakeError::MalformedHttp)?;
                    }

                    return Ok(Self {
                        stream,
                        reader,
                        writer: FrameWriter::new(Role::Client),
                        write_buf: WriteBuf::new(write_cap, 14),
                        max_read_size,
                    });
                }
                Ok(None) => {} // need more bytes
                Err(_) => return Err(HandshakeError::MalformedHttp.into()),
            }
        }
    }

    // =========================================================================
    // Internal — async accept (server-side)
    // =========================================================================

    async fn accept_impl(
        mut stream: S,
        reader_builder: FrameReaderBuilder,
        write_cap: usize,
        max_read_size: usize,
    ) -> Result<Self, WsError> {
        let mut req_reader = nexus_net::http::RequestReader::new(HTTP_HANDSHAKE_BUFFER);

        let ws_key;
        loop {
            // Pre-check the WireStream::poll_fill_into precondition
            // (sink.spare() non-empty). If full without a parsed
            // request, the head exceeds capacity — treat as malformed.
            if req_reader.spare().is_empty() {
                return Err(HandshakeError::MalformedHttp.into());
            }
            // Direct-feed via WireStream — bytes land in
            // req_reader.spare() without a tmp slice intermediate.
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

        let accept = nexus_net::ws::handshake::compute_accept_key(&ws_key);
        let accept_str = std::str::from_utf8(&accept).expect("base64 output is valid ASCII");

        let resp_headers = [
            ("Upgrade", "websocket"),
            ("Connection", "Upgrade"),
            ("Sec-WebSocket-Accept", accept_str),
        ];
        let resp_size = nexus_net::http::response_size("Switching Protocols", &resp_headers);
        let mut resp_buf = vec![0u8; resp_size];
        let n = nexus_net::http::write_response(
            101,
            "Switching Protocols",
            &resp_headers,
            &mut resp_buf,
        )
        .map_err(|_| HandshakeError::MalformedHttp)?;
        write_all_async(&mut stream, &resp_buf[..n]).await?;

        let mut reader = reader_builder.role(Role::Server).build();
        let remainder = req_reader.remainder();
        if !remainder.is_empty() {
            reader
                .read(remainder)
                .map_err(|_| HandshakeError::MalformedHttp)?;
        }

        Ok(Self {
            stream,
            reader,
            writer: FrameWriter::new(Role::Server),
            write_buf: WriteBuf::new(write_cap, 14),
            max_read_size,
        })
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Builder for [`WsStream`].
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

    /// Resolve max_read_size: user override clamped to buffer, or default 1/8 of buffer.
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
    /// See [`FrameReaderBuilder::compact_at`](nexus_net::ws::FrameReaderBuilder::compact_at)
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
    pub async fn connect(self, url: &str) -> Result<WsStream<MaybeTls>, WsError> {
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

        #[allow(unused_mut)] // mut needed when tls feature is enabled
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
        WsStream::connect_impl(
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
    ) -> Result<WsStream<S>, WsError> {
        let max_read_size = self.resolved_max_read_size();
        WsStream::connect_impl(
            stream,
            url,
            self.reader_builder,
            self.write_buf_capacity,
            max_read_size,
        )
        .await
    }

    /// Accept an incoming WebSocket connection (server-side).
    pub async fn accept<S: WireStream + Unpin>(self, stream: S) -> Result<WsStream<S>, WsError> {
        let max_read_size = self.resolved_max_read_size();
        WsStream::accept_impl(
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
    use std::io::Cursor;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use crate::NexusAsyncReadAdapter;
    use nexus_async_rt::{AsyncRead, AsyncWrite};

    /// Mock async stream backed by a byte buffer.
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

    /// Minimal single-poll executor for futures that resolve immediately.
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

    fn ws_from_bytes(data: Vec<u8>) -> WsStream<NexusAsyncReadAdapter<MockStream>> {
        let mock = NexusAsyncReadAdapter::new(MockStream::from_bytes(data));
        let reader = FrameReader::builder().role(Role::Client).build();
        let writer = FrameWriter::new(Role::Client);
        WsStream::from_parts(mock, reader, writer)
    }

    #[test]
    fn recv_text() {
        let frame = make_frame(true, 0x1, b"Hello");
        let mut ws = ws_from_bytes(frame);
        block_on(async {
            match ws.recv().await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "Hello"),
                other => panic!("expected Text, got {other:?}"),
            }
        });
    }

    #[test]
    fn recv_binary() {
        let frame = make_frame(true, 0x2, &[0x42; 100]);
        let mut ws = ws_from_bytes(frame);
        block_on(async {
            match ws.recv().await.unwrap().unwrap() {
                Message::Binary(b) => assert_eq!(b.len(), 100),
                other => panic!("expected Binary, got {other:?}"),
            }
        });
    }

    #[test]
    fn recv_ping() {
        let frame = make_frame(true, 0x9, b"ping");
        let mut ws = ws_from_bytes(frame);
        block_on(async {
            match ws.recv().await.unwrap().unwrap() {
                Message::Ping(p) => assert_eq!(p, b"ping"),
                other => panic!("expected Ping, got {other:?}"),
            }
        });
    }

    #[test]
    fn recv_fragmented_text() {
        let mut data = make_frame(false, 0x1, b"Hel");
        data.extend_from_slice(&make_frame(true, 0x0, b"lo"));
        let mut ws = ws_from_bytes(data);
        block_on(async {
            match ws.recv().await.unwrap().unwrap() {
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
        let mut ws = ws_from_bytes(data);
        block_on(async {
            // Ping first
            match ws.recv().await.unwrap().unwrap() {
                Message::Ping(p) => assert_eq!(p, b"ping"),
                other => panic!("expected Ping, got {other:?}"),
            }
            // Then assembled text
            match ws.recv().await.unwrap().unwrap() {
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
        let mut ws = ws_from_bytes(frame);
        block_on(async {
            match ws.recv().await.unwrap().unwrap() {
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
        let mut ws = ws_from_bytes(Vec::new());
        block_on(async {
            assert!(ws.recv().await.unwrap().is_none());
        });
    }

    #[test]
    fn fifo_three_messages() {
        let mut data = make_frame(true, 0x1, b"first");
        data.extend_from_slice(&make_frame(true, 0x1, b"second"));
        data.extend_from_slice(&make_frame(true, 0x1, b"third"));
        let mut ws = ws_from_bytes(data);

        block_on(async {
            match ws.recv().await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "first"),
                other => panic!("expected first, got {other:?}"),
            }
            match ws.recv().await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "second"),
                other => panic!("expected second, got {other:?}"),
            }
            match ws.recv().await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "third"),
                other => panic!("expected third, got {other:?}"),
            }
        });
    }

    #[test]
    fn send_text_writes_frame() {
        let mut ws = ws_from_bytes(Vec::new());
        block_on(async {
            ws.send_text("hello").await.unwrap();
        });
        // Verify bytes were written to the mock (peek through the adapter).
        assert!(!ws.stream.get_ref().written.is_empty());
    }

    #[test]
    fn send_binary_writes_frame() {
        let mut ws = ws_from_bytes(Vec::new());
        block_on(async {
            ws.send_binary(&[1, 2, 3]).await.unwrap();
        });
        assert!(!ws.stream.get_ref().written.is_empty());
    }

    #[test]
    fn from_parts_construction() {
        let mock =
            NexusAsyncReadAdapter::new(MockStream::from_bytes(make_frame(true, 0x1, b"test")));
        let reader = FrameReader::builder().role(Role::Client).build();
        let writer = FrameWriter::new(Role::Client);
        let mut ws = WsStream::from_parts(mock, reader, writer);

        block_on(async {
            match ws.recv().await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "test"),
                other => panic!("expected Text, got {other:?}"),
            }
        });
    }

    /// A mock stream that fails all writes with BrokenPipe.
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
        let reader = FrameReader::builder().role(Role::Client).build();
        let writer = FrameWriter::new(Role::Client);
        let mut ws = WsStream::from_parts(mock, reader, writer);

        block_on(async {
            let result = ws.send_text("hello").await;
            assert!(result.is_err(), "send on broken stream should return error");

            let result = ws.send_binary(&[1, 2, 3]).await;
            assert!(result.is_err(), "subsequent send should also fail");
        });
    }
}

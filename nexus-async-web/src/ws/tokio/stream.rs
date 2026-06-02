//! Async WebSocket — tokio backend.
//!
//! Builder + handshake logic. `WsReader`/`WsWriter` (the primary API)
//! live in the shared `ws::parts` module.

use std::pin::Pin;

use nexus_net::WireStream;
use nexus_net::buf::WriteBuf;
#[cfg(feature = "tls")]
use nexus_net::tls::TlsConfig;
use nexus_web::http::HTTP_HANDSHAKE_BUFFER;
use nexus_web::ws::{
    CloseCode, Error as WsError, FrameReader, FrameReaderBuilder, FrameWriter, HandshakeError,
    Role, parse_ws_url,
};
use tokio::net::TcpStream;

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
// WsStream — Stream/Sink ecosystem adapter
// =============================================================================

/// Bundled WebSocket stream for `Stream`/`Sink` ecosystem compatibility.
///
/// This type exists solely to implement `futures_core::Stream` and
/// `futures_sink::Sink<OwnedMessage>`. It uses owned messages
/// (allocates per message) and cannot overlap read/write borrows.
///
/// For performance-sensitive code, use [`WsReader`] and [`WsWriter`]
/// directly — they provide zero-copy messages and independent borrows.
///
/// # Construction
///
/// ```ignore
/// // Connect, then bundle for Stream/Sink usage
/// let (reader, writer, conn) = WsStreamBuilder::new()
///     .connect("ws://localhost:8080/ws")
///     .await?;
/// let ws = WsStream::from_parts(reader, writer, conn);
/// ```
pub struct WsStream<S> {
    stream: S,
    reader: FrameReader,
    writer: FrameWriter,
    write_buf: WriteBuf,
    max_read_size: usize,
}

impl<S> WsStream<S> {
    /// Construct from decomposed parts.
    ///
    /// Inverse of [`into_parts`](Self::into_parts).
    pub fn from_parts(reader: WsReader, writer: WsWriter, stream: S) -> Self {
        Self {
            stream,
            max_read_size: reader.max_read_size,
            reader: reader.reader,
            writer: writer.writer,
            write_buf: writer.write_buf,
        }
    }

    /// Decompose into reader, writer, and transport stream.
    pub fn into_parts(self) -> (WsReader, WsWriter, S) {
        (
            WsReader {
                reader: self.reader,
                max_read_size: self.max_read_size,
            },
            WsWriter {
                writer: self.writer,
                write_buf: self.write_buf,
            },
            self.stream,
        )
    }

    /// Low-level construction from raw nexus-web types.
    ///
    /// For testing or custom handshakes. Prefer [`from_parts`](Self::from_parts)
    /// with builder-produced components.
    pub fn from_raw_parts(stream: S, reader: FrameReader, writer: FrameWriter) -> Self {
        Self {
            stream,
            reader,
            writer,
            write_buf: WriteBuf::new(65_536, 14),
            max_read_size: usize::MAX,
        }
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Builder for WebSocket connections.
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
    ///
    /// **Note:** This only affects `WsReader::recv()`. The `Stream`
    /// implementation on [`WsStream`] uses the full spare slice for
    /// compatibility with `StreamExt` combinators. For latency-sensitive
    /// code, use `WsReader::recv()` directly.
    #[must_use]
    pub fn max_read_size(mut self, n: usize) -> Self {
        self.max_read_size = Some(n);
        self
    }

    /// Fraction of buffer capacity consumed before proactive compaction.
    ///
    /// See [`FrameReaderBuilder::compact_at`](nexus_web::ws::FrameReaderBuilder::compact_at)
    /// for details. Default: 0.5.
    ///
    /// **Note:** This only affects `WsReader::recv()`. The `Stream`
    /// implementation does not use proactive compaction.
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
    pub async fn connect(self, url: &str) -> Result<(WsReader, WsWriter, MaybeTls), WsError> {
        let parsed = parse_ws_url(url)?;
        let addr = format!("{}:{}", parsed.host, parsed.port);

        let tcp = match self.connect_timeout {
            Some(timeout) => tokio::time::timeout(timeout, TcpStream::connect(&addr))
                .await
                .map_err(|_| {
                    WsError::Io(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "connect timeout",
                    ))
                })??,
            None => TcpStream::connect(&addr).await?,
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

                let connector =
                    tokio_rustls::TlsConnector::from(tls_config.client_config().clone());
                let server_name =
                    tokio_rustls::rustls::pki_types::ServerName::try_from(parsed.host.to_owned())
                        .map_err(|_| {
                        WsError::Tls(nexus_net::tls::TlsError::InvalidHostname(
                            parsed.host.to_string(),
                        ))
                    })?;
                let tls_stream = connector
                    .connect(server_name, tcp)
                    .await
                    .map_err(|e| WsError::Tls(nexus_net::tls::TlsError::Io(e)))?;
                MaybeTls::Tls(Box::new(tls_stream))
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
        let sock = socket2::SockRef::from(tcp);
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
// Stream + Sink (ecosystem compat — allocates per message)
// =============================================================================

use std::task::{Context, Poll};

use futures_core::Stream;
use futures_sink::Sink;
use nexus_web::ws::OwnedMessage;

impl<S: WireStream + Unpin> Stream for WsStream<S> {
    type Item = Result<OwnedMessage, WsError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            match this.reader.poll() {
                Ok(true) => {
                    return match this.reader.next() {
                        Ok(Some(msg)) => Poll::Ready(Some(Ok(msg.into_owned()))),
                        Ok(None) => Poll::Ready(None),
                        Err(e) => Poll::Ready(Some(Err(e.into()))),
                    };
                }
                Ok(false) => {}
                Err(e) => return Poll::Ready(Some(Err(e.into()))),
            }

            if this.reader.should_compact() {
                this.reader.compact();
            }
            if this.reader.spare().is_empty() {
                this.reader.compact();
                if this.reader.spare().is_empty() {
                    return Poll::Ready(Some(Err(std::io::Error::other(
                        "websocket read buffer full",
                    )
                    .into())));
                }
            }

            match Pin::new(&mut this.stream).poll_fill_into(
                cx,
                &mut this.reader,
                this.max_read_size,
            ) {
                Poll::Ready(Ok(0)) => return Poll::Ready(None),
                Poll::Ready(Ok(_)) => {}
                Poll::Ready(Err(e)) => return Poll::Ready(Some(Err(e.into()))),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl<S: WireStream + Unpin> Sink<OwnedMessage> for WsStream<S> {
    type Error = WsError;

    fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(self: Pin<&mut Self>, item: OwnedMessage) -> Result<(), Self::Error> {
        let this = self.get_mut();
        match &item {
            OwnedMessage::Text(s) => {
                this.writer
                    .encode_text_into(s.as_bytes(), &mut this.write_buf);
            }
            OwnedMessage::Binary(b) => {
                this.writer.encode_binary_into(b, &mut this.write_buf);
            }
            OwnedMessage::Ping(b) => {
                this.writer
                    .encode_ping_into(b, &mut this.write_buf)
                    .map_err(WsError::Encode)?;
            }
            OwnedMessage::Pong(b) => {
                this.writer
                    .encode_pong_into(b, &mut this.write_buf)
                    .map_err(WsError::Encode)?;
            }
            OwnedMessage::Close(cf) => {
                if cf.code == CloseCode::NoStatus {
                    let mut dst = [0u8; 14];
                    let n = this.writer.encode_empty_close(&mut dst);
                    this.write_buf.clear();
                    this.write_buf.append(&dst[..n]);
                } else {
                    this.writer
                        .encode_close_into(
                            cf.code.as_u16(),
                            cf.reason.as_bytes(),
                            &mut this.write_buf,
                        )
                        .map_err(WsError::Encode)?;
                }
            }
        }
        Ok(())
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.get_mut();
        while !this.write_buf.is_empty() {
            let data = this.write_buf.data();
            match Pin::new(&mut this.stream).poll_write(cx, data) {
                Poll::Ready(Ok(0)) => {
                    return Poll::Ready(Err(WsError::Io(std::io::Error::new(
                        std::io::ErrorKind::WriteZero,
                        "write returned 0",
                    ))));
                }
                Poll::Ready(Ok(n)) => {
                    this.write_buf.advance(n);
                }
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e.into())),
                Poll::Pending => return Poll::Pending,
            }
        }
        Pin::new(&mut this.stream)
            .poll_flush(cx)
            .map_err(WsError::Io)
    }

    fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        match <Self as Sink<OwnedMessage>>::poll_flush(self.as_mut(), cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
            Poll::Ready(Ok(())) => {}
        }
        let this = self.get_mut();
        Pin::new(&mut this.stream)
            .poll_shutdown(cx)
            .map_err(WsError::Io)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AsyncReadAdapter;
    use nexus_web::ws::Message;
    use std::io::Cursor;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

    struct MockStream(Cursor<Vec<u8>>);

    impl AsyncRead for MockStream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            let n = std::io::Read::read(&mut self.0, buf.initialize_unfilled())?;
            buf.advance(n);
            Poll::Ready(Ok(()))
        }
    }

    impl AsyncWrite for MockStream {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Poll::Ready(Ok(buf.len()))
        }
        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
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

    fn parts_from_bytes(data: Vec<u8>) -> (WsReader, WsWriter, AsyncReadAdapter<MockStream>) {
        let mock = AsyncReadAdapter::new(MockStream(Cursor::new(data)));
        let reader = FrameReader::builder().role(Role::Client).build();
        let writer = FrameWriter::new(Role::Client);
        let ws = WsStream::from_raw_parts(mock, reader, writer);
        ws.into_parts()
    }

    // -- Primary API tests (WsReader / WsWriter) -----------------------------

    #[tokio::test]
    async fn recv_text() {
        let frame = make_frame(true, 0x1, b"Hello");
        let (mut reader, _writer, mut conn) = parts_from_bytes(frame);
        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Text(s) => assert_eq!(s, "Hello"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recv_binary() {
        let frame = make_frame(true, 0x2, &[0x42; 100]);
        let (mut reader, _writer, mut conn) = parts_from_bytes(frame);
        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Binary(b) => assert_eq!(b.len(), 100),
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recv_ping() {
        let frame = make_frame(true, 0x9, b"ping");
        let (mut reader, _writer, mut conn) = parts_from_bytes(frame);
        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Ping(p) => assert_eq!(p, b"ping"),
            other => panic!("expected Ping, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recv_fragmented_text() {
        let mut data = make_frame(false, 0x1, b"Hel");
        data.extend_from_slice(&make_frame(true, 0x0, b"lo"));
        let (mut reader, _writer, mut conn) = parts_from_bytes(data);
        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Text(s) => assert_eq!(s, "Hello"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recv_fragment_with_control() {
        let mut data = make_frame(false, 0x1, b"Hel");
        data.extend_from_slice(&make_frame(true, 0x9, b"ping"));
        data.extend_from_slice(&make_frame(true, 0x0, b"lo"));
        let (mut reader, _writer, mut conn) = parts_from_bytes(data);
        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Ping(p) => assert_eq!(p, b"ping"),
            other => panic!("expected Ping, got {other:?}"),
        }
        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Text(s) => assert_eq!(s, "Hello"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recv_close() {
        let mut payload = vec![];
        payload.extend_from_slice(&1000u16.to_be_bytes());
        payload.extend_from_slice(b"bye");
        let frame = make_frame(true, 0x8, &payload);
        let (mut reader, _writer, mut conn) = parts_from_bytes(frame);
        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Close(cf) => {
                assert_eq!(cf.code, CloseCode::Normal);
                assert_eq!(cf.reason, "bye");
            }
            other => panic!("expected Close, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn eof_returns_none() {
        let (mut reader, _writer, mut conn) = parts_from_bytes(Vec::new());
        assert!(reader.recv(&mut conn).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn fifo_three_messages() {
        let mut data = make_frame(true, 0x1, b"first");
        data.extend_from_slice(&make_frame(true, 0x1, b"second"));
        data.extend_from_slice(&make_frame(true, 0x1, b"third"));
        let (mut reader, _writer, mut conn) = parts_from_bytes(data);

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
    }

    #[tokio::test]
    async fn ping_echo_split_borrow() {
        let mut data = make_frame(true, 0x9, b"ping-data");
        data.extend_from_slice(&make_frame(true, 0x1, b"hello"));
        let (mut reader, mut writer, mut conn) = parts_from_bytes(data);

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
    }

    #[tokio::test]
    async fn text_response_while_holding_message() {
        let data = make_frame(true, 0x1, b"request");
        let (mut reader, mut writer, mut conn) = parts_from_bytes(data);

        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Text(req) => {
                assert_eq!(req, "request");
                let response = format!("echo: {req}");
                writer.send_text(&mut conn, &response).await.unwrap();
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    // -- Stream/Sink tests (WsStream) ----------------------------------------

    #[tokio::test]
    async fn stream_yields_owned_messages() {
        use std::pin::pin;

        let mut data = make_frame(true, 0x1, b"hello");
        data.extend_from_slice(&make_frame(true, 0x2, &[0x42]));
        let (reader, writer, conn) = parts_from_bytes(data);
        let ws = WsStream::from_parts(reader, writer, conn);
        let mut ws = pin!(ws);

        let poll_result = futures_core::Stream::poll_next(ws.as_mut(), &mut noop_cx());
        match poll_result {
            Poll::Ready(Some(Ok(OwnedMessage::Text(s)))) => assert_eq!(s, "hello"),
            other => panic!("expected Text, got {other:?}"),
        }
        let poll_result = futures_core::Stream::poll_next(ws.as_mut(), &mut noop_cx());
        match poll_result {
            Poll::Ready(Some(Ok(OwnedMessage::Binary(b)))) => assert_eq!(b, vec![0x42]),
            other => panic!("expected Binary, got {other:?}"),
        }
        let poll_result = futures_core::Stream::poll_next(ws.as_mut(), &mut noop_cx());
        assert!(matches!(poll_result, Poll::Ready(None)));
    }

    fn noop_cx() -> Context<'static> {
        use std::task::{RawWaker, RawWakerVTable, Waker};
        const VTABLE: RawWakerVTable =
            RawWakerVTable::new(|p| RawWaker::new(p, &VTABLE), |_| {}, |_| {}, |_| {});
        // SAFETY: The vtable functions (clone/wake/wake_by_ref/drop) are all no-ops
        // that never dereference the data pointer, so the null data pointer is sound.
        // The vtable is 'static (const) and correctly returns a valid RawWaker on clone.
        let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
        let waker = Box::leak(Box::new(waker));
        Context::from_waker(waker)
    }

    #[tokio::test]
    async fn accept_server_side() {
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
                .accept(AsyncReadAdapter::new(tcp))
                .await
                .unwrap();
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "hello from client"),
                other => panic!("expected Text, got {other:?}"),
            }
            writer
                .send_text(&mut conn, "hello from server")
                .await
                .unwrap();
        });

        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        let url = format!("ws://127.0.0.1:{}/ws", addr.port());
        let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
            .connect_with(AsyncReadAdapter::new(tcp), &url)
            .await
            .unwrap();

        writer
            .send_text(&mut conn, "hello from client")
            .await
            .unwrap();

        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Text(s) => assert_eq!(s, "hello from server"),
            other => panic!("expected Text, got {other:?}"),
        }

        server.await.unwrap();
    }

    struct BrokenWriteStream(Cursor<Vec<u8>>);

    impl AsyncRead for BrokenWriteStream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            let n = std::io::Read::read(&mut self.0, buf.initialize_unfilled())?;
            buf.advance(n);
            Poll::Ready(Ok(()))
        }
    }

    impl AsyncWrite for BrokenWriteStream {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Poll::Ready(Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "connection lost",
            )))
        }
        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn send_on_broken_stream_returns_error() {
        let mock = AsyncReadAdapter::new(BrokenWriteStream(Cursor::new(Vec::new())));
        let reader = FrameReader::builder().role(Role::Client).build();
        let writer = FrameWriter::new(Role::Client);
        let (_, mut ws_writer, mut conn) =
            WsStream::from_raw_parts(mock, reader, writer).into_parts();

        let result = ws_writer.send_text(&mut conn, "hello").await;
        assert!(result.is_err(), "send on broken stream should return error");

        let result = ws_writer.send_binary(&mut conn, &[1, 2, 3]).await;
        assert!(result.is_err(), "subsequent send should also fail");
    }

    #[tokio::test]
    async fn from_parts_roundtrip() {
        let data = make_frame(true, 0x1, b"test");
        let (reader, writer, conn) = parts_from_bytes(data);
        let ws = WsStream::from_parts(reader, writer, conn);
        let (mut reader, _writer, mut conn) = ws.into_parts();

        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Text(s) => assert_eq!(s, "test"),
            other => panic!("expected Text, got {other:?}"),
        }
    }
}

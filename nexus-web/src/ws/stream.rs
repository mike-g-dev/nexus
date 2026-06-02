//! WebSocket stream — I/O wrapper with HTTP upgrade handshake.

use std::time::Duration;

use super::error::ProtocolError;
use super::frame::Role;
use super::frame_reader::{FrameReader, FrameReaderBuilder};
use super::frame_writer::FrameWriter;
use super::message::{CloseCode, Message};
use nexus_net::buf::WriteBuf;

use super::handshake;
use super::handshake::HandshakeError;
use std::io::{self, Read, Write};

#[cfg(feature = "tls")]
use nexus_net::tls::{TlsConfig, TlsError};

// =============================================================================
// URL parsing
// =============================================================================

/// Parsed WebSocket URL.
#[non_exhaustive]
pub struct ParsedUrl<'a> {
    /// Whether the URL is `wss://` (true) or `ws://` (false).
    pub tls: bool,
    /// Host portion (no port).
    pub host: &'a str,
    /// Port — explicit if present, otherwise the scheme default
    /// (80 for ws, 443 for wss).
    pub port: u16,
    /// Path portion (everything after the host:port, including the
    /// leading `/`). Defaults to `/` when absent in the input URL.
    pub path: &'a str,
}

impl ParsedUrl<'_> {
    /// Host header value: includes port if non-default.
    pub fn host_header(&self) -> String {
        let default = if self.tls { 443 } else { 80 };
        if self.port == default {
            self.host.to_string()
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

/// Parse a `ws://` or `wss://` URL into its scheme, host, port, and
/// path. Supports IPv6 bracket notation (`[::1]:8080`). Returns
/// [`Error::InvalidUrl`] on a malformed input or missing scheme.
pub fn parse_ws_url(url: &str) -> Result<ParsedUrl<'_>, Error> {
    let (tls, rest) = if let Some(r) = url.strip_prefix("wss://") {
        (true, r)
    } else if let Some(r) = url.strip_prefix("ws://") {
        (false, r)
    } else {
        return Err(Error::InvalidUrl(url.to_string()));
    };

    let (host_port, path) = rest
        .find('/')
        .map_or((rest, "/"), |i| (&rest[..i], &rest[i..]));

    if host_port.is_empty() {
        return Err(Error::InvalidUrl(format!("empty host: {url}")));
    }

    let default_port = if tls { 443 } else { 80 };

    // IPv6 bracket notation: [::1]:8080
    let (host, port) = if host_port.starts_with('[') {
        match host_port.find(']') {
            Some(end) => {
                let h = &host_port[1..end];
                let rest = &host_port[end + 1..];
                if let Some(port_str) = rest.strip_prefix(':') {
                    let p = port_str
                        .parse::<u16>()
                        .map_err(|_| Error::InvalidUrl(format!("invalid port: {url}")))?;
                    (h, p)
                } else {
                    (h, default_port)
                }
            }
            None => return Err(Error::InvalidUrl(format!("unclosed bracket: {url}"))),
        }
    } else {
        match host_port.rfind(':') {
            None => (host_port, default_port),
            Some(i) => {
                let port_str = &host_port[i + 1..];
                if port_str.is_empty() {
                    (&host_port[..i], default_port)
                } else {
                    let p = port_str
                        .parse::<u16>()
                        .map_err(|_| Error::InvalidUrl(format!("invalid port: {url}")))?;
                    (&host_port[..i], p)
                }
            }
        }
    };

    Ok(ParsedUrl {
        tls,
        host,
        port,
        path,
    })
}

// =============================================================================
// Error
// =============================================================================

/// Unified error type for WebSocket stream operations.
#[derive(Debug)]
pub enum Error {
    /// I/O error from the underlying stream.
    Io(std::io::Error),
    /// WebSocket protocol error.
    Protocol(ProtocolError),
    /// Encoding error (e.g., control frame payload too large).
    Encode(super::frame_writer::EncodeError),
    /// HTTP handshake failed.
    Handshake(HandshakeError),
    /// TLS error during connection setup (handshake, certificate
    /// validation, SNI hostname verification).
    ///
    /// **Steady-state TLS protocol errors** (decrypt failure, peer
    /// alert, malformed record received during a frame) on the async
    /// `nexus-async-web` paths surface as [`Error::Io`](Self::Io)
    /// instead — the underlying [`TlsError`](nexus_net::tls::TlsError) is
    /// wrapped via `io::Error::other` and reachable via
    /// `io_err.source()` or `io_err.get_ref()`. This asymmetry stems
    /// from the [`WireStream`](crate::WireStream) trait returning
    /// `io::Result` for poll methods. Sync WS surfaces `Tls` directly
    /// because its `TlsStream` exposes `TlsError` natively. Pattern-
    /// match on both `Io` and `Tls` if you need to distinguish TLS-
    /// protocol failures from generic transport failures across both
    /// surfaces.
    #[cfg(feature = "tls")]
    Tls(TlsError),
    /// Invalid WebSocket URL.
    InvalidUrl(String),
    /// `wss://` URL used without the `tls` feature enabled.
    TlsNotEnabled,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Protocol(e) => write!(f, "protocol error: {e}"),
            Self::Encode(e) => write!(f, "encode error: {e}"),
            Self::Handshake(e) => write!(f, "handshake error: {e}"),
            #[cfg(feature = "tls")]
            Self::Tls(e) => write!(f, "TLS error: {e}"),
            Self::InvalidUrl(u) => write!(f, "invalid WebSocket URL: {u}"),
            Self::TlsNotEnabled => write!(f, "wss:// requires the 'tls' feature"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<ProtocolError> for Error {
    fn from(e: ProtocolError) -> Self {
        Self::Protocol(e)
    }
}
impl From<super::frame_writer::EncodeError> for Error {
    fn from(e: super::frame_writer::EncodeError) -> Self {
        Self::Encode(e)
    }
}
impl From<HandshakeError> for Error {
    fn from(e: HandshakeError) -> Self {
        Self::Handshake(e)
    }
}
#[cfg(feature = "tls")]
impl From<TlsError> for Error {
    fn from(e: TlsError) -> Self {
        match e {
            TlsError::Io(io) => Self::Io(io),
            other => Self::Tls(other),
        }
    }
}

// =============================================================================
// ClientBuilder
// =============================================================================

/// Builder for [`Client`].
///
/// Configures buffer sizes, socket options, and optional TLS.
///
/// # Examples
///
/// ```ignore
/// let mut ws = Client::builder()
///     .disable_nagle()
///     .buffer_capacity(2 * 1024 * 1024)
///     .connect("wss://exchange.com/ws")?;
/// ```
pub struct ClientBuilder {
    pub(crate) reader_builder: FrameReaderBuilder,
    pub(crate) write_buf_capacity: usize,
    pub(crate) write_buf_headroom: usize,
    #[cfg(feature = "tls")]
    pub(crate) tls_config: Option<TlsConfig>,
    pub(crate) tcp_nodelay: bool,
    #[cfg(feature = "socket-opts")]
    pub(crate) recv_buf_size: Option<usize>,
    #[cfg(feature = "socket-opts")]
    pub(crate) send_buf_size: Option<usize>,
    pub(crate) connect_timeout: Option<Duration>,
    pub(crate) read_timeout: Option<Duration>,
}

impl ClientBuilder {
    /// Create a new builder with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            reader_builder: FrameReader::builder(),
            write_buf_capacity: 65_536,
            write_buf_headroom: 14,
            #[cfg(feature = "tls")]
            tls_config: None,
            tcp_nodelay: false,
            #[cfg(feature = "socket-opts")]
            recv_buf_size: None,
            #[cfg(feature = "socket-opts")]
            send_buf_size: None,
            connect_timeout: None,
            read_timeout: None,
        }
    }

    /// ReadBuf capacity. Default: 1MB.
    #[must_use]
    pub fn buffer_capacity(mut self, n: usize) -> Self {
        self.reader_builder = self.reader_builder.buffer_capacity(n);
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

    /// Set `TCP_NODELAY` (disable Nagle's algorithm).
    #[must_use]
    pub fn disable_nagle(mut self) -> Self {
        self.tcp_nodelay = true;
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

    /// TCP connect timeout.
    #[must_use]
    pub fn connect_timeout(mut self, d: Duration) -> Self {
        self.connect_timeout = Some(d);
        self
    }

    /// Socket read timeout.
    #[must_use]
    pub fn read_timeout(mut self, d: Duration) -> Self {
        self.read_timeout = Some(d);
        self
    }

    /// Set a custom TLS configuration.
    ///
    /// If not set, `wss://` URLs use [`TlsConfig::new()`] (system defaults).
    #[cfg(feature = "tls")]
    #[must_use]
    pub fn tls(mut self, config: &TlsConfig) -> Self {
        self.tls_config = Some(config.clone());
        self
    }

    /// Connect to a WebSocket server (blocking).
    ///
    /// Creates a TCP socket, applies socket options, and performs the
    /// full handshake (TLS if `wss://`, then HTTP upgrade).
    ///
    /// When the `tls` feature is enabled, returns `Client<MaybeTls<TcpStream>>`
    /// regardless of scheme — `ws://` uses `MaybeTls::Plain`, `wss://` uses
    /// `MaybeTls::Tls`. Without the `tls` feature, returns `Client<TcpStream>`
    /// and errors on `wss://`.
    #[cfg(feature = "tls")]
    pub fn connect(
        self,
        url: &str,
    ) -> Result<Client<nexus_net::MaybeTls<std::net::TcpStream>>, Error> {
        let parsed = parse_ws_url(url)?;
        let addr = format!("{}:{}", parsed.host, parsed.port);

        let tcp = match self.connect_timeout {
            Some(timeout) => {
                let addrs: Vec<std::net::SocketAddr> =
                    std::net::ToSocketAddrs::to_socket_addrs(&addr)
                        .map_err(Error::Io)?
                        .collect();
                let first = addrs
                    .first()
                    .ok_or_else(|| Error::Io(io::Error::other("DNS resolution failed")))?;
                std::net::TcpStream::connect_timeout(first, timeout)?
            }
            None => std::net::TcpStream::connect(&addr)?,
        };

        self.apply_socket_opts(&tcp)?;

        let stream = if parsed.tls {
            let config = match self.tls_config {
                Some(c) => c,
                None => TlsConfig::new().map_err(Error::Tls)?,
            };
            let codec = nexus_net::tls::TlsCodec::new(&config, parsed.host)?;
            let tls = nexus_net::tls::TlsStream::connect(tcp, codec).map_err(Error::Tls)?;
            nexus_net::MaybeTls::Tls(Box::new(tls))
        } else {
            nexus_net::MaybeTls::Plain(tcp)
        };

        let host_header = parsed.host_header();
        Client::connect_impl(
            stream,
            &host_header,
            parsed.path,
            self.reader_builder,
            self.write_buf_capacity,
            self.write_buf_headroom,
        )
    }

    /// Connect to a WebSocket server (blocking, no TLS feature).
    #[cfg(not(feature = "tls"))]
    pub fn connect(self, url: &str) -> Result<Client<std::net::TcpStream>, Error> {
        let parsed = parse_ws_url(url)?;
        if parsed.tls {
            return Err(Error::TlsNotEnabled);
        }
        let addr = format!("{}:{}", parsed.host, parsed.port);

        let tcp = match self.connect_timeout {
            Some(timeout) => {
                let addrs: Vec<std::net::SocketAddr> =
                    std::net::ToSocketAddrs::to_socket_addrs(&addr)
                        .map_err(Error::Io)?
                        .collect();
                let first = addrs
                    .first()
                    .ok_or_else(|| Error::Io(io::Error::other("DNS resolution failed")))?;
                std::net::TcpStream::connect_timeout(first, timeout)?
            }
            None => std::net::TcpStream::connect(&addr)?,
        };

        self.apply_socket_opts(&tcp)?;

        let host_header = parsed.host_header();
        Client::connect_impl(
            tcp,
            &host_header,
            parsed.path,
            self.reader_builder,
            self.write_buf_capacity,
            self.write_buf_headroom,
        )
    }

    /// Connect using a pre-connected stream.
    ///
    /// The stream must already handle TLS if connecting to `wss://`.
    /// For example, pass a `TlsStream<TcpStream>` or `MaybeTls<TcpStream>`.
    /// This method only performs the HTTP upgrade handshake.
    pub fn connect_with<S: Read + Write>(self, stream: S, url: &str) -> Result<Client<S>, Error> {
        let parsed = parse_ws_url(url)?;
        let host_header = parsed.host_header();
        Client::connect_impl(
            stream,
            &host_header,
            parsed.path,
            self.reader_builder,
            self.write_buf_capacity,
            self.write_buf_headroom,
        )
    }

    /// Accept an incoming WebSocket connection (server-side).
    pub fn accept<S: Read + Write>(self, stream: S) -> Result<Client<S>, Error> {
        Client::accept_impl(
            stream,
            self.reader_builder,
            self.write_buf_capacity,
            self.write_buf_headroom,
        )
    }

    fn apply_socket_opts(&self, tcp: &std::net::TcpStream) -> Result<(), Error> {
        if self.tcp_nodelay {
            tcp.set_nodelay(true)?;
        }
        if let Some(timeout) = self.read_timeout {
            tcp.set_read_timeout(Some(timeout))?;
        }
        #[cfg(feature = "socket-opts")]
        {
            let sock = socket2::SockRef::from(tcp);
            if let Some(size) = self.recv_buf_size {
                sock.set_recv_buffer_size(size)?;
            }
            if let Some(size) = self.send_buf_size {
                sock.set_send_buffer_size(size)?;
            }
        }
        Ok(())
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Client
// =============================================================================

/// WebSocket stream — owns a socket, reader, writer, and buffers.
///
/// Handles both plain `ws://` and encrypted `wss://` connections.
/// The URL scheme determines whether TLS is used — no separate type needed.
///
/// # Usage
///
/// ```ignore
/// use nexus_web::ws::Client;
/// use nexus_web::tls::TlsConfig;
///
/// // Plain WebSocket
/// let mut ws = Client::builder().connect("ws://localhost:8080/ws")?;
///
/// // TLS WebSocket (requires 'tls' feature)
/// let tls = TlsConfig::new()?;
/// let mut ws = Client::builder().tls(&tls).connect("wss://exchange.com/ws")?;
///
/// // Same API for both:
/// ws.send_text("Hello!")?;
/// while let Some(msg) = ws.recv()? {
///     // ...
/// }
/// ```
pub struct Client<S> {
    pub(crate) stream: S,
    pub(crate) reader: FrameReader,
    pub(crate) writer: FrameWriter,
    pub(crate) write_buf: WriteBuf,
    pub(crate) poisoned: bool,
}

impl Client<std::net::TcpStream> {
    /// Create a builder for configuring buffer sizes, socket options, and TLS.
    #[must_use]
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }
}

// -- Unbounded impl: accessors and constructors that need no I/O traits -------

impl<S> Client<S> {
    /// Create from pre-existing parts. For testing or custom handshakes.
    pub fn from_parts(stream: S, reader: FrameReader, writer: FrameWriter) -> Self {
        Self {
            stream,
            reader,
            writer,
            write_buf: WriteBuf::new(65_536, 14),
            poisoned: false,
        }
    }

    /// Internal constructor with all fields. Used by Connecting::finish().
    pub(crate) fn from_parts_internal(
        stream: S,
        reader: FrameReader,
        writer: FrameWriter,
        write_buf: WriteBuf,
    ) -> Self {
        Self {
            stream,
            reader,
            writer,
            write_buf,
            poisoned: false,
        }
    }

    /// Whether the stream is poisoned (I/O error occurred during send).
    ///
    /// A poisoned stream should not be reused — the connection may be
    /// in an indeterminate state (partial frame written).
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
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
}

// -- Blocking I/O impl --------------------------------------------------------

impl<S: Read + Write> Client<S> {
    /// Connect using a pre-connected socket with default configuration.
    ///
    /// IPv6 addresses must use bracket notation: `ws://[::1]:8080/path`.
    pub fn connect_with(stream: S, url: &str) -> Result<Self, Error> {
        ClientBuilder::new().connect_with(stream, url)
    }

    /// Accept an incoming WebSocket connection (server-side).
    pub fn accept(stream: S) -> Result<Self, Error> {
        ClientBuilder::new().accept(stream)
    }

    /// Receive the next message. Reads from the socket as needed.
    ///
    /// Returns `Ok(None)` on EOF, buffer full, or `WouldBlock` (non-blocking sockets).
    pub fn recv(&mut self) -> Result<Option<Message<'_>>, Error> {
        loop {
            if self.reader.poll()? {
                return Ok(self.reader.next()?);
            }
            match self.read_into_reader() {
                Ok(0) => return Ok(None),
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(None),
                Err(e) => return Err(Error::Io(e)),
            }
        }
    }

    /// Send a text message.
    pub fn send_text(&mut self, text: &str) -> Result<(), Error> {
        self.writer
            .encode_text_into(text.as_bytes(), &mut self.write_buf);
        self.flush_write_buf_or_poison()
    }

    /// Send a binary message.
    pub fn send_binary(&mut self, data: &[u8]) -> Result<(), Error> {
        self.writer.encode_binary_into(data, &mut self.write_buf);
        self.flush_write_buf_or_poison()
    }

    /// Send a ping.
    pub fn send_ping(&mut self, data: &[u8]) -> Result<(), Error> {
        self.writer
            .encode_ping_into(data, &mut self.write_buf)
            .map_err(Error::Encode)?;
        self.flush_write_buf_or_poison()
    }

    /// Send a pong.
    pub fn send_pong(&mut self, data: &[u8]) -> Result<(), Error> {
        self.writer
            .encode_pong_into(data, &mut self.write_buf)
            .map_err(Error::Encode)?;
        self.flush_write_buf_or_poison()
    }

    /// Initiate close handshake.
    pub fn close(&mut self, code: CloseCode, reason: &str) -> Result<(), Error> {
        if code == CloseCode::NoStatus {
            let mut dst = [0u8; 14];
            let n = self.writer.encode_empty_close(&mut dst);
            self.write_raw(&dst[..n]).inspect_err(|_| {
                self.poisoned = true;
            })
        } else {
            self.writer
                .encode_close_into(code.as_u16(), reason.as_bytes(), &mut self.write_buf)
                .map_err(Error::Encode)?;
            self.flush_write_buf_or_poison()
        }
    }

    // =========================================================================
    // Internal — read/write with optional TLS
    // =========================================================================

    /// Read bytes into the FrameReader.
    ///
    /// TLS is now handled at the stream level (`TlsStream<S>` or
    /// `MaybeTls<S>`), so this always reads plaintext from `S`.
    fn read_into_reader(&mut self) -> io::Result<usize> {
        self.reader.read_from(&mut self.stream)
    }

    /// Flush write_buf, poisoning on I/O error.
    fn flush_write_buf_or_poison(&mut self) -> Result<(), Error> {
        self.flush_write_buf().inspect_err(|_| {
            self.poisoned = true;
        })
    }

    /// Flush the write_buf to the socket.
    fn flush_write_buf(&mut self) -> Result<(), Error> {
        self.stream.write_all(self.write_buf.data())?;
        Ok(())
    }

    /// Write raw bytes to the socket.
    fn write_raw(&mut self, data: &[u8]) -> Result<(), Error> {
        self.stream.write_all(data)?;
        Ok(())
    }

    // =========================================================================
    // Internal — handshake
    // =========================================================================

    /// Perform the HTTP upgrade handshake on a stream that is already
    /// plaintext-ready (TLS handled at the stream level).
    pub(crate) fn connect_impl(
        mut stream: S,
        host: &str,
        path: &str,
        reader_builder: FrameReaderBuilder,
        write_cap: usize,
        write_headroom: usize,
    ) -> Result<Self, Error> {
        let key = handshake::generate_key();
        let key_str = std::str::from_utf8(&key).expect("base64 output is valid ASCII");

        let headers = [
            ("Host", host),
            ("Upgrade", "websocket"),
            ("Connection", "Upgrade"),
            ("Sec-WebSocket-Key", key_str),
            ("Sec-WebSocket-Version", "13"),
        ];
        let req_size = crate::http::request_size("GET", path, &headers);
        let mut req_buf = vec![0u8; req_size];
        let n = crate::http::write_request("GET", path, &headers, &mut req_buf)
            .map_err(|_| HandshakeError::MalformedHttp)?;

        stream.write_all(&req_buf[..n])?;

        let mut resp_reader = crate::http::ResponseReader::new(4096);
        let mut tmp = [0u8; 4096];
        loop {
            let bytes_read = stream.read(&mut tmp)?;
            if bytes_read == 0 {
                return Err(HandshakeError::MalformedHttp.into());
            }

            resp_reader
                .read(&tmp[..bytes_read])
                .map_err(|_| HandshakeError::MalformedHttp)?;
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
                    if !contains_ignore_case(conn, "upgrade") {
                        return Err(HandshakeError::MissingConnection.into());
                    }
                    let accept = resp
                        .header("Sec-WebSocket-Accept")
                        .ok_or(HandshakeError::InvalidAcceptKey)?;
                    if !handshake::validate_accept(key_str, accept) {
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
                        write_buf: WriteBuf::new(write_cap, write_headroom),
                        poisoned: false,
                    });
                }
                Ok(None) => {} // need more bytes
                Err(_) => return Err(HandshakeError::MalformedHttp.into()),
            }
        }
    }

    fn accept_impl(
        mut stream: S,
        reader_builder: FrameReaderBuilder,
        write_cap: usize,
        write_headroom: usize,
    ) -> Result<Self, Error> {
        let mut req_reader = crate::http::RequestReader::new(4096);
        let mut tmp = [0u8; 4096];

        let ws_key;
        loop {
            let n = stream.read(&mut tmp)?;
            if n == 0 {
                return Err(HandshakeError::MalformedHttp.into());
            }
            req_reader
                .read(&tmp[..n])
                .map_err(|_| HandshakeError::MalformedHttp)?;
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
                    if !contains_ignore_case(conn, "upgrade") {
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

        let accept = handshake::compute_accept_key(&ws_key);
        let accept_str = std::str::from_utf8(&accept).expect("base64 output is valid ASCII");

        let resp_headers = [
            ("Upgrade", "websocket"),
            ("Connection", "Upgrade"),
            ("Sec-WebSocket-Accept", accept_str),
        ];
        let resp_size = crate::http::response_size("Switching Protocols", &resp_headers);
        let mut resp_buf = vec![0u8; resp_size];
        let n =
            crate::http::write_response(101, "Switching Protocols", &resp_headers, &mut resp_buf)
                .map_err(|_| HandshakeError::MalformedHttp)?;
        stream.write_all(&resp_buf[..n])?;

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
            write_buf: WriteBuf::new(write_cap, write_headroom),
            poisoned: false,
        })
    }
}

/// Create a matched FrameReader + FrameWriter pair.
///
/// Prevents mismatched roles between reader and writer.
pub fn pair(role: Role) -> (FrameReader, FrameWriter) {
    (
        FrameReader::builder().role(role).build(),
        FrameWriter::new(role),
    )
}

/// Create a pair with a configured FrameReader.
pub fn pair_with(role: Role, reader_builder: FrameReaderBuilder) -> (FrameReader, FrameWriter) {
    (reader_builder.role(role).build(), FrameWriter::new(role))
}

fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    haystack
        .as_bytes()
        .windows(needle.len())
        .any(|w| w.eq_ignore_ascii_case(needle.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // URL parsing
    // =========================================================================

    #[test]
    fn parse_ws_url_plain() {
        let p = parse_ws_url("ws://localhost:8080/ws").unwrap();
        assert!(!p.tls);
        assert_eq!(p.host, "localhost");
        assert_eq!(p.port, 8080);
        assert_eq!(p.path, "/ws");
    }

    #[test]
    fn parse_ws_url_tls() {
        let p = parse_ws_url("wss://exchange.com/ws/v1").unwrap();
        assert!(p.tls);
        assert_eq!(p.host, "exchange.com");
        assert_eq!(p.port, 443);
        assert_eq!(p.path, "/ws/v1");
    }

    #[test]
    fn parse_ws_url_default_port() {
        let p = parse_ws_url("ws://host/path").unwrap();
        assert_eq!(p.port, 80);

        let p = parse_ws_url("wss://host/path").unwrap();
        assert_eq!(p.port, 443);
    }

    #[test]
    fn parse_ws_url_no_path() {
        let p = parse_ws_url("ws://host").unwrap();
        assert_eq!(p.path, "/");
    }

    #[test]
    fn parse_ws_url_invalid_scheme() {
        assert!(parse_ws_url("http://host").is_err());
        assert!(parse_ws_url("host/path").is_err());
    }

    // =========================================================================
    // Blocking Client tests
    // =========================================================================

    mod sync_tests {
        use super::*;
        use std::io::{self, Read, Write};

        #[test]
        fn pair_creates_matching_roles() {
            let (mut reader, _writer) = pair(Role::Client);
            let frame = make_frame(true, 0x1, b"test");
            reader.read(&frame).unwrap();
            let msg = reader.next().unwrap().unwrap();
            assert!(matches!(msg, Message::Text(s) if s == "test"));
        }

        struct ByteAtATimeStream {
            data: Vec<u8>,
            pos: usize,
        }

        impl ByteAtATimeStream {
            fn new(data: Vec<u8>) -> Self {
                Self { data, pos: 0 }
            }
        }

        impl Read for ByteAtATimeStream {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if self.pos >= self.data.len() {
                    return Ok(0);
                }
                buf[0] = self.data[self.pos];
                self.pos += 1;
                Ok(1)
            }
        }

        impl Write for ByteAtATimeStream {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
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

        fn ws_from_bytes(data: Vec<u8>) -> Client<ByteAtATimeStream> {
            let mock = ByteAtATimeStream::new(data);
            let reader = FrameReader::builder().role(Role::Client).build();
            let writer = FrameWriter::new(Role::Client);
            Client::from_parts(mock, reader, writer)
        }

        #[test]
        fn recv_text() {
            let frame = make_frame(true, 0x1, b"Hello");
            let mut ws = ws_from_bytes(frame);
            match ws.recv().unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "Hello"),
                other => panic!("expected Text, got {other:?}"),
            }
        }

        #[test]
        fn recv_ping() {
            let frame = make_frame(true, 0x9, &[0x42; 125]);
            let mut ws = ws_from_bytes(frame);
            match ws.recv().unwrap().unwrap() {
                Message::Ping(p) => assert_eq!(p.len(), 125),
                other => panic!("expected Ping, got {other:?}"),
            }
        }

        #[test]
        fn recv_fragmented_text() {
            let mut data = make_frame(false, 0x1, b"Hel");
            data.extend_from_slice(&make_frame(true, 0x0, b"lo"));
            let mut ws = ws_from_bytes(data);
            match ws.recv().unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "Hello"),
                other => panic!("expected Text, got {other:?}"),
            }
        }

        #[test]
        fn recv_fragment_with_ping() {
            let mut data = make_frame(false, 0x1, b"Hel");
            data.extend_from_slice(&make_frame(true, 0x9, b"ping"));
            data.extend_from_slice(&make_frame(true, 0x0, b"lo"));
            let mut ws = ws_from_bytes(data);
            match ws.recv().unwrap().unwrap() {
                Message::Ping(p) => assert_eq!(p, b"ping"),
                other => panic!("expected Ping, got {other:?}"),
            }
            match ws.recv().unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "Hello"),
                other => panic!("expected Text, got {other:?}"),
            }
        }

        #[test]
        fn recv_close() {
            let mut payload = vec![];
            payload.extend_from_slice(&1000u16.to_be_bytes());
            payload.extend_from_slice(b"bye");
            let frame = make_frame(true, 0x8, &payload);
            let mut ws = ws_from_bytes(frame);
            match ws.recv().unwrap().unwrap() {
                Message::Close(cf) => {
                    assert_eq!(cf.code, CloseCode::Normal);
                    assert_eq!(cf.reason, "bye");
                }
                other => panic!("expected Close, got {other:?}"),
            }
        }

        #[test]
        fn eof_returns_none() {
            let mut ws = ws_from_bytes(Vec::new());
            assert!(ws.recv().unwrap().is_none());
        }

        #[test]
        fn would_block_returns_none() {
            struct WouldBlockStream;
            impl Read for WouldBlockStream {
                fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
                    Err(io::Error::new(io::ErrorKind::WouldBlock, "would block"))
                }
            }
            impl Write for WouldBlockStream {
                fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                    Ok(buf.len())
                }
                fn flush(&mut self) -> io::Result<()> {
                    Ok(())
                }
            }

            let reader = FrameReader::builder().role(Role::Client).build();
            let writer = FrameWriter::new(Role::Client);
            let mut ws = Client::from_parts(WouldBlockStream, reader, writer);
            assert!(ws.recv().unwrap().is_none());
        }
    }

    // =========================================================================
    // ws::Error variant coverage
    // =========================================================================

    #[test]
    fn ws_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "broken");
        let err = Error::from(io_err);
        assert!(matches!(err, Error::Io(_)));
        assert!(err.to_string().contains("broken"));
    }

    #[test]
    fn ws_error_protocol() {
        let proto = ProtocolError::InvalidUtf8;
        let err = Error::from(proto);
        assert!(matches!(err, Error::Protocol(ProtocolError::InvalidUtf8)));
        assert!(err.to_string().contains("protocol error"));
    }

    #[test]
    fn ws_error_encode() {
        let enc = crate::ws::EncodeError::ControlPayloadTooLarge(200);
        let err = Error::from(enc);
        assert!(matches!(err, Error::Encode(_)));
        assert!(err.to_string().contains("encode error"));
    }

    #[test]
    fn ws_error_handshake() {
        let hs = HandshakeError::MissingUpgrade;
        let err = Error::from(hs);
        assert!(matches!(
            err,
            Error::Handshake(HandshakeError::MissingUpgrade)
        ));
        assert!(err.to_string().contains("handshake error"));
    }

    #[test]
    fn ws_error_invalid_url() {
        let err = Error::InvalidUrl("bad://url".into());
        assert!(matches!(err, Error::InvalidUrl(_)));
        assert!(err.to_string().contains("bad://url"));
    }

    #[test]
    fn ws_error_tls_not_enabled() {
        let err = Error::TlsNotEnabled;
        assert!(matches!(err, Error::TlsNotEnabled));
        assert!(err.to_string().contains("tls"));
    }
}

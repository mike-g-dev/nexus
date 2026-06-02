//! HTTP/1.1 keep-alive connection — pure transport.
//!
//! `Client<S>` is a thin I/O wrapper. It sends request bytes and
//! reads response bytes. All protocol logic (request encoding, response
//! parsing) lives in [`RequestWriter`](super::RequestWriter) and
//! [`ResponseReader`](crate::http::ResponseReader).

use std::time::Duration;

use super::error::RestError;
use super::request::RequestWriter;

use super::request::Request;
use super::response::RestResponse;
use crate::http::{HttpError, ResponseReader};
use std::io::{self, Read, Write};

#[cfg(feature = "tls")]
use nexus_net::tls::TlsConfig;

// =============================================================================
// URL parsing
// =============================================================================

/// Parsed HTTP URL.
#[non_exhaustive]
pub struct ParsedUrl<'a> {
    /// Whether the URL is `https://` (true) or `http://` (false).
    pub tls: bool,
    /// Host portion (no port).
    pub host: &'a str,
    /// Port — explicit if present, otherwise the scheme default
    /// (80 for http, 443 for https).
    pub port: u16,
    /// Path portion (everything after the host:port, including the
    /// leading `/`). Empty if the URL had no path.
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

/// Parse an `http://` or `https://` URL into its scheme, host, port,
/// and path. Returns [`RestError::InvalidUrl`] on a malformed input or
/// missing scheme.
pub fn parse_base_url(url: &str) -> Result<ParsedUrl<'_>, RestError> {
    let (tls, rest) = if let Some(r) = url.strip_prefix("https://") {
        (true, r)
    } else if let Some(r) = url.strip_prefix("http://") {
        (false, r)
    } else {
        return Err(RestError::InvalidUrl(url.to_string()));
    };

    // Split host:port from path
    let (host_port, path) = rest
        .find('/')
        .map_or((rest, ""), |i| (&rest[..i], &rest[i..]));

    if host_port.is_empty() {
        return Err(RestError::InvalidUrl(format!("empty host: {url}")));
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
                        .map_err(|_| RestError::InvalidUrl(format!("invalid port: {url}")))?;
                    (h, p)
                } else {
                    (h, default_port)
                }
            }
            None => return Err(RestError::InvalidUrl(format!("unclosed bracket: {url}"))),
        }
    } else {
        match host_port.rfind(':') {
            None => (host_port, default_port),
            Some(i) => {
                let port_str = &host_port[i + 1..];
                if port_str.is_empty() {
                    // Trailing colon with no port: "host:" → strip colon
                    (&host_port[..i], default_port)
                } else {
                    let p = port_str
                        .parse::<u16>()
                        .map_err(|_| RestError::InvalidUrl(format!("invalid port: {url}")))?;
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
// ClientBuilder
// =============================================================================

/// Builder for [`Client`].
///
/// Configures transport: TLS, timeouts, socket options.
/// Protocol configuration (host, headers, base path) lives on
/// [`RequestWriter`].
pub struct ClientBuilder {
    #[cfg(feature = "tls")]
    tls_config: Option<TlsConfig>,
    tcp_nodelay: bool,
    connect_timeout: Option<Duration>,
    read_timeout: Option<Duration>,
}

impl ClientBuilder {
    /// Create a new builder with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "tls")]
            tls_config: None,
            tcp_nodelay: false,
            connect_timeout: None,
            read_timeout: None,
        }
    }

    /// Set a custom TLS configuration.
    ///
    /// If not set, `https://` URLs use [`TlsConfig::new()`] (system defaults).
    #[cfg(feature = "tls")]
    #[must_use]
    pub fn tls(mut self, config: &TlsConfig) -> Self {
        self.tls_config = Some(config.clone());
        self
    }

    /// Set `TCP_NODELAY` (disable Nagle's algorithm).
    #[must_use]
    pub fn disable_nagle(mut self) -> Self {
        self.tcp_nodelay = true;
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

    /// Connect to an HTTP(S) endpoint (blocking).
    ///
    /// TLS is auto-detected from the URL scheme. When the `tls` feature is
    /// enabled, returns `Client<MaybeTls<TcpStream>>` — `https://` uses
    /// `MaybeTls::Tls`, `http://` uses `MaybeTls::Plain`. Without the `tls`
    /// feature, returns `Client<TcpStream>` and errors on `https://`.
    #[cfg(feature = "tls")]
    pub fn connect(
        self,
        url: &str,
    ) -> Result<Client<nexus_net::MaybeTls<std::net::TcpStream>>, RestError> {
        let parsed = parse_base_url(url)?;
        let addr = format!("{}:{}", parsed.host, parsed.port);

        let tcp = match self.connect_timeout {
            Some(timeout) => {
                let addrs: Vec<std::net::SocketAddr> =
                    std::net::ToSocketAddrs::to_socket_addrs(&addr)
                        .map_err(RestError::Io)?
                        .collect();
                let first = addrs
                    .first()
                    .ok_or_else(|| RestError::Io(io::Error::other("DNS resolution failed")))?;
                std::net::TcpStream::connect_timeout(first, timeout)?
            }
            None => std::net::TcpStream::connect(&addr)?,
        };

        if self.tcp_nodelay {
            tcp.set_nodelay(true)?;
        }
        if let Some(timeout) = self.read_timeout {
            tcp.set_read_timeout(Some(timeout))?;
        }

        let stream = if parsed.tls {
            let config = match self.tls_config {
                Some(c) => c,
                None => TlsConfig::new().map_err(RestError::Tls)?,
            };
            let codec = nexus_net::tls::TlsCodec::new(&config, parsed.host)?;
            let tls = nexus_net::tls::TlsStream::connect(tcp, codec).map_err(RestError::Tls)?;
            nexus_net::MaybeTls::Tls(Box::new(tls))
        } else {
            nexus_net::MaybeTls::Plain(tcp)
        };

        Ok(Client {
            stream,
            poisoned: false,
        })
    }

    /// Connect to an HTTP(S) endpoint (blocking, no TLS feature).
    #[cfg(not(feature = "tls"))]
    pub fn connect(self, url: &str) -> Result<Client<std::net::TcpStream>, RestError> {
        let parsed = parse_base_url(url)?;
        if parsed.tls {
            return Err(RestError::TlsNotEnabled);
        }
        let addr = format!("{}:{}", parsed.host, parsed.port);

        let tcp = match self.connect_timeout {
            Some(timeout) => {
                let addrs: Vec<std::net::SocketAddr> =
                    std::net::ToSocketAddrs::to_socket_addrs(&addr)
                        .map_err(RestError::Io)?
                        .collect();
                let first = addrs
                    .first()
                    .ok_or_else(|| RestError::Io(io::Error::other("DNS resolution failed")))?;
                std::net::TcpStream::connect_timeout(first, timeout)?
            }
            None => std::net::TcpStream::connect(&addr)?,
        };

        if self.tcp_nodelay {
            tcp.set_nodelay(true)?;
        }
        if let Some(timeout) = self.read_timeout {
            tcp.set_read_timeout(Some(timeout))?;
        }

        Ok(Client {
            stream: tcp,
            poisoned: false,
        })
    }

    /// Connect using a pre-connected socket.
    ///
    /// The stream must already handle TLS if connecting to `https://`.
    /// For example, pass a `TlsStream<TcpStream>` or `MaybeTls<TcpStream>`.
    pub fn connect_with<S: Read + Write>(
        self,
        stream: S,
        url: &str,
    ) -> Result<Client<S>, RestError> {
        // Validate the URL even though we don't use it for connection —
        // catches malformed URLs early rather than at first request.
        parse_base_url(url)?;
        Ok(Client::new(stream))
    }

    /// Create a `RequestWriter` configured for this URL.
    ///
    /// Convenience: extracts host and path from the URL to create
    /// a writer with the correct Host header and base path.
    pub fn writer_for(url: &str) -> Result<RequestWriter, RestError> {
        let parsed = parse_base_url(url)?;
        let host_header = parsed.host_header();
        let mut writer = RequestWriter::new(&host_header)?;
        if !parsed.path.is_empty() {
            writer.set_base_path(parsed.path)?;
        }
        Ok(writer)
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Client — pure transport
// =============================================================================

/// HTTP/1.1 keep-alive connection — pure transport.
///
/// Sends request bytes and reads response bytes. All protocol logic
/// lives in [`RequestWriter`] (request encoding) and
/// [`ResponseReader`](crate::http::ResponseReader) (response parsing).
///
/// # Usage
///
/// ```ignore
/// use nexus_web::rest::{Client, RequestWriter};
/// use nexus_web::http::ResponseReader;
/// use nexus_web::tls::TlsConfig;
///
/// // Protocol (sans-IO)
/// let mut writer = RequestWriter::new("api.binance.com").unwrap();
/// let mut reader = ResponseReader::new(32 * 1024);
///
/// // Transport
/// let tls = TlsConfig::new()?;
/// let mut conn = Client::builder().tls(&tls).connect("https://api.binance.com")?;
///
/// // Build + send
/// let req = writer.get("/orders").query("symbol", "BTC").finish()?;
/// let resp = conn.send(req, &mut reader)?;
/// ```
pub struct Client<S> {
    pub(crate) stream: S,
    pub(crate) poisoned: bool,
}

impl Client<std::net::TcpStream> {
    /// Create a transport builder.
    #[must_use]
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Set read timeout on the socket.
    ///
    /// **Strongly recommended for production.** Without a timeout, reads
    /// block indefinitely on stale connections.
    pub fn set_read_timeout(&self, timeout: Option<std::time::Duration>) -> Result<(), RestError> {
        self.stream.set_read_timeout(timeout).map_err(RestError::Io)
    }

    /// Set TCP keepalive on the underlying socket.
    ///
    /// Enables OS-level dead connection detection. The kernel sends
    /// probes after `idle` of inactivity.
    #[cfg(feature = "socket-opts")]
    pub fn set_tcp_keepalive(&self, idle: std::time::Duration) -> Result<(), RestError> {
        let sock = socket2::SockRef::from(&self.stream);
        let keepalive = socket2::TcpKeepalive::new().with_time(idle);
        sock.set_tcp_keepalive(&keepalive).map_err(RestError::Io)
    }
}

// -- Unbounded impl: accessors and constructors that need no I/O traits -------

impl<S> Client<S> {
    /// Wrap a pre-connected stream.
    pub fn new(stream: S) -> Self {
        Self {
            stream,
            poisoned: false,
        }
    }

    /// Whether the connection is poisoned (I/O error occurred).
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
}

// -- Blocking I/O impl --------------------------------------------------------

impl<S: Read + Write> Client<S> {
    /// Send a request and read the response.
    ///
    /// `req` provides the outbound bytes (from [`RequestWriter`]).
    /// `reader` receives and parses the response (body size limit
    /// configured on the reader via [`ResponseReader::max_body_size`]).
    ///
    /// Read timeout is a stream-level concern — configure via the builder
    /// (`read_timeout`) or `conn.stream().set_read_timeout()` for
    /// `TcpStream`. Without a timeout, reads block indefinitely.
    ///
    /// `Response` borrows from `reader` — drop before next send.
    #[allow(clippy::needless_pass_by_value)] // Move by design — request is consumed after send.
    pub fn send<'r>(
        &mut self,
        req: Request<'_>,
        reader: &'r mut ResponseReader,
    ) -> Result<RestResponse<'r>, RestError> {
        if self.poisoned {
            return Err(RestError::ConnectionPoisoned);
        }

        // Send request bytes
        if let Err(e) = self.write_all(req.as_bytes()) {
            self.poisoned = true;
            return Err(e);
        }

        // Read response
        match self.read_response(reader) {
            Ok(resp) => Ok(resp),
            Err(e) => self.handle_send_error(e),
        }
    }

    /// Cold path: diagnose send failure.
    #[cold]
    fn handle_send_error<T>(&mut self, err: RestError) -> Result<T, RestError> {
        self.poisoned = true;
        // On timeout, check if the socket is actually dead (stale connection)
        // vs the server just being slow.
        if let RestError::Io(ref io_err) = err
            && (io_err.kind() == std::io::ErrorKind::TimedOut
                || io_err.kind() == std::io::ErrorKind::WouldBlock)
        {
            if self.peek_is_dead() {
                return Err(RestError::ConnectionStale);
            }
            return Err(RestError::ReadTimeout);
        }
        Err(err)
    }

    /// Check if the socket has been closed by the peer.
    ///
    /// For generic streams we can't peek, so we assume alive and
    /// report `ReadTimeout`. The connection is poisoned either way.
    #[allow(clippy::unused_self)]
    fn peek_is_dead(&self) -> bool {
        // For generic S, assume alive (report ReadTimeout not ConnectionStale).
        // The caller still gets an error; it's just less specific.
        false
    }

    // =========================================================================
    // Internal — I/O with optional TLS
    // =========================================================================

    fn write_all(&mut self, data: &[u8]) -> Result<(), RestError> {
        self.stream.write_all(data)?;
        self.stream.flush()?;
        Ok(())
    }

    fn read_into_reader(&mut self, reader: &mut ResponseReader) -> Result<usize, RestError> {
        let n = reader.read_from(&mut self.stream)?;
        Ok(n)
    }

    fn read_response<'r>(
        &mut self,
        reader: &'r mut ResponseReader,
    ) -> Result<RestResponse<'r>, RestError> {
        // Consume previous response, preserving pipelined bytes.
        reader.consume_response();

        // Read until headers are complete.
        loop {
            match reader.next() {
                Ok(Some(_)) => break,
                Ok(None) => {}
                Err(e) => {
                    self.poisoned = true;
                    return Err(e.into());
                }
            }
            match self.read_into_reader(reader) {
                Ok(0) => {
                    self.poisoned = true;
                    return Err(RestError::ConnectionClosed(
                        "server closed before response headers",
                    ));
                }
                Ok(_) => {}
                Err(e) => {
                    self.poisoned = true;
                    return Err(e);
                }
            }
        }

        // Validate using cached values from try_parse.
        let status = reader.status();

        // RFC 7230: 1xx, 204, 304 have no body.
        if matches!(status, 100..=199 | 204 | 304) {
            reader.set_body_consumed(0);
            return Ok(RestResponse::new(status, 0, reader));
        }

        if reader.is_chunked() {
            let body = self.read_chunked_body(reader)?;
            // All remainder bytes were consumed (decoded or framing),
            // plus whatever was read from the socket during decode.
            // For consume_response, we need the total raw bytes in the
            // reader's buffer that belong to this response's body.
            // Since chunked body goes into a Vec (not the reader), the
            // remainder bytes are all raw chunked wire data that should
            // be skipped on consume.
            reader.set_body_consumed(reader.body_remaining());
            return Ok(RestResponse::new_chunked(status, body, reader));
        }

        let content_length = match reader.content_length() {
            Some(Ok(n)) => n,
            Some(Err(())) => {
                return Err(RestError::Http(HttpError::Malformed(
                    "invalid Content-Length header",
                )));
            }
            None => {
                // No Content-Length and not chunked — can't determine body
                // boundaries for keep-alive. Error instead of silent empty body.
                self.poisoned = true;
                return Err(RestError::Http(HttpError::Malformed(
                    "no Content-Length and not chunked",
                )));
            }
        };

        let max_body = reader.max_body_size_limit();
        if max_body > 0 && content_length > max_body {
            self.poisoned = true;
            return Err(RestError::BodyTooLarge {
                size: content_length,
                max: max_body,
            });
        }

        // Read remaining body bytes (Content-Length delimited).
        while reader.body_remaining() < content_length {
            match self.read_into_reader(reader) {
                Ok(0) => {
                    self.poisoned = true;
                    return Err(RestError::ConnectionClosed(
                        "server closed during body read",
                    ));
                }
                Ok(_) => {}
                Err(e) => {
                    self.poisoned = true;
                    return Err(e);
                }
            }
        }

        reader.set_body_consumed(content_length);
        Ok(RestResponse::new(status, content_length, reader))
    }

    /// Read a chunked transfer-encoded body. Returns decoded body bytes.
    ///
    /// One allocation: the Vec for the decoded body. The chunk framing
    /// is stripped and only payload bytes are accumulated.
    fn read_chunked_body(&mut self, reader: &ResponseReader) -> Result<Vec<u8>, RestError> {
        use crate::http::ChunkedDecoder;

        let max_body = reader.max_body_size_limit();
        let mut decoder = ChunkedDecoder::new();
        let mut body = Vec::with_capacity(4096);
        let mut wire_buf = [0u8; 4096];
        let mut decode_buf = [0u8; 4096];

        // Decode any chunk data that arrived with the headers.
        let remainder = reader.remainder();
        if !remainder.is_empty() {
            let mut pos = 0;
            while pos < remainder.len() && !decoder.is_done() {
                let (consumed, produced) = decoder
                    .decode(&remainder[pos..], &mut decode_buf)
                    .map_err(RestError::Http)?;
                pos += consumed;
                if produced > 0 {
                    body.extend_from_slice(&decode_buf[..produced]);
                    if max_body > 0 && body.len() > max_body {
                        self.poisoned = true;
                        return Err(RestError::BodyTooLarge {
                            size: body.len(),
                            max: max_body,
                        });
                    }
                }
                if consumed == 0 && produced == 0 {
                    break;
                }
            }
        }

        // Read from socket until all chunks decoded.
        while !decoder.is_done() {
            let n = self.read_wire_bytes(&mut wire_buf)?;
            if n == 0 {
                self.poisoned = true;
                return Err(RestError::ConnectionClosed(
                    "server closed during chunked body",
                ));
            }

            let mut pos = 0;
            while pos < n && !decoder.is_done() {
                let (consumed, produced) = decoder
                    .decode(&wire_buf[pos..n], &mut decode_buf)
                    .map_err(RestError::Http)?;
                pos += consumed;
                if produced > 0 {
                    body.extend_from_slice(&decode_buf[..produced]);
                    // Check body size limit after each decode, not per read.
                    if max_body > 0 && body.len() > max_body {
                        self.poisoned = true;
                        return Err(RestError::BodyTooLarge {
                            size: body.len(),
                            max: max_body,
                        });
                    }
                }
                if consumed == 0 && produced == 0 {
                    break;
                }
            }
        }

        Ok(body)
    }

    /// Read raw bytes from the socket.
    fn read_wire_bytes(&mut self, buf: &mut [u8]) -> Result<usize, RestError> {
        Ok(self.stream.read(buf)?)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read, Write};
    use std::net::{TcpListener, TcpStream};

    struct MockStream {
        written: Vec<u8>,
        response: Cursor<Vec<u8>>,
    }

    impl MockStream {
        fn new(response: &[u8]) -> Self {
            Self {
                written: Vec::new(),
                response: Cursor::new(response.to_vec()),
            }
        }

        fn written_str(&self) -> &str {
            std::str::from_utf8(&self.written).unwrap()
        }
    }

    impl Read for MockStream {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.response.read(buf)
        }
    }

    impl Write for MockStream {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.written.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn ok_response(body: &str) -> Vec<u8> {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        )
        .into_bytes()
    }

    /// Helper: build request + send via mock.
    #[allow(dead_code)]
    fn send_get<'r>(
        writer: &mut RequestWriter,
        conn: &mut Client<MockStream>,
        reader: &'r mut ResponseReader,
        path: &str,
    ) -> Result<RestResponse<'r>, RestError> {
        let req = writer.get(path).finish()?;
        conn.send(req, reader)
    }

    // --- Request format ---

    #[test]
    fn get_request_format() {
        let resp = ok_response(r#"{"ok":true}"#);
        let mock = MockStream::new(&resp);
        let mut writer = RequestWriter::new("api.example.com").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(mock);

        let req = writer.get("/api/v1/status").finish().unwrap();
        let resp = conn.send(req, &mut reader).unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.body_str().unwrap(), r#"{"ok":true}"#);

        let written = conn.stream().written_str();
        assert!(written.starts_with("GET /api/v1/status HTTP/1.1\r\n"));
        assert!(written.contains("Host: api.example.com\r\n"));
        assert!(written.contains("Connection: keep-alive\r\n"));
        assert!(written.ends_with("\r\n\r\n"));
    }

    #[test]
    fn post_with_body() {
        let resp = ok_response(r#"{"filled":true}"#);
        let mock = MockStream::new(&resp);
        let mut writer = RequestWriter::new("api.example.com").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(mock);

        let body = br#"{"symbol":"BTC","side":"buy"}"#;
        let req = writer.post("/api/v3/order").body(body).finish().unwrap();
        let resp = conn.send(req, &mut reader).unwrap();
        assert_eq!(resp.status(), 200);

        let written = conn.stream().written_str();
        assert!(written.starts_with("POST /api/v3/order HTTP/1.1\r\n"));
        assert!(written.contains(&format!("Content-Length: {}\r\n", body.len())));
        assert!(written.ends_with(std::str::from_utf8(body).unwrap()));
    }

    #[test]
    fn post_body_writer() {
        let resp = ok_response(r#"{"ok":true}"#);
        let mock = MockStream::new(&resp);
        let mut writer = RequestWriter::new("host").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(mock);

        let body = br#"{"symbol":"BTC","side":"buy"}"#;
        let req = writer
            .post("/order")
            .body_writer(|w| {
                use std::io::Write;
                w.write_all(body)
            })
            .finish()
            .unwrap();

        let written_before = std::str::from_utf8(req.as_bytes()).unwrap().to_string();
        // Verify Content-Length is backfilled correctly (exact digits)
        assert!(written_before.contains("Content-Length:"));
        assert!(written_before.contains(&format!("{}", body.len())));
        assert!(written_before.ends_with(std::str::from_utf8(body).unwrap()));

        let resp = conn.send(req, &mut reader).unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[test]
    fn body_writer_from_headers_phase() {
        let mut writer = RequestWriter::new("host").unwrap();
        let body = b"test-body";
        let req = writer
            .post("/order")
            .header("X-Custom", "val")
            .body_writer(|w| {
                use std::io::Write;
                w.write_all(body)
            })
            .finish()
            .unwrap();

        let data = std::str::from_utf8(req.as_bytes()).unwrap();
        assert!(data.contains("X-Custom: val\r\n"));
        assert!(data.contains(&format!("{}", body.len())));
        assert!(data.ends_with("test-body"));
    }

    #[test]
    fn body_writer_empty() {
        let mut writer = RequestWriter::new("host").unwrap();
        let req = writer
            .post("/order")
            .body_writer(|_w| Ok::<(), std::io::Error>(()))
            .finish()
            .unwrap();

        let data = std::str::from_utf8(req.as_bytes()).unwrap();
        // Content-Length should be 0
        assert!(data.contains("Content-Length:"));
        assert!(data.contains("0\r\n\r\n"));
    }

    #[test]
    fn body_writer_matches_body() {
        // Verify body_writer produces identical wire bytes to body()
        let mut writer1 = RequestWriter::new("host").unwrap();
        let mut writer2 = RequestWriter::new("host").unwrap();

        let body = b"identical-content";

        let req1 = writer1.post("/test").body(body).finish().unwrap();
        let req2 = writer2
            .post("/test")
            .body_writer(|w| {
                use std::io::Write;
                w.write_all(body)
            })
            .finish()
            .unwrap();

        // Both paths produce identical wire format.
        let d1 = std::str::from_utf8(req1.as_bytes()).unwrap();
        let d2 = std::str::from_utf8(req2.as_bytes()).unwrap();
        assert_eq!(d1, d2);
    }

    #[test]
    fn all_methods() {
        for (method, expected) in [
            (super::super::request::Method::Put, "PUT"),
            (super::super::request::Method::Delete, "DELETE"),
            (super::super::request::Method::Patch, "PATCH"),
        ] {
            let resp = ok_response("{}");
            let mock = MockStream::new(&resp);
            let mut writer = RequestWriter::new("host").unwrap();
            let mut reader = ResponseReader::new(4096);
            let mut conn = Client::new(mock);

            let req = writer.request(method, "/test").finish().unwrap();
            let _ = conn.send(req, &mut reader).unwrap();
            assert!(
                conn.stream()
                    .written_str()
                    .starts_with(&format!("{expected} /test HTTP/1.1\r\n"))
            );
        }
    }

    #[test]
    fn default_headers_included() {
        let resp = ok_response("{}");
        let mock = MockStream::new(&resp);
        let mut writer = RequestWriter::new("api.example.com").unwrap();
        writer.default_header("X-API-KEY", "secret123").unwrap();
        writer
            .default_header("Content-Type", "application/json")
            .unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(mock);

        let req = writer.get("/test").finish().unwrap();
        let _ = conn.send(req, &mut reader).unwrap();

        let written = conn.stream().written_str();
        assert!(written.contains("X-API-KEY: secret123\r\n"));
        assert!(written.contains("Content-Type: application/json\r\n"));
    }

    #[test]
    fn extra_headers() {
        let resp = ok_response("{}");
        let mock = MockStream::new(&resp);
        let mut writer = RequestWriter::new("api.example.com").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(mock);

        let req = writer
            .get("/test")
            .header("X-Custom", "value1")
            .header("Authorization", "Bearer tok")
            .finish()
            .unwrap();
        let _ = conn.send(req, &mut reader).unwrap();

        let written = conn.stream().written_str();
        assert!(written.contains("X-Custom: value1\r\n"));
        assert!(written.contains("Authorization: Bearer tok\r\n"));
    }

    // --- Query parameters ---

    #[test]
    fn query_params_encoded() {
        let mut writer = RequestWriter::new("host").unwrap();
        let req = writer
            .get("/orders")
            .query("symbol", "BTC-USD")
            .query("limit", "100")
            .finish()
            .unwrap();
        let data = std::str::from_utf8(req.as_bytes()).unwrap();
        assert!(data.starts_with("GET /orders?symbol=BTC-USD&limit=100 HTTP/1.1\r\n"));
    }

    #[test]
    fn query_encodes_special_chars() {
        let mut writer = RequestWriter::new("host").unwrap();
        let req = writer
            .get("/search")
            .query("q", "hello world&more=yes")
            .finish()
            .unwrap();
        let data = std::str::from_utf8(req.as_bytes()).unwrap();
        assert!(data.starts_with("GET /search?q=hello%20world%26more%3Dyes HTTP/1.1\r\n"));
    }

    #[test]
    fn query_raw_no_encoding() {
        let mut writer = RequestWriter::new("host").unwrap();
        let req = writer
            .get("/orders")
            .query_raw("symbol", "BTC-USD")
            .finish()
            .unwrap();
        let data = std::str::from_utf8(req.as_bytes()).unwrap();
        assert!(data.starts_with("GET /orders?symbol=BTC-USD HTTP/1.1\r\n"));
    }

    #[test]
    fn query_then_header() {
        let mut writer = RequestWriter::new("host").unwrap();
        let req = writer
            .get("/orders")
            .query("sym", "ETH")
            .header("X-Nonce", "123")
            .finish()
            .unwrap();
        let data = std::str::from_utf8(req.as_bytes()).unwrap();
        assert!(data.starts_with("GET /orders?sym=ETH HTTP/1.1\r\n"));
        assert!(data.contains("X-Nonce: 123\r\n"));
    }

    #[test]
    fn path_with_existing_query() {
        let mut writer = RequestWriter::new("host").unwrap();
        let req = writer
            .get("/path?existing=true")
            .query("extra", "val")
            .finish()
            .unwrap();
        let data = std::str::from_utf8(req.as_bytes()).unwrap();
        assert!(data.starts_with("GET /path?existing=true&extra=val HTTP/1.1\r\n"));
    }

    #[test]
    fn base_path_prepended() {
        let mut writer = RequestWriter::new("host").unwrap();
        writer.set_base_path("/api/v3").unwrap();
        let req = writer.get("/orders").finish().unwrap();
        let data = std::str::from_utf8(req.as_bytes()).unwrap();
        assert!(data.starts_with("GET /api/v3/orders HTTP/1.1\r\n"));
    }

    #[test]
    fn get_raw_skips_query_phase() {
        let mut writer = RequestWriter::new("host").unwrap();
        let req = writer
            .get_raw("/orders?symbol=BTC&limit=100")
            .finish()
            .unwrap();
        let data = std::str::from_utf8(req.as_bytes()).unwrap();
        assert!(data.starts_with("GET /orders?symbol=BTC&limit=100 HTTP/1.1\r\n"));
    }

    // --- Validation ---

    #[test]
    fn crlf_in_header_rejected() {
        let mut writer = RequestWriter::new("host").unwrap();
        let result = writer.get("/test").header("X-Bad\r\n", "val").finish();
        assert!(matches!(result, Err(RestError::CrlfInjection)));
    }

    #[test]
    fn crlf_in_path_rejected() {
        let mut writer = RequestWriter::new("host").unwrap();
        let result = writer.get("/path\r\nEvil: yes").finish();
        assert!(matches!(result, Err(RestError::CrlfInjection)));
    }

    #[test]
    fn crlf_in_default_header_rejected() {
        let mut writer = RequestWriter::new("host").unwrap();
        assert!(matches!(
            writer.default_header("X-Bad\n", "val"),
            Err(RestError::CrlfInjection)
        ));
    }

    #[test]
    fn crlf_in_query_raw_rejected() {
        let mut writer = RequestWriter::new("host").unwrap();
        let result = writer.get("/test").query_raw("k", "v\r\n").finish();
        assert!(matches!(result, Err(RestError::CrlfInjection)));
    }

    #[test]
    fn crlf_in_host_rejected() {
        assert!(matches!(
            RequestWriter::new("evil.com\r\nX-Injected: yes"),
            Err(RestError::CrlfInjection)
        ));
    }

    // --- Response handling ---

    #[test]
    fn response_headers_accessible() {
        let resp_bytes = b"HTTP/1.1 200 OK\r\nX-Request-Id: abc123\r\nX-RateLimit-Remaining: 42\r\nContent-Length: 2\r\n\r\n{}";
        let mock = MockStream::new(resp_bytes);
        let mut writer = RequestWriter::new("host").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(mock);

        let req = writer.get("/test").finish().unwrap();
        let resp = conn.send(req, &mut reader).unwrap();
        assert_eq!(resp.header("X-Request-Id"), Some("abc123"));
        assert_eq!(resp.header("X-RateLimit-Remaining"), Some("42"));
    }

    #[test]
    fn chunked_encoding_decoded() {
        let resp_bytes = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n7\r\nMozilla\r\n11\r\nDeveloper Network\r\n0\r\n\r\n";
        let mock = MockStream::new(resp_bytes);
        let mut writer = RequestWriter::new("host").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(mock);

        let req = writer.get("/test").finish().unwrap();
        let resp = conn.send(req, &mut reader).unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.body_str().unwrap(), "MozillaDeveloper Network");
    }

    #[test]
    fn chunked_empty_body() {
        let resp_bytes = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n";
        let mock = MockStream::new(resp_bytes);
        let mut writer = RequestWriter::new("host").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(mock);

        let req = writer.get("/test").finish().unwrap();
        let resp = conn.send(req, &mut reader).unwrap();
        assert_eq!(resp.body().len(), 0);
    }

    #[test]
    fn chunked_json_response() {
        // Simulates a CDN/proxy chunking a JSON response
        let body = r#"{"orderId":12345,"status":"FILLED"}"#;
        let chunked = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
            body.len(),
            body
        );
        let mock = MockStream::new(chunked.as_bytes());
        let mut writer = RequestWriter::new("host").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(mock);

        let req = writer.get("/test").finish().unwrap();
        let resp = conn.send(req, &mut reader).unwrap();
        assert_eq!(resp.body_str().unwrap(), body);
    }

    #[test]
    fn malformed_content_length_rejected() {
        let resp_bytes = b"HTTP/1.1 200 OK\r\nContent-Length: abc\r\n\r\nbody";
        let mock = MockStream::new(resp_bytes);
        let mut writer = RequestWriter::new("host").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(mock);

        let req = writer.get("/test").finish().unwrap();
        let result = conn.send(req, &mut reader);
        assert!(matches!(result, Err(RestError::Http(_))));
    }

    #[test]
    fn body_too_large_rejected() {
        let resp_bytes = b"HTTP/1.1 200 OK\r\nContent-Length: 999999\r\n\r\n";
        let mock = MockStream::new(resp_bytes);
        let mut writer = RequestWriter::new("host").unwrap();
        let mut reader = ResponseReader::new(4096).max_body_size(32 * 1024);
        let mut conn = Client::new(mock);

        let req = writer.get("/test").finish().unwrap();
        let result = conn.send(req, &mut reader);
        assert!(matches!(
            result,
            Err(RestError::BodyTooLarge { size: 999_999, .. })
        ));
    }

    #[test]
    fn status_204_no_body() {
        let resp_bytes = b"HTTP/1.1 204 No Content\r\nContent-Length: 5\r\n\r\nxxxxx";
        let mock = MockStream::new(resp_bytes);
        let mut writer = RequestWriter::new("host").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(mock);

        let req = writer.get("/test").finish().unwrap();
        let resp = conn.send(req, &mut reader).unwrap();
        assert_eq!(resp.status(), 204);
        assert_eq!(resp.body().len(), 0);
    }

    #[test]
    fn connection_poisoned_after_io_error() {
        let resp_bytes = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial";
        let mock = MockStream::new(resp_bytes);
        let mut writer = RequestWriter::new("host").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(mock);

        let req = writer.get("/test").finish().unwrap();
        let result = conn.send(req, &mut reader);
        assert!(matches!(result, Err(RestError::ConnectionClosed(_))));

        let req = writer.get("/test2").finish().unwrap();
        let result = conn.send(req, &mut reader);
        assert!(matches!(result, Err(RestError::ConnectionPoisoned)));
    }

    // --- URL parsing ---

    #[test]
    fn url_parsing() {
        let parsed = parse_base_url("https://api.binance.com").unwrap();
        assert!(parsed.tls);
        assert_eq!(parsed.host, "api.binance.com");
        assert_eq!(parsed.port, 443);
        assert_eq!(parsed.path, "");

        let parsed = parse_base_url("http://localhost:8080").unwrap();
        assert!(!parsed.tls);
        assert_eq!(parsed.host, "localhost");
        assert_eq!(parsed.port, 8080);

        let parsed = parse_base_url("https://api.example.com/v1/foo").unwrap();
        assert_eq!(parsed.path, "/v1/foo");

        assert!(parse_base_url("ftp://host").is_err());
        assert!(parse_base_url("http://").is_err());
    }

    #[test]
    fn ipv6_url_parsing() {
        let parsed = parse_base_url("http://[::1]:8080").unwrap();
        assert_eq!(parsed.host, "::1");
        assert_eq!(parsed.port, 8080);

        let parsed = parse_base_url("http://[::1]").unwrap();
        assert_eq!(parsed.host, "::1");
        assert_eq!(parsed.port, 80);

        assert!(parse_base_url("http://[::1").is_err());
    }

    // --- Keep-alive ---

    #[test]
    fn keep_alive_sequential_requests() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let server = std::thread::spawn(move || {
            let (mut tcp, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];

            let n = tcp.read(&mut buf).unwrap();
            assert!(
                std::str::from_utf8(&buf[..n])
                    .unwrap()
                    .contains("GET /first")
            );
            let body1 = r#"{"id":1}"#;
            let resp1 = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                body1.len(),
                body1
            );
            tcp.write_all(resp1.as_bytes()).unwrap();

            let n = tcp.read(&mut buf).unwrap();
            assert!(
                std::str::from_utf8(&buf[..n])
                    .unwrap()
                    .contains("GET /second")
            );
            let body2 = r#"{"id":2}"#;
            let resp2 = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                body2.len(),
                body2
            );
            tcp.write_all(resp2.as_bytes()).unwrap();
        });

        let tcp = TcpStream::connect(addr).unwrap();
        let mut writer = RequestWriter::new("localhost").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(tcp);

        let req = writer.get("/first").finish().unwrap();
        let resp = conn.send(req, &mut reader).unwrap();
        assert_eq!(resp.body_str().unwrap(), r#"{"id":1}"#);
        drop(resp);

        let req = writer.get("/second").finish().unwrap();
        let resp = conn.send(req, &mut reader).unwrap();
        assert_eq!(resp.body_str().unwrap(), r#"{"id":2}"#);

        server.join().unwrap();
    }

    // --- Display ---

    #[test]
    fn method_display() {
        use super::super::request::Method;
        assert_eq!(format!("{}", Method::Get), "GET");
        assert_eq!(format!("{}", Method::Post), "POST");
        assert_eq!(format!("{}", Method::Delete), "DELETE");
    }
}

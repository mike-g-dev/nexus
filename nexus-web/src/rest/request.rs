//! Sans-IO HTTP request encoder with typestate builder.
//!
//! `RequestWriter` is the protocol-level request encoder. It owns a
//! `WriteBuf` and endpoint configuration (host, default headers, base path).
//! The typestate builder writes directly into the buffer. `finish()`
//! returns a `Request<'_>` borrowing the assembled wire bytes.
//!
//! No I/O, no sockets, no async — pure state machine.
//!
//! ```text
//! Query   → query()   → Query
//! Query   → header()  → Headers   (seals request line)
//! Query   → body()    → Ready     (seals + writes body)
//! Query   → finish()  → Request<'_>
//!
//! Headers → header()  → Headers
//! Headers → body()    → Ready
//! Headers → finish()  → Request<'_>
//!
//! Ready   → finish()  → Request<'_>
//! ```

use std::marker::PhantomData;

use super::error::RestError;
use nexus_net::buf::WriteBuf;

// ---------------------------------------------------------------------------
// Phase markers
// ---------------------------------------------------------------------------

/// Request is in the query-parameter phase.
pub struct Query;
/// Request is in the headers phase.
pub struct Headers;
/// Request is fully assembled, ready to send.
pub struct Ready;

mod sealed {
    pub trait Phase {}
    impl Phase for super::Query {}
    impl Phase for super::Headers {}
    impl Phase for super::Ready {}
}

// ---------------------------------------------------------------------------
// Method
// ---------------------------------------------------------------------------

/// HTTP method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// `GET` — retrieve a resource.
    Get,
    /// `POST` — submit data.
    Post,
    /// `PUT` — replace a resource.
    Put,
    /// `DELETE` — remove a resource.
    Delete,
    /// `PATCH` — partial update.
    Patch,
}

impl Method {
    /// Wire representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Patch => "PATCH",
        }
    }
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Request (the output — borrows WriteBuf data)
// ---------------------------------------------------------------------------

/// A built HTTP request. Borrows from the `RequestWriter`'s buffer.
///
/// Must be consumed or dropped before the next request on the same writer.
/// Same lifecycle as [`Message<'_>`](crate::ws::Message) in WebSocket.
///
/// `Clone` is cheap (copies a pointer + length, no allocation). Use it
/// to archive request bytes before sending:
///
/// ```ignore
/// let req = writer.post("/order").body(json).finish()?;
/// let archived = req.clone();
/// conn.send(req, &mut reader)?;
/// archive_log.write(archived.as_bytes());
/// ```
#[derive(Clone)]
pub struct Request<'a> {
    data: &'a [u8],
}

impl<'a> Request<'a> {
    /// The complete HTTP request as wire bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }

    /// Consume the request, returning the raw wire bytes.
    ///
    /// Releases the borrow on the `RequestWriter` while keeping
    /// access to the bytes (they remain valid until the writer
    /// is used again).
    ///
    /// ```ignore
    /// let payload = writer.post("/order").body(json).finish()?.into_bytes();
    /// archive_log.write(payload);
    /// ```
    pub fn into_bytes(self) -> &'a [u8] {
        self.data
    }

    /// Request size in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the request is empty (should never be true after `finish()`).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl std::fmt::Debug for Request<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Request")
            .field("len", &self.data.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Percent-encoding (RFC 3986)
// ---------------------------------------------------------------------------

/// Unreserved characters: A-Z a-z 0-9 - . _ ~
const UNRESERVED: [bool; 256] = {
    let mut table = [false; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = matches!(
            i as u8,
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~'
        );
        i += 1;
    }
    table
};

const HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";

/// Percent-encode `input` per RFC 3986 directly into the WriteBuf.
/// Percent-encode `input` per RFC 3986 directly into the WriteBuf.
/// Batch-scans for runs of unreserved bytes to minimize checked_append calls.
fn append_percent_encoded(buf: &mut WriteBuf, input: &[u8], error: &mut Option<RestError>) {
    if error.is_some() {
        return;
    }
    let mut i = 0;
    while i < input.len() {
        // Scan for a run of unreserved bytes.
        let run_start = i;
        while i < input.len() && UNRESERVED[input[i] as usize] {
            i += 1;
        }
        // Append the unreserved run in one bulk copy.
        if i > run_start {
            checked_append(buf, &input[run_start..i], error);
            if error.is_some() {
                return;
            }
        }
        // Encode the next reserved byte (if any).
        if i < input.len() {
            let b = input[i];
            checked_append(
                buf,
                &[
                    b'%',
                    HEX_UPPER[(b >> 4) as usize],
                    HEX_UPPER[(b & 0xf) as usize],
                ],
                error,
            );
            if error.is_some() {
                return;
            }
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Append to WriteBuf with deferred overflow error.
fn checked_append(buf: &mut WriteBuf, src: &[u8], error: &mut Option<RestError>) {
    if error.is_some() {
        return;
    }
    if src.len() > buf.tailroom() {
        *error = Some(RestError::RequestTooLarge {
            capacity: buf.len() + buf.tailroom(),
        });
        return;
    }
    buf.append(src);
}

/// Check for CR/LF bytes.
fn has_crlf(s: &str) -> bool {
    s.bytes().any(|b| b == b'\r' || b == b'\n')
}

/// Write a usize as ASCII digits without allocation.
fn write_usize_ascii(buf: &mut WriteBuf, n: usize, error: &mut Option<RestError>) {
    if n == 0 {
        checked_append(buf, b"0", error);
        return;
    }
    let mut digits = [0u8; 20]; // max digits for usize
    let mut i = 20;
    let mut val = n;
    while val > 0 {
        i -= 1;
        digits[i] = (val % 10) as u8 + b'0';
        val /= 10;
    }
    checked_append(buf, &digits[i..], error);
}

/// Write " HTTP/1.1\r\n" + host_wire + default_headers_wire.
fn seal_request_line(writer: &mut RequestWriter, error: &mut Option<RestError>) {
    checked_append(&mut writer.write_buf, b" HTTP/1.1\r\n", error);
    // Split borrow: write_buf (mut) and host_wire/default_headers_wire (shared).
    checked_append(&mut writer.write_buf, &writer.host_wire, error);
    if !writer.default_headers_wire.is_empty() {
        checked_append(&mut writer.write_buf, &writer.default_headers_wire, error);
    }
}

/// Write Content-Length + \r\n separator + body bytes.
fn write_body(buf: &mut WriteBuf, body: &[u8], error: &mut Option<RestError>) {
    checked_append(buf, b"Content-Length: ", error);
    write_usize_ascii(buf, body.len(), error);
    checked_append(buf, b"\r\n\r\n", error);
    checked_append(buf, body, error);
}

/// Write Content-Length header + \r\n separator (no body).
fn write_body_header(buf: &mut WriteBuf, body_len: usize, error: &mut Option<RestError>) {
    checked_append(buf, b"Content-Length: ", error);
    write_usize_ascii(buf, body_len, error);
    checked_append(buf, b"\r\n\r\n", error);
}

/// Content-Length placeholder: "Content-Length: XXXXXXXXXX\r\n\r\n"
/// 20 digits supports all possible `usize` values on 64-bit.
const CL_PREFIX: &[u8] = b"Content-Length: ";
const CL_PAD_LEN: usize = 20;
const CL_SUFFIX: &[u8] = b"\r\n\r\n";
/// Write a padded Content-Length placeholder. Returns the offset
/// within the WriteBuf where the 10-digit number starts.
fn write_content_length_placeholder(buf: &mut WriteBuf, error: &mut Option<RestError>) -> usize {
    checked_append(buf, CL_PREFIX, error);
    let num_offset = buf.len();
    checked_append(buf, b"00000000000000000000", error); // 20 zeros
    checked_append(buf, CL_SUFFIX, error);
    num_offset
}

/// Write adapter for direct serialization into the request buffer.
///
/// Type alias for [`WriteBufWriter`](nexus_net::buf::WriteBufWriter).
/// Implements `std::io::Write`.
pub type BodyWriter<'a> = nexus_net::buf::WriteBufWriter<'a>;

/// Backfill the Content-Length placeholder with the actual body length.
/// Writes exact digits, shifts body left to close the gap.
///
/// Produces clean wire format: `Content-Length: 42\r\n\r\n` (no extra spaces).
fn backfill_content_length(buf: &mut WriteBuf, num_offset: usize, body_len: usize) {
    // Format the actual digits on the stack.
    let mut digits = [0u8; 20];
    let digit_len = if body_len == 0 {
        digits[0] = b'0';
        1
    } else {
        let mut val = body_len;
        let mut i = 20;
        while val > 0 {
            i -= 1;
            digits[i] = (val % 10) as u8 + b'0';
            val /= 10;
        }
        let len = 20 - i;
        digits.copy_within(i..20, 0);
        len
    };

    let gap = CL_PAD_LEN - digit_len;

    {
        let data = buf.data_mut();

        // Overwrite placeholder with actual digits.
        data[num_offset..num_offset + digit_len].copy_from_slice(&digits[..digit_len]);

        // Write \r\n\r\n right after the digits.
        let suffix_dst = num_offset + digit_len;
        data[suffix_dst..suffix_dst + CL_SUFFIX.len()].copy_from_slice(CL_SUFFIX);

        // Shift body bytes left to close the gap.
        if gap > 0 {
            let body_start = num_offset + CL_PAD_LEN + CL_SUFFIX.len();
            let body_end = data.len();
            if body_start < body_end {
                data.copy_within(body_start..body_end, body_start - gap);
            }
        }
    }

    // Shrink the buffer to remove the gap bytes.
    if gap > 0 {
        buf.shrink_tail(gap);
    }
}

// ---------------------------------------------------------------------------
// RequestWriter
// ---------------------------------------------------------------------------

/// Sans-IO HTTP request encoder.
///
/// Owns a `WriteBuf` and endpoint configuration. The typestate builder
/// methods serialize HTTP requests directly into the buffer. `finish()`
/// returns `Request<'_>` borrowing the assembled bytes.
///
/// # Usage
///
/// ```ignore
/// use nexus_web::rest::RequestWriter;
///
/// let mut writer = RequestWriter::new("api.binance.com")?;
/// writer.set_base_path("/api/v3")?;
/// writer.default_header("X-API-KEY", &key)?;
///
/// let req = writer.get("/orders")
///     .query("symbol", "BTC-USD")
///     .finish()?;
///
/// // req.as_bytes() contains the complete HTTP request wire bytes
/// ```
pub struct RequestWriter {
    write_buf: WriteBuf,
    /// Pre-serialized: "Host: hostname\r\nConnection: keep-alive\r\n"
    host_wire: Vec<u8>,
    /// Pre-serialized default headers: "Name: Value\r\n..."
    default_headers_wire: Vec<u8>,
    /// Base path prefix prepended to all request paths.
    base_path: Vec<u8>,
}

impl RequestWriter {
    /// Create a new writer for the given host.
    ///
    /// Pre-serializes the Host and Connection: keep-alive headers.
    /// Default write buffer: 32KB.
    ///
    /// # Errors
    ///
    /// Returns [`RestError::CrlfInjection`] if `host` contains CR/LF.
    pub fn new(host: &str) -> Result<Self, RestError> {
        if host.bytes().any(|b| b == b'\r' || b == b'\n') {
            return Err(RestError::CrlfInjection);
        }

        let mut host_wire = Vec::with_capacity(host.len() + 32);
        host_wire.extend_from_slice(b"Host: ");
        host_wire.extend_from_slice(host.as_bytes());
        host_wire.extend_from_slice(b"\r\nConnection: keep-alive\r\n");

        Ok(Self {
            write_buf: WriteBuf::new(32 * 1024, 0),
            host_wire,
            default_headers_wire: Vec::new(),
            base_path: Vec::new(),
        })
    }

    /// Set the write buffer capacity. Default: 32KB.
    ///
    /// Must be called before any requests are built.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0.
    pub fn set_write_buffer_capacity(&mut self, capacity: usize) {
        self.write_buf = WriteBuf::new(capacity, 0);
    }

    /// Add a default header sent with every request.
    ///
    /// Pre-serializes into wire format. Append-only.
    ///
    /// # Errors
    ///
    /// Returns [`RestError::CrlfInjection`] if name or value contains CR/LF.
    pub fn default_header(&mut self, name: &str, value: &str) -> Result<(), RestError> {
        if has_crlf(name) || has_crlf(value) {
            return Err(RestError::CrlfInjection);
        }
        self.default_headers_wire.extend_from_slice(name.as_bytes());
        self.default_headers_wire.extend_from_slice(b": ");
        self.default_headers_wire
            .extend_from_slice(value.as_bytes());
        self.default_headers_wire.extend_from_slice(b"\r\n");
        Ok(())
    }

    /// Set a base path prefix prepended to all request paths.
    ///
    /// Trailing slashes are stripped. Request paths should start with `/`.
    ///
    /// # Errors
    ///
    /// Returns [`RestError::CrlfInjection`] if the path contains CR/LF.
    pub fn set_base_path(&mut self, path: &str) -> Result<(), RestError> {
        if has_crlf(path) {
            return Err(RestError::CrlfInjection);
        }
        self.base_path = path.trim_end_matches('/').as_bytes().to_vec();
        Ok(())
    }

    // =========================================================================
    // Request builders — Query phase
    // =========================================================================

    /// Build a GET request.
    pub fn get(&mut self, path: &str) -> RequestBuilder<'_> {
        self.request(Method::Get, path)
    }

    /// Build a POST request.
    pub fn post(&mut self, path: &str) -> RequestBuilder<'_> {
        self.request(Method::Post, path)
    }

    /// Build a PUT request.
    pub fn put(&mut self, path: &str) -> RequestBuilder<'_> {
        self.request(Method::Put, path)
    }

    /// Build a DELETE request.
    pub fn delete(&mut self, path: &str) -> RequestBuilder<'_> {
        self.request(Method::Delete, path)
    }

    /// Build a request with the given method.
    pub fn request(&mut self, method: Method, path: &str) -> RequestBuilder<'_> {
        RequestBuilder::new(self, method, path)
    }

    // =========================================================================
    // Request builders — Headers phase (pre-formed URL)
    // =========================================================================

    /// Build a GET with a pre-formed URL path (including any query string).
    ///
    /// Skips the [`Query`] phase — returns [`Headers`] directly.
    pub fn get_raw(&mut self, path: &str) -> RequestBuilder<'_, Headers> {
        self.request_raw(Method::Get, path)
    }

    /// Build a POST with a pre-formed URL path.
    pub fn post_raw(&mut self, path: &str) -> RequestBuilder<'_, Headers> {
        self.request_raw(Method::Post, path)
    }

    /// Build a PUT with a pre-formed URL path.
    pub fn put_raw(&mut self, path: &str) -> RequestBuilder<'_, Headers> {
        self.request_raw(Method::Put, path)
    }

    /// Build a DELETE with a pre-formed URL path.
    pub fn delete_raw(&mut self, path: &str) -> RequestBuilder<'_, Headers> {
        self.request_raw(Method::Delete, path)
    }

    /// Build a request with a pre-formed URL path.
    pub fn request_raw(&mut self, method: Method, path: &str) -> RequestBuilder<'_, Headers> {
        RequestBuilder::new_sealed(self, method, path)
    }
}

// ---------------------------------------------------------------------------
// RequestBuilder
// ---------------------------------------------------------------------------

/// Typestate request builder. Writes directly into a `RequestWriter`'s
/// buffer — no intermediate storage, no stream type parameter.
///
/// Phase type parameter enforces correct wire ordering at compile time:
/// query parameters before headers, headers before body.
#[must_use = "request must be finished with .finish()"]
pub struct RequestBuilder<'a, P: sealed::Phase = Query> {
    writer: &'a mut RequestWriter,
    has_query: bool,
    error: Option<RestError>,
    _phase: PhantomData<P>,
}

// =========================================================================
// Query phase
// =========================================================================

impl<'a> RequestBuilder<'a, Query> {
    pub(crate) fn new(writer: &'a mut RequestWriter, method: Method, path: &str) -> Self {
        writer.write_buf.clear();
        let mut error = if has_crlf(path) {
            Some(RestError::CrlfInjection)
        } else {
            None
        };
        checked_append(
            &mut writer.write_buf,
            method.as_str().as_bytes(),
            &mut error,
        );
        checked_append(&mut writer.write_buf, b" ", &mut error);
        if !writer.base_path.is_empty() {
            checked_append(&mut writer.write_buf, &writer.base_path, &mut error);
        }
        checked_append(&mut writer.write_buf, path.as_bytes(), &mut error);
        Self {
            writer,
            has_query: path.contains('?'),
            error,
            _phase: PhantomData,
        }
    }

    pub(crate) fn new_sealed(
        writer: &'a mut RequestWriter,
        method: Method,
        path: &str,
    ) -> RequestBuilder<'a, Headers> {
        writer.write_buf.clear();
        let mut error = if has_crlf(path) {
            Some(RestError::CrlfInjection)
        } else {
            None
        };
        checked_append(
            &mut writer.write_buf,
            method.as_str().as_bytes(),
            &mut error,
        );
        checked_append(&mut writer.write_buf, b" ", &mut error);
        if !writer.base_path.is_empty() {
            checked_append(&mut writer.write_buf, &writer.base_path, &mut error);
        }
        checked_append(&mut writer.write_buf, path.as_bytes(), &mut error);
        seal_request_line(writer, &mut error);
        RequestBuilder {
            writer,
            has_query: false,
            error,
            _phase: PhantomData,
        }
    }

    /// Add a query parameter. Percent-encodes key and value per RFC 3986.
    pub fn query(mut self, key: &str, value: &str) -> Self {
        let sep = if self.has_query { b"&" as &[u8] } else { b"?" };
        checked_append(&mut self.writer.write_buf, sep, &mut self.error);
        append_percent_encoded(&mut self.writer.write_buf, key.as_bytes(), &mut self.error);
        checked_append(&mut self.writer.write_buf, b"=", &mut self.error);
        append_percent_encoded(
            &mut self.writer.write_buf,
            value.as_bytes(),
            &mut self.error,
        );
        self.has_query = true;
        self
    }

    /// Add a pre-encoded query parameter. No percent-encoding applied.
    ///
    /// Caller is responsible for correct encoding. Validates no CR/LF.
    pub fn query_raw(mut self, key: &str, value: &str) -> Self {
        if has_crlf(key) || has_crlf(value) {
            self.error = Some(RestError::CrlfInjection);
            return self;
        }
        let sep = if self.has_query { b"&" as &[u8] } else { b"?" };
        checked_append(&mut self.writer.write_buf, sep, &mut self.error);
        checked_append(&mut self.writer.write_buf, key.as_bytes(), &mut self.error);
        checked_append(&mut self.writer.write_buf, b"=", &mut self.error);
        checked_append(
            &mut self.writer.write_buf,
            value.as_bytes(),
            &mut self.error,
        );
        self.has_query = true;
        self
    }

    /// Add a request header. Transitions to the headers phase.
    pub fn header(mut self, name: &str, value: &str) -> RequestBuilder<'a, Headers> {
        seal_request_line(self.writer, &mut self.error);
        let mut next = RequestBuilder {
            writer: self.writer,
            has_query: self.has_query,
            error: self.error,
            _phase: PhantomData,
        };
        next.append_header(name, value);
        next
    }

    /// Set the request body. Transitions to the ready phase.
    pub fn body(mut self, body: &[u8]) -> RequestBuilder<'a, Ready> {
        seal_request_line(self.writer, &mut self.error);
        write_body(&mut self.writer.write_buf, body, &mut self.error);
        RequestBuilder {
            writer: self.writer,
            has_query: self.has_query,
            error: self.error,
            _phase: PhantomData,
        }
    }

    /// Write the body directly into the buffer via a closure.
    ///
    /// Zero-alloc: the closure writes into the WriteBuf's spare region
    /// via [`BodyWriter`] (implements `std::io::Write`). Content-Length
    /// is computed and backfilled automatically.
    ///
    /// ```ignore
    /// let req = writer.post("/order")
    ///     .body_writer(|w| serde_json::to_writer(w, &order))
    ///     .finish()?;
    /// ```
    pub fn body_writer<F, E>(mut self, f: F) -> RequestBuilder<'a, Ready>
    where
        F: FnOnce(&mut BodyWriter<'_>) -> Result<(), E>,
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        seal_request_line(self.writer, &mut self.error);
        if self.error.is_some() {
            return RequestBuilder {
                writer: self.writer,
                has_query: self.has_query,
                error: self.error,
                _phase: PhantomData,
            };
        }
        let num_offset =
            write_content_length_placeholder(&mut self.writer.write_buf, &mut self.error);
        if self.error.is_some() {
            return RequestBuilder {
                writer: self.writer,
                has_query: self.has_query,
                error: self.error,
                _phase: PhantomData,
            };
        }
        let body_len = {
            let mut bw = BodyWriter::new(&mut self.writer.write_buf);
            if let Err(e) = f(&mut bw) {
                // If the buffer is full, this is a capacity issue.
                self.error = Some(if self.writer.write_buf.tailroom() == 0 {
                    RestError::RequestTooLarge {
                        capacity: self.writer.write_buf.len() + self.writer.write_buf.tailroom(),
                    }
                } else {
                    RestError::Io(std::io::Error::other(e))
                });
                0
            } else {
                bw.written()
            }
        }; // bw dropped here, releasing write_buf borrow
        if self.error.is_none() {
            backfill_content_length(&mut self.writer.write_buf, num_offset, body_len);
        }
        RequestBuilder {
            writer: self.writer,
            has_query: self.has_query,
            error: self.error,
            _phase: PhantomData,
        }
    }

    /// Write a fixed-size body via closure with direct `&mut [u8]` access.
    ///
    /// The caller specifies the exact body size upfront. Content-Length
    /// is written with exact digits (no backfill). The closure receives
    /// a zeroed `&mut [u8]` slice of exactly `len` bytes to write into.
    ///
    /// For binary wire formats (SBE, FIX, protobuf) where the message
    /// size is known at construction time.
    ///
    /// ```ignore
    /// let req = writer.post("/order")
    ///     .body_fixed(128, |buf| {
    ///         order.encode_sbe(buf);
    ///     })
    ///     .finish()?;
    /// ```
    pub fn body_fixed(
        mut self,
        len: usize,
        f: impl FnOnce(&mut [u8]),
    ) -> RequestBuilder<'a, Ready> {
        seal_request_line(self.writer, &mut self.error);
        // Write exact Content-Length (size known upfront).
        write_body_header(&mut self.writer.write_buf, len, &mut self.error);
        if self.error.is_some() {
            return RequestBuilder {
                writer: self.writer,
                has_query: self.has_query,
                error: self.error,
                _phase: PhantomData,
            };
        }
        // Reserve `len` bytes in the WriteBuf and let the closure write.
        let buf = &mut self.writer.write_buf;
        if len > buf.tailroom() {
            self.error = Some(RestError::RequestTooLarge {
                capacity: buf.len() + buf.tailroom(),
            });
        } else {
            let start = buf.len();
            // Extend the buffer with zeros, then give the closure mutable access.
            buf.extend_zeroed(len);
            let data = buf.data_mut();
            f(&mut data[start..start + len]);
        }
        RequestBuilder {
            writer: self.writer,
            has_query: self.has_query,
            error: self.error,
            _phase: PhantomData,
        }
    }

    /// Finish building. Returns the assembled request bytes.
    pub fn finish(mut self) -> Result<Request<'a>, RestError> {
        seal_request_line(self.writer, &mut self.error);
        checked_append(&mut self.writer.write_buf, b"\r\n", &mut self.error);
        if let Some(e) = self.error {
            return Err(e);
        }
        Ok(Request {
            data: self.writer.write_buf.data(),
        })
    }
}

// =========================================================================
// Headers phase
// =========================================================================

impl<'a> RequestBuilder<'a, Headers> {
    /// Add a request header.
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.append_header(name, value);
        self
    }

    /// Set the request body. Transitions to the ready phase.
    pub fn body(mut self, body: &[u8]) -> RequestBuilder<'a, Ready> {
        write_body(&mut self.writer.write_buf, body, &mut self.error);
        RequestBuilder {
            writer: self.writer,
            has_query: self.has_query,
            error: self.error,
            _phase: PhantomData,
        }
    }

    /// Write the body directly into the buffer via a closure.
    ///
    /// Same as [`RequestBuilder<Query>::body_writer`] — see its docs.
    pub fn body_writer<F, E>(mut self, f: F) -> RequestBuilder<'a, Ready>
    where
        F: FnOnce(&mut BodyWriter<'_>) -> Result<(), E>,
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        if self.error.is_some() {
            return RequestBuilder {
                writer: self.writer,
                has_query: self.has_query,
                error: self.error,
                _phase: PhantomData,
            };
        }
        let num_offset =
            write_content_length_placeholder(&mut self.writer.write_buf, &mut self.error);
        if self.error.is_some() {
            return RequestBuilder {
                writer: self.writer,
                has_query: self.has_query,
                error: self.error,
                _phase: PhantomData,
            };
        }
        let body_len = {
            let mut bw = BodyWriter::new(&mut self.writer.write_buf);
            if let Err(e) = f(&mut bw) {
                // If the buffer is full, this is a capacity issue.
                self.error = Some(if self.writer.write_buf.tailroom() == 0 {
                    RestError::RequestTooLarge {
                        capacity: self.writer.write_buf.len() + self.writer.write_buf.tailroom(),
                    }
                } else {
                    RestError::Io(std::io::Error::other(e))
                });
                0
            } else {
                bw.written()
            }
        }; // bw dropped here, releasing write_buf borrow
        if self.error.is_none() {
            backfill_content_length(&mut self.writer.write_buf, num_offset, body_len);
        }
        RequestBuilder {
            writer: self.writer,
            has_query: self.has_query,
            error: self.error,
            _phase: PhantomData,
        }
    }

    /// Write a fixed-size body via closure with direct `&mut [u8]` access.
    ///
    /// Same as [`RequestBuilder<Query>::body_fixed`] — see its docs.
    pub fn body_fixed(
        mut self,
        len: usize,
        f: impl FnOnce(&mut [u8]),
    ) -> RequestBuilder<'a, Ready> {
        write_body_header(&mut self.writer.write_buf, len, &mut self.error);
        if self.error.is_some() {
            return RequestBuilder {
                writer: self.writer,
                has_query: self.has_query,
                error: self.error,
                _phase: PhantomData,
            };
        }
        let buf = &mut self.writer.write_buf;
        if len > buf.tailroom() {
            self.error = Some(RestError::RequestTooLarge {
                capacity: buf.len() + buf.tailroom(),
            });
        } else {
            let start = buf.len();
            buf.extend_zeroed(len);
            let data = buf.data_mut();
            f(&mut data[start..start + len]);
        }
        RequestBuilder {
            writer: self.writer,
            has_query: self.has_query,
            error: self.error,
            _phase: PhantomData,
        }
    }

    /// Finish building. Returns the assembled request bytes.
    pub fn finish(mut self) -> Result<Request<'a>, RestError> {
        checked_append(&mut self.writer.write_buf, b"\r\n", &mut self.error);
        if let Some(e) = self.error {
            return Err(e);
        }
        Ok(Request {
            data: self.writer.write_buf.data(),
        })
    }

    fn append_header(&mut self, name: &str, value: &str) {
        if self.error.is_some() {
            return;
        }
        if has_crlf(name) || has_crlf(value) {
            self.error = Some(RestError::CrlfInjection);
            return;
        }
        checked_append(&mut self.writer.write_buf, name.as_bytes(), &mut self.error);
        checked_append(&mut self.writer.write_buf, b": ", &mut self.error);
        checked_append(
            &mut self.writer.write_buf,
            value.as_bytes(),
            &mut self.error,
        );
        checked_append(&mut self.writer.write_buf, b"\r\n", &mut self.error);
    }
}

// =========================================================================
// Ready phase
// =========================================================================

impl<'a> RequestBuilder<'a, Ready> {
    /// Finish building. Returns the assembled request bytes.
    pub fn finish(self) -> Result<Request<'a>, RestError> {
        if let Some(e) = self.error {
            return Err(e);
        }
        Ok(Request {
            data: self.writer.write_buf.data(),
        })
    }
}

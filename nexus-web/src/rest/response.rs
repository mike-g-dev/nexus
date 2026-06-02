//! REST response — borrows from the ResponseReader's buffer.

use crate::http::ResponseReader;

/// HTTP response. Borrows from the connection's ResponseReader.
///
/// Must be dropped before the next request on the same connection
/// (same pattern as WebSocket `Message<'_>`).
pub struct RestResponse<'a> {
    pub(crate) status: u16,
    pub(crate) body_len: usize,
    pub(crate) resp_reader: &'a ResponseReader,
    /// Decoded body for chunked responses. None = use reader remainder.
    pub(crate) chunked_body: Option<Vec<u8>>,
}

impl<'a> RestResponse<'a> {
    /// Create a response from parsed data (Content-Length delimited).
    ///
    /// Typically called by transport layers after reading and parsing
    /// the response.
    pub fn new(status: u16, body_len: usize, resp_reader: &'a ResponseReader) -> Self {
        Self {
            status,
            body_len,
            resp_reader,
            chunked_body: None,
        }
    }

    /// Create a response with a decoded chunked body.
    pub fn new_chunked(status: u16, body: Vec<u8>, resp_reader: &'a ResponseReader) -> Self {
        let body_len = body.len();
        Self {
            status,
            body_len,
            resp_reader,
            chunked_body: Some(body),
        }
    }

    /// HTTP status code.
    pub fn status(&self) -> u16 {
        self.status
    }

    /// Look up a response header by name (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.resp_reader.header(name)
    }

    /// Response body as bytes.
    pub fn body(&self) -> &[u8] {
        if let Some(ref chunked) = self.chunked_body {
            return chunked;
        }
        let remainder = self.resp_reader.remainder();
        &remainder[..self.body_len.min(remainder.len())]
    }

    /// Response body as a UTF-8 string.
    pub fn body_str(&self) -> Result<&str, std::str::Utf8Error> {
        std::str::from_utf8(self.body())
    }

    /// Response body length (from Content-Length header).
    pub fn body_len(&self) -> usize {
        self.body_len
    }

    /// Copy the response body into `bytes::Bytes`.
    ///
    /// Allocates once — copies the body slice into a `Bytes` handle
    /// that is `Send + Clone` for cross-thread passing.
    ///
    /// ```ignore
    /// let resp = conn.send(req, &mut reader)?;
    /// let body: Bytes = resp.body_to_bytes();
    /// tx.send(body)?;
    /// ```
    #[cfg(feature = "bytes")]
    pub fn body_to_bytes(&self) -> bytes::Bytes {
        bytes::Bytes::copy_from_slice(self.body())
    }

    /// Number of response headers.
    pub fn header_count(&self) -> usize {
        self.resp_reader.header_count()
    }
}

impl std::fmt::Debug for RestResponse<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RestResponse")
            .field("status", &self.status)
            .field("body_len", &self.body_len)
            .finish()
    }
}

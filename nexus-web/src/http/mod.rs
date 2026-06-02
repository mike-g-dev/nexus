//! Sans-IO HTTP/1.x protocol primitives.
//!
//! Built on [`httparse`] for SIMD-accelerated header parsing.
//! Uses [`ReadBuf`](nexus_net::buf::ReadBuf) for incremental byte buffering.
//!
//! - [`ResponseReader`] — parse inbound HTTP responses (used by REST client)
//! - [`ChunkedDecoder`] — chunked transfer encoding decoder
//! - [`write_request`] / [`write_response`] — construct outbound HTTP messages
//!
//! The HTTP client API is in [`rest`](crate::rest).
//! `RequestReader` is internal (used for WebSocket upgrade handshake).

mod chunked;
mod error;
mod request;
mod response;

/// Default capacity for HTTP read/decode/scratch buffers.
///
/// Sized to comfortably fit a typical HTTP/1.1 head section (request
/// line + headers up to ~3-4 KiB) in a single allocation, which is the
/// dominant use site. Also used by `nexus-async-web` as the per-recv
/// read cap during WebSocket upgrade and REST request/response cycles,
/// and as the initial capacity of intermediate body / wire / decode
/// scratch buffers.
///
/// Currently a hardcoded internal default. Callers with unusually large
/// HTTP heads (very long cookies, many or large header values) would
/// today need to work around by sending fewer headers; a builder knob
/// is a separate concern.
pub const HTTP_HANDSHAKE_BUFFER: usize = 4096;

pub use chunked::ChunkedDecoder;
pub use error::HttpError;
// RequestReader parses inbound HTTP requests (used for WS upgrade handshake).
// The public HTTP client API is in `rest::`.
pub use request::RequestReader;
pub use response::{
    Response, ResponseReader, request_size, response_size, write_request, write_response,
};

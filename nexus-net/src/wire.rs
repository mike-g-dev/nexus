//! Composition seam between transports and parsers.
//!
//! [`WireStream`] is the trait `WsStream`/`HttpConnection` consume —
//! abstracts over plain TCP, TLS-wrapped TCP, and any user-provided
//! transport (via the `AsyncReadAdapter` types in `nexus-async-web`,
//! one per supported runtime).
//! [`ParserSink`] is the parser-side buffer; implemented by
//! nexus-web's `FrameReader` and the HTTP response parser, plus any
//! third-party parser that follows the `spare`/`filled` discipline.
//!
//! The split lets implementations skip the `&mut [u8]` intermediate
//! that `AsyncRead`'s contract requires when a faster path is
//! available — notably, the nexus-async-rt TLS adapter copies bytes
//! from rustls's plaintext queue straight into the parser's spare
//! region (one memcpy instead of two per recv).

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A bidirectional byte-level stream that fills a parser's buffer.
///
/// Abstracts over plain TCP, TLS-wrapped TCP, and any user-provided
/// transport (wrap in `nexus_async_web::AsyncReadAdapter` /
/// `NexusAsyncReadAdapter` for the latter, depending on backend).
///
/// Implementations may take advantage of zero-copy paths when
/// available — e.g. `MaybeTls`'s TLS variant feeds plaintext directly
/// from rustls's queue into the parser's buffer, skipping the
/// intermediate `&mut [u8]` copy that `AsyncRead`'s contract requires.
pub trait WireStream {
    /// Read bytes into `sink.spare()`. Returns the number of bytes
    /// actually delivered to the sink. `Ok(0)` indicates EOF.
    ///
    /// `max` caps the bytes pulled in one call — useful for tail
    /// latency control. Implementations must respect it.
    ///
    /// # Preconditions
    ///
    /// - `max > 0`
    /// - `sink.spare()` is non-empty
    ///
    /// Implementations return `Err(io::ErrorKind::InvalidInput)` if
    /// either precondition is violated. With the preconditions met,
    /// `Ok(0)` unambiguously signals EOF — callers do not need to
    /// inspect sink state to distinguish "no buffer space" from
    /// "transport closed."
    fn poll_fill_into<P: ParserSink>(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        sink: &mut P,
        max: usize,
    ) -> Poll<io::Result<usize>>;

    /// Write `buf` to the transport. Returns the number of bytes
    /// accepted (may be less than `buf.len()`). `Ok(0)` may indicate
    /// the transport cannot accept more right now or that the
    /// underlying connection has been closed for writes — same
    /// semantics as `tokio::io::AsyncWrite::poll_write`. Implementors
    /// of new transports should mirror that contract.
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>>;

    /// Flush bytes previously written via `poll_write` to the
    /// transport. Returns `Poll::Ready(Ok(()))` once all buffered
    /// data has been pushed to the underlying socket. Mirrors
    /// `tokio::io::AsyncWrite::poll_flush`.
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>>;

    /// Shut down the write side of the transport. For TLS-bearing
    /// implementations, send `close_notify` first and flush before
    /// closing the underlying connection. Mirrors
    /// `tokio::io::AsyncWrite::poll_shutdown`.
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>>;
}

/// Parser-side buffer that a [`WireStream`] fills.
///
/// Implementations expose a writable spare region via [`spare()`](Self::spare)
/// and commit the bytes via [`filled(n)`](Self::filled). This is the
/// same shape as `nexus_net::buf::ReadBuf` — any parser already using
/// that pattern can implement `ParserSink` in two methods.
pub trait ParserSink {
    /// Writable region where new bytes go.
    fn spare(&mut self) -> &mut [u8];

    /// Commit `n` bytes written into [`spare()`](Self::spare).
    ///
    /// # Panics
    /// Implementations panic if `n > spare().len()`.
    fn filled(&mut self, n: usize);
}

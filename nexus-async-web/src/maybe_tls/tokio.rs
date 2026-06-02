//! MaybeTls — plain TCP or TLS, unified async I/O.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;

/// Async stream that may or may not be TLS-wrapped.
///
/// Created by [`WsStreamBuilder::connect()`](crate::ws::WsStreamBuilder::connect)
/// or [`HttpConnectionBuilder::connect()`](crate::rest::HttpConnectionBuilder::connect)
/// based on the URL scheme.
pub enum MaybeTls {
    /// Plain TCP (ws://, http://).
    Plain(TcpStream),
    /// TLS over TCP (wss://, https://).
    #[cfg(feature = "tls")]
    Tls(Box<tokio_rustls::client::TlsStream<TcpStream>>),
}

impl AsyncRead for MaybeTls {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(s) => Pin::new(s).poll_read(cx, buf),
            #[cfg(feature = "tls")]
            Self::Tls(s) => Pin::new(s).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for MaybeTls {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Self::Plain(s) => Pin::new(s).poll_write(cx, buf),
            #[cfg(feature = "tls")]
            Self::Tls(s) => Pin::new(s).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(s) => Pin::new(s).poll_flush(cx),
            #[cfg(feature = "tls")]
            Self::Tls(s) => Pin::new(s).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(s) => Pin::new(s).poll_shutdown(cx),
            #[cfg(feature = "tls")]
            Self::Tls(s) => Pin::new(s).poll_shutdown(cx),
        }
    }
}

// =============================================================================
// WireStream
// =============================================================================
//
// On the tokio backend the TLS variant goes through `tokio_rustls`,
// which buffers plaintext internally and exposes only the
// `AsyncRead` interface — there's no direct way to drain that
// buffer into a `ParserSink` without a slice intermediate. So both
// variants here delegate to the slow path (poll_read into
// `sink.spare()`); the type-system seam is consistent across
// backends, but the zero-copy win is nexus-async-rt-only.

impl nexus_net::WireStream for MaybeTls {
    fn poll_fill_into<P: nexus_net::ParserSink>(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        sink: &mut P,
        max: usize,
    ) -> Poll<io::Result<usize>> {
        if max == 0 || sink.spare().is_empty() {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "poll_fill_into called with no buffer space \
                 (max == 0 or sink.spare() is empty)",
            )));
        }
        match self.get_mut() {
            Self::Plain(s) => fill_via_async_read(Pin::new(s), cx, sink, max),
            #[cfg(feature = "tls")]
            Self::Tls(s) => fill_via_async_read(Pin::new(s), cx, sink, max),
        }
    }

    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        <Self as AsyncWrite>::poll_write(self, cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        <Self as AsyncWrite>::poll_flush(self, cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        <Self as AsyncWrite>::poll_shutdown(self, cx)
    }
}

/// Slow-path helper: poll_read from a tokio AsyncRead source
/// directly into `sink.spare()`, capped at `max`. Inlined here
/// rather than shared with `AsyncReadAdapter` (in this crate) to
/// keep the cross-crate seam minimal.
///
/// Caller (the `WireStream::poll_fill_into` impl above) already
/// validated `max > 0` and `sink.spare()` non-empty per the trait
/// contract — no need to re-check.
fn fill_via_async_read<S, P>(
    stream: Pin<&mut S>,
    cx: &mut Context<'_>,
    sink: &mut P,
    max: usize,
) -> Poll<io::Result<usize>>
where
    S: AsyncRead + ?Sized,
    P: nexus_net::ParserSink,
{
    let spare = sink.spare();
    let cap = spare.len().min(max);
    let mut tmp_buf = ReadBuf::new(&mut spare[..cap]);
    match stream.poll_read(cx, &mut tmp_buf) {
        Poll::Ready(Ok(())) => {
            let n = tmp_buf.filled().len();
            if n > 0 {
                sink.filled(n);
            }
            Poll::Ready(Ok(n))
        }
        Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
        Poll::Pending => Poll::Pending,
    }
}

//! MaybeTls — plain TCP or TLS, unified async I/O (nexus-async-rt backend).
//!
//! Unlike the tokio variant which delegates TLS to `tokio-rustls`, this
//! drives nexus-net's sans-IO [`TlsCodec`] at the poll level. The codec
//! handles encrypt/decrypt; we shuttle bytes between it and the TCP stream.
//!
//! The TLS variant is constructed atomically via [`TlsInner::connect`],
//! which allocates the per-connection buffers and drives the handshake
//! to completion before returning. A `TlsInner` value is always
//! post-handshake; there is no "construct, then handshake later"
//! two-step.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use nexus_async_rt::{AsyncRead, AsyncWrite, TcpStream};
#[cfg(feature = "tls")]
use nexus_net::buf::{ReadBuf, WriteBuf};
#[cfg(feature = "tls")]
use nexus_net::tls::{TlsBufferCapacities, TlsCodec, TlsError};

/// Async stream that may or may not be TLS-wrapped.
///
/// Created by connection builders based on the URL scheme.
///
/// # Composition
///
/// Implements [`nexus_net::WireStream`] in addition to
/// [`AsyncRead`] / [`AsyncWrite`]. The `WireStream` impl fast-paths
/// the TLS variant by copying bytes from rustls's plaintext queue
/// straight into a [`ParserSink`](nexus_net::ParserSink)'s spare
/// region — one memcpy per recv instead of two on the AsyncRead path
/// (which mandates a `&mut [u8]` intermediate). `WsStream` and
/// `HttpConnection` consume `MaybeTls` through `WireStream` to pick
/// up the fast path automatically.
///
/// # Shutdown (TLS variant)
///
/// `poll_shutdown` queues a TLS `close_notify` alert, flushes the
/// resulting ciphertext to the transport, then closes the underlying
/// transport. Callers do not need to flush manually — `poll_shutdown`
/// drives any pending plaintext through to the wire as part of its
/// shutdown sequence.
///
/// If the caller drops the stream without calling `poll_shutdown`,
/// any pending plaintext (in rustls's outbound queue) and ciphertext
/// (in `pending_write`) is discarded, and the peer sees TCP FIN
/// without close_notify — which rustls treats as a truncation alert.
/// Callers needing graceful termination must call `shutdown().await`
/// (or drive `poll_shutdown` to `Ready`) before drop.
///
/// # Memory (TLS variant)
///
/// Steady-state per connection: approximately 35 KiB of heap-resident
/// buffers. Worst-case under bursty inbound: up to ~99 KiB if rustls's
/// outbound plaintext queue fills to its default 64 KiB cap.
///
/// | Buffer | Default size | Purpose |
/// |---|---|---|
/// | `pending_read` | 18 KiB | Inbound ciphertext FIFO (transport read target + codec input). 18 KiB covers max TLS 1.3 record (16,384 plaintext + AEAD overhead). |
/// | `pending_write` | 16 KiB | Outbound ciphertext FIFO (drains to socket) |
/// | rustls state | ~1 KiB | Crypto state + small fixed buffers |
/// | rustls plaintext queue | up to 64 KiB | Outbound plaintext awaiting encryption; cap is rustls's `DEFAULT_BUFFER_LIMIT`, configurable via `TlsBufferCapacities::rustls_plaintext_limit`. |
///
/// Trading workloads with strict per-connection memory budgets can
/// drop the rustls plaintext cap to 8–16 KiB and reduce
/// `pending_write` similarly. Bulk-transfer workloads (large
/// snapshots, file uploads) can raise both via the connection
/// builder's `tls_buffer_capacities` setter (takes a
/// [`TlsBufferCapacities`]).
pub enum MaybeTls {
    /// Plain TCP (ws://, http://).
    Plain(TcpStream),
    /// TLS over TCP (wss://, https://).
    #[cfg(feature = "tls")]
    Tls(Box<TlsInner>),
}

/// TLS state: a TCP stream plus the sans-IO codec and cursor-based
/// staging buffers for ciphertext in both directions.
///
/// Construct via [`TlsInner::connect`] — the handshake completes
/// before the value is returned. Post-construction, the type
/// implements [`AsyncRead`] / [`AsyncWrite`] for steady-state I/O.
///
/// Opaque to users — fields are `pub(crate)`. Exposed only because
/// [`MaybeTls::Tls`] holds a `Box<TlsInner>`.
#[cfg(feature = "tls")]
pub struct TlsInner {
    pub(crate) stream: TcpStream,
    pub(crate) codec: TlsCodec,
    /// Inbound ciphertext FIFO. The transport reads directly into
    /// `spare()` and the codec reads from `data()` — single buffer,
    /// no scratch tmp needed.
    pending_read: ReadBuf,
    /// Outbound ciphertext FIFO. Cursor-based, O(1) advance with
    /// auto-reset when fully drained.
    pending_write: WriteBuf,
}

#[cfg(feature = "tls")]
impl TlsInner {
    /// Construct + drive the TLS handshake atomically. Allocates the
    /// per-connection buffers per `capacities`, then drives the
    /// handshake to completion before returning.
    ///
    /// On success the returned `TlsInner` is ready for plaintext
    /// I/O. On failure (handshake error, transport error) the buffers
    /// and the codec are dropped.
    ///
    /// The returned future is `!Send` because `nexus_async_rt::TcpStream`
    /// is `!Send` by design — the nexus-async-rt runtime is
    /// single-threaded and pins IO state to the local thread.
    #[allow(clippy::future_not_send)]
    pub async fn connect(
        stream: TcpStream,
        mut codec: TlsCodec,
        capacities: TlsBufferCapacities,
    ) -> Result<Self, TlsError> {
        // Apply the rustls plaintext queue cap (if specified) before
        // any encrypts can happen. None keeps rustls's default.
        if let Some(limit) = capacities.rustls_plaintext_limit() {
            codec.set_buffer_limit(Some(limit));
        }
        let mut inner = Self {
            stream,
            codec,
            pending_read: ReadBuf::with_capacity(capacities.read_chunk()),
            pending_write: WriteBuf::new(capacities.pending_write(), 0),
        };
        inner.drive_handshake().await?;
        Ok(inner)
    }

    /// Drive the TLS handshake to completion using this struct's own
    /// buffers. The post-handshake state of `pending_read` / `codec`
    /// flows naturally into steady-state `poll_read` — including any
    /// piggybacked TLS 1.3 app-data records that arrived in the same
    /// burst as `ServerFinished`. Allocation-free past `connect`'s
    /// initial buffer construction.
    ///
    /// `!Send` for the same reason as `connect` — nexus-async-rt's
    /// IO types are intentionally single-threaded.
    #[allow(clippy::future_not_send)]
    async fn drive_handshake(&mut self) -> Result<(), TlsError> {
        while self.codec.is_handshaking() {
            // Drain outbound first (ClientHello, then client Finished
            // after we've consumed ServerHello + Certificate + ...).
            while self.codec.wants_write() {
                if self.pending_write.spare().is_empty() {
                    handshake_drain_pending(self).await?;
                    if self.pending_write.spare().is_empty() {
                        return Err(TlsError::Io(io::Error::new(
                            io::ErrorKind::WriteZero,
                            "pending_write full and socket cannot accept \
                             during handshake",
                        )));
                    }
                }
                let n = self.codec.write_tls_to(&mut self.pending_write.spare())?;
                if n == 0 {
                    // wants_write said yes, spare was non-empty (checked
                    // above), yet rustls produced 0 bytes. Contract
                    // violation — silent break would loop the outer
                    // while forever with wants_write still true.
                    return Err(TlsError::Io(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "rustls reported wants_write but produced 0 bytes \
                         into a non-empty buffer during handshake",
                    )));
                }
                self.pending_write.filled(n);
                handshake_drain_pending(self).await?;
            }
            handshake_drain_pending(self).await?;

            if !self.codec.is_handshaking() {
                break;
            }

            // Read directly into pending_read.spare() — same buffer
            // steady-state poll_read uses, so any piggybacked app-data
            // remainder lands where the streaming reader expects it.
            if self.pending_read.spare().is_empty() {
                // Buffer is full but rustls couldn't decode a record from
                // what we have — same condition as the steady-state
                // poll_read branch. Match its kind for consistency.
                return Err(TlsError::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "pending_read full mid-handshake but rustls cannot \
                     decode a record",
                )));
            }
            let n = handshake_read_into_spare(self).await?;
            if n == 0 {
                return Err(TlsError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "connection closed during TLS handshake",
                )));
            }

            // Step pending_read until the handshake completes or the
            // buffer is exhausted. If handshake completes mid-buffer,
            // the remainder stays in pending_read for steady-state
            // poll_read to pick up — zero allocation.
            while !self.pending_read.is_empty() && self.codec.is_handshaking() {
                let consumed = self.codec.read_tls(self.pending_read.data())?;
                if consumed == 0 {
                    // Deframer needs more bytes than we have in the
                    // buffer; loop back to the outer wants_read branch.
                    break;
                }
                self.pending_read.advance(consumed);
            }
        }

        // Final flush: covers the case where `is_handshaking()`
        // flipped false mid-burst with one extra record (typically
        // the client Finished) still queued in the codec. The inner
        // loop above drains between writes, so this block is rarely
        // reached and usually does zero work — but the rustls API
        // doesn't promise wants_write is empty just because handshake
        // completed.
        while self.codec.wants_write() {
            if self.pending_write.spare().is_empty() {
                handshake_drain_pending(self).await?;
                if self.pending_write.spare().is_empty() {
                    return Err(TlsError::Io(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "pending_write full and socket cannot accept \
                         during handshake",
                    )));
                }
            }
            let n = self.codec.write_tls_to(&mut self.pending_write.spare())?;
            if n == 0 {
                // Same contract violation as the inner write loop —
                // surface explicitly rather than silently exiting with
                // wants_write still true.
                return Err(TlsError::Io(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "rustls reported wants_write but produced 0 bytes \
                     into a non-empty buffer during handshake flush",
                )));
            }
            self.pending_write.filled(n);
        }
        handshake_drain_pending(self).await?;

        Ok(())
    }
}

impl MaybeTls {
    /// Whether this connection is TLS-wrapped.
    pub fn is_tls(&self) -> bool {
        match self {
            Self::Plain(_) => false,
            #[cfg(feature = "tls")]
            Self::Tls(_) => true,
        }
    }
}

// =============================================================================
// AsyncRead
// =============================================================================

impl AsyncRead for MaybeTls {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            MaybeTls::Plain(s) => Pin::new(s).poll_read(cx, buf),
            #[cfg(feature = "tls")]
            MaybeTls::Tls(inner) => {
                if buf.is_empty() {
                    return Poll::Ready(Ok(0));
                }

                loop {
                    // 1. Drain plaintext rustls has already decrypted.
                    let n = inner.codec.read_plaintext(buf).map_err(tls_to_io)?;
                    if n > 0 {
                        return Poll::Ready(Ok(n));
                    }

                    // 2. Step buffered ciphertext one packet at a time.
                    if !inner.pending_read.is_empty() {
                        let consumed = inner
                            .codec
                            .read_tls(inner.pending_read.data())
                            .map_err(tls_to_io)?;
                        if consumed == 0 {
                            // Deframer needs more bytes; fall through to
                            // the transport-read branch. Keep pending_read
                            // intact.
                        } else {
                            // State invariant: every error leg above this
                            // line MUST return before reaching here. New
                            // error returns must be placed BEFORE this
                            // side-effect — pending_read can be left
                            // inconsistent if advance() is half-applied.
                            inner.pending_read.advance(consumed);
                            continue;
                        }
                    }

                    // 3. Pull fresh ciphertext directly into
                    //    pending_read.spare(). No separate tmp needed.
                    if inner.pending_read.spare().is_empty() {
                        // Buffer is full but no records have decoded —
                        // would be a malformed stream. Return error.
                        return Poll::Ready(Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "pending_read full but rustls cannot decode \
                             a record",
                        )));
                    }
                    match Pin::new(&mut inner.stream).poll_read(cx, inner.pending_read.spare()) {
                        Poll::Ready(Ok(0)) => return Poll::Ready(Ok(0)), // EOF
                        Poll::Ready(Ok(filled)) => {
                            inner.pending_read.filled(filled);
                        }
                        Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                        Poll::Pending => return Poll::Pending,
                    }
                }
            }
        }
    }
}

// =============================================================================
// AsyncWrite
// =============================================================================

impl AsyncWrite for MaybeTls {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            MaybeTls::Plain(s) => Pin::new(s).poll_write(cx, buf),
            #[cfg(feature = "tls")]
            MaybeTls::Tls(inner) => {
                // Empty-buffer write is a no-op: AsyncWrite allows it,
                // and skipping the drain/encrypt cycle avoids the
                // defensive `wake_by_ref` busy-spin path below firing
                // on `try_encrypt(b"")` returning 0.
                if buf.is_empty() {
                    return Poll::Ready(Ok(0));
                }
                // 1. Drain pending ciphertext to free pending_write space.
                drain_pending(inner, cx)?;
                if !inner.pending_write.is_empty() {
                    return Poll::Pending;
                }

                // 2. Pull queued ciphertext from rustls into pending_write
                //    and on to the socket. Frees rustls's plaintext queue
                //    so encrypt has room for new bytes.
                drain_codec_to_pending(inner, cx)?;
                drain_pending(inner, cx)?;
                if !inner.pending_write.is_empty() {
                    return Poll::Pending;
                }

                // 3. Encrypt as much of buf as rustls's queue can accept.
                let consumed = inner.codec.encrypt(buf).map_err(tls_to_io)?;
                if consumed == 0 {
                    // Steps 1-3 above just drained both pending_write
                    // and rustls's outbound plaintext queue. If encrypt
                    // STILL returns 0, `buf.len()` exceeds the rustls
                    // plaintext queue limit (default 64 KiB). A
                    // wake_by_ref + Pending here would busy-spin
                    // forever (next poll repeats the same path with
                    // the same buf, same result). Mirror sync
                    // TlsStream::write — surface as WriteZero so the
                    // caller sees a hard, actionable failure.
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "rustls plaintext queue limit smaller than \
                         remaining input — raise via \
                         TlsBufferCapacities::rustls_plaintext_limit \
                         or chunk the write into smaller pieces",
                    )));
                }

                // 4. Best-effort flush of what we just produced.
                drain_codec_to_pending(inner, cx)?;
                drain_pending(inner, cx)?;

                Poll::Ready(Ok(consumed))
            }
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            MaybeTls::Plain(s) => Pin::new(s).poll_flush(cx),
            #[cfg(feature = "tls")]
            MaybeTls::Tls(inner) => {
                drain_codec_to_pending(inner, cx)?;
                drain_pending(inner, cx)?;
                if !inner.pending_write.is_empty() {
                    return Poll::Pending;
                }
                Pin::new(&mut inner.stream).poll_flush(cx)
            }
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            MaybeTls::Plain(s) => Pin::new(s).poll_shutdown(cx),
            #[cfg(feature = "tls")]
            MaybeTls::Tls(inner) => {
                // 1. Queue close_notify (idempotent).
                inner.codec.send_close_notify();
                // 2. Drain rustls's queue (now including close_notify
                //    ciphertext) into pending_write.
                drain_codec_to_pending(inner, cx)?;
                // 3. Flush pending_write to the transport.
                drain_pending(inner, cx)?;
                if !inner.pending_write.is_empty() {
                    return Poll::Pending;
                }
                // 4. Now safe to shutdown the transport.
                Pin::new(&mut inner.stream).poll_shutdown(cx)
            }
        }
    }
}

// =============================================================================
// WireStream — zero-copy parser feed
// =============================================================================

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
            MaybeTls::Plain(s) => fill_via_nexus_async_read(Pin::new(s), cx, sink, max),
            #[cfg(feature = "tls")]
            MaybeTls::Tls(inner) => {
                loop {
                    // 1. Fast path: drain plaintext rustls already
                    //    decrypted. Direct copy from rustls's queue
                    //    into sink.spare() — no &mut [u8] intermediate.
                    let mut limited = LimitedSink::new(sink, max);
                    let n = inner
                        .codec
                        .drain_plaintext_into(&mut limited)
                        .map_err(tls_to_io)?;
                    if n > 0 {
                        return Poll::Ready(Ok(n));
                    }

                    // 2. Step buffered ciphertext one packet at a time.
                    if !inner.pending_read.is_empty() {
                        let consumed = inner
                            .codec
                            .read_tls(inner.pending_read.data())
                            .map_err(tls_to_io)?;
                        if consumed == 0 {
                            // Deframer needs more bytes; fall through to
                            // the transport-read branch.
                        } else {
                            // State invariant: every error leg above this
                            // line MUST return before reaching here.
                            inner.pending_read.advance(consumed);
                            continue;
                        }
                    }

                    // 3. Pull fresh ciphertext into pending_read.spare().
                    if inner.pending_read.spare().is_empty() {
                        return Poll::Ready(Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "pending_read full but rustls cannot decode \
                             a record",
                        )));
                    }
                    match Pin::new(&mut inner.stream).poll_read(cx, inner.pending_read.spare()) {
                        Poll::Ready(Ok(0)) => return Poll::Ready(Ok(0)), // EOF
                        Poll::Ready(Ok(filled)) => {
                            inner.pending_read.filled(filled);
                        }
                        Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                        Poll::Pending => return Poll::Pending,
                    }
                }
            }
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

/// Wraps a `ParserSink` to cap the spare region — used by
/// `poll_fill_into` to enforce the `max` argument when delegating
/// to `drain_plaintext_into`.
#[cfg(feature = "tls")]
struct LimitedSink<'a, P: nexus_net::ParserSink> {
    inner: &'a mut P,
    remaining: usize,
}

#[cfg(feature = "tls")]
impl<'a, P: nexus_net::ParserSink> LimitedSink<'a, P> {
    fn new(inner: &'a mut P, max: usize) -> Self {
        Self {
            inner,
            remaining: max,
        }
    }
}

#[cfg(feature = "tls")]
impl<P: nexus_net::ParserSink> nexus_net::ParserSink for LimitedSink<'_, P> {
    fn spare(&mut self) -> &mut [u8] {
        let s = self.inner.spare();
        let n = s.len().min(self.remaining);
        &mut s[..n]
    }
    fn filled(&mut self, n: usize) {
        self.inner.filled(n);
        // saturating_sub is defensive — by contract `n <= spare().len()`
        // and our `spare()` caps at `remaining`, so `n <= remaining`
        // always. Saturation is unreachable but cheap.
        self.remaining = self.remaining.saturating_sub(n);
    }
}

/// Slow-path helper: drive a nexus-async-rt `AsyncRead` source
/// directly into `sink.spare()`, capped at `max`.
///
/// Caller (the `WireStream::poll_fill_into` impl above) already
/// validated `max > 0` and `sink.spare()` non-empty per the trait
/// contract — no need to re-check.
fn fill_via_nexus_async_read<S, P>(
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
    match stream.poll_read(cx, &mut spare[..cap]) {
        Poll::Ready(Ok(n)) => {
            if n > 0 {
                sink.filled(n);
            }
            Poll::Ready(Ok(n))
        }
        Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
        Poll::Pending => Poll::Pending,
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Drain the `pending_write` buffer to the transport, writing as much
/// as the socket will accept without blocking.
#[cfg(feature = "tls")]
fn drain_pending(inner: &mut TlsInner, cx: &mut Context<'_>) -> io::Result<()> {
    while !inner.pending_write.is_empty() {
        match Pin::new(&mut inner.stream).poll_write(cx, inner.pending_write.data()) {
            Poll::Ready(Ok(0)) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "transport write returned 0",
                ));
            }
            Poll::Ready(Ok(n)) => {
                inner.pending_write.advance(n);
            }
            Poll::Ready(Err(e)) => return Err(e),
            Poll::Pending => return Ok(()),
        }
    }
    Ok(())
}

/// Move ciphertext rustls wants to write into `pending_write`,
/// draining `pending_write` to the socket between iterations so a
/// single big encrypt can't outrun `pending_write`'s fixed capacity.
#[cfg(feature = "tls")]
fn drain_codec_to_pending(inner: &mut TlsInner, cx: &mut Context<'_>) -> io::Result<()> {
    while inner.codec.wants_write() {
        if inner.pending_write.spare().is_empty() {
            drain_pending(inner, cx)?;
            if inner.pending_write.spare().is_empty() {
                return Ok(());
            }
        }
        let n = inner.codec.write_tls_to(&mut inner.pending_write.spare())?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "rustls reported wants_write but produced 0 bytes \
                 into a non-empty buffer",
            ));
        }
        inner.pending_write.filled(n);
        drain_pending(inner, cx)?;
    }
    Ok(())
}

/// Convert a [`TlsError`] into an [`io::Error`].
#[cfg(feature = "tls")]
fn tls_to_io(e: TlsError) -> io::Error {
    match e {
        TlsError::Io(io_err) => io_err,
        other => io::Error::other(other),
    }
}

// =============================================================================
// Handshake helpers (async wrappers around poll_fn for nexus-async-rt)
// =============================================================================

#[cfg(feature = "tls")]
#[allow(clippy::future_not_send)] // Single-threaded runtime — TcpStream is !Send by design.
async fn handshake_drain_pending(inner: &mut TlsInner) -> Result<(), TlsError> {
    use std::future::poll_fn;
    while !inner.pending_write.is_empty() {
        let n =
            poll_fn(|cx| Pin::new(&mut inner.stream).poll_write(cx, inner.pending_write.data()))
                .await
                .map_err(TlsError::Io)?;
        if n == 0 {
            return Err(TlsError::Io(io::Error::new(
                io::ErrorKind::WriteZero,
                "transport write returned 0 during TLS handshake",
            )));
        }
        inner.pending_write.advance(n);
    }
    poll_fn(|cx| Pin::new(&mut inner.stream).poll_flush(cx))
        .await
        .map_err(TlsError::Io)?;
    Ok(())
}

#[cfg(feature = "tls")]
#[allow(clippy::future_not_send)] // Single-threaded runtime — TcpStream is !Send by design.
async fn handshake_read_into_spare(inner: &mut TlsInner) -> Result<usize, TlsError> {
    use std::future::poll_fn;
    let n = poll_fn(|cx| {
        let spare = inner.pending_read.spare();
        Pin::new(&mut inner.stream).poll_read(cx, spare)
    })
    .await
    .map_err(TlsError::Io)?;
    inner.pending_read.filled(n);
    Ok(n)
}

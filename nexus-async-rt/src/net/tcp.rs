//! Async TCP stream, listener, and pre-bind socket configuration.
//!
//! Wraps mio's TCP types with the runtime's IO driver for readiness-based
//! async IO. Sockets register with mio lazily on first poll — the task
//! pointer comes from the `Context`'s waker.
//!
//! # Split
//!
//! [`TcpStream::split`] borrows the stream into separate read/write halves
//! for concurrent IO within a single task. [`TcpStream::into_split`]
//! consumes the stream into owned halves that can be moved to different
//! tasks.

use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, RawFd};
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use mio::{Interest, Token};

use super::{AsyncRead, AsyncWrite, waker_to_ptr};
use crate::io::IoHandle;

// =============================================================================
// TcpStream
// =============================================================================

/// Async TCP stream backed by mio.
///
/// Created via [`TcpListener::accept`], [`TcpStream::connect`], or
/// [`TcpSocket::connect`]. Implements [`AsyncRead`] and [`AsyncWrite`].
///
/// The stream registers with mio lazily on the first read or write.
/// Uses edge-triggered epoll — registration happens once and persists.
pub struct TcpStream {
    inner: mio::net::TcpStream,
    io: IoHandle,
    token: Option<Token>,
    /// Task pointer from the last registration. Used to detect when the
    /// stream moves to a different task (e.g., via `into_split`) and
    /// reregister with the IO driver to wake the correct task.
    registered_task: *mut u8,
}

impl TcpStream {
    /// Wrap a mio TcpStream. Registration deferred to first poll.
    pub(crate) fn new(inner: mio::net::TcpStream, io: IoHandle) -> Self {
        Self {
            inner,
            io,
            token: None,
            registered_task: std::ptr::null_mut(),
        }
    }

    /// Initiate an async TCP connection to `addr`.
    ///
    /// The connection completes asynchronously. The first read or write
    /// will register with mio and detect when the connection is
    /// established.
    ///
    /// # Panics
    ///
    /// Panics if called outside a [`Runtime::block_on`](crate::Runtime::block_on)
    /// context — fetches the runtime's [`IoHandle`] internally.
    pub fn connect(addr: SocketAddr) -> io::Result<Self> {
        let inner = mio::net::TcpStream::connect(addr)?;
        Ok(Self::new(inner, IoHandle::current()))
    }

    /// Convert from a `std::net::TcpStream`.
    ///
    /// The stream must be set to non-blocking mode before calling this.
    ///
    /// # Panics
    ///
    /// Panics if called outside a runtime context.
    pub fn from_std(stream: std::net::TcpStream) -> io::Result<Self> {
        let inner = mio::net::TcpStream::from_std(stream);
        Ok(Self::new(inner, IoHandle::current()))
    }

    /// Convert into a `std::net::TcpStream`.
    ///
    /// Deregisters from mio. The returned stream is still non-blocking.
    pub fn into_std(mut self) -> io::Result<std::net::TcpStream> {
        if let Some(token) = self.token.take() {
            // SAFETY: IoHandle's raw pointers are valid for the Runtime
            // lifetime (block_on borrows &mut Runtime which outlives all IO).
            let _ = unsafe { self.io.deregister(&mut self.inner, token) };
        }
        let fd = self.inner.as_raw_fd();
        std::mem::forget(self); // skip Drop (already deregistered)
        // SAFETY: fd is valid — we own it via mio::net::TcpStream and
        // just prevented Drop from closing it via mem::forget.
        Ok(unsafe { std::net::TcpStream::from_raw_fd(fd) })
    }

    // =========================================================================
    // Address
    // =========================================================================

    /// Returns the local address of this stream.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    /// Returns the remote address of this stream.
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.inner.peer_addr()
    }

    // =========================================================================
    // Socket options (via socket2)
    // =========================================================================

    /// Helper: get a socket2::Socket reference for option access.
    fn socket_ref(&self) -> socket2::SockRef<'_> {
        socket2::SockRef::from(&self.inner)
    }

    /// Get TCP_NODELAY.
    pub fn nodelay(&self) -> io::Result<bool> {
        self.inner.nodelay()
    }

    /// Set TCP_NODELAY (disable Nagle's algorithm).
    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        self.inner.set_nodelay(nodelay)
    }

    /// Get IP_TTL.
    pub fn ttl(&self) -> io::Result<u32> {
        self.socket_ref().ttl()
    }

    /// Set IP_TTL.
    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        self.socket_ref().set_ttl(ttl)
    }

    /// Get SO_LINGER.
    pub fn linger(&self) -> io::Result<Option<Duration>> {
        self.socket_ref().linger()
    }

    /// Set SO_LINGER.
    pub fn set_linger(&self, duration: Option<Duration>) -> io::Result<()> {
        self.socket_ref().set_linger(duration)
    }

    /// Get SO_KEEPALIVE.
    pub fn keepalive(&self) -> io::Result<bool> {
        self.socket_ref().keepalive()
    }

    /// Set SO_KEEPALIVE.
    pub fn set_keepalive(&self, keepalive: bool) -> io::Result<()> {
        self.socket_ref().set_keepalive(keepalive)
    }

    /// Get SO_SNDBUF.
    pub fn send_buffer_size(&self) -> io::Result<usize> {
        self.socket_ref().send_buffer_size()
    }

    /// Set SO_SNDBUF.
    pub fn set_send_buffer_size(&self, size: usize) -> io::Result<()> {
        self.socket_ref().set_send_buffer_size(size)
    }

    /// Get SO_RCVBUF.
    pub fn recv_buffer_size(&self) -> io::Result<usize> {
        self.socket_ref().recv_buffer_size()
    }

    /// Set SO_RCVBUF.
    pub fn set_recv_buffer_size(&self, size: usize) -> io::Result<()> {
        self.socket_ref().set_recv_buffer_size(size)
    }

    /// Get SO_ERROR and clear it.
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        self.socket_ref().take_error()
    }

    // =========================================================================
    // Non-blocking try methods (no context needed)
    // =========================================================================

    /// Try to read without blocking. Returns `WouldBlock` if not ready.
    pub fn try_read(&self, buf: &mut [u8]) -> io::Result<usize> {
        (&self.inner).read(buf)
    }

    /// Try to write without blocking. Returns `WouldBlock` if not ready.
    pub fn try_write(&self, buf: &[u8]) -> io::Result<usize> {
        (&self.inner).write(buf)
    }

    /// Read without consuming from the buffer (MSG_PEEK).
    pub fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        // SAFETY: u8 and MaybeUninit<u8> have identical layout (size, alignment).
        // Transmuting an initialized &mut [u8] to &mut [MaybeUninit<u8>] is
        // sound — the bytes are initialized, and peek() will read into them.
        let buf = unsafe { &mut *(buf as *mut [u8] as *mut [std::mem::MaybeUninit<u8>]) };
        self.socket_ref().peek(buf)
    }

    // =========================================================================
    // Async convenience methods
    // =========================================================================

    /// Read bytes from the stream. Returns when at least 1 byte is read
    /// or EOF (0 bytes).
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        std::future::poll_fn(|cx| Pin::new(&mut *self).poll_read(cx, buf)).await
    }

    /// Write bytes to the stream. Returns when at least 1 byte is written.
    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        std::future::poll_fn(|cx| Pin::new(&mut *self).poll_write(cx, buf)).await
    }

    /// Write all bytes to the stream.
    pub async fn write_all(&mut self, mut buf: &[u8]) -> io::Result<()> {
        while !buf.is_empty() {
            let n = self.write(buf).await?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write whole buffer",
                ));
            }
            buf = &buf[n..];
        }
        Ok(())
    }

    /// Poll for read readiness without performing IO.
    ///
    /// Returns `Ready(Ok(()))` if the socket has been reported readable
    /// by epoll. Returns `Pending` if not yet ready. Use this for
    /// sans-IO codecs that want to check readiness before feeding bytes.
    pub fn poll_read_ready(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        if let Err(e) = self.ensure_registered(cx) {
            return Poll::Ready(Err(e));
        }
        if let Some(token) = self.token
            && self.io.readiness(token).readable
        {
            return Poll::Ready(Ok(()));
        }
        Poll::Pending
    }

    /// Poll for write readiness without performing IO.
    pub fn poll_write_ready(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        if let Err(e) = self.ensure_registered(cx) {
            return Poll::Ready(Err(e));
        }
        if let Some(token) = self.token
            && self.io.readiness(token).writable
        {
            return Poll::Ready(Ok(()));
        }
        Poll::Pending
    }

    /// Wait until the stream is readable.
    ///
    /// Returns when epoll reports the socket as readable. After this
    /// returns, [`try_read`](Self::try_read) should succeed.
    pub async fn readable(&mut self) -> io::Result<()> {
        std::future::poll_fn(|cx| self.poll_read_ready(cx)).await
    }

    /// Wait until the stream is writable.
    pub async fn writable(&mut self) -> io::Result<()> {
        std::future::poll_fn(|cx| self.poll_write_ready(cx)).await
    }

    // Note: after a successful read or WouldBlock, the readable flag is
    // Correctly implementing them requires tracking readiness state from
    // epoll events (like tokio's internal readiness tracking). Zero-length
    // reads/writes don't reliably probe socket readiness on Linux.
    // Use poll_read/poll_write or try_read/try_write instead.

    // =========================================================================
    // Split
    // =========================================================================

    /// Split into borrowed read and write halves.
    ///
    /// Both halves borrow the stream — they can be used concurrently
    /// within a single task but cannot be moved to different tasks.
    pub fn split(&mut self) -> (ReadHalf<'_>, WriteHalf<'_>) {
        let ptr = std::ptr::from_mut(self);
        (
            ReadHalf {
                stream: ptr,
                _marker: std::marker::PhantomData,
            },
            WriteHalf {
                stream: ptr,
                _marker: std::marker::PhantomData,
            },
        )
    }

    /// Split into owned read and write halves.
    ///
    /// The halves can be moved to different spawned tasks on the same
    /// single-threaded runtime (`!Send` — not across threads). The IO
    /// driver automatically updates the task pointer when a half is
    /// polled from a different task. Use [`OwnedReadHalf::reunite`]
    /// to reassemble the stream.
    pub fn into_split(self) -> (OwnedReadHalf, OwnedWriteHalf) {
        use std::rc::Rc;
        let shared = Rc::new(std::cell::UnsafeCell::new(self));
        (
            OwnedReadHalf {
                stream: Rc::clone(&shared),
            },
            OwnedWriteHalf { stream: shared },
        )
    }

    // =========================================================================
    // Registration (internal)
    // =========================================================================

    /// Ensure registered with mio and the correct task waker.
    ///
    /// First call: registers with mio. Subsequent calls: checks if the
    /// task pointer changed (stream moved to a different task via
    /// `into_split`). If so, updates the IO driver's waker.
    #[inline(always)]
    fn ensure_registered(&mut self, cx: &Context<'_>) -> io::Result<()> {
        let task_ptr = waker_to_ptr(cx);
        if let Some(token) = self.token {
            // Already registered — check if task changed.
            if task_ptr != self.registered_task {
                self.io.set_waker(token, cx.waker().clone());
                self.registered_task = task_ptr;
            }
            return Ok(());
        }
        self.do_register(task_ptr, cx.waker().clone())
    }

    #[cold]
    fn do_register(&mut self, task_ptr: *mut u8, waker: Waker) -> io::Result<()> {
        let interest = Interest::READABLE | Interest::WRITABLE;
        let token = self.io.register(&mut self.inner, interest, waker)?;
        self.token = Some(token);
        self.registered_task = task_ptr;
        Ok(())
    }
}

impl AsyncRead for TcpStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if let Err(e) = this.ensure_registered(cx) {
            return Poll::Ready(Err(e));
        }
        match this.inner.read(buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                // Clear readable — wait for next epoll notification.
                if let Some(token) = this.token {
                    this.io.clear_readable(token);
                }
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

impl AsyncWrite for TcpStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if let Err(e) = this.ensure_registered(cx) {
            return Poll::Ready(Err(e));
        }
        match this.inner.write(buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                if let Some(token) = this.token {
                    this.io.clear_writable(token);
                }
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if let Err(e) = this.ensure_registered(cx) {
            return Poll::Ready(Err(e));
        }
        match this.inner.flush() {
            Ok(()) => Poll::Ready(Ok(())),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                if let Some(token) = this.token {
                    this.io.clear_writable(token);
                }
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        match this.inner.shutdown(std::net::Shutdown::Write) {
            Ok(()) => Poll::Ready(Ok(())),
            Err(e) if e.kind() == io::ErrorKind::NotConnected => Poll::Ready(Ok(())),
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

impl std::fmt::Debug for TcpStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpStream")
            .field("fd", &self.inner.as_raw_fd())
            .field("registered", &self.token.is_some())
            .finish()
    }
}

impl AsFd for TcpStream {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.inner.as_fd()
    }
}

impl AsRawFd for TcpStream {
    fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }
}

impl Drop for TcpStream {
    fn drop(&mut self) {
        if let Some(token) = self.token {
            // SAFETY: IoHandle's raw pointers are valid for the Runtime
            // lifetime. The stream is being dropped, so deregistering
            // from mio is the correct cleanup.
            let _ = unsafe { self.io.deregister(&mut self.inner, token) };
        }
    }
}

// =============================================================================
// ReadHalf / WriteHalf (borrowed split)
// =============================================================================

/// Borrowed read half of a [`TcpStream`].
///
/// Created by [`TcpStream::split`]. Borrows the stream — cannot be moved
/// to a different task. Implements [`AsyncRead`].
pub struct ReadHalf<'a> {
    stream: *mut TcpStream,
    // Tie lifetime to the borrow of the stream.
    _marker: std::marker::PhantomData<&'a mut TcpStream>,
}

// The split constructor actually gives us two raw pointers to the same stream.
// This is safe because ReadHalf only reads and WriteHalf only writes — no
// aliased mutation of the same fields. Single-threaded.
impl ReadHalf<'_> {
    fn stream(&mut self) -> &mut TcpStream {
        // SAFETY: Borrowed from split(), single-threaded. ReadHalf and
        // WriteHalf hold raw pointers to the same TcpStream, but
        // ReadHalf only calls poll_read and WriteHalf only calls
        // poll_write — no aliased mutation of the same mio fields.
        // The PhantomData<&'a mut TcpStream> ties the lifetime.
        unsafe { &mut *self.stream }
    }
}

impl AsyncRead for ReadHalf<'_> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        Pin::new(this.stream()).poll_read(cx, buf)
    }
}

/// Borrowed write half of a [`TcpStream`].
///
/// Created by [`TcpStream::split`]. Borrows the stream — cannot be moved
/// to a different task. Implements [`AsyncWrite`].
pub struct WriteHalf<'a> {
    stream: *mut TcpStream,
    _marker: std::marker::PhantomData<&'a mut TcpStream>,
}

impl WriteHalf<'_> {
    fn stream(&mut self) -> &mut TcpStream {
        // SAFETY: Borrowed from split(), single-threaded. WriteHalf only
        // calls poll_write/poll_flush/poll_shutdown — disjoint from
        // ReadHalf's poll_read. No aliased mutation.
        unsafe { &mut *self.stream }
    }
}

impl AsyncWrite for WriteHalf<'_> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        Pin::new(this.stream()).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        Pin::new(this.stream()).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        Pin::new(this.stream()).poll_shutdown(cx)
    }
}

// =============================================================================
// OwnedReadHalf / OwnedWriteHalf (owned split)
// =============================================================================

/// Owned read half of a [`TcpStream`].
///
/// Created by [`TcpStream::into_split`]. Can be moved to a different task.
pub struct OwnedReadHalf {
    stream: std::rc::Rc<std::cell::UnsafeCell<TcpStream>>,
}

impl OwnedReadHalf {
    /// Reassemble the stream from its halves.
    ///
    /// Returns `Err` if the halves don't belong to the same stream.
    pub fn reunite(self, write: OwnedWriteHalf) -> Result<TcpStream, ReuniteError> {
        if std::rc::Rc::ptr_eq(&self.stream, &write.stream) {
            drop(write);
            let cell = std::rc::Rc::try_unwrap(self.stream).map_err(|_| ReuniteError)?;
            Ok(cell.into_inner())
        } else {
            Err(ReuniteError)
        }
    }

    /// Returns the peer address.
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        // SAFETY: single-threaded runtime. Shared ref to immutable fields
        // (peer_addr reads from the kernel, doesn't mutate TcpStream).
        unsafe { &*self.stream.get() }.peer_addr()
    }

    /// Returns the local address.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        // SAFETY: same as peer_addr — single-threaded, immutable access.
        unsafe { &*self.stream.get() }.local_addr()
    }
}

impl AsyncRead for OwnedReadHalf {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        // SAFETY: single-threaded runtime. OwnedReadHalf is the only
        // half that calls poll_read; OwnedWriteHalf only calls poll_write.
        // No aliased mutation of the same TcpStream fields.
        let stream = unsafe { &mut *self.stream.get() };
        Pin::new(stream).poll_read(cx, buf)
    }
}

/// Owned write half of a [`TcpStream`].
///
/// Created by [`TcpStream::into_split`]. Can be moved to a different task.
pub struct OwnedWriteHalf {
    stream: std::rc::Rc<std::cell::UnsafeCell<TcpStream>>,
}

impl OwnedWriteHalf {
    /// Reassemble the stream from its halves.
    pub fn reunite(self, read: OwnedReadHalf) -> Result<TcpStream, ReuniteError> {
        read.reunite(self)
    }

    /// Returns the peer address.
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        // SAFETY: single-threaded, immutable access to kernel state.
        unsafe { &*self.stream.get() }.peer_addr()
    }

    /// Returns the local address.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        // SAFETY: same as peer_addr — single-threaded, immutable access.
        unsafe { &*self.stream.get() }.local_addr()
    }
}

impl AsyncWrite for OwnedWriteHalf {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        // SAFETY: single-threaded. OwnedWriteHalf is the only half that
        // calls poll_write; OwnedReadHalf only calls poll_read.
        let stream = unsafe { &mut *self.stream.get() };
        Pin::new(stream).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // SAFETY: same as poll_write — single-threaded, write-side only.
        let stream = unsafe { &mut *self.stream.get() };
        Pin::new(stream).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // SAFETY: same as poll_write — single-threaded, write-side only.
        let stream = unsafe { &mut *self.stream.get() };
        Pin::new(stream).poll_shutdown(cx)
    }
}

/// Error returned by [`OwnedReadHalf::reunite`] when the halves don't match.
#[derive(Debug)]
pub struct ReuniteError;

impl std::fmt::Display for ReuniteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "halves do not belong to the same TcpStream")
    }
}

impl std::error::Error for ReuniteError {}

// =============================================================================
// TcpListener
// =============================================================================

/// Async TCP listener backed by mio.
///
/// Bind with [`TcpListener::bind`] or [`TcpSocket::listen`], then call
/// [`accept`](Self::accept) to await incoming connections.
pub struct TcpListener {
    inner: mio::net::TcpListener,
    io: IoHandle,
    token: Option<Token>,
    registered_task: *mut u8,
}

impl TcpListener {
    /// Bind to `addr`. Registration deferred to first `accept` poll.
    ///
    /// # Panics
    ///
    /// Panics if called outside a [`Runtime::block_on`](crate::Runtime::block_on)
    /// context — fetches the runtime's [`IoHandle`] internally.
    pub fn bind(addr: SocketAddr) -> io::Result<Self> {
        let inner = mio::net::TcpListener::bind(addr)?;
        Ok(Self {
            inner,
            io: IoHandle::current(),
            token: None,
            registered_task: std::ptr::null_mut(),
        })
    }

    /// Convert from a `std::net::TcpListener`.
    ///
    /// # Panics
    ///
    /// Panics if called outside a runtime context.
    pub fn from_std(listener: std::net::TcpListener) -> io::Result<Self> {
        let inner = mio::net::TcpListener::from_std(listener);
        Ok(Self {
            inner,
            io: IoHandle::current(),
            token: None,
            registered_task: std::ptr::null_mut(),
        })
    }

    /// Returns the local address this listener is bound to.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    /// Get IP_TTL.
    pub fn ttl(&self) -> io::Result<u32> {
        socket2::SockRef::from(&self.inner).ttl()
    }

    /// Set IP_TTL.
    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        socket2::SockRef::from(&self.inner).set_ttl(ttl)
    }

    /// Accept a new TCP connection.
    pub fn accept(&mut self) -> Accept<'_> {
        Accept { listener: self }
    }

    /// Ensure registered with mio and the correct task waker.
    #[inline(always)]
    fn ensure_registered(&mut self, cx: &Context<'_>) -> io::Result<()> {
        let task_ptr = waker_to_ptr(cx);
        if let Some(token) = self.token {
            if task_ptr != self.registered_task {
                self.io.set_waker(token, cx.waker().clone());
                self.registered_task = task_ptr;
            }
            return Ok(());
        }
        self.do_register(task_ptr, cx.waker().clone())
    }

    #[cold]
    fn do_register(&mut self, task_ptr: *mut u8, waker: Waker) -> io::Result<()> {
        let token = self
            .io
            .register(&mut self.inner, Interest::READABLE, waker)?;
        self.token = Some(token);
        self.registered_task = task_ptr;
        Ok(())
    }
}

impl std::fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpListener")
            .field("fd", &self.inner.as_raw_fd())
            .field("registered", &self.token.is_some())
            .finish()
    }
}

impl AsFd for TcpListener {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.inner.as_fd()
    }
}

impl AsRawFd for TcpListener {
    fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }
}

impl Drop for TcpListener {
    fn drop(&mut self) {
        if let Some(token) = self.token {
            // SAFETY: IoHandle's raw pointers are valid for the Runtime
            // lifetime. The listener is being dropped, so deregistering
            // from mio is the correct cleanup.
            let _ = unsafe { self.io.deregister(&mut self.inner, token) };
        }
    }
}

/// Future returned by [`TcpListener::accept`].
pub struct Accept<'a> {
    listener: &'a mut TcpListener,
}

impl std::future::Future for Accept<'_> {
    type Output = io::Result<(TcpStream, SocketAddr)>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if let Err(e) = this.listener.ensure_registered(cx) {
            return Poll::Ready(Err(e));
        }
        match this.listener.inner.accept() {
            Ok((stream, addr)) => {
                let tcp = TcpStream::new(stream, this.listener.io);
                Poll::Ready(Ok((tcp, addr)))
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Poll::Pending,
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

// =============================================================================
// TcpSocket — pre-bind configuration
// =============================================================================

/// TCP socket builder for configuring options before bind/connect.
///
/// Wraps `socket2::Socket` to provide access to socket options that
/// must be set before binding (SO_REUSEADDR, SO_REUSEPORT, buffer
/// sizes, etc.).
///
/// # Examples
///
/// ```ignore
/// let socket = TcpSocket::new_v4()?;
/// socket.set_reuseaddr(true)?;
/// socket.set_recv_buffer_size(1024 * 1024)?;
/// let listener = socket.listen(1024, io)?;
/// ```
pub struct TcpSocket {
    inner: socket2::Socket,
}

impl TcpSocket {
    /// Create a new IPv4 TCP socket.
    pub fn new_v4() -> io::Result<Self> {
        let inner = socket2::Socket::new(
            socket2::Domain::IPV4,
            socket2::Type::STREAM,
            Some(socket2::Protocol::TCP),
        )?;
        inner.set_nonblocking(true)?;
        Ok(Self { inner })
    }

    /// Create a new IPv6 TCP socket.
    pub fn new_v6() -> io::Result<Self> {
        let inner = socket2::Socket::new(
            socket2::Domain::IPV6,
            socket2::Type::STREAM,
            Some(socket2::Protocol::TCP),
        )?;
        inner.set_nonblocking(true)?;
        Ok(Self { inner })
    }

    // -- Socket options --

    /// Set SO_REUSEADDR.
    pub fn set_reuseaddr(&self, reuseaddr: bool) -> io::Result<()> {
        self.inner.set_reuse_address(reuseaddr)
    }

    /// Get SO_REUSEADDR.
    pub fn reuseaddr(&self) -> io::Result<bool> {
        self.inner.reuse_address()
    }

    /// Set SO_REUSEPORT (Unix only).
    #[cfg(unix)]
    pub fn set_reuseport(&self, reuseport: bool) -> io::Result<()> {
        self.inner.set_reuse_port(reuseport)
    }

    /// Get SO_REUSEPORT (Unix only).
    #[cfg(unix)]
    pub fn reuseport(&self) -> io::Result<bool> {
        self.inner.reuse_port()
    }

    /// Set SO_KEEPALIVE.
    pub fn set_keepalive(&self, keepalive: bool) -> io::Result<()> {
        self.inner.set_keepalive(keepalive)
    }

    /// Get SO_KEEPALIVE.
    pub fn keepalive(&self) -> io::Result<bool> {
        self.inner.keepalive()
    }

    /// Set TCP_NODELAY.
    pub fn set_nodelay(&self, nodelay: bool) -> io::Result<()> {
        self.inner.set_nodelay(nodelay)
    }

    /// Get TCP_NODELAY.
    pub fn nodelay(&self) -> io::Result<bool> {
        self.inner.nodelay()
    }

    /// Set SO_LINGER.
    pub fn set_linger(&self, duration: Option<Duration>) -> io::Result<()> {
        self.inner.set_linger(duration)
    }

    /// Get SO_LINGER.
    pub fn linger(&self) -> io::Result<Option<Duration>> {
        self.inner.linger()
    }

    /// Set SO_SNDBUF.
    pub fn set_send_buffer_size(&self, size: usize) -> io::Result<()> {
        self.inner.set_send_buffer_size(size)
    }

    /// Get SO_SNDBUF.
    pub fn send_buffer_size(&self) -> io::Result<usize> {
        self.inner.send_buffer_size()
    }

    /// Set SO_RCVBUF.
    pub fn set_recv_buffer_size(&self, size: usize) -> io::Result<()> {
        self.inner.set_recv_buffer_size(size)
    }

    /// Get SO_RCVBUF.
    pub fn recv_buffer_size(&self) -> io::Result<usize> {
        self.inner.recv_buffer_size()
    }

    /// Set IP_TTL.
    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        self.inner.set_ttl(ttl)
    }

    /// Get IP_TTL.
    pub fn ttl(&self) -> io::Result<u32> {
        self.inner.ttl()
    }

    // -- Bind, connect, listen --

    /// Bind the socket to `addr`.
    pub fn bind(&self, addr: SocketAddr) -> io::Result<()> {
        self.inner.bind(&addr.into())
    }

    /// Connect to `addr` and return a [`TcpStream`].
    ///
    /// The connection completes asynchronously (non-blocking socket).
    /// The first read or write will detect when the connection is
    /// established.
    ///
    /// # Panics
    ///
    /// Panics if called outside a [`Runtime::block_on`](crate::Runtime::block_on)
    /// context — fetches the runtime's [`IoHandle`] internally.
    pub fn connect(self, addr: SocketAddr) -> io::Result<TcpStream> {
        // Non-blocking connect returns EINPROGRESS/EALREADY — that's
        // normal, not an error. Suppress these.
        match self.inner.connect(&addr.into()) {
            Ok(()) => {}
            Err(e)
                if e.raw_os_error() == Some(libc::EINPROGRESS)
                    || e.raw_os_error() == Some(libc::EALREADY) => {}
            Err(e) => return Err(e),
        }
        let std_stream: std::net::TcpStream = self.inner.into();
        let mio_stream = mio::net::TcpStream::from_std(std_stream);
        Ok(TcpStream::new(mio_stream, IoHandle::current()))
    }

    /// Start listening with the given backlog and return a [`TcpListener`].
    ///
    /// # Panics
    ///
    /// Panics if called outside a runtime context.
    pub fn listen(self, backlog: i32) -> io::Result<TcpListener> {
        self.inner.listen(backlog)?;
        let std_listener: std::net::TcpListener = self.inner.into();
        let mio_listener = mio::net::TcpListener::from_std(std_listener);
        Ok(TcpListener {
            inner: mio_listener,
            io: IoHandle::current(),
            token: None,
            registered_task: std::ptr::null_mut(),
        })
    }
}

impl std::fmt::Debug for TcpSocket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpSocket")
            .field("fd", &self.inner.as_raw_fd())
            .finish()
    }
}

impl AsFd for TcpSocket {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.inner.as_fd()
    }
}

impl AsRawFd for TcpSocket {
    fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(
    unused_must_use,
    clippy::float_cmp,
    dead_code,
    clippy::ref_option,
    clippy::redundant_closure_for_method_calls,
    clippy::let_underscore_future,
    clippy::semicolon_if_nothing_returned
)]
mod tests {
    use super::*;
    use crate::{Runtime, spawn_boxed};
    use nexus_rt::WorldBuilder;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    #[cfg_attr(miri, ignore)] // Requires real TCP sockets — not miri-compatible.
    fn tcp_echo() {
        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let mut rt = Runtime::new(&mut world);

        let done = Rc::new(Cell::new(false));
        let done2 = done.clone();

        rt.block_on(async move {
            let listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind failed");
            let addr = listener.local_addr().unwrap();
            spawn_boxed(async move {
                let mut listener = listener;
                let (mut stream, _peer) = listener.accept().await.unwrap();
                let mut buf = [0u8; 64];
                let n = stream.read(&mut buf).await.unwrap();
                stream.write_all(&buf[..n]).await.unwrap();
            });

            let flag = done2;
            spawn_boxed(async move {
                crate::context::sleep(std::time::Duration::from_millis(10)).await;
                let mut client = TcpStream::connect(addr).unwrap();
                client.write_all(b"hello").await.unwrap();
                let mut buf = [0u8; 64];
                let n = client.read(&mut buf).await.unwrap();
                assert_eq!(&buf[..n], b"hello");
                flag.set(true);
            });

            crate::context::sleep(std::time::Duration::from_millis(500)).await;
        });

        assert!(done.get(), "echo exchange never completed");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Requires real TCP sockets — not miri-compatible.
    fn tcp_socket_builder() {
        let socket = TcpSocket::new_v4().unwrap();
        socket.set_reuseaddr(true).unwrap();
        assert!(socket.reuseaddr().unwrap());
        socket.set_nodelay(true).unwrap();
        assert!(socket.nodelay().unwrap());
        socket.set_send_buffer_size(65536).unwrap();
        // Buffer size may be rounded up by the kernel.
        assert!(socket.send_buffer_size().unwrap() >= 65536);
    }
}

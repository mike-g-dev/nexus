//! Async UDP socket.
//!
//! Wraps `mio::net::UdpSocket` with the runtime's IO driver. Supports
//! both connected (`send`/`recv`) and unconnected (`send_to`/`recv_from`)
//! modes.
//!
//! Registers with mio lazily on first IO attempt.

use std::io;
use std::net::SocketAddr;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, RawFd};
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use mio::{Interest, Token};

use super::waker_to_ptr;
use crate::io::IoHandle;

/// Async UDP socket backed by mio.
///
/// Created via [`UdpSocket::bind`]. Supports both connected and
/// unconnected modes:
///
/// - **Unconnected**: `send_to(buf, addr).await` / `recv_from(buf).await`
/// - **Connected**: call `connect(addr)`, then `send(buf).await` / `recv(buf).await`
pub struct UdpSocket {
    inner: mio::net::UdpSocket,
    io: IoHandle,
    token: Option<Token>,
    registered_task: *mut u8,
}

impl UdpSocket {
    /// Bind to `addr`. Registration deferred to first IO poll.
    ///
    /// # Panics
    ///
    /// Panics if called outside a [`Runtime::block_on`](crate::Runtime::block_on)
    /// context — fetches the runtime's [`IoHandle`] internally.
    pub fn bind(addr: SocketAddr) -> io::Result<Self> {
        let inner = mio::net::UdpSocket::bind(addr)?;
        Ok(Self {
            inner,
            io: IoHandle::current(),
            token: None,
            registered_task: std::ptr::null_mut(),
        })
    }

    /// Returns the local address this socket is bound to.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    /// Connect the socket to a remote address.
    ///
    /// After connecting, use `send`/`recv` instead of `send_to`/`recv_from`.
    /// Filters incoming datagrams to only those from the connected address.
    pub fn connect(&self, addr: SocketAddr) -> io::Result<()> {
        self.inner.connect(addr)
    }

    /// Returns the remote address this socket is connected to, if any.
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.inner.peer_addr()
    }

    // =========================================================================
    // Socket options
    // =========================================================================

    /// Set the value of `SO_BROADCAST`.
    pub fn set_broadcast(&self, on: bool) -> io::Result<()> {
        self.inner.set_broadcast(on)
    }

    /// Get the value of `SO_BROADCAST`.
    pub fn broadcast(&self) -> io::Result<bool> {
        self.inner.broadcast()
    }

    /// Set the value of `IP_MULTICAST_LOOP`.
    pub fn set_multicast_loop_v4(&self, on: bool) -> io::Result<()> {
        self.inner.set_multicast_loop_v4(on)
    }

    /// Get the value of `IP_MULTICAST_LOOP`.
    pub fn multicast_loop_v4(&self) -> io::Result<bool> {
        self.inner.multicast_loop_v4()
    }

    /// Set the value of `IP_MULTICAST_TTL`.
    pub fn set_multicast_ttl_v4(&self, ttl: u32) -> io::Result<()> {
        self.inner.set_multicast_ttl_v4(ttl)
    }

    /// Get the value of `IP_MULTICAST_TTL`.
    pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
        self.inner.multicast_ttl_v4()
    }

    /// Set the value of `IP_TTL`.
    pub fn set_ttl(&self, ttl: u32) -> io::Result<()> {
        self.inner.set_ttl(ttl)
    }

    /// Get the value of `IP_TTL`.
    pub fn ttl(&self) -> io::Result<u32> {
        self.inner.ttl()
    }

    /// Join a multicast group on the given interface.
    pub fn join_multicast_v4(
        &self,
        multiaddr: &std::net::Ipv4Addr,
        interface: &std::net::Ipv4Addr,
    ) -> io::Result<()> {
        self.inner.join_multicast_v4(multiaddr, interface)
    }

    /// Leave a multicast group on the given interface.
    pub fn leave_multicast_v4(
        &self,
        multiaddr: &std::net::Ipv4Addr,
        interface: &std::net::Ipv4Addr,
    ) -> io::Result<()> {
        self.inner.leave_multicast_v4(multiaddr, interface)
    }

    /// Join an IPv6 multicast group.
    pub fn join_multicast_v6(
        &self,
        multiaddr: &std::net::Ipv6Addr,
        interface: u32,
    ) -> io::Result<()> {
        self.inner.join_multicast_v6(multiaddr, interface)
    }

    /// Leave an IPv6 multicast group.
    pub fn leave_multicast_v6(
        &self,
        multiaddr: &std::net::Ipv6Addr,
        interface: u32,
    ) -> io::Result<()> {
        self.inner.leave_multicast_v6(multiaddr, interface)
    }

    /// Get SO_ERROR and clear it.
    pub fn take_error(&self) -> io::Result<Option<io::Error>> {
        socket2::SockRef::from(&self.inner).take_error()
    }

    /// Get SO_SNDBUF.
    pub fn send_buffer_size(&self) -> io::Result<usize> {
        socket2::SockRef::from(&self.inner).send_buffer_size()
    }

    /// Set SO_SNDBUF.
    pub fn set_send_buffer_size(&self, size: usize) -> io::Result<()> {
        socket2::SockRef::from(&self.inner).set_send_buffer_size(size)
    }

    /// Get SO_RCVBUF.
    pub fn recv_buffer_size(&self) -> io::Result<usize> {
        socket2::SockRef::from(&self.inner).recv_buffer_size()
    }

    /// Set SO_RCVBUF.
    pub fn set_recv_buffer_size(&self, size: usize) -> io::Result<()> {
        socket2::SockRef::from(&self.inner).set_recv_buffer_size(size)
    }

    // =========================================================================
    // Conversion
    // =========================================================================

    /// Convert from a `std::net::UdpSocket`.
    ///
    /// The socket must be set to non-blocking mode before calling this.
    ///
    /// # Panics
    ///
    /// Panics if called outside a runtime context.
    pub fn from_std(socket: std::net::UdpSocket) -> io::Result<Self> {
        let inner = mio::net::UdpSocket::from_std(socket);
        Ok(Self {
            inner,
            io: IoHandle::current(),
            token: None,
            registered_task: std::ptr::null_mut(),
        })
    }

    /// Convert into a `std::net::UdpSocket`.
    ///
    /// Deregisters from mio. The returned socket is still non-blocking.
    pub fn into_std(mut self) -> io::Result<std::net::UdpSocket> {
        if let Some(token) = self.token.take() {
            // SAFETY: IoHandle's raw pointers are valid for the Runtime
            // lifetime. Deregistering before conversion.
            let _ = unsafe { self.io.deregister(&mut self.inner, token) };
        }
        let fd = self.inner.as_raw_fd();
        std::mem::forget(self);
        // SAFETY: fd is valid — we own it via mio::net::UdpSocket and
        // prevented Drop from closing it via mem::forget.
        Ok(unsafe { std::net::UdpSocket::from_raw_fd(fd) })
    }

    // =========================================================================
    // Non-blocking try methods (no context needed)
    // =========================================================================

    /// Try to send a datagram without blocking.
    pub fn try_send(&self, buf: &[u8]) -> io::Result<usize> {
        self.inner.send(buf)
    }

    /// Try to receive a datagram without blocking.
    pub fn try_recv(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.recv(buf)
    }

    /// Try to send a datagram to `target` without blocking.
    pub fn try_send_to(&self, buf: &[u8], target: SocketAddr) -> io::Result<usize> {
        self.inner.send_to(buf, target)
    }

    /// Try to receive a datagram without blocking, returning source address.
    pub fn try_recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        self.inner.recv_from(buf)
    }

    // =========================================================================
    // Registration
    // =========================================================================

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
        let interest = Interest::READABLE | Interest::WRITABLE;
        let token = self.io.register(&mut self.inner, interest, waker)?;
        self.token = Some(token);
        self.registered_task = task_ptr;
        Ok(())
    }

    // =========================================================================
    // Unconnected IO (send_to / recv_from)
    // =========================================================================

    /// Poll to send a datagram to `target`.
    pub fn poll_send_to(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
        target: SocketAddr,
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if let Err(e) = this.ensure_registered(cx) {
            return Poll::Ready(Err(e));
        }
        match this.inner.send_to(buf, target) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Poll::Pending,
            Err(e) => Poll::Ready(Err(e)),
        }
    }

    /// Poll to receive a datagram, returning the number of bytes read
    /// and the source address.
    pub fn poll_recv_from(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<(usize, SocketAddr)>> {
        let this = self.get_mut();
        // Ensure registered BEFORE attempting IO to avoid missing
        // edge-triggered events that arrive between WouldBlock and register.
        if let Err(e) = this.ensure_registered(cx) {
            return Poll::Ready(Err(e));
        }
        match this.inner.recv_from(buf) {
            Ok((n, addr)) => Poll::Ready(Ok((n, addr))),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Poll::Pending,
            Err(e) => Poll::Ready(Err(e)),
        }
    }

    /// Send a datagram to `target`. Returns when the data is sent.
    pub async fn send_to(&mut self, buf: &[u8], target: SocketAddr) -> io::Result<usize> {
        std::future::poll_fn(|cx| Pin::new(&mut *self).poll_send_to(cx, buf, target)).await
    }

    /// Receive a datagram. Returns the number of bytes read and the
    /// source address.
    pub async fn recv_from(&mut self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        std::future::poll_fn(|cx| Pin::new(&mut *self).poll_recv_from(cx, buf)).await
    }

    // =========================================================================
    // Connected IO (send / recv)
    // =========================================================================

    /// Poll to send a datagram on a connected socket.
    pub fn poll_send(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if let Err(e) = this.ensure_registered(cx) {
            return Poll::Ready(Err(e));
        }
        match this.inner.send(buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Poll::Pending,
            Err(e) => Poll::Ready(Err(e)),
        }
    }

    /// Poll to receive a datagram on a connected socket.
    pub fn poll_recv(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if let Err(e) = this.ensure_registered(cx) {
            return Poll::Ready(Err(e));
        }
        match this.inner.recv(buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Poll::Pending,
            Err(e) => Poll::Ready(Err(e)),
        }
    }

    /// Send a datagram on a connected socket.
    pub async fn send(&mut self, buf: &[u8]) -> io::Result<usize> {
        std::future::poll_fn(|cx| Pin::new(&mut *self).poll_send(cx, buf)).await
    }

    /// Receive a datagram on a connected socket.
    pub async fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        std::future::poll_fn(|cx| Pin::new(&mut *self).poll_recv(cx, buf)).await
    }

    // =========================================================================
    // Peek
    // =========================================================================

    /// Receive a datagram without removing it from the queue.
    pub async fn peek_from(&mut self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        std::future::poll_fn(|cx| {
            let this = &mut *self;
            if let Err(e) = this.ensure_registered(cx) {
                return Poll::Ready(Err(e));
            }
            match this.inner.peek_from(buf) {
                Ok((n, addr)) => Poll::Ready(Ok((n, addr))),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => Poll::Pending,
                Err(e) => Poll::Ready(Err(e)),
            }
        })
        .await
    }

    /// Peek on a connected socket.
    pub async fn peek(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        std::future::poll_fn(|cx| {
            let this = &mut *self;
            if let Err(e) = this.ensure_registered(cx) {
                return Poll::Ready(Err(e));
            }
            match this.inner.peek(buf) {
                Ok(n) => Poll::Ready(Ok(n)),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => Poll::Pending,
                Err(e) => Poll::Ready(Err(e)),
            }
        })
        .await
    }
}

impl std::fmt::Debug for UdpSocket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UdpSocket")
            .field("fd", &self.inner.as_raw_fd())
            .field("registered", &self.token.is_some())
            .finish()
    }
}

impl AsFd for UdpSocket {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.inner.as_fd()
    }
}

impl AsRawFd for UdpSocket {
    fn as_raw_fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }
}

impl Drop for UdpSocket {
    fn drop(&mut self) {
        if let Some(token) = self.token {
            // SAFETY: IoHandle valid (Runtime lifetime).
            let _ = unsafe { self.io.deregister(&mut self.inner, token) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Runtime, spawn_boxed};
    use nexus_rt::WorldBuilder;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::time::Duration;

    #[test]
    #[cfg_attr(miri, ignore)] // Requires real UDP sockets — not miri-compatible.
    fn udp_send_recv() {
        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let mut rt = Runtime::new(&mut world);

        let done = Rc::new(Cell::new(false));
        let done2 = done.clone();

        rt.block_on(async move {
            let recv_sock = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
            let recv_addr = recv_sock.local_addr().unwrap();
            // Receiver task.
            let flag = done2;
            spawn_boxed(async move {
                let mut sock = recv_sock;
                let mut buf = [0u8; 64];
                let (n, _from) = sock.recv_from(&mut buf).await.unwrap();
                assert_eq!(&buf[..n], b"test");
                flag.set(true);
            });

            // Sender task.
            spawn_boxed(async move {
                crate::context::sleep(Duration::from_millis(10)).await;
                let mut sock = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
                sock.send_to(b"test", recv_addr).await.unwrap();
            });

            crate::context::sleep(Duration::from_millis(500)).await;
        });

        assert!(done.get(), "UDP recv never completed");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Requires real UDP sockets — not miri-compatible.
    fn udp_echo() {
        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let mut rt = Runtime::new(&mut world);

        let done = Rc::new(Cell::new(false));
        let done2 = done.clone();

        rt.block_on(async move {
            let server_sock = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).expect("bind failed");
            let server_addr = server_sock.local_addr().unwrap();

            // Server task: receive one datagram, echo back.
            spawn_boxed(async move {
                let mut server = server_sock;
                let mut buf = [0u8; 64];
                let (n, peer) = server.recv_from(&mut buf).await.unwrap();
                server.send_to(&buf[..n], peer).await.unwrap();
            });

            // Client task: send datagram, receive echo.
            let flag = done2;
            spawn_boxed(async move {
                crate::context::sleep(Duration::from_millis(10)).await;
                let client_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
                let mut client = UdpSocket::bind(client_addr).unwrap();
                client.send_to(b"hello udp", server_addr).await.unwrap();
                let mut buf = [0u8; 64];
                let (n, _from) = client.recv_from(&mut buf).await.unwrap();
                assert_eq!(&buf[..n], b"hello udp");
                flag.set(true);
            });

            crate::context::sleep(Duration::from_millis(500)).await;
        });

        assert!(done.get(), "UDP echo never completed");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Requires real UDP sockets — not miri-compatible.
    fn udp_connected() {
        let wb = WorldBuilder::new();
        let mut world = wb.build();
        let mut rt = Runtime::new(&mut world);

        let done = Rc::new(Cell::new(false));
        let done2 = done.clone();

        rt.block_on(async move {
            let a_sock = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
            let b_sock = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
            let a_addr = a_sock.local_addr().unwrap();
            let b_addr = b_sock.local_addr().unwrap();
            // A sends to B via connected mode.
            spawn_boxed(async move {
                let mut a = a_sock;
                a.connect(b_addr).unwrap();
                a.send(b"connected").await.unwrap();
            });

            let flag = done2;
            spawn_boxed(async move {
                crate::context::sleep(Duration::from_millis(10)).await;
                let mut b = b_sock;
                b.connect(a_addr).unwrap();
                let mut buf = [0u8; 64];
                let n = b.recv(&mut buf).await.unwrap();
                assert_eq!(&buf[..n], b"connected");
                flag.set(true);
            });

            crate::context::sleep(Duration::from_millis(500)).await;
        });

        assert!(done.get(), "UDP connected exchange never completed");
    }
}

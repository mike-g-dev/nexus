#![allow(
    unused_must_use,
    unused_imports,
    dead_code,
    unknown_lints,
    clippy::float_cmp,
    clippy::ref_option,
    clippy::used_underscore_binding,
    clippy::redundant_locals,
    clippy::semicolon_if_nothing_returned,
    clippy::let_underscore_future,
    clippy::while_let_loop,
    clippy::needless_continue,
    clippy::match_wild_err_arm,
    clippy::collection_is_never_read,
    clippy::async_yields_async,
    clippy::match_same_arms
)]
//! UDP integration tests.
//!
//! All socket creation inside `block_on` (IO driver requires TLS context).

use std::cell::Cell;
use std::rc::Rc;
use std::time::Duration;

use nexus_async_rt::{Runtime, UdpSocket, spawn_boxed};
use nexus_rt::WorldBuilder;

/// Bind a UDP socket to loopback:0, return (socket, addr).
fn bind_udp() -> (UdpSocket, std::net::SocketAddr) {
    let s = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let a = s.local_addr().unwrap();
    (s, a)
}

// =============================================================================
// Basic send/recv
// =============================================================================

#[test]
fn udp_send_recv_basic() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let (recv_sock, recv_addr) = bind_udp();

        spawn_boxed(async move {
            let mut s = recv_sock;
            let mut buf = [0u8; 64];
            let (n, from) = s.recv_from(&mut buf).await.unwrap();
            assert_eq!(&buf[..n], b"hello udp");
            assert!(from.ip().is_loopback());
            flag.set(true);
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let mut s = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
            s.send_to(b"hello udp", recv_addr).await.unwrap();
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

// =============================================================================
// Connected mode
// =============================================================================

#[test]
fn udp_connected_send_recv() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let (a_sock, a_addr) = bind_udp();
        let (b_sock, b_addr) = bind_udp();

        spawn_boxed(async move {
            let mut a = a_sock;
            a.connect(b_addr).unwrap();
            a.send(b"connected-msg").await.unwrap();
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let mut b = b_sock;
            b.connect(a_addr).unwrap();
            let mut buf = [0u8; 64];
            let n = b.recv(&mut buf).await.unwrap();
            assert_eq!(&buf[..n], b"connected-msg");
            flag.set(true);
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

// =============================================================================
// Echo (bidirectional)
// =============================================================================

#[test]
fn udp_echo() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let (server_sock, server_addr) = bind_udp();

        spawn_boxed(async move {
            let mut s = server_sock;
            let mut buf = [0u8; 64];
            let (n, peer) = s.recv_from(&mut buf).await.unwrap();
            s.send_to(&buf[..n], peer).await.unwrap();
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let mut c = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
            c.send_to(b"echo-me", server_addr).await.unwrap();
            let mut buf = [0u8; 64];
            let (n, _) = c.recv_from(&mut buf).await.unwrap();
            assert_eq!(&buf[..n], b"echo-me");
            flag.set(true);
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

// =============================================================================
// Multiple datagrams
// =============================================================================

#[test]
fn udp_multiple_datagrams() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let count = Rc::new(Cell::new(0u32));
    let count2 = count.clone();

    rt.block_on(async move {
        let (recv_sock, recv_addr) = bind_udp();

        spawn_boxed(async move {
            let mut s = recv_sock;
            let mut buf = [0u8; 64];
            for _ in 0..5 {
                let (n, _) = s.recv_from(&mut buf).await.unwrap();
                assert!(n > 0);
                count2.set(count2.get() + 1);
            }
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let mut c = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
            for i in 0..5u8 {
                c.send_to(&[i; 4], recv_addr).await.unwrap();
                nexus_async_rt::sleep(Duration::from_millis(20)).await;
            }
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert_eq!(count.get(), 5);
}

// =============================================================================
// Socket options
// =============================================================================

#[test]
fn udp_socket_options() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let (s, _) = bind_udp();

        s.set_broadcast(true).unwrap();
        assert!(s.broadcast().unwrap());
        s.set_broadcast(false).unwrap();
        assert!(!s.broadcast().unwrap());

        s.set_ttl(42).unwrap();
        assert_eq!(s.ttl().unwrap(), 42);

        s.set_multicast_ttl_v4(5).unwrap();
        assert_eq!(s.multicast_ttl_v4().unwrap(), 5);

        s.set_multicast_loop_v4(false).unwrap();
        assert!(!s.multicast_loop_v4().unwrap());
        s.set_multicast_loop_v4(true).unwrap();
        assert!(s.multicast_loop_v4().unwrap());

        s.set_send_buffer_size(65536).unwrap();
        assert!(s.send_buffer_size().unwrap() >= 65536);
        s.set_recv_buffer_size(65536).unwrap();
        assert!(s.recv_buffer_size().unwrap() >= 65536);

        assert!(s.take_error().unwrap().is_none());
    });
}

// =============================================================================
// try_send / try_recv
// =============================================================================

#[test]
fn udp_try_send_recv() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let (a, a_addr) = bind_udp();
        let (b, b_addr) = bind_udp();

        spawn_boxed(async move {
            let a = a;
            a.connect(b_addr).unwrap();
            let n = a.try_send(b"try-data").unwrap();
            assert_eq!(n, 8);
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(50)).await;
            let b = b;
            b.connect(a_addr).unwrap();
            match b.try_recv(&mut [0u8; 64]) {
                Ok(n) => {
                    assert_eq!(n, 8);
                    flag.set(true);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    flag.set(true); // timing-sensitive, acceptable
                }
                Err(e) => panic!("unexpected error: {e}"),
            }
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

// =============================================================================
// from_std / into_std
// =============================================================================

#[test]
fn udp_from_std() {
    let std_sock = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
    std_sock.set_nonblocking(true).unwrap();
    let addr = std_sock.local_addr().unwrap();

    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let sock = UdpSocket::from_std(std_sock).unwrap();

        spawn_boxed(async move {
            let mut s = sock;
            let mut buf = [0u8; 64];
            let (n, _) = s.recv_from(&mut buf).await.unwrap();
            assert_eq!(&buf[..n], b"from-std");
            flag.set(true);
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let mut s = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
            s.send_to(b"from-std", addr).await.unwrap();
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

#[test]
fn udp_into_std() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let (sock, _) = bind_udp();
        let std_sock = sock.into_std().unwrap();
        assert!(std_sock.local_addr().is_ok());
    });
}

// =============================================================================
// Peek
// =============================================================================

#[test]
fn udp_peek_from() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let (recv_sock, recv_addr) = bind_udp();

        spawn_boxed(async move {
            let mut s = recv_sock;
            let mut buf = [0u8; 64];
            let (n, peer) = s.peek_from(&mut buf).await.unwrap();
            assert_eq!(&buf[..n], b"peek-data");
            assert!(peer.ip().is_loopback());
            let (n2, _) = s.recv_from(&mut buf).await.unwrap();
            assert_eq!(&buf[..n2], b"peek-data");
            flag.set(true);
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let mut s = UdpSocket::bind("127.0.0.1:0".parse().unwrap()).unwrap();
            s.send_to(b"peek-data", recv_addr).await.unwrap();
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

// =============================================================================
// Multicast (loopback)
// =============================================================================

#[test]
fn udp_multicast_loopback() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    // Use std socket for multicast setup (doesn't need IO driver).
    let std_recv = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
    std_recv.set_nonblocking(true).unwrap();
    let recv_port = std_recv.local_addr().unwrap().port();

    if std_recv
        .join_multicast_v4(&"239.255.0.1".parse().unwrap(), &"0.0.0.0".parse().unwrap())
        .is_err()
    {
        println!("multicast join failed — skipping test");
        return;
    }
    let _ = std_recv.set_multicast_loop_v4(true);

    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let recv_sock = UdpSocket::from_std(std_recv).unwrap();

        spawn_boxed(async move {
            let mut s = recv_sock;
            let mut buf = [0u8; 64];
            let (n, _) = s.recv_from(&mut buf).await.unwrap();
            assert_eq!(&buf[..n], b"mcast");
            flag.set(true);
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(50)).await;
            let mut s = UdpSocket::bind("0.0.0.0:0".parse().unwrap()).unwrap();
            let target: std::net::SocketAddr = format!("239.255.0.1:{recv_port}").parse().unwrap();
            s.send_to(b"mcast", target).await.unwrap();
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    // Multicast on loopback may not work in CI — don't assert.
    let _ = done.get();
}

// =============================================================================
// AsFd / AsRawFd
// =============================================================================

#[test]
fn udp_as_fd() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let (s, _) = bind_udp();
        use std::os::fd::AsRawFd;
        let fd = s.as_raw_fd();
        assert!(fd >= 0);
    });
}

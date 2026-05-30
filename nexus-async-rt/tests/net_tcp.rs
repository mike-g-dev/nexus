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
//! TCP integration tests.
//!
//! Every test creates its own Runtime + World. All socket creation is
//! inside `block_on` (IO driver requires TLS context).

use std::cell::Cell;
use std::rc::Rc;
use std::time::Duration;

use nexus_async_rt::{Runtime, TcpListener, TcpSocket, TcpStream, spawn_boxed};
use nexus_rt::WorldBuilder;

// =============================================================================
// Basic connectivity
// =============================================================================

#[test]
fn tcp_echo_basic() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = listener.local_addr().unwrap();

        spawn_boxed(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 128];
            let n = s.read(&mut buf).await.unwrap();
            s.write_all(&buf[..n]).await.unwrap();
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let mut c = TcpStream::connect(addr).unwrap();
            c.write_all(b"hello world").await.unwrap();
            let mut buf = [0u8; 128];
            let n = c.read(&mut buf).await.unwrap();
            assert_eq!(&buf[..n], b"hello world");
            flag.set(true);
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

#[test]
fn tcp_multiple_clients() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let count = Rc::new(Cell::new(0u32));
    let count2 = count.clone();

    rt.block_on(async move {
        let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = listener.local_addr().unwrap();

        spawn_boxed(async move {
            for _ in 0..3 {
                let (mut s, _) = listener.accept().await.unwrap();
                let mut buf = [0u8; 64];
                let n = s.read(&mut buf).await.unwrap();
                s.write_all(&buf[..n]).await.unwrap();
                count2.set(count2.get() + 1);
            }
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            for i in 0..3u8 {
                let mut c = TcpStream::connect(addr).unwrap();
                let msg = [b'A' + i; 4];
                c.write_all(&msg).await.unwrap();
                let mut buf = [0u8; 64];
                let n = c.read(&mut buf).await.unwrap();
                assert_eq!(&buf[..n], &msg);
            }
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert_eq!(count.get(), 3);
}

// =============================================================================
// Large transfer
// =============================================================================

#[test]
fn tcp_large_transfer() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();
    let data: Vec<u8> = (0..1_000_000).map(|i| (i % 251) as u8).collect();
    let expected = data.clone();

    rt.block_on(async move {
        let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = listener.local_addr().unwrap();

        spawn_boxed(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let mut received = Vec::new();
            let mut buf = [0u8; 8192];
            loop {
                let n = s.read(&mut buf).await.unwrap();
                if n == 0 {
                    break;
                }
                received.extend_from_slice(&buf[..n]);
            }
            assert_eq!(received.len(), expected.len());
            assert_eq!(received, expected);
            flag.set(true);
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let mut c = TcpStream::connect(addr).unwrap();
            c.write_all(&data).await.unwrap();
        });

        nexus_async_rt::sleep(Duration::from_secs(5)).await;
    });

    assert!(done.get(), "large transfer did not complete");
}

// =============================================================================
// Split
// =============================================================================

#[test]
fn tcp_split_borrowed() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = listener.local_addr().unwrap();

        spawn_boxed(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let (mut rd, mut wr) = s.split();
            let mut buf = [0u8; 64];
            use nexus_async_rt::{AsyncRead, AsyncWrite};
            let n = std::future::poll_fn(|cx| std::pin::Pin::new(&mut rd).poll_read(cx, &mut buf))
                .await
                .unwrap();
            std::future::poll_fn(|cx| std::pin::Pin::new(&mut wr).poll_write(cx, &buf[..n]))
                .await
                .unwrap();
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let mut c = TcpStream::connect(addr).unwrap();
            c.write_all(b"split").await.unwrap();
            let mut buf = [0u8; 64];
            let n = c.read(&mut buf).await.unwrap();
            assert_eq!(&buf[..n], b"split");
            flag.set(true);
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

#[test]
fn tcp_into_split_reunite() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async move {
        let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = listener.local_addr().unwrap();

        spawn_boxed(async move {
            let (s, _) = listener.accept().await.unwrap();
            let (read_half, write_half) = s.into_split();
            let _stream = read_half.reunite(write_half).unwrap();
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let _c = TcpStream::connect(addr).unwrap();
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });
}

// =============================================================================
// Socket options
// =============================================================================

#[test]
fn tcp_socket_options_on_stream() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = listener.local_addr().unwrap();

        spawn_boxed(async move {
            let (s, _) = listener.accept().await.unwrap();
            s.set_nodelay(true).unwrap();
            assert!(s.nodelay().unwrap());
            s.set_keepalive(true).unwrap();
            assert!(s.keepalive().unwrap());
            s.set_ttl(64).unwrap();
            assert_eq!(s.ttl().unwrap(), 64);
            s.set_send_buffer_size(32768).unwrap();
            assert!(s.send_buffer_size().unwrap() >= 32768);
            s.set_linger(Some(Duration::from_secs(5))).unwrap();
            assert!(s.linger().unwrap().is_some());
            assert!(s.take_error().unwrap().is_none());
            flag.set(true);
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let _c = TcpStream::connect(addr).unwrap();
            nexus_async_rt::sleep(Duration::from_millis(100)).await;
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

#[test]
fn tcp_socket_builder_bind_listen() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    let socket = TcpSocket::new_v4().unwrap();
    socket.set_reuseaddr(true).unwrap();
    assert!(socket.reuseaddr().unwrap());
    socket.set_nodelay(true).unwrap();
    socket.set_send_buffer_size(65536).unwrap();
    assert!(socket.send_buffer_size().unwrap() >= 65536);
    socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();

    rt.block_on(async move {
        let mut listener = socket.listen(128).unwrap();
        let addr = listener.local_addr().unwrap();

        spawn_boxed(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 16];
            let n = s.read(&mut buf).await.unwrap();
            assert_eq!(&buf[..n], b"via-socket");
            flag.set(true);
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let mut c = TcpStream::connect(addr).unwrap();
            c.write_all(b"via-socket").await.unwrap();
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

// =============================================================================
// try_read / try_write
// =============================================================================

#[test]
fn tcp_try_read_write() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = listener.local_addr().unwrap();

        spawn_boxed(async move {
            let (s, _) = listener.accept().await.unwrap();
            match s.try_write(b"data") {
                Ok(n) => assert!(n > 0),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => panic!("unexpected error: {e}"),
            }
            flag.set(true);
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let _c = TcpStream::connect(addr).unwrap();
            nexus_async_rt::sleep(Duration::from_millis(100)).await;
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

// =============================================================================
// from_std / into_std
// =============================================================================

#[test]
fn tcp_from_std() {
    let std_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    std_listener.set_nonblocking(true).unwrap();
    let addr = std_listener.local_addr().unwrap();

    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let mut listener = TcpListener::from_std(std_listener).unwrap();

        spawn_boxed(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 16];
            let n = s.read(&mut buf).await.unwrap();
            assert_eq!(&buf[..n], b"from_std");
            flag.set(true);
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let mut c = TcpStream::connect(addr).unwrap();
            c.write_all(b"from_std").await.unwrap();
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

#[test]
fn tcp_into_std() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = listener.local_addr().unwrap();

        spawn_boxed(async move {
            let (s, _) = listener.accept().await.unwrap();
            let std_stream = s.into_std().unwrap();
            assert!(std_stream.peer_addr().is_ok());
            flag.set(true);
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let _c = TcpStream::connect(addr).unwrap();
            nexus_async_rt::sleep(Duration::from_millis(100)).await;
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

// =============================================================================
// Error paths
// =============================================================================

#[test]
fn tcp_connect_refused() {
    let tmp = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let closed_addr = tmp.local_addr().unwrap();
    drop(tmp);

    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        spawn_boxed(async move {
            match TcpStream::connect(closed_addr) {
                Err(_) => flag.set(true),
                Ok(mut c) => {
                    nexus_async_rt::sleep(Duration::from_millis(50)).await;
                    let result = c.write(b"test").await;
                    assert!(result.is_err(), "expected connection refused");
                    flag.set(true);
                }
            }
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

#[test]
fn tcp_read_after_peer_close() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);
    let done = Rc::new(Cell::new(false));
    let flag = done.clone();

    rt.block_on(async move {
        let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = listener.local_addr().unwrap();

        spawn_boxed(async move {
            let (_s, _) = listener.accept().await.unwrap();
        });

        spawn_boxed(async move {
            nexus_async_rt::sleep(Duration::from_millis(10)).await;
            let mut c = TcpStream::connect(addr).unwrap();
            nexus_async_rt::sleep(Duration::from_millis(50)).await;
            let mut buf = [0u8; 64];
            let n = c.read(&mut buf).await.unwrap();
            assert_eq!(n, 0, "expected EOF");
            flag.set(true);
        });

        nexus_async_rt::sleep(Duration::from_secs(2)).await;
    });

    assert!(done.get());
}

// =============================================================================
// Listener TTL
// =============================================================================

#[test]
fn tcp_listener_ttl() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        listener.set_ttl(42).unwrap();
        assert_eq!(listener.ttl().unwrap(), 42);
    });
}

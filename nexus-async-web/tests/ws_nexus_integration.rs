//! Integration tests for the nexus-async-rt WebSocket backend.
//!
//! Real TCP loopback — server and client run as spawned tasks in
//! the same single-threaded executor.
//!
//! Run: `cargo test -p nexus-async-web --no-default-features --features nexus --test ws_nexus_integration`

#![cfg(feature = "nexus")]

use nexus_async_rt::{Runtime, TcpListener, TcpStream, spawn_boxed};
use nexus_async_web::NexusAsyncReadAdapter;
use nexus_async_web::ws::WsStreamBuilder;
use nexus_rt::WorldBuilder;
use nexus_web::ws::{CloseCode, Message};

use std::net::SocketAddr;

// =============================================================================
// Text echo over TCP loopback
// =============================================================================

#[test]
fn text_echo_loopback() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let mut listener = TcpListener::bind(addr).unwrap();
        let local_addr = listener.local_addr().unwrap();

        let server = spawn_boxed(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
                .accept(NexusAsyncReadAdapter::new(tcp))
                .await
                .unwrap();
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Text(s) => assert_eq!(s, "hello from client"),
                other => panic!("server expected Text, got {other:?}"),
            }
            writer
                .send_text(&mut conn, "hello from server")
                .await
                .unwrap();
        });

        let tcp = TcpStream::connect(local_addr).unwrap();
        let url = format!("ws://127.0.0.1:{}/ws", local_addr.port());
        let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
            .connect_with(NexusAsyncReadAdapter::new(tcp), &url)
            .await
            .unwrap();

        writer
            .send_text(&mut conn, "hello from client")
            .await
            .unwrap();

        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Text(s) => assert_eq!(s, "hello from server"),
            other => panic!("client expected Text, got {other:?}"),
        }

        server.await;
    });
}

// =============================================================================
// Binary roundtrip
// =============================================================================

#[test]
fn binary_roundtrip_loopback() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let mut listener = TcpListener::bind(addr).unwrap();
        let local_addr = listener.local_addr().unwrap();

        let server = spawn_boxed(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
                .accept(NexusAsyncReadAdapter::new(tcp))
                .await
                .unwrap();
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Binary(b) => {
                    assert_eq!(b.len(), 256);
                    assert_eq!(b[0], 0xAB);
                }
                other => panic!("server expected Binary, got {other:?}"),
            }
            writer.send_binary(&mut conn, &[0xCD; 128]).await.unwrap();
        });

        let tcp = TcpStream::connect(local_addr).unwrap();
        let url = format!("ws://127.0.0.1:{}/ws", local_addr.port());
        let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
            .connect_with(NexusAsyncReadAdapter::new(tcp), &url)
            .await
            .unwrap();

        writer.send_binary(&mut conn, &[0xAB; 256]).await.unwrap();

        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Binary(b) => {
                assert_eq!(b.len(), 128);
                assert_eq!(b[0], 0xCD);
            }
            other => panic!("client expected Binary, got {other:?}"),
        }

        server.await;
    });
}

// =============================================================================
// Ping / pong
// =============================================================================

#[test]
fn ping_pong_loopback() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let mut listener = TcpListener::bind(addr).unwrap();
        let local_addr = listener.local_addr().unwrap();

        let server = spawn_boxed(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
                .accept(NexusAsyncReadAdapter::new(tcp))
                .await
                .unwrap();
            let ping_data = match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Ping(p) => {
                    assert_eq!(p, b"heartbeat");
                    p.to_vec()
                }
                other => panic!("server expected Ping, got {other:?}"),
            };
            writer.send_pong(&mut conn, &ping_data).await.unwrap();
        });

        let tcp = TcpStream::connect(local_addr).unwrap();
        let url = format!("ws://127.0.0.1:{}/ws", local_addr.port());
        let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
            .connect_with(NexusAsyncReadAdapter::new(tcp), &url)
            .await
            .unwrap();

        writer.send_ping(&mut conn, b"heartbeat").await.unwrap();

        match reader.recv(&mut conn).await.unwrap().unwrap() {
            Message::Pong(p) => assert_eq!(p, b"heartbeat"),
            other => panic!("client expected Pong, got {other:?}"),
        }

        server.await;
    });
}

// =============================================================================
// Close handshake
// =============================================================================

#[test]
fn close_handshake_loopback() {
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let mut listener = TcpListener::bind(addr).unwrap();
        let local_addr = listener.local_addr().unwrap();

        let server = spawn_boxed(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let (mut reader, _writer, mut conn) = WsStreamBuilder::new()
                .accept(NexusAsyncReadAdapter::new(tcp))
                .await
                .unwrap();
            match reader.recv(&mut conn).await.unwrap().unwrap() {
                Message::Close(cf) => {
                    assert_eq!(cf.code, CloseCode::Normal);
                    assert_eq!(cf.reason, "done");
                }
                other => panic!("server expected Close, got {other:?}"),
            }
        });

        let tcp = TcpStream::connect(local_addr).unwrap();
        let url = format!("ws://127.0.0.1:{}/ws", local_addr.port());
        let (_reader, mut writer, mut conn) = WsStreamBuilder::new()
            .connect_with(NexusAsyncReadAdapter::new(tcp), &url)
            .await
            .unwrap();

        writer
            .close(&mut conn, CloseCode::Normal, "done")
            .await
            .unwrap();

        server.await;
    });
}

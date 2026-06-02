//! Hermetic TLS backpressure tests for the **nexus-async-rt
//! backend**. Mirrors `nexus-net/tests/tls_stream_async_backpressure.rs`
//! but drives `MaybeTls`'s `poll_read` / `poll_write` / `poll_shutdown`
//! through `nexus_async_rt::Runtime` instead of tokio.
//!
//! Three scenarios:
//!
//! - `maybe_tls_handles_oversize_app_data_burst` — server pushes
//!   256 KiB to client, client receives it all via `recv()`.
//!   Exercises `MaybeTls::poll_read` over a large steady-state burst.
//! - `maybe_tls_handles_large_write_via_chunking` — client sends
//!   256 KiB to server. Exercises `try_encrypt`'s chunking path
//!   in `MaybeTls::poll_write` (256 KiB > rustls's 64 KiB plaintext
//!   queue → multiple poll_write cycles).
//! - `maybe_tls_oversize_write_with_tiny_pending_write` — client
//!   sends 32 KiB with `pending_write_cap = TMP_SIZE` (8 KiB),
//!   exercising the legitimate-backpressure exit path of
//!   `drain_codec_to_pending`.
//!
//! Run with:
//! ```text
//! cargo test -p nexus-async-web --no-default-features \
//!     --features nexus,tls --test maybe_tls_nexus_backpressure
//! ```

#![cfg(all(feature = "nexus", feature = "tls"))]

use std::net::TcpListener;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use nexus_async_rt::Runtime;
use nexus_async_web::ws::WsStreamBuilder;
use nexus_net::tls::TlsConfig;
use nexus_rt::WorldBuilder;
use nexus_web::ws::{Client as SyncWsClient, CloseCode, Message};

const PAYLOAD_LEN: usize = 256 * 1024;

// =============================================================================
// Server config (simple self-signed cert; chain-depth tests are covered
// by the existing wss_loopback test).
// =============================================================================

fn make_server_config() -> Arc<rustls::ServerConfig> {
    let cert_kp =
        rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).expect("cert generation");
    let chain = vec![rustls::pki_types::CertificateDer::from(
        cert_kp.cert.der().to_vec(),
    )];
    let key = rustls::pki_types::PrivateKeyDer::try_from(cert_kp.key_pair.serialize_der())
        .expect("server key");
    Arc::new(
        rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(chain, key)
            .expect("server config"),
    )
}

// =============================================================================
// Sync server companions — one per test scenario.
// =============================================================================

/// Server that accepts a WS connection, sends one big binary
/// message, then drains incoming until close.
#[allow(clippy::needless_pass_by_value)]
fn run_burst_send_server(
    listener: TcpListener,
    server_config: Arc<rustls::ServerConfig>,
    payload: Vec<u8>,
) {
    let (tcp, _addr) = listener.accept().expect("server accept");
    tcp.set_nodelay(true).ok();
    tcp.set_read_timeout(Some(Duration::from_secs(15))).ok();
    tcp.set_write_timeout(Some(Duration::from_secs(15))).ok();

    let server_conn = rustls::ServerConnection::new(server_config).expect("server conn");
    let tls_stream = rustls::StreamOwned::new(server_conn, tcp);

    // Need a write_buffer big enough for the whole frame.
    let mut ws = SyncWsClient::builder()
        .write_buffer_capacity(payload.len() + 1024)
        .accept(tls_stream)
        .expect("server WS accept");
    ws.send_binary(&payload).expect("server send burst");

    // Drain until close.
    while let Some(msg) = ws.recv().expect("server recv") {
        if let Message::Close(_) = msg {
            break;
        }
    }
}

/// Server that accepts a WS connection, reads one big binary message,
/// asserts the expected length, then closes.
#[allow(clippy::needless_pass_by_value)]
fn run_burst_recv_server(
    listener: TcpListener,
    server_config: Arc<rustls::ServerConfig>,
    expected_len: usize,
    expected_byte: u8,
) {
    let (tcp, _addr) = listener.accept().expect("server accept");
    tcp.set_nodelay(true).ok();
    tcp.set_read_timeout(Some(Duration::from_secs(15))).ok();
    tcp.set_write_timeout(Some(Duration::from_secs(15))).ok();

    let server_conn = rustls::ServerConnection::new(server_config).expect("server conn");
    let tls_stream = rustls::StreamOwned::new(server_conn, tcp);

    // Reader buffer must hold the whole inbound frame.
    let mut ws = SyncWsClient::builder()
        .buffer_capacity(2 * expected_len.max(65_536))
        .max_frame_size(expected_len as u64 + 1024)
        .max_message_size(expected_len + 1024)
        .accept(tls_stream)
        .expect("server WS accept");
    let msg = ws
        .recv()
        .expect("server recv")
        .expect("server saw EOF before payload");
    match msg {
        Message::Binary(b) => {
            assert_eq!(b.len(), expected_len, "server payload length");
            assert!(
                b.iter().all(|&x| x == expected_byte),
                "server payload bytes intact"
            );
        }
        other => panic!("expected Binary, got {other:?}"),
    }

    // Acknowledge close from client.
    while let Some(msg) = ws.recv().expect("server recv after payload") {
        if let Message::Close(_) = msg {
            break;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[test]
fn maybe_tls_handles_oversize_app_data_burst() {
    let server_config = make_server_config();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("local_addr").port();

    let payload = vec![b'x'; PAYLOAD_LEN];
    let server_handle =
        thread::spawn(move || run_burst_send_server(listener, server_config, payload));

    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async move {
        let tls_config = TlsConfig::builder()
            .danger_no_verify()
            .build()
            .expect("client tls config");

        let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
            .tls(&tls_config)
            .buffer_capacity(2 * PAYLOAD_LEN)
            .max_frame_size(PAYLOAD_LEN as u64 + 1024)
            .max_message_size(PAYLOAD_LEN + 1024)
            .connect_timeout(Duration::from_secs(15))
            .connect(&format!("wss://127.0.0.1:{port}/"))
            .await
            .expect("client wss connect");

        let msg = reader
            .recv(&mut conn)
            .await
            .expect("client recv")
            .expect("server closed early");
        match msg {
            Message::Binary(b) => {
                assert_eq!(b.len(), PAYLOAD_LEN, "client payload length");
                assert!(b.iter().all(|&x| x == b'x'), "client payload intact");
            }
            other => panic!("expected Binary, got {other:?}"),
        }

        writer
            .close(&mut conn, CloseCode::Normal, "done")
            .await
            .expect("client close");
    });

    server_handle.join().expect("server join");
}

#[test]
fn maybe_tls_handles_large_write_via_chunking() {
    let server_config = make_server_config();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("local_addr").port();

    let server_handle =
        thread::spawn(move || run_burst_recv_server(listener, server_config, PAYLOAD_LEN, b'z'));

    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async move {
        let tls_config = TlsConfig::builder()
            .danger_no_verify()
            .build()
            .expect("client tls config");

        // write_buffer_capacity must hold one whole frame; 256 KiB
        // payload + WS header.
        let (mut _reader, mut writer, mut conn) = WsStreamBuilder::new()
            .tls(&tls_config)
            .write_buffer_capacity(PAYLOAD_LEN + 1024)
            .connect_timeout(Duration::from_secs(15))
            .connect(&format!("wss://127.0.0.1:{port}/"))
            .await
            .expect("client wss connect");

        let payload = vec![b'z'; PAYLOAD_LEN];
        writer
            .send_binary(&mut conn, &payload)
            .await
            .expect("client send big");

        writer
            .close(&mut conn, CloseCode::Normal, "done")
            .await
            .expect("client close");
    });

    server_handle.join().expect("server join");
}

#[test]
fn maybe_tls_oversize_write_with_tiny_pending_write() {
    let server_config = make_server_config();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("local_addr").port();

    // 32 KiB stays under rustls's 64 KiB outbound plaintext queue cap
    // (so encrypt itself doesn't error). With pending_write_cap = 8 KiB
    // (= TMP_SIZE), the drain loop in poll_write must iterate ~4 times
    // to flush all the ciphertext.
    let payload_len = 32 * 1024;
    let server_handle =
        thread::spawn(move || run_burst_recv_server(listener, server_config, payload_len, b'y'));

    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async move {
        let tls_config = TlsConfig::builder()
            .danger_no_verify()
            .build()
            .expect("client tls config");

        // 8 KiB pending_write forces the drain-and-refill loop in
        // poll_write to iterate multiple times for the 32 KiB binary
        // frame.
        let capacities = nexus_net::tls::TlsBufferCapacities::builder()
            .pending_write(8 * 1024)
            .build();
        let (mut _reader, mut writer, mut conn) = WsStreamBuilder::new()
            .tls(&tls_config)
            .write_buffer_capacity(payload_len + 1024)
            .tls_buffer_capacities(capacities)
            .connect_timeout(Duration::from_secs(15))
            .connect(&format!("wss://127.0.0.1:{port}/"))
            .await
            .expect("client wss connect");

        let payload = vec![b'y'; payload_len];
        writer
            .send_binary(&mut conn, &payload)
            .await
            .expect("client send");

        writer
            .close(&mut conn, CloseCode::Normal, "done")
            .await
            .expect("client close");
    });

    server_handle.join().expect("server join");
}

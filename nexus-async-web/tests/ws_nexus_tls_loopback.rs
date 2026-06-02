//! Hermetic TLS + WebSocket echo test for the **nexus-async-rt
//! backend**. This is the exact code path that has the issue #200
//! bug — the buggy `read_and_process_tls` calls live in
//! `nexus-async-web/src/ws/nexus/stream.rs:71`,
//! `nexus-async-web/src/maybe_tls/nexus.rs:85`, and
//! `nexus-async-web/src/rest/nexus/connection.rs:71`. Each reads
//! ciphertext into a `tmp` buffer from the async TCP socket and
//! feeds the slice to the codec. Pre-fix, large handshake bursts
//! got truncated to 4096 bytes and the handshake stalled.
//!
//! This test forces the server's first handshake burst over rustls's
//! `READ_SIZE = 4096` per-call cap by using a 10-cert ECDSA-P256
//! chain (chain depth gets the bytes; ECDSA keygen stays fast), then
//! drives a real wss:// connect through the async client. If the
//! helper's loop regresses, this test fails with the same symptom
//! birch reported: `Io(UnexpectedEof, "closed during TLS handshake")`
//! after the server times out.
//!
//! Run with:
//! ```text
//! cargo test -p nexus-async-web --no-default-features \
//!     --features nexus,tls --test ws_nexus_tls_loopback
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

// ============================================================================
// 10-cert ECDSA-P256 chain. Chain depth pushes the TLS 1.3 Certificate
// message past rustls's 4096-byte per-call deframer cap (~5KB of cert
// bytes); ECDSA keygen is microseconds, so the test stays fast.
// ============================================================================

fn generate_oversize_ecdsa_chain() -> (Vec<rustls::pki_types::CertificateDer<'static>>, Vec<u8>) {
    use rcgen::{BasicConstraints, CertificateParams, IsCa, KeyPair};

    const CHAIN_DEPTH: usize = 10;

    let mut keys: Vec<KeyPair> = Vec::with_capacity(CHAIN_DEPTH);
    let mut certs: Vec<rcgen::Certificate> = Vec::with_capacity(CHAIN_DEPTH);

    let root_key = KeyPair::generate().expect("root key");
    let mut root_params = CertificateParams::new(Vec::<String>::new()).expect("root params");
    root_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let root_cert = root_params.self_signed(&root_key).expect("root self-sign");
    keys.push(root_key);
    certs.push(root_cert);

    for _ in 0..(CHAIN_DEPTH - 2) {
        let key = KeyPair::generate().expect("int key");
        let mut params = CertificateParams::new(Vec::<String>::new()).expect("int params");
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let parent_cert = certs.last().expect("parent");
        let parent_key = keys.last().expect("parent key");
        let cert = params
            .signed_by(&key, parent_cert, parent_key)
            .expect("int signed");
        keys.push(key);
        certs.push(cert);
    }

    let leaf_key = KeyPair::generate().expect("leaf key");
    let leaf_params = CertificateParams::new(vec!["localhost".to_string()]).expect("leaf params");
    let parent_cert = certs.last().expect("parent");
    let parent_key = keys.last().expect("parent key");
    let leaf_cert = leaf_params
        .signed_by(&leaf_key, parent_cert, parent_key)
        .expect("leaf signed");

    let mut chain: Vec<rustls::pki_types::CertificateDer<'static>> =
        Vec::with_capacity(CHAIN_DEPTH);
    chain.push(rustls::pki_types::CertificateDer::from(
        leaf_cert.der().to_vec(),
    ));
    for cert in certs.iter().rev() {
        chain.push(rustls::pki_types::CertificateDer::from(cert.der().to_vec()));
    }
    (chain, leaf_key.serialize_der())
}

// ============================================================================
// Sync TLS+WS echo server (runs on a dedicated OS thread). nexus-async-rt
// is single-threaded; can't run both client and server on the same
// runtime when both need to make blocking-style progress on the
// handshake. Loopback TCP bridges the two.
// ============================================================================

// Pass-by-value gives this function ownership so the listener drops
// when the server thread exits — that's the desired teardown.
#[allow(clippy::needless_pass_by_value)]
fn run_echo_server(listener: TcpListener, server_config: Arc<rustls::ServerConfig>) {
    let (tcp, _addr) = listener.accept().expect("server accept");
    tcp.set_nodelay(true).ok();
    tcp.set_read_timeout(Some(Duration::from_secs(15))).ok();
    tcp.set_write_timeout(Some(Duration::from_secs(15))).ok();

    let server_conn = rustls::ServerConnection::new(server_config).expect("server conn");
    let tls_stream = rustls::StreamOwned::new(server_conn, tcp);

    let mut ws = SyncWsClient::accept(tls_stream).expect("server WS accept");
    while let Some(msg) = ws.recv().expect("server recv") {
        match msg {
            Message::Text(s) => {
                let owned = s.to_string();
                ws.send_text(&owned).expect("server send text");
            }
            Message::Binary(b) => {
                let owned = b.to_vec();
                ws.send_binary(&owned).expect("server send binary");
            }
            Message::Ping(payload) => {
                let owned = payload.to_vec();
                ws.send_pong(&owned).expect("server pong");
            }
            Message::Pong(_) => {}
            Message::Close(_) => break,
        }
    }
}

// ============================================================================
// The test — async client over real TCP+TLS+WS, exercising the
// `read_and_process_tls` path in nexus-async-web's nexus backend.
// ============================================================================

#[test]
fn nexus_async_wss_echo_with_oversize_handshake_burst() {
    // Generate ECDSA cert chain up-front (~10ms) before spawning the
    // server thread.
    let (chain, key_der) = generate_oversize_ecdsa_chain();
    let key = rustls::pki_types::PrivateKeyDer::try_from(key_der).expect("server key");
    let server_config = Arc::new(
        rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(chain, key)
            .expect("server config"),
    );

    // Bind on loopback IPv4 (matches the URL host below). Ephemeral port.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
    let local_addr = listener.local_addr().expect("local_addr");
    let port = local_addr.port();

    let server_handle = thread::spawn(move || run_echo_server(listener, server_config));

    // Build nexus-async-rt runtime and drive the client side.
    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    rt.block_on(async move {
        // TLS config — self-signed chain with root not in any system
        // trust store, so verification must be skipped (test-only).
        let tls_config = TlsConfig::builder()
            .danger_no_verify()
            .build()
            .expect("client tls config");

        // `WsStreamBuilder::connect()` constructs the TLS stream
        // internally and goes through `handshake_tls`. That's the
        // async TLS handshake which calls `read_and_process_tls` on
        // each ciphertext chunk read from the async socket — the bug
        // surface for #200.
        let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
            .tls(&tls_config)
            .connect_timeout(Duration::from_secs(15))
            .connect(&format!("wss://127.0.0.1:{port}/"))
            .await
            .expect("client wss connect — handshake byte-loss bug should NOT fire");

        // Text echo round-trip.
        let probe = "hello-from-#200-async-regression-test";
        writer
            .send_text(&mut conn, probe)
            .await
            .expect("client send");
        match reader
            .recv(&mut conn)
            .await
            .expect("client recv")
            .expect("server closed early")
        {
            Message::Text(s) => assert_eq!(s, probe, "echo must match"),
            other => panic!("expected Text, got {other:?}"),
        }

        // Larger payload to keep the data path honest.
        let big = "x".repeat(8192);
        writer
            .send_text(&mut conn, &big)
            .await
            .expect("client send big");
        match reader
            .recv(&mut conn)
            .await
            .expect("client recv big")
            .expect("server closed early")
        {
            Message::Text(s) => assert_eq!(s.len(), 8192, "big echo length must match"),
            other => panic!("expected Text, got {other:?}"),
        }

        writer
            .close(&mut conn, CloseCode::Normal, "done")
            .await
            .expect("client close");
    });

    server_handle.join().expect("server thread join");
}

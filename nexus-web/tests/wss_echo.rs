//! Hermetic TLS + WebSocket echo test (sync `nexus-web` stack).
//!
//! Drives a real TLS handshake + WebSocket upgrade + frame echo
//! between two `nexus_web::ws::Client`s — one acting as server (via
//! `Client::accept`), one as client (via `ClientBuilder::connect`,
//! which handles the TLS + HTTP-upgrade plumbing internally for the
//! `wss://` scheme). Both halves run in the same process; the test
//! connects over loopback TCP. No external network dependencies,
//! runs in `cargo test`.
//!
//! **What this proves:**
//!
//! - The sync TLS + HTTP-upgrade + WebSocket-frame stack works
//!   end-to-end against a real server (not just in-memory codec
//!   tests). The sync `TlsStream::handshake()` path uses
//!   `read_tls_from(&mut socket)` (rustls drives its own loop), so
//!   this test does NOT exercise the issue #200 multi-iteration
//!   surface — that lives in the async paths and is covered by
//!   `nexus-async-web/tests/ws_nexus_tls_loopback.rs`. This test
//!   provides broad regression coverage for the sync stack
//!   (handshake, upgrade, frame I/O).
//!
//! The handshake burst is forced over 4096 bytes by using a 10-cert
//! ECDSA-P256 chain — same shape as the in-process codec test in
//! `tls/codec.rs::tests::read_and_process_tls_handles_oversize_burst`.
//! The chain depth produces > 4KB of cert bytes; ECDSA keygen is
//! microseconds so the test stays fast.
//!
//! **Why localhost, not public servers:** public WSS echo servers are
//! unreliable for tests (geoblocks, Cloudflare bot mitigation,
//! deprecated endpoints, HTTP/2 negotiation). Hermetic localhost
//! tests are deterministic and run in CI without network access.

#![cfg(feature = "tls")]

use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use nexus_web::tls::TlsConfig;
use nexus_web::ws::{Client, CloseCode, Message};

// ============================================================================
// 10-cert ECDSA-P256 chain — matches the in-process codec test. Forces
// the server's first handshake burst past rustls's READ_SIZE = 4096
// via chain depth (~5KB of cert bytes), not key size. ECDSA keygen is
// microseconds vs RSA-4096's seconds, so the test stays fast.
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
// Server side: a tiny TLS-wrapped WebSocket echo. Uses rustls's sync
// `StreamOwned` for the TLS layer + nexus_net's `Client::accept` for
// the WebSocket upgrade and frame echo.
// ============================================================================

// Pass-by-value gives this function ownership so the listener drops
// when the server thread exits — that's the desired teardown.
#[allow(clippy::needless_pass_by_value)]
fn run_echo_server(listener: TcpListener, server_config: std::sync::Arc<rustls::ServerConfig>) {
    let (tcp, _addr) = listener.accept().expect("accept");
    tcp.set_nodelay(true).ok();
    tcp.set_read_timeout(Some(Duration::from_secs(10))).ok();
    tcp.set_write_timeout(Some(Duration::from_secs(10))).ok();

    let server_conn = rustls::ServerConnection::new(server_config).expect("server conn");
    let tls_stream = rustls::StreamOwned::new(server_conn, tcp);

    let mut ws = Client::accept(tls_stream).expect("server WS accept");
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
// Smoke check — basic ECDSA cert, no chain, just verifies the
// listener+thread+rustls+nexus_net stack works at all.
// ============================================================================

fn smoke_check_simple_cert() {
    let cert_kp =
        rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).expect("simple cert");
    let chain = vec![rustls::pki_types::CertificateDer::from(
        cert_kp.cert.der().to_vec(),
    )];
    let key = rustls::pki_types::PrivateKeyDer::try_from(cert_kp.key_pair.serialize_der())
        .expect("simple key");
    let server_config = std::sync::Arc::new(
        rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(chain, key)
            .expect("smoke server config"),
    );

    let listener = TcpListener::bind("127.0.0.1:0").expect("smoke bind");
    let port = listener.local_addr().expect("smoke local_addr").port();
    let server_handle = thread::spawn(move || run_echo_server(listener, server_config));

    let tls_config = TlsConfig::builder()
        .danger_no_verify()
        .build()
        .expect("smoke tls");
    let mut ws = nexus_web::ws::ClientBuilder::new()
        .tls(&tls_config)
        .connect_timeout(Duration::from_secs(10))
        .connect(&format!("wss://127.0.0.1:{port}/"))
        .expect("smoke WSS connect");
    ws.send_text("smoke").expect("smoke send");
    match ws.recv().expect("smoke recv").expect("smoke close") {
        Message::Text(s) => assert_eq!(s, "smoke"),
        other => panic!("smoke: expected Text, got {other:?}"),
    }
    ws.close(CloseCode::Normal, "done").expect("smoke close");
    server_handle.join().expect("smoke server join");
}

// ============================================================================
// The test
// ============================================================================

#[test]
fn local_wss_echo_with_oversize_handshake_burst() {
    // First a smoke check with a simple ECDSA cert to verify the test
    // infrastructure works at all.
    smoke_check_simple_cert();

    // Generate the cert chain up-front (fast: ECDSA chain ~10ms total)
    // so the server thread can signal ready immediately on spawn.
    let (chain, key_der) = generate_oversize_ecdsa_chain();
    let key = rustls::pki_types::PrivateKeyDer::try_from(key_der).expect("server key");
    let server_config = std::sync::Arc::new(
        rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(chain, key)
            .expect("server config"),
    );

    // Bind to an ephemeral port on loopback. OS assigns a free port.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
    let port = listener.local_addr().expect("local_addr").port();

    let server_handle = thread::spawn(move || run_echo_server(listener, server_config));

    // Client side: real TLS + WS upgrade, going through every code
    // path the production tls/stream.rs and ws/stream.rs use.
    //
    // `danger_no_verify` because our chain root is self-signed and not
    // in any system trust store. This is test-only.
    let tls_config = TlsConfig::builder()
        .danger_no_verify()
        .build()
        .expect("client tls config");

    let mut ws = nexus_web::ws::ClientBuilder::new()
        .tls(&tls_config)
        .write_buffer_capacity(64 * 1024)
        .connect_timeout(Duration::from_secs(10))
        .connect(&format!("wss://127.0.0.1:{port}/"))
        .expect("client WSS connect + upgrade");

    // Text echo round-trip.
    let probe = "hello-from-#200-regression-test";
    ws.send_text(probe).expect("client send");
    match ws
        .recv()
        .expect("client recv")
        .expect("close before message")
    {
        Message::Text(s) => assert_eq!(s, probe, "echo must match"),
        other => panic!("expected Text echo, got {other:?}"),
    }

    // Larger payload to keep the data path honest.
    let big = "x".repeat(8192);
    ws.send_text(&big).expect("client send big");
    match ws
        .recv()
        .expect("client recv big")
        .expect("close before message")
    {
        Message::Text(s) => assert_eq!(s.len(), 8192, "big echo length must match"),
        other => panic!("expected Text echo, got {other:?}"),
    }

    ws.close(CloseCode::Normal, "done").expect("client close");
    server_handle.join().expect("server thread");
}

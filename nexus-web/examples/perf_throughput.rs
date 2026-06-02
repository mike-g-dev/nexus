//! Throughput benchmark: nexus-web vs tungstenite.
//!
//! Three benchmark modes:
//! - **Parse-only**: Pure WS framing speed (binary frames).
//! - **Parse + deserialize**: WS text frame → sonic-rs → typed struct.
//! - **Deser-only**: Isolate JSON cost to show the breakdown.
//!
//! Usage:
//!   cargo run --release -p nexus-web --example perf_throughput

use std::hint::black_box;
use std::io::{Cursor, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use serde::Deserialize;

// =============================================================================
// Realistic JSON payloads
// =============================================================================

/// Small quote tick (~90 bytes JSON).
#[derive(Deserialize)]
#[allow(dead_code)]
struct QuoteTick {
    s: String,
    b: f64,
    a: f64,
    bs: f64,
    #[serde(rename = "as")]
    as_: f64,
    t: u64,
}

fn quote_tick_json() -> String {
    r#"{"s":"BTC-USD","b":67234.50,"a":67234.75,"bs":1.5,"as":2.3,"t":1700000000000}"#.to_string()
}

/// Medium order update (~250 bytes JSON).
#[derive(Deserialize)]
#[allow(dead_code)]
struct OrderUpdate {
    s: String,
    bids: Vec<[f64; 2]>,
    asks: Vec<[f64; 2]>,
    t: u64,
    u: u64,
}

fn order_update_json() -> String {
    r#"{"s":"BTC-USD","bids":[[67234.50,1.5],[67234.25,3.2],[67234.00,5.0]],"asks":[[67234.75,2.3],[67235.00,4.1],[67235.25,1.8]],"t":1700000000000,"u":42}"#.to_string()
}

/// Large book snapshot (~1KB JSON).
#[derive(Deserialize)]
#[allow(dead_code)]
struct BookSnapshot {
    s: String,
    bids: Vec<[f64; 2]>,
    asks: Vec<[f64; 2]>,
    t: u64,
    u: u64,
    #[serde(rename = "type")]
    type_: String,
}

fn book_snapshot_json() -> String {
    let mut bids = Vec::new();
    let mut asks = Vec::new();
    for i in 0..20 {
        bids.push(format!(
            "[{:.2},{:.1}]",
            (i as f64).mul_add(-0.25, 67234.50),
            (i as f64).mul_add(0.3, 1.0)
        ));
        asks.push(format!(
            "[{:.2},{:.1}]",
            (i as f64).mul_add(0.25, 67234.75),
            (i as f64).mul_add(0.2, 1.0)
        ));
    }
    format!(
        r#"{{"s":"BTC-USD","bids":[{}],"asks":[{}],"t":1700000000000,"u":42,"type":"snapshot"}}"#,
        bids.join(","),
        asks.join(","),
    )
}

// =============================================================================
// Frame construction
// =============================================================================

/// Build an unmasked text frame (server→client).
fn make_text_frame(payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::new();
    frame.push(0x81); // FIN + Text
    if payload.len() <= 125 {
        frame.push(payload.len() as u8);
    } else if payload.len() <= 65535 {
        frame.push(126);
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    frame.extend_from_slice(payload);
    frame
}

/// Build an unmasked binary frame (server→client).
fn make_binary_frame(payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::new();
    frame.push(0x82); // FIN + Binary
    if payload.len() <= 125 {
        frame.push(payload.len() as u8);
    } else if payload.len() <= 65535 {
        frame.push(126);
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    frame.extend_from_slice(payload);
    frame
}

fn build_text_wire(json: &str, msg_count: u64) -> Vec<u8> {
    let frame = make_text_frame(json.as_bytes());
    let mut wire = Vec::with_capacity(frame.len() * msg_count as usize);
    for _ in 0..msg_count {
        wire.extend_from_slice(&frame);
    }
    wire
}

fn build_binary_wire(payload_size: usize, msg_count: u64) -> Vec<u8> {
    let payload = vec![0x42u8; payload_size];
    let frame = make_binary_frame(&payload);
    let mut wire = Vec::with_capacity(frame.len() * msg_count as usize);
    for _ in 0..msg_count {
        wire.extend_from_slice(&frame);
    }
    wire
}

// =============================================================================
// Cursor wrappers
// =============================================================================

struct CursorWrap<'a>(Cursor<&'a [u8]>);
impl Read for CursorWrap<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}
impl Write for CursorWrap<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct ReadWriteWrap(Cursor<Vec<u8>>);
impl Read for ReadWriteWrap {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}
impl Write for ReadWriteWrap {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// =============================================================================
// Parse-only benchmarks (binary frames, no deser)
// =============================================================================

fn bench_parse_nexus(wire: &[u8], msg_count: u64) -> (Duration, u64) {
    use nexus_web::ws::{Client, FrameReader, FrameWriter, Message, Role};

    let cursor = CursorWrap(Cursor::new(wire));
    let reader = FrameReader::builder()
        .role(Role::Client)
        .buffer_capacity(64 * 1024)
        .build();
    let mut ws = Client::from_parts(cursor, reader, FrameWriter::new(Role::Client));

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.recv().unwrap() {
            Some(Message::Binary(d)) => {
                black_box(&d);
                received += 1;
            }
            Some(_) => {}
            None => break,
        }
    }
    (start.elapsed(), received)
}

fn bench_parse_tungstenite(wire: &[u8], msg_count: u64) -> (Duration, u64) {
    use tungstenite::protocol::{Role, WebSocket, WebSocketConfig};

    let mut config = WebSocketConfig::default();
    config.max_frame_size = Some(64 * 1024 * 1024);
    config.max_message_size = Some(64 * 1024 * 1024);

    let cursor = ReadWriteWrap(Cursor::new(wire.to_vec()));
    let mut ws = WebSocket::from_raw_socket(cursor, Role::Client, Some(config));

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.read() {
            Ok(msg) => {
                black_box(&msg);
                received += 1;
            }
            Err(_) => break,
        }
    }
    (start.elapsed(), received)
}

// =============================================================================
// Parse + deserialize benchmarks (text frames, JSON deser)
// =============================================================================

fn bench_parse_deser_nexus<T: for<'de> Deserialize<'de>>(
    wire: &[u8],
    msg_count: u64,
) -> (Duration, u64) {
    use nexus_web::ws::{Client, FrameReader, FrameWriter, Message, Role};

    let cursor = CursorWrap(Cursor::new(wire));
    let reader = FrameReader::builder()
        .role(Role::Client)
        .buffer_capacity(64 * 1024)
        .build();
    let mut ws = Client::from_parts(cursor, reader, FrameWriter::new(Role::Client));

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.recv().unwrap() {
            Some(Message::Text(s)) => {
                let val: T = sonic_rs::from_str(s).unwrap();
                black_box(&val);
                received += 1;
            }
            Some(_) => {}
            None => break,
        }
    }
    (start.elapsed(), received)
}

fn bench_parse_deser_tungstenite<T: for<'de> Deserialize<'de>>(
    wire: &[u8],
    msg_count: u64,
) -> (Duration, u64) {
    use tungstenite::protocol::{Role, WebSocket, WebSocketConfig};

    let mut config = WebSocketConfig::default();
    config.max_frame_size = Some(64 * 1024 * 1024);
    config.max_message_size = Some(64 * 1024 * 1024);

    let cursor = ReadWriteWrap(Cursor::new(wire.to_vec()));
    let mut ws = WebSocket::from_raw_socket(cursor, Role::Client, Some(config));

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.read() {
            Ok(tungstenite::Message::Text(s)) => {
                let val: T = sonic_rs::from_str(&s).unwrap();
                black_box(&val);
                received += 1;
            }
            Ok(_) => {
                received += 1;
            }
            Err(_) => break,
        }
    }
    (start.elapsed(), received)
}

// =============================================================================
// Deser-only benchmark (isolate JSON cost)
// =============================================================================

fn bench_deser_only<T: for<'de> Deserialize<'de>>(json: &str, msg_count: u64) -> (Duration, u64) {
    let start = Instant::now();
    for _ in 0..msg_count {
        let val: T = sonic_rs::from_str(json).unwrap();
        black_box(&val);
    }
    (start.elapsed(), msg_count)
}

// =============================================================================
// Loopback TCP
// =============================================================================

fn run_loopback(
    port: u16,
    wire: &[u8],
    msg_count: u64,
    client_fn: fn(TcpStream, u64) -> (Duration, u64),
) -> (Duration, u64) {
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr).unwrap();

    let wire_owned = wire.to_vec();
    let server = std::thread::spawn(move || {
        let (tcp, _) = listener.accept().unwrap();
        tcp.set_nodelay(true).unwrap();
        let mut ws = tungstenite::accept(tcp).unwrap();
        let raw_tcp = ws.get_mut();
        raw_tcp.write_all(&wire_owned).unwrap();
        raw_tcp.flush().unwrap();
        std::thread::sleep(Duration::from_secs(5));
    });

    std::thread::sleep(Duration::from_millis(50));
    let tcp = TcpStream::connect(&addr).unwrap();
    tcp.set_nodelay(true).unwrap();
    let result = client_fn(tcp, msg_count);
    let _ = server.join();
    result
}

fn loopback_nexus_client(tcp: TcpStream, msg_count: u64) -> (Duration, u64) {
    use nexus_web::ws::{Client, Message};
    let mut ws = Client::connect_with(tcp, "ws://127.0.0.1/").unwrap();
    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.recv().unwrap() {
            Some(Message::Binary(d)) => {
                black_box(&d);
                received += 1;
            }
            Some(_) => {}
            None => break,
        }
    }
    (start.elapsed(), received)
}

fn loopback_tungstenite_client(tcp: TcpStream, msg_count: u64) -> (Duration, u64) {
    let (mut ws, _) = tungstenite::client::client("ws://127.0.0.1/", tcp).unwrap();
    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.read() {
            Ok(msg) => {
                black_box(&msg);
                received += 1;
            }
            Err(_) => break,
        }
    }
    (start.elapsed(), received)
}

// =============================================================================
// TLS loopback infrastructure
// =============================================================================

/// Run a TLS loopback benchmark. Server does TLS accept + WS accept + blast frames.
fn run_tls_loopback(
    port: u16,
    wire: &[u8],
    msg_count: u64,
    client_fn: impl FnOnce(TcpStream, nexus_web::tls::TlsConfig, u64) -> (Duration, u64),
) -> (Duration, u64) {
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr).unwrap();

    let generated = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
    let cert_der_bytes = generated.cert.der().to_vec();
    let key_der_bytes = generated.key_pair.serialize_der();

    let server_config = std::sync::Arc::new(
        rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(
                vec![rustls::pki_types::CertificateDer::from(
                    cert_der_bytes.clone(),
                )],
                rustls::pki_types::PrivateKeyDer::Pkcs8(
                    rustls::pki_types::PrivatePkcs8KeyDer::from(key_der_bytes),
                ),
            )
            .unwrap(),
    );

    let client_tls_config = nexus_web::tls::TlsConfig::builder()
        .skip_system_certs()
        .add_root_cert(rustls::pki_types::CertificateDer::from(cert_der_bytes))
        .build()
        .unwrap();

    let wire_owned = wire.to_vec();
    let server = std::thread::spawn(move || {
        let (tcp, _) = listener.accept().unwrap();
        tcp.set_nodelay(true).unwrap();

        // TLS accept
        let tls_conn = rustls::ServerConnection::new(server_config).unwrap();
        let tls_stream = rustls::StreamOwned::new(tls_conn, tcp);

        // Use tungstenite for WS accept over the TLS stream
        let mut ws = tungstenite::accept(tls_stream).unwrap();
        let raw = ws.get_mut();

        // Blast raw WS frames through TLS
        let _ = raw.write_all(&wire_owned);
        let _ = raw.flush();
        std::thread::sleep(Duration::from_secs(10));
    });

    std::thread::sleep(Duration::from_millis(100));

    let tcp = TcpStream::connect(&addr).unwrap();
    tcp.set_nodelay(true).unwrap();
    let result = client_fn(tcp, client_tls_config, msg_count);
    let _ = server.join();
    result
}

#[allow(clippy::needless_pass_by_value)] // signature constrained by run_tls_loopback callback
fn tls_loopback_nexus_json_client<T: for<'de> Deserialize<'de>>(
    tcp: TcpStream,
    tls_config: nexus_web::tls::TlsConfig,
    msg_count: u64,
) -> (Duration, u64) {
    use nexus_web::ws::{Client, Message};

    let mut ws = match Client::builder()
        .tls(&tls_config)
        .connect_with(tcp, "wss://localhost/")
    {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("  [nexus TLS connect failed: {e}]");
            return (Duration::from_secs(0), 0);
        }
    };

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.recv() {
            Ok(Some(Message::Text(s))) => {
                let val: T = sonic_rs::from_str(s).unwrap();
                black_box(&val);
                received += 1;
            }
            Ok(Some(_)) => {}
            Ok(None) => {
                if received == 0 {
                    eprintln!("  [nexus: recv returned None after 0 msgs]");
                }
                break;
            }
            Err(e) => {
                eprintln!("  [nexus recv error after {received} msgs: {e}]");
                break;
            }
        }
    }
    (start.elapsed(), received)
}

fn tls_loopback_tungstenite_json_client<T: for<'de> Deserialize<'de>>(
    tcp: TcpStream,
    _tls_config: nexus_web::tls::TlsConfig,
    msg_count: u64,
) -> (Duration, u64) {
    // Set up rustls ClientConnection with no-verify for benchmark
    let config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(std::sync::Arc::new(NoVerifyBench))
        .with_no_client_auth();

    let server_name = rustls::pki_types::ServerName::try_from("localhost")
        .unwrap()
        .to_owned();
    let tls_conn = rustls::ClientConnection::new(std::sync::Arc::new(config), server_name).unwrap();
    let tls_stream = rustls::StreamOwned::new(tls_conn, tcp);

    // tungstenite over the already-established TLS stream
    let (mut ws, _) = tungstenite::client::client("ws://localhost/", tls_stream).unwrap();

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.read() {
            Ok(tungstenite::Message::Text(s)) => {
                let val: T = sonic_rs::from_str(&s).unwrap();
                black_box(&val);
                received += 1;
            }
            Ok(_) => {
                received += 1;
            }
            Err(_) => break,
        }
    }
    (start.elapsed(), received)
}

/// NoVerify for benchmark TLS connections.
#[derive(Debug)]
struct NoVerifyBench;

impl rustls::client::danger::ServerCertVerifier for NoVerifyBench {
    fn verify_server_cert(
        &self,
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &[rustls::pki_types::CertificateDer<'_>],
        _: &rustls::pki_types::ServerName<'_>,
        _: &[u8],
        _: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        _: &[u8],
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn verify_tls13_signature(
        &self,
        _: &[u8],
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::aws_lc_rs::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

// =============================================================================
// Reporting
// =============================================================================

fn report(label: &str, elapsed: Duration, count: u64) {
    let secs = elapsed.as_secs_f64();
    let rate = count as f64 / secs;
    let ns_per = (secs * 1e9) / count as f64;
    let rate_str = if rate >= 1_000_000.0 {
        format!("{:.1}M msg/sec", rate / 1_000_000.0)
    } else {
        format!("{:.0}K msg/sec", rate / 1_000.0)
    };
    println!("  {:<40} {:>12} = {:>7.0}ns/msg", label, rate_str, ns_per);
}

fn section(name: &str) {
    println!("\n  --- {name} ---");
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let n = 1_000_000u64;

    // =========================================================================
    // 1. Parse-only (binary frames, no deser)
    // =========================================================================

    println!("\n  === Parse-Only (binary frames, in-memory) ===");

    for &(size, label) in &[(40, "40B"), (128, "128B"), (512, "512B")] {
        section(label);
        let wire = build_binary_wire(size, n);
        // warmup
        let _ = bench_parse_nexus(&wire, n);
        let _ = bench_parse_tungstenite(&wire, n);

        let (e, c) = bench_parse_nexus(&wire, n);
        report("nexus-web", e, c);
        let (e, c) = bench_parse_tungstenite(&wire, n);
        report("tungstenite", e, c);
    }

    // =========================================================================
    // 2. Parse + deserialize (text frames, JSON)
    // =========================================================================

    println!("\n\n  === Parse + Deserialize (JSON text frames, in-memory) ===");

    // Quote tick (~90B)
    {
        let json = quote_tick_json();
        section(&format!("quote tick ({}B JSON)", json.len()));
        let wire = build_text_wire(&json, n);

        let _ = bench_parse_deser_nexus::<QuoteTick>(&wire, n);
        let _ = bench_parse_deser_tungstenite::<QuoteTick>(&wire, n);

        let (e, c) = bench_parse_deser_nexus::<QuoteTick>(&wire, n);
        report("nexus-web  (parse+deser)", e, c);
        let (e, c) = bench_parse_deser_tungstenite::<QuoteTick>(&wire, n);
        report("tungstenite (parse+deser)", e, c);
        let (e, c) = bench_deser_only::<QuoteTick>(&json, n);
        report("sonic-rs only (deser)", e, c);
    }

    // Order update (~250B)
    {
        let json = order_update_json();
        section(&format!("order update ({}B JSON)", json.len()));
        let wire = build_text_wire(&json, n);

        let _ = bench_parse_deser_nexus::<OrderUpdate>(&wire, n);
        let _ = bench_parse_deser_tungstenite::<OrderUpdate>(&wire, n);

        let (e, c) = bench_parse_deser_nexus::<OrderUpdate>(&wire, n);
        report("nexus-web  (parse+deser)", e, c);
        let (e, c) = bench_parse_deser_tungstenite::<OrderUpdate>(&wire, n);
        report("tungstenite (parse+deser)", e, c);
        let (e, c) = bench_deser_only::<OrderUpdate>(&json, n);
        report("sonic-rs only (deser)", e, c);
    }

    // Book snapshot (~1KB)
    {
        let json = book_snapshot_json();
        let snap_n = 500_000;
        section(&format!("book snapshot ({}B JSON)", json.len()));
        let wire = build_text_wire(&json, snap_n);

        let _ = bench_parse_deser_nexus::<BookSnapshot>(&wire, snap_n);
        let _ = bench_parse_deser_tungstenite::<BookSnapshot>(&wire, snap_n);

        let (e, c) = bench_parse_deser_nexus::<BookSnapshot>(&wire, snap_n);
        report("nexus-web  (parse+deser)", e, c);
        let (e, c) = bench_parse_deser_tungstenite::<BookSnapshot>(&wire, snap_n);
        report("tungstenite (parse+deser)", e, c);
        let (e, c) = bench_deser_only::<BookSnapshot>(&json, snap_n);
        report("sonic-rs only (deser)", e, c);
    }

    // =========================================================================
    // 3. Loopback TCP (binary, real I/O)
    // =========================================================================

    println!("\n\n  === Loopback TCP (binary, real I/O) ===");

    let tcp_n = 500_000u64;
    let wire = build_binary_wire(40, tcp_n);

    section(&format!("40B binary, {tcp_n} msgs"));
    let (e, c) = run_loopback(19100, &wire, tcp_n, loopback_nexus_client);
    report("nexus-web", e, c);
    let (e, c) = run_loopback(19101, &wire, tcp_n, loopback_tungstenite_client);
    report("tungstenite", e, c);

    // =========================================================================
    // 4. TLS loopback + JSON deser (full production stack)
    // =========================================================================

    println!("\n\n  === TLS Loopback + JSON Deserialize (full stack) ===");

    let tls_n = 200_000u64;

    // Quote tick
    {
        let json = quote_tick_json();
        section(&format!("quote tick ({}B JSON) over TLS", json.len()));
        let wire = build_text_wire(&json, tls_n);

        let (e, c) = run_tls_loopback(
            19200,
            &wire,
            tls_n,
            tls_loopback_nexus_json_client::<QuoteTick>,
        );
        report("nexus-web  (TLS+parse+deser)", e, c);
        let (e, c) = run_tls_loopback(
            19201,
            &wire,
            tls_n,
            tls_loopback_tungstenite_json_client::<QuoteTick>,
        );
        report("tungstenite (TLS+parse+deser)", e, c);
        let (e, c) = bench_deser_only::<QuoteTick>(&json, tls_n);
        report("sonic-rs only (deser)", e, c);
    }

    // Order update
    {
        let json = order_update_json();
        section(&format!("order update ({}B JSON) over TLS", json.len()));
        let wire = build_text_wire(&json, tls_n);

        let (e, c) = run_tls_loopback(
            19202,
            &wire,
            tls_n,
            tls_loopback_nexus_json_client::<OrderUpdate>,
        );
        report("nexus-web  (TLS+parse+deser)", e, c);
        let (e, c) = run_tls_loopback(
            19203,
            &wire,
            tls_n,
            tls_loopback_tungstenite_json_client::<OrderUpdate>,
        );
        report("tungstenite (TLS+parse+deser)", e, c);
        let (e, c) = bench_deser_only::<OrderUpdate>(&json, tls_n);
        report("sonic-rs only (deser)", e, c);
    }

    // Book snapshot
    {
        let json = book_snapshot_json();
        let snap_tls_n = 100_000u64;
        section(&format!("book snapshot ({}B JSON) over TLS", json.len()));
        let wire = build_text_wire(&json, snap_tls_n);

        let (e, c) = run_tls_loopback(
            19204,
            &wire,
            snap_tls_n,
            tls_loopback_nexus_json_client::<BookSnapshot>,
        );
        report("nexus-web  (TLS+parse+deser)", e, c);
        let (e, c) = run_tls_loopback(
            19205,
            &wire,
            snap_tls_n,
            tls_loopback_tungstenite_json_client::<BookSnapshot>,
        );
        report("tungstenite (TLS+parse+deser)", e, c);
        let (e, c) = bench_deser_only::<BookSnapshot>(&json, snap_tls_n);
        report("sonic-rs only (deser)", e, c);
    }

    println!();
}

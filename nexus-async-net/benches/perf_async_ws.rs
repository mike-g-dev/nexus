//! Async throughput benchmark: nexus-async-net vs tokio-tungstenite.
//!
//! Three benchmarks:
//! - **In-memory**: Async recv() from a mock stream (isolate parse speed).
//! - **Loopback TCP**: Real tokio TCP, single-threaded runtime.
//! - **Stream trait**: Same as in-memory but using StreamExt (owned messages).
//!
//! All benchmarks use current_thread runtime — closest to a real
//! trading system's single-threaded event loop.
//!
//! Usage:
//!   cargo run --release -p nexus-async-net --example perf_async_ws

use std::hint::black_box;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use nexus_async_net::AsyncReadAdapter;
use serde::Deserialize;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

// =============================================================================
// Frame construction
// =============================================================================

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

fn build_binary_wire(size: usize, count: u64) -> Vec<u8> {
    let payload = vec![0x42u8; size];
    let frame = make_binary_frame(&payload);
    let mut wire = Vec::with_capacity(frame.len() * count as usize);
    for _ in 0..count {
        wire.extend_from_slice(&frame);
    }
    wire
}

fn build_text_wire(json: &str, count: u64) -> Vec<u8> {
    let frame = make_text_frame(json.as_bytes());
    let mut wire = Vec::with_capacity(frame.len() * count as usize);
    for _ in 0..count {
        wire.extend_from_slice(&frame);
    }
    wire
}

// =============================================================================
// Mock async stream
// =============================================================================

/// Async reader backed by a byte slice. Synchronous — always Ready.
struct MockAsyncReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl AsyncRead for MockAsyncReader<'_> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let remaining = &self.data[self.pos..];
        let n = remaining.len().min(buf.remaining());
        buf.put_slice(&remaining[..n]);
        self.pos += n;
        Poll::Ready(Ok(()))
    }
}

impl AsyncWrite for MockAsyncReader<'_> {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Poll::Ready(Ok(buf.len()))
    }
    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

// Same but for tungstenite (needs Read + Write, not AsyncRead + AsyncWrite).
// Wrap in a tokio::io::DuplexStream? No — tokio-tungstenite can work with
// any AsyncRead + AsyncWrite. Let's use the same mock.

// =============================================================================
// In-memory benchmarks
// =============================================================================

async fn bench_inmemory_nexus(wire: &[u8], msg_count: u64) -> (Duration, u64) {
    use nexus_async_net::ws::WsStream;
    use nexus_net::ws::{FrameReader, FrameWriter, Message, Role};

    let mock = AsyncReadAdapter::new(MockAsyncReader { data: wire, pos: 0 });
    let reader = FrameReader::builder()
        .role(Role::Client)
        .buffer_capacity(64 * 1024)
        .build();
    let mut ws = WsStream::from_parts(mock, reader, FrameWriter::new(Role::Client));

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.recv().await.unwrap() {
            Some(Message::Binary(d)) => {
                black_box(&d);
                received += 1;
            }
            Some(Message::Text(s)) => {
                black_box(&s);
                received += 1;
            }
            Some(_) => {}
            None => break,
        }
    }
    (start.elapsed(), received)
}

async fn bench_inmemory_tungstenite(wire: &[u8], msg_count: u64) -> (Duration, u64) {
    use futures_util::StreamExt;

    let mock = MockAsyncReader { data: wire, pos: 0 };
    let ws = tokio_tungstenite::WebSocketStream::from_raw_socket(
        mock,
        tokio_tungstenite::tungstenite::protocol::Role::Client,
        None,
    )
    .await;
    let mut ws = ws;

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.next().await {
            Some(Ok(msg)) => {
                black_box(&msg);
                received += 1;
            }
            _ => break,
        }
    }
    (start.elapsed(), received)
}

// =============================================================================
// JSON payloads
// =============================================================================

#[derive(Deserialize)]
struct QuoteTick {
    #[allow(dead_code)]
    s: String,
    #[allow(dead_code)]
    b: f64,
    #[allow(dead_code)]
    a: f64,
    #[allow(dead_code)]
    bs: f64,
    #[serde(rename = "as")]
    #[allow(dead_code)]
    as_: f64,
    #[allow(dead_code)]
    t: u64,
}

#[derive(Deserialize)]
struct OrderUpdate {
    #[allow(dead_code)]
    s: String,
    #[allow(dead_code)]
    bids: Vec<[f64; 2]>,
    #[allow(dead_code)]
    asks: Vec<[f64; 2]>,
    #[allow(dead_code)]
    t: u64,
    #[allow(dead_code)]
    u: u64,
}

#[derive(Deserialize)]
struct BookSnapshot {
    #[allow(dead_code)]
    s: String,
    #[allow(dead_code)]
    bids: Vec<[f64; 2]>,
    #[allow(dead_code)]
    asks: Vec<[f64; 2]>,
    #[allow(dead_code)]
    t: u64,
    #[allow(dead_code)]
    u: u64,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    type_: String,
}

fn quote_tick_json() -> String {
    r#"{"s":"BTC-USD","b":67234.50,"a":67234.75,"bs":1.5,"as":2.3,"t":1700000000000}"#.to_string()
}

fn order_update_json() -> String {
    r#"{"s":"BTC-USD","bids":[[67234.50,1.5],[67234.25,3.2],[67234.00,5.0]],"asks":[[67234.75,2.3],[67235.00,4.1],[67235.25,1.8]],"t":1700000000000,"u":42}"#.to_string()
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
// JSON parse+deser benchmarks
// =============================================================================

async fn bench_json_nexus<T: for<'de> Deserialize<'de>>(
    wire: &[u8],
    msg_count: u64,
) -> (Duration, u64) {
    use nexus_net::ws::{FrameReader, FrameWriter, Message, Role};

    let mock = AsyncReadAdapter::new(MockAsyncReader { data: wire, pos: 0 });
    let reader = FrameReader::builder()
        .role(Role::Client)
        .buffer_capacity(64 * 1024)
        .build();
    let mut ws =
        nexus_async_net::ws::WsStream::from_parts(mock, reader, FrameWriter::new(Role::Client));

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.recv().await.unwrap() {
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

async fn bench_json_tungstenite<T: for<'de> Deserialize<'de>>(
    wire: &[u8],
    msg_count: u64,
) -> (Duration, u64) {
    use futures_util::StreamExt;

    let mock = MockAsyncReader { data: wire, pos: 0 };
    let ws = tokio_tungstenite::WebSocketStream::from_raw_socket(
        mock,
        tokio_tungstenite::tungstenite::protocol::Role::Client,
        None,
    )
    .await;
    let mut ws = ws;

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.next().await {
            Some(Ok(tokio_tungstenite::tungstenite::Message::Text(s))) => {
                let val: T = sonic_rs::from_str(&s).unwrap();
                black_box(&val);
                received += 1;
            }
            Some(Ok(_)) => {
                received += 1;
            }
            _ => break,
        }
    }
    (start.elapsed(), received)
}

fn bench_deser_only<T: for<'de> Deserialize<'de>>(json: &str, msg_count: u64) -> (Duration, u64) {
    let start = Instant::now();
    for _ in 0..msg_count {
        let val: T = sonic_rs::from_str(json).unwrap();
        black_box(&val);
    }
    (start.elapsed(), msg_count)
}

// =============================================================================
// Stream trait benchmark (nexus — owned messages)
// =============================================================================

async fn bench_stream_nexus(wire: &[u8], msg_count: u64) -> (Duration, u64) {
    use futures_util::StreamExt;
    use nexus_net::ws::{FrameReader, FrameWriter, Role};

    let mock = AsyncReadAdapter::new(MockAsyncReader { data: wire, pos: 0 });
    let reader = FrameReader::builder()
        .role(Role::Client)
        .buffer_capacity(64 * 1024)
        .build();
    let mut ws =
        nexus_async_net::ws::WsStream::from_parts(mock, reader, FrameWriter::new(Role::Client));

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.next().await {
            Some(Ok(msg)) => {
                black_box(&msg);
                received += 1;
            }
            _ => break,
        }
    }
    (start.elapsed(), received)
}

// =============================================================================
// Loopback TCP
// =============================================================================

async fn bench_loopback_nexus(port: u16, wire: Vec<u8>, msg_count: u64) -> (Duration, u64) {
    use nexus_async_net::ws::WsStream;
    use nexus_net::ws::Message;

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}"))
        .await
        .unwrap();

    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        tcp.set_nodelay(true).unwrap();
        // Accept WS handshake using tungstenite (server side)
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        // Get raw stream, blast frames
        let raw = ws.get_mut();
        tokio::io::AsyncWriteExt::write_all(raw, &wire)
            .await
            .unwrap();
        tokio::io::AsyncWriteExt::flush(raw).await.unwrap();
        tokio::time::sleep(Duration::from_secs(5)).await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let tcp = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    tcp.set_nodelay(true).unwrap();
    let mut ws = WsStream::connect_with(
        AsyncReadAdapter::new(tcp),
        &format!("ws://127.0.0.1:{port}/"),
    )
    .await
    .unwrap();

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.recv().await.unwrap() {
            Some(Message::Binary(d)) => {
                black_box(&d);
                received += 1;
            }
            Some(_) => {}
            None => break,
        }
    }
    let elapsed = start.elapsed();

    server.abort();
    (elapsed, received)
}

async fn bench_loopback_tungstenite(port: u16, wire: Vec<u8>, msg_count: u64) -> (Duration, u64) {
    use futures_util::StreamExt;

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}"))
        .await
        .unwrap();

    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        tcp.set_nodelay(true).unwrap();
        let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        let raw = ws.get_mut();
        tokio::io::AsyncWriteExt::write_all(raw, &wire)
            .await
            .unwrap();
        tokio::io::AsyncWriteExt::flush(raw).await.unwrap();
        tokio::time::sleep(Duration::from_secs(5)).await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let tcp = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    tcp.set_nodelay(true).unwrap();
    let (mut ws, _) = tokio_tungstenite::client_async(format!("ws://127.0.0.1:{port}/"), tcp)
        .await
        .unwrap();

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.next().await {
            Some(Ok(msg)) => {
                black_box(&msg);
                received += 1;
            }
            _ => break,
        }
    }
    let elapsed = start.elapsed();

    server.abort();
    (elapsed, received)
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
    println!("  {:<45} {:>12} = {:>7.0}ns/msg", label, rate_str, ns_per);
}

fn section(name: &str) {
    println!("\n  --- {name} ---");
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let n = 1_000_000u64;

        // =================================================================
        // 1. In-memory parse (binary, recv() — zero-copy)
        // =================================================================

        println!("\n  === In-Memory Parse (binary, async recv — zero-copy) ===");

        for &(size, label) in &[(40, "40B"), (128, "128B"), (512, "512B")] {
            section(label);
            let wire = build_binary_wire(size, n);

            // warmup
            let _ = bench_inmemory_nexus(&wire, n).await;
            let _ = bench_inmemory_tungstenite(&wire, n).await;

            let (e, c) = bench_inmemory_nexus(&wire, n).await;
            report("nexus-async-net (recv)", e, c);
            let (e, c) = bench_inmemory_tungstenite(&wire, n).await;
            report("tokio-tungstenite (next)", e, c);
        }

        // =================================================================
        // 2. In-memory parse (binary, Stream trait — owned)
        // =================================================================

        println!("\n\n  === In-Memory Parse (binary, Stream trait — owned) ===");

        for &(size, label) in &[(40, "40B"), (128, "128B")] {
            section(label);
            let wire = build_binary_wire(size, n);

            let _ = bench_stream_nexus(&wire, n).await;

            let (e, c) = bench_stream_nexus(&wire, n).await;
            report("nexus-async-net (Stream)", e, c);
            let (e, c) = bench_inmemory_tungstenite(&wire, n).await;
            report("tokio-tungstenite (Stream)", e, c);
        }

        // =================================================================
        // 3. JSON parse+deser (text frames, sonic-rs)
        // =================================================================

        println!("\n\n  === JSON Parse + Deserialize (text frames, async) ===");

        {
            let json = quote_tick_json();
            section(&format!("quote tick ({}B JSON)", json.len()));
            let wire = build_text_wire(&json, n);

            let _ = bench_json_nexus::<QuoteTick>(&wire, n).await;
            let _ = bench_json_tungstenite::<QuoteTick>(&wire, n).await;

            let (e, c) = bench_json_nexus::<QuoteTick>(&wire, n).await;
            report("nexus (recv+deser)", e, c);
            let (e, c) = bench_json_tungstenite::<QuoteTick>(&wire, n).await;
            report("tungstenite (next+deser)", e, c);
            let (e, c) = bench_deser_only::<QuoteTick>(&json, n);
            report("sonic-rs only", e, c);
        }

        {
            let json = order_update_json();
            section(&format!("order update ({}B JSON)", json.len()));
            let wire = build_text_wire(&json, n);

            let _ = bench_json_nexus::<OrderUpdate>(&wire, n).await;
            let _ = bench_json_tungstenite::<OrderUpdate>(&wire, n).await;

            let (e, c) = bench_json_nexus::<OrderUpdate>(&wire, n).await;
            report("nexus (recv+deser)", e, c);
            let (e, c) = bench_json_tungstenite::<OrderUpdate>(&wire, n).await;
            report("tungstenite (next+deser)", e, c);
            let (e, c) = bench_deser_only::<OrderUpdate>(&json, n);
            report("sonic-rs only", e, c);
        }

        {
            let json = book_snapshot_json();
            let snap_n = 500_000;
            section(&format!("book snapshot ({}B JSON)", json.len()));
            let wire = build_text_wire(&json, snap_n);

            let _ = bench_json_nexus::<BookSnapshot>(&wire, snap_n).await;
            let _ = bench_json_tungstenite::<BookSnapshot>(&wire, snap_n).await;

            let (e, c) = bench_json_nexus::<BookSnapshot>(&wire, snap_n).await;
            report("nexus (recv+deser)", e, c);
            let (e, c) = bench_json_tungstenite::<BookSnapshot>(&wire, snap_n).await;
            report("tungstenite (next+deser)", e, c);
            let (e, c) = bench_deser_only::<BookSnapshot>(&json, snap_n);
            report("sonic-rs only", e, c);
        }

        // =================================================================
        // 4. Loopback TCP (binary, real I/O)
        // =================================================================

        println!("\n\n  === Loopback TCP (binary, single-threaded tokio) ===");

        let tcp_n = 500_000u64;
        let wire = build_binary_wire(40, tcp_n);

        section(&format!("40B binary, {tcp_n} msgs"));
        let (e, c) = bench_loopback_nexus(19300, wire.clone(), tcp_n).await;
        report("nexus-async-net (recv)", e, c);
        let (e, c) = bench_loopback_tungstenite(19301, wire, tcp_n).await;
        report("tokio-tungstenite (next)", e, c);

        // =================================================================
        // 5. Blocking vs Async — in-memory (no syscalls)
        // =================================================================

        println!("\n\n  === Blocking vs Async — In-Memory (no syscalls) ===");

        for &(size, label) in &[(40, "40B"), (128, "128B")] {
            section(label);
            let wire = build_binary_wire(size, n);

            let (e, c) = bench_inmemory_nexus(&wire, n).await;
            report("nexus async recv()", e, c);

            let (e, c) = bench_blocking_nexus(&wire, n);
            report("nexus blocking recv()", e, c);
        }

        // =================================================================
        // 6. Blocking vs Async — TCP loopback (real syscalls)
        // =================================================================

        println!("\n\n  === Blocking vs Async — TCP Loopback (real syscalls) ===");

        let tcp_n = 500_000u64;

        for &(size, label) in &[(40, "40B"), (128, "128B")] {
            section(label);
            let wire = build_binary_wire(size, tcp_n);

            let port_async = 19400 + size as u16;
            let port_tung = 19450 + size as u16;
            let port_block = 19500 + size as u16;

            let (e, c) = bench_loopback_nexus(port_async, wire.clone(), tcp_n).await;
            report("nexus-async-net (tokio)", e, c);

            let (e, c) = bench_loopback_tungstenite(port_tung, wire.clone(), tcp_n).await;
            report("tokio-tungstenite", e, c);

            let (e, c) = bench_loopback_blocking(port_block, wire, tcp_n);
            report("nexus-net (blocking)", e, c);
        }

        // =================================================================
        // 7. TLS loopback — all three
        // =================================================================

        println!("\n\n  === TLS Loopback — All Three ===");

        let tls_n = 200_000u64;

        for &(size, label) in &[(40, "40B"), (128, "128B")] {
            section(label);
            let wire = build_binary_wire(size, tls_n);

            let port_async = 19600 + size as u16;
            let port_tung = 19650 + size as u16;
            let port_block = 19700 + size as u16;

            let (e, c) = bench_tls_loopback_nexus(port_async, wire.clone(), tls_n).await;
            report("nexus-async-net (tokio+TLS)", e, c);

            let (e, c) = bench_tls_loopback_tungstenite(port_tung, wire.clone(), tls_n).await;
            report("tokio-tungstenite (+TLS)", e, c);

            let (e, c) = bench_tls_loopback_blocking(port_block, wire, tls_n);
            report("nexus-net (blocking+TLS)", e, c);
        }

        println!();
    });
}

fn bench_loopback_blocking(port: u16, wire: Vec<u8>, msg_count: u64) -> (Duration, u64) {
    use nexus_net::ws::{Client, Message};
    use std::net::{TcpListener, TcpStream};

    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr).unwrap();

    let server = std::thread::spawn(move || {
        let (tcp, _) = listener.accept().unwrap();
        tcp.set_nodelay(true).unwrap();
        // Use tungstenite for server-side accept
        let mut ws = tokio_tungstenite::tungstenite::accept(tcp).unwrap();
        let raw = ws.get_mut();
        std::io::Write::write_all(raw, &wire).unwrap();
        std::io::Write::flush(raw).unwrap();
        std::thread::sleep(Duration::from_secs(5));
    });

    std::thread::sleep(Duration::from_millis(50));

    let tcp = TcpStream::connect(&addr).unwrap();
    tcp.set_nodelay(true).unwrap();
    let mut ws = Client::connect_with(tcp, &format!("ws://{addr}/")).unwrap();

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
    let elapsed = start.elapsed();
    let _ = server.join();
    (elapsed, received)
}

// =============================================================================
// TLS loopback helpers
// =============================================================================

fn make_tls_server_config() -> std::sync::Arc<rustls::ServerConfig> {
    let generated = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
    let cert_der = generated.cert.der().to_vec();
    let key_der = generated.key_pair.serialize_der();
    std::sync::Arc::new(
        rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(
                vec![rustls::pki_types::CertificateDer::from(cert_der)],
                rustls::pki_types::PrivateKeyDer::Pkcs8(
                    rustls::pki_types::PrivatePkcs8KeyDer::from(key_der),
                ),
            )
            .unwrap(),
    )
}

fn make_no_verify_client_config() -> std::sync::Arc<rustls::ClientConfig> {
    std::sync::Arc::new(
        rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(std::sync::Arc::new(NoVerifyBench))
            .with_no_client_auth(),
    )
}

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

async fn tls_accept_and_blast(
    listener: tokio::net::TcpListener,
    server_config: std::sync::Arc<rustls::ServerConfig>,
    wire: Vec<u8>,
) {
    let (tcp, _) = listener.accept().await.unwrap();
    tcp.set_nodelay(true).unwrap();
    let acceptor = tokio_rustls::TlsAcceptor::from(server_config);
    let tls = acceptor.accept(tcp).await.unwrap();
    // WS accept over TLS, then blast raw frames
    let mut ws = tokio_tungstenite::accept_async(tls).await.unwrap();
    let raw = ws.get_mut();
    tokio::io::AsyncWriteExt::write_all(raw, &wire)
        .await
        .unwrap();
    tokio::io::AsyncWriteExt::flush(raw).await.unwrap();
    tokio::time::sleep(Duration::from_secs(5)).await;
}

async fn bench_tls_loopback_nexus(port: u16, wire: Vec<u8>, msg_count: u64) -> (Duration, u64) {
    use nexus_async_net::ws::WsStream;
    use nexus_net::ws::Message;

    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    let server_config = make_tls_server_config();

    let server = tokio::spawn(tls_accept_and_blast(listener, server_config, wire));
    tokio::time::sleep(Duration::from_millis(50)).await;

    let tcp = tokio::net::TcpStream::connect(&addr).await.unwrap();
    tcp.set_nodelay(true).unwrap();

    let client_config = make_no_verify_client_config();
    let connector = tokio_rustls::TlsConnector::from(client_config);
    let server_name = rustls::pki_types::ServerName::try_from("localhost")
        .unwrap()
        .to_owned();
    let tls_stream = connector.connect(server_name, tcp).await.unwrap();

    let mut ws = WsStream::connect_with(AsyncReadAdapter::new(tls_stream), "ws://localhost/")
        .await
        .unwrap();

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.recv().await.unwrap() {
            Some(Message::Binary(d)) => {
                black_box(&d);
                received += 1;
            }
            Some(_) => {}
            None => break,
        }
    }
    let elapsed = start.elapsed();
    server.abort();
    (elapsed, received)
}

async fn bench_tls_loopback_tungstenite(
    port: u16,
    wire: Vec<u8>,
    msg_count: u64,
) -> (Duration, u64) {
    use futures_util::StreamExt;

    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    let server_config = make_tls_server_config();

    let server = tokio::spawn(tls_accept_and_blast(listener, server_config, wire));
    tokio::time::sleep(Duration::from_millis(50)).await;

    let tcp = tokio::net::TcpStream::connect(&addr).await.unwrap();
    tcp.set_nodelay(true).unwrap();

    let client_config = make_no_verify_client_config();
    let connector = tokio_rustls::TlsConnector::from(client_config);
    let server_name = rustls::pki_types::ServerName::try_from("localhost")
        .unwrap()
        .to_owned();
    let tls_stream = connector.connect(server_name, tcp).await.unwrap();

    let (mut ws, _) = tokio_tungstenite::client_async("ws://localhost/", tls_stream)
        .await
        .unwrap();

    let start = Instant::now();
    let mut received = 0u64;
    while received < msg_count {
        match ws.next().await {
            Some(Ok(msg)) => {
                black_box(&msg);
                received += 1;
            }
            _ => break,
        }
    }
    let elapsed = start.elapsed();
    server.abort();
    (elapsed, received)
}

fn bench_tls_loopback_blocking(port: u16, wire: Vec<u8>, msg_count: u64) -> (Duration, u64) {
    use nexus_net::ws::Message;

    let addr = format!("127.0.0.1:{port}");
    let server_config = make_tls_server_config();

    let server = std::thread::spawn(move || {
        let listener = std::net::TcpListener::bind(&addr).unwrap();
        let (tcp, _) = listener.accept().unwrap();
        tcp.set_nodelay(true).unwrap();
        let tls_conn = rustls::ServerConnection::new(server_config).unwrap();
        let tls_stream = rustls::StreamOwned::new(tls_conn, tcp);
        let mut ws = tokio_tungstenite::tungstenite::accept(tls_stream).unwrap();
        let raw = ws.get_mut();
        std::io::Write::write_all(raw, &wire).unwrap();
        std::io::Write::flush(raw).unwrap();
        std::thread::sleep(Duration::from_secs(5));
    });

    std::thread::sleep(Duration::from_millis(100));

    let tcp = std::net::TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    tcp.set_nodelay(true).unwrap();

    let tls_config = nexus_net::tls::TlsConfig::builder()
        .danger_no_verify()
        .build()
        .unwrap();
    let codec = nexus_net::tls::TlsCodec::new(&tls_config, "localhost").unwrap();
    let tls = nexus_net::tls::TlsStream::connect(tcp, codec).unwrap();

    let mut ws = nexus_net::ws::ClientBuilder::new()
        .connect_with(tls, "wss://localhost/")
        .unwrap();

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
    let elapsed = start.elapsed();
    let _ = server.join();
    (elapsed, received)
}

// =============================================================================
// In-memory blocking comparison
// =============================================================================

fn bench_blocking_nexus(wire: &[u8], msg_count: u64) -> (Duration, u64) {
    use nexus_net::ws::{Client, FrameReader, FrameWriter, Message, Role};
    use std::io::{Cursor, Read, Write};

    struct CursorWrap<'a>(Cursor<&'a [u8]>);
    impl Read for CursorWrap<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.0.read(buf)
        }
    }
    impl Write for CursorWrap<'_> {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

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

//! REST throughput benchmark: max requests/sec on mock stream.
//!
//! Measures the protocol layer throughput — how fast we can build
//! requests and parse responses without real network latency.
//!
//! Usage:
//!   cargo run --release -p nexus-web --example throughput_rest

use std::io::{self, Cursor, Read, Write};
use std::time::Instant;

use nexus_web::http::ResponseReader;
use nexus_web::rest::{Client, RequestWriter};

const CANNED_RESPONSE: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\nX-RateLimit-Remaining: 42\r\n\r\n{\"orderId\":123}";

struct MockStream {
    response: Cursor<&'static [u8]>,
}

impl MockStream {
    fn new() -> Self {
        Self {
            response: Cursor::new(CANNED_RESPONSE),
        }
    }
    fn reset(&mut self) {
        self.response.set_position(0);
    }
}

impl Read for MockStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.response.read(buf)
    }
}

impl Write for MockStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn main() {
    let mut writer = RequestWriter::new("api.binance.com").unwrap();
    writer.set_base_path("/api/v3").unwrap();
    writer
        .default_header(
            "X-MBX-APIKEY",
            "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        )
        .unwrap();
    writer
        .default_header("Content-Type", "application/json")
        .unwrap();

    let mut reader = ResponseReader::new(4096).max_body_size(32 * 1024);
    let mut conn = Client::new(MockStream::new());

    let order_body = br#"{"symbol":"BTCUSDT","side":"BUY","type":"LIMIT","timeInForce":"GTC","quantity":"0.001","price":"67234.50"}"#;
    let timestamp = "1700000000000";
    let signature = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    // Warmup
    for _ in 0..10_000 {
        conn.stream_mut().reset();
        let req = writer
            .post("/order")
            .header("X-MBX-TIMESTAMP", timestamp)
            .header("X-MBX-SIGNATURE", signature)
            .body(order_body)
            .finish()
            .unwrap();
        let resp = conn.send(req, &mut reader).unwrap();
        std::hint::black_box(resp.status());
    }

    // Benchmark
    let iterations = 1_000_000u64;
    let start = Instant::now();

    for _ in 0..iterations {
        conn.stream_mut().reset();
        let req = writer
            .post("/order")
            .header("X-MBX-TIMESTAMP", timestamp)
            .header("X-MBX-SIGNATURE", signature)
            .body(order_body)
            .finish()
            .unwrap();
        let resp = conn.send(req, &mut reader).unwrap();
        std::hint::black_box(resp.status());
    }

    let elapsed = start.elapsed();
    let rps = iterations as f64 / elapsed.as_secs_f64();
    let ns_per_req = elapsed.as_nanos() as f64 / iterations as f64;

    println!("\n  REST throughput benchmark (mock I/O, single-threaded)");
    println!("  POST /api/v3/order + 4 headers + JSON body + response parse\n");
    println!("  Iterations:     {iterations:>12}");
    println!("  Elapsed:        {:>12.2?}", elapsed);
    println!("  Requests/sec:   {:>12.0}", rps);
    println!("  ns/request:     {:>12.1}", ns_per_req);
    println!(
        "  cycles/request: {:>12.0} (est. at 3GHz)",
        ns_per_req * 3.0
    );
    println!();
}

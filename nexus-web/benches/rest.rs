//! REST client benchmarks — request construction + response parsing.
//!
//! Measures the full protocol path: RequestWriter::finish() → send → ResponseReader parse.
//! Uses mock I/O to isolate protocol overhead from syscalls.
//!
//! Run with:
//!   cargo bench -p nexus-web --bench rest

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

use std::io::{self, Cursor, Read, Write};

use nexus_web::http::ResponseReader;
use nexus_web::rest::{Client, RequestWriter};

// =============================================================================
// Mock stream
// =============================================================================

const CANNED_RESPONSE: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\nX-RateLimit-Remaining: 1200\r\n\r\n{\"orderId\":123}";

const CHUNKED_RESPONSE: &[u8] =
    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nf\r\n{\"orderId\":123}\r\n0\r\n\r\n";

struct MockStream {
    response: Cursor<&'static [u8]>,
}

impl MockStream {
    fn new(response: &'static [u8]) -> Self {
        Self {
            response: Cursor::new(response),
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

// =============================================================================
// Benchmarks
// =============================================================================

fn bench_request_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("rest/request_construction");

    // GET — no body, no extra headers
    group.bench_function("GET_simple", |b| {
        let mut writer = RequestWriter::new("api.binance.com").unwrap();
        b.iter(|| {
            let req = writer.get("/api/v3/ticker/price").finish().unwrap();
            black_box(req.as_bytes());
        });
    });

    // GET — with query params
    group.bench_function("GET_query_params", |b| {
        let mut writer = RequestWriter::new("api.binance.com").unwrap();
        b.iter(|| {
            let req = writer
                .get("/api/v3/ticker/price")
                .query("symbol", "BTCUSDT")
                .finish()
                .unwrap();
            black_box(req.as_bytes());
        });
    });

    // POST — 4 headers + JSON body (exchange order entry)
    group.bench_function("POST_order_entry", |b| {
        let mut writer = RequestWriter::new("api.binance.com").unwrap();
        writer
            .default_header("X-MBX-APIKEY", "abcdef1234567890abcdef1234567890")
            .unwrap();
        writer
            .default_header("Content-Type", "application/json")
            .unwrap();

        let body = br#"{"symbol":"BTCUSDT","side":"BUY","type":"LIMIT","quantity":"0.001","price":"67234.50"}"#;
        let timestamp = "1700000000000";
        let signature = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

        b.iter(|| {
            let req = writer
                .post("/api/v3/order")
                .header("X-MBX-TIMESTAMP", timestamp)
                .header("X-MBX-SIGNATURE", signature)
                .body(body)
                .finish()
                .unwrap();
            black_box(req.as_bytes());
        });
    });

    group.finish();
}

fn bench_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("rest/round_trip");

    // GET round-trip (build + write + parse)
    group.bench_function("GET_simple", |b| {
        let mut writer = RequestWriter::new("host").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(MockStream::new(CANNED_RESPONSE));

        b.iter(|| {
            conn.stream_mut().reset();
            let req = writer.get("/test").finish().unwrap();
            let resp = conn.send(req, &mut reader).unwrap();
            black_box(resp.status());
        });
    });

    // POST round-trip (build + write + parse)
    group.bench_function("POST_order_entry", |b| {
        let mut writer = RequestWriter::new("api.binance.com").unwrap();
        writer
            .default_header("X-MBX-APIKEY", "abcdef1234567890abcdef1234567890")
            .unwrap();
        writer
            .default_header("Content-Type", "application/json")
            .unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(MockStream::new(CANNED_RESPONSE));

        let body = br#"{"symbol":"BTCUSDT","side":"BUY","type":"LIMIT","quantity":"0.001","price":"67234.50"}"#;
        let timestamp = "1700000000000";
        let signature = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

        b.iter(|| {
            conn.stream_mut().reset();
            let req = writer
                .post("/api/v3/order")
                .header("X-MBX-TIMESTAMP", timestamp)
                .header("X-MBX-SIGNATURE", signature)
                .body(body)
                .finish()
                .unwrap();
            let resp = conn.send(req, &mut reader).unwrap();
            black_box(resp.status());
        });
    });

    // Chunked response round-trip
    group.bench_function("GET_chunked_response", |b| {
        let mut writer = RequestWriter::new("host").unwrap();
        let mut reader = ResponseReader::new(4096);
        let mut conn = Client::new(MockStream::new(CHUNKED_RESPONSE));

        b.iter(|| {
            conn.stream_mut().reset();
            let req = writer.get("/test").finish().unwrap();
            let resp = conn.send(req, &mut reader).unwrap();
            black_box(resp.body());
        });
    });

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("rest/throughput");

    for &(name, body_size) in &[
        ("no_body", 0usize),
        ("100B_json", 100),
        ("1KB_json", 1024),
        ("4KB_json", 4096),
    ] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("POST", name), &body_size, |b, &size| {
            let mut writer = RequestWriter::new("host").unwrap();
            writer
                .default_header("Content-Type", "application/json")
                .unwrap();
            let mut reader = ResponseReader::new(4096);
            let mut conn = Client::new(MockStream::new(CANNED_RESPONSE));

            let body = vec![b'x'; size];

            b.iter(|| {
                conn.stream_mut().reset();
                let req = if size > 0 {
                    writer.post("/test").body(&body).finish().unwrap()
                } else {
                    writer.post("/test").finish().unwrap()
                };
                let resp = conn.send(req, &mut reader).unwrap();
                black_box(resp.status());
            });
        });
    }

    group.finish();
}

fn bench_query_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("rest/query_encoding");

    group.bench_function("3_clean_params", |b| {
        let mut writer = RequestWriter::new("host").unwrap();
        b.iter(|| {
            let req = writer
                .get("/api")
                .query("symbol", "BTCUSDT")
                .query("limit", "100")
                .query("recvWindow", "5000")
                .finish()
                .unwrap();
            black_box(req.as_bytes());
        });
    });

    group.bench_function("3_params_needing_encoding", |b| {
        let mut writer = RequestWriter::new("host").unwrap();
        b.iter(|| {
            let req = writer
                .get("/api")
                .query("q", "hello world")
                .query("filter", "price>100&qty<50")
                .query("note", "a=b&c=d")
                .finish()
                .unwrap();
            black_box(req.as_bytes());
        });
    });

    group.finish();
}

fn bench_loopback(c: &mut Criterion) {
    use std::net::{TcpListener, TcpStream};

    let mut group = c.benchmark_group("rest/loopback_tcp");

    // Start a minimal HTTP server
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    let response_body = b"{\"orderId\":12345,\"status\":\"FILLED\"}";
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nX-RateLimit-Remaining: 1200\r\n\r\n{}",
        response_body.len(),
        std::str::from_utf8(response_body).unwrap(),
    );
    let resp_bytes: &'static [u8] = Box::leak(response.into_bytes().into_boxed_slice());

    let server = std::thread::spawn(move || {
        let (mut tcp, _) = listener.accept().unwrap();
        tcp.set_nodelay(true).unwrap();
        let mut buf = [0u8; 4096];
        loop {
            match tcp.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if tcp.write_all(resp_bytes).is_err() {
                        break;
                    }
                }
            }
        }
    });

    let tcp = TcpStream::connect(addr).unwrap();
    tcp.set_nodelay(true).unwrap();
    let mut writer = RequestWriter::new("localhost").unwrap();
    writer
        .default_header("X-MBX-APIKEY", "abcdef1234567890abcdef1234567890")
        .unwrap();
    writer
        .default_header("Content-Type", "application/json")
        .unwrap();
    let mut reader = ResponseReader::new(4096);
    let mut conn = Client::new(tcp);

    let order_body = br#"{"symbol":"BTCUSDT","side":"BUY","type":"LIMIT","quantity":"0.001","price":"67234.50"}"#;

    // Warmup
    for _ in 0..1000 {
        let req = writer
            .post("/api/v3/order")
            .header("X-MBX-TIMESTAMP", "1700000000000")
            .header("X-MBX-SIGNATURE", "e3b0c44298fc1c149afbf4c8996fb924")
            .body(order_body)
            .finish()
            .unwrap();
        let resp = conn.send(req, &mut reader).unwrap();
        black_box(resp.status());
    }

    group.throughput(Throughput::Elements(1));

    group.bench_function("POST_order_entry", |b| {
        b.iter(|| {
            let req = writer
                .post("/api/v3/order")
                .header("X-MBX-TIMESTAMP", "1700000000000")
                .header("X-MBX-SIGNATURE", "e3b0c44298fc1c149afbf4c8996fb924")
                .body(order_body)
                .finish()
                .unwrap();
            let resp = conn.send(req, &mut reader).unwrap();
            black_box(resp.status());
        });
    });

    group.finish();
    drop(conn);
    let _ = server.join();
}

criterion_group!(
    benches,
    bench_request_construction,
    bench_round_trip,
    bench_throughput,
    bench_query_encoding,
    bench_loopback,
);
criterion_main!(benches);

//! Client pool throughput benchmark over real TCP.
//!
//! Measures: requests/sec through the pool with N connections.
//! Single-threaded (current_thread + LocalSet).
//!
//! Usage:
//!   cargo run --release -p nexus-async-net --example bench_pool_throughput

use std::time::Instant;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use nexus_async_net::rest::MaybeTls;
use nexus_async_net::rest::{ClientSlot, HttpConnection};
use nexus_net::http::ResponseReader;
use nexus_net::rest::RequestWriter;
use nexus_pool::local::Pool;

const RESPONSE: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\nX-RateLimit-Remaining: 42\r\n\r\n{\"orderId\":123}";

async fn run_server(listener: TcpListener) {
    loop {
        let Ok((mut tcp, _)) = listener.accept().await else {
            break;
        };
        tcp.set_nodelay(true).unwrap();
        tokio::task::spawn_local(async move {
            let mut buf = [0u8; 4096];
            loop {
                match tcp.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if tcp.write_all(RESPONSE).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });
    }
}

async fn bench_single_connection(addr: std::net::SocketAddr, iterations: u64) {
    let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
    tcp.set_nodelay(true).unwrap();
    let stream = MaybeTls::Plain(tcp);
    let mut conn = HttpConnection::new(stream);
    let mut writer = RequestWriter::new(&addr.to_string()).unwrap();
    writer
        .default_header("Content-Type", "application/json")
        .unwrap();
    let mut reader = ResponseReader::new(4096);

    let body = br#"{"symbol":"BTCUSDT","side":"BUY"}"#;

    // Warmup
    for _ in 0..1000 {
        let req = writer.post("/order").body(body).finish().unwrap();
        let resp = conn.send(req, &mut reader).await.unwrap();
        std::hint::black_box(resp.status());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let req = writer.post("/order").body(body).finish().unwrap();
        let resp = conn.send(req, &mut reader).await.unwrap();
        std::hint::black_box(resp.status());
    }
    let elapsed = start.elapsed();
    let rps = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "  {:<45} {:>9.0} req/sec  ({:.2?})",
        "single connection (no pool)", rps, elapsed,
    );
}

#[allow(clippy::future_not_send)] // intentionally uses non-Send pool + LocalSet
async fn bench_pool(addr: std::net::SocketAddr, pool_size: usize, iterations: u64) {
    // Build pool manually with loopback connections.
    let pool: Pool<ClientSlot> = Pool::new(
        || ClientSlot {
            writer: RequestWriter::new("placeholder").unwrap(),
            reader: ResponseReader::new(4096),
            conn: None,
        },
        |slot| {
            if slot.needs_reconnect() {
                slot.conn = None;
            }
        },
    );

    // Create connected slots.
    for _ in 0..pool_size {
        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        tcp.set_nodelay(true).unwrap();
        let stream = MaybeTls::Plain(tcp);
        let conn = HttpConnection::new(stream);
        let mut writer = RequestWriter::new(&addr.to_string()).unwrap();
        writer
            .default_header("Content-Type", "application/json")
            .unwrap();
        pool.put(ClientSlot {
            writer,
            reader: ResponseReader::new(4096),
            conn: Some(conn),
        });
    }

    let body = br#"{"symbol":"BTCUSDT","side":"BUY"}"#;

    // Warmup
    for _ in 0..1000 {
        let mut slot = pool.acquire();
        let s: &mut ClientSlot = &mut slot;
        let req = s.writer.post("/order").body(body).finish().unwrap();
        let conn = s.conn.as_mut().unwrap();
        let resp = conn.send(req, &mut s.reader).await.unwrap();
        std::hint::black_box(resp.status());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let mut slot = pool.acquire();
        let s: &mut ClientSlot = &mut slot;
        let req = s.writer.post("/order").body(body).finish().unwrap();
        let conn = s.conn.as_mut().unwrap();
        let resp = conn.send(req, &mut s.reader).await.unwrap();
        std::hint::black_box(resp.status());
    }
    let elapsed = start.elapsed();
    let rps = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "  pool ({pool_size} conn, sequential)                        {:>9.0} req/sec  ({:.2?})",
        rps, elapsed,
    );
}

#[allow(clippy::future_not_send)] // intentionally uses Rc + LocalSet
async fn bench_pool_concurrent(
    addr: std::net::SocketAddr,
    pool_size: usize,
    tasks: usize,
    iterations_per_task: u64,
) {
    use std::rc::Rc;

    let pool: Rc<Pool<ClientSlot>> = Rc::new(Pool::new(
        || ClientSlot {
            writer: RequestWriter::new("placeholder").unwrap(),
            reader: ResponseReader::new(4096),
            conn: None,
        },
        |slot| {
            if slot.needs_reconnect() {
                slot.conn = None;
            }
        },
    ));

    for _ in 0..pool_size {
        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        tcp.set_nodelay(true).unwrap();
        let stream = MaybeTls::Plain(tcp);
        let conn = HttpConnection::new(stream);
        let mut writer = RequestWriter::new(&addr.to_string()).unwrap();
        writer
            .default_header("Content-Type", "application/json")
            .unwrap();
        pool.put(ClientSlot {
            writer,
            reader: ResponseReader::new(4096),
            conn: Some(conn),
        });
    }

    let body = br#"{"symbol":"BTCUSDT","side":"BUY"}"#;

    // Warmup
    for _ in 0..500 {
        let mut slot = pool.acquire();
        let s: &mut ClientSlot = &mut slot;
        let req = s.writer.post("/order").body(body).finish().unwrap();
        let conn = s.conn.as_mut().unwrap();
        let resp = conn.send(req, &mut s.reader).await.unwrap();
        std::hint::black_box(resp.status());
    }

    let total = tasks as u64 * iterations_per_task;
    let start = Instant::now();

    let mut handles = Vec::with_capacity(tasks);
    for _ in 0..tasks {
        let pool = pool.clone();
        handles.push(tokio::task::spawn_local(async move {
            for _ in 0..iterations_per_task {
                let mut slot = pool.acquire();
                let s: &mut ClientSlot = &mut slot;
                let req = s.writer.post("/order").body(body).finish().unwrap();
                let conn = s.conn.as_mut().unwrap();
                let resp = conn.send(req, &mut s.reader).await.unwrap();
                std::hint::black_box(resp.status());
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let elapsed = start.elapsed();
    let rps = total as f64 / elapsed.as_secs_f64();
    println!(
        "  pool ({pool_size} conn, {tasks} tasks concurrent)             {:>9.0} req/sec  ({:.2?})",
        rps, elapsed,
    );
}

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let local = tokio::task::LocalSet::new();

    local.block_on(&rt, async {
        println!("\n  HTTP client pool throughput benchmark (real TCP, localhost)");
        println!("  POST + 2 headers + JSON body → 200 OK + JSON response\n");

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Start server.
        tokio::task::spawn_local(run_server(listener));

        // Allow server to start.
        tokio::task::yield_now().await;

        let iterations = 100_000;

        bench_single_connection(addr, iterations).await;
        bench_pool(addr, 1, iterations).await;
        bench_pool(addr, 4, iterations).await;
        bench_pool_concurrent(addr, 4, 4, iterations / 4).await;
        bench_pool_concurrent(addr, 8, 8, iterations / 8).await;

        println!();
    });
}

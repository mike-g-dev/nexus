//! Single-threaded async HTTP client pool.
//!
//! Uses [`nexus_pool::local::Pool`] for LIFO acquire/release with RAII guards.
//! Inline reconnect on acquire when a connection dies.
//!
//! For `current_thread` runtime + `LocalSet`.

use std::future::poll_fn;
use std::pin::Pin;

#[cfg(feature = "tls")]
use nexus_net::tls::TlsConfig;
use nexus_pool::local::{Pool, Pooled};
#[cfg(test)]
use nexus_web::http::HTTP_HANDSHAKE_BUFFER;
use nexus_web::http::ResponseReader;
use nexus_web::rest::{RequestWriter, RestError};
use tokio::io::AsyncWrite;

use super::connection::{HttpConnection, HttpConnectionBuilder};
use crate::maybe_tls::MaybeTls;

/// Drive `poll_shutdown` on a healthy connection during pool-build
/// cleanup when a later slot fails — sends TLS `close_notify` + TCP
/// FIN gracefully so the peer doesn't see a truncation alert. Errors
/// are ignored: best-effort cleanup on an already-failing path.
async fn graceful_shutdown(conn: &mut HttpConnection<MaybeTls>) {
    let stream = conn.stream_mut();
    let _ = poll_fn(|cx| Pin::new(&mut *stream).poll_shutdown(cx)).await;
}

// =============================================================================
// ClientSlot — the item stored in the pool
// =============================================================================

/// A complete request/response pipeline: writer + reader + transport.
///
/// Each slot in the pool owns its own set of protocol primitives.
/// Acquired via [`ClientPool::acquire`], auto-returned on drop.
///
/// Fields are public for split borrows through `Pooled<T>`'s `DerefMut`.
/// Deref explicitly to split, then build + send:
///
/// ```ignore
/// let s: &mut ClientSlot = &mut slot;  // explicit deref, enables split borrows
/// let req = s.writer.post("/order").body(json).finish()?;
/// let (conn, reader) = (s.conn.as_mut().unwrap(), &mut s.reader);
///
/// // With timeout (recommended for production):
/// let resp = tokio::time::timeout(timeout, conn.send(req, reader)).await??;
///
/// // Without timeout (prototyping only):
/// let resp = conn.send(req, reader).await?;
/// ```
pub struct ClientSlot {
    /// Request encoder (sans-IO). Build requests here.
    pub writer: RequestWriter,
    /// Response parser. Fed by the connection during send.
    pub reader: ResponseReader,
    /// Transport. `None` if connection died and needs reconnect.
    pub conn: Option<HttpConnection<MaybeTls>>,
}

impl ClientSlot {
    /// Whether the connection is dead and needs reconnect.
    pub fn needs_reconnect(&self) -> bool {
        self.conn.as_ref().is_none_or(HttpConnection::is_poisoned)
    }

    /// Split borrow: get mutable references to conn + reader
    /// while writer is borrowed by a `Request<'_>`.
    ///
    /// This exists because `Pooled<ClientSlot>` goes through `DerefMut`
    /// which prevents the compiler from seeing disjoint field borrows.
    pub fn conn_and_reader(
        &mut self,
    ) -> Result<(&mut HttpConnection<MaybeTls>, &mut ResponseReader), RestError> {
        let conn = self.conn.as_mut().ok_or(RestError::ConnectionPoisoned)?;
        Ok((conn, &mut self.reader))
    }
}

// =============================================================================
// ClientPool
// =============================================================================

/// Single-threaded async HTTP client pool.
///
/// Pre-allocated slots with LIFO acquire for cache locality. Each slot
/// owns a [`RequestWriter`], [`ResponseReader`], and
/// [`HttpConnection`].
///
/// # Usage
///
/// ```ignore
/// let pool = ClientPool::builder()
///     .url("https://api.binance.com")
///     .base_path("/api/v3")
///     .default_header("X-API-KEY", &key)?
///     .connections(4)
///     .tls(&tls)
///     .build()
///     .await?;
///
/// // Fast path (trading) — no reconnect, no wait
/// let mut slot = pool.try_acquire().unwrap();
/// // Patient path (background) — waits, reconnects with backoff
/// let mut slot = pool.acquire().await?;
///
/// let s: &mut ClientSlot = &mut slot;
/// let req = s.writer.post("/order").body(json).finish()?;
/// let conn = s.conn.as_mut().unwrap();
/// let resp = conn.send(req, &mut s.reader).await?;
/// // drop(slot) returns to pool
/// ```
pub struct ClientPool {
    pool: Pool<ClientSlot>,
    reconnect_config: ReconnectConfig,
}

#[derive(Clone)]
struct ReconnectConfig {
    url: String,
    #[cfg(feature = "tls")]
    tls_config: Option<TlsConfig>,
    nodelay: bool,
    #[cfg(feature = "socket-opts")]
    tcp_keepalive: Option<std::time::Duration>,
    #[cfg(feature = "socket-opts")]
    recv_buf_size: Option<usize>,
    #[cfg(feature = "socket-opts")]
    send_buf_size: Option<usize>,
}

#[allow(clippy::future_not_send)] // Intentionally !Send — single-threaded pool for LocalSet.
impl ClientPool {
    /// Create a builder.
    #[must_use]
    pub fn builder() -> ClientPoolBuilder {
        ClientPoolBuilder::new()
    }

    /// Try to acquire a healthy client slot (LIFO).
    ///
    /// Checks available slots for a healthy connection. Dead slots are
    /// ejected from the pool and a reconnect task is spawned for each.
    /// When reconnection succeeds, the slot returns to the pool
    /// automatically.
    ///
    /// Returns `None` if all slots are in use or currently reconnecting.
    ///
    /// This is the trading hot path — O(1) when the top slot is healthy.
    pub fn try_acquire(&self) -> Option<Pooled<ClientSlot>> {
        loop {
            let slot = self.pool.try_acquire()?;
            if !slot.needs_reconnect() {
                return Some(slot);
            }
            // Spawn a task to heal this slot. The task owns the Pooled
            // guard — when it reconnects, dropping the guard returns
            // the healthy slot to the pool.
            self.spawn_reconnect(slot);
            // Try next slot.
        }
    }

    /// Acquire a client slot, waiting until one is available.
    ///
    /// If no healthy slots are available, waits for reconnect tasks
    /// to finish healing dead connections. Returns error if no slot
    /// becomes available within the retry limit.
    ///
    /// This is the off-hot-path API for background tasks and REST calls.
    pub async fn acquire(&self) -> Result<Pooled<ClientSlot>, RestError> {
        const MAX_BACKOFF_MS: u64 = 1_000;
        const MAX_ATTEMPTS: u32 = 20;
        let mut backoff_ms = 1u64;

        for _ in 0..MAX_ATTEMPTS {
            if let Some(slot) = self.try_acquire() {
                return Ok(slot);
            }
            // All slots are either in use or being reconnected.
            // Wait for a reconnect task to finish and return a slot.
            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
            backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
        }

        Err(RestError::ConnectionClosed(
            "pool acquire timed out: no healthy slots available",
        ))
    }

    /// Number of slots currently available (not acquired).
    pub fn available(&self) -> usize {
        self.pool.available()
    }

    /// Spawn a local task to reconnect a dead slot.
    ///
    /// The task owns the `Pooled` guard. On successful reconnect, the
    /// guard drops and returns the healthy slot to the pool. On failure,
    /// retries with exponential backoff.
    fn spawn_reconnect(&self, mut slot: Pooled<ClientSlot>) {
        let config = self.reconnect_config.clone();
        tokio::task::spawn_local(async move {
            const MAX_BACKOFF_MS: u64 = 5_000;
            let mut backoff_ms = 100u64;

            loop {
                if let Ok(conn) = Self::connect_one_with(&config).await {
                    slot.conn = Some(conn);
                    slot.reader.reset();
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
            }
        });
    }

    async fn connect_one_with(
        config: &ReconnectConfig,
    ) -> Result<HttpConnection<MaybeTls>, RestError> {
        let mut builder = HttpConnectionBuilder::new();
        #[cfg(feature = "tls")]
        if let Some(ref tls) = config.tls_config {
            builder = builder.tls(tls);
        }
        if config.nodelay {
            builder = builder.disable_nagle();
        }
        #[cfg(feature = "socket-opts")]
        {
            if let Some(idle) = config.tcp_keepalive {
                builder = builder.tcp_keepalive(idle);
            }
            if let Some(size) = config.recv_buf_size {
                builder = builder.recv_buffer_size(size);
            }
            if let Some(size) = config.send_buf_size {
                builder = builder.send_buffer_size(size);
            }
        }
        builder.connect(&config.url).await
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Builder for [`ClientPool`].
pub struct ClientPoolBuilder {
    url: String,
    base_path: String,
    default_headers: Vec<(String, String)>,
    connections: usize,
    #[cfg(feature = "tls")]
    tls_config: Option<TlsConfig>,
    nodelay: bool,
    #[cfg(feature = "socket-opts")]
    tcp_keepalive: Option<std::time::Duration>,
    #[cfg(feature = "socket-opts")]
    recv_buf_size: Option<usize>,
    #[cfg(feature = "socket-opts")]
    send_buf_size: Option<usize>,
    write_buffer_capacity: usize,
    response_buffer_capacity: usize,
    max_body_size: usize,
}

impl ClientPoolBuilder {
    /// Create a builder with default settings: empty URL/base path, no
    /// default headers, one connection, TCP nodelay off, default buffer
    /// capacities.
    #[must_use]
    pub fn new() -> Self {
        Self {
            url: String::new(),
            base_path: String::new(),
            default_headers: Vec::new(),
            connections: 1,
            #[cfg(feature = "tls")]
            tls_config: None,
            nodelay: false,
            #[cfg(feature = "socket-opts")]
            tcp_keepalive: None,
            #[cfg(feature = "socket-opts")]
            recv_buf_size: None,
            #[cfg(feature = "socket-opts")]
            send_buf_size: None,
            write_buffer_capacity: 32 * 1024,
            response_buffer_capacity: 32 * 1024,
            max_body_size: 0,
        }
    }

    /// Target URL (scheme + host + optional port + optional path).
    #[must_use]
    pub fn url(mut self, url: &str) -> Self {
        self.url = url.to_string();
        self
    }

    /// Base path prefix for all requests.
    #[must_use]
    pub fn base_path(mut self, path: &str) -> Self {
        self.base_path = path.to_string();
        self
    }

    /// Add a default header sent with every request.
    pub fn default_header(mut self, name: &str, value: &str) -> Result<Self, RestError> {
        if name.bytes().any(|b| b == b'\r' || b == b'\n')
            || value.bytes().any(|b| b == b'\r' || b == b'\n')
        {
            return Err(RestError::CrlfInjection);
        }
        self.default_headers
            .push((name.to_string(), value.to_string()));
        Ok(self)
    }

    /// Number of pre-allocated connections. Default: 1.
    #[must_use]
    pub fn connections(mut self, n: usize) -> Self {
        self.connections = n;
        self
    }

    /// Custom TLS configuration.
    #[must_use]
    #[cfg(feature = "tls")]
    pub fn tls(mut self, config: &TlsConfig) -> Self {
        self.tls_config = Some(config.clone());
        self
    }

    /// Disable Nagle's algorithm on each connection.
    #[must_use]
    pub fn disable_nagle(mut self) -> Self {
        self.nodelay = true;
        self
    }

    /// Set TCP keepalive idle time on each connection.
    #[cfg(feature = "socket-opts")]
    #[must_use]
    pub fn tcp_keepalive(mut self, idle: std::time::Duration) -> Self {
        self.tcp_keepalive = Some(idle);
        self
    }

    /// Set `SO_RCVBUF` on each connection.
    #[cfg(feature = "socket-opts")]
    #[must_use]
    pub fn recv_buffer_size(mut self, n: usize) -> Self {
        self.recv_buf_size = Some(n);
        self
    }

    /// Set `SO_SNDBUF` on each connection.
    #[cfg(feature = "socket-opts")]
    #[must_use]
    pub fn send_buffer_size(mut self, n: usize) -> Self {
        self.send_buf_size = Some(n);
        self
    }

    /// Write buffer capacity per slot. Default: 32KB.
    #[must_use]
    pub fn write_buffer_capacity(mut self, n: usize) -> Self {
        self.write_buffer_capacity = n;
        self
    }

    /// Response buffer capacity per slot. Default: 32KB.
    #[must_use]
    pub fn response_buffer_capacity(mut self, n: usize) -> Self {
        self.response_buffer_capacity = n;
        self
    }

    /// Maximum response body size per slot. Default: 0 (no limit).
    #[must_use]
    pub fn max_body_size(mut self, n: usize) -> Self {
        self.max_body_size = n;
        self
    }

    /// Build the pool, establishing all connections.
    pub async fn build(self) -> Result<ClientPool, RestError> {
        if self.url.is_empty() {
            return Err(RestError::InvalidUrl("url is required".to_string()));
        }
        if self.connections == 0 {
            return Err(RestError::InvalidUrl("connections must be > 0".to_string()));
        }

        let parsed = nexus_web::rest::parse_base_url(&self.url)?;
        let host_header = parsed.host_header();

        let reconnect_config = ReconnectConfig {
            url: self.url.clone(),
            #[cfg(feature = "tls")]
            tls_config: self.tls_config.clone(),
            nodelay: self.nodelay,
            #[cfg(feature = "socket-opts")]
            tcp_keepalive: self.tcp_keepalive,
            #[cfg(feature = "socket-opts")]
            recv_buf_size: self.recv_buf_size,
            #[cfg(feature = "socket-opts")]
            send_buf_size: self.send_buf_size,
        };

        // Connect all slots sequentially (cold path — startup only).
        // If a later slot fails, gracefully shut down the already-built
        // healthy slots so the peer doesn't see TCP FIN without TLS
        // close_notify (which rustls peers log as a truncation alert).
        let mut initial_slots: Vec<ClientSlot> = Vec::with_capacity(self.connections);
        for _ in 0..self.connections {
            let slot_result: Result<ClientSlot, RestError> = async {
                let mut builder = HttpConnectionBuilder::new();
                #[cfg(feature = "tls")]
                if let Some(ref tls) = self.tls_config {
                    builder = builder.tls(tls);
                }
                if self.nodelay {
                    builder = builder.disable_nagle();
                }
                #[cfg(feature = "socket-opts")]
                {
                    if let Some(idle) = self.tcp_keepalive {
                        builder = builder.tcp_keepalive(idle);
                    }
                    if let Some(size) = self.recv_buf_size {
                        builder = builder.recv_buffer_size(size);
                    }
                    if let Some(size) = self.send_buf_size {
                        builder = builder.send_buffer_size(size);
                    }
                }
                let conn = builder.connect(&self.url).await?;

                let mut writer = RequestWriter::new(&host_header)?;
                if !self.base_path.is_empty() {
                    writer.set_base_path(&self.base_path)?;
                }
                writer.set_write_buffer_capacity(self.write_buffer_capacity);
                for (name, value) in &self.default_headers {
                    writer.default_header(name, value)?;
                }

                let reader = ResponseReader::new(self.response_buffer_capacity)
                    .max_body_size(self.max_body_size);

                Ok(ClientSlot {
                    writer,
                    reader,
                    conn: Some(conn),
                })
            }
            .await;

            match slot_result {
                Ok(slot) => initial_slots.push(slot),
                Err(e) => {
                    for slot in &mut initial_slots {
                        if let Some(ref mut c) = slot.conn {
                            graceful_shutdown(c).await;
                        }
                    }
                    return Err(e);
                }
            }
        }

        // Create pool with factory + reset.
        let host = host_header.clone();
        let base = self.base_path.clone();
        let headers = self.default_headers.clone();
        let wbuf_cap = self.write_buffer_capacity;
        let rbuf_cap = self.response_buffer_capacity;
        let max_body = self.max_body_size;

        let pool = Pool::new(
            move || {
                let mut writer = RequestWriter::new(&host).expect("host already validated");
                if !base.is_empty() {
                    writer
                        .set_base_path(&base)
                        .expect("base_path already validated");
                }
                writer.set_write_buffer_capacity(wbuf_cap);
                for (name, value) in &headers {
                    writer
                        .default_header(name, value)
                        .expect("headers already validated");
                }
                ClientSlot {
                    writer,
                    reader: ResponseReader::new(rbuf_cap).max_body_size(max_body),
                    conn: None,
                }
            },
            |slot: &mut ClientSlot| {
                if slot.needs_reconnect() {
                    slot.conn = None;
                    // Clear stale response data so the next request
                    // after reconnect starts with a clean buffer.
                    slot.reader.reset();
                }
            },
        );

        // Pre-populate with connected slots.
        for slot in initial_slots {
            pool.put(slot);
        }

        Ok(ClientPool {
            pool,
            reconnect_config,
        })
    }
}

impl Default for ClientPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_disconnected_slot() -> ClientSlot {
        ClientSlot {
            writer: RequestWriter::new("host").unwrap(),
            reader: ResponseReader::new(HTTP_HANDSHAKE_BUFFER),
            conn: None,
        }
    }

    #[test]
    fn slot_needs_reconnect_when_no_conn() {
        let slot = make_disconnected_slot();
        assert!(slot.needs_reconnect());
    }

    #[test]
    fn pool_acquire_release_cycle() {
        let pool = Pool::new(make_disconnected_slot, |_| {});
        pool.put(make_disconnected_slot());

        assert_eq!(pool.available(), 1);

        let slot = pool.acquire();
        assert_eq!(pool.available(), 0);

        drop(slot);
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn pool_acquire_returns_available() {
        let pool: Pool<ClientSlot> = Pool::new(make_disconnected_slot, |_| {});
        pool.put(make_disconnected_slot());
        pool.put(make_disconnected_slot());

        assert_eq!(pool.available(), 2);
        let _s1 = pool.acquire();
        assert_eq!(pool.available(), 1);
        let _s2 = pool.acquire();
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn pool_reset_clears_dead_conn() {
        let pool = Pool::new(make_disconnected_slot, |slot| {
            if slot.needs_reconnect() {
                slot.conn = None;
            }
        });
        pool.put(make_disconnected_slot());

        let slot = pool.acquire();
        assert!(slot.conn.is_none());
        assert!(slot.needs_reconnect());
        drop(slot);

        // After return + reset, slot is still disconnected.
        let slot = pool.acquire();
        assert!(slot.conn.is_none());
    }

    #[test]
    fn pool_multiple_slots() {
        let pool = Pool::new(make_disconnected_slot, |_| {});
        for _ in 0..4 {
            pool.put(make_disconnected_slot());
        }
        assert_eq!(pool.available(), 4);

        let s1 = pool.acquire();
        let s2 = pool.acquire();
        assert_eq!(pool.available(), 2);

        drop(s1);
        assert_eq!(pool.available(), 3);
        drop(s2);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn try_acquire_returns_none_when_all_in_use() {
        let pool = Pool::new(make_disconnected_slot, |_| {});
        pool.put(make_disconnected_slot());
        pool.put(make_disconnected_slot());

        let s1 = pool.try_acquire().unwrap();
        let s2 = pool.try_acquire().unwrap();
        // All slots are held — not dead, just in use.
        assert!(pool.try_acquire().is_none());
        assert_eq!(pool.available(), 0);

        // Return one.
        drop(s1);
        drop(s2);
        assert_eq!(pool.available(), 2);
        assert!(pool.try_acquire().is_some());
    }

    #[test]
    fn try_acquire_returns_some_after_slot_released() {
        // Verify that when all slots are held, try_acquire returns None,
        // and after release it returns Some.
        let pool = Pool::new(make_disconnected_slot, |_| {});
        pool.put(make_disconnected_slot());

        let held = pool.try_acquire().unwrap();
        assert!(pool.try_acquire().is_none());

        drop(held);
        assert!(pool.try_acquire().is_some());
    }

    /// Tests that 4 connections in the pool all work. Each connection
    /// is set up and used one at a time (sequential, same TCP listener).
    #[tokio::test(flavor = "current_thread")]
    async fn pool_four_connections_all_succeed() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Server: accept 4 sequential connections, each gets one request.
        tokio::spawn(async move {
            for _ in 0..4 {
                let (mut tcp, _) = listener.accept().await.unwrap();
                let mut buf = [0u8; HTTP_HANDSHAKE_BUFFER];
                let _ = tcp.read(&mut buf).await.unwrap();
                let resp = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok";
                tcp.write_all(resp).await.unwrap();
            }
        });

        let pool = Pool::new(make_disconnected_slot, |slot| {
            if slot.needs_reconnect() {
                slot.conn = None;
                slot.reader.reset();
            }
        });

        // Connect, send, and verify — 4 times. Each iteration creates
        // a new connection (simulating pool with 4 slots used sequentially).
        let mut success_count = 0u8;
        for _ in 0..4u8 {
            let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
            tcp.set_nodelay(true).unwrap();
            let stream = MaybeTls::Plain(tcp);
            let conn = HttpConnection::new(stream);
            pool.put(ClientSlot {
                writer: RequestWriter::new(&addr.to_string()).unwrap(),
                reader: ResponseReader::new(HTTP_HANDSHAKE_BUFFER),
                conn: Some(conn),
            });

            let mut slot = pool.try_acquire().unwrap();
            let s: &mut ClientSlot = &mut slot;
            let req = s.writer.get("/test").finish().unwrap();
            let conn = s.conn.as_mut().unwrap();
            let resp = conn.send(req, &mut s.reader).await.unwrap();
            assert_eq!(resp.status(), 200);
            assert_eq!(resp.body_str().unwrap(), "ok");
            success_count += 1;
        }

        assert_eq!(success_count, 4);
    }

    // Integration test with real TCP is in tests/httpbin.rs (ignored, needs network).
    // The loopback test below uses tokio to verify the full send path.
    #[tokio::test(flavor = "current_thread")]
    async fn pool_loopback_send() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Server: read request, send canned response.
        tokio::spawn(async move {
            let (mut tcp, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; HTTP_HANDSHAKE_BUFFER];
            let _ = tcp.read(&mut buf).await.unwrap();
            let resp = b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"orderId\":123}";
            tcp.write_all(resp).await.unwrap();
        });

        // Client: connect, wrap in MaybeTls, create slot, send.
        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        let stream = MaybeTls::Plain(tcp);
        let conn = HttpConnection::new(stream);

        let mut slot = ClientSlot {
            writer: RequestWriter::new(&addr.to_string()).unwrap(),
            reader: ResponseReader::new(HTTP_HANDSHAKE_BUFFER),
            conn: Some(conn),
        };

        let req = slot.writer.get("/test").finish().unwrap();
        let conn = slot.conn.as_mut().unwrap();
        let resp = conn.send(req, &mut slot.reader).await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.body_str().unwrap(), r#"{"orderId":123}"#);
    }

    #[test]
    fn try_acquire_returns_none_when_exhausted() {
        let pool = Pool::new(make_disconnected_slot, |_| {});
        pool.put(make_disconnected_slot());

        let _s1 = pool.try_acquire().unwrap();
        assert!(pool.try_acquire().is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pool_keep_alive_multiple_requests() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Server: handle two sequential requests on the same connection.
        tokio::spawn(async move {
            let (mut tcp, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; HTTP_HANDSHAKE_BUFFER];

            // Request 1
            let _ = tcp.read(&mut buf).await.unwrap();
            tcp.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\n{\"r\":1}")
                .await
                .unwrap();

            // Request 2 — same TCP connection
            let _ = tcp.read(&mut buf).await.unwrap();
            tcp.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\n{\"r\":2}")
                .await
                .unwrap();
        });

        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        tcp.set_nodelay(true).unwrap();
        let stream = MaybeTls::Plain(tcp);
        let conn = HttpConnection::new(stream);

        let pool = Pool::new(make_disconnected_slot, |slot| {
            if slot.needs_reconnect() {
                slot.conn = None;
                slot.reader.reset();
            }
        });
        pool.put(ClientSlot {
            writer: RequestWriter::new(&addr.to_string()).unwrap(),
            reader: ResponseReader::new(HTTP_HANDSHAKE_BUFFER),
            conn: Some(conn),
        });

        // First request
        {
            let mut slot = pool.acquire();
            let s: &mut ClientSlot = &mut slot;
            let req = s.writer.get("/first").finish().unwrap();
            let conn = s.conn.as_mut().unwrap();
            let resp = conn.send(req, &mut s.reader).await.unwrap();
            assert_eq!(resp.body_str().unwrap(), r#"{"r":1}"#);
        } // slot returned

        // Second request — same slot, same connection (keep-alive)
        {
            let mut slot = pool.acquire();
            let s: &mut ClientSlot = &mut slot;
            let req = s.writer.get("/second").finish().unwrap();
            let conn = s.conn.as_mut().unwrap();
            let resp = conn.send(req, &mut s.reader).await.unwrap();
            assert_eq!(resp.body_str().unwrap(), r#"{"r":2}"#);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn builder_validates_empty_url() {
        let result = ClientPool::builder().connections(1).build().await;
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn builder_validates_zero_connections() {
        let result = ClientPool::builder()
            .url("http://localhost")
            .connections(0)
            .build()
            .await;
        assert!(result.is_err());
    }

    /// Reproduces the reqwest connection pool bug: server closes the
    /// connection while idle, client writes into the dead socket,
    /// read hangs forever. Our timeout prevents the hang.
    #[tokio::test(flavor = "current_thread")]
    async fn stale_connection_timeout_not_hang() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Server: accept, respond to first request, then close the connection.
        tokio::spawn(async move {
            let (mut tcp, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; HTTP_HANDSHAKE_BUFFER];

            // First request — respond normally.
            let _ = tcp.read(&mut buf).await.unwrap();
            tcp.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                .await
                .unwrap();

            // Close the connection — simulates server idle timeout.
            drop(tcp);
        });

        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        tcp.set_nodelay(true).unwrap();
        let stream = MaybeTls::Plain(tcp);
        let conn = HttpConnection::new(stream);

        let pool = Pool::new(make_disconnected_slot, |slot| {
            if slot.needs_reconnect() {
                slot.conn = None;
                slot.reader.reset();
            }
        });
        pool.put(ClientSlot {
            writer: RequestWriter::new(&addr.to_string()).unwrap(),
            reader: ResponseReader::new(HTTP_HANDSHAKE_BUFFER),
            conn: Some(conn),
        });

        // First request — succeeds.
        {
            let mut slot = pool.acquire();
            let s: &mut ClientSlot = &mut slot;
            let req = s.writer.get("/first").finish().unwrap();
            let conn = s.conn.as_mut().unwrap();
            let resp = conn.send(req, &mut s.reader).await.unwrap();
            assert_eq!(resp.status(), 200);
        }

        // Small delay to let server close.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Second request — server has closed. Without timeout this hangs.
        // With tokio::time::timeout, it fails fast.
        {
            let mut slot = pool.acquire();
            let s: &mut ClientSlot = &mut slot;
            let req = s.writer.get("/second").finish().unwrap();
            let conn = s.conn.as_mut().unwrap();

            let timeout = std::time::Duration::from_millis(500);
            let result = tokio::time::timeout(timeout, conn.send(req, &mut s.reader)).await;

            match result {
                Ok(Ok(_)) => panic!("stale connection should not succeed"),
                Ok(Err(_)) => {} // Connection error — correct (ConnectionClosed)
                Err(_elapsed) => {
                    // Timeout — also correct. This is what reqwest hangs on.
                    // We don't hang — we fail.
                }
            }

            // The connection should be dead.
            assert!(
                s.needs_reconnect(),
                "slot should be poisoned after stale connection"
            );
        }
    }

    /// Dead connection heals automatically via spawned reconnect task.
    ///
    /// Scenario:
    /// 1. Server accepts one connection, responds, then closes
    /// 2. Client sends on the now-dead connection → poisoned
    /// 3. Slot returns to pool (dead)
    /// 4. Server starts listening again
    /// 5. try_acquire() spawns reconnect task for dead slot
    /// 6. acquire() waits until reconnect task finishes
    /// 7. Slot is healthy — request succeeds
    #[tokio::test(flavor = "current_thread")]
    #[allow(clippy::large_futures)]
    async fn dead_connection_heals_via_reconnect_task() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                // Phase 1: Start server, build pool with 1 connection.
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let addr = listener.local_addr().unwrap();

                let server_handle = tokio::task::spawn_local({
                    let listener_fd = listener.into_std().unwrap();
                    async move {
                        let listener = TcpListener::from_std(listener_fd).unwrap();
                        // Accept first connection — respond then close.
                        let (mut tcp, _) = listener.accept().await.unwrap();
                        let mut buf = [0u8; HTTP_HANDSHAKE_BUFFER];
                        let _ = tcp.read(&mut buf).await.unwrap();
                        tcp.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nfirst")
                            .await
                            .unwrap();
                        drop(tcp); // kill connection

                        // Accept second connection (reconnect) — respond.
                        let (mut tcp, _) = listener.accept().await.unwrap();
                        let _ = tcp.read(&mut buf).await.unwrap();
                        tcp.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nhealed")
                            .await
                            .unwrap();
                    }
                });

                let pool = ClientPool::builder()
                    .url(&format!("http://{addr}"))
                    .connections(1)
                    .build()
                    .await
                    .unwrap();

                // Phase 2: First request succeeds.
                {
                    let mut slot = pool.try_acquire().unwrap();
                    let s: &mut ClientSlot = &mut slot;
                    let req = s.writer.get("/first").finish().unwrap();
                    let conn = s.conn.as_mut().unwrap();
                    let resp = conn.send(req, &mut s.reader).await.unwrap();
                    assert_eq!(resp.body_str().unwrap(), "first");
                }

                // Phase 3: Server closed the connection. Next send will fail.
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                {
                    let mut slot = pool.try_acquire().unwrap();
                    let s: &mut ClientSlot = &mut slot;
                    let req = s.writer.get("/dead").finish().unwrap();
                    let conn = s.conn.as_mut().unwrap();
                    let timeout = std::time::Duration::from_millis(500);
                    let _ = tokio::time::timeout(timeout, conn.send(req, &mut s.reader)).await;
                    assert!(s.needs_reconnect(), "should be poisoned");
                }
                // Slot returned to pool (dead).

                // Phase 4: try_acquire sees dead slot → spawns reconnect task.
                // Server is still listening (waiting for second accept).
                // acquire() waits for the reconnect task to finish.
                let mut slot = pool.acquire().await.unwrap();
                let s: &mut ClientSlot = &mut slot;
                let req = s.writer.get("/healed").finish().unwrap();
                let conn = s.conn.as_mut().unwrap();
                let resp = conn.send(req, &mut s.reader).await.unwrap();
                assert_eq!(resp.body_str().unwrap(), "healed");

                server_handle.await.unwrap();
            })
            .await;
    }
}

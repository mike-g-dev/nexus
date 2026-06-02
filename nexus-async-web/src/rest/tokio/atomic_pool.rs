//! Thread-safe async HTTP client pool.
//!
//! Uses [`nexus_pool::sync::Pool`] for LIFO acquire/release with RAII guards.
//! Single acquirer, any thread can return. Inline reconnect on acquire.
//!
//! For multi-threaded tokio runtime.

use std::future::poll_fn;
use std::pin::Pin;

#[cfg(feature = "tls")]
use nexus_net::tls::TlsConfig;
use nexus_pool::sync::{Pool, Pooled};
#[cfg(test)]
use nexus_web::http::HTTP_HANDSHAKE_BUFFER;
use nexus_web::http::ResponseReader;
use nexus_web::rest::{RequestWriter, RestError};
use tokio::io::AsyncWrite;

use super::connection::{HttpConnection, HttpConnectionBuilder};
use crate::maybe_tls::MaybeTls;

/// Drive `poll_shutdown` on a healthy connection during pool-build
/// cleanup when a later slot fails — sends TLS `close_notify` + TCP
/// FIN gracefully so the peer doesn't see a truncation alert.
async fn graceful_shutdown(conn: &mut HttpConnection<MaybeTls>) {
    let stream = conn.stream_mut();
    let _ = poll_fn(|cx| Pin::new(&mut *stream).poll_shutdown(cx)).await;
}

// =============================================================================
// AtomicClientSlot
// =============================================================================

/// Thread-safe client slot. Same type as [`ClientSlot`](super::ClientSlot) —
/// the slot is identical regardless of pool type.
pub type AtomicClientSlot = super::ClientSlot;

// =============================================================================
// AtomicClientPool
// =============================================================================

/// Thread-safe async HTTP client pool.
///
/// Pre-allocated slots with LIFO acquire. **Single acquirer, any
/// returner** — acquire from one task, guards can be dropped from
/// any thread. Uses atomic CAS for lock-free release.
///
/// Note: `sync::Pool` is `Send` but not `Sync`. Wrap in `Mutex` if
/// you need shared acquire across multiple tasks on different threads.
///
/// # Usage
///
/// ```ignore
/// let pool = AtomicClientPool::builder()
///     .url("https://api.binance.com")
///     .base_path("/api/v3")
///     .default_header("X-API-KEY", &key)?
///     .connections(4)
///     .tls(&tls)
///     .build()
///     .await?;
///
/// let mut slot = pool.try_acquire().unwrap();
/// let req = slot.writer.post("/order").body(json).finish()?;
/// let conn = slot.conn.as_mut().unwrap();
/// let resp = conn.send(req, &mut slot.reader).await?;
/// // drop(slot) returns to pool from any thread
/// ```
pub struct AtomicClientPool {
    pool: Pool<AtomicClientSlot>,
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

impl AtomicClientPool {
    /// Create a builder.
    #[must_use]
    pub fn builder() -> AtomicClientPoolBuilder {
        AtomicClientPoolBuilder::new()
    }

    /// Try to acquire a healthy slot. Fast path — no reconnect.
    ///
    /// Dead slots are ejected and a reconnect task is spawned for each.
    /// Returns `None` if all slots are in use or currently reconnecting.
    pub fn try_acquire(&self) -> Option<Pooled<AtomicClientSlot>> {
        loop {
            let slot = self.pool.try_acquire()?;
            if !slot.needs_reconnect() {
                return Some(slot);
            }
            self.spawn_reconnect(slot);
        }
    }

    /// Acquire a slot, waiting until one is available.
    ///
    /// If no healthy slots exist, waits for spawned reconnect tasks
    /// to heal dead connections and return them to the pool.
    pub async fn acquire(&self) -> Result<Pooled<AtomicClientSlot>, RestError> {
        const MAX_BACKOFF_MS: u64 = 1_000;
        const MAX_ATTEMPTS: u32 = 20;
        let mut backoff_ms = 1u64;

        for _ in 0..MAX_ATTEMPTS {
            if let Some(slot) = self.try_acquire() {
                return Ok(slot);
            }
            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
            backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
        }

        Err(RestError::ConnectionClosed(
            "pool acquire timed out: no healthy slots available",
        ))
    }

    /// Number of slots currently available.
    pub fn available(&self) -> usize {
        self.pool.available()
    }

    /// Spawn a task to reconnect a dead slot.
    ///
    /// Uses `tokio::spawn` (Send-compatible). The task owns the guard;
    /// dropping it after reconnect returns the healthy slot to the pool.
    fn spawn_reconnect(&self, mut slot: Pooled<AtomicClientSlot>) {
        let config = self.reconnect_config.clone();
        tokio::spawn(async move {
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

/// Builder for [`AtomicClientPool`].
pub struct AtomicClientPoolBuilder {
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

impl AtomicClientPoolBuilder {
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

    /// Target URL.
    #[must_use]
    pub fn url(mut self, url: &str) -> Self {
        self.url = url.to_string();
        self
    }

    /// Base path prefix.
    #[must_use]
    pub fn base_path(mut self, path: &str) -> Self {
        self.base_path = path.to_string();
        self
    }

    /// Add a default header.
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

    /// Number of connections. Default: 1.
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

    /// Disable Nagle's algorithm.
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
    pub async fn build(self) -> Result<AtomicClientPool, RestError> {
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

        // Build the init and reset closures for sync::Pool.
        let host = host_header.clone();
        let base = self.base_path.clone();
        let headers = self.default_headers.clone();
        let wbuf_cap = self.write_buffer_capacity;
        let rbuf_cap = self.response_buffer_capacity;
        let max_body = self.max_body_size;

        let pool = Pool::new(
            self.connections,
            // Init — creates disconnected slots.
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
                AtomicClientSlot {
                    writer,
                    reader: ResponseReader::new(rbuf_cap).max_body_size(max_body),
                    conn: None,
                }
            },
            // Reset — clear dead connections on return.
            |slot: &mut AtomicClientSlot| {
                if slot.needs_reconnect() {
                    slot.conn = None;
                    slot.reader.reset();
                }
            },
        );

        // Replace disconnected slots with connected ones.
        let mut built = 0usize;
        for _ in 0..self.connections {
            let connect_result: Result<HttpConnection<MaybeTls>, RestError> = async {
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
                builder.connect(&self.url).await
            }
            .await;

            match connect_result {
                Ok(conn) => {
                    let mut slot = pool
                        .try_acquire()
                        .expect("pool should have slots during initial setup");
                    slot.conn = Some(conn);
                    built += 1;
                    // Drop returns it to the pool with the healthy conn.
                }
                Err(e) => {
                    // Shutdown the already-built slots before propagating.
                    // Re-acquire each one (they returned to the pool when
                    // their guards dropped above), shutdown the conn, drop.
                    for _ in 0..built {
                        if let Some(mut slot) = pool.try_acquire()
                            && let Some(ref mut c) = slot.conn
                        {
                            graceful_shutdown(c).await;
                        }
                    }
                    return Err(e);
                }
            }
        }

        Ok(AtomicClientPool {
            pool,
            reconnect_config,
        })
    }
}

impl Default for AtomicClientPoolBuilder {
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

    fn make_pool(n: usize) -> Pool<AtomicClientSlot> {
        Pool::new(
            n,
            || AtomicClientSlot {
                writer: RequestWriter::new("host").unwrap(),
                reader: ResponseReader::new(HTTP_HANDSHAKE_BUFFER),
                conn: None,
            },
            |slot| {
                if slot.needs_reconnect() {
                    slot.conn = None;
                }
            },
        )
    }

    #[test]
    fn atomic_pool_acquire_release() {
        let pool = make_pool(2);

        assert_eq!(pool.available(), 2);
        let s1 = pool.try_acquire().unwrap();
        assert_eq!(pool.available(), 1);
        let s2 = pool.try_acquire().unwrap();
        assert_eq!(pool.available(), 0);
        assert!(pool.try_acquire().is_none());

        drop(s1);
        assert_eq!(pool.available(), 1);
        drop(s2);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn atomic_slot_needs_reconnect() {
        let slot = AtomicClientSlot {
            writer: RequestWriter::new("host").unwrap(),
            reader: ResponseReader::new(HTTP_HANDSHAKE_BUFFER),
            conn: None,
        };
        assert!(slot.needs_reconnect());
    }

    #[tokio::test]
    async fn atomic_pool_loopback() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut tcp, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; HTTP_HANDSHAKE_BUFFER];
            let _ = tcp.read(&mut buf).await.unwrap();
            let resp = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok";
            tcp.write_all(resp).await.unwrap();
        });

        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        let stream = MaybeTls::Plain(tcp);
        let conn = HttpConnection::new(stream);

        let pool = make_pool(1);
        {
            let mut slot = pool.try_acquire().unwrap();
            slot.writer = RequestWriter::new(&addr.to_string()).unwrap();
            slot.conn = Some(conn);
        } // return to pool

        let mut slot = pool.try_acquire().unwrap();
        let s: &mut AtomicClientSlot = &mut slot;
        let req = s.writer.get("/test").finish().unwrap();
        let conn = s.conn.as_mut().unwrap();
        let resp = conn.send(req, &mut s.reader).await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.body_str().unwrap(), "ok");
    }

    #[test]
    fn atomic_try_acquire_none_when_exhausted() {
        let pool = make_pool(1);
        let _s1 = pool.try_acquire().unwrap();
        assert!(pool.try_acquire().is_none());
    }

    #[test]
    fn atomic_conn_and_reader_error_when_no_conn() {
        let pool = make_pool(1);
        let mut slot = pool.try_acquire().unwrap();
        assert!(slot.conn_and_reader().is_err());
    }

    #[test]
    fn atomic_try_acquire_none_when_all_in_use() {
        let pool = make_pool(2);
        let s1 = pool.try_acquire().unwrap();
        let s2 = pool.try_acquire().unwrap();
        // All slots acquired (not dead — just in use).
        assert!(pool.try_acquire().is_none());
        assert_eq!(pool.available(), 0);

        drop(s1);
        drop(s2);
        assert_eq!(pool.available(), 2);
        assert!(pool.try_acquire().is_some());
    }

    #[test]
    fn atomic_try_acquire_returns_some_after_release() {
        // Verify that when all slots are held, try_acquire returns None,
        // and after release it returns Some.
        let pool = make_pool(1);
        let held = pool.try_acquire().unwrap();
        assert!(pool.try_acquire().is_none());

        drop(held);
        assert!(pool.try_acquire().is_some());
    }

    /// Tests that 4 connections in the pool all work correctly.
    #[tokio::test]
    async fn atomic_pool_four_connections() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Server: accept 4 sequential connections.
        tokio::spawn(async move {
            for _ in 0..4 {
                let (mut tcp, _) = listener.accept().await.unwrap();
                let mut buf = [0u8; HTTP_HANDSHAKE_BUFFER];
                let _ = tcp.read(&mut buf).await.unwrap();
                let resp = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok";
                tcp.write_all(resp).await.unwrap();
            }
        });

        let pool = make_pool(4);

        // Connect, send, verify — 4 times sequentially.
        let mut success_count = 0u8;
        for _ in 0..4u8 {
            {
                let mut slot = pool.try_acquire().unwrap();
                let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
                tcp.set_nodelay(true).unwrap();
                let stream = MaybeTls::Plain(tcp);
                slot.conn = Some(HttpConnection::new(stream));
                slot.writer = RequestWriter::new(&addr.to_string()).unwrap();

                let s: &mut AtomicClientSlot = &mut slot;
                let req = s.writer.get("/test").finish().unwrap();
                let conn = s.conn.as_mut().unwrap();
                let resp = conn.send(req, &mut s.reader).await.unwrap();
                assert_eq!(resp.status(), 200);
                assert_eq!(resp.body_str().unwrap(), "ok");
                success_count += 1;
            }
        }

        assert_eq!(success_count, 4);
    }

    #[tokio::test]
    async fn atomic_builder_validates_empty_url() {
        let result = AtomicClientPool::builder().connections(1).build().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn atomic_builder_validates_zero_connections() {
        let result = AtomicClientPool::builder()
            .url("http://localhost")
            .connections(0)
            .build()
            .await;
        assert!(result.is_err());
    }
}

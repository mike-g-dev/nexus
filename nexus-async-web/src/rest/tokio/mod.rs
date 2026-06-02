//! Async HTTP REST client — tokio adapter for nexus-web.
//!
//! Same [`RequestWriter`], same [`ResponseReader`], same zero-alloc
//! protocol layer. The only difference is `.await` on socket I/O.
//!
//! # Usage
//!
//! ```ignore
//! use nexus_web::rest::RequestWriter;
//! use nexus_web::http::ResponseReader;
//! use nexus_async_web::rest::HttpConnectionBuilder;
//!
//! let mut writer = RequestWriter::new("api.binance.com")?;
//! writer.default_header("X-API-KEY", &key)?;
//! let mut reader = ResponseReader::new(32 * 1024);
//!
//! let tls = nexus_net::tls::TlsConfig::new()?;
//! let mut conn = HttpConnectionBuilder::new()
//!     .tls(&tls)
//!     .connect("https://api.binance.com")
//!     .await?;
//!
//! let req = writer.get("/orders").query("symbol", "BTC").finish()?;
//! let resp = conn.send(req, &mut reader).await?;
//! println!("{}", resp.body_str()?);
//! ```

mod atomic_pool;
mod connection;
mod pool;

pub use crate::maybe_tls::MaybeTls;
pub use atomic_pool::{AtomicClientPool, AtomicClientPoolBuilder, AtomicClientSlot};
pub use connection::{HttpConnection, HttpConnectionBuilder};
pub use pool::{ClientPool, ClientPoolBuilder, ClientSlot};

//! Async HTTP REST client -- nexus-async-rt adapter for nexus-web.
//!
//! Same [`RequestWriter`](nexus_web::rest::RequestWriter), same
//! [`ResponseReader`](nexus_web::http::ResponseReader), same zero-alloc
//! protocol layer. The only difference is `.await` on socket I/O backed
//! by nexus-async-rt instead of tokio.

mod connection;
mod pool;

pub use crate::maybe_tls::MaybeTls;
pub use connection::{HttpConnection, HttpConnectionBuilder};
pub use pool::{ClientPool, ClientPoolBuilder, ClientSlot};

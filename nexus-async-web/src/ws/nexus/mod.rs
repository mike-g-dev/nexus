//! Async WebSocket — nexus-async-rt backend.
//!
//! Provides [`WsStreamBuilder`] for connection setup. The primary API
//! types ([`WsReader`](super::WsReader) / [`WsWriter`](super::WsWriter))
//! are re-exported from the parent `ws` module.

mod stream;

pub use crate::maybe_tls::MaybeTls;
pub use stream::WsStreamBuilder;

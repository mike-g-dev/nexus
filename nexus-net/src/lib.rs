//! nexus-net — low-latency networking primitives.
//!
//! Buffer types, TLS codec, and wire abstractions for protocol engines.
//!
//! # Modules
//!
//! - [`buf`] — Buffer primitives (`ReadBuf`, `WriteBuf`, `WriteBufWriter`)
//! - [`maybe_tls`] — Plain-or-TLS stream wrapper
//! - [`wire`] — `WireStream` and `ParserSink` composition traits
//! - `tls` — TLS codec via rustls (feature: `tls`)
//!
//! # Protocol implementations
//!
//! WebSocket, HTTP, and REST moved to [`nexus-web`](https://docs.rs/nexus-web)
//! in 0.8.0.

#![warn(missing_docs)]

pub mod buf;
pub mod maybe_tls;
#[cfg(feature = "tls")]
pub mod tls;
pub mod wire;

pub use maybe_tls::MaybeTls;
pub use wire::{ParserSink, WireStream};

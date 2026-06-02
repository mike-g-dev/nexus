//! nexus-web — low-latency web protocol primitives.
//!
//! Sans-IO protocol implementations that operate on byte slices.
//! No async runtime, no I/O layer — pure protocol state machines.
//!
//! Extracted from [`nexus-net`](https://docs.rs/nexus-net) 0.7.x.
//! Protocol code moved here; networking primitives (buffers, TLS,
//! wire abstractions) remain in nexus-net and are re-exported below.
//!
//! # Modules
//!
//! - [`ws`] — WebSocket framing (RFC 6455)
//! - [`http`] — HTTP/1.1 response parsing, chunked decoding, request/response writers
//! - [`rest`] — HTTP/1.1 REST client (`RequestWriter`, `Client`, typestate builder)
//!
//! # Primitives
//!
//! Buffer types, TLS, and wire abstractions live in
//! [`nexus-net`](https://docs.rs/nexus-net) and are re-exported here
//! for convenience.

#![warn(missing_docs)]

pub mod http;
pub mod rest;
pub mod ws;

// Re-export nexus-net primitives used in our public API so downstream
// crates don't need to depend on nexus-net directly.
pub use nexus_net::buf;
pub use nexus_net::maybe_tls;
pub use nexus_net::wire;
pub use nexus_net::{MaybeTls, ParserSink, WireStream};

#[cfg(feature = "tls")]
pub use nexus_net::tls;

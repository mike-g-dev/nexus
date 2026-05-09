//! nexus-net — low-latency network protocol primitives.
//!
//! Sans-IO protocol implementations that operate on byte slices.
//! No async runtime, no I/O layer — pure protocol state machines.
//!
//! # Modules
//!
//! - [`buf`] — Buffer primitives (`ReadBuf`, `WriteBuf`, `WriteBufWriter`)
//! - [`ws`] — WebSocket framing (RFC 6455)
//! - [`http`] — HTTP/1.1 response parsing, chunked decoding, request/response writers
//! - [`rest`] — HTTP/1.1 REST client (`RequestWriter`, `Client`, typestate builder)
//! - `tls` — TLS codec via rustls (feature: `tls`)

#![warn(missing_docs)]

pub mod buf;
pub mod http;
pub mod maybe_tls;
pub mod rest;
#[cfg(feature = "tls")]
pub mod tls;
pub mod wire;
pub mod ws;

pub use maybe_tls::MaybeTls;
pub use wire::{ParserSink, WireStream};

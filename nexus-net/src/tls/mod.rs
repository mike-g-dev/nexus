//! TLS codec — sans-IO encrypt/decrypt via rustls.
//!
//! Sits between the socket and protocol parsers:
//!
//! ```text
//! socket → TlsCodec (decrypt) → FrameReader / ResponseReader → Message
//! Request → TlsCodec (encrypt) → socket
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use nexus_net::tls::TlsConfig;
//! use nexus_web::ws::Client;
//!
//! let tls = TlsConfig::new()?;
//! let mut ws = Client::builder()
//!     .tls(&tls)
//!     .connect("wss://exchange.com/ws/v1")?;
//!
//! while let Some(msg) = ws.recv()? {
//!     process(msg);
//! }
//! ```
//!
//! # Codec primitives
//!
//! - [`TlsCodec::read_tls`] feeds buffered ciphertext one packet step
//!   at a time (the canonical streaming path).
//! - [`TlsCodec::read_tls_from`] drives a sync [`Read`](std::io::Read)
//!   source directly.
//! - [`TlsCodec::read_and_process_tls`] loops over a bounded buffer
//!   (use for in-memory tests or custom adapters that pre-buffer
//!   known-bounded ciphertext; no production callers in this crate).
//! - [`TlsCodec::read_plaintext`] / [`TlsCodec::drain_plaintext_into`]
//!   drain decrypted plaintext (the latter feeds a [`ParserSink`](crate::ParserSink)
//!   directly, zero-copy).
//! - [`TlsCodec::encrypt`] / [`TlsCodec::write_tls_to`] handle the
//!   outbound side.

mod capacities;
mod codec;
mod config;
mod error;
mod stream;

pub use capacities::{TlsBufferCapacities, TlsBufferCapacitiesBuilder};
pub use codec::TlsCodec;
pub use config::{TlsConfig, TlsConfigBuilder};
pub use error::TlsError;
pub use stream::TlsStream;

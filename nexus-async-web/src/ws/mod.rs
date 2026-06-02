//! Async WebSocket — adapts nexus-web for async runtimes.
//!
//! The primary API is [`WsReader`] and [`WsWriter`] — sans-IO types
//! that own the frame parser and encoder independently. The transport
//! connection (`conn`) is passed to each call, enabling zero-copy
//! messages and independent read/write borrows.
//!
//! ```ignore
//! let (mut reader, mut writer, mut conn) = WsStreamBuilder::new()
//!     .connect("ws://localhost:8080/ws")
//!     .await?;
//!
//! while let Some(msg) = reader.recv(&mut conn).await? {
//!     match msg {
//!         Message::Ping(data) => writer.send_pong(&mut conn, data).await?,
//!         Message::Text(text) => {
//!             let response = process(text);
//!             writer.send_text(&mut conn, &response).await?;
//!         }
//!         _ => {}
//!     }
//! }
//! ```
//!
//! The tokio backend also provides [`WsStream`] — a bundled adapter
//! that implements `futures_core::Stream` and `futures_sink::Sink` for
//! ecosystem compatibility. This uses `OwnedMessage` (allocates per
//! message) and cannot overlap read/write borrows. Use it when you
//! need `StreamExt`/`SinkExt` combinators; use `WsReader`/`WsWriter`
//! for performance-sensitive code.

mod parts;

#[cfg(feature = "nexus")]
mod nexus;
#[cfg(feature = "tokio-rt")]
mod tokio;

pub use parts::{WsReader, WsWriter};

#[cfg(feature = "nexus")]
pub use self::nexus::*;
#[cfg(feature = "tokio-rt")]
pub use self::tokio::*;

//! Sans-IO HTTP/1.1 REST primitives + transport.
//!
//! **Protocol layer (sans-IO):**
//! - [`RequestWriter`] — typestate request encoder, produces [`Request`]
//! - [`ResponseReader`](crate::http::ResponseReader) — response parser
//!
//! **Transport layer:**
//! - [`Client`] — sends request bytes, reads response bytes
//!
//! # Usage
//!
//! ```ignore
//! use nexus_web::rest::{Client, RequestWriter};
//! use nexus_web::http::ResponseReader;
//!
//! let mut writer = RequestWriter::new("api.exchange.com")?;
//! writer.default_header("Authorization", "Bearer token123")?;
//! let mut reader = ResponseReader::new(32 * 1024);
//!
//! let mut conn = Client::builder().connect("http://api.exchange.com")?;
//!
//! let req = writer.get("/api/v1/orders")
//!     .query("symbol", "BTC-USD")
//!     .finish()?;
//! let resp = conn.send(req, &mut reader)?;
//! ```

mod connection;
mod error;
mod request;
mod response;

pub use connection::{Client, ClientBuilder, ParsedUrl, parse_base_url};
pub use error::RestError;
pub use request::{
    BodyWriter, Headers, Method, Query, Ready, Request, RequestBuilder, RequestWriter,
};
pub use response::RestResponse;

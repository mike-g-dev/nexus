use std::fmt;

use crate::http::HttpError;

/// REST client error.
#[derive(Debug)]
pub enum RestError {
    /// I/O error.
    Io(std::io::Error),
    /// HTTP protocol error.
    Http(HttpError),
    /// Response body exceeds max size.
    BodyTooLarge {
        /// Size reported by Content-Length (or accumulated for chunked).
        size: usize,
        /// Configured maximum body size in bytes.
        max: usize,
    },
    /// Request exceeds WriteBuf capacity.
    RequestTooLarge {
        /// Capacity of the write buffer in bytes.
        capacity: usize,
    },
    /// Header name/value or query parameter contains CR/LF bytes.
    CrlfInjection,
    /// Connection is poisoned after an I/O error mid-response.
    ConnectionPoisoned,
    /// Read timed out waiting for response.
    ReadTimeout,
    /// Connection is stale (dead socket detected after timeout).
    ConnectionStale,
    /// Connection closed before response complete.
    ConnectionClosed(&'static str),
    /// Invalid URL.
    InvalidUrl(String),
    /// `https://` URL used without the `tls` feature enabled.
    TlsNotEnabled,
    /// TLS error during connection setup (handshake, certificate
    /// validation, hostname resolution).
    ///
    /// **Steady-state TLS protocol errors** (decrypt failure, peer
    /// alert, malformed record received during a request) on the
    /// async `nexus-async-net` paths surface as
    /// [`RestError::Io`](Self::Io) instead — the underlying
    /// [`TlsError`](crate::tls::TlsError) is wrapped via
    /// `io::Error::other` and reachable via `io_err.source()` or
    /// `io_err.get_ref()`. This asymmetry stems from the
    /// `WireStream` trait returning `io::Result` for poll
    /// methods. Sync REST surfaces `Tls` directly because its
    /// `TlsStream` exposes `TlsError` natively. Pattern-match on
    /// both `Io` and `Tls` if you need to distinguish TLS-protocol
    /// failures from generic transport failures across both
    /// surfaces.
    #[cfg(feature = "tls")]
    Tls(crate::tls::TlsError),
}

impl fmt::Display for RestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Http(e) => write!(f, "HTTP error: {e}"),
            Self::BodyTooLarge { size, max } => {
                write!(f, "response body too large: {size} bytes (max: {max})")
            }
            Self::RequestTooLarge { capacity } => {
                write!(
                    f,
                    "request exceeds write buffer capacity ({capacity} bytes)"
                )
            }
            Self::CrlfInjection => {
                write!(f, "header or query parameter contains CR/LF")
            }
            Self::ConnectionPoisoned => write!(f, "connection poisoned after I/O error"),
            Self::ReadTimeout => write!(f, "read timed out waiting for response"),
            Self::ConnectionStale => write!(f, "connection stale (dead socket)"),
            Self::TlsNotEnabled => write!(f, "https:// requires the `tls` feature"),
            Self::ConnectionClosed(ctx) => write!(f, "connection closed: {ctx}"),
            Self::InvalidUrl(u) => write!(f, "invalid URL: {u}"),
            #[cfg(feature = "tls")]
            Self::Tls(e) => write!(f, "TLS error: {e}"),
        }
    }
}

impl std::error::Error for RestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Http(e) => Some(e),
            #[cfg(feature = "tls")]
            Self::Tls(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RestError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<HttpError> for RestError {
    fn from(e: HttpError) -> Self {
        Self::Http(e)
    }
}

#[cfg(feature = "tls")]
impl From<crate::tls::TlsError> for RestError {
    fn from(e: crate::tls::TlsError) -> Self {
        match e {
            crate::tls::TlsError::Io(io) => Self::Io(io),
            other => Self::Tls(other),
        }
    }
}

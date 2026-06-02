//! REST client error types.
//!
//! ## TLS error handling
//!
//! `From<TlsError> for RestError` partially preserves the TLS layer:
//! non-IO `TlsError` variants (decrypt failure, peer alert, malformed
//! record) surface as [`RestError::Tls`]; `TlsError::Io` flattens to
//! [`RestError::Io`] because it represents a genuine `io::Error` that
//! happened during TLS operations and the underlying async transport
//! ([`WireStream`](crate::WireStream)) returns `io::Result` either
//! way. The original `TlsError::Io` is preserved as the source of the
//! resulting `io::Error` and reachable via `io_err.source()` /
//! `io_err.get_ref()`. See [`RestError::Tls`] for the full
//! sync-vs-async asymmetry note.

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
    /// async `nexus-async-web` paths surface as
    /// [`RestError::Io`](Self::Io) instead — the underlying
    /// [`TlsError`](nexus_net::tls::TlsError) is wrapped via
    /// `io::Error::other` and reachable via `io_err.source()` or
    /// `io_err.get_ref()`. This asymmetry stems from the
    /// `WireStream` trait returning `io::Result` for poll
    /// methods. Sync REST surfaces `Tls` directly because its
    /// `TlsStream` exposes `TlsError` natively. Pattern-match on
    /// both `Io` and `Tls` if you need to distinguish TLS-protocol
    /// failures from generic transport failures across both
    /// surfaces.
    #[cfg(feature = "tls")]
    Tls(nexus_net::tls::TlsError),
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
impl From<nexus_net::tls::TlsError> for RestError {
    fn from(e: nexus_net::tls::TlsError) -> Self {
        match e {
            nexus_net::tls::TlsError::Io(io) => Self::Io(io),
            other => Self::Tls(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn rest_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");
        let err = RestError::from(io_err);
        assert!(matches!(err, RestError::Io(_)));
        assert!(err.to_string().contains("timeout"));
        assert!(err.source().is_some());
    }

    #[test]
    fn rest_error_http() {
        let http_err = HttpError::TooManyHeaders;
        let err = RestError::from(http_err);
        assert!(matches!(err, RestError::Http(_)));
        assert!(err.to_string().contains("too many"));
        assert!(err.source().is_some());
    }

    #[test]
    fn rest_error_body_too_large() {
        let err = RestError::BodyTooLarge {
            size: 10_000,
            max: 4096,
        };
        assert!(matches!(
            err,
            RestError::BodyTooLarge {
                size: 10_000,
                max: 4096,
            }
        ));
        assert_eq!(
            err.to_string(),
            "response body too large: 10000 bytes (max: 4096)"
        );
    }

    #[test]
    fn rest_error_request_too_large() {
        let err = RestError::RequestTooLarge { capacity: 32768 };
        assert!(matches!(
            err,
            RestError::RequestTooLarge { capacity: 32768 }
        ));
        assert!(
            err.to_string()
                .contains("exceeds write buffer capacity (32768 bytes)")
        );
    }

    #[test]
    fn rest_error_crlf_injection() {
        let err = RestError::CrlfInjection;
        assert!(matches!(err, RestError::CrlfInjection));
        assert_eq!(err.to_string(), "header or query parameter contains CR/LF");
    }

    #[test]
    fn rest_error_connection_poisoned() {
        let err = RestError::ConnectionPoisoned;
        assert!(matches!(err, RestError::ConnectionPoisoned));
        assert_eq!(err.to_string(), "connection poisoned after I/O error");
    }

    #[test]
    fn rest_error_read_timeout() {
        let err = RestError::ReadTimeout;
        assert!(matches!(err, RestError::ReadTimeout));
        assert_eq!(err.to_string(), "read timed out waiting for response");
    }

    #[test]
    fn rest_error_connection_stale() {
        let err = RestError::ConnectionStale;
        assert!(matches!(err, RestError::ConnectionStale));
        assert_eq!(err.to_string(), "connection stale (dead socket)");
    }

    #[test]
    fn rest_error_connection_closed() {
        let err = RestError::ConnectionClosed("during body read");
        assert!(matches!(
            err,
            RestError::ConnectionClosed("during body read")
        ));
        assert_eq!(err.to_string(), "connection closed: during body read");
    }

    #[test]
    fn rest_error_invalid_url() {
        let err = RestError::InvalidUrl("ftp://bad".into());
        assert!(matches!(err, RestError::InvalidUrl(_)));
        assert_eq!(err.to_string(), "invalid URL: ftp://bad");
    }

    #[test]
    fn rest_error_tls_not_enabled() {
        let err = RestError::TlsNotEnabled;
        assert!(matches!(err, RestError::TlsNotEnabled));
        assert_eq!(err.to_string(), "https:// requires the `tls` feature");
    }

    #[test]
    fn rest_error_source_none_for_leaf_variants() {
        assert!(RestError::CrlfInjection.source().is_none());
        assert!(RestError::ConnectionPoisoned.source().is_none());
        assert!(RestError::ReadTimeout.source().is_none());
        assert!(RestError::ConnectionStale.source().is_none());
        assert!(RestError::TlsNotEnabled.source().is_none());
        assert!(RestError::InvalidUrl("x".into()).source().is_none());
        assert!(RestError::ConnectionClosed("x").source().is_none());
        assert!(
            RestError::BodyTooLarge { size: 1, max: 1 }
                .source()
                .is_none()
        );
        assert!(
            RestError::RequestTooLarge { capacity: 1 }
                .source()
                .is_none()
        );
    }

    #[cfg(feature = "tls")]
    #[test]
    fn rest_error_from_tls_io_flattens() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "broken");
        let tls_err = nexus_net::tls::TlsError::Io(io_err);
        let rest_err = RestError::from(tls_err);
        // TlsError::Io should flatten to RestError::Io
        assert!(matches!(rest_err, RestError::Io(_)));
    }

    #[cfg(feature = "tls")]
    #[test]
    fn rest_error_from_tls_non_io_preserves() {
        let tls_err = nexus_net::tls::TlsError::NoRootCerts;
        let rest_err = RestError::from(tls_err);
        // Non-IO TlsError should become RestError::Tls
        assert!(matches!(rest_err, RestError::Tls(_)));
        assert!(rest_err.source().is_some());
    }
}

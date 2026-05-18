use std::fmt;

/// TLS operation error.
#[derive(Debug)]
pub enum TlsError {
    /// rustls protocol error (handshake failure, certificate error, etc.)
    Rustls(rustls::Error),
    /// I/O error during TLS operations.
    Io(std::io::Error),
    /// Invalid hostname for SNI.
    InvalidHostname(String),
    /// No system root certificates found.
    NoRootCerts,
}

impl fmt::Display for TlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rustls(e) => write!(f, "TLS error: {e}"),
            Self::Io(e) => write!(f, "TLS I/O error: {e}"),
            Self::InvalidHostname(h) => write!(f, "invalid TLS hostname: {h}"),
            Self::NoRootCerts => write!(f, "no system root certificates found"),
        }
    }
}

impl std::error::Error for TlsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rustls(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<rustls::Error> for TlsError {
    fn from(e: rustls::Error) -> Self {
        Self::Rustls(e)
    }
}

impl From<std::io::Error> for TlsError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn tls_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::ConnectionReset, "reset");
        let err = TlsError::from(io_err);
        assert!(matches!(err, TlsError::Io(_)));
        assert!(err.to_string().contains("reset"));
        assert!(err.source().is_some());
    }

    #[test]
    fn tls_error_rustls() {
        let rustls_err = rustls::Error::General("test error".into());
        let err = TlsError::from(rustls_err);
        assert!(matches!(err, TlsError::Rustls(_)));
        assert!(err.to_string().contains("test error"));
        assert!(err.source().is_some());
    }

    #[test]
    fn tls_error_invalid_hostname() {
        let err = TlsError::InvalidHostname("not a host!".into());
        assert!(matches!(err, TlsError::InvalidHostname(_)));
        assert_eq!(err.to_string(), "invalid TLS hostname: not a host!");
        assert!(err.source().is_none());
    }

    #[test]
    fn tls_error_no_root_certs() {
        let err = TlsError::NoRootCerts;
        assert!(matches!(err, TlsError::NoRootCerts));
        assert_eq!(err.to_string(), "no system root certificates found");
        assert!(err.source().is_none());
    }
}

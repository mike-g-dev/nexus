//! Stream that may or may not be wrapped in TLS (sync only).
//!
//! Implements `Read + Write` by delegating to either the plain
//! stream or the `TlsStream` wrapper (requires `tls` feature).
//!
//! Protocol clients use `MaybeTls<S>` as their stream type when the
//! TLS decision happens at runtime (`ws://` vs `wss://`). For async
//! TLS, see `nexus-async-web::maybe_tls`.

use std::io::{self, Read, Write};

#[cfg(feature = "tls")]
use crate::tls::TlsStream;

/// A stream that may or may not be wrapped in TLS.
///
/// The `Tls` variant is boxed because `TlsStream` includes rustls's
/// ~1KB connection state. TLS connections are established once at
/// startup — the box indirection is not on the hot path.
pub enum MaybeTls<S> {
    /// Plaintext stream.
    Plain(S),
    /// TLS-wrapped stream.
    #[cfg(feature = "tls")]
    Tls(Box<TlsStream<S>>),
}

impl<S> MaybeTls<S> {
    /// Whether this is a TLS-wrapped stream.
    pub fn is_tls(&self) -> bool {
        #[cfg(feature = "tls")]
        if matches!(self, Self::Tls(_)) {
            return true;
        }
        false
    }
}

// =============================================================================
// Read + Write (blocking)
// =============================================================================

impl<S: Read + Write> Read for MaybeTls<S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Plain(s) => s.read(buf),
            #[cfg(feature = "tls")]
            Self::Tls(s) => s.read(buf),
        }
    }
}

impl<S: Read + Write> Write for MaybeTls<S> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Plain(s) => s.write(buf),
            #[cfg(feature = "tls")]
            Self::Tls(s) => s.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Plain(s) => s.flush(),
            #[cfg(feature = "tls")]
            Self::Tls(s) => s.flush(),
        }
    }
}

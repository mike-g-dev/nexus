//! WebSocket HTTP upgrade handshake (RFC 6455 §4).

use sha1::{Digest, Sha1};

/// The WebSocket magic GUID used in Sec-WebSocket-Accept computation.
const WS_GUID: &[u8] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// Compute the Sec-WebSocket-Accept value from a Sec-WebSocket-Key.
///
/// `accept = base64(SHA-1(key + GUID))`
pub fn compute_accept_key(key: &str) -> [u8; 28] {
    let mut hasher = Sha1::new();
    hasher.update(key.as_bytes());
    hasher.update(WS_GUID);
    let hash = hasher.finalize();
    let hash_arr: [u8; 20] = hash.into();
    base64_encode_20(&hash_arr)
}

/// Generate a random 16-byte Sec-WebSocket-Key, base64-encoded (24 chars).
///
/// Uses OS randomness via `getrandom` per RFC 6455 §4.1 which requires
/// the key to be randomly selected.
pub fn generate_key() -> [u8; 24] {
    let mut raw = [0u8; 16];
    getrandom::fill(&mut raw).expect("OS randomness unavailable");
    base64_encode_16(&raw)
}

/// Validate a Sec-WebSocket-Accept value against the expected key.
pub fn validate_accept(key: &str, accept: &str) -> bool {
    let expected = compute_accept_key(key);
    accept.as_bytes() == &expected[..]
}

/// Handshake error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandshakeError {
    /// Response was not HTTP 101.
    UnexpectedStatus(u16),
    /// Missing or wrong Upgrade header.
    MissingUpgrade,
    /// Missing or wrong Connection header.
    MissingConnection,
    /// Sec-WebSocket-Accept doesn't match.
    InvalidAcceptKey,
    /// Missing Sec-WebSocket-Key in client request.
    MissingKey,
    /// Unsupported WebSocket version.
    UnsupportedVersion,
    /// HTTP response/request malformed or too large.
    MalformedHttp,
    /// I/O error.
    Io(String),
}

impl std::fmt::Display for HandshakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedStatus(s) => write!(f, "unexpected HTTP status: {s}"),
            Self::MissingUpgrade => write!(f, "missing Upgrade: websocket header"),
            Self::MissingConnection => write!(f, "missing Connection: Upgrade header"),
            Self::InvalidAcceptKey => write!(f, "Sec-WebSocket-Accept mismatch"),
            Self::MissingKey => write!(f, "missing Sec-WebSocket-Key header"),
            Self::UnsupportedVersion => write!(f, "unsupported WebSocket version"),
            Self::MalformedHttp => write!(f, "malformed HTTP"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for HandshakeError {}

impl From<std::io::Error> for HandshakeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

// =============================================================================
// Base64 (inline, standard alphabet, no padding for 16-byte, padding for 20-byte)
// =============================================================================

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Base64 encode exactly 16 bytes → 24 chars (with padding).
fn base64_encode_16(input: &[u8; 16]) -> [u8; 24] {
    let mut out = [0u8; 24];
    base64_encode_into(input, &mut out);
    out
}

/// Base64 encode exactly 20 bytes → 28 chars (with padding).
fn base64_encode_20(input: &[u8; 20]) -> [u8; 28] {
    let mut out = [0u8; 28];
    base64_encode_into(input, &mut out);
    out
}

fn base64_encode_into(input: &[u8], out: &mut [u8]) {
    let mut i = 0;
    let mut o = 0;
    while i + 3 <= input.len() {
        let n =
            (u32::from(input[i]) << 16) | (u32::from(input[i + 1]) << 8) | u32::from(input[i + 2]);
        out[o] = B64[((n >> 18) & 0x3F) as usize];
        out[o + 1] = B64[((n >> 12) & 0x3F) as usize];
        out[o + 2] = B64[((n >> 6) & 0x3F) as usize];
        out[o + 3] = B64[(n & 0x3F) as usize];
        i += 3;
        o += 4;
    }
    let remaining = input.len() - i;
    if remaining == 2 {
        let n = (u32::from(input[i]) << 16) | (u32::from(input[i + 1]) << 8);
        out[o] = B64[((n >> 18) & 0x3F) as usize];
        out[o + 1] = B64[((n >> 12) & 0x3F) as usize];
        out[o + 2] = B64[((n >> 6) & 0x3F) as usize];
        out[o + 3] = b'=';
    } else if remaining == 1 {
        let n = u32::from(input[i]) << 16;
        out[o] = B64[((n >> 18) & 0x3F) as usize];
        out[o + 1] = B64[((n >> 12) & 0x3F) as usize];
        out[o + 2] = b'=';
        out[o + 3] = b'=';
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc_6455_accept_key() {
        // RFC 6455 §4.2.2 example
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        let accept = compute_accept_key(key);
        assert_eq!(
            std::str::from_utf8(&accept).unwrap(),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn validate_accept_correct() {
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        assert!(validate_accept(key, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
    }

    #[test]
    fn validate_accept_wrong() {
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        assert!(!validate_accept(key, "wrongvalue"));
    }

    #[test]
    fn generate_key_is_24_chars() {
        let key = generate_key();
        assert_eq!(key.len(), 24);
        // Should be valid base64
        for &b in &key {
            assert!(
                b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'=',
                "invalid base64 char: {b}"
            );
        }
    }

    #[test]
    fn generate_key_not_constant() {
        let k1 = generate_key();
        let k2 = generate_key();
        // Two consecutive keys should differ (astronomically unlikely to match)
        assert_ne!(k1, k2);
    }

    #[test]
    fn base64_encode_16_known() {
        let input = [0u8; 16];
        let encoded = base64_encode_16(&input);
        assert_eq!(
            std::str::from_utf8(&encoded).unwrap(),
            "AAAAAAAAAAAAAAAAAAAAAA=="
        );
    }

    // =========================================================================
    // HandshakeError variant coverage
    // =========================================================================

    #[test]
    fn handshake_error_unexpected_status() {
        let err = HandshakeError::UnexpectedStatus(403);
        assert!(matches!(err, HandshakeError::UnexpectedStatus(403)));
        assert_eq!(err.to_string(), "unexpected HTTP status: 403");
    }

    #[test]
    fn handshake_error_missing_upgrade() {
        let err = HandshakeError::MissingUpgrade;
        assert!(matches!(err, HandshakeError::MissingUpgrade));
        assert_eq!(err.to_string(), "missing Upgrade: websocket header");
    }

    #[test]
    fn handshake_error_missing_connection() {
        let err = HandshakeError::MissingConnection;
        assert!(matches!(err, HandshakeError::MissingConnection));
        assert_eq!(err.to_string(), "missing Connection: Upgrade header");
    }

    #[test]
    fn handshake_error_invalid_accept_key() {
        let err = HandshakeError::InvalidAcceptKey;
        assert!(matches!(err, HandshakeError::InvalidAcceptKey));
        assert_eq!(err.to_string(), "Sec-WebSocket-Accept mismatch");
    }

    #[test]
    fn handshake_error_missing_key() {
        let err = HandshakeError::MissingKey;
        assert!(matches!(err, HandshakeError::MissingKey));
        assert_eq!(err.to_string(), "missing Sec-WebSocket-Key header");
    }

    #[test]
    fn handshake_error_unsupported_version() {
        let err = HandshakeError::UnsupportedVersion;
        assert!(matches!(err, HandshakeError::UnsupportedVersion));
        assert_eq!(err.to_string(), "unsupported WebSocket version");
    }

    #[test]
    fn handshake_error_malformed_http() {
        let err = HandshakeError::MalformedHttp;
        assert!(matches!(err, HandshakeError::MalformedHttp));
        assert_eq!(err.to_string(), "malformed HTTP");
    }

    #[test]
    fn handshake_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broken");
        let err = HandshakeError::from(io_err);
        assert!(matches!(err, HandshakeError::Io(_)));
        assert!(err.to_string().contains("pipe broken"));
    }

    #[test]
    fn handshake_error_is_std_error() {
        let err: &dyn std::error::Error = &HandshakeError::MalformedHttp;
        assert!(err.source().is_none());
    }

    #[test]
    fn handshake_error_eq() {
        assert_eq!(
            HandshakeError::UnexpectedStatus(404),
            HandshakeError::UnexpectedStatus(404)
        );
        assert_ne!(
            HandshakeError::UnexpectedStatus(404),
            HandshakeError::UnexpectedStatus(500)
        );
        assert_ne!(
            HandshakeError::MissingUpgrade,
            HandshakeError::MissingConnection
        );
    }
}

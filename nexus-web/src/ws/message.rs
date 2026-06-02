use super::error::ProtocolError;

/// WebSocket close status codes (RFC 6455 §7.4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseCode {
    /// 1000 — normal closure.
    Normal,
    /// 1001 — endpoint going away.
    GoingAway,
    /// 1002 — protocol error.
    Protocol,
    /// 1003 — received unsupported data type.
    Unsupported,
    /// 1005 — no status code present.
    NoStatus,
    /// 1007 — payload data not consistent with message type.
    InvalidPayload,
    /// 1008 — policy violation.
    PolicyViolation,
    /// 1009 — message too big.
    MessageTooBig,
    /// 1010 — client expected server to negotiate an extension.
    MandatoryExtension,
    /// 1011 — server encountered an unexpected condition.
    InternalError,
    /// Application-defined code (3000-4999).
    Other(u16),
}

impl CloseCode {
    /// Parse a close code from its wire representation.
    ///
    /// # Errors
    /// Returns `ProtocolError::InvalidCloseCode` for codes outside the
    /// valid ranges defined in RFC 6455 §7.4.2.
    pub fn from_u16(code: u16) -> Result<Self, ProtocolError> {
        match code {
            1000 => Ok(Self::Normal),
            1001 => Ok(Self::GoingAway),
            1002 => Ok(Self::Protocol),
            1003 => Ok(Self::Unsupported),
            // 1005 is reserved — MUST NOT appear on the wire (RFC 6455 §7.4.1)
            1007 => Ok(Self::InvalidPayload),
            1008 => Ok(Self::PolicyViolation),
            1009 => Ok(Self::MessageTooBig),
            1010 => Ok(Self::MandatoryExtension),
            1011 => Ok(Self::InternalError),
            3000..=4999 => Ok(Self::Other(code)),
            _ => Err(ProtocolError::InvalidCloseCode(code)),
        }
    }

    /// Convert to the wire representation.
    pub fn as_u16(&self) -> u16 {
        match self {
            Self::Normal => 1000,
            Self::GoingAway => 1001,
            Self::Protocol => 1002,
            Self::Unsupported => 1003,
            Self::NoStatus => 1005,
            Self::InvalidPayload => 1007,
            Self::PolicyViolation => 1008,
            Self::MessageTooBig => 1009,
            Self::MandatoryExtension => 1010,
            Self::InternalError => 1011,
            Self::Other(code) => *code,
        }
    }
}

/// Parsed close frame: status code + UTF-8 reason.
#[derive(Debug, Clone)]
pub struct CloseFrame<'a> {
    /// The close status code.
    pub code: CloseCode,
    /// UTF-8 reason string (validated, may be empty).
    pub reason: &'a str,
}

/// Owned close frame.
#[derive(Debug, Clone)]
pub struct OwnedCloseFrame {
    /// The close status code.
    pub code: CloseCode,
    /// UTF-8 reason string.
    pub reason: String,
}

/// A complete WebSocket message.
///
/// Text payloads are validated UTF-8. Close frames are parsed into
/// structured code + reason. No continuation frames are exposed.
///
/// Borrows from the reader's internal buffer — drop before calling
/// [`FrameReader::next()`](super::FrameReader) again.
#[derive(Debug, Clone)]
pub enum Message<'a> {
    /// UTF-8 text message (validated).
    Text(&'a str),
    /// Binary message.
    Binary(&'a [u8]),
    /// Ping control frame.
    Ping(&'a [u8]),
    /// Pong control frame.
    Pong(&'a [u8]),
    /// Connection close.
    Close(CloseFrame<'a>),
}

impl<'a> Message<'a> {
    /// Payload as bytes, regardless of message type.
    ///
    /// - `Text` → UTF-8 bytes
    /// - `Binary` / `Ping` / `Pong` → raw bytes
    /// - `Close` → reason string as bytes (excludes the 2-byte status code)
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Text(s) => s.as_bytes(),
            Self::Binary(b) | Self::Ping(b) | Self::Pong(b) => b,
            Self::Close(cf) => cf.reason.as_bytes(),
        }
    }

    /// Consume the message, returning the payload as a byte slice.
    ///
    /// Releases the borrow on the `FrameReader` while keeping access
    /// to the payload (valid until the reader is advanced).
    pub fn into_bytes(self) -> &'a [u8] {
        match self {
            Self::Text(s) => s.as_bytes(),
            Self::Binary(b) | Self::Ping(b) | Self::Pong(b) => b,
            Self::Close(cf) => cf.reason.as_bytes(),
        }
    }

    /// Take ownership. Copies payload out of borrowed buffer.
    pub fn into_owned(self) -> OwnedMessage {
        match self {
            Self::Text(s) => OwnedMessage::Text(s.to_owned()),
            Self::Binary(b) => OwnedMessage::Binary(b.to_vec()),
            Self::Ping(b) => OwnedMessage::Ping(b.to_vec()),
            Self::Pong(b) => OwnedMessage::Pong(b.to_vec()),
            Self::Close(cf) => OwnedMessage::Close(OwnedCloseFrame {
                code: cf.code,
                reason: cf.reason.to_owned(),
            }),
        }
    }
}

/// An owned WebSocket message, detached from reader buffers.
#[derive(Debug, Clone)]
pub enum OwnedMessage {
    /// UTF-8 text message.
    Text(String),
    /// Binary message.
    Binary(Vec<u8>),
    /// Ping control frame.
    Ping(Vec<u8>),
    /// Pong control frame.
    Pong(Vec<u8>),
    /// Connection close.
    Close(OwnedCloseFrame),
}

impl OwnedMessage {
    /// Payload as bytes, regardless of message type.
    ///
    /// - `Text` → UTF-8 bytes
    /// - `Binary` / `Ping` / `Pong` → raw bytes
    /// - `Close` → reason string as bytes (excludes the 2-byte status code)
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Text(s) => s.as_bytes(),
            Self::Binary(b) | Self::Ping(b) | Self::Pong(b) => b,
            Self::Close(cf) => cf.reason.as_bytes(),
        }
    }

    /// Convert to `bytes::Bytes`. Zero-copy — takes ownership of the
    /// underlying `Vec`/`String` allocation without copying.
    ///
    /// ```ignore
    /// let msg = ws.recv()?.unwrap().into_owned();
    /// let shared: Bytes = msg.to_bytes();
    /// tx.send(shared)?;  // Send + Clone, cheap to share
    /// ```
    #[cfg(feature = "bytes")]
    pub fn to_bytes(self) -> bytes::Bytes {
        match self {
            Self::Text(s) => bytes::Bytes::from(s.into_bytes()),
            Self::Binary(b) | Self::Ping(b) | Self::Pong(b) => bytes::Bytes::from(b),
            Self::Close(cf) => bytes::Bytes::from(cf.reason.into_bytes()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_code_round_trip() {
        let codes = [
            (1000, CloseCode::Normal),
            (1001, CloseCode::GoingAway),
            (1002, CloseCode::Protocol),
            (1003, CloseCode::Unsupported),
            (1007, CloseCode::InvalidPayload),
            (1008, CloseCode::PolicyViolation),
            (1009, CloseCode::MessageTooBig),
            (1010, CloseCode::MandatoryExtension),
            (1011, CloseCode::InternalError),
            (3000, CloseCode::Other(3000)),
            (4999, CloseCode::Other(4999)),
        ];
        for (raw, expected) in &codes {
            let parsed = CloseCode::from_u16(*raw).unwrap();
            assert_eq!(parsed, *expected);
            assert_eq!(parsed.as_u16(), *raw);
        }
    }

    #[test]
    fn close_code_rejects_invalid() {
        let invalid = [0, 999, 1004, 1005, 1006, 1015, 1016, 2999, 5000, u16::MAX];
        for code in &invalid {
            assert!(
                CloseCode::from_u16(*code).is_err(),
                "should reject code {code}"
            );
        }
    }

    #[test]
    fn message_into_owned() {
        let text = Message::Text("hello");
        let owned = text.into_owned();
        assert!(matches!(owned, OwnedMessage::Text(s) if s == "hello"));

        let binary = Message::Binary(&[1, 2, 3]);
        let owned = binary.into_owned();
        assert!(matches!(owned, OwnedMessage::Binary(b) if b == vec![1, 2, 3]));

        let close = Message::Close(CloseFrame {
            code: CloseCode::Normal,
            reason: "bye",
        });
        let owned = close.into_owned();
        assert!(matches!(
            owned,
            OwnedMessage::Close(OwnedCloseFrame { code: CloseCode::Normal, reason }) if reason == "bye"
        ));
    }

    #[test]
    fn owned_message_as_bytes() {
        assert_eq!(OwnedMessage::Text("hello".into()).as_bytes(), b"hello");
        assert_eq!(OwnedMessage::Binary(vec![1, 2, 3]).as_bytes(), &[1, 2, 3]);
        assert_eq!(OwnedMessage::Ping(vec![4, 5]).as_bytes(), &[4, 5]);
        assert_eq!(OwnedMessage::Pong(vec![6]).as_bytes(), &[6]);
        // Close returns reason bytes only (excludes 2-byte status code)
        let close = OwnedMessage::Close(OwnedCloseFrame {
            code: CloseCode::Normal,
            reason: "bye".into(),
        });
        assert_eq!(close.as_bytes(), b"bye");
    }

    #[cfg(feature = "bytes")]
    #[test]
    fn owned_message_to_bytes() {
        let text = OwnedMessage::Text("hello".into());
        let b = text.to_bytes();
        assert_eq!(&b[..], b"hello");

        let binary = OwnedMessage::Binary(vec![1, 2, 3]);
        let b = binary.to_bytes();
        assert_eq!(&b[..], &[1, 2, 3]);

        let ping = OwnedMessage::Ping(vec![4, 5]);
        let b = ping.to_bytes();
        assert_eq!(&b[..], &[4, 5]);

        // Close → reason bytes only
        let close = OwnedMessage::Close(OwnedCloseFrame {
            code: CloseCode::Normal,
            reason: "bye".into(),
        });
        let b = close.to_bytes();
        assert_eq!(&b[..], b"bye");
    }
}

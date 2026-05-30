/// Protocol error from WebSocket frame decoding.
///
/// Each variant is a specific RFC 6455 violation. No catch-all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    /// Frame header contains an unrecognized opcode.
    InvalidOpcode(u8),
    /// Reserved bits (RSV1-3) are set without a negotiated extension.
    ReservedBitsSet {
        /// The RSV bits that were set (bits 4-6 of byte 0).
        bits: u8,
    },
    /// Server sent a masked frame (RFC 6455 §5.1: server MUST NOT mask).
    MaskedFrameFromServer,
    /// Client sent an unmasked frame (RFC 6455 §5.1: client MUST mask).
    UnmaskedFrameFromClient,
    /// Frame payload exceeds the configured maximum frame size.
    PayloadTooLarge {
        /// Declared payload size.
        size: u64,
        /// Configured maximum.
        max: u64,
    },
    /// Control frame payload exceeds 125 bytes (RFC 6455 §5.5).
    ControlFrameTooLarge {
        /// Declared payload size.
        size: u64,
    },
    /// Control frame is fragmented (RFC 6455 §5.5: MUST NOT be fragmented).
    FragmentedControlFrame,
    /// Close frame has invalid status code.
    InvalidCloseCode(u16),
    /// Close frame reason is not valid UTF-8.
    InvalidUtf8InCloseReason,
    /// Close frame payload is 1 byte (must be 0 or >= 2).
    CloseFrameTooShort,
    /// Received a continuation frame with no preceding start frame.
    ContinuationWithoutStart,
    /// Received a new data frame (Text/Binary) while assembling fragments.
    NewMessageDuringAssembly,
    /// Text message payload is not valid UTF-8.
    InvalidUtf8,
    /// Assembled message exceeds the configured maximum message size.
    MessageTooLarge {
        /// Accumulated size so far.
        accumulated: usize,
        /// Configured maximum.
        max: usize,
    },
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOpcode(op) => write!(f, "invalid opcode: 0x{op:X}"),
            Self::ReservedBitsSet { bits } => {
                write!(f, "reserved bits set: 0b{bits:03b}")
            }
            Self::MaskedFrameFromServer => write!(f, "server sent masked frame"),
            Self::UnmaskedFrameFromClient => write!(f, "client sent unmasked frame"),
            Self::PayloadTooLarge { size, max } => {
                write!(f, "payload too large: {size} bytes (max {max})")
            }
            Self::ControlFrameTooLarge { size } => {
                write!(f, "control frame too large: {size} bytes (max 125)")
            }
            Self::FragmentedControlFrame => write!(f, "fragmented control frame"),
            Self::InvalidCloseCode(code) => write!(f, "invalid close code: {code}"),
            Self::InvalidUtf8InCloseReason => write!(f, "invalid UTF-8 in close reason"),
            Self::CloseFrameTooShort => {
                write!(f, "close frame too short (1 byte, must be 0 or >= 2)")
            }
            Self::ContinuationWithoutStart => {
                write!(f, "continuation frame without preceding start frame")
            }
            Self::NewMessageDuringAssembly => {
                write!(f, "new data frame received during fragment assembly")
            }
            Self::InvalidUtf8 => write!(f, "text message contains invalid UTF-8"),
            Self::MessageTooLarge { accumulated, max } => {
                write!(
                    f,
                    "assembled message too large: {accumulated} bytes (max {max})"
                )
            }
        }
    }
}

impl std::error::Error for ProtocolError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_invalid_opcode() {
        let err = ProtocolError::InvalidOpcode(0x3);
        assert_eq!(err.to_string(), "invalid opcode: 0x3");
    }

    #[test]
    fn display_reserved_bits_set() {
        let err = ProtocolError::ReservedBitsSet { bits: 0b110 };
        assert_eq!(err.to_string(), "reserved bits set: 0b110");
    }

    #[test]
    fn display_masked_frame_from_server() {
        assert_eq!(
            ProtocolError::MaskedFrameFromServer.to_string(),
            "server sent masked frame"
        );
    }

    #[test]
    fn display_unmasked_frame_from_client() {
        assert_eq!(
            ProtocolError::UnmaskedFrameFromClient.to_string(),
            "client sent unmasked frame"
        );
    }

    #[test]
    fn display_payload_too_large() {
        let err = ProtocolError::PayloadTooLarge {
            size: 200,
            max: 125,
        };
        assert_eq!(err.to_string(), "payload too large: 200 bytes (max 125)");
    }

    #[test]
    fn display_control_frame_too_large() {
        let err = ProtocolError::ControlFrameTooLarge { size: 130 };
        assert_eq!(
            err.to_string(),
            "control frame too large: 130 bytes (max 125)"
        );
    }

    #[test]
    fn display_fragmented_control_frame() {
        assert_eq!(
            ProtocolError::FragmentedControlFrame.to_string(),
            "fragmented control frame"
        );
    }

    #[test]
    fn display_invalid_close_code() {
        let err = ProtocolError::InvalidCloseCode(999);
        assert_eq!(err.to_string(), "invalid close code: 999");
    }

    #[test]
    fn display_invalid_utf8_in_close_reason() {
        assert_eq!(
            ProtocolError::InvalidUtf8InCloseReason.to_string(),
            "invalid UTF-8 in close reason"
        );
    }

    #[test]
    fn display_close_frame_too_short() {
        assert_eq!(
            ProtocolError::CloseFrameTooShort.to_string(),
            "close frame too short (1 byte, must be 0 or >= 2)"
        );
    }

    #[test]
    fn display_continuation_without_start() {
        assert_eq!(
            ProtocolError::ContinuationWithoutStart.to_string(),
            "continuation frame without preceding start frame"
        );
    }

    #[test]
    fn display_new_message_during_assembly() {
        assert_eq!(
            ProtocolError::NewMessageDuringAssembly.to_string(),
            "new data frame received during fragment assembly"
        );
    }

    #[test]
    fn display_invalid_utf8() {
        assert_eq!(
            ProtocolError::InvalidUtf8.to_string(),
            "text message contains invalid UTF-8"
        );
    }

    #[test]
    fn display_message_too_large() {
        let err = ProtocolError::MessageTooLarge {
            accumulated: 2000,
            max: 1024,
        };
        assert_eq!(
            err.to_string(),
            "assembled message too large: 2000 bytes (max 1024)"
        );
    }

    #[test]
    fn protocol_error_eq() {
        assert_eq!(
            ProtocolError::InvalidOpcode(0x3),
            ProtocolError::InvalidOpcode(0x3)
        );
        assert_ne!(
            ProtocolError::InvalidOpcode(0x3),
            ProtocolError::InvalidOpcode(0x4)
        );
        assert_ne!(
            ProtocolError::MaskedFrameFromServer,
            ProtocolError::UnmaskedFrameFromClient
        );
    }
}

/// Wire-level opcode (internal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RawOpcode {
    Continuation,
    Text,
    Binary,
    Close,
    Ping,
    Pong,
}

impl RawOpcode {
    pub(crate) fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x0 => Some(Self::Continuation),
            0x1 => Some(Self::Text),
            0x2 => Some(Self::Binary),
            0x8 => Some(Self::Close),
            0x9 => Some(Self::Ping),
            0xA => Some(Self::Pong),
            _ => None,
        }
    }

    pub(crate) fn is_control(self) -> bool {
        matches!(self, Self::Close | Self::Ping | Self::Pong)
    }
}

/// Determines masking behavior per RFC 6455.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// WebSocket client: must mask outbound, expects unmasked inbound.
    Client,
    /// WebSocket server: must not mask outbound, expects masked inbound.
    Server,
}

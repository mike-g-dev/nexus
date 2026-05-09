//! Error types for ID string parsing.
//!
//! Each error type contains only the variants reachable by the parse methods
//! that return it. Users never need unreachable match arms.
//!
//! | Error Type | Used By |
//! |---|---|
//! | [`ParseError`] | [`HexId64`], [`UuidCompact`], [`Ulid`] |
//! | [`UuidParseError`] | [`Uuid`] |
//! | [`DecodeError`] | [`Base62Id`], [`Base36Id`] |
//! | [`TypeIdParseError`] | [`TypeId`] |

use core::fmt;

// =============================================================================
// ParseError — InvalidLength + InvalidChar
// =============================================================================

/// Error parsing a fixed-format ID string.
///
/// Returned by [`HexId64`](crate::HexId64), [`UuidCompact`](crate::UuidCompact),
/// and [`Ulid`](crate::Ulid).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Input length doesn't match expected format length.
    InvalidLength {
        /// The length the format requires.
        expected: usize,
        /// The length of the input that was provided.
        got: usize,
    },
    /// Invalid character at the given position.
    InvalidChar {
        /// Byte index in the input where the invalid character was found.
        position: usize,
        /// The invalid byte value.
        byte: u8,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { expected, got } => {
                write!(f, "invalid length: expected {}, got {}", expected, got)
            }
            Self::InvalidChar { position, byte } => {
                write!(
                    f,
                    "invalid character 0x{:02x} at position {}",
                    byte, position
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {}

// =============================================================================
// UuidParseError — InvalidLength + InvalidChar + InvalidFormat
// =============================================================================

/// Error parsing a UUID string.
///
/// Returned by [`Uuid`](crate::Uuid).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UuidParseError {
    /// Input length doesn't match expected format length.
    InvalidLength {
        /// The length the format requires.
        expected: usize,
        /// The length of the input that was provided.
        got: usize,
    },
    /// Invalid character at the given position.
    InvalidChar {
        /// Byte index in the input where the invalid character was found.
        position: usize,
        /// The invalid byte value.
        byte: u8,
    },
    /// Structural format error (missing or misplaced dashes).
    InvalidFormat,
}

impl fmt::Display for UuidParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { expected, got } => {
                write!(f, "invalid length: expected {}, got {}", expected, got)
            }
            Self::InvalidChar { position, byte } => {
                write!(
                    f,
                    "invalid character 0x{:02x} at position {}",
                    byte, position
                )
            }
            Self::InvalidFormat => write!(f, "invalid UUID format (expected dashes at 8-13-18-23)"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for UuidParseError {}

impl From<ParseError> for UuidParseError {
    #[inline]
    fn from(e: ParseError) -> Self {
        match e {
            ParseError::InvalidLength { expected, got } => Self::InvalidLength { expected, got },
            ParseError::InvalidChar { position, byte } => Self::InvalidChar { position, byte },
        }
    }
}

// =============================================================================
// DecodeError — InvalidLength + InvalidChar + Overflow
// =============================================================================

/// Error parsing a base-N encoded ID string.
///
/// Returned by [`Base62Id`](crate::Base62Id) and [`Base36Id`](crate::Base36Id).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Input length doesn't match expected format length.
    InvalidLength {
        /// The length the format requires.
        expected: usize,
        /// The length of the input that was provided.
        got: usize,
    },
    /// Invalid character at the given position.
    InvalidChar {
        /// Byte index in the input where the invalid character was found.
        position: usize,
        /// The invalid byte value.
        byte: u8,
    },
    /// Value overflows the target integer type.
    Overflow,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { expected, got } => {
                write!(f, "invalid length: expected {}, got {}", expected, got)
            }
            Self::InvalidChar { position, byte } => {
                write!(
                    f,
                    "invalid character 0x{:02x} at position {}",
                    byte, position
                )
            }
            Self::Overflow => write!(f, "value overflows target type"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecodeError {}

impl From<ParseError> for DecodeError {
    #[inline]
    fn from(e: ParseError) -> Self {
        match e {
            ParseError::InvalidLength { expected, got } => Self::InvalidLength { expected, got },
            ParseError::InvalidChar { position, byte } => Self::InvalidChar { position, byte },
        }
    }
}

// =============================================================================
// TypeIdParseError — InvalidLength + InvalidChar + InvalidFormat + InvalidPrefix
// =============================================================================

/// Error parsing or constructing a TypeId.
///
/// Returned by [`TypeId`](crate::TypeId).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeIdParseError {
    /// Input length doesn't match expected capacity or format.
    InvalidLength {
        /// The length the format requires.
        expected: usize,
        /// The length of the input that was provided.
        got: usize,
    },
    /// Invalid character at the given position.
    InvalidChar {
        /// Byte index in the input where the invalid character was found.
        position: usize,
        /// The invalid byte value.
        byte: u8,
    },
    /// Structural format error (missing underscore separator).
    InvalidFormat,
    /// Prefix is empty or contains non-lowercase-ASCII characters.
    InvalidPrefix,
}

impl fmt::Display for TypeIdParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { expected, got } => {
                write!(f, "invalid length: expected max {}, got {}", expected, got)
            }
            Self::InvalidChar { position, byte } => {
                write!(
                    f,
                    "invalid character 0x{:02x} at position {}",
                    byte, position
                )
            }
            Self::InvalidFormat => {
                write!(f, "invalid TypeId format (expected prefix_suffix)")
            }
            Self::InvalidPrefix => {
                write!(f, "invalid prefix: must be non-empty lowercase ASCII [a-z]")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TypeIdParseError {}

impl From<ParseError> for TypeIdParseError {
    #[inline]
    fn from(e: ParseError) -> Self {
        match e {
            ParseError::InvalidLength { expected, got } => Self::InvalidLength { expected, got },
            ParseError::InvalidChar { position, byte } => Self::InvalidChar { position, byte },
        }
    }
}

// =============================================================================
// Validation helpers
// =============================================================================

/// Validate and decode a hex character, returning value or error.
#[allow(dead_code)] // Utility available for future parse implementations
#[inline]
pub(crate) const fn validate_hex(b: u8, position: usize) -> Result<u8, ParseError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(ParseError::InvalidChar { position, byte: b }),
    }
}

/// Sentinel value for invalid Crockford Base32 characters.
const CROCKFORD32_INVALID: u8 = 0xFF;

/// Lookup table: ASCII byte → Crockford Base32 value (0-31), or 0xFF if invalid.
/// 256 bytes = 4 cache lines. Stays warm after first use.
pub(crate) static CROCKFORD32_DECODE: [u8; 256] = {
    let mut table = [CROCKFORD32_INVALID; 256];

    // Digits 0-9
    table[b'0' as usize] = 0;
    table[b'1' as usize] = 1;
    table[b'2' as usize] = 2;
    table[b'3' as usize] = 3;
    table[b'4' as usize] = 4;
    table[b'5' as usize] = 5;
    table[b'6' as usize] = 6;
    table[b'7' as usize] = 7;
    table[b'8' as usize] = 8;
    table[b'9' as usize] = 9;

    // Letters (uppercase) — Crockford excludes I, L, O, U
    table[b'A' as usize] = 10;
    table[b'B' as usize] = 11;
    table[b'C' as usize] = 12;
    table[b'D' as usize] = 13;
    table[b'E' as usize] = 14;
    table[b'F' as usize] = 15;
    table[b'G' as usize] = 16;
    table[b'H' as usize] = 17;
    table[b'J' as usize] = 18;
    table[b'K' as usize] = 19;
    table[b'M' as usize] = 20;
    table[b'N' as usize] = 21;
    table[b'P' as usize] = 22;
    table[b'Q' as usize] = 23;
    table[b'R' as usize] = 24;
    table[b'S' as usize] = 25;
    table[b'T' as usize] = 26;
    table[b'V' as usize] = 27;
    table[b'W' as usize] = 28;
    table[b'X' as usize] = 29;
    table[b'Y' as usize] = 30;
    table[b'Z' as usize] = 31;

    // Letters (lowercase)
    table[b'a' as usize] = 10;
    table[b'b' as usize] = 11;
    table[b'c' as usize] = 12;
    table[b'd' as usize] = 13;
    table[b'e' as usize] = 14;
    table[b'f' as usize] = 15;
    table[b'g' as usize] = 16;
    table[b'h' as usize] = 17;
    table[b'j' as usize] = 18;
    table[b'k' as usize] = 19;
    table[b'm' as usize] = 20;
    table[b'n' as usize] = 21;
    table[b'p' as usize] = 22;
    table[b'q' as usize] = 23;
    table[b'r' as usize] = 24;
    table[b's' as usize] = 25;
    table[b't' as usize] = 26;
    table[b'v' as usize] = 27;
    table[b'w' as usize] = 28;
    table[b'x' as usize] = 29;
    table[b'y' as usize] = 30;
    table[b'z' as usize] = 31;

    // Crockford aliases
    table[b'O' as usize] = 0; // O → 0
    table[b'o' as usize] = 0;
    table[b'I' as usize] = 1; // I → 1
    table[b'i' as usize] = 1;
    table[b'L' as usize] = 1; // L → 1
    table[b'l' as usize] = 1;

    table
};

/// Validate and decode a Crockford Base32 character via lookup table.
#[inline]
pub(crate) fn validate_crockford32(b: u8, position: usize) -> Result<u8, ParseError> {
    let val = CROCKFORD32_DECODE[b as usize];
    if val == CROCKFORD32_INVALID {
        Err(ParseError::InvalidChar { position, byte: b })
    } else {
        Ok(val)
    }
}

/// Validate and decode a base62 character.
#[inline]
pub(crate) const fn validate_base62(b: u8, position: usize) -> Result<u8, ParseError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'A'..=b'Z' => Ok(b - b'A' + 10),
        b'a'..=b'z' => Ok(b - b'a' + 36),
        _ => Err(ParseError::InvalidChar { position, byte: b }),
    }
}

/// Validate and decode a base36 character.
#[inline]
pub(crate) const fn validate_base36(b: u8, position: usize) -> Result<u8, ParseError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'z' => Ok(b - b'a' + 10),
        b'A'..=b'Z' => Ok(b - b'A' + 10),
        _ => Err(ParseError::InvalidChar { position, byte: b }),
    }
}

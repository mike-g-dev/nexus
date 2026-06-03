use core::fmt;

/// Error during FIX message decoding.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// Message is too short to contain required header fields.
    Truncated,
    /// No `=` separator found in a tag=value field.
    MissingSeparator,
    /// Tag number is zero or contains non-digit bytes.
    InvalidTag,
    /// BeginString (tag 8) is missing or not the first field.
    MissingBeginString,
    /// BodyLength (tag 9) is missing or not the second field.
    MissingBodyLength,
    /// Checksum validation failed.
    Checksum(ChecksumError),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("message truncated"),
            Self::MissingSeparator => f.write_str("missing '=' separator in field"),
            Self::InvalidTag => f.write_str("invalid tag number"),
            Self::MissingBeginString => f.write_str("missing or misplaced BeginString (tag 8)"),
            Self::MissingBodyLength => f.write_str("missing or misplaced BodyLength (tag 9)"),
            Self::Checksum(e) => write!(f, "checksum: {}", e),
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Checksum(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ChecksumError> for DecodeError {
    fn from(e: ChecksumError) -> Self {
        Self::Checksum(e)
    }
}

/// Checksum validation failure.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ChecksumError {
    pub expected: u8,
    pub computed: u8,
}

impl fmt::Display for ChecksumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "expected {:03}, computed {:03}",
            self.expected, self.computed
        )
    }
}

impl std::error::Error for ChecksumError {}

/// Value-level parse failure.
///
/// Returned when a field's value bytes are present but cannot be parsed
/// into the requested type. This is distinct from two other concerns:
/// frame-structure errors ([`DecodeError`]) and field *absence* — an
/// optional field that simply was not sent is modeled as `Option` at the
/// lookup layer, never as an error here. A present-but-empty value
/// (`44=\x01`) is [`FixValueError::Empty`], not absence.
///
/// `Copy` and allocation-free; an error value is only constructed on the
/// cold failure path, so it costs nothing on a successful parse.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FixValueError {
    /// Field was present but its value was zero-length.
    Empty,
    /// A non-digit byte appeared where digits were required.
    NotNumeric,
    /// The digits were valid but the value exceeds the target type's range.
    Overflow,
    /// Structurally well-formed but semantically invalid (month 13,
    /// hour 24, a tenor count of 0, ...).
    OutOfRange,
    /// The value does not match the expected shape (missing separator,
    /// wrong fixed width, bad sign placement, unknown unit letter, ...).
    BadFormat,
    /// A text field contained a control or non-ASCII byte.
    NotPrintable,
}

impl fmt::Display for FixValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "empty field value",
            Self::NotNumeric => "non-digit byte in numeric field",
            Self::Overflow => "value exceeds target range",
            Self::OutOfRange => "value out of valid range",
            Self::BadFormat => "malformed field value",
            Self::NotPrintable => "non-printable byte in text field",
        })
    }
}

impl std::error::Error for FixValueError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_error_display() {
        let e = ChecksumError {
            expected: 178,
            computed: 42,
        };
        assert_eq!(e.to_string(), "expected 178, computed 042");
    }

    #[test]
    fn decode_error_from_checksum() {
        let ce = ChecksumError {
            expected: 1,
            computed: 2,
        };
        let de: DecodeError = ce.into();
        assert_eq!(de, DecodeError::Checksum(ce));
    }

    #[test]
    fn decode_error_display() {
        assert_eq!(DecodeError::Truncated.to_string(), "message truncated");
        assert_eq!(
            DecodeError::MissingSeparator.to_string(),
            "missing '=' separator in field"
        );
        assert_eq!(DecodeError::InvalidTag.to_string(), "invalid tag number");
        assert_eq!(
            DecodeError::MissingBeginString.to_string(),
            "missing or misplaced BeginString (tag 8)"
        );
        assert_eq!(
            DecodeError::MissingBodyLength.to_string(),
            "missing or misplaced BodyLength (tag 9)"
        );
        let ce = ChecksumError {
            expected: 178,
            computed: 42,
        };
        assert_eq!(
            DecodeError::Checksum(ce).to_string(),
            "checksum: expected 178, computed 042"
        );
    }

    #[test]
    fn decode_error_source_chain() {
        use std::error::Error;

        assert!(DecodeError::Truncated.source().is_none());
        assert!(DecodeError::MissingSeparator.source().is_none());
        assert!(DecodeError::InvalidTag.source().is_none());

        let ce = ChecksumError {
            expected: 1,
            computed: 2,
        };
        let de = DecodeError::Checksum(ce);
        let src = de.source().unwrap();
        let downcasted = src.downcast_ref::<ChecksumError>().unwrap();
        assert_eq!(*downcasted, ce);
    }

    #[test]
    fn fix_value_error_display() {
        assert_eq!(FixValueError::Empty.to_string(), "empty field value");
        assert_eq!(
            FixValueError::NotNumeric.to_string(),
            "non-digit byte in numeric field"
        );
        assert_eq!(
            FixValueError::Overflow.to_string(),
            "value exceeds target range"
        );
        assert_eq!(
            FixValueError::OutOfRange.to_string(),
            "value out of valid range"
        );
        assert_eq!(
            FixValueError::BadFormat.to_string(),
            "malformed field value"
        );
        assert_eq!(
            FixValueError::NotPrintable.to_string(),
            "non-printable byte in text field"
        );
    }
}

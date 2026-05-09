/// HTTP parsing error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpError {
    /// Request or response head is malformed.
    Malformed(&'static str),
    /// Too many headers (exceeds configured limit).
    TooManyHeaders,
    /// Head section exceeds size limit.
    HeadTooLarge {
        /// Configured maximum head size in bytes.
        max: usize,
    },
    /// Read buffer full.
    BufferFull {
        /// Bytes required to make progress.
        needed: usize,
        /// Bytes currently free in the buffer.
        available: usize,
    },
    /// Write buffer too small for the HTTP message.
    BufferTooSmall {
        /// Bytes required to write the message.
        needed: usize,
        /// Bytes available in the supplied buffer.
        available: usize,
    },
    /// Header name or value contains invalid characters (CR/LF).
    InvalidHeaderValue,
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(ctx) => write!(f, "malformed HTTP: {ctx}"),
            Self::TooManyHeaders => write!(f, "too many HTTP headers"),
            Self::HeadTooLarge { max } => write!(f, "HTTP head exceeds {max} bytes"),
            Self::BufferFull { needed, available } => {
                write!(f, "buffer full: need {needed}, {available} available")
            }
            Self::BufferTooSmall { needed, available } => {
                write!(
                    f,
                    "write buffer too small: need {needed} bytes, have {available}"
                )
            }
            Self::InvalidHeaderValue => {
                write!(f, "header name or value contains CR/LF")
            }
        }
    }
}

impl std::error::Error for HttpError {}

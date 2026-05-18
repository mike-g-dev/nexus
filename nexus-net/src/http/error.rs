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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_error_malformed() {
        let err = HttpError::Malformed("missing status line");
        assert!(matches!(err, HttpError::Malformed("missing status line")));
        assert_eq!(err.to_string(), "malformed HTTP: missing status line");
    }

    #[test]
    fn http_error_too_many_headers() {
        let err = HttpError::TooManyHeaders;
        assert!(matches!(err, HttpError::TooManyHeaders));
        assert_eq!(err.to_string(), "too many HTTP headers");
    }

    #[test]
    fn http_error_head_too_large() {
        let err = HttpError::HeadTooLarge { max: 8192 };
        assert!(matches!(err, HttpError::HeadTooLarge { max: 8192 }));
        assert_eq!(err.to_string(), "HTTP head exceeds 8192 bytes");
    }

    #[test]
    fn http_error_buffer_full() {
        let err = HttpError::BufferFull {
            needed: 1024,
            available: 256,
        };
        assert!(matches!(
            err,
            HttpError::BufferFull {
                needed: 1024,
                available: 256,
            }
        ));
        assert_eq!(err.to_string(), "buffer full: need 1024, 256 available");
    }

    #[test]
    fn http_error_buffer_too_small() {
        let err = HttpError::BufferTooSmall {
            needed: 512,
            available: 128,
        };
        assert!(matches!(
            err,
            HttpError::BufferTooSmall {
                needed: 512,
                available: 128,
            }
        ));
        assert_eq!(
            err.to_string(),
            "write buffer too small: need 512 bytes, have 128"
        );
    }

    #[test]
    fn http_error_invalid_header_value() {
        let err = HttpError::InvalidHeaderValue;
        assert!(matches!(err, HttpError::InvalidHeaderValue));
        assert_eq!(err.to_string(), "header name or value contains CR/LF");
    }

    #[test]
    fn http_error_eq() {
        assert_eq!(HttpError::TooManyHeaders, HttpError::TooManyHeaders);
        assert_ne!(HttpError::TooManyHeaders, HttpError::InvalidHeaderValue);
        assert_eq!(
            HttpError::HeadTooLarge { max: 100 },
            HttpError::HeadTooLarge { max: 100 }
        );
    }
}

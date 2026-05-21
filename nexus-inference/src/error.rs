use core::fmt;

/// Errors during model loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// Model file is malformed or missing required fields.
    Parse(&'static str),
    /// Feature count, tree structure, or metadata is inconsistent.
    Validation(&'static str),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(msg) => write!(f, "parse error: {msg}"),
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LoadError {}

/// Input contains NaN.
///
/// Returned by checked `predict` / `predict_into` methods on MLP and LUT
/// when any input value is NaN. Use `predict_unchecked` to skip the scan
/// and let NaN propagate through the computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NanInput;

impl fmt::Display for NanInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("input contains NaN")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NanInput {}

use core::fmt;

/// Errors during model loading.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LoadError {
    /// Model file is malformed or missing required fields.
    Parse(&'static str),
    /// Feature count, tree structure, or metadata is inconsistent.
    Validation(&'static str),
    /// A required tensor was not found in the model file.
    TensorNotFound(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(msg) => write!(f, "parse error: {msg}"),
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
        }
    }
}

impl std::error::Error for LoadError {}

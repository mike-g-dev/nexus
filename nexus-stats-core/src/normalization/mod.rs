//! Online feature normalization.

#[cfg(any(feature = "std", feature = "libm"))]
mod zscore;
#[cfg(any(feature = "std", feature = "libm"))]
pub use zscore::*;

mod minmax;
pub use minmax::*;

#[cfg(feature = "alloc")]
mod quantile;
#[cfg(feature = "alloc")]
pub use quantile::*;

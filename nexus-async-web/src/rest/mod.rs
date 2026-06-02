//! Async HTTP REST client — adapts nexus-web for async runtimes.

#[cfg(feature = "nexus")]
mod nexus;
#[cfg(feature = "tokio-rt")]
mod tokio;

#[cfg(feature = "nexus")]
pub use self::nexus::*;
#[cfg(feature = "tokio-rt")]
pub use self::tokio::*;

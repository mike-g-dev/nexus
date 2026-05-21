#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! ML inference engine for pre-trained models.
//!
//! This crate provides low-latency inference for models trained in
//! external frameworks (LightGBM). No training — just fast prediction
//! on the hot path.
//!
//! # Available Types
//!
//! - [`GbdtF64`] / [`GbdtF32`] — Gradient-boosted decision tree ensemble

#[cfg(feature = "alloc")]
extern crate alloc;

mod error;
mod gbdt;

#[cfg(feature = "loader-lightgbm")]
mod loader;

pub use error::LoadError;
#[cfg(feature = "alloc")]
pub use gbdt::{GbdtF32, GbdtF64};

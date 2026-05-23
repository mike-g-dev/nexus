#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! Real-time CPU inference for small, pre-trained models.
//!
//! This crate runs tiny models on the CPU — the kind that sit in an
//! event loop and make decisions at wire speed. Models are trained
//! externally (LightGBM, PyTorch), loaded once via `from_parts`, and
//! called millions of times. Sub-microsecond prediction, zero allocation
//! after construction.
//!
//! # Stateless (single prediction)
//!
//! - [`GbdtF64`] / [`GbdtF32`] — Gradient-boosted decision tree ensemble
//! - [`MlpF64`] / [`MlpF32`] — Feedforward neural network (multi-layer perceptron)
//! - [`LutF64`] / [`LutF32`] — Lookup table predictor (discretized features)
//!
//! # Stateful (streaming temporal)
//!
//! - [`TinyLstmF32`] — LSTM with hidden and cell state carried between steps
//! - [`TinyGruF32`] — GRU, ~75% of LSTM cost, simpler memory model
//! - [`Causal1dConvF32`] — Causal 1D convolution over a sliding window

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
mod activation;
#[cfg(feature = "alloc")]
mod dot;
mod error;
mod gbdt;
mod lut;
mod mlp;

#[cfg(any(feature = "std", feature = "libm"))]
mod rnn;

#[cfg(feature = "alloc")]
mod conv;

#[cfg(feature = "loader-lightgbm")]
mod loader;

#[cfg(feature = "alloc")]
pub use activation::Activation;
#[cfg(feature = "alloc")]
pub use conv::Causal1dConvF32;
pub use error::LoadError;
#[cfg(feature = "alloc")]
pub use gbdt::{GbdtF32, GbdtF64};
#[cfg(feature = "alloc")]
pub use lut::{LutF32, LutF64};
#[cfg(feature = "alloc")]
pub use mlp::{MlpF32, MlpF64};
#[cfg(any(feature = "std", feature = "libm"))]
pub use rnn::{TinyGruF32, TinyLstmF32};

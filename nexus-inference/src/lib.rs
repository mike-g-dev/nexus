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
//! - [`Gbdt`] — Gradient-boosted decision tree ensemble
//! - [`Mlp`] — Feedforward neural network (multi-layer perceptron)
//! - [`Lut`] — Lookup table predictor (discretized features)
//! - [`Bnn`] — Binary neural network (XNOR+popcount inference)
//! - [`QuantizedMlp`] — Int8-quantized MLP (i8 matmul, f32 activations)
//!
//! # Stateful (streaming temporal)
//!
//! - [`TinyLstm`] — Single-layer LSTM
//! - [`TinyGru`] — Single-layer GRU, ~75% of LSTM cost
//! - [`StackedLstm`] — Multi-layer LSTM (PyTorch `nn.LSTM(num_layers=N)`)
//! - [`StackedGru`] — Multi-layer GRU (PyTorch `nn.GRU(num_layers=N)`)
//! - [`LinearSsm`] — Linear state-space model (S4/S4D), no transcendentals
//! - [`Causal1dConv`] — Causal 1D convolution over a sliding window
//! - [`TinyTcn`] — Temporal convolutional network (dilated causal conv stack)

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
#[cfg(feature = "alloc")]
mod quantized_mlp;

#[cfg(any(feature = "std", feature = "libm"))]
mod rnn;

#[cfg(feature = "alloc")]
mod bnn;

#[cfg(feature = "alloc")]
mod ssm;

#[cfg(feature = "alloc")]
mod conv;

#[cfg(any(feature = "loader-lightgbm", feature = "safetensors"))]
mod loader;

#[cfg(feature = "alloc")]
pub use activation::Activation;
#[cfg(feature = "alloc")]
pub use bnn::Bnn;
#[cfg(feature = "alloc")]
pub use conv::{Causal1dConv, TinyTcn};
pub use error::LoadError;
#[cfg(feature = "alloc")]
pub use gbdt::Gbdt;
#[cfg(feature = "alloc")]
pub use lut::Lut;
#[cfg(feature = "alloc")]
pub use mlp::Mlp;
#[cfg(feature = "alloc")]
pub use quantized_mlp::QuantizedMlp;
#[cfg(any(feature = "std", feature = "libm"))]
pub use rnn::{StackedGru, StackedLstm, TinyGru, TinyLstm};
#[cfg(feature = "alloc")]
pub use ssm::LinearSsm;

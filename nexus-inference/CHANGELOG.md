# Changelog

All notable changes to nexus-inference are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **`MlpF64` / `MlpF32`** — Feedforward neural network inference.
  Runtime-configured layer sizes, row-major weight layout (PyTorch-compatible).
  Activation functions: `Relu`, `LeakyRelu`, `Tanh`, `Sigmoid` (hidden layers only,
  output layer is raw linear). `predict()` for single-output, `predict_into()`
  for multi-output. Requires `alloc`. `Tanh`/`Sigmoid` require `std` or `libm`.
- **`LutF64` / `LutF32`** — Lookup table predictor. Uniform bin spacing,
  Horner indexing, clamped out-of-range features. O(1) prediction.
  Requires `alloc`.
- **`Activation`** enum — `Relu`, `LeakyRelu(f64)`, `Tanh`, `Sigmoid`.
- **GBDT API additions** — `predict_into()`, `predict_into_unchecked()`,
  `n_outputs()` for API consistency with MLP and LUT.
- **`libm` feature** — enables `Tanh`/`Sigmoid` activations in `no_std`
  environments via the `libm` crate.

## [0.1.0] — 2026-05-21

### Added

- **`GbdtF64` / `GbdtF32`** — Gradient-boosted decision tree ensemble inference.
  Flat node arrays, depth-first layout, 16-byte nodes. `predict()` with
  NaN routing (LightGBM-compatible), `predict_unchecked()` without NaN checks,
  `predict_n()` for partial ensemble evaluation. Requires `alloc`.
- **LightGBM text format loader** — `GbdtF64::from_lightgbm(&[u8])` /
  `GbdtF32::from_lightgbm(&[u8])`. Parses LightGBM model text files.
  Requires `loader-lightgbm` feature (implies `std`).

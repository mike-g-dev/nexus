# Changelog

All notable changes to nexus-inference are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-05-21

### Added

- **`GbdtF64` / `GbdtF32`** — Gradient-boosted decision tree ensemble inference.
  Flat node arrays, depth-first layout, 16-byte nodes. `predict()` with
  NaN routing (LightGBM-compatible), `predict_unchecked()` without NaN checks,
  `predict_n()` for partial ensemble evaluation. Requires `alloc`.
- **LightGBM text format loader** — `GbdtF64::from_lightgbm(&[u8])` /
  `GbdtF32::from_lightgbm(&[u8])`. Parses LightGBM model text files.
  Requires `loader-lightgbm` feature (implies `std`).

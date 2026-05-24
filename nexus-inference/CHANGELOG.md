# Changelog

All notable changes to nexus-inference are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **`bias=False` support** — MLP safetensors loader handles PyTorch
  `nn.Linear(bias=False)`. Missing bias tensors are treated as zero bias.
- **BatchNorm fusion** — `nn.BatchNorm1d` layers between linear layers
  are detected by `running_mean` presence and fused into the preceding
  linear layer's weights at load time. `fused_weight = scale * W`,
  `fused_bias = scale * (b - mean) + beta`. Zero runtime cost. Both
  `affine=True` (learned gamma/beta) and `affine=False` (gamma=1,
  beta=0) are supported. Requires `std` or `libm` for `sqrt`.
- **LayerNorm support** — `nn.LayerNorm` layers between linear layers
  are detected by 1D `.weight` tensors (without `running_mean`) and
  applied at inference time: `y = gamma * (x - mean) / sqrt(var + eps) + beta`.
  Cannot be fused because statistics depend on each input. New
  `from_parts_with_layer_norm` constructor for manual construction.
  Uses eps=1e-5 (PyTorch default). Requires `std` or `libm`.
- **`StackedLstmF32`** — Multi-layer LSTM matching PyTorch's
  `nn.LSTM(num_layers=N)`. Each layer's hidden state feeds as input to
  the next; output projection applied only to the final layer.
  `from_safetensors` auto-detects `num_layers` from consecutive
  `weight_ih_l{k}` tensors. `from_parts` accepts per-layer weight
  slices. Same SIMD gate processing as `TinyLstmF32`.
- **`StackedGruF32`** — Multi-layer GRU matching PyTorch's
  `nn.GRU(num_layers=N)`. Same stacking model as `StackedLstmF32`,
  ~75% compute per layer. Auto-detects `num_layers` from safetensors.
- **`TinyLstmF32`** — Single-layer LSTM for streaming temporal inference.
  Four gates (input, forget, cell candidate, output) with hidden and cell
  state carried between `step` calls. Fused `(4H, I+H)` gate matrix for
  single-matmul fast path. PyTorch `nn.LSTM` weight layout. Requires `std`
  or `libm`.
- **`TinyGruF32`** — Single-layer GRU for streaming temporal inference.
  Three gates (reset, update, candidate), ~75% of LSTM compute. PyTorch
  `nn.GRU` weight layout with reset applied after hidden matmul. Requires
  `std` or `libm`.
- **`Causal1dConvF32`** — Streaming causal 1D convolution. Circular buffer,
  configurable activation, no future leakage. `is_primed()` tracks buffer
  fill. Requires `alloc`.
- **AVX2 vectorized gate processing** — Pade [7,6] rational polynomial for
  sigmoid/tanh (~1.2e-7 relative error), 8-wide SIMD gate loop for
  LSTM/GRU. 2-5x faster than scalar glibc transcendentals.
- **`matvec_bias_f32` / `matvec_f32`** — Shared tiled matrix-vector product
  helpers in dot module. 4-at-a-time processing via `dot4_f32`.
- **`MlpF64` / `MlpF32`** — Feedforward neural network inference.
  Runtime-configured layer sizes, row-major weight layout (PyTorch-compatible).
  Activation functions: `Relu`, `LeakyRelu`, `Tanh`, `Sigmoid` (hidden layers only,
  output layer is raw linear). `predict()` for single-output, `predict_into()`
  for multi-output. Requires `alloc`. `Tanh`/`Sigmoid` require `std` or `libm`.
- **`LutF64` / `LutF32`** — Lookup table predictor. Uniform bin spacing,
  Horner indexing, clamped out-of-range features. O(1) prediction.
  Requires `alloc`.
- **`Activation`** enum — `Relu`, `LeakyRelu(f64)`, `Tanh`, `Sigmoid`,
  `Identity`, `Elu(f64)`, `Gelu`, `Swish`.
- **GBDT API additions** — `predict_into()`, `predict_into_unchecked()`,
  `n_outputs()` for API consistency with MLP and LUT.
- **`libm` feature** — enables `Tanh`/`Sigmoid` activations and LSTM/GRU
  in `no_std` environments via the `libm` crate.

## [0.1.0] — 2026-05-21

### Added

- **`GbdtF64` / `GbdtF32`** — Gradient-boosted decision tree ensemble inference.
  Flat node arrays, depth-first layout, 16-byte nodes. `predict()` with
  NaN routing (LightGBM-compatible), `predict_unchecked()` without NaN checks,
  `predict_n()` for partial ensemble evaluation. Requires `alloc`.
- **LightGBM text format loader** — `GbdtF64::from_lightgbm(&[u8])` /
  `GbdtF32::from_lightgbm(&[u8])`. Parses LightGBM model text files.
  Requires `loader-lightgbm` feature (implies `alloc`).

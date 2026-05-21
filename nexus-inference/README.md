# nexus-inference

ML inference engine for pre-trained models. Sub-microsecond prediction on the hot path.

## Types

- **`GbdtF64` / `GbdtF32`** — Gradient-boosted decision tree ensemble inference.
  16-byte nodes, depth-first layout. NaN-aware prediction (LightGBM-compatible)
  and unchecked fast path.

## Features

- `std` (default) — standard library support
- `alloc` — enables `GbdtF64` / `GbdtF32` (requires heap allocation for tree storage)
- `loader-lightgbm` — LightGBM text format parser (implies `std`)

## Usage

```rust
use nexus_inference::GbdtF64;

// Load a LightGBM text model
let model = GbdtF64::from_lightgbm(model_bytes).unwrap();

// Predict with NaN routing
let score = model.predict(&features);

// Fast path: caller guarantees finite inputs
let score = model.predict_unchecked(&features);

// Partial ensemble evaluation
let score = model.predict_n(&features, 50);
```

## Performance

Target latency for typical trading models:

| Configuration | Target |
|--------------|--------|
| 50 trees × depth 6, 8 features | < 500 ns |
| 100 trees × depth 6, 8 features | < 1 µs |
| 200 trees × depth 8, 16 features | < 3 µs |

# Quickstart

## Add the dependency

```toml
[dependencies]
nexus-inference = { version = "0.1", features = ["loader-lightgbm"] }
```

## Load and predict with a GBDT

```rust
use nexus_inference::Gbdt;

// Load from LightGBM text format (model_text.txt)
let bytes = std::fs::read("model_text.txt").unwrap();
let model = Gbdt::from_lightgbm(&bytes).unwrap();

let features = vec![0.5, 1.2, -0.3, 0.8, 2.1, 0.0, -1.5, 3.3];
let score = model.predict(&features);

// NaN-aware routing (when features may contain NaN)
let score = model.predict_nan_aware(&features);
```

## Load and predict with an MLP

```rust
use nexus_inference::{Mlp, Activation};

// Weights exported from PyTorch (see python-export.md)
let layer_sizes = &[4, 8, 1];  // 4 inputs → 8 hidden → 1 output
let weights: Vec<f32> = load_weights();  // 4*8 + 8*1 = 40 values
let biases: Vec<f32> = load_biases();    // 8 + 1 = 9 values

let model = Mlp::from_parts(
    layer_sizes, &weights, &biases, Activation::Relu,
).unwrap();

let score = model.predict(&[0.5, 1.2, -0.3, 0.8]);
```

## Load and predict with a LUT

```rust
use nexus_inference::Lut;

// Pre-computed table: 2 features, 10 bins each
let table: Vec<f32> = load_table();  // 100 values

let model = Lut::from_parts(
    2,              // n_features
    10,             // n_bins
    &[0.0, 0.0],   // feature minimums
    &[1.0, 1.0],   // feature maximums
    &table,
).unwrap();

let value = model.predict(&[0.35, 0.72]);
```

## Multi-output MLP

```rust
use nexus_inference::{Mlp, Activation};

// 4 inputs → 8 hidden → 3 outputs
let model = Mlp::from_parts(
    &[4, 8, 3], &weights, &biases, Activation::Relu,
).unwrap();

// predict() panics for multi-output — use predict_into
let mut output = [0.0_f32; 3];
model.predict_into(&[0.5, 1.2, -0.3, 0.8], &mut output);
// output[0], output[1], output[2] now contain the three predictions
```

## LSTM — streaming temporal inference

```rust
use nexus_inference::TinyLstm;

// Weights exported from PyTorch nn.LSTM + nn.Linear
let mut lstm = TinyLstm::from_parts(
    4, 16, 1,   // 4 inputs, 16 hidden, 1 output
    &weight_ih, &weight_hh,
    &bias_ih, &bias_hh,
    &w_out, &b_out,
).unwrap();

// Process a sequence — state carries between calls
let score1 = lstm.predict(&[0.5, 1.2, -0.3, 0.8]);
let score2 = lstm.predict(&[0.3, 0.9, -0.1, 1.1]);

// Reset for a new sequence
lstm.reset();
```

## GRU — lighter temporal inference

```rust
use nexus_inference::TinyGru;

// Same API as LSTM, ~25% less compute
let mut gru = TinyGru::from_parts(
    4, 16, 1,
    &weight_ih, &weight_hh,
    &bias_ih, &bias_hh,
    &w_out, &b_out,
).unwrap();

let score = gru.predict(&[0.5, 1.2, -0.3, 0.8]);
```

## Causal 1D Convolution — fixed-window temporal

```rust
use nexus_inference::{Causal1dConv, Activation};

// 4 input channels, kernel 3, 8 filters, 1 output
let mut conv = Causal1dConv::from_parts(
    4, 3, 8, 1,
    &w_conv, &b_conv,
    &w_out, &b_out,
    Activation::Relu,
).unwrap();

let score = conv.predict(&[0.5, 1.0, 0.2, 0.8]);
assert!(!conv.is_primed());  // needs 3 steps to fill kernel buffer

conv.predict(&[0.3, 0.9, 0.1, 1.1]);
conv.predict(&[0.1, 0.4, 0.6, 0.3]);
assert!(conv.is_primed());   // buffer fully populated
```

## Handling errors

```rust
use nexus_inference::{Mlp, Activation, LoadError};

// Construction errors
let result = Mlp::from_parts(&[2, 0, 1], &[], &[], Activation::Relu);
match result {
    Err(LoadError::Validation(msg)) => eprintln!("bad model: {msg}"),
    Err(LoadError::Parse(msg)) => eprintln!("parse error: {msg}"),
    Ok(model) => { /* use model */ }
}
```

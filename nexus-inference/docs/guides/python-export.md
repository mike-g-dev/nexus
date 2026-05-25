# Exporting from Python

Models are trained in Python and loaded into Rust via `from_parts()`.
This guide covers how to extract weights from common frameworks.

## LightGBM → GBDT

LightGBM has a native text export. Use `from_lightgbm()` directly:

```python
import lightgbm as lgb

model = lgb.train(params, train_data)
model.save_model("model.txt")
```

```rust
let bytes = std::fs::read("model.txt").unwrap();
let model = Gbdt::from_lightgbm(&bytes).unwrap();
```

### Limitations

- **Categorical features**: Not supported. Use integer encoding
  before training.
- **Linear trees**: `is_linear=1` is rejected. Train without
  `linear_tree=True`.
- **Multiclass**: Only single-output (`num_class=1`) models.
  For multiclass, train one-vs-rest and load each as a separate model.

## PyTorch → MLP

PyTorch's `nn.Linear` stores weights in row-major (output-major)
format, which matches `from_parts()` directly.

```python
import torch
import json

model = YourModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Extract layer sizes from the architecture
layer_sizes = [model.input_dim]
weights = []
biases = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        layer_sizes.append(module.out_features)
        # .weight is already (out_features, in_features) — row-major
        weights.extend(module.weight.detach().numpy().flatten().tolist())
        biases.extend(module.bias.detach().numpy().flatten().tolist())

# Save as JSON (simple, portable)
with open("mlp_weights.json", "w") as f:
    json.dump({
        "layer_sizes": layer_sizes,
        "weights": weights,
        "biases": biases,
        "activation": "relu",
    }, f)
```

```rust
use nexus_inference::{Mlp, Activation};

let data: serde_json::Value = serde_json::from_str(&json_str).unwrap();
let layer_sizes: Vec<usize> = /* parse from data */;
let weights: Vec<f32> = /* parse from data */;
let biases: Vec<f32> = /* parse from data */;

let model = Mlp::from_parts(
    &layer_sizes, &weights, &biases, Activation::Relu,
).unwrap();
```

### Activation mapping

Map your PyTorch activation module to the Rust enum:

| PyTorch | Rust `Activation` | Note |
|---------|-------------------|------|
| `nn.ReLU` | `Relu` | |
| `nn.LeakyReLU(negative_slope=α)` | `LeakyRelu(α)` | |
| `nn.Tanh` | `Tanh` | |
| `nn.Sigmoid` | `Sigmoid` | |
| `nn.Identity` | `Identity` | |
| `nn.ELU(alpha=α)` | `Elu(α)` | typically α=1.0 |
| `nn.GELU(approximate='tanh')` | `Gelu` | must use tanh mode |
| `nn.SiLU` | `Swish` | |

**GELU approximation**: We use the tanh approximation
(`0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715x³)))`), which
matches `nn.GELU(approximate='tanh')`. If your model was trained
with `approximate='none'` (exact erf), expect ~1e-4 numerical
drift. Retrain with `approximate='tanh'` for exact match, or
accept the drift if it's within your tolerance.

### Activation limitation

Currently `from_parts` takes a single `Activation` applied to all
hidden layers. If your PyTorch model uses different activations per
layer (e.g., relu on layer 1, tanh on layer 2), you'll need to
retrain with uniform activations or wait for the per-layer builder
API (planned).

### Weight layout verification

If you're unsure about the weight layout, verify with a known input:

```python
# Python: compute expected output
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
with torch.no_grad():
    expected = model(x)
print(f"Expected: {expected.item()}")
```

```rust
// Rust: must match
let result = model.predict(&[1.0, 2.0, 3.0, 4.0]).unwrap();
assert!((result - expected).abs() < 1e-6);
```

## scikit-learn → MLP

scikit-learn's `MLPRegressor`/`MLPClassifier` stores weights
transposed relative to PyTorch — each weight matrix is
`(in_features, out_features)`. Transpose before export.

```python
from sklearn.neural_network import MLPRegressor
import json

model = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu')
model.fit(X_train, y_train)

layer_sizes = [model.n_features_in_]
weights = []
biases = []

for W, b in zip(model.coefs_, model.intercepts_):
    layer_sizes.append(W.shape[1])
    # Transpose: sklearn is (in, out), we need (out, in) row-major
    weights.extend(W.T.flatten().tolist())
    biases.extend(b.flatten().tolist())

with open("mlp_weights.json", "w") as f:
    json.dump({
        "layer_sizes": layer_sizes,
        "weights": weights,
        "biases": biases,
    }, f)
```

## Any framework → LUT

LUT tables are framework-agnostic — you just need to evaluate your
function on a grid:

```python
import numpy as np
import itertools
import json

n_features = 2
n_bins = 20
mins = [0.0, 0.0]
maxs = [1.0, 1.0]

# Compute bin centers
steps = [(maxs[i] - mins[i]) / n_bins for i in range(n_features)]
grids = [
    [mins[i] + (j + 0.5) * steps[i] for j in range(n_bins)]
    for i in range(n_features)
]

# Evaluate on grid (first feature varies slowest)
table = []
for point in itertools.product(*grids):
    table.append(your_model.predict([point])[0])

with open("lut.json", "w") as f:
    json.dump({
        "n_features": n_features,
        "n_bins": n_bins,
        "mins": mins,
        "maxs": maxs,
        "table": table,
    }, f)
```

The table ordering matters: first feature varies slowest (row-major).
`itertools.product` produces this order naturally.

## Binary formats

JSON is simple but verbose for large models. For production:

- **Raw f32 bytes**: Write weights as little-endian f32, read with
  `bytemuck::cast_slice` or manual `from_le_bytes`
- **MessagePack**: Compact binary, `rmp-serde` crate
- **Flatbuffers/Cap'n Proto**: Zero-copy deserialization

The `from_parts()` API takes slices, so any format that gives you
`&[f32]` works.

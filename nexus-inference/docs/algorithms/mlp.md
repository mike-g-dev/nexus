# MLP — Multi-Layer Perceptron

**Feedforward neural network.** Layers of neurons connected by weight
matrices, with nonlinear activation functions between layers. Learns
arbitrary continuous functions from data.

| Property | Value |
|----------|-------|
| Prediction cost | ~0.5 ns per FMA (scalar), dominated by matmul |
| Memory | `Σ(layer[i] x layer[i+1])` weights + `Σ(layer[i+1])` biases |
| Types | `Mlp` |
| Construction | `from_parts(layer_sizes, weights, biases, activation)` |
| Output | Single scalar or multi-output vector |

## What It Does

```
  Input         Hidden layer (relu)      Output (linear)
  features      ┌────────────────┐       ┌──────────┐

   x0 ──────────┤ w*x + b → relu ├──┐
                └────────────────┘  │
   x1 ──────────┤ w*x + b → relu ├──┼────┤ w*h + b ├──── score
                └────────────────┘  │    └──────────┘
   x2 ──────────┤ w*x + b → relu ├──┘
                └────────────────┘

  Forward pass for one layer:
    output[j] = activation( bias[j] + Σ(weights[j,k] * input[k]) )

  Output layer has NO activation — produces raw linear scores.
  Caller applies sigmoid/softmax if needed (same as GBDT).
```

Each layer performs a matrix-vector multiply followed by an activation
function. The network topology is defined at construction by
`layer_sizes` — e.g., `[8, 16, 1]` means 8 inputs, 16 hidden neurons
with activation, 1 linear output.

## Weight Layout

Weights are stored **row-major (output-major)** — each row contains
the weights for one output neuron. This matches PyTorch's
`nn.Linear.weight` layout directly.

```
  Layer connecting 3 inputs to 2 outputs:

  weights = [w00, w01, w02,    ← row 0: weights for output neuron 0
             w10, w11, w12]    ← row 1: weights for output neuron 1

  output[0] = bias[0] + w00*in[0] + w01*in[1] + w02*in[2]
  output[1] = bias[1] + w10*in[0] + w11*in[1] + w12*in[2]
```

All weight matrices are concatenated into a single flat array,
layer by layer. Same for biases.

## Activation Functions

A single activation function is applied to all hidden layers.
The output layer is always linear.

| Activation | Formula | Feature required | Use case |
|-----------|---------|-----------------|----------|
| `Relu` | `max(0, x)` | None | Default, most common |
| `LeakyRelu(alpha)` | `x >= 0 ? x : alpha*x` | None | Prevents dead neurons |
| `Identity` | `x` | None | No transformation |
| `Tanh` | `tanh(x)` | | Bounded output [-1, 1] |
| `Sigmoid` | `1 / (1 + exp(-x))` | | Bounded output [0, 1] |
| `Elu(alpha)` | `x >= 0 ? x : alpha*(exp(x)-1)` | | Smooth negative region |
| `Gelu` | `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` | | Transformer default |
| `Swish` | `x * sigmoid(x)` | | aka SiLU in PyTorch |

**Design note:** The current API uses a single activation for the
entire model. Per-layer activations (e.g., relu hidden + tanh final
hidden) would require a builder API and is a potential future extension.

## NaN Handling

NaN inputs propagate through the computation — the caller is
responsible for ensuring clean inputs (standard ML convention).
NaN propagates correctly through all activations:
- **Relu**: NaN passes through (three-branch comparison, matches PyTorch)
- **LeakyRelu**: `NaN * alpha = NaN`
- **Identity**: direct passthrough
- **Tanh/Sigmoid/Elu/Gelu/Swish**: transcendentals propagate NaN

Unlike GBDT, MLP has no learned NaN behavior — there is no meaningful
"default direction" for missing features.

## Scratch Buffers

The forward pass uses pre-allocated ping-pong buffers stored in
the struct. These are allocated once at construction, sized to
the maximum layer dimension. No allocation happens on the
prediction path.

Scratch buffers use interior mutability, so `predict` methods
take `&self`. Models can be shared via `Arc<Mlp>` without
contention.

## When to Use It

**Use MLP when:**
- You have a trained neural network from PyTorch/TensorFlow
- Inputs are dense numeric vectors (not sparse tabular features)
- The relationship is nonlinear and can't be tabulated
- Prediction budget is 100ns-2us (small networks, 1-3 hidden layers)

**Don't use MLP when:**
- Features are sparse/tabular with many categorical variables (use [GBDT](gbdt.md))
- The function can be precomputed over a small grid (use [LUT](lut.md))
- You need sub-10ns predictions (use [LUT](lut.md))
- Network has >128 neurons per layer (weights spill to L2)

## Code Example

```rust
use nexus_inference::{Mlp, Activation};

// 4 inputs → 8 hidden (relu) → 1 output
let model = Mlp::from_parts(
    &[4, 8, 1],
    &weights,  // 4*8 + 8*1 = 40 weights, row-major
    &biases,   // 8 + 1 = 9 biases
    Activation::Relu,
).unwrap();

let score = model.predict(&[0.5, 1.2, -0.3, 0.8]);

// Multi-output
let model = Mlp::from_parts(&[4, 8, 3], &w, &b, Activation::Relu).unwrap();
let mut output = [0.0_f32; 3];
model.predict_into(&[0.5, 1.2, -0.3, 0.8], &mut output);
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Construction | O(total_weights) | O(total_weights + total_biases) |
| `predict` | O(Σ layer[i] x layer[i+1]) | O(max_layer_size) scratch |

The cost is dominated by FMA count:

| Topology | FMAs | Latency (AVX2+FMA) |
|----------|------|--------------------|
| 8→16→1 | 144 | 53 ns |
| 16→32→8→1 | 776 | 133 ns |
| 64→64→1 | 4,160 | 373 ns |

With AVX2+FMA tiled GEMV (4 neurons sharing input loads), the
hot path is load-port bound, not compute bound. Compile with
`RUSTFLAGS="-C target-cpu=native"` for AVX2+FMA dispatch.

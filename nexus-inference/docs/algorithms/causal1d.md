# Causal 1D Convolution

**Streaming causal convolution over a sliding window.** Maintains a
circular buffer of the last `kernel_size` inputs. Each step convolves
the buffer with learned filters, applies a configurable activation,
and projects to output. Only past and current inputs contribute — no
future leakage.

| Property | Value |
|----------|-------|
| Step cost | 50ns (8 filters) to 168ns (32 filters) with AVX2+FMA |
| Memory | `(F, K, C)` conv weights + output projection + circular buffer |
| Type | `Causal1dConv` |
| Construction | `from_parts(input_ch, kernel_size, filters, output_size, w_conv, b_conv, w_out, b_out, activation)` |
| Output | Single scalar or multi-output vector |

## What It Does

```
  step(input):
    1. buffer[write_idx] = input              — store in circular buffer
    2. for each filter f:                     — convolve
         sum = bias[f]
         for k in 0..kernel_size:
           sum += dot(w[f][k], buffer[(write_idx - k) % K])
         filter_out[f] = activation(sum)
    3. output = W_out @ filter_out + b_out    — output projection
    4. advance write_idx
```

The convolution has a **fixed receptive field** of exactly `kernel_size`
timesteps. Unlike LSTM/GRU, there is no recurrent state — the model sees
a sliding window, nothing more.

## Weight Layout

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `w_conv` | `(filters, kernel_size, input_ch)` | Convolution weights, flat |
| `b_conv` | `filters` | Per-filter bias |
| `w_out` | `(O, filters)` | Output projection, row-major |
| `b_out` | `O` | Output bias |

The convolution weight layout is `(filters, kernel_size, input_ch)` —
not PyTorch's `nn.Conv1d` layout which is `(out_channels, in_channels,
kernel_size)`. The dimension ordering is transposed to match the
circular buffer access pattern.

## Circular Buffer

The buffer stores the last `kernel_size` inputs as a ring. Each step
writes the new input at `write_idx`, then the convolution reads
backwards through the ring:

```
  k=0: current input (just written)
  k=1: previous input
  ...
  k=K-1: oldest input in the window
```

Buffer index: `(write_idx + kernel_size - k) % kernel_size`.

Internally, the buffer is linearized into a contiguous scratch array
before convolution, enabling tiled `dot4_f32` over the full
`kernel * input_ch` length instead of K separate short dot products.

## Priming

Before `kernel_size` steps have been processed, the buffer is partially
filled (zero-padded). `is_primed()` returns `true` once the full window
is populated.

```rust
let mut conv = Causal1dConv::from_parts(
    2, 3, 4, 1, /* weights... */ Activation::Relu,
).unwrap();

assert!(!conv.is_primed());    // kernel_size=3, need 3 steps
conv.predict(&[0.5, 1.0]);
conv.predict(&[0.2, 0.3]);
conv.predict(&[0.1, 0.4]);
assert!(conv.is_primed());     // buffer fully populated
```

Output before priming is valid but computed over zero-padded history.
Many applications discard or ignore pre-priming output.

## Activation Functions

Unlike LSTM/GRU (hardcoded sigmoid/tanh), the convolution layer has
a configurable activation function — same set as [MLP](mlp.md):
Relu, LeakyRelu, Tanh, Sigmoid, Identity, Elu, Gelu, Swish.

The convolution is a feature extractor, not a gating mechanism, so
the activation choice depends on the task. Relu is the most common
default.

## NaN Handling

NaN inputs are written to the circular buffer and propagate through
dot products and activation. The buffer retains NaN for `kernel_size`
steps (until the NaN entry rotates out). All activations propagate
NaN correctly.

## When to Use It

**Use Causal1dConv when:**
- Temporal patterns have a known, fixed horizon (e.g., "look at the last 8 ticks")
- You need cheaper inference than LSTM/GRU
- The task is local pattern detection: micro-bursts, short-term momentum,
  periodic signals with known period
- You don't need long-range memory across the full sequence

**Don't use Causal1dConv when:**
- The model needs to remember indefinitely (use [LSTM](lstm.md) or [GRU](gru.md))
- There's no temporal component (use [GBDT](gbdt.md) or [MLP](mlp.md))
- You need sub-10ns predictions (use [LUT](../algorithms/lut.md))

## Code Example

```rust
use nexus_inference::{Causal1dConv, Activation};

let mut conv = Causal1dConv::from_parts(
    4, 8, 16, 1,      // 4 input channels, kernel 8, 16 filters, 1 output
    &w_conv, &b_conv,
    &w_out, &b_out,
    Activation::Relu,
).unwrap();

// Process a stream
for frame in data_stream {
    let score = conv.predict(&frame);
    if conv.is_primed() {
        // Act on score — buffer is fully populated
    }
}

// Reset for new sequence
conv.reset();
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Construction | O(F*K*C + O*F) | Weights + buffer + scratch |
| `predict` | O(F*K*C + O*F) | No allocation |

| Configuration | FMAs | Latency (AVX2+FMA) |
|---------------|------|--------------------|
| 4ch x 4k x 8f → 1 | 128 + 8 | 50 ns |
| 4ch x 8k x 16f → 1 | 512 + 16 | 87 ns |
| 8ch x 8k x 32f → 1 | 2,048 + 32 | 168 ns |

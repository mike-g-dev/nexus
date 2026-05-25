# GRU — Gated Recurrent Unit

**Single-layer GRU for streaming temporal inference.** Three gates
(reset, update, candidate) with hidden state carried between calls.
~75% of LSTM compute for comparable quality on many tasks. Trained
externally in PyTorch, loaded via `from_parts`.

| Property | Value |
|----------|-------|
| Step cost | 165ns (H=16) to 1.1us (H=64) with AVX2+FMA |
| Memory | Separate `(3H, I)` and `(3H, H)` matrices + output projection + state |
| Type | `TinyGru` |
| Construction | `from_parts(input, hidden, output, weight_ih, weight_hh, bias_ih, bias_hh, w_out, b_out)` |
| Output | Single scalar or multi-output vector |

## What It Does

```
  step(input):
    1. ih = W_ih @ input                         — input-to-hidden
    2. hh = W_hh @ hidden                        — hidden-to-hidden
    3. r = sigmoid(ih[0..H] + b_ih[0..H]
                 + hh[0..H] + b_hh[0..H])       — reset gate
       z = sigmoid(ih[H..2H] + b_ih[H..2H]
                 + hh[H..2H] + b_hh[H..2H])     — update gate
       n = tanh(ih[2H..3H] + b_ih[2H..3H]
              + r * (hh[2H..3H] + b_hh[2H..3H]))— candidate
    4. hidden' = (1 - z) * n + z * hidden        — interpolate
    5. output = W_out @ hidden' + b_out           — output projection
```

The update gate `z` controls interpolation between the old hidden
state and the new candidate. `z=1` means "keep old state entirely"
(PyTorch convention). The reset gate `r` controls how much of the
previous hidden state influences the candidate.

## Weight Layout

Parameters map directly to PyTorch's `nn.GRU` + `nn.Linear`:

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `weight_ih` | `(3*H, I)` | Input-to-hidden, row-major. Gate order: R, Z, N |
| `weight_hh` | `(3*H, H)` | Hidden-to-hidden, row-major. Same gate order |
| `bias_ih` | `3*H` | Input gate biases |
| `bias_hh` | `3*H` | Hidden gate biases |
| `w_out` | `(O, H)` | Output projection, row-major |
| `b_out` | `O` | Output bias |

Unlike LSTM, GRU stores `weight_ih` and `weight_hh` separately
(no fusion). The candidate gate applies the reset gate between the
input-to-hidden and hidden-to-hidden products, so they can't be
combined into a single matrix.

## Why Two Matmuls

The candidate gate equation is:

```
n = tanh( W_in @ x + b_in + r * (W_hn @ h + b_hn) )
```

The reset gate `r` multiplies only the hidden-to-hidden part. If
we fused the matrices like LSTM, we'd compute `W_fused @ [x; h]`
which pre-combines both halves — there's no place to insert the
reset gate between them. Two separate matmuls are required.

## Gate Activations

Hardcoded: sigmoid for reset/update gates, tanh for candidate.
Same Pade [7,6] rational polynomial as LSTM. AVX2 vectorized
for 8 hidden units at a time.

## NaN Handling

Same as LSTM — NaN propagates through all gates and into the hidden
state. Call `reset()` to recover.

## When to Use It

**Use GRU when:**
- Same temporal modeling needs as LSTM, but you want ~25% less compute
- Your temporal patterns are relatively simple (GRU matches LSTM on many tasks)
- Hidden size is 32-64 where GRU's cache behavior is better than LSTM
- You're evaluating both — train LSTM and GRU, keep whichever works better

**Don't use GRU when:**
- Your task specifically benefits from the cell state mechanism (long-range memory)
- Hidden size is very small (H<=16) where LSTM's fused matmul is actually faster
- Temporal patterns have a fixed horizon (use [Causal1dConv](causal1d.md))
- No temporal context needed (use [GBDT](gbdt.md) or [MLP](mlp.md))

## Code Example

```rust
use nexus_inference::TinyGru;

let mut gru = TinyGru::from_parts(
    4, 16, 1,          // 4 inputs, 16 hidden, 1 output
    &weight_ih, &weight_hh,
    &bias_ih, &bias_hh,
    &w_out, &b_out,
).unwrap();

// Process a sequence
let score = gru.predict(&[0.5, 1.2, -0.3, 0.8]);
let score = gru.predict(&[0.3, 0.9, -0.1, 1.1]);  // carries state

// Reset for new sequence
gru.reset();
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Construction | O(3*H*(I+H) + O*H) | Separate ih/hh matrices + state + scratch |
| `predict` | O(3*H*I + 3*H*H + O*H) | No allocation |

| Configuration | FMAs | Latency (AVX2+FMA) |
|---------------|------|--------------------|
| 8→16→1 | 1,152 + 16 | 165 ns |
| 8→32→1 | 3,840 + 32 | 356 ns |
| 16→64→1 | 15,360 + 64 | 1,061 ns |

# LSTM — Long Short-Term Memory

**Single-layer LSTM for streaming temporal inference.** Four gates
(input, forget, cell candidate, output) with hidden and cell state
carried between steps. Trained externally in PyTorch, loaded via
`from_parts`, one timestep per `step` call.

| Property | Value |
|----------|-------|
| Step cost | 105ns (H=8) to 1.3us (H=64) with AVX2+FMA |
| Memory | Fused `(4H, I+H)` gate matrix + output projection + state |
| Type | `TinyLstm` |
| Construction | `from_parts(input, hidden, output, weight_ih, weight_hh, bias_ih, bias_hh, w_out, b_out)` |
| Output | Single scalar or multi-output vector |

## What It Does

```
  step(input):
    1. concat = [input; hidden_state]
    2. gates = W_fused @ concat + b_fused        — single matmul
    3. i = sigmoid(gates[0..H])                  — input gate
       f = sigmoid(gates[H..2H])                 — forget gate
       g = tanh(gates[2H..3H])                   — cell candidate
       o = sigmoid(gates[3H..4H])                — output gate
    4. cell' = f * cell + i * g                  — selective memory
    5. hidden' = o * tanh(cell')                 — gated output
    6. output = W_out @ hidden' + b_out          — output projection
```

The cell state acts as long-term memory. The forget gate controls
what to discard, the input gate controls what new information to
store. This selective memory mechanism is what distinguishes LSTM
from simpler recurrent networks.

## Weight Layout

Parameters map directly to PyTorch's `nn.LSTM` + `nn.Linear`:

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `weight_ih` | `(4*H, I)` | Input-to-hidden, row-major. Gate order: I, F, G, O |
| `weight_hh` | `(4*H, H)` | Hidden-to-hidden, row-major. Same gate order |
| `bias_ih` | `4*H` | Input gate biases |
| `bias_hh` | `4*H` | Hidden gate biases |
| `w_out` | `(O, H)` | Output projection, row-major |
| `b_out` | `O` | Output bias |

Internally, `from_parts` fuses `weight_ih` and `weight_hh` into a
single `(4H, I+H)` matrix and pre-sums biases. This enables a
single matrix-vector product per step instead of two.

## Gate Activations

Hardcoded: sigmoid for input/forget/output gates, tanh for cell
candidate and output nonlinearity. This is the standard LSTM
formulation — changing gate activations would break the gating
mechanism.

Sigmoid and tanh use a Pade [7,6] rational polynomial approximation
with full f32 precision (~1.2e-7 relative error). On AVX2 targets,
8 hidden units are processed simultaneously with vectorized gates.

## NaN Handling

NaN inputs propagate through the computation — sigmoid(NaN) = NaN,
tanh(NaN) = NaN, and arithmetic with NaN produces NaN. The hidden
and cell state will become NaN and remain so until `reset()`.
This matches the standard ML convention (caller responsibility).

## State Management

```rust
let mut lstm = TinyLstm::from_parts(/* ... */).unwrap();

// Process a sequence
let s1 = lstm.step(&frame_1);   // hidden state initialized to zero
let s2 = lstm.step(&frame_2);   // carries h and c from previous step
let s3 = lstm.step(&frame_3);   // accumulates temporal context

// Start a new sequence
lstm.reset();              // clears h and c to zero
let s1 = lstm.step(&new_frame);  // fresh start
```

## When to Use It

**Use LSTM when:**
- You need to model temporal patterns that unfold over many timesteps
- The model needs to selectively remember and forget (regime detection,
  accumulating flow toxicity, tracking session state)
- You have a PyTorch-trained LSTM with hidden size 8-64
- Prediction budget is 100ns-2us per step

**Don't use LSTM when:**
- Temporal patterns have a known fixed horizon (use [Causal1dConv](causal1d.md))
- You don't need the expressiveness of 4 gates (use [GRU](gru.md), ~25% cheaper)
- Hidden size > 64 (weights spill L1d cache, use GPU instead)
- Input is a single snapshot with no temporal context (use [GBDT](gbdt.md) or [MLP](mlp.md))

## LSTM vs GRU

| Property | LSTM | GRU |
|----------|------|-----|
| Gates | 4 (input, forget, cell candidate, output) | 3 (reset, update, candidate) |
| State | Hidden + cell (separate long-term memory) | Hidden only |
| Matmuls per step | 1 (fused) | 2 (separate, candidate needs reset between halves) |
| FMA count | ~33% more than GRU at same H | Baseline |
| Cache behavior at H=64 | Fused matrix spills L1d | Separate matrices fit L1d |
| Expressiveness | More — cell state decouples memory from output | Less — hidden state serves both roles |

At H<=32, LSTM's single fused matmul is faster despite more FMAs.
At H>=64, GRU's smaller matrices have better cache behavior. Try
both and keep whichever trains better for your task.

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Construction | O(4*H*(I+H) + O*H) | Fused weights + state + scratch |
| `step` | O(4*H*(I+H) + O*H) | No allocation |

| Configuration | FMAs | Latency (AVX2+FMA) |
|---------------|------|--------------------|
| 4→8→1 | 384 + 8 | 105 ns |
| 8→16→1 | 1,536 + 16 | 155 ns |
| 8→32→1 | 5,120 + 32 | 351 ns |
| 16→64→1 | 20,480 + 64 | 1,306 ns |

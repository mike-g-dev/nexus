# GBDT — Gradient-Boosted Decision Trees

**Ensemble of decision trees trained sequentially.** Each tree corrects
the residual errors of the previous ones. Prediction sums the leaf
values across all trees.

| Property | Value |
|----------|-------|
| Prediction cost | ~5 cycles/node (branchless cmov), ~8 cycles/node (NaN-aware) |
| Memory | 8 bytes/node + 4 bytes/tree offset |
| Type | `Gbdt` |
| Construction | `from_parts()` (raw nodes) or `from_lightgbm()` (text format) |
| Output | Single scalar (raw ensemble score, f32) |

## What It Does

```
  Input features: [price_change, spread, volume, ...]
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
           Tree 0      Tree 1      Tree 2     ...  Tree N
           ┌─┐         ┌─┐         ┌─┐
          ╱   ╲       ╱   ╲       ╱   ╲
         ╱     ╲     ╱     ╲     ╱     ╲
        ■       ■   ■       ■   ■       ■    (leaf values)
       0.3    -0.1 0.2     0.1 -0.05   0.15
              │           │           │
              └─────┬─────┘─────┬─────┘
                    ▼           ▼
          base_score + Σ(leaf values) = raw prediction
              0.0   +    0.3 + 0.2 + (-0.05) = 0.45
```

Each tree is a binary decision tree:
- Internal nodes compare `features[i] <= threshold`
- Leaf nodes store a numeric value
- NaN features route via a learned default direction (left or right)

The ensemble prediction is `base_score + sum(leaf_values)`.

## Tree Storage

Trees use a **false-branch-next** layout: nodes are ordered so the
right (false) child is always at `index + 1`. This means ~50% of
tree traversal follows sequential memory, which the hardware
prefetcher serves from L1.

```
  Logical tree:          Memory layout (DFS right-first):
       A                  [A] [C] [D] [E] [B] [F] [G]
      ╱ ╲                  0   1   2   3   4   5   6
     B   C
    ╱ ╲ ╱ ╲              A.left = 4 (B)    false branch: idx+1 = 1 (C)
   F  G D  E             C.left = 3 (E)    false branch: idx+1 = 2 (D)
                          B.left = 6 (G)    false branch: idx+1 = 5 (F)
```

Compact 8-byte `repr(C)` nodes:
- `value: f32` (threshold for splits, prediction for leaves)
- `feature_idx: u16` (bit 15 = leaf flag, bit 14 = default_left, bits 13:0 = feature index)
- `left: u16` (only stored for left/true branch)

The 8-byte power-of-2 stride means pointer arithmetic is a shift (LEA
with scale 8), not a multiply. Traversal uses `select_unpredictable` to
produce a single cmov per tree level — fully branchless, deterministic
latency regardless of input data.

## NaN Handling

| Method | NaN behavior | Cost |
|--------|-------------|------|
| `predict` | NaN routes right (`NaN <= threshold` is false) | ~5 cycles/node |
| `predict_nan_aware` | Routes NaN via learned default direction (3 cmovs) | ~8 cycles/node |

GBDT is unique among the three model types: it can *handle* NaN
rather than just propagating it. The training framework (LightGBM)
learns which direction gives better predictions when a feature is
missing. Use `predict_nan_aware` when features may contain NaN.

## When to Use It

**Use GBDT when:**
- Features are tabular (numeric columns, not raw signals)
- You have a trained LightGBM model to deploy
- Prediction budget is 200ns-3us (50-200 trees, depth 6-8)
- You need NaN-tolerant inference (missing features are common)

**Don't use GBDT when:**
- Inputs are dense vectors from an upstream neural network (use [MLP](mlp.md))
- The relationship is a known function that can be tabulated (use [LUT](lut.md))
- You need sub-10ns predictions (use [LUT](lut.md))

## Output Interpretation

`predict()` returns the **raw ensemble score** — the sum of leaf
values plus base score. This is NOT a probability or class label.

For classification models:
- Binary classification (`objective=binary`): apply sigmoid to get probability
- Multiclass: not supported (single-output only)
- Poisson regression: apply `exp()` to get the rate

## Code Example

```rust
use nexus_inference::Gbdt;

// Load from LightGBM text format
let model = Gbdt::from_lightgbm(model_bytes).unwrap();

let features = vec![0.5_f32, 1.2, -0.3, 0.8];
let score = model.predict(&features);

// NaN-aware routing (when features may contain NaN)
let score = model.predict_nan_aware(&features);

// Buffer form
let mut output = [0.0_f32];
model.predict_into(&features, &mut output);
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Construction (`from_lightgbm`) | O(total_nodes) | O(total_nodes) |
| `predict` | O(trees x depth) | O(1) |
| `predict` | O(trees x depth) | O(1) |
| `predict_n` | O(n x depth) | O(1) |

Typical configurations:

| Trees | Depth | Nodes/tree | Total nodes | Working set | Approx. latency |
|-------|-------|-----------|-------------|-------------|----------------|
| 50 | 6 | 63 | 3,150 | 25 KB | ~280 ns |
| 100 | 6 | 63 | 6,300 | 50 KB | ~560 ns |
| 200 | 8 | 255 | 51,000 | 400 KB | ~2.5 us |

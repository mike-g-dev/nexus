# Choosing a Model Type

## Decision Tree

```
  What are you deploying?
  │
  ├── A trained LightGBM model
  │   └── GBDT (use from_lightgbm)
  │
  ├── A trained neural network (PyTorch)
  │   ├── Feedforward (nn.Linear layers)
  │   │   └── MLP (export weights to from_parts)
  │   ├── LSTM (nn.LSTM)
  │   │   └── TinyLstm (streaming step-by-step)
  │   ├── GRU (nn.GRU)
  │   │   └── TinyGru (streaming step-by-step)
  │   └── 1D convolution (nn.Conv1d, causal)
  │       └── Causal1dConv (streaming step-by-step)
  │
  ├── A pre-computed function over a small grid
  │   └── LUT (tabulate in Python, load flat array)
  │
  └── Not sure yet — what matters most?
      │
      ├── Latency under 10ns
      │   └── LUT (O(1), ~5ns for 2 features)
      │
      ├── Tabular features with missing values
      │   └── GBDT (learned NaN routing)
      │
      ├── Dense numeric inputs, nonlinear relationships
      │   └── MLP (universal function approximation)
      │
      ├── Temporal patterns, long-range memory
      │   └── LSTM or GRU (hidden state accumulates over time)
      │
      ├── Temporal patterns, fixed window
      │   └── Causal1dConv (sees exactly last K timesteps)
      │
      └── Simple monotonic relationship, 1-2 features
          └── LUT (precompute, avoid model complexity)
```

## Comparison

### Stateless types

| Criterion | GBDT | MLP | LUT |
|-----------|------|-----|-----|
| **Input type** | Tabular features | Dense vectors | 1-3 numeric features |
| **Latency** | 200ns - 3us | 100ns - 2us | 5-10ns |
| **Missing data** | Learned NaN routing | No (propagate) | No (clamp to bin 0) |
| **Output** | Single scalar | Single or multi-output | Single scalar |
| **Model source** | LightGBM | PyTorch/TF/sklearn | Python script |
| **Memory** | 16B/node | 4B/weight | 4B/bin^features |
| **Loader** | `from_lightgbm()` | `from_parts()` | `from_parts()` |

### Temporal types

| Criterion | LSTM | GRU | Causal1dConv |
|-----------|------|-----|--------------|
| **Input type** | Per-timestep vector | Per-timestep vector | Per-timestep vector |
| **Latency** | 105ns - 1.3us | 165ns - 1.1us | 50ns - 168ns |
| **Memory model** | Hidden + cell state | Hidden state | Circular buffer (fixed window) |
| **Temporal range** | Unbounded (learned) | Unbounded (learned) | Fixed (kernel_size) |
| **Gates** | 4 (sigmoid/tanh) | 3 (sigmoid/tanh) | None (configurable activation) |
| **Output** | Single or multi | Single or multi | Single or multi |
| **Model source** | PyTorch `nn.LSTM` | PyTorch `nn.GRU` | PyTorch `nn.Conv1d` |
| **API** | `predict` / `predict_into` | `predict` / `predict_into` | `predict` / `predict_into` |

## When to Combine Types

In trading systems, it's common to use multiple model types together:

- **GBDT for feature selection** → extract top features, feed into MLP
  for final prediction
- **LUT for fast pre-filters** → coarse signal check in <10ns, then
  GBDT/MLP for the full model only when the filter fires
- **MLP for embeddings** → neural network produces a dense vector,
  GBDT consumes it as features alongside tabular data
- **GBDT signals → MLP combination** → individual microstructure signals
  from trees or streaming stats, stacked into a dense vector, combined
  by a small MLP that learns nonlinear interactions
- **LSTM/GRU for regime detection** → temporal model outputs a regime
  score, GBDT or MLP conditions on it alongside snapshot features

The `predict_into` API makes composition straightforward
— one model's output buffer feeds directly into the next model's input.

## Model Size Guidelines

| Type | Small | Medium | Large |
|------|-------|--------|-------|
| GBDT | 50 trees x depth 6 (~220ns) | 100 x 6 (~410ns) | 200 x 8 (~2.2us) |
| MLP | 8→16→1 (~100ns) | 16→32→8→1 (~370ns) | 64→64→1 (~2us) |
| LUT | 1 feat x 10 bins | 2 feat x 10 bins | 3 feat x 20 bins |
| LSTM | 4→8→1 (105ns) | 8→16→1 (155ns) | 16→64→1 (1.3us) |
| GRU | 8→16→1 (165ns) | 8→32→1 (356ns) | 16→64→1 (1.1us) |
| Conv | 4ch×4k×8f (50ns) | 4ch×8k×16f (87ns) | 8ch×8k×32f (168ns) |

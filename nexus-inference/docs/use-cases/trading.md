# Trading Systems

## The Inference Pipeline

In a trading system, inference sits between feature computation and
order execution. The prediction must be fast (microseconds matter)
and correct (wrong predictions cost money).

```
  Market data  →  Feature pipeline  →  Inference  →  Decision  →  Execution
   (ticks)        (nexus-stats)       (this crate)   (strategy)   (orders)
```

The feature pipeline (often using `nexus-stats` types like EMA, KAMA,
Drawdown, etc.) produces a numeric feature vector. The inference
engine scores it. The strategy layer decides whether and how to act.

## Common Model Deployments

### Signal scoring (GBDT)

The most common pattern: LightGBM model trained on tabular features,
deployed for real-time scoring.

```rust
use nexus_inference::Gbdt;

// Load once at startup
let model = Gbdt::from_lightgbm(&model_bytes).unwrap();

// On each market data update:
let features = [
    spread_bps,
    mid_price_return_1s,
    volume_imbalance,
    ema_deviation,
    drawdown_pct,
    volatility_ratio,
    order_flow_toxicity,
    queue_position_ratio,
];

// NaN-aware — missing features route via learned direction
let signal = model.predict_nan_aware(&features);
```

**Why GBDT here:** Tabular features from different data sources with
different update frequencies. Some features may be stale or missing
(NaN). GBDTs handle this natively.

### Nonlinear combination (MLP)

When the relationship between features is nonlinear and can't be
captured by a single tree ensemble:

```rust
use nexus_inference::{Mlp, Activation};

let model = Mlp::from_parts(
    &[8, 16, 1], &weights, &biases, Activation::Relu,
).unwrap();

// Features must be clean — MLP rejects NaN
let features = impute_and_normalize(&raw_features);
let signal = model.predict(&features);
```

**Why MLP here:** The signal depends on interactions between features
that tree splits can't capture efficiently. Common for
embedding-based features or when the input is already a dense
representation from another model.

### Fast approximation (LUT)

For functions that are too expensive to compute per-tick but can be
pre-tabulated:

```rust
use nexus_inference::Lut;

// Pre-computed: spread_model(volatility, time_of_day) → fair_spread
let spread_lut = Lut::from_parts(
    2, 50,
    &[0.0, 0.0],     // min vol, min time_frac
    &[0.05, 1.0],    // max vol, max time_frac
    &table,           // 2500 pre-computed spread values
).unwrap();

// ~5ns lookup instead of ~500ns model evaluation
let fair_spread = spread_lut.predict(&[current_vol, time_frac]);
```

**Why LUT here:** The fair spread model is expensive (involves
multiple GBDT evaluations, historical lookups, etc.) but the input
space is small (2 features). Pre-compute once per parameter update,
serve at ~5ns per tick.

## Model Composition

Models can be chained — one model's output feeds another:

```rust
// Stage 1: GBDT feature extraction (NaN-tolerant)
let gbdt_score = gbdt_model.predict_nan_aware(&raw_features);

// Stage 2: MLP combines GBDT score with embedding features
let mlp_features = [
    gbdt_score,
    embedding[0],
    embedding[1],
    embedding[2],
];
let final_signal = mlp_model.predict(&mlp_features);
```

## Output Interpretation

All model types return **raw scores**, not probabilities or actions.
The strategy layer decides what to do:

```rust
let score = model.predict(&features);

// Strategy layer owns the decision
if score > entry_threshold {
    place_order(Side::Buy, size_from_signal(score));
} else if score < -entry_threshold {
    place_order(Side::Sell, size_from_signal(-score));
}
```

For classification models, apply the link function yourself:

```rust
let logit = model.predict(&features);
let probability = 1.0 / (1.0 + (-logit).exp());  // sigmoid
```

## Performance Budget

Typical latency budgets for trading inference:

| Component | Budget | What fits |
|-----------|--------|----------|
| Feature computation | 1-5 us | nexus-stats types, 10-20 features |
| Model inference | 100ns - 2us | GBDT 50-100 trees, MLP 8→16→1 |
| Decision logic | 50-200 ns | Threshold checks, position sizing |
| Order construction | 100-500 ns | Message building, risk checks |
| **Total** | **2-8 us** | **End-to-end tick-to-order** |

Inference is typically 5-25% of the total pipeline. Optimize the
feature pipeline first — that's where the most time goes.

## Model Updates

To update a live model, swap via `Arc`:

```rust
use std::sync::Arc;

// GBDT: prediction takes &self, so Arc sharing is natural
let model = Arc::new(Gbdt::from_lightgbm(&bytes).unwrap());
let m = model.clone();
let score = m.predict(&features);

// Cold path: load new model, swap the Arc
let new_model = Arc::new(Gbdt::from_lightgbm(&new_bytes).unwrap());
```

**MLP/LUT note:** MLP prediction takes `&self` (scratch buffers
use interior mutability). Models can be shared via `Arc<Mlp>`
directly — no mutex needed, no contention.

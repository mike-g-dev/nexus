# LUT — Lookup Table Predictor

**Discretized function approximation.** Divides each feature into
uniform bins and indexes a pre-computed flat table. O(1) prediction
regardless of the underlying function's complexity.

| Property | Value |
|----------|-------|
| Prediction cost | ~5-8 ns (1-3 features) |
| Memory | `n_bins^n_features * 4` bytes (f32 table) |
| Types | `Lut` |
| Construction | `from_parts(n_features, n_bins, mins, maxs, table)` |
| Output | Single scalar |

## What It Does

```
  1D example: price_change → predicted_spread
  (4 bins over [0.0, 1.0), step = 0.25)

  table = [10.0, 20.0, 30.0, 40.0]
           bin0   bin1   bin2   bin3

  predict(0.1) → bin 0 → 10.0
  predict(0.3) → bin 1 → 20.0
  predict(0.6) → bin 2 → 30.0
  predict(0.8) → bin 3 → 40.0

  2D example: two features, 3 bins each → 9 table entries

  feature 1 (bins →)
           bin0  bin1  bin2
  bin0 ┌─────┬─────┬─────┐
       │  0  │  1  │  2  │   feature 0
  bin1 ├─────┼─────┼─────┤   (bins ↓)
       │  3  │  4  │  5  │
  bin2 ├─────┼─────┼─────┤
       │  6  │  7  │  8  │
       └─────┴─────┴─────┘

  Flat index via Horner's method:
    idx = bin0 * n_bins + bin1
  
  predict([2.5, 0.5]) → bin0=2, bin1=0 → idx = 2*3 + 0 = 6
```

The bin index for each feature is computed as:
```
bin = floor((value - min) / step)
```
where `step = (max - min) / n_bins`. Out-of-range values clamp to
the first or last bin.

## Table Size

The table grows exponentially with features:

| Features | Bins | Table entries | Memory (f32) |
|----------|------|--------------|-------------|
| 1 | 10 | 10 | 40 B |
| 2 | 10 | 100 | 400 B |
| 3 | 10 | 1,000 | 4 KB |
| 2 | 50 | 2,500 | 10 KB |
| 3 | 20 | 8,000 | 32 KB |
| 4 | 10 | 10,000 | 40 KB |
| 3 | 50 | 125,000 | 500 KB |

In practice, LUTs work best with 1-3 features. Beyond that, the
table size explodes and you're better off with an MLP.

## NaN Handling

NaN features map to bin 0 (Rust's saturating float-to-int cast
maps `NaN as usize` to 0). The result is a valid number from the
table but meaningless. Validate inputs in the feature pipeline.

## When to Use It

**Use LUT when:**
- The function has few inputs (1-3 features)
- The relationship can be precomputed over a grid
- You need absolute minimum latency (<10 ns)
- Accuracy at bin resolution is acceptable (piecewise constant)

**Don't use LUT when:**
- More than 3-4 features (table size explodes)
- You need smooth interpolation between grid points (LUT is piecewise constant)
- The feature ranges change over time (fixed min/max at construction)
- Features have non-uniform importance (uniform bins waste resolution)

**Common use cases:**
- Volatility surface lookups (strike x expiry → implied vol)
- Fee schedules (volume tier → fee rate)
- Pre-computed signal surfaces (feature1 x feature2 → alpha)
- Fast approximations of expensive functions (replacing `exp`, `log`, etc.)

## Code Example

```rust
use nexus_inference::Lut;

// 2 features, 10 bins each, ranges [0, 1)
let model = Lut::from_parts(
    2,              // n_features
    10,             // n_bins per feature
    &[0.0, 0.0],   // mins
    &[1.0, 1.0],   // maxs
    &table,         // 100 pre-computed values
).unwrap();

let value = model.predict(&[0.35, 0.72]);
```

## Building the Table

The table is typically computed in Python:

```python
import numpy as np

n_features, n_bins = 2, 10
mins = np.array([0.0, 0.0])
maxs = np.array([1.0, 1.0])

# Create grid points
grids = [np.linspace(mins[i], maxs[i], n_bins, endpoint=False) 
         + (maxs[i] - mins[i]) / n_bins / 2  # bin centers
         for i in range(n_features)]

# Evaluate your function on the grid
table = np.zeros(n_bins ** n_features)
for idx, point in enumerate(itertools.product(*grids)):
    table[idx] = your_function(*point)

# Flatten in row-major order (first feature varies slowest)
# This is the natural order from itertools.product
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Construction | O(n_bins^n_features) | O(n_bins^n_features) |
| `predict` | O(n_features) | O(1) |

Prediction cost is constant regardless of table size — the table
access is a single indexed read after computing the flat index.
The O(n_features) cost is for the per-feature division and Horner's
method accumulation.

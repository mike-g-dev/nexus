# Performance

All benchmarks: `RUSTFLAGS="-C target-cpu=native"` (AVX2+FMA),
pinned with `taskset -c 0`, criterion. For reproducible results,
disable turbo boost (`echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo`).

## How it works

Every model type follows the same pattern: pre-allocate all memory
at construction, then touch only stack and pre-allocated buffers
during inference. No allocations, no syscalls, no locks on the hot
path.

The core bottleneck across all model types is matrix-vector
multiplication. The dot product implementation uses tiled
multi-accumulator loops (4 or 8 rows sharing input loads) to
saturate FMA throughput and minimize memory bandwidth. Compile-time
SIMD dispatch selects AVX-512, AVX2+FMA, or scalar fallback — no
runtime feature detection overhead.

**Stateless types** (GBDT, MLP, LUT) compute output directly from
input features. **Temporal types** (LSTM, GRU, Conv) carry hidden
state between `step` calls — the state buffers are pre-allocated
at construction and mutated in place.

## GBDT

16-byte nodes in false-branch-next (depth-first) layout. The right
child is always at `idx + 1`, so ~50% of tree traversal is
sequential memory access served by the hardware prefetcher.

| Configuration | `predict` |
|--------------|----------:|
| 50 trees x depth 6, 8 features | 264 ns |
| 100 trees x depth 6, 8 features | 550 ns |
| 200 trees x depth 8, 16 features | 2.47 us |

Per-node cost: ~4.7 cycles — within ~1 cycle of the L1 load latency
floor for data-dependent tree traversal.

## MLP

SIMD-tiled matrix-vector product with fused bias + activation in
registers (f32 path). 4 or 8 output neurons computed simultaneously,
sharing input vector loads. Relu and Identity activations are fused
in SIMD; other activations use scalar fallback.

### MLP f32

| Configuration | Latency |
|--------------|--------:|
| 8→16→1 relu | 53 ns |
| 16→32→8→1 relu | 106 ns |
| 64→64→1 relu | 187 ns |
| 32→32→32→32→1 relu | 229 ns |
| 64→64→64→1 relu | 409 ns |

### MLP f64

No SIMD-tiled path — uses generic dot4 + scalar activation.

| Configuration | Latency | f32 speedup |
|--------------|--------:|------------:|
| 8→16→1 relu | 65 ns | 1.2x |
| 16→32→8→1 relu | 170 ns | 1.6x |
| 64→64→1 relu | 455 ns | 2.1x |

The f32 advantage grows with layer width because the tiled SIMD path
fuses bias + relu in registers and the 8-wide dot product halves
iteration count.

### LayerNorm

BatchNorm layers are fused into the preceding linear layer's weights
at load time — zero runtime cost.

LayerNorm cannot be fused (statistics depend on each input). It uses
a SIMD-vectorized 3-pass implementation: mean, variance, then
normalize + affine transform + activation. Overhead is 35-53% vs the
same model without LayerNorm, depending on hidden layer width.

## LUT

O(1) prediction via discretized feature lookup. One division and one
array index per feature.

| Configuration | `predict` |
|--------------|----------:|
| 2 features x 10 bins | 6.6 ns |
| 3 features x 20 bins | 8.5 ns |

## LSTM

Fuses `weight_ih` and `weight_hh` into a single `(4H, I+H)` gate
matrix per layer at construction — one matrix-vector multiply per
step instead of two. Gate activations (sigmoid, tanh) use a Pade
[7,6] rational polynomial approximation vectorized 8-wide (AVX2) or
16-wide (AVX-512), replacing scalar glibc transcendentals.

### Single-layer (TinyLstmF32)

| Configuration | Gate matrix | Latency |
|--------------|-------------|--------:|
| 4→8→1 | 32×12 | 105 ns |
| 8→16→1 | 64×24 | 137 ns |
| 8→32→1 | 128×40 | 297 ns |
| 16→64→1 | 256×80 | 1066 ns |

### Stacked (StackedLstmF32)

Each layer's hidden state feeds as input to the next. Output
projection applied only to the final layer.

| Configuration | Latency |
|--------------|--------:|
| 8→32→1 x 2 layers | 739 ns |
| 8→32→1 x 3 layers | 1239 ns |

Scaling is roughly linear with layer count. Non-first layers have
larger input dimensions (`hidden + hidden` vs `input + hidden`), so
per-layer cost is slightly higher than the single-layer baseline.

## GRU

Three gates instead of four, no separate cell state — ~75% of LSTM
compute per layer. Weights are stored separately (not fused) because
the candidate gate applies the reset gate between the input-hidden
and hidden-hidden matrix-vector products. Same Pade sigmoid/tanh
approximation as LSTM.

### Single-layer (TinyGruF32)

| Configuration | Latency |
|--------------|--------:|
| 8→16→1 | 173 ns |
| 8→32→1 | 325 ns |
| 16→64→1 | 909 ns |

### Stacked (StackedGruF32)

| Configuration | Latency |
|--------------|--------:|
| 8→32→1 x 2 layers | 711 ns |
| 8→32→1 x 3 layers | 1201 ns |

## Causal 1D Conv

Circular buffer linearized into contiguous scratch before convolution,
enabling tiled dot products over the full `kernel x channels` length.
SIMD-tiled with fused bias + activation (same pattern as MLP).

| Configuration | conv_len | Latency |
|--------------|----------|--------:|
| 4ch x 4k x 8f → 1 | 16 | 48 ns |
| 4ch x 8k x 16f → 1 | 32 | 68 ns |
| 8ch x 8k x 32f → 1 | 64 | 115 ns |

## Complexity

| Type | Predict | Construction |
|------|---------|-------------|
| GBDT | O(trees x depth) | O(total_nodes) |
| MLP | O(Σ layer[i] x layer[i+1]) | O(total_weights) |
| LUT | O(n_features) | O(n_bins^n_features) |
| LSTM | O(H x (I+H)) per layer | O(total_weights) |
| GRU | O(H x I + H x H) per layer | O(total_weights) |
| Conv | O(filters x kernel x channels) | O(total_weights) |

## Memory

All weights stored as `Box<[f32]>` or `Box<[f64]>`. Scratch buffers
pre-allocated at construction.

| Type | Weight memory | Example |
|------|--------------|---------|
| GBDT | 16B/node + 4B/tree | 100 trees x 63 nodes = 101 KB |
| MLP | 4B/weight (f32) or 8B (f64) | 8→16→1: 164 B (f32) |
| LUT | 8B/entry + 16B/feature | 2 feat x 10 bins: 835 B |
| LSTM | 4B x (4H(I+H) + 4H + 6H) per layer | 8→32→1 x 2L: ~42 KB |
| GRU | 4B x (3HI + 3HH + 3H + 3H + 7H) per layer | 8→32→1 x 2L: ~30 KB |
| Conv | 4B x (F x K x C + F + O x F + O) | 8ch x 8k x 32f→1: ~8 KB |

# nexus-inference Performance Optimization Catalog

Systematic record of SIMD and algorithmic optimizations applied to each
model type. Intended as an audit reference so future work doesn't
rediscover dead ends or miss context on why something was done.

All SIMD work targets AVX2+FMA with AVX-512F where available. Scalar
fallbacks exist for all code paths. Benchmarks run pinned (`taskset -c 0`)
with turbo boost disabled where noted.

---

## Shared: dot product primitives (`src/dot/`)

The foundation everything else builds on. All model types bottleneck on
matrix-vector products, so dot product throughput is the single largest
lever.

### dot_f32 / dot_f64 — single dot product

- **4 independent accumulators** to hide FMA latency (4-5 cycle on most
  x86). Inner loop processes 32 f32s (4×8-wide FMA) or 16 f64s (4×4-wide
  FMA) per iteration.
- Unrolled main loop + 8-element cleanup loop + scalar tail.
- AVX2 and AVX-512 implementations with compile-time dispatch via `cfg`.

### dot4_f32 / dot4_f64 — 4 simultaneous dot products

- **Shared input loads**: one `_mm256_loadu` per input vector feeds all 4
  row accumulators. Cuts input bandwidth by 4×.
- **2 accumulators per row (A/B split, 8 total)** to hide FMA latency.
  Inner loop processes 16 f32s or 8 f64s per iteration.
- Scalar tail for remainder.

### dot4_f32_m128 — 4 dots returning packed `__m128`

- Same accumulation as `dot4_f32` but returns `__m128` instead of
  `[f32; 4]`.
- **Paired hadd reduction**: `lo0/lo1 → hadd → h01`, `lo2/lo3 → hadd →
  h23`, then `hadd(h01, h23)` produces all 4 sums in the target lanes.
  11 reduction instructions vs 28 for 4 separate `hsum_f32` calls.
- Enables callers (matvec, MLP tiled, conv tiled) to fuse
  bias-add + activation + store in SIMD without scalar round-trip.
- LLVM never inlines this (~225 asm instructions). `#[inline]` hint is
  present but the function is too large. This is fine — the function call
  overhead is amortized over the inner loop work.

### dot8_f32_m256 — 8 simultaneous dot products (newest)

- **8 independent accumulators** (1 per row), single input load per
  iteration. 8 FMA chains hide latency without A/B splitting.
- Returns `__m256` for direct store or fused operations.
- **Reduction**: 8 cross-lane folds (`extractf128 + add`), 3 levels of
  `hadd`, final `insertf128` to pack `__m256`.
- **AVX-512 variant**: 16 `__m512` accumulators (A/B split per row),
  processes 32 elements per inner iteration. 3-stage reduction:
  `__m512 → __m256 → __m128 → hadd → insertf128`.
- **Threshold gating**: only called when `in_size >= 32`. Below that, the
  heavier reduction cost isn't amortized and `dot4_f32_m128` is faster.
  Crossover verified empirically: models with `in_size < 32` show no
  improvement from dot8, models with `in_size >= 40` show 5-19%.

### matvec_bias_f32 / matvec_f32 — tiled matrix-vector product

- Outer loop: dot8 (8 rows at a time) when `in_size >= 32`, then dot4
  (4 rows), then scalar tail.
- `#[inline]` — inlined into LSTM/GRU gate computation for zero call
  overhead.
- Used by: LSTM, GRU, stacked LSTM/GRU, Conv output projection.

---

## GBDT (`src/gbdt.rs`)

### False-branch-next node layout

- Compact 16-byte `Node` struct (`repr(C)`): feature_idx (u16), left (u16),
  flags (u16), value (f64). The `right` child field is **absent**.
- DFS right-first tree reordering: the false/right child is always at
  `idx + 1`. Eliminates a stored index per node and makes ~50% of
  decisions (the false path) sequential — served from L1 by the hardware
  prefetcher.
- `reorder_and_compact()` converts from `RawNode` (explicit left/right)
  to this layout during model construction.

### 12-byte packed layout (rejected)

- Benchmarked `repr(C, packed)` at 12 bytes per node (no padding after
  `flags`). The 25% smaller working set doesn't shift the L2-vs-L3 cache
  tier for any tested configuration, and the non-power-of-2 stride (×12 vs
  ×16) plus unaligned access overhead **regressed L2-resident cases by
  ~25%**. 16-byte aligned is the measured optimum.

### predict_n — partial ensemble

- `predict_n(features, n_trees)` sums only the first `n` trees. Enables
  early-stopping at inference time and A/B testing of ensemble depth.

---

## LUT (`src/lut.rs`)

O(1) prediction via discretized feature lookup. No SIMD optimization
needed — the operation is a division + array index per feature, already
<10ns. Perf-sensitive work here is table construction, not prediction.

---

## MLP (`src/mlp.rs`)

### SIMD tiled path (f32 only)

- `mlp_tiled_simd_f32`: `#[inline(never)]` free function processing the
  `out_size_4` portion of each layer.
- **Fused bias + activation + store** in SIMD registers. No scalar
  round-trip between dot product and activation.
- Relu path: `_mm_max_ps(bias + dots, zero)` (or `_mm256_max_ps` for dot8).
- Identity/last-layer path: `bias + dots` directly.
- **dot8→dot4 cascade** with `in_size >= 32` threshold: groups of 8 use
  `dot8_f32_m256`, remainder of 4 uses `dot4_f32_m128`.
- `mlp_tiled_noop<T>`: generic no-op for `MlpF64` (returns 0, compiler
  eliminates). Keeps the macro signature uniform without dead-code in the
  f64 path.

### 3-branch borrow checker pattern

- `predict_into` dispatches to `$tiled_fn` with one of three disjoint
  src/dst pairs: `(scratch_a → scratch_b)`, `(scratch_b → scratch_a)`, or
  `(scratch → output)`. Each branch is separate so Rust proves disjoint
  borrows. One branch per layer, not per element.

### cfg-gated `let mut j` pattern

- `#[cfg(SIMD)] let mut j = { tiled_fn(...) };` and
  `#[cfg(not(SIMD))] let mut j = 0usize;` avoids `unused_assignments`
  warnings from the scalar fallback overwriting a previously-assigned `j`.

### Measured results

See Results section for absolute latencies and f32-vs-f64 comparison.
Deeper configs benefit more — the tiled path runs per layer.

### f64 — no SIMD tiled path

`MlpF64` uses the generic `dot4_f64` + scalar activation fallback. The
tiled approach would work but f64 MLP is not a hot-path use case. If
needed, add `mlp_tiled_simd_f64` following the f32 pattern.

---

## LSTM (`src/rnn/lstm.rs`, `src/rnn/avx2_gates.rs`, `src/rnn/avx512_gates.rs`)

### Architecture

Two hot operations per step:
1. **Gate matvec**: single fused `matvec_bias_f32` over concatenated
   `[input, hidden]` → 4H-dimensional gate vector. Matrix shape is
   `(4×hidden, input_size + hidden_size)`. The large `in_size`
   (input+hidden combined) is what makes dot8 effective here.
2. **Gate activation + cell/hidden update**: sigmoid(i,f,o), tanh(g),
   cell update, tanh(cell) → hidden update.

### SIMD gate processing (AVX2)

- `lstm_gates_avx2`: processes 8 hidden units at a time.
- **Padé [7,6] rational approximation** for tanh — 7th degree numerator,
  6th degree denominator. Evaluated with FMA chains (3 FMA per
  num/den). Accuracy ~1e-5 max error over [-4.97, 4.97].
- **NaN preservation**: `_mm256_cmp_ps(x, x, _CMP_UNORD_Q)` detects NaN
  lanes before clamping, then `_mm256_blendv_ps` restores them. Without
  this, `min/max` clamping silently converts NaN to clip values.
- **Sigmoid via tanh**: `0.5 + 0.5 * tanh(x * 0.5)`. One function, not
  two approximations.
- Cell update: `c_new = fg * c_old + ig * cg` — single FMA.
- Hidden update: `h_new = og * tanh(c_new)` — reuses tanh_8wide.
- Scalar tail for `hidden % 8 != 0`.

### SIMD gate processing (AVX-512)

- `lstm_gates_avx512`: same algorithm, 16 lanes at a time.
- `tanh_16wide` / `sigmoid_16wide` using `__m512` intrinsics.
- NaN detection via `_mm512_cmp_ps_mask` + `_mm512_mask_blend_ps`
  (k-mask variant).

### matvec improvements (from dot8)

The gate matvec calls `matvec_bias_f32` which now uses the
dot8→dot4 cascade. See Results section for measured deltas.

### What wasn't done (and why)

- **Fused gate matvec**: computing the full gate matrix as a single fused
  operation (interleaving matvec rows with activation) would avoid writing
  the intermediate gate buffer. Not done because (a) the gate buffer is
  hot in L1 immediately after matvec, so the reload is ~4 cycles, and
  (b) fusing would prevent code reuse with the shared `matvec_bias_f32`.
- **Quantized weights (int8)**: would halve memory bandwidth but requires
  scale factors, dequantization overhead, and complicates the loader.
  Worth revisiting if models grow beyond L2.
- **Blocked/tiled matvec for cache**: the gate matrices are small enough
  (max ~256×80 = 80KB at f32) to fit in L2. Cache-blocking would add
  complexity without benefit at these sizes.

---

## GRU (`src/rnn/gru.rs`, `src/rnn/avx2_gates.rs`, `src/rnn/avx512_gates.rs`)

### Architecture

Three hot operations per step:
1. **input-hidden matvec**: `matvec_f32(w_ih, input)` → 3H gate vector.
   Matrix shape: `(3×hidden, input_size)`. No bias (applied in gate step).
2. **hidden-hidden matvec**: `matvec_f32(w_hh, hidden)` → 3H gate vector.
   Matrix shape: `(3×hidden, hidden_size)`.
3. **Gate activation + hidden update**: computes reset, update, candidate
   gates and blends old/new hidden state.

GRU splits the matvec into two calls (input-hidden and hidden-hidden)
because the candidate gate applies the reset gate between them:
`n = tanh(ih_cand + r * hh_cand)`. This is inherent to the GRU
architecture and can't be fused into a single matvec.

**Why GRU improves less than LSTM from dot8**: LSTM concatenates input
and hidden into one large vector (`in_size = input_size + hidden_size`)
for a single matvec. GRU keeps them separate — two smaller matvecs.
For GRU 16→64: the ih matvec has `in_size=16` (below dot8 threshold),
only the hh matvec (`in_size=64`) benefits. LSTM 16→64 gets
`in_size=80` on its single matvec — both halves benefit.

### SIMD gate processing

- `gru_gates_avx2` / `gru_gates_avx512`: 8-wide / 16-wide processing.
- Same Padé tanh/sigmoid as LSTM (shared functions).
- Reset gate: `r = sigmoid(ih + bias_ih + hh + bias_hh)` — 4 loads + 3
  adds + sigmoid.
- Update gate: same structure.
- Candidate: `n = tanh(ih_cand + bias_ih + r * (hh_cand + bias_hh))` —
  FMA for reset-gated term.
- Hidden blend: `h' = (1-z)*n + z*h` — sub + FMA.

### matvec improvements (from dot8)

GRU uses `matvec_f32` (no-bias variant). See Results section for
measured deltas.

### What wasn't done (and why)

- **Fused matvec** (same reasoning as LSTM — buffer fits in L1).
- **Single matvec for all gates**: GRU's reset-gate-before-candidate
  structure prevents this. The candidate's hidden-hidden contribution
  depends on `r`, which depends on the first matvec.

---

## Causal 1D Convolution (`src/conv/causal1d.rs`)

### Architecture

Two phases per step:
1. **Convolution**: `n_filters` dot products over the linearized circular
   buffer (length = `kernel_size × input_channels`).
2. **Output projection**: `matvec_bias_f32(w_out, filter_scratch)`.

### Circular buffer linearization

- Maintains a circular write buffer of the last `kernel_size` inputs.
- Each step linearizes into `lin_buf` before convolution. The memcpy
  cost is small (typically 16-128 f32s) and enables contiguous dot
  products without modular indexing in the inner loop.

### SIMD tiled convolution

- `conv_tiled_simd`: `#[inline(never)]` free function.
- **dot8→dot4 cascade** with `conv_len >= 32` threshold.
- **Fused bias + activation + store**: Relu path uses
  `_mm256_max_ps(bias + dots, zero)` / `_mm_max_ps` variant. Identity
  path skips the max.
- Handles Relu and Identity activations in SIMD. Other activations fall
  through to scalar.

### Measured results

See Results section for before/after with dot8 cascade.

### What wasn't done (and why)

- **im2col / GEMM-based convolution**: standard for large CNNs but
  overkill for our use case (small kernel sizes, streaming single-step).
  The linearized dot product approach has no materialization overhead.
- **Winograd convolution**: only helps for kernel_size=3 or 5, and the
  overhead dominates at our filter counts.

---

## Stacked LSTM / Stacked GRU (`src/rnn/stacked_lstm.rs`, `src/rnn/stacked_gru.rs`)

Same optimizations as single-layer variants — they call the same
`matvec_bias_f32` / `matvec_f32` and gate functions. The stacked models
benefit more from dot8 because non-first layers have `in_size =
hidden + hidden` (typically ≥ 32), clearing the threshold.

---

## BNN (`src/bnn.rs`)

### Architecture

Pipeline: fp32 input matmul → binarize → N × XNOR+popcount layers →
fused output (masked accumulation from bits).

Binary layers pack ±1 weights as u64 words (1 bit per weight).
Inference replaces multiply-add with XNOR + popcount: `popcount(
!(weight_row XOR input_bits)) >= threshold`. Hidden size must be a
multiple of 64 for clean bit packing.

### Fused output — masked accumulation from bits

Original path: `unpack_bits` (64 conditional stores to ±1.0 f32 array)
→ `matvec_bias_f32` (dot product reading those f32s back). This writes
256B to float_scratch and reads it back — one full cache-line round-trip.

Replacement: `output_from_bits_simd` computes the output directly from
the packed bit pattern. The math:

```
y = Σ w[i] * sign(bit_i) = 2 * Σ(w[i] where bit=1) - Σ w[i]
```

`Σ w[i]` is precomputed at construction (`w_output_row_sum`). The masked
sum uses AVX2: expand each byte of the u64 to an 8-lane mask via
`set1 → AND → cmpeq` with `[1,2,4,8,16,32,64,128]`, then
`AND` with loaded weights, accumulate with `addps`. 8 iterations for
H=64 (one byte per iteration, 8 f32 weights per byte).

`#[inline(never)]` — the function has enough register pressure and
loop structure that inlining would bloat callers.

### Fused input — matvec + binarize in one pass

Original path: `matvec_bias_f32` writes 64 f32 results to float_scratch,
then `binarize` reads them back and packs into u64.

Replacement: `matvec_bias_binarize_f32` computes dot products using
`dot4_f32_m128` / `dot8_f32_m256` (same tiled infrastructure as MLP/LSTM),
adds bias, compares against zero with `cmpge_ps`, extracts comparison
results with `movemask_ps`, and shifts into the u64 word directly. Never
materializes the intermediate f32 values.

For `in_size >= 32`: dot8 path produces 8 dot products per iteration,
`_mm256_movemask_ps` → 8 bits. 8 iterations per 64-bit word.

For `in_size < 32`: dot4 path produces 4 dot products per iteration,
`_mm_movemask_ps` → 4 bits. 16 iterations per 64-bit word.

The combined effect: `float_scratch` is entirely eliminated from the
SIMD path (cfg-gated out of the struct). Only the scalar fallback
allocates it.

### LLVM auto-vectorization of binary_layer_forward

The binary layer hot loop is NOT hand-written SIMD — LLVM
auto-vectorizes it remarkably well:

- **vpshufb + vpsadbw**: SIMD nibble-lookup popcount. Processes 32 bytes
  of XNOR results per vpshufb instruction, horizontal byte-sum via
  vpsadbw.
- **vgf2p8affineqb**: GF(2^8) affine transformation (GFNI, Zen3+/Ice
  Lake+). Used as an alternative bit-manipulation path alongside vpshufb.
- **4× unrolled**: processes 16 neurons per loop iteration (4 groups
  of 4 u64s in YMM registers).
- **Single scalar `popcnt`** only for the remainder path.

This codegen matches or exceeds what hand-written SIMD would produce.
No manual intervention needed.

### Measured results

Before: original implementation with `unpack_bits + matvec_bias_f32`
output and separate `matvec_bias_f32 + binarize` input.

| Config | Before | After | Delta |
|--------|--------|-------|-------|
| BNN 8→64→1 (0 binary) | 110ns | 83ns | **-25%** |
| BNN 8→64→1 (1 binary) | 217ns | 195ns | **-10%** |
| BNN 8→64→1 (2 binary) | 325ns | 309ns | **-5%** |
| BNN 8→128→1 (2 binary) | 709ns | 666ns | **-6%** |

Binary layer marginal cost: ~112ns (H=64, wpr=1). Unchanged by the
optimizations — the savings are purely in the fp32 bookends.

### vs GBDT (competitive positioning)

| BNN Config | BNN | GBDT equivalent | GBDT | BNN advantage |
|------------|-----|-----------------|------|---------------|
| 1 binary layer (H=64) | 195ns | 50 trees × depth 6 | 233ns | **16% faster** |
| 2 binary layers (H=64) | 309ns | 100 trees × depth 6 | 487ns | **37% faster** |

### What wasn't done (and why)

- **Fused final-binary-layer + output**: would combine the last binary
  layer's popcount-threshold decision with output weight accumulation.
  Rejected because it mixes integer (popcount) and float (weight
  accumulation) pipelines in one loop body, creating loop-carried float
  dependencies that prevent LLVM from vectorizing across neurons. The
  current two-pass approach (pure-integer binary layer → pure-float
  output) lets LLVM vectorize each pass independently, which is faster.

- **Manual SIMD for binary_layer_forward**: LLVM already produces
  optimal code (vpshufb+vpsadbw popcount, 4× unrolled, 16 neurons/iter).
  Hand-writing this would match but not exceed the auto-vectorized output.

- **AVX-512 VPOPCNTDQ**: would replace the vpshufb popcount with native
  vector popcount (4 u64s in one instruction). Available only on Ice
  Lake+ and Zen4+. The vpshufb approach is already fast enough that the
  target-restriction isn't worth the codegen complexity.

- **Specialize for wpr=1**: the inner `for k in 0..wpr` loop is a
  runtime value. For H=64 (wpr=1), the loop is trivially one iteration.
  LLVM already handles this — the vectorized path loads contiguous
  neuron weights (which for wpr=1 are adjacent u64s), so no loop
  overhead exists in the vectorized case.

---

## Cross-cutting: things that apply everywhere

### `#[inline(never)]` for SIMD helpers

Both `mlp_tiled_simd_f32` and `conv_tiled_simd` use `#[inline(never)]`.
LLVM otherwise inlines these large functions into every call site,
bloating the caller's instruction footprint. The function call overhead
(~5 cycles) is negligible relative to the matvec work.

### Compile-time SIMD dispatch

All SIMD paths are selected at compile time via `cfg(target_feature)`,
not runtime `is_x86_feature_detected!()`. This means:
- Zero runtime dispatch cost.
- Build with `RUSTFLAGS="-C target-cpu=native"` for best codegen.
- The binary is not portable across CPU generations (acceptable for
  inference workloads deployed to known hardware).

### Scalar fallbacks

Every SIMD function has a scalar fallback compiled on non-x86 or
non-AVX2 targets. The scalar paths use the same algorithmic structure
(dot4 tiling, multi-accumulator) so correctness tests cover both paths.

### Activation functions

- Relu: `max(x, 0)` — trivially vectorized as `_mm*_max_ps(x, zero)`.
- Identity: no-op — just bias-add.
- Tanh/Sigmoid: Padé [7,6] rational approximation (LSTM/GRU gates).
  Not yet vectorized in MLP/Conv paths — those use scalar `activate_f32`
  for non-Relu activations. Vectorizing Tanh/Sigmoid for MLP would help
  if those activations become common in deployed models.

---

## Benchmark methodology

```bash
# Build
RUSTFLAGS="-C target-cpu=native" cargo bench --bench temporal_bench -p nexus-inference --no-run
RUSTFLAGS="-C target-cpu=native" cargo bench --bench predict_bench -p nexus-inference --features loader-lightgbm --no-run

# Baseline
taskset -c 0 ./target/release/deps/temporal_bench-* --bench --save-baseline <name>

# Compare
taskset -c 0 ./target/release/deps/temporal_bench-* --bench --baseline <name>
```

Turbo boost should be disabled for stable results:
```bash
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# ... run benchmarks ...
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

Without turbo disabled, individual runs can vary 5-15%. Always do A/B
comparison (`--save-baseline` → `--baseline`), never trust absolute
numbers from a single run.

---

## Results (2026-05-24)

All measurements: `RUSTFLAGS="-C target-cpu=native"`, pinned with
`taskset -c 0`, criterion A/B comparison (`--save-baseline` →
`--baseline`), turbo boost NOT disabled (adds ~5-15% noise on small
models, large-model deltas are reliable). AVX2+FMA target.

Baseline: commit `1e871a6` (pre MLP-SIMD, pre dot8). Current: commit
`3a60197` + dot8 threshold refinement.

### LSTM (single-layer)

| Config | Gate matrix | Before | After | Delta |
|--------|-------------|--------|-------|-------|
| 4→8→1 | 32×12 | 113ns | 105ns | ~-7% |
| 8→16→1 | 64×24 | 138ns | 137ns | ~-1% |
| 8→32→1 | 128×40 | 331ns | 297ns | **-10%** |
| 16→64→1 | 256×80 | 1313ns | 1066ns | **-19%** |

### LSTM (stacked)

| Config | Before | After | Delta |
|--------|--------|-------|-------|
| 8→32→1 ×2L | 853ns | 739ns | **-13%** |
| 8→32→1 ×3L | 1414ns | 1239ns | **-12%** |

### GRU (single-layer)

| Config | Before | After | Delta |
|--------|--------|-------|-------|
| 8→16→1 | 170ns | 173ns | ~0% (noise) |
| 8→32→1 | 343ns | 325ns | **-5%** |
| 16→64→1 | 1031ns | 909ns | **-12%** |

### GRU (stacked)

| Config | Before | After | Delta |
|--------|--------|-------|-------|
| 8→32→1 ×2L | 760ns | 711ns | **-6%** |
| 8→32→1 ×3L | 1254ns | 1201ns | **-4%** |

### Causal 1D Conv

| Config | conv_len | Before | After | Delta |
|--------|----------|--------|-------|-------|
| 4ch×4k×8f→1 | 16 | 49ns | 48ns | ~0% |
| 4ch×8k×16f→1 | 32 | 72ns | 68ns | **-6%** |
| 8ch×8k×32f→1 | 64 | 140ns | 115ns | **-18%** |

### MLP f32

MlpF32 benchmarks were added as part of this optimization work, so no
pre/post A/B comparison exists for them. Absolute latencies with all
optimizations (SIMD tiled + dot8):

| Config | Latency | Notes |
|--------|---------|-------|
| 8→16→1 relu | 53ns | in_size < 32, dot4 only |
| 16→32→8→1 relu | 106ns | mixed: some layers below threshold |
| 64→64→1 relu | 187ns | dot8 active |
| 32→32→32→32→1 relu | 229ns | dot8 active, 4 layers |
| 64→64→64→1 relu | 409ns | dot8 active, 3 layers |

For reference, equivalent MlpF64 configs (no SIMD tiled path):

| Config | MlpF64 | MlpF32 | f32 speedup |
|--------|--------|--------|-------------|
| 8→16→1 | 65ns | 53ns | 1.2× |
| 16→32→8→1 | 170ns | 106ns | 1.6× |
| 64→64→1 | 397ns | 187ns | 2.1× |

The f32 speedup exceeds 2× for 64-wide layers — the SIMD tiled path
fuses bias+relu in registers, and dot8 halves function call overhead.
The 1.2× for 8→16→1 is purely from f32 halving bandwidth; no SIMD
tiling fires (both dimensions below threshold).

### MLP f64 (control)

| Config | Before | After | Delta |
|--------|--------|-------|-------|
| 8→16→1 | 66ns | 65ns | ~0% |
| 16→32→8→1 | 156ns | 170ns | ~0% (noise) |
| 64→64→1 | 467ns | 455ns | ~0% |

No SIMD changes for f64 — confirms the improvements are from the
optimization work, not system state changes.

### GBDT (control)

| Config | Latency | Delta vs baseline |
|--------|---------|-------------------|
| 50×6 trees, 8 feat | 264ns | ~0% |
| 100×6 trees, 8 feat | 550ns | ~0% |
| 200×8 trees, 16 feat | 2.47µs | ~0% |

No changes to GBDT — already optimized with false-branch-next layout.

### LUT (control)

| Config | Latency |
|--------|---------|
| 2 feat × 10 bins | 6.6ns |
| 3 feat × 20 bins | 8.5ns |

O(1) lookup, no optimization needed.

### BNN

| Config | Latency | Binary layer marginal |
|--------|---------|----------------------|
| BNN 8→64→1 (0 binary) | 83ns | — (fp32 overhead only) |
| BNN 8→64→1 (1 binary) | 195ns | 112ns |
| BNN 8→64→1 (2 binary) | 309ns | 113ns |
| BNN 8→128→1 (2 binary) | 666ns | ~291ns |

Binary layer cost scales with H²/64 XNOR+popcount operations.
H=128 (wpr=2) costs ~2.6× the H=64 (wpr=1) layer — better than
the 4× theoretical increase due to SIMD amortization with more data.

### Current bottlenecks

Not profiled — these are architectural observations, not measured splits.

- **LSTM / GRU**: gate matvec dominates. The activation (Padé
  tanh/sigmoid) and output projection are small relative to the
  matrix-vector multiply. Further matvec improvement requires AVX-512
  on wider hardware, or reducing the matrix (quantization, pruning).
  GRU additionally can't fuse its two matvecs due to the reset gate.

- **MLP f32**: matvec across layers. Relu is free (fused in SIMD).
  Already near FMA throughput wall at 64-wide — diminishing returns
  from dot-product restructuring. LayerNorm now SIMD-vectorized
  (3-pass: mean, variance, normalize+affine+activate). Remaining LN
  overhead is 35-53% vs the same model without LN.

- **Conv**: split between linearization (memcpy circular buffer into
  contiguous layout) and convolution dot products. The linearization
  cost is fixed and doesn't shrink with SIMD improvements.

- **BNN**: binary layer dominates (54-57% of predict time for 1-binary
  config). Already auto-vectorized with optimal vpshufb+vpsadbw
  popcount. Further improvement requires AVX-512 VPOPCNTDQ (native
  vector popcount) or algorithmic change.

---

## Summary: what moves the needle

| Optimization | Where | Impact |
|---|---|---|
| dot4 shared input loads | everywhere | foundational — 4× input bandwidth reduction |
| dot4_f32_m128 batched hadd | matvec, MLP, Conv | eliminates scalar hsum round-trip |
| dot8_f32_m256 (in_size≥32) | matvec, MLP, Conv | 5-19% on medium/large models |
| Padé tanh/sigmoid 8-wide | LSTM/GRU gates | eliminates scalar activation bottleneck |
| MLP fused bias+relu in SIMD | MLP f32 | ~2× vs MlpF64 at 64-wide |
| Conv fused bias+relu in SIMD | Conv f32 | 6-18% vs scalar |
| GBDT false-branch-next layout | GBDT | ~50% of traversals sequential in L1 |
| SIMD LayerNorm (3-pass f32) | MLP f32 | ~4× vs scalar LN (65-74% reduction) |
| BNN fused output (masked sum from bits) | BNN | -14ns: eliminates unpack + output matmul |
| BNN fused input (matvec+binarize+movemask) | BNN | -14ns: eliminates float_scratch round-trip |
| `#[inline(never)]` on tiled helpers | MLP, Conv, BNN | prevents caller I-cache bloat |

---

## Future directions

Ordered by expected impact, not effort.

1. **AVX-512 on production hardware**: all dot8/dot4 code has AVX-512
   variants already written. Deploying to AVX-512-capable hardware
   doubles the SIMD width — expect another 30-50% on matvec-bound
   models without code changes (just a different target CPU).

2. **Vectorized Tanh/Sigmoid in MLP/Conv**: currently only LSTM/GRU
   gates use the SIMD Padé approximation. MLP and Conv fall back to
   scalar `activate_f32` for Tanh/Sigmoid/Gelu/Swish. If these
   activations are deployed, the same 8-wide Padé can be applied in
   the tiled helpers.

3. **Int8 quantized matvec**: halves memory bandwidth for weight loads.
   Relevant when models grow large enough that weights spill L2. Adds
   loader complexity (scale/zero-point per row or per tensor).

4. **GRU fused ih+hh matvec**: the ih and hh matrices could be
   concatenated into `(3H, input+hidden)` and a single matvec used,
   with the reset gate applied after splitting the output. This matches
   how LSTM already works. Would bring GRU improvements in line with
   LSTM. Requires validating numerical parity with PyTorch's split
   formulation.

5. **Layer Norm LSTM / Layer Norm GRU**: the same SIMD LayerNorm
   function (`layer_norm_simd_f32`) can be applied post-gate in LSTM
   and GRU variants. Literature shows LN stabilizes training for
   temporal models. Implementation would normalize hidden state after
   gate application — same 3-pass pattern, applied to hidden_size
   elements per step. Discuss with Martin whether this is worth the
   API surface (new constructors / weight format) vs keeping it
   MLP-only.

6. **Profiled bottleneck decomposition**: run `perf stat` or
   `perf record` on the temporal bench to get actual cycle attribution
   (matvec vs gates vs projection vs overhead). Current "bottleneck"
   claims are architectural reasoning, not measurement.

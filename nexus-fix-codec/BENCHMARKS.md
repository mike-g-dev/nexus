# nexus-fix-codec Benchmarks

Final cycle-level numbers for the FIX read/write primitives. For *why*
each path is the way it is — including the optimizations that were tried
and rejected — see [PERF_CATALOG.md](./PERF_CATALOG.md).

## Machine & methodology

- **CPU:** Intel Core Ultra 7 165U (Meteor Lake). AVX2 + BMI2, **no**
  AVX-512. Hybrid P/E cores; benchmarks pinned to a P-core.
- **Pinning:** `taskset -c 0`, best-of-N runs to suppress turbo/thermal
  noise (turbo left on; p50 is the stable signal, p99/p999 carry run
  noise).
- **Timing:** batched + fenced. Each sample times `BATCH` back-to-back
  calls between one `lfence`-serialized `rdtsc`/`rdtscp` pair, divided by
  `BATCH`. This resolves per-call costs below the ~16–30 cycle
  single-shot measurement floor (without batching, every sub-floor
  kernel reads a meaningless flat ~16 cyc). Numbers are sustained
  per-call cost. See `examples/perf_scan.rs`.
- **Workload:** a 15-field FIX 4.4 NewOrderSingle, 144 bytes. The
  encode/decode benches use **runtime** tags (worst case); generated
  encoders use *constant* tags, which constant-fold and run faster.

```bash
# AVX2 (deploy target)
RUSTFLAGS="-C target-cpu=native" cargo build --release --example perf_scan -p nexus-fix-codec
taskset -c 0 ./target/release/examples/perf_scan

# SSE2 (default cargo build)
cargo build --release --example perf_scan -p nexus-fix-codec
taskset -c 0 ./target/release/examples/perf_scan
```

---

## Headline: writer optimization

`encode_field` rewritten from per-byte bounds-checked stores +
`copy_from_slice` to a single up-front capacity check + unchecked
interior with `copy_nonoverlapping`. Same harness, only `writer.rs`
differs.

| Encode 15-field NewOrderSingle | before | after | Δ |
|--------------------------------|-------:|------:|---|
| AVX2, batched/fenced (best-of-15) | 178 cyc | **155 cyc** | **−13%** |
| AVX2, cyc/field | 11.8 | **10.3** | |
| SSE2, single-shot (best-of-12) | 172 cyc | **132 cyc** | **−23%** |

The bounds-hoist is SIMD-independent, so it helps both tiers; the larger
SSE2 single-shot delta reflects the simpler default-build codegen.
Real-world constant-tag encoding gains more (tag-width math and digit
writes constant-fold away). The rewrite also fixes a latent bug:
6–10 digit tags now encode correctly (the old path emitted a corrupt
`:` byte for tags ≥ 100000).

---

## Current numbers

### AVX2 (`target-cpu=native`)

| Path | p50 | cyc/field | cyc/byte |
|------|----:|----------:|---------:|
| `soh_iter` scan (mask-cached) | 21 | 1.4 | 0.15 |
| `FieldReader` decode (scan + tag + checksum) | 171 | 11.4 | 1.19 |
| `FieldWriter` encode | ~150 | ~10 | ~1.03 |

`checksum()` standalone body sum:

| Length | p50 (cyc) | cyc/byte |
|-------:|----------:|---------:|
| 16 B | 4 | 0.25 |
| 64 B | 7 | 0.11 |
| 143 B | 9 | 0.06 |
| 256 B | 7 | 0.03 |
| 512 B | 11 | 0.02 |

### SSE2 (default `cargo build`)

| Path | p50 | cyc/field |
|------|----:|----------:|
| `FieldReader` decode | 179 | 11.9 |
| `FieldWriter` encode | 152 | 10.1 |
| `checksum()` 512 B | 23 | 0.04 cyc/byte |

---

## What is *not* changed, and why

- **Decode (`FieldReader::next_field`)** — already near-optimal. The
  decode machine code is **byte-identical before/after this pass**
  (`find_next_soh` and the inlined `next_field` verified via `cargo asm`
  diff). `parse_tag` is a `lea`-based `tag*10+digit` chain; the scan is
  ~21 cyc and **AVX2 ≈ SSE2** for full decode → it is per-field-work
  bound, not SIMD bound. The ~150-cycle "overhead vs scan" is genuine
  tag-parse + checksum + field-yield work, already lean.
- **`checksum()`** — LLVM auto-vectorizes it (`vpaddb` byte-lane mod-256
  accumulation + a final `vpsadbw`); 0.02 cyc/byte at 512 B. Nothing to
  win.
- **Scan tiers** — unchanged.

Four reader/checksum micro-optimizations were implemented, measured, and
**reverted** (branchless SWAR gather, `parse_tag` digit cap, tag-10 cold
helper, deferred PSADBW reduction). The reasons — mostly "the machine
said no" — are in [PERF_CATALOG.md](./PERF_CATALOG.md#rejected-approaches).

---

## Verification

- 86 unit tests + 4 doc-tests pass on both SSE2 and AVX2.
- `cargo clippy --lib -- -D warnings`: clean on both tiers.
- The writer's `unsafe` is miri-clean and was adversarially reviewed
  (full input-grid brute-force + all 2³² tag values).

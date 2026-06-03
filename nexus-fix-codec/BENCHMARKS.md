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

## Type parsers (SWAR digit parsing)

Domain type parsers for FIX field values. All numeric parsing uses SWAR
(SIMD Within A Register) — 8 ASCII digits processed per block in 3
multiply+shift stages.

> **Methodology:** measured on the **batched/fenced** `perf_parse_cycles`
> harness — each sample times `BATCH = 100` back-to-back calls between one
> `lfence`-fenced `rdtsc`/`rdtscp` pair and divides, so sub-floor per-call
> costs resolve (a single-shot `rdtsc` pair bottoms out at a ~16–20 cycle floor
> that masks every cheap kernel). `taskset -c 0`, **turbo on** → p50 is the
> stable signal; p99 carries run noise. These are sustained back-to-back
> per-call costs. For publication-grade absolutes, re-run with `no_turbo` and
> best-of-N. A subset of these types only spans `Option<T>`/scalar paths today;
> the richer integration is mapped in [PERF_CATALOG.md](./PERF_CATALOG.md).
>
> Earlier revisions of this section carried *single-shot* numbers that were
> floor-inflated by ~16–20 cyc — the figures below supersede them.

```bash
cargo build --release --bench perf_parse_cycles -p nexus-fix-codec
taskset -c 0 ./target/release/deps/perf_parse_cycles-*
```

### Integer parsing

| Parser | Input | p50 | p99 |
|--------|-------|----:|----:|
| `parse_fix_bool` | `"Y"` | 2 | 2 |
| `parse_fix_int` | 8-digit | 22 | 29 |
| `parse_fix_int` | 16-digit | 22 | 40 |
| `parse_fix_int` | 19-digit (i64::MAX) | 21 | 25 |
| `parse_fix_int` | negative 8-digit | 21 | 31 |
| `parse_fix_uint` | `"256"` | 28 | 37 |
| `parse_fix_seqnum` | `"1000000"` | 36 | 40 |

`parse_fix_bool` is a 2-cycle single-byte match. Integer parsing is
~21–39 cycles, roughly flat across digit count (the SWAR block structure
processes 8 digits at once, so 8- and 16-digit inputs cost the same). The
short 1–4 digit cases (~38) carry the SWAR setup; ≥8 digits amortize it.

### Decimal parsing

| Input | p50 | p99 |
|-------|----:|----:|
| integer `"12345678"` | 28 | 40 |
| 12-digit `"50123.45000000"` | 54 | 71 |
| 4-digit `"99.50"` | 70 | 93 |
| 16-digit `"1234567.890123456"` | 75 | 104 |
| negative `"-123.456"` | 71 | 107 |
| sub-penny `"0.00000001"` | 52 | 79 |

Decimal adds dot-finding + split + recombine on top of the SWAR blocks.
~28 cycles for an integer-valued price, 52–75 for fractional prices. The
fractional cases pay a second SWAR block + the `10^scale` recombine.

### Date/time parsing

| Parser | Input | p50 | p99 |
|--------|-------|----:|----:|
| `FixDate::parse` | `"20260602"` | 9 | 9 |
| `FixTime::parse` | `"14:30:00"` | 18 | 20 |
| `FixTime::parse` | `"14:30:00.123456"` | 26 | 27 |
| `FixTimestamp::parse` | no fractional | 33 | 47 |
| `FixTimestamp::parse` | millis | 38 | 52 |
| `FixTimestamp::parse` | micros | 42 | 62 |
| `FixTimestamp::parse` | nanos | 44 | 82 |

`FixDate` is 9 cy (four fixed-width scalar digit reads + the Hinnant
rata-die). Timestamps compose date + time + epoch conversion additively,
33–44 cy for the full parse — no `i128` decompose on the parse side (that
cost is encode-only; see PERF_CATALOG.md).

---

## Type encoders (digit-pair LUT)

Domain type encoders — the inverse of the parsers above. Encoding uses
a 200-byte digit-pair lookup table (LUT): two ASCII digits per entry for
`00..99`, extracting digit pairs via `value % 100` / `value / 100`.
Zero-padded variants for date/time components and fractional parts.
Same measurement methodology as the parsers.

### Integer encoding

| Encoder | Input | p50 | p99 |
|---------|-------|----:|----:|
| `encode_fix_bool` | `true` | 1 | 1 |
| `encode_fix_int` | 8-digit | 19 | 40 |
| `encode_fix_int` | 16-digit | 25 | 33 |
| `encode_fix_int` | negative 8-digit | 20 | 24 |
| `encode_fix_uint` | `256` | 10 | 14 |
| `encode_fix_seqnum` | `1000000` | 18 | 20 |

### Decimal encoding

| Input | p50 | p99 |
|-------|----:|----:|
| integer `"12345678"` | 21 | 28 |
| 4-digit `"99.50"` | 27 | 42 |
| 8-digit `"50123.450"` | 30 | 51 |
| 16-digit `"1234567.890123456"` | 43 | 62 |
| negative `"-123.456"` | 25 | 37 |

### Date/time encoding

| Encoder | Input | p50 | p99 |
|---------|-------|----:|----:|
| `FixDate::encode` | `"20260602"` | 6 | 7 |
| `FixTime::encode` | `"14:30:00"` | 10 | 10 |
| `FixTime::encode` | `"14:30:00.123456"` | 23 | 39 |
| `FixTimestamp::encode` | no fractional | 60 | 98 |
| `FixTimestamp::encode` | millis | 70 | 149 |
| `FixTimestamp::encode` | micros | 73 | 108 |
| `FixTimestamp::encode` | nanos | 76 | 135 |

`FixDate`/`FixTime` encode in 6–23 cyc (digit-pair LUT writes). `FixDecimal`
encode is 21–43. **`FixTimestamp::encode` (60–76) is the heaviest path** —
it runs the reverse Hinnant algorithm (civil-from-days) to `decompose()` the
`i128` instant into date + time, and the `i128` `div_euclid`/`rem_euclid` in
that split dominates the cost. It is the one value-layer optimization target if
a timestamp encoder ever lands on a hot loop (reciprocal-multiply, or keep the
epoch math in `i64` — the wire domain fits). Parse is cheaper (33–44) because it
builds the instant additively with no decompose. See
[PERF_CATALOG.md](./PERF_CATALOG.md) for the full design rationale.

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

- 278 unit tests (267 default + 11 behind `nexus-decimal`) + 9 doc-tests pass
  on both SSE2 and AVX2.
- **12 property tests** (`tests/property.rs`, proptest): every value parser is
  proven to never panic on arbitrary / printable-ASCII / structured wire bytes,
  and every type round-trips (`construct → encode → parse`) across its full
  domain — all `i64`×scale decimals, all dates, all times including the
  leap-second band, all tenors, every `MonthYear` form, and the TZ types.
- `cargo clippy --all-targets --all-features -- -D warnings`: clean on both tiers.
- The writer's `unsafe` is miri-clean and was adversarially reviewed
  (full input-grid brute-force + all 2³² tag values). The value layer's one
  new `unsafe` (the `MultipleStringValue` borrowing iterator) is miri-clean.

The cycle harness (`benches/perf_parse_cycles.rs`) and the criterion suite
(`benches/parse_types.rs`) both cover parse **and** encode for every new type
(`FixMonthYear`, `FixTenor`, `FixTzTime`/`FixTzTimestamp`, `char`/text/
multi-value, plus the leap-second path). Numbers still need a controlled
(`taskset`, turbo-off) run to be authoritative.

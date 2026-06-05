# Changelog

All notable changes to nexus-stats are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/),
with the project-specific allowance that a minor bump may carry small,
narrowly-scoped breaking changes when external blast radius is
contained.

## [Unreleased]

## [6.0.1] — 2026-06-05

### Changed

- **nexus-stats-core** — Replaced `EventRateF64` with `EventRateU64` / `EventRateI64` using integer timestamps and bit-shift EMA. See core CHANGELOG for details.

## [6.0.0] — 2026-05-28

Type-variant audit: removed F32, I32, and I128 type variants across all
sub-crates. Use the F64/I64 equivalents. See each sub-crate's CHANGELOG
for the full removal list.

### Removed

- **nexus-stats-core 3.0.0** — 46 removed types (smoothing, control,
  monitoring, statistics modules). See core CHANGELOG.
- **nexus-stats-smoothing 2.0.0** — `HoltF32`, `SpringF32`, `Kalman1dF32`,
  `KamaF32`, `WindowedMedianF32`, `WindowedMedianI32` and their builders.
- **nexus-stats-detection 2.0.0** — `TrendAlertF32`, `MosumF32/I32/I128`,
  `MultiGateF32`, `PageHinkleyF32`, `AdwinF32`, `DistDriftF32`,
  `AutocorrelationF32/I32`, `CrossCorrelationF32`, `EntropyF32`,
  `PredictiveInfoBoundF32` and their builders.
- **nexus-stats-regression 2.0.0** — 20 removed types across regression,
  estimation, and learning modules. See regression CHANGELOG.
- **nexus-stats-control 2.0.0** — `PeakDetectorF32`, `PeakDetectorI32`,
  `PeakDetectorI128`.

## [5.1.0] — 2026-05-26

### Added

- **`BipowerVariationF64/F32`** — jump-robust volatility (no_std).
- **`RollSpreadF64/F32`** — Roll's implicit spread with Hasbrouck adjustment (std/libm).
- **`TwoScaleRvF64`** — noise-corrected realized variance (alloc+std/libm).
- **`HawkesIntensityF64/F32`** — self-exciting point process intensity (std/libm).
- **`PageHinkleyF64/F32`** — Page-Hinkley change detection (detection feature).
- **`AdwinF64/F32`** — ADWIN adaptive windowing (detection feature).
- **`PredictiveInfoBoundF64/F32`** — predictive information bound (detection feature).
- **`Ucb1F64/F32`** — UCB1 multi-armed bandit (regression feature).
- **`ThompsonBetaF64/F32`** — Thompson Sampling with Beta prior (regression feature).
- **`ThompsonGammaF64/F32`** — Thompson Sampling with Gamma prior (regression feature).
- **`EpsilonGreedyF64/F32`** — epsilon-greedy bandit (regression feature).
- **`Exp3F64/F32`** — EXP3 adversarial bandit (regression feature).

## [5.0.0] — 2026-05-18

Breaking: tracks nexus-stats-core 2.0.0.

### Added

- **`clock` module** re-exported from nexus-stats-core. `Clock` trait,
  `WallClock`, `EpochClock`.

### Changed

- `std` feature now also provides `WallClock`.

### Removed

- All `Instant`-based stats types (see nexus-stats-core 2.0.0 CHANGELOG).
- `Raw` suffix dropped from windowed/CoDel type names.

## [4.x] — workspace re-export pattern

`nexus-stats` is the umbrella crate that re-exports from the
focused subcrates: `nexus-stats-core`, `nexus-stats-control`,
`nexus-stats-detection`, `nexus-stats-regression`, and
`nexus-stats-smoothing`. The umbrella version tracks the workspace
release cadence; subcrate versions track per-area changes.

For per-algorithm or per-type changes, see the relevant subcrate's
CHANGELOG (or its git history).

## [4.2.2] and earlier

Earlier history is not documented in this CHANGELOG. See git history,
GitHub release notes, and the per-subcrate CHANGELOGs for details.

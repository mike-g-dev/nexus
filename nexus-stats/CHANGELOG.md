# Changelog

All notable changes to nexus-stats are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/),
with the project-specific allowance that a minor bump may carry small,
narrowly-scoped breaking changes when external blast radius is
contained.

## [Unreleased]

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

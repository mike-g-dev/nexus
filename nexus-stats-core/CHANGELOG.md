# Changelog

All notable changes to nexus-stats-core are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/),
with the project-specific allowance that a minor bump may carry small,
narrowly-scoped breaking changes when external blast radius is
contained.

## [Unreleased]

## [2.2.0] — 2026-05-20

### Added

- **`LpmF64/F32`** — Lower Partial Moments. Streaming downside risk
  with configurable target and integer order. Order 0 = shortfall
  probability, 1 = expected shortfall, 2 = semivariance.
  Convenience: `LpmF64::semivariance(target)`.
- **`CvarF64/F32`** — Conditional Value at Risk (Expected Shortfall).
  Streaming CVaR at configurable confidence level. Composes P²
  percentile internally.
- **`normalization` module** — Online feature normalization:
  - `ZScoreNormF64/F32` — EW z-score: `(x - mean) / std_dev`.
    Requires `std` or `libm`.
  - `MinMaxNormF64/F32` — Windowed min-max scaling to [0, 1].
  - `QuantileNormF64/F32` — Quantile transform via P² grid.
    Requires `alloc`.

## [2.1.0] — 2026-05-18

### Added

- **`BipowerVariationF64/F32`** — jump-robust volatility estimator using
  products of consecutive absolute returns. Barndorff-Nielsen & Shephard (2004).
- **`RollSpreadF64/F32`** — Roll's implicit spread estimator from
  autocovariance of consecutive price changes, with Hasbrouck (2009) adjustment.
  Requires `std` or `libm`.
- **`TwoScaleRvF64`** — two-scale realized variance, noise-corrected volatility
  estimator. Zhang, Mykland, Ait-Sahalia (2005). Requires `alloc` + `std`/`libm`.
- **`HawkesIntensityF64/F32`** — self-exciting point process intensity
  estimator. Models bursty event arrivals with exponential decay. Requires
  `std` or `libm`.

## [2.0.0] — 2026-05-18

Clock trait and Instant type removal.

### Added

- **`clock` module.** `Clock` trait (`stamp() -> u64`), `WallClock` (std,
  wraps `Instant`, returns elapsed nanos), `EpochClock` (manual/test clock
  with `set()`/`advance()`). Stats types accept `u64` timestamps; the caller
  owns the clock.
- **`elapsed(from, to)` helper** — saturating u64 subtraction.

### Removed

- **All `Instant`-based stats types.** `WindowedMax/MinF64/F32/I64/I32/I128`,
  `CoDelF64/F32/I64/I32/I128` (Instant variants), `LivenessInstant`,
  `EventRateInstant`, and `BucketAccumulator::update_instant()`. Epoch
  management belongs in the Clock, not the stats type.

### Changed

- **Renamed Raw variants to canonical.** `WindowedMaxF64Raw` → `WindowedMaxF64`,
  `CoDelF64Raw` → `CoDelF64`, etc. The `Raw` suffix was disambiguation for
  the now-removed Instant variants.

## [1.2.1] and earlier

Earlier history is not documented in this CHANGELOG. See git history
and GitHub release notes for details.

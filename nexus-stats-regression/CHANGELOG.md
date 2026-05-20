# Changelog

All notable changes to nexus-stats-regression are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/),
with the project-specific allowance that a minor bump may carry small,
narrowly-scoped breaking changes when external blast radius is
contained.

## [Unreleased]

## [1.3.0] — 2026-05-19

### Added

- **`Ucb1F64/F32`** — UCB1 multi-armed bandit. Deterministic selection,
  no RNG needed. Auer, Cesa-Bianchi, Fischer (2002).
- **`ThompsonBetaF64/F32`** — Thompson Sampling with Beta conjugate
  prior for binary/[0,1] rewards. Thompson (1933).
- **`ThompsonGammaF64/F32`** — Thompson Sampling with Gamma conjugate
  prior for positive continuous rewards.
- **`EpsilonGreedyF64/F32`** — Epsilon-greedy bandit. Simplest baseline.
- **`Exp3F64/F32`** — EXP3 adversarial bandit. Robust to non-stochastic
  rewards. Auer, Cesa-Bianchi, Freund, Schapire (2002).
- UCB1, ThompsonBeta, ThompsonGamma, and EpsilonGreedy support
  exponential discounting via `decay` parameter for non-stationary
  reward environments. EXP3 handles non-stationarity through its
  `gamma` exploration mixing rate.
- Internal sampling utilities (Marsaglia polar, Marsaglia-Tsang Gamma,
  Beta from Gamma ratio).

All bandit types require `alloc` + (`std` or `libm`).

## [1.2.0] and earlier

Earlier history is not documented in this CHANGELOG. See git history
and GitHub release notes for details.

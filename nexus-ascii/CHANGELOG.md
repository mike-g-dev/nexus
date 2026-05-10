# Changelog

All notable changes to nexus-ascii are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/),
with the project-specific allowance that a minor bump may carry small,
narrowly-scoped breaking changes when external blast radius is
contained.

## [Unreleased]

## [1.6.2] — 2026-05-10

Doc-only release. No API or behavior change.

### Changed

- Missing-doc additions across the public surface. Workspace-wide
  `#![warn(missing_docs)]` now in effect for the crate.

## [1.6.1] — 2026-05-08

### Fixed

- Minor follow-ups in `format.rs`, `parse.rs`, and `text.rs` after the
  flat-cap relaxation work in 1.6.0.

## [1.6.0] — 2026-05-08

The "non-multiple-of-8 flat capacities" release. `FlatAsciiString<CAP>`
and `FlatAsciiText<CAP>` now accept any `CAP >= 1`, lifting the prior
`CAP >= 8 && CAP % 8 == 0` constraint that ruled out short capacities
like `4` for fixed-width order tags.

### Added

- **`FlatAsciiString4`** and **`FlatAsciiText4`** type aliases for the
  newly-supported 4-byte capacity.

### Changed

- **Compile-time check relaxed** on `FlatAsciiString<CAP>` /
  `FlatAsciiText<CAP>` from `CAP >= 8 && CAP % 8 == 0` to `CAP >= 1`.
  Reads now fall back from u64 chunked loads to byte-by-byte loads
  when the capacity is non-multiple-of-8.
- `widen` and `tighten` now compile-time reject `NEW_CAP == 0`.

## [1.5.2] and earlier

Earlier history is not documented in this CHANGELOG. See git history
and GitHub release notes for details.

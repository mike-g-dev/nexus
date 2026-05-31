# Changelog

All notable changes to nexus-shm are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Foundation layer: `Pod` trait, segment control block, mmap-backed
  `Segment`, two-tier liveness (atomic status + OFD lock).
- `Segment::create` / `Segment::attach` for owner and peer roles.
- `Status` (Tier 1, atomic) and `Liveness` (Tier 2, OFD `F_OFD_GETLK`)
  for two-tier peer liveness detection.
- `MapOptions` with `populate` and `huge_pages` flags.
- Criterion benchmark for OFD liveness query cost.

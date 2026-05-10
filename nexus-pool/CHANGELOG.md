# Changelog

All notable changes to nexus-pool are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/),
with the project-specific allowance that a minor bump may carry small,
narrowly-scoped breaking changes when external blast radius is
contained.

## [Unreleased]

## [1.0.4] — 2026-05-08

### Added

- **`#[must_use]` on `Pooled<T>`** in both `local::Pool` and
  `sync::Pool`. Dropping the guard immediately returns the object to
  the pool, so silently discarding the result of `pool.acquire()` was
  almost always a bug — the lint now catches it.

## [1.0.3] and earlier

Earlier history is not documented in this CHANGELOG. See git history
and GitHub release notes for details.

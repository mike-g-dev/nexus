# Changelog

All notable changes to nexus-rt are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/),
with the project-specific allowance that a minor bump may carry small,
narrowly-scoped breaking changes when external blast radius is
contained.

## [Unreleased]

## [2.3.0] — 2026-05-08

Ergonomics around `Res<T>` and `ResMut<T>`. Lets handler bodies pass
the wrappers themselves (not just `&T` / `&mut T`) into inner functions
without moving.

### Added

- **`Res<T>: Copy + Clone`**, regardless of `T`. Manual impls (not
  derived) so the bounds depend only on the inner `&T` field, which is
  always `Copy`. A derive would have erroneously required `T: Clone`.
  This means user code can now pass `Res<T>` to inner functions
  multiple times without `.clone()` ceremony.
- **`ResMut::reborrow(&mut self) -> ResMut<'_, T>`**. The exclusive-
  borrow counterpart to `Res<T>: Copy`. Pass `ResMut<T>` to inner
  functions without moving — the original is frozen for the duration
  of the reborrow, then usable again. Analogous to `&mut *x` reborrow
  for `&mut T`.

### Notes on breakage

- This release is a **minor bump** even though existing user code that
  shadowed an outer `Res<T>` with a different value via something like
  `let res = res.clone();` will now silently `Copy` instead. Behavior
  is the same in practice, but the inferred `Clone` bound on user
  generics may shift. Watch for diagnostic regressions, not runtime
  ones.

## [2.2.0] and earlier

Earlier history is not documented in this CHANGELOG. See git history
and GitHub release notes for details.

# Changelog

All notable changes to nexus-logbuf are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/),
with the project-specific allowance that a minor bump may carry small,
narrowly-scoped breaking changes when external blast radius is
contained (see "Migration notes" below).

## [Unreleased]

## [2.2.0] — 2026-05-08

The "honest error contracts" release. `len == 0` is a wire-format
sentinel (the "uncommitted" marker in record headers) — letting it
through silently hangs the consumer. Prior versions treated `len == 0`
as a recoverable runtime error (`ZeroLength` variant in three error
types); this release treats it as the precondition violation it
always was, panicking at the queue layer. Closes
[#171](https://github.com/Abso1ut3Zer0/nexus/issues/171).

Also rebuilds the channel-layer error types around their actual
runtime states: `Sender::send` has exactly one runtime failure mode
(receiver gone), so `SendError` collapses from a 2-variant enum to a
unit struct `ChannelClosed`. The "`SendError` is overloaded" naming
collision with `nexus-channel::SendError<T>(T)` and
`nexus-async-rt::SendError<T>(T)` is resolved as a side-effect.

### Breaking changes

- **`pub enum TryClaimError` removed.** Replaced by `pub struct BufferFull`.
  - Before: `try_claim(len) -> Result<WriteClaim<'_>, TryClaimError>`
  - After:  `try_claim(len) -> Result<WriteClaim<'_>, BufferFull>`
  - Match arms drop the wrapper:
    ```rust
    // before
    match prod.try_claim(n) {
        Ok(c) => ...,
        Err(TryClaimError::Full) => retry,
        Err(TryClaimError::ZeroLength) => bug(),
    }
    // after
    match prod.try_claim(n) {
        Ok(c) => ...,
        Err(BufferFull) => retry,
    }
    ```
- **`pub enum SendError` removed.** Replaced by `pub struct ChannelClosed`.
  Affects both `channel::spsc::Sender::send` and `channel::mpsc::Sender::send`:
  - Before: `send(len) -> Result<WriteClaim<'_>, SendError>`
  - After:  `send(len) -> Result<WriteClaim<'_>, ChannelClosed>`
- **`TrySendError::ZeroLength` removed.** `TrySendError` now has only
  `Full` and `Disconnected` variants. Both `channel::spsc` and
  `channel::mpsc`.
- **`Producer::try_claim`, `Sender::send`, `Sender::try_send` now panic
  on `len == 0`.** Previously returned an error variant. The check is
  structural: `len == 0` is reserved by the wire format as the
  "uncommitted" sentinel in record headers, so allowing it through
  would silently hang the consumer. Always-on `assert!`, not
  `debug_assert!` — the cost is identical to the old `if len == 0
  return Err(...)` (one cmp+branch, predictable), only the response
  to violation changes. Aborting a non-zero claim (drop the
  `WriteClaim` without committing) is fully supported and writes a
  skip marker the consumer handles correctly.

### Migration notes

Search-and-replace covers most callers:

| 2.1.x | 2.2.0 |
|---|---|
| `TryClaimError::Full` (in match) | `BufferFull` |
| `TryClaimError::ZeroLength` (in match) | remove arm; assert it can't happen, or fix the bug |
| `SendError::Disconnected` | `ChannelClosed` |
| `SendError::ZeroLength` | remove arm |
| `TrySendError::ZeroLength` | remove arm |
| `Result<_, TryClaimError>` (function sig) | `Result<_, BufferFull>` |
| `Result<_, SendError>` (function sig) | `Result<_, ChannelClosed>` |

If your code has a path that actually called `try_claim(0)` /
`send(0)` / `try_send(0)` and handled the error, that path was
already a programmer-bug-as-soft-error — the fix is to ensure
`len > 0` upstream, not to handle the panic.

Since this is a minor bump with breaking changes, the `2.1.5`
release will be yanked once `2.2.0` publishes. External consumers
on a default `^2` spec will pick up `2.2.0` on `cargo update`; pin
to `=2.1.5` if you need to defer.

## [2.1.5] and earlier

Earlier 2.x versions are not documented in this CHANGELOG. See
git history and GitHub release notes for details.

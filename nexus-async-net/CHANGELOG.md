# Changelog

All notable changes to nexus-async-net are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.6.1] — 2026-05-05

The "TLS handshake byte-loss" release. Picks up the
[nexus-net 0.6.1](../nexus-net/CHANGELOG.md) fix at all three async
TLS call sites and bumps the `nexus-async-rt` dependency declaration
to 0.5.0 (the hardening-series release that landed earlier today).

### Fixed

- `ws::nexus::stream::handshake_tls` (line 71),
  `rest::nexus::connection` (line 71), and
  `maybe_tls::nexus::TlsInner::poll_read` (line 83) now use
  `TlsCodec::read_and_process_tls` instead of the bare
  `read_tls` + `process_new_packets` pair. Closes the production
  bug birch reported in
  [#200](https://github.com/Abso1ut3Zer0/nexus/issues/200) — the
  async client could silently drop unconsumed handshake bytes when
  the server sent a multi-record burst exceeding rustls's per-call
  read cap (long cert chains, large certs, OCSP stapling). The
  observed symptom was
  `Io(UnexpectedEof, "closed during TLS handshake")` after the
  server's timeout (~15-30s).

### Changed

- Dependency declaration: `nexus-async-rt` 0.4.0 → 0.5.0. The 0.5.0
  release was the production-hardening version of nexus-async-rt
  (see its own CHANGELOG for details). nexus-async-net's `nexus`
  backend benefits from those fixes transitively (TaskRef-based
  refcount discipline, `dispose_terminal` routing, intrusive
  cancellation list, `shutdown_quiesce` API).

### Internal

- New end-to-end regression test:
  `tests/ws_nexus_tls_loopback.rs::nexus_async_wss_echo_with_oversize_handshake_burst`
  drives a real wss:// connect through the async client against a
  localhost TLS+WS echo server, with a 10-cert ECDSA-P256 chain
  forcing the handshake burst past rustls's 4096-byte per-call
  deframer cap. Pre-fix this test reproduces birch's exact symptom
  (`Io(UnexpectedEof, "closed during TLS handshake")`); post-fix it
  passes. Hermetic, no network access required.

### Migration notes

`cargo update -p nexus-async-net` is the only change required.
Public API is unchanged.

## [0.6.0] — 2026-04-15

Earlier 0.6.x and prior versions are not documented in this
CHANGELOG. See git history and GitHub release notes for details.

# Changelog

All notable changes to nexus-net are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.6.1] — 2026-05-05

The "TLS handshake byte-loss" release. Closes
[#200](https://github.com/Abso1ut3Zer0/nexus/issues/200) — TLS
handshakes against servers that emit multi-record handshake bursts
exceeding rustls's per-call read cap (e.g., long cert chains, OCSP
stapling, large certs) could silently drop unconsumed bytes,
stalling the handshake and producing
`Io(UnexpectedEof, "closed during TLS handshake")` after the
server's timeout.

### Added

- `TlsCodec::read_and_process_tls(&[u8]) -> Result<usize, TlsError>` —
  helper that loops `read_tls` + `process_new_packets` until the
  entire input slice is consumed. Use anywhere code reads ciphertext
  bytes into a buffer first (async paths, sans-IO pipelines, IO
  drivers without a `Read` trait) and then needs to push them into
  the codec. Returns `Ok(src.len())` on success or
  `TlsError::Io(InvalidData)` if rustls's deframer can't make
  progress despite intervening `process_new_packets` calls.

### Fixed

- `TlsStream::handshake_async` (line 166) and `poll_read` (line 207)
  now use `read_and_process_tls` instead of the bare
  `read_tls` + `process_new_packets` pair. Pre-fix,
  `rustls::Connection::read_tls(&mut Cursor)` could consume only part
  of the slice and the unconsumed tail was silently lost when `tmp`
  got overwritten on the next loop iteration. This was the bug birch
  reported in #200 against `wss://ws-subscriptions-clob.polymarket.com`.

### Internal

- New regression tests: `tls::codec::tests::read_and_process_tls_handles_oversize_burst`
  (in-process, deterministic; uses 10-cert ECDSA-P256 chain to push
  the burst past rustls's `READ_SIZE = 4096`) and
  `wss_echo::local_wss_echo_with_oversize_handshake_burst` (hermetic
  localhost TLS+WS+frame-echo). Plus
  `tls::codec::tests::bare_read_tls_partially_consumes_large_slice`
  documents rustls's contract directly.
- `read_tls`'s doc-comment now points readers at
  `read_and_process_tls` for the common case (callers who already
  have bytes in a buffer).

### Migration notes

`cargo update -p nexus-net` is the only change required for most
users. The bare `read_tls` API is unchanged — anyone implementing
their own consume loop on top of it continues to work. New code
that pre-buffers bytes should reach for `read_and_process_tls`
instead.

## [0.6.0] — 2026-04-15

Earlier 0.6.x and prior versions are not documented in this
CHANGELOG. See git history and GitHub release notes for details.

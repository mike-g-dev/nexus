# Changelog

All notable changes to nexus-web are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Initial release. WebSocket, HTTP/1.1, and REST protocol implementations
extracted from nexus-net 0.7.x
([#413](https://github.com/Abso1ut3Zer0/nexus/issues/413)).

### Added

- **`ws`** — WebSocket RFC 6455: `FrameReader`, `FrameWriter`, `Client`,
  `Message`, SIMD XOR masking, UTF-8 validation. 517/517 Autobahn
  conformance.
- **`http`** — HTTP/1.1: `ResponseReader`, `RequestReader`,
  `ChunkedDecoder`, `write_request`/`write_response`. httparse-backed,
  SIMD-accelerated header parsing.
- **`rest`** — REST client: `RequestWriter` (typestate builder),
  `Client` (transport), `RestResponse`. Zero per-request allocation.
- Re-exports nexus-net primitives (`buf`, `tls`, `wire`, `MaybeTls`,
  `ParserSink`, `WireStream`) so downstream crates don't need to
  depend on nexus-net directly.

### Migration from nexus-net

Replace `nexus_net::ws`, `nexus_net::http`, `nexus_net::rest` paths
with `nexus_web::ws`, `nexus_web::http`, `nexus_web::rest`. All
primitives (`buf`, `tls`, `wire`) remain in nexus-net but are also
re-exported from nexus-web.

//! HTTP/1.1 REST client integration tests against httpbin.org.
//!
//! These tests validate the full stack over real TLS connections:
//! request construction, wire format, response parsing, keep-alive,
//! status codes, headers, and body handling.
//!
//! Equivalent of Autobahn for WebSocket — proves protocol conformance
//! against a real server.
//!
//! Run with:
//!   cargo test -p nexus-web --all-features --test httpbin -- --ignored --nocapture
//!
//! Requires network access to httpbin.org and the `tls` feature.

#![cfg(feature = "tls")]

use nexus_web::MaybeTls;
use nexus_web::http::ResponseReader;
use nexus_web::rest::{Client, RequestWriter};
use nexus_web::tls::TlsConfig;

fn setup() -> (
    RequestWriter,
    ResponseReader,
    Client<MaybeTls<std::net::TcpStream>>,
) {
    let tls = TlsConfig::new().unwrap();
    let mut writer = RequestWriter::new("httpbin.org").unwrap();
    let _ = writer.default_header("Accept", "application/json");
    let reader = ResponseReader::new(64 * 1024).max_body_size(64 * 1024);
    let conn = Client::builder()
        .tls(&tls)
        .disable_nagle()
        .connect("https://httpbin.org")
        .unwrap();
    (writer, reader, conn)
}

// =========================================================================
// GET
// =========================================================================

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_get() {
    let (mut writer, mut reader, mut conn) = setup();

    let req = writer.get("/get").finish().unwrap();
    let resp = conn.send(req, &mut reader).unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.body_str().unwrap();
    assert!(body.contains("\"url\""));
    assert!(body.contains("httpbin.org/get"));
}

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_get_with_query_params() {
    let (mut writer, mut reader, mut conn) = setup();

    let req = writer
        .get("/get")
        .query("symbol", "BTC-USD")
        .query("limit", "100")
        .finish()
        .unwrap();
    let resp = conn.send(req, &mut reader).unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.body_str().unwrap();
    assert!(body.contains("\"symbol\""));
    assert!(body.contains("BTC-USD"));
    assert!(body.contains("\"limit\""));
    assert!(body.contains("100"));
}

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_get_with_special_chars_in_query() {
    let (mut writer, mut reader, mut conn) = setup();

    let req = writer
        .get("/get")
        .query("q", "hello world&more=yes")
        .finish()
        .unwrap();
    let resp = conn.send(req, &mut reader).unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.body_str().unwrap();
    assert!(body.contains("hello world&more=yes") || body.contains("hello+world"));
}

// =========================================================================
// POST
// =========================================================================

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_post_json() {
    let (mut writer, mut reader, mut conn) = setup();

    let json = r#"{"symbol":"BTC-USD","side":"buy","quantity":"0.001"}"#;
    let req = writer
        .post("/post")
        .header("Content-Type", "application/json")
        .body(json.as_bytes())
        .finish()
        .unwrap();
    let resp = conn.send(req, &mut reader).unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.body_str().unwrap();
    assert!(body.contains("BTC-USD"));
}

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_post_with_custom_headers() {
    let (mut writer, mut reader, mut conn) = setup();

    let req = writer
        .post("/post")
        .header("X-Custom-Header", "test-value-123")
        .header("X-Another", "another-value")
        .body(b"{}")
        .finish()
        .unwrap();
    let resp = conn.send(req, &mut reader).unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.body_str().unwrap();
    assert!(body.contains("X-Custom-Header"));
    assert!(body.contains("test-value-123"));
}

// =========================================================================
// PUT / DELETE / PATCH
// =========================================================================

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_put() {
    let (mut writer, mut reader, mut conn) = setup();

    let req = writer
        .put("/put")
        .body(b"{\"update\":true}")
        .finish()
        .unwrap();
    let resp = conn.send(req, &mut reader).unwrap();

    assert_eq!(resp.status(), 200);
    assert!(resp.body_str().unwrap().contains("update"));
}

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_delete() {
    let (mut writer, mut reader, mut conn) = setup();

    let req = writer.delete("/delete").finish().unwrap();
    let resp = conn.send(req, &mut reader).unwrap();

    assert_eq!(resp.status(), 200);
}

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_patch() {
    let (mut writer, mut reader, mut conn) = setup();

    let req = writer
        .request(nexus_web::rest::Method::Patch, "/patch")
        .body(b"{\"patch\":true}")
        .finish()
        .unwrap();
    let resp = conn.send(req, &mut reader).unwrap();

    assert_eq!(resp.status(), 200);
    assert!(resp.body_str().unwrap().contains("patch"));
}

// =========================================================================
// Status codes
// =========================================================================

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_status_404() {
    let (mut writer, mut reader, mut conn) = setup();

    let req = writer.get("/status/404").finish().unwrap();
    let resp = conn.send(req, &mut reader).unwrap();
    assert_eq!(resp.status(), 404);
}

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_status_500() {
    let (mut writer, mut reader, mut conn) = setup();

    let req = writer.get("/status/500").finish().unwrap();
    let resp = conn.send(req, &mut reader).unwrap();
    assert_eq!(resp.status(), 500);
}

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_status_204() {
    let (mut writer, mut reader, mut conn) = setup();

    let req = writer.get("/status/204").finish().unwrap();
    let resp = conn.send(req, &mut reader).unwrap();
    assert_eq!(resp.status(), 204);
    assert_eq!(resp.body().len(), 0);
}

// =========================================================================
// Response headers
// =========================================================================

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_response_headers() {
    let (mut writer, mut reader, mut conn) = setup();

    let req = writer
        .get("/response-headers")
        .query("X-Test-Header", "nexus-web")
        .finish()
        .unwrap();
    let resp = conn.send(req, &mut reader).unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.header("X-Test-Header"), Some("nexus-web"));
}

// =========================================================================
// Keep-alive / connection reuse
// =========================================================================

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_keep_alive() {
    let (mut writer, mut reader, mut conn) = setup();

    // First request
    let req = writer.get("/get").finish().unwrap();
    let resp = conn.send(req, &mut reader).unwrap();
    assert_eq!(resp.status(), 200);
    drop(resp);

    // Second request on same connection
    let req = writer.get("/get").query("req", "2").finish().unwrap();
    let resp = conn.send(req, &mut reader).unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.body_str().unwrap().contains("\"req\""));
    drop(resp);

    // Third request — POST
    let req = writer.post("/post").body(b"third").finish().unwrap();
    let resp = conn.send(req, &mut reader).unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.body_str().unwrap().contains("third"));
}

// =========================================================================
// Large response
// =========================================================================

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_large_response() {
    let (mut writer, _, mut conn) = setup();
    let mut reader = ResponseReader::new(64 * 1024).max_body_size(64 * 1024);

    let req = writer.get("/bytes/16384").finish().unwrap();
    let resp = conn.send(req, &mut reader).unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.body().len(), 16384);
}

// =========================================================================
// Raw URL path
// =========================================================================

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_raw_url() {
    let (mut writer, mut reader, mut conn) = setup();

    let req = writer.get_raw("/get?pre=formed&url=true").finish().unwrap();
    let resp = conn.send(req, &mut reader).unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.body_str().unwrap();
    assert!(body.contains("\"pre\""));
    assert!(body.contains("\"url\""));
}

// =========================================================================
// Default headers
// =========================================================================

#[test]
#[ignore = "requires network access to httpbin.org"]
fn httpbin_default_headers_sent() {
    let tls = TlsConfig::new().unwrap();
    let mut writer = RequestWriter::new("httpbin.org").unwrap();
    writer
        .default_header("X-Default-Test", "default-val")
        .unwrap();
    let mut reader = ResponseReader::new(64 * 1024).max_body_size(64 * 1024);
    let mut conn = Client::builder()
        .tls(&tls)
        .disable_nagle()
        .connect("https://httpbin.org")
        .unwrap();

    let req = writer.get("/headers").finish().unwrap();
    let resp = conn.send(req, &mut reader).unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.body_str().unwrap();
    assert!(body.contains("X-Default-Test"));
    assert!(body.contains("default-val"));
}

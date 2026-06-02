//! Async HTTP/1.1 REST client integration tests against httpbin.org.
//!
//! Mirrors nexus-net's httpbin tests but uses HttpConnection.
//! Proves the async adapter produces identical results to sync.
//!
//! Run with:
//!   cargo test -p nexus-async-web --test httpbin -- --ignored --nocapture

use nexus_async_web::rest::HttpConnection;
use nexus_net::tls::TlsConfig;
use nexus_web::http::ResponseReader;
use nexus_web::rest::RequestWriter;

async fn setup() -> (
    RequestWriter,
    ResponseReader,
    HttpConnection<nexus_async_web::rest::MaybeTls>,
) {
    let tls = TlsConfig::new().unwrap();
    let mut writer = RequestWriter::new("httpbin.org").unwrap();
    let _ = writer.default_header("Accept", "application/json");
    let reader = ResponseReader::new(64 * 1024).max_body_size(64 * 1024);
    let conn = nexus_async_web::rest::HttpConnectionBuilder::new()
        .tls(&tls)
        .disable_nagle()
        .connect("https://httpbin.org")
        .await
        .unwrap();
    (writer, reader, conn)
}

#[tokio::test]
#[ignore = "requires network access to httpbin.org"]
async fn async_httpbin_get() {
    let (mut writer, mut reader, mut conn) = setup().await;

    let req = writer.get("/get").finish().unwrap();
    let resp = conn.send(req, &mut reader).await.unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.body_str().unwrap();
    assert!(body.contains("\"url\""));
    assert!(body.contains("httpbin.org/get"));
}

#[tokio::test]
#[ignore = "requires network access to httpbin.org"]
async fn async_httpbin_get_with_query() {
    let (mut writer, mut reader, mut conn) = setup().await;

    let req = writer
        .get("/get")
        .query("symbol", "ETH-USD")
        .query("limit", "50")
        .finish()
        .unwrap();
    let resp = conn.send(req, &mut reader).await.unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.body_str().unwrap();
    assert!(body.contains("ETH-USD"));
    assert!(body.contains("50"));
}

#[tokio::test]
#[ignore = "requires network access to httpbin.org"]
async fn async_httpbin_post_json() {
    let (mut writer, mut reader, mut conn) = setup().await;

    let json = r#"{"action":"place_order"}"#;
    let req = writer
        .post("/post")
        .header("Content-Type", "application/json")
        .body(json.as_bytes())
        .finish()
        .unwrap();
    let resp = conn.send(req, &mut reader).await.unwrap();

    assert_eq!(resp.status(), 200);
    assert!(resp.body_str().unwrap().contains("place_order"));
}

#[tokio::test]
#[ignore = "requires network access to httpbin.org"]
async fn async_httpbin_keep_alive() {
    let (mut writer, mut reader, mut conn) = setup().await;

    // Three sequential requests on the same connection
    for i in 1..=3 {
        let req = writer
            .get("/get")
            .query("req", &i.to_string())
            .finish()
            .unwrap();
        let resp = conn.send(req, &mut reader).await.unwrap();
        assert_eq!(resp.status(), 200);
        drop(resp);
    }
}

#[tokio::test]
#[ignore = "requires network access to httpbin.org"]
async fn async_httpbin_status_codes() {
    let (mut writer, mut reader, mut conn) = setup().await;

    for (path, expected) in [
        ("/status/200", 200),
        ("/status/404", 404),
        ("/status/204", 204),
    ] {
        let req = writer.get(path).finish().unwrap();
        let resp = conn.send(req, &mut reader).await.unwrap();
        assert_eq!(resp.status(), expected, "failed for {path}");
        drop(resp);
    }
}

#[tokio::test]
#[ignore = "requires network access to httpbin.org"]
async fn async_httpbin_response_headers() {
    let (mut writer, mut reader, mut conn) = setup().await;

    let req = writer
        .get("/response-headers")
        .query("X-Async-Test", "works")
        .finish()
        .unwrap();
    let resp = conn.send(req, &mut reader).await.unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.header("X-Async-Test"), Some("works"));
}

#[tokio::test]
#[ignore = "requires network access to httpbin.org"]
async fn async_httpbin_large_response() {
    let (mut writer, _, mut conn) = setup().await;
    let mut reader = ResponseReader::new(64 * 1024).max_body_size(64 * 1024);

    let req = writer.get("/bytes/8192").finish().unwrap();
    let resp = conn.send(req, &mut reader).await.unwrap();

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.body().len(), 8192);
}

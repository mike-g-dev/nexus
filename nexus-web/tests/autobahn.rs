//! Autobahn WebSocket conformance test (517 cases).
//!
//! Runs against Autobahn's `fuzzingserver` via Podman.
//! Skipped in normal `cargo test` and CI.
//!
//! **Run after any changes to:** `ws/frame_reader.rs`, `ws/frame_writer.rs`,
//! `ws/message.rs`, `ws/stream.rs`, `ws/mask.rs`, `ws/handshake.rs`,
//! or buffer primitives.
//!
//! ```bash
//! # Start Autobahn (requires Podman)
//! podman run --rm -d --network=host \
//!     -v "${PWD}/nexus-web/tests/autobahn:/config:Z" \
//!     -v "${PWD}/target/autobahn-reports:/reports:Z" \
//!     --name autobahn \
//!     docker.io/crossbario/autobahn-testsuite \
//!     wstest -m fuzzingserver -s /config/fuzzingserver.json
//!
//! # Run tests
//! cargo test -p nexus-web --test autobahn -- --ignored --nocapture
//!
//! # Stop
//! podman stop autobahn
//! ```

use nexus_web::ws::{Client, CloseCode, Error, Message, OwnedMessage, ProtocolError};
use std::net::TcpStream;

const AUTOBAHN_HOST: &str = "127.0.0.1:9001";
const AGENT: &str = "nexus-web";

fn make_ws(path: &str) -> Client<TcpStream> {
    let tcp = TcpStream::connect(AUTOBAHN_HOST).expect("connect failed");
    let url = format!("ws://{AUTOBAHN_HOST}{path}");
    nexus_web::ws::ClientBuilder::new()
        .buffer_capacity(16 * 1024 * 1024 + 4096) // 16MB + header room
        .max_frame_size(16 * 1024 * 1024)
        .max_message_size(16 * 1024 * 1024)
        .write_buffer_capacity(16 * 1024 * 1024 + 4096) // match read capacity for echo
        .connect_with(tcp, &url)
        .expect("handshake failed")
}

#[test]
#[ignore = "requires autobahn fuzzingserver via podman"]
fn autobahn_conformance() {
    let case_count = get_case_count();
    println!("Autobahn: {case_count} test cases");

    for case in 1..=case_count {
        print!("  Case {case}/{case_count}...");
        run_case(case);
        println!(" ok");
    }

    update_reports();
    println!("Autobahn: reports generated. Check target/autobahn-reports/");
}

fn get_case_count() -> u32 {
    let mut ws = make_ws("/getCaseCount");
    match ws.recv().expect("read failed").expect("no message") {
        Message::Text(s) => s.parse().expect("invalid case count"),
        other => panic!("expected Text, got {other:?}"),
    }
}

fn run_case(case: u32) {
    let path = format!("/runCase?case={case}&agent={AGENT}");
    let mut ws = make_ws(&path);

    loop {
        let msg = match ws.recv() {
            Ok(Some(msg)) => msg.into_owned(),
            Err(Error::Protocol(ProtocolError::InvalidUtf8)) => {
                let _ = ws.close(CloseCode::InvalidPayload, "invalid UTF-8");
                break;
            }
            Err(Error::Protocol(_)) => {
                let _ = ws.close(CloseCode::Protocol, "protocol error");
                break;
            }
            Ok(None) | Err(_) => break,
        };

        match msg {
            OwnedMessage::Text(s) => {
                if ws.send_text(&s).is_err() {
                    break;
                }
            }
            OwnedMessage::Binary(b) => {
                if ws.send_binary(&b).is_err() {
                    break;
                }
            }
            OwnedMessage::Ping(p) => {
                if ws.send_pong(&p).is_err() {
                    break;
                }
            }
            OwnedMessage::Close(_) => {
                let _ = ws.close(CloseCode::Normal, "");
                break;
            }
            OwnedMessage::Pong(_) => {}
        }
    }
}

fn update_reports() {
    let path = format!("/updateReports?agent={AGENT}");
    let mut ws = make_ws(&path);
    let _ = ws.recv();
}

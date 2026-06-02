//! Fuzz ResponseReader with arbitrary bytes.
//!
//! Proves: no panics, no UB, no infinite loops on any input.
//! The parser must gracefully handle any byte sequence a malicious
//! server could send.

#![no_main]
use libfuzzer_sys::fuzz_target;

use nexus_web::http::ResponseReader;

fuzz_target!(|data: &[u8]| {
    // Test 1: Feed all bytes at once
    {
        let mut reader = ResponseReader::new(8192).max_body_size(4096);
        // Ignore errors — we're testing for panics/UB, not correctness.
        let _ = reader.read(data);
        let _ = reader.next();
        // Access cached values
        let _ = reader.status();
        let _ = reader.content_length();
        let _ = reader.is_chunked();
        let _ = reader.body_remaining();
        let _ = reader.remainder();
        let _ = reader.header("Content-Length");
        let _ = reader.header("Transfer-Encoding");
        let _ = reader.header_count();
    }

    // Test 2: Feed bytes one at a time (different parse boundaries)
    if data.len() < 2048 {
        let mut reader = ResponseReader::new(8192);
        for chunk in data.chunks(1) {
            let _ = reader.read(chunk);
            if reader.next().ok().flatten().is_some() {
                // Headers parsed — check accessors
                let _ = reader.status();
                let _ = reader.content_length();
                let _ = reader.is_chunked();
                let _ = reader.body_remaining();
                let _ = reader.header("Host");
                break;
            }
        }
    }

    // Test 3: Feed in random-sized chunks
    if data.len() >= 2 && data.len() < 4096 {
        let mut reader = ResponseReader::new(8192);
        let chunk_size = (data[0] as usize % 64) + 1;
        for chunk in data[1..].chunks(chunk_size) {
            let _ = reader.read(chunk);
            if reader.next().ok().flatten().is_some() {
                break;
            }
        }
        // Try consume + reparse (keep-alive simulation)
        reader.consume_response();
        let _ = reader.next();
    }
});

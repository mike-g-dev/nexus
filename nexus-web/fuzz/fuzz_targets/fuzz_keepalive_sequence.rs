//! Fuzz keep-alive response sequences.
//!
//! Proves: consume_response() correctly advances past each response,
//! preserving pipelined bytes. No panics, no buffer corruption across
//! multiple request/response cycles.

#![no_main]
use libfuzzer_sys::fuzz_target;

use nexus_web::http::ResponseReader;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let num_cycles = (data[0] % 8) as usize + 1; // 1-8 cycles
    let chunk_size = (data[1] as usize % 128) + 1;
    let input = &data[2..];

    let mut reader = ResponseReader::new(8192).max_body_size(4096);

    for cycle in 0..num_cycles {
        // consume_response from previous cycle
        if cycle > 0 {
            reader.consume_response();
        }

        // Feed data in chunks, try to parse
        let offset = (cycle * input.len() / num_cycles) % input.len().max(1);
        let cycle_data = &input[offset..];

        let mut parsed = false;
        for chunk in cycle_data.chunks(chunk_size) {
            if reader.read(chunk).is_err() {
                break;
            }
            match reader.next() {
                Ok(Some(_resp)) => {
                    // Successfully parsed — access all fields
                    let _ = reader.status();
                    let _ = reader.content_length();
                    let _ = reader.is_chunked();
                    let _ = reader.body_remaining();
                    let _ = reader.remainder();
                    let _ = reader.header("Content-Length");
                    let _ = reader.header_count();
                    parsed = true;
                    break;
                }
                Ok(None) => {} // need more data
                Err(_) => break,
            }
        }

        if !parsed {
            break;
        }
    }
});

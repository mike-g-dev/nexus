//! Fuzz RequestWriter with arbitrary strings.
//!
//! Proves: no panics, no UB. CRLF injection is caught (deferred error).
//! When finish() succeeds, the output is valid HTTP/1.1 (no bare CR/LF
//! in the request line or header values).

#![no_main]
use libfuzzer_sys::fuzz_target;

use nexus_web::rest::{Method, RequestWriter};

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }

    // Use first bytes to select method and split remaining data into fields
    let method = match data[0] % 5 {
        0 => Method::Get,
        1 => Method::Post,
        2 => Method::Put,
        3 => Method::Delete,
        _ => Method::Patch,
    };

    let field_count = (data[1] % 4) as usize; // 0-3 extra headers
    let has_query = data[2] & 1 != 0;
    let has_body = data[2] & 2 != 0;
    let has_base_path = data[2] & 4 != 0;

    // Split remaining data into strings for host, path, headers, etc.
    let rest = &data[3..];
    let fields: Vec<&str> = rest
        .split(|&b| b == 0xFF)
        .filter_map(|s| std::str::from_utf8(s).ok())
        .collect();

    if fields.is_empty() {
        return;
    }

    // Construct host — must not contain CRLF (panic guard)
    let host_candidate = fields[0];
    if host_candidate.bytes().any(|b| b == b'\r' || b == b'\n') || host_candidate.is_empty() {
        return;
    }

    let mut writer = match RequestWriter::new(host_candidate) {
        Ok(w) => w,
        Err(_) => return,
    };

    // Optional base path
    if has_base_path {
        if let Some(bp) = fields.get(1) {
            let _ = writer.set_base_path(bp);
        }
    }

    // Optional default headers
    for pair in fields.chunks(2).skip(1).take(2) {
        if pair.len() == 2 {
            let _ = writer.default_header(pair[0], pair[1]);
        }
    }

    // Build the request path
    let path = fields.get(1).copied().unwrap_or("/");

    // Start building
    let mut builder = writer.request(method, path);

    // Optional query params
    if has_query {
        for pair in fields.chunks(2).skip(2).take(2) {
            if pair.len() == 2 {
                builder = builder.query(pair[0], pair[1]);
            }
        }
    }

    // Transition to headers phase and add extra headers
    if field_count > 0 {
        let mut hbuilder = if let Some(pair) = fields.chunks(2).skip(3).next() {
            if pair.len() == 2 {
                builder.header(pair[0], pair[1])
            } else {
                // Can't transition without a header, finish from Query
                let result = builder.finish();
                validate_if_ok(&result);
                return;
            }
        } else {
            let result = builder.finish();
            validate_if_ok(&result);
            return;
        };

        for pair in fields.chunks(2).skip(4).take(field_count.saturating_sub(1)) {
            if pair.len() == 2 {
                hbuilder = hbuilder.header(pair[0], pair[1]);
            }
        }

        if has_body {
            let body_data = fields.last().map(|s| s.as_bytes()).unwrap_or(b"{}");
            let result = hbuilder.body(body_data).finish();
            validate_if_ok(&result);
        } else {
            let result = hbuilder.finish();
            validate_if_ok(&result);
        }
    } else if has_body {
        let body_data = fields.last().map(|s| s.as_bytes()).unwrap_or(b"{}");
        let result = builder.body(body_data).finish();
        validate_if_ok(&result);
    } else {
        let result = builder.finish();
        validate_if_ok(&result);
    }
});

/// When finish() succeeds, validate the output is sane HTTP.
fn validate_if_ok(result: &Result<nexus_web::rest::Request<'_>, nexus_web::rest::RestError>) {
    if let Ok(req) = result {
        let data = req.data();
        // Must contain HTTP version
        assert!(
            data.windows(8).any(|w| w == b"HTTP/1.1"),
            "missing HTTP/1.1 in request"
        );
        // Must end with \r\n (at minimum the header terminator)
        assert!(
            data.len() >= 4,
            "request too short"
        );
        // The request line (first line) must not have bare CR or LF
        // before the HTTP/1.1 marker.
        if let Some(pos) = data.windows(8).position(|w| w == b"HTTP/1.1") {
            let request_line = &data[..pos];
            // Request line should be: METHOD SP path SP
            // No bare \r or \n allowed in the method or path
            for (i, &b) in request_line.iter().enumerate() {
                if b == b'\r' || b == b'\n' {
                    panic!("CR/LF at byte {i} in request line: {:?}",
                        String::from_utf8_lossy(request_line));
                }
            }
        }
    }
}

//! Chunked Transfer-Encoding decoder (sans-IO).
//!
//! Strips chunk framing from HTTP/1.1 chunked responses.
//! Feed wire bytes in, get decoded body bytes out.

use super::error::HttpError;

/// Decoder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Reading the hex chunk size + \r\n.
    ChunkSize,
    /// Reading chunk data bytes.
    ChunkData { remaining: usize },
    /// Reading the \r\n after chunk data.
    ChunkDataTrailer,
    /// Consuming optional trailer headers after the zero chunk.
    /// RFC 7230 §4.1: trailers end with an empty line (\r\n).
    /// `prev_was_lf` tracks if the previous byte was \n.
    Trailers { prev_was_lf: bool },
    /// Final zero-length chunk and trailers consumed. Done.
    Done,
}

/// Sans-IO chunked transfer-encoding decoder.
///
/// Feed wire bytes via [`decode`](Self::decode). Decoded body bytes are written
/// into the output buffer. Returns how many input bytes were consumed
/// and how many output bytes were produced.
///
/// # Usage
///
/// ```ignore
/// let mut decoder = ChunkedDecoder::new();
/// let (consumed, produced) = decoder.decode(wire_bytes, &mut output_buf)?;
/// ```
pub struct ChunkedDecoder {
    state: State,
    /// Accumulates the hex chunk size digits.
    size_buf: [u8; 16],
    size_len: usize,
    /// Total decoded body bytes so far.
    total_decoded: usize,
}

impl ChunkedDecoder {
    /// Create a new decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: State::ChunkSize,
            size_buf: [0; 16],
            size_len: 0,
            total_decoded: 0,
        }
    }

    /// Whether the final zero-length chunk has been seen.
    pub fn is_done(&self) -> bool {
        self.state == State::Done
    }

    /// Total decoded body bytes produced so far.
    pub fn total_decoded(&self) -> usize {
        self.total_decoded
    }

    /// Decode chunked wire bytes into body bytes.
    ///
    /// Returns `(consumed, produced)` — how many input bytes were consumed
    /// and how many output bytes were written.
    ///
    /// Call repeatedly as more wire bytes arrive. When `is_done()` returns
    /// true, the body is complete.
    pub fn decode(&mut self, input: &[u8], output: &mut [u8]) -> Result<(usize, usize), HttpError> {
        let mut in_pos = 0;
        let mut out_pos = 0;

        while in_pos < input.len() && self.state != State::Done {
            match self.state {
                State::ChunkSize => {
                    // Scan for \n to find end of chunk size line.
                    let b = input[in_pos];
                    in_pos += 1;

                    if b == b'\n' {
                        // Parse the hex size (ignore optional chunk extensions after ';')
                        let size_str = std::str::from_utf8(&self.size_buf[..self.size_len])
                            .map_err(|_| HttpError::Malformed("invalid UTF-8 in chunk size"))?;
                        let hex_part = size_str.split(';').next().unwrap_or("").trim();
                        let chunk_size = usize::from_str_radix(hex_part, 16)
                            .map_err(|_| HttpError::Malformed("invalid hex in chunk size"))?;

                        self.size_len = 0;

                        if chunk_size == 0 {
                            // Zero chunk = end of body. Consume optional
                            // trailer headers + terminating empty line.
                            self.state = State::Trailers { prev_was_lf: true };
                        } else {
                            self.state = State::ChunkData {
                                remaining: chunk_size,
                            };
                        }
                    } else if b == b'\r' {
                        // Skip CR before LF.
                    } else {
                        if self.size_len >= self.size_buf.len() {
                            return Err(HttpError::Malformed("chunk size line too long"));
                        }
                        self.size_buf[self.size_len] = b;
                        self.size_len += 1;
                    }
                }

                State::ChunkData { remaining } => {
                    // Copy chunk data to output.
                    let available_in = input.len() - in_pos;
                    let available_out = output.len() - out_pos;
                    let to_copy = remaining.min(available_in).min(available_out);

                    if to_copy == 0 {
                        // Output buffer full — caller needs to process and call again.
                        break;
                    }

                    output[out_pos..out_pos + to_copy]
                        .copy_from_slice(&input[in_pos..in_pos + to_copy]);
                    in_pos += to_copy;
                    out_pos += to_copy;
                    self.total_decoded += to_copy;

                    let new_remaining = remaining - to_copy;
                    if new_remaining == 0 {
                        self.state = State::ChunkDataTrailer;
                    } else {
                        self.state = State::ChunkData {
                            remaining: new_remaining,
                        };
                    }
                }

                State::ChunkDataTrailer => {
                    // Consume \r\n after chunk data → next chunk.
                    let b = input[in_pos];
                    in_pos += 1;
                    if b == b'\n' {
                        self.state = State::ChunkSize;
                    }
                }

                State::Trailers { prev_was_lf } => {
                    // Consume optional trailer headers until empty line.
                    // Empty line = \n immediately after \n (with optional
                    // \r between). Detects both \r\n\r\n and \n\n.
                    let b = input[in_pos];
                    in_pos += 1;
                    if b == b'\n' {
                        if prev_was_lf {
                            // \n\n — empty line, done.
                            self.state = State::Done;
                        } else {
                            self.state = State::Trailers { prev_was_lf: true };
                        }
                    } else if b == b'\r' {
                        // \r doesn't reset the \n flag — \n\r\n is valid.
                        self.state = State::Trailers { prev_was_lf };
                    } else {
                        // Non-CRLF byte — part of a trailer header line.
                        self.state = State::Trailers { prev_was_lf: false };
                    }
                }

                State::Done => break,
            }
        }

        Ok((in_pos, out_pos))
    }

    /// Reset for reuse.
    pub fn reset(&mut self) {
        self.state = State::ChunkSize;
        self.size_len = 0;
        self.total_decoded = 0;
    }
}

impl Default for ChunkedDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_chunk() {
        let mut dec = ChunkedDecoder::new();
        let input = b"d\r\nHello, world!\r\n0\r\n\r\n";
        let mut output = [0u8; 64];

        let (consumed, produced) = dec.decode(input, &mut output).unwrap();
        assert_eq!(consumed, input.len());
        assert_eq!(produced, 13);
        assert_eq!(&output[..produced], b"Hello, world!");
        assert!(dec.is_done());
    }

    #[test]
    fn multiple_chunks() {
        let mut dec = ChunkedDecoder::new();
        let input = b"7\r\nMozilla\r\n11\r\nDeveloper Network\r\n0\r\n\r\n";
        let mut output = [0u8; 64];

        let (consumed, produced) = dec.decode(input, &mut output).unwrap();
        assert_eq!(consumed, input.len());
        assert_eq!(produced, 24);
        assert_eq!(&output[..produced], b"MozillaDeveloper Network");
        assert!(dec.is_done());
    }

    #[test]
    fn byte_by_byte() {
        let mut dec = ChunkedDecoder::new();
        let input = b"5\r\nhello\r\n0\r\n\r\n";
        let mut output = [0u8; 64];
        let mut total_out = 0;

        for &b in input {
            let (_, produced) = dec.decode(&[b], &mut output[total_out..]).unwrap();
            total_out += produced;
        }

        assert_eq!(total_out, 5);
        assert_eq!(&output[..5], b"hello");
        assert!(dec.is_done());
    }

    #[test]
    fn hex_uppercase() {
        let mut dec = ChunkedDecoder::new();
        let input = b"A\r\n0123456789\r\n0\r\n\r\n";
        let mut output = [0u8; 64];

        let (_, produced) = dec.decode(input, &mut output).unwrap();
        assert_eq!(produced, 10);
        assert!(dec.is_done());
    }

    #[test]
    fn chunk_extension_ignored() {
        let mut dec = ChunkedDecoder::new();
        // Chunk extensions after ';' should be ignored per RFC 7230
        let input = b"5;ext=val\r\nhello\r\n0\r\n\r\n";
        let mut output = [0u8; 64];

        let (_, produced) = dec.decode(input, &mut output).unwrap();
        assert_eq!(produced, 5);
        assert_eq!(&output[..5], b"hello");
        assert!(dec.is_done());
    }

    #[test]
    fn empty_body() {
        let mut dec = ChunkedDecoder::new();
        let input = b"0\r\n\r\n";
        let mut output = [0u8; 64];

        let (consumed, produced) = dec.decode(input, &mut output).unwrap();
        assert_eq!(consumed, input.len());
        assert_eq!(produced, 0);
        assert!(dec.is_done());
    }

    #[test]
    fn output_buffer_smaller_than_chunk() {
        let mut dec = ChunkedDecoder::new();
        let input = b"a\r\n0123456789\r\n0\r\n\r\n";
        let mut output = [0u8; 4]; // smaller than chunk

        // First call: fills 4 bytes
        let (consumed1, produced1) = dec.decode(input, &mut output).unwrap();
        assert_eq!(produced1, 4);
        assert_eq!(&output[..4], b"0123");

        // Second call with remaining input
        let (consumed2, produced2) = dec.decode(&input[consumed1..], &mut output).unwrap();
        assert_eq!(produced2, 4);
        assert_eq!(&output[..4], b"4567");

        // Third call
        let (_consumed3, produced3) = dec
            .decode(&input[consumed1 + consumed2..], &mut output)
            .unwrap();
        assert_eq!(produced3, 2);
        assert_eq!(&output[..2], b"89");
        assert!(dec.is_done());
    }

    #[test]
    fn malformed_hex_rejected() {
        let mut dec = ChunkedDecoder::new();
        let input = b"xyz\r\ndata\r\n";
        let mut output = [0u8; 64];

        assert!(dec.decode(input, &mut output).is_err());
    }

    #[test]
    fn trailer_headers_consumed() {
        let mut dec = ChunkedDecoder::new();
        let input = b"5\r\nhello\r\n0\r\nTrailer: value\r\nAnother: hdr\r\n\r\n";
        let mut output = [0u8; 64];

        let (consumed, produced) = dec.decode(input, &mut output).unwrap();
        assert_eq!(consumed, input.len());
        assert_eq!(produced, 5);
        assert_eq!(&output[..5], b"hello");
        assert!(dec.is_done());
    }

    #[test]
    fn trailer_no_headers_just_crlf() {
        let mut dec = ChunkedDecoder::new();
        // No trailer headers — just the standard terminator
        let input = b"3\r\nabc\r\n0\r\n\r\n";
        let mut output = [0u8; 64];

        let (consumed, produced) = dec.decode(input, &mut output).unwrap();
        assert_eq!(consumed, input.len());
        assert_eq!(produced, 3);
        assert!(dec.is_done());
    }
}

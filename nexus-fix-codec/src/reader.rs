//! FIX field reader with SIMD-accelerated SOH scanning and checksum.
//!
//! Combines SOH delimiter scanning, tag=value reading, and byte-sum
//! checksum accumulation in a single pass. Each SIMD chunk load
//! performs both `cmpeq` (SOH detection) and `PSADBW` (checksum
//! accumulation), avoiding a second pass over the data.
//!
//! Dispatch cascade (widest first, remainder flows down):
//!
//! - AVX-512: 64 bytes/iter (`target_feature = "avx512bw"`)
//! - AVX2: 32 bytes/iter (`target_feature = "avx2"`)
//! - SSE2: 16 bytes/iter (baseline x86_64)
//! - SWAR: 8 bytes/iter (all platforms)
//! - Scalar: byte-by-byte tail

use crate::FieldSpan;
use crate::error::ChecksumError;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const HI: u64 = 0x8080_8080_8080_8080;
const LO: u64 = 0x0101_0101_0101_0101;

/// A parsed FIX field: tag number and value span.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RawField {
    pub tag: u32,
    pub value: FieldSpan,
}

/// FIX field reader with fused checksum accumulation.
///
/// Iterates over `tag=value\x01` fields, yielding [`RawField`] pairs.
/// Accumulates a running byte-sum checksum via SIMD PSADBW alongside
/// the SOH scan — no second pass needed.
///
/// The checksum is only meaningful after a complete scan of the
/// message body. Partial iteration produces a partial sum.
///
/// # Example
///
/// ```
/// use nexus_fix_codec::reader::FieldReader;
///
/// let msg = b"8=FIX.4.4\x0135=D\x0149=SENDER\x01";
/// let mut parser = FieldReader::new(msg, 0);
///
/// while let Some(field) = parser.next_field() {
///     // field.tag, field.value
/// }
///
/// let _checksum = parser.checksum();
/// ```
pub struct FieldReader<'a> {
    buf: &'a [u8],
    scan_pos: usize,
    field_start: usize,
    checksum: u32,
    soh_mask: u64,
    mask_base: usize,
}

impl<'a> FieldReader<'a> {
    #[inline]
    pub fn new(buf: &'a [u8], start: usize) -> Self {
        Self {
            buf,
            scan_pos: start,
            field_start: start,
            checksum: 0,
            soh_mask: 0,
            mask_base: 0,
        }
    }

    /// FIX checksum: byte sum mod 256 of all scanned bytes, excluding
    /// the checksum field itself (`10=XXX\x01`).
    ///
    /// Only meaningful after a complete scan of the message body.
    /// Tag 10 bytes are automatically excluded when encountered
    /// during parsing.
    #[inline]
    pub fn checksum(&self) -> u8 {
        (self.checksum & 0xFF) as u8
    }

    /// Where the next field would start (after the last SOH + 1).
    #[inline]
    pub fn pos(&self) -> usize {
        self.field_start
    }

    /// Parse the next `tag=value\x01` field.
    ///
    /// Returns `None` at end of buffer or if the remaining bytes
    /// contain no valid `tag=value\x01` structure.
    #[inline]
    pub fn next_field(&mut self) -> Option<RawField> {
        let field_start = self.field_start;
        let soh_pos = self.find_next_soh()?;
        self.field_start = soh_pos + 1;

        let field_bytes = self.buf.get(field_start..soh_pos)?;
        let (tag, tag_len) = parse_tag(field_bytes);

        if tag_len == 0 || tag_len >= field_bytes.len() || field_bytes[tag_len] != b'=' {
            return None;
        }

        if tag == 10 {
            for &b in &self.buf[field_start..=soh_pos] {
                self.checksum = self.checksum.wrapping_sub(b as u32);
            }
        }

        let value_start = field_start + tag_len + 1;
        let value_len = soh_pos - value_start;

        Some(RawField {
            tag,
            value: FieldSpan::new(value_start as u32, value_len as u32),
        })
    }
}

impl Iterator for FieldReader<'_> {
    type Item = RawField;

    #[inline]
    fn next(&mut self) -> Option<RawField> {
        self.next_field()
    }
}

// =============================================================================
// Tag number parsing
// =============================================================================

/// Parse a FIX tag number from ASCII digits.
///
/// Reads sequential digits from the start of `buf` until a non-digit
/// byte or end of buffer. Returns `(tag_number, digits_consumed)`.
#[inline]
pub fn parse_tag(buf: &[u8]) -> (u32, usize) {
    let mut tag = 0u32;
    let mut i = 0;
    while i < buf.len() && buf[i] >= b'0' && buf[i] <= b'9' {
        tag = tag * 10 + (buf[i] - b'0') as u32;
        i += 1;
    }
    (tag, i)
}

// =============================================================================
// Standalone helpers
// =============================================================================

/// Compute the FIX checksum: byte sum mod 256.
///
/// Simple scalar sum — useful for encoder checksum computation
/// and anywhere a standalone checksum is needed without a full parse.
///
/// Uses `wrapping_add` because only the low 8 bits matter (the
/// result is mod 256), so intermediate overflow is harmless and
/// the function accepts any input length without debug panics.
#[inline]
pub fn checksum(data: &[u8]) -> u8 {
    data.iter().fold(0u32, |acc, &b| acc.wrapping_add(b as u32)) as u8
}

/// Find the first field with a specific tag number.
///
/// Scans from `start` using [`FieldReader`], returning the value
/// [`FieldSpan`] of the first field with matching tag number.
#[inline]
pub fn find_tag(buf: &[u8], start: usize, tag: u32) -> Option<FieldSpan> {
    FieldReader::new(buf, start)
        .find(|f| f.tag == tag)
        .map(|f| f.value)
}

/// Validate a FIX message checksum against tag 10.
///
/// Parses all fields via [`FieldReader`] (fused PSADBW checksum),
/// then compares the computed byte sum against the declared tag 10
/// value.
///
/// Returns `Ok(())` if valid, `Err(ChecksumError)` on mismatch.
/// Returns `Ok(())` if tag 10 is absent (nothing to validate
/// against — structural validation is the decoder's job).
pub fn validate_checksum(msg: &[u8]) -> Result<(), ChecksumError> {
    let mut parser = FieldReader::new(msg, 0);
    let mut expected_span = None;

    while let Some(field) = parser.next_field() {
        if field.tag == 10 {
            expected_span = Some(field.value);
        }
    }

    let Some(span) = expected_span else {
        return Ok(());
    };

    let expected = parse_checksum_bytes(span.slice(msg));
    let computed = parser.checksum();
    if expected == computed {
        Ok(())
    } else {
        Err(ChecksumError { expected, computed })
    }
}

fn parse_checksum_bytes(bytes: &[u8]) -> u8 {
    let mut val = 0u32;
    for &b in bytes {
        let digit = b.wrapping_sub(b'0');
        if digit > 9 {
            return 0;
        }
        val = val * 10 + digit as u32;
    }
    (val & 0xFF) as u8
}

// =============================================================================
// SOH scanning with fused checksum accumulation
// =============================================================================

impl FieldReader<'_> {
    #[inline(always)]
    fn emit_soh_mask(&mut self, mask: u64, chunk_offset: usize, chunk_size: usize) -> usize {
        self.mask_base = self.scan_pos + chunk_offset;
        self.scan_pos = self.scan_pos + chunk_offset + chunk_size;
        self.soh_mask = mask;
        let bit = self.soh_mask.trailing_zeros() as usize;
        self.soh_mask &= self.soh_mask - 1;
        self.mask_base + bit
    }

    fn find_next_soh(&mut self) -> Option<usize> {
        if self.soh_mask != 0 {
            let bit = self.soh_mask.trailing_zeros() as usize;
            self.soh_mask &= self.soh_mask - 1;
            return Some(self.mask_base + bit);
        }

        let bytes = self.buf.get(self.scan_pos..)?;
        if bytes.is_empty() {
            return None;
        }

        let mut i = 0;

        // ---- x86_64 SIMD tiers (widest first) ----
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: target_feature cfgs guarantee SIMD instruction availability.
            // All pointer arithmetic is bounds-checked by the while conditions.
            unsafe {
                #[cfg(target_feature = "avx512bw")]
                {
                    let soh = _mm512_set1_epi8(0x01_i8);
                    let zero = _mm512_setzero_si512();
                    while i + 64 <= bytes.len() {
                        let chunk = _mm512_loadu_si512(bytes.as_ptr().add(i).cast());
                        let sad = _mm512_sad_epu8(chunk, zero);
                        self.checksum += Self::hsum_sad_512(sad);
                        let m = _mm512_cmpeq_epi8_mask(chunk, soh);
                        if m != 0 {
                            return Some(self.emit_soh_mask(m, i, 64));
                        }
                        i += 64;
                    }
                }

                #[cfg(target_feature = "avx2")]
                {
                    let soh = _mm256_set1_epi8(0x01_i8);
                    let zero = _mm256_setzero_si256();
                    while i + 32 <= bytes.len() {
                        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i).cast());
                        let sad = _mm256_sad_epu8(chunk, zero);
                        self.checksum += Self::hsum_sad_256(sad);
                        let cmp = _mm256_cmpeq_epi8(chunk, soh);
                        let m = _mm256_movemask_epi8(cmp) as u32 as u64;
                        if m != 0 {
                            return Some(self.emit_soh_mask(m, i, 32));
                        }
                        i += 32;
                    }
                }

                // SSE2 — baseline on x86_64
                {
                    let soh = _mm_set1_epi8(0x01_i8);
                    let zero = _mm_setzero_si128();
                    while i + 16 <= bytes.len() {
                        let chunk = _mm_loadu_si128(bytes.as_ptr().add(i).cast());
                        let sad = _mm_sad_epu8(chunk, zero);
                        self.checksum += Self::hsum_sad_128(sad);
                        let cmp = _mm_cmpeq_epi8(chunk, soh);
                        let m = _mm_movemask_epi8(cmp) as u32 as u64;
                        if m != 0 {
                            return Some(self.emit_soh_mask(m, i, 16));
                        }
                        i += 16;
                    }
                }
            }
        }

        // ---- SWAR (8 bytes at a time) ----
        {
            let splat = LO;
            while i + 8 <= bytes.len() {
                // SAFETY: bounds checked by the while condition
                let chunk: [u8; 8] =
                    unsafe { bytes.as_ptr().add(i).cast::<[u8; 8]>().read_unaligned() };
                for &b in &chunk {
                    self.checksum += b as u32;
                }
                let word = u64::from_ne_bytes(chunk);
                let xored = word ^ splat;
                let swar_mask = xored.wrapping_sub(LO) & !xored & HI;
                if swar_mask != 0 {
                    let packed = swar_to_byte_mask(swar_mask);
                    return Some(self.emit_soh_mask(packed, i, 8));
                }
                i += 8;
            }
        }

        // ---- Scalar tail (< 8 bytes) ----
        while i < bytes.len() {
            self.checksum += bytes[i] as u32;
            if bytes[i] == 0x01 {
                let result = self.scan_pos + i;
                self.scan_pos = result + 1;
                return Some(result);
            }
            i += 1;
        }

        self.scan_pos = self.buf.len();
        None
    }
}

// =============================================================================
// SIMD horizontal sum helpers
// =============================================================================

#[cfg(target_arch = "x86_64")]
impl FieldReader<'_> {
    #[inline(always)]
    fn hsum_sad_128(v: __m128i) -> u32 {
        // SAFETY: SSE2 is baseline on x86_64. Pure SIMD arithmetic.
        unsafe {
            let hi = _mm_srli_si128(v, 8);
            let sum = _mm_add_epi64(v, hi);
            _mm_cvtsi128_si32(sum) as u32
        }
    }

    #[cfg(target_feature = "avx2")]
    #[inline(always)]
    fn hsum_sad_256(v: __m256i) -> u32 {
        // SAFETY: AVX2 guaranteed by cfg. Pure SIMD arithmetic.
        unsafe {
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256(v, 1);
            let sum = _mm_add_epi64(lo, hi);
            Self::hsum_sad_128(sum)
        }
    }

    #[cfg(target_feature = "avx512bw")]
    #[inline(always)]
    fn hsum_sad_512(v: __m512i) -> u32 {
        // SAFETY: AVX-512BW guaranteed by cfg. Pure SIMD arithmetic.
        unsafe {
            let lo = _mm512_castsi512_si256(v);
            let hi = _mm512_extracti64x4_epi64(v, 1);
            let sum = _mm256_add_epi64(lo, hi);
            Self::hsum_sad_256(sum)
        }
    }
}

// =============================================================================
// SWAR mask conversion
// =============================================================================

/// Convert SWAR match mask to bit-per-byte mask.
///
/// SWAR sets the high bit of each matching byte (bits 7, 15, 23, ...).
/// This converts to one-bit-per-byte (bits 0, 1, 2, ...) for uniform
/// handling via `emit_soh_mask`.
///
/// A branchless multiply-shift gather (shift right 7, multiply by the
/// magic `0x0102_0408_1020_4080`, shift right 56) is correct and tempting
/// here, but measured slower on realistic decode and was rejected: this
/// result feeds `emit_soh_mask` on the critical path, so the multiply's
/// full latency lands on the returned position. For the common sparse case
/// (one SOH in the tail window) the loop runs a single cheap iteration and
/// wins by doing less work for the common input.
#[inline]
fn swar_to_byte_mask(swar_mask: u64) -> u64 {
    let mut packed = 0u64;
    let mut m = swar_mask;
    while m != 0 {
        let bit = m.trailing_zeros();
        packed |= 1u64 << (bit / 8);
        m &= m - 1;
    }
    packed
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_buffer() {
        let mut parser = FieldReader::new(b"", 0);
        assert!(parser.next_field().is_none());
    }

    #[test]
    fn single_field() {
        let msg = b"35=D\x01";
        let mut parser = FieldReader::new(msg, 0);
        let field = parser.next_field().unwrap();
        assert_eq!(field.tag, 35);
        assert_eq!(field.value.slice(msg), b"D");
        assert!(parser.next_field().is_none());
    }

    #[test]
    fn multiple_fields() {
        let msg = b"8=FIX.4.4\x0135=D\x0149=SENDER\x01";
        let mut parser = FieldReader::new(msg, 0);

        let f1 = parser.next_field().unwrap();
        assert_eq!(f1.tag, 8);
        assert_eq!(f1.value.slice(msg), b"FIX.4.4");

        let f2 = parser.next_field().unwrap();
        assert_eq!(f2.tag, 35);
        assert_eq!(f2.value.slice(msg), b"D");

        let f3 = parser.next_field().unwrap();
        assert_eq!(f3.tag, 49);
        assert_eq!(f3.value.slice(msg), b"SENDER");

        assert!(parser.next_field().is_none());
    }

    #[test]
    fn from_offset() {
        let msg = b"8=FIX.4.4\x0135=D\x01";
        let mut parser = FieldReader::new(msg, 10);
        let field = parser.next_field().unwrap();
        assert_eq!(field.tag, 35);
        assert_eq!(field.value.slice(msg), b"D");
        assert!(parser.next_field().is_none());
    }

    #[test]
    fn tag_numbers() {
        let msg = b"8=v\x0135=v\x01150=v\x015592=v\x01";
        let tags: Vec<u32> = FieldReader::new(msg, 0).map(|f| f.tag).collect();
        assert_eq!(tags, vec![8, 35, 150, 5592]);
    }

    #[test]
    fn value_spans() {
        let msg = b"44=50000.00\x0155=BTC-USD\x01";
        let mut parser = FieldReader::new(msg, 0);

        let f1 = parser.next_field().unwrap();
        assert_eq!(f1.value.slice(msg), b"50000.00");

        let f2 = parser.next_field().unwrap();
        assert_eq!(f2.value.slice(msg), b"BTC-USD");
    }

    #[test]
    fn checksum_matches_byte_sum() {
        let msg = b"8=FIX.4.4\x0135=D\x0149=SENDER\x01";
        let mut parser = FieldReader::new(msg, 0);
        while parser.next_field().is_some() {}

        let expected: u8 = msg.iter().map(|&b| b as u32).sum::<u32>() as u8;
        assert_eq!(parser.checksum(), expected);
    }

    #[test]
    fn checksum_from_offset() {
        let msg = b"8=FIX.4.4\x0135=D\x01";
        let mut parser = FieldReader::new(msg, 10);
        while parser.next_field().is_some() {}

        let expected: u8 = msg[10..].iter().map(|&b| b as u32).sum::<u32>() as u8;
        assert_eq!(parser.checksum(), expected);
    }

    #[test]
    fn checksum_various_lengths() {
        for len in 1..=200 {
            let value = "X".repeat(len);
            let msg = format!("1={}\x01", value);
            let msg = msg.as_bytes();

            let expected: u8 = msg.iter().map(|&b| b as u32).sum::<u32>() as u8;
            let mut parser = FieldReader::new(msg, 0);
            while parser.next_field().is_some() {}
            assert_eq!(parser.checksum(), expected, "len={}", len);
        }
    }

    #[test]
    fn malformed_no_equals() {
        let msg = b"garbage\x01";
        let mut parser = FieldReader::new(msg, 0);
        assert!(parser.next_field().is_none());
    }

    #[test]
    fn iterator_count() {
        let msg = b"8=FIX.4.4\x019=65\x0135=D\x0149=SENDER\x0156=TARGET\x01\
                     34=1\x0152=20260530-12:00:00\x0111=order1\x0155=BTC-USD\x01\
                     54=1\x0138=100\x0140=2\x0144=50000.00\x0110=123\x01";
        let count = FieldReader::new(msg, 0).count();
        assert_eq!(count, 14);
    }

    #[test]
    fn realistic_fix_message() {
        let msg = b"8=FIX.4.4\x019=120\x0135=D\x0149=SENDER\x0156=TARGET\x01\
                     34=42\x0152=20260530-12:00:00.000\x0111=order-001\x01\
                     55=BTC-USD\x0154=1\x0138=1.50000000\x0140=2\x01\
                     44=67500.00\x0159=0\x0110=178\x01";

        let mut parser = FieldReader::new(msg, 0);
        let mut fields = Vec::new();
        while let Some(field) = parser.next_field() {
            fields.push(field);
        }

        assert_eq!(fields.len(), 15);
        assert_eq!(fields[0].tag, 8);
        assert_eq!(fields[0].value.slice(msg), b"FIX.4.4");
        assert_eq!(fields[2].tag, 35);
        assert_eq!(fields[2].value.slice(msg), b"D");
        assert_eq!(fields[14].tag, 10);
        assert_eq!(fields[14].value.slice(msg), b"178");

        // Checksum excludes the tag 10 field
        let tag10_field = b"10=178\x01";
        let body_sum: u32 = msg.iter().map(|&b| b as u32).sum::<u32>()
            - tag10_field.iter().map(|&b| b as u32).sum::<u32>();
        assert_eq!(parser.checksum(), (body_sum & 0xFF) as u8);
    }

    #[test]
    fn checksum_excludes_tag_10() {
        let msg = b"35=D\x0149=SENDER\x0110=099\x01";
        let mut parser = FieldReader::new(msg, 0);
        while parser.next_field().is_some() {}

        let body = b"35=D\x0149=SENDER\x01";
        let expected: u8 = body.iter().map(|&b| b as u32).sum::<u32>() as u8;
        assert_eq!(parser.checksum(), expected);
    }

    #[test]
    fn long_value_exercises_simd() {
        let value = "X".repeat(300);
        let msg = format!("1={}\x01", value);
        let msg = msg.as_bytes();

        let mut parser = FieldReader::new(msg, 0);
        let field = parser.next_field().unwrap();
        assert_eq!(field.tag, 1);
        assert_eq!(field.value.len, 300);
        assert!(parser.next_field().is_none());

        let expected: u8 = msg.iter().map(|&b| b as u32).sum::<u32>() as u8;
        assert_eq!(parser.checksum(), expected);
    }

    #[test]
    fn many_short_fields() {
        let mut msg = Vec::new();
        for i in 1..=50 {
            msg.extend_from_slice(format!("{}=v\x01", i).as_bytes());
        }

        let mut parser = FieldReader::new(&msg, 0);
        let mut count = 0;
        while parser.next_field().is_some() {
            count += 1;
        }
        assert_eq!(count, 50);

        let tag10_field = b"10=v\x01";
        let expected: u8 = (msg.iter().map(|&b| b as u32).sum::<u32>()
            - tag10_field.iter().map(|&b| b as u32).sum::<u32>()) as u8;
        assert_eq!(parser.checksum(), expected);
    }

    #[test]
    fn dense_soh_in_swar_chunk() {
        // Multiple SOH bytes within a single 8-byte window
        let msg = b"1=A\x012=B\x013=C\x01";
        let fields: Vec<(u32, &[u8])> = FieldReader::new(msg, 0)
            .map(|f| (f.tag, f.value.slice(msg)))
            .collect();
        assert_eq!(
            fields,
            vec![(1, b"A" as &[u8]), (2, b"B" as &[u8]), (3, b"C" as &[u8])]
        );
    }

    #[test]
    fn pos_advances() {
        let msg = b"8=FIX.4.4\x0135=D\x01";
        let mut parser = FieldReader::new(msg, 0);
        assert_eq!(parser.pos(), 0);
        parser.next_field();
        assert_eq!(parser.pos(), 10);
        parser.next_field();
        assert_eq!(parser.pos(), 15);
    }

    #[test]
    fn parse_tag_standalone() {
        assert_eq!(parse_tag(b"8=FIX"), (8, 1));
        assert_eq!(parse_tag(b"35=D"), (35, 2));
        assert_eq!(parse_tag(b"150=2"), (150, 3));
        assert_eq!(parse_tag(b"5592=CUSTOM"), (5592, 4));
        assert_eq!(parse_tag(b"10000=X"), (10000, 5));
    }

    #[test]
    fn parse_tag_empty() {
        assert_eq!(parse_tag(b""), (0, 0));
        assert_eq!(parse_tag(b"=value"), (0, 0));
    }

    #[test]
    fn swar_mask_conversion() {
        // Single match at byte 0: bit 7 set
        assert_eq!(swar_to_byte_mask(0x80), 0x01);
        // Single match at byte 3: bit 31 set
        assert_eq!(swar_to_byte_mask(0x80_00_00_00), 0x08);
        // Matches at bytes 0 and 7
        assert_eq!(swar_to_byte_mask(0x80_00_00_00_00_00_00_80), 0x81);
    }

    #[test]
    fn swar_mask_conversion_exhaustive() {
        // For every subset of matched bytes, build the SWAR mask (high bit
        // per matched byte) and check `swar_to_byte_mask` against the
        // reference mapping: matched byte k -> output bit k.
        for pattern in 0u32..256 {
            let mut swar = 0u64;
            let mut expected = 0u64;
            for k in 0..8 {
                if pattern & (1 << k) != 0 {
                    swar |= 0x80u64 << (k * 8);
                    expected |= 1u64 << k;
                }
            }
            assert_eq!(
                swar_to_byte_mask(swar),
                expected,
                "pattern={:#010b}",
                pattern
            );
        }
    }

    // =========================================================================
    // Standalone checksum
    // =========================================================================

    #[test]
    fn checksum_standalone() {
        assert_eq!(checksum(b""), 0);
        assert_eq!(checksum(b"\x01"), 1);
        let msg = b"8=FIX.4.4\x0135=D\x01";
        let expected: u8 = msg.iter().map(|&b| b as u32).sum::<u32>() as u8;
        assert_eq!(checksum(msg), expected);
    }

    // =========================================================================
    // find_tag
    // =========================================================================

    #[test]
    fn find_tag_present() {
        let msg = b"8=FIX.4.4\x0135=D\x0149=SENDER\x01";
        let span = find_tag(msg, 0, 35).unwrap();
        assert_eq!(span.slice(msg), b"D");
    }

    #[test]
    fn find_tag_absent() {
        let msg = b"8=FIX.4.4\x0135=D\x01";
        assert!(find_tag(msg, 0, 99).is_none());
    }

    #[test]
    fn find_tag_with_offset() {
        let msg = b"8=FIX.4.4\x0135=D\x0149=SENDER\x01";
        assert!(find_tag(msg, 10, 8).is_none());
        let span = find_tag(msg, 10, 35).unwrap();
        assert_eq!(span.slice(msg), b"D");
    }

    #[test]
    fn find_tag_first_match() {
        let msg = b"58=first\x0158=second\x01";
        let span = find_tag(msg, 0, 58).unwrap();
        assert_eq!(span.slice(msg), b"first");
    }

    // =========================================================================
    // validate_checksum
    // =========================================================================

    #[test]
    fn validate_checksum_valid() {
        let body = b"8=FIX.4.4\x0135=D\x0149=SENDER\x01";
        let sum = checksum(body);
        let msg = format!("8=FIX.4.4\x0135=D\x0149=SENDER\x0110={:03}\x01", sum);
        assert!(validate_checksum(msg.as_bytes()).is_ok());
    }

    #[test]
    fn validate_checksum_invalid() {
        let msg = b"8=FIX.4.4\x0135=D\x0149=SENDER\x0110=000\x01";
        let result = validate_checksum(msg);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.expected, 0);
        assert_ne!(err.computed, 0);
    }

    #[test]
    fn validate_checksum_no_tag10() {
        let msg = b"8=FIX.4.4\x0135=D\x01";
        assert!(validate_checksum(msg).is_ok());
    }

    #[test]
    fn validate_checksum_realistic() {
        // Build a message, compute correct checksum, then validate
        let body = b"8=FIX.4.4\x019=65\x0135=D\x0149=SENDER\x0156=TARGET\x01";
        let sum = checksum(body);
        let mut msg = body.to_vec();
        msg.extend_from_slice(format!("10={:03}\x01", sum).as_bytes());
        assert!(validate_checksum(&msg).is_ok());
    }

    #[test]
    fn binary_value() {
        let msg = b"96=\x00\xFF\x80\x7F\x01";
        let mut parser = FieldReader::new(msg, 0);
        let field = parser.next_field().unwrap();
        assert_eq!(field.tag, 96);
        assert_eq!(field.value.slice(msg), b"\x00\xFF\x80\x7F");
        assert!(parser.next_field().is_none());

        let expected: u8 = msg.iter().map(|&b| b as u32).sum::<u32>() as u8;
        assert_eq!(parser.checksum(), expected);
    }

    #[test]
    fn checksum_excludes_tag_10_mid_message() {
        let msg = b"35=D\x0110=099\x0149=SENDER\x01";
        let mut parser = FieldReader::new(msg, 0);
        let fields: Vec<_> = parser.by_ref().collect();
        assert_eq!(fields.len(), 3);

        let body_sum: u32 = b"35=D\x0149=SENDER\x01".iter().map(|&b| b as u32).sum();
        assert_eq!(parser.checksum(), (body_sum & 0xFF) as u8);
    }

    #[test]
    fn validate_checksum_malformed_tag10() {
        let msg = b"35=D\x0110=XYZ\x01";
        let result = validate_checksum(msg);
        // parse_checksum_bytes on non-digits wraps around, producing a mismatch
        assert!(result.is_err());
    }

    // =========================================================================
    // Repeating group pattern (simulates codegen usage)
    // =========================================================================

    #[test]
    fn repeating_group_walk() {
        // MarketDataSnapshot with NoMDEntries (tag 268) = 3 entries.
        // Delimiter tag is 269 (MDEntryType). Entries have varying field counts.
        let msg = b"35=W\x01\
                     268=3\x01\
                     269=0\x01270=50000.00\x01271=1.5\x01\
                     269=1\x01270=49999.00\x01\
                     269=2\x01270=50001.00\x01271=0.8\x01272=XBTO\x01\
                     10=999\x01";

        let delimiter_tag = 269u32;
        let mut reader = FieldReader::new(msg, 0);

        // Walk to the count tag (268), simulating what the scanner does.
        let mut group_count = 0u16;
        let mut group_offset = 0u32;
        while let Some(field) = reader.next_field() {
            if field.tag == 268 {
                let count_bytes = field.value.slice(msg);
                group_count = core::str::from_utf8(count_bytes).unwrap().parse().unwrap();
                group_offset = reader.pos() as u32;
                break;
            }
        }

        let group = crate::GroupSpan::new(group_offset, group_count);
        assert_eq!(group.count, 3);

        // Now read the group: iterate from group.offset, using dictionary
        // knowledge of which tags belong to the group to detect boundaries.
        let group_tags: &[u32] = &[269, 270, 271, 272];
        let mut group_reader = FieldReader::new(msg, group.offset as usize);
        let mut entries: Vec<Vec<RawField>> = Vec::new();
        let mut current_entry: Vec<RawField> = Vec::new();

        while let Some(field) = group_reader.next_field() {
            if !group_tags.contains(&field.tag) {
                break;
            }
            if field.tag == delimiter_tag && !current_entry.is_empty() {
                entries.push(core::mem::take(&mut current_entry));
            }
            current_entry.push(field);
        }
        if !current_entry.is_empty() {
            entries.push(current_entry);
        }

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].len(), 3); // 269, 270, 271
        assert_eq!(entries[1].len(), 2); // 269, 270
        assert_eq!(entries[2].len(), 4); // 269, 270, 271, 272

        assert_eq!(entries[0][0].value.slice(msg), b"0");
        assert_eq!(entries[0][1].value.slice(msg), b"50000.00");
        assert_eq!(entries[1][0].value.slice(msg), b"1");
        assert_eq!(entries[2][3].value.slice(msg), b"XBTO");
    }

    #[test]
    fn repeating_group_single_entry() {
        let msg = b"35=W\x01268=1\x01269=0\x01270=50000\x0110=999\x01";

        let group_tags: &[u32] = &[269, 270];
        let mut reader = FieldReader::new(msg, 0);
        while let Some(f) = reader.next_field() {
            if f.tag == 268 {
                break;
            }
        }
        let group = crate::GroupSpan::new(reader.pos() as u32, 1);

        let mut group_reader = FieldReader::new(msg, group.offset as usize);
        let mut entries: Vec<Vec<RawField>> = Vec::new();
        let mut current: Vec<RawField> = Vec::new();
        while let Some(f) = group_reader.next_field() {
            if !group_tags.contains(&f.tag) {
                break;
            }
            if f.tag == 269 && !current.is_empty() {
                entries.push(core::mem::take(&mut current));
            }
            current.push(f);
        }
        if !current.is_empty() {
            entries.push(current);
        }

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].len(), 2);
        assert_eq!(entries[0][0].value.slice(msg), b"0");
        assert_eq!(entries[0][1].value.slice(msg), b"50000");
    }

    #[test]
    fn repeating_group_minimal_entries() {
        // Entries with only the delimiter tag (all optional fields absent).
        let msg = b"268=3\x01269=0\x01269=1\x01269=2\x0110=999\x01";

        let group_tags: &[u32] = &[269, 270, 271];
        let mut reader = FieldReader::new(msg, 0);
        reader.next_field(); // skip 268
        let group = crate::GroupSpan::new(reader.pos() as u32, 3);

        let mut group_reader = FieldReader::new(msg, group.offset as usize);
        let mut entries: Vec<Vec<RawField>> = Vec::new();
        let mut current: Vec<RawField> = Vec::new();
        while let Some(f) = group_reader.next_field() {
            if !group_tags.contains(&f.tag) {
                break;
            }
            if f.tag == 269 && !current.is_empty() {
                entries.push(core::mem::take(&mut current));
            }
            current.push(f);
        }
        if !current.is_empty() {
            entries.push(current);
        }

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].len(), 1);
        assert_eq!(entries[1].len(), 1);
        assert_eq!(entries[2].len(), 1);
    }

    #[test]
    fn repeating_group_at_end_of_message() {
        // Group is the last thing in the buffer — no trailing non-group tag.
        let msg = b"268=2\x01269=0\x01270=100\x01269=1\x01270=200\x01";

        let group_tags: &[u32] = &[269, 270];
        let mut reader = FieldReader::new(msg, 0);
        reader.next_field(); // skip 268
        let group = crate::GroupSpan::new(reader.pos() as u32, 2);

        let mut group_reader = FieldReader::new(msg, group.offset as usize);
        let mut entries: Vec<Vec<RawField>> = Vec::new();
        let mut current: Vec<RawField> = Vec::new();
        while let Some(f) = group_reader.next_field() {
            if !group_tags.contains(&f.tag) {
                break;
            }
            if f.tag == 269 && !current.is_empty() {
                entries.push(core::mem::take(&mut current));
            }
            current.push(f);
        }
        if !current.is_empty() {
            entries.push(current);
        }

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0][1].value.slice(msg), b"100");
        assert_eq!(entries[1][1].value.slice(msg), b"200");
    }

    // =========================================================================
    // Repeating group error cases
    // =========================================================================

    #[test]
    fn repeating_group_truncated() {
        // Count says 3 but buffer only has 2 entries.
        let msg = b"268=3\x01269=0\x01270=100\x01269=1\x01270=200\x01";

        let group_tags: &[u32] = &[269, 270];
        let mut reader = FieldReader::new(msg, 0);
        reader.next_field();
        let group = crate::GroupSpan::new(reader.pos() as u32, 3);

        let mut group_reader = FieldReader::new(msg, group.offset as usize);
        let mut entry_count = 0u16;
        while let Some(f) = group_reader.next_field() {
            if !group_tags.contains(&f.tag) {
                break;
            }
            if f.tag == 269 {
                entry_count += 1;
            }
        }

        // Reader yields what's there without panicking.
        // Codegen detects the mismatch: found 2, expected 3.
        assert_eq!(entry_count, 2);
        assert!(entry_count < group.count);
    }

    #[test]
    fn repeating_group_no_delimiter() {
        // Count says 2 but the delimiter tag (269) never appears.
        let msg = b"268=2\x01270=100\x01271=200\x0110=999\x01";

        let group_tags: &[u32] = &[269, 270, 271];
        let mut reader = FieldReader::new(msg, 0);
        reader.next_field();
        let group = crate::GroupSpan::new(reader.pos() as u32, 2);

        let mut group_reader = FieldReader::new(msg, group.offset as usize);
        let mut entry_count = 0u16;
        while let Some(f) = group_reader.next_field() {
            if !group_tags.contains(&f.tag) {
                break;
            }
            if f.tag == 269 {
                entry_count += 1;
            }
        }

        // No delimiter seen — codegen would flag this as malformed.
        assert_eq!(entry_count, 0);
    }

    #[test]
    fn repeating_group_count_zero() {
        // Count is 0 — group has no entries.
        let msg = b"268=0\x0135=W\x01";

        let mut reader = FieldReader::new(msg, 0);
        reader.next_field();
        let group = crate::GroupSpan::new(reader.pos() as u32, 0);

        assert!(!group.is_present());

        // Reading from the offset just yields the next non-group field.
        let mut group_reader = FieldReader::new(msg, group.offset as usize);
        let next = group_reader.next_field().unwrap();
        assert_eq!(next.tag, 35);
    }

    // =========================================================================
    // Bogus / adversarial input
    // =========================================================================

    #[test]
    fn bogus_no_soh() {
        let mut reader = FieldReader::new(b"just random garbage", 0);
        assert!(reader.next_field().is_none());
    }

    #[test]
    fn bogus_adjacent_soh() {
        // Empty spans between SOH bytes.
        let mut reader = FieldReader::new(b"\x01\x01\x01", 0);
        assert!(reader.next_field().is_none());
    }

    #[test]
    fn bogus_equals_no_tag() {
        let mut reader = FieldReader::new(b"=value\x01", 0);
        assert!(reader.next_field().is_none());
    }

    #[test]
    fn bogus_non_digit_in_tag() {
        // parse_tag reads "1", stops at "a", then field_bytes[1] != '='.
        let mut reader = FieldReader::new(b"1a2=value\x01", 0);
        assert!(reader.next_field().is_none());
    }

    #[test]
    fn bogus_all_soh() {
        let buf = vec![0x01; 64];
        let mut reader = FieldReader::new(&buf, 0);
        assert!(reader.next_field().is_none());
    }

    #[test]
    fn bogus_before_valid_field() {
        // Garbage field stops the iterator; valid field after it is not reached.
        let msg = b"garbage\x0135=D\x01";
        let count = FieldReader::new(msg, 0).count();
        assert_eq!(count, 0);
    }

    #[test]
    fn bogus_only_equals_and_soh() {
        let mut reader = FieldReader::new(b"=\x01=\x01=\x01", 0);
        assert!(reader.next_field().is_none());
    }

    #[test]
    fn bogus_tag_no_value_no_equals() {
        let mut reader = FieldReader::new(b"35\x01", 0);
        assert!(reader.next_field().is_none());
    }
}

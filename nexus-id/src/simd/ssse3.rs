//! SSSE3 hex encode implementation.
//!
//! Uses `pshufb` as a 16-entry lookup table to convert nibble values (0-15)
//! directly to hex ASCII characters in a single instruction. Replaces the
//! 256-byte scalar HEX_TABLE with a 16-byte LUT in a register.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(test)]
use super::scalar;

/// Hex character lookup table for pshufb.
/// Maps nibble values 0-15 to ASCII '0'-'9','a'-'f'.
#[inline(always)]
fn hex_lut() -> __m128i {
    // SAFETY: _mm_setr_epi8 is always safe on SSSE3 targets
    unsafe {
        _mm_setr_epi8(
            b'0' as i8, b'1' as i8, b'2' as i8, b'3' as i8, b'4' as i8, b'5' as i8, b'6' as i8,
            b'7' as i8, b'8' as i8, b'9' as i8, b'a' as i8, b'b' as i8, b'c' as i8, b'd' as i8,
            b'e' as i8, b'f' as i8,
        )
    }
}

/// Encode a u64 as 16 lowercase hex bytes using SSSE3 pshufb.
///
/// Algorithm:
/// 1. Load 8 bytes (big-endian) into XMM register
/// 2. Split each byte into high/low nibbles via shift+mask
/// 3. Interleave nibbles to get 16 values in output order
/// 4. pshufb against hex LUT: nibble value → hex char
#[inline]
pub fn hex_encode_u64(value: u64) -> [u8; 16] {
    let bytes = value.to_be_bytes();
    let lut = hex_lut();

    // SAFETY: All intrinsics are SSSE3 or below, guaranteed available by the
    // module-level `cfg(target_feature = "ssse3")` gate. `bytes` is a stack
    // [u8; 8]; `_mm_loadl_epi64` reads 8 bytes (unaligned-safe). `buf` is a
    // stack [u8; 16]; `_mm_storeu_si128` writes 16 bytes (unaligned-safe).
    unsafe {
        let mask_0f = _mm_set1_epi8(0x0F);
        // Load 8 bytes into low 64 bits of XMM register
        let input = _mm_loadl_epi64(bytes.as_ptr().cast());

        // Split each byte into high and low nibbles.
        // _mm_srli_epi16 shifts 16-bit words, but AND with 0x0F isolates per-byte >> 4:
        // After shift, low 4 bits of each byte = (original_byte >> 4), upper bits are garbage.
        // AND with 0x0F keeps only the correct nibble value.
        let hi_nibbles = _mm_and_si128(_mm_srli_epi16(input, 4), mask_0f);
        let lo_nibbles = _mm_and_si128(input, mask_0f);

        // Interleave: [hi[0], lo[0], hi[1], lo[1], ..., hi[7], lo[7]]
        // Produces 16 nibble values in the correct output order.
        let nibbles = _mm_unpacklo_epi8(hi_nibbles, lo_nibbles);

        // pshufb: use each nibble (0-15) as index into hex LUT.
        // Result: each byte is the corresponding hex character.
        let hex_chars = _mm_shuffle_epi8(lut, nibbles);

        let mut buf = [0u8; 16];
        _mm_storeu_si128(buf.as_mut_ptr().cast(), hex_chars);
        buf
    }
}

/// Encode two u64s as 32 lowercase hex bytes using SSSE3 pshufb.
#[inline]
pub fn hex_encode_u128(hi: u64, lo: u64) -> [u8; 32] {
    let lut = hex_lut();

    // SAFETY: All intrinsics are SSSE3 or below, guaranteed available by the
    // module-level `cfg(target_feature = "ssse3")` gate. `hi_bytes`/`lo_bytes`
    // are stack [u8; 8]; `_mm_loadl_epi64` reads 8 bytes (unaligned-safe).
    // `buf` is a stack [u8; 32]; both `_mm_storeu_si128` writes are within bounds
    // (offsets 0 and 16 into a 32-byte buffer).
    unsafe {
        let mask_0f = _mm_set1_epi8(0x0F);
        let mut buf = [0u8; 32];

        // Encode hi half
        let hi_bytes = hi.to_be_bytes();
        let hi_input = _mm_loadl_epi64(hi_bytes.as_ptr().cast());
        let hi_hi = _mm_and_si128(_mm_srli_epi16(hi_input, 4), mask_0f);
        let hi_lo = _mm_and_si128(hi_input, mask_0f);
        let hi_nibbles = _mm_unpacklo_epi8(hi_hi, hi_lo);
        let hi_chars = _mm_shuffle_epi8(lut, hi_nibbles);
        _mm_storeu_si128(buf.as_mut_ptr().cast(), hi_chars);

        // Encode lo half
        let lo_bytes = lo.to_be_bytes();
        let lo_input = _mm_loadl_epi64(lo_bytes.as_ptr().cast());
        let lo_hi = _mm_and_si128(_mm_srli_epi16(lo_input, 4), mask_0f);
        let lo_lo = _mm_and_si128(lo_input, mask_0f);
        let lo_nibbles = _mm_unpacklo_epi8(lo_hi, lo_lo);
        let lo_chars = _mm_shuffle_epi8(lut, lo_nibbles);
        _mm_storeu_si128(buf.as_mut_ptr().add(16).cast(), lo_chars);

        buf
    }
}

/// Decode a 36-byte dashed UUID string to (hi, lo) u64 pair using SSSE3
/// compaction + SSE2 decode.
///
/// Uses `pshufb` to compact 32 hex chars (stripping 4 dashes) into two
/// contiguous 16-byte XMM registers, then decodes each with SSE2 parallel
/// range classification. Stays entirely in registers — no intermediate
/// memory round-trip, no store-forwarding stalls.
///
/// Caller must validate: `bytes.len() == 36` and dashes at positions 8,13,18,23.
/// Returns `Err(position)` in the compacted 32-char hex space.
#[inline]
pub fn uuid_decode_dashed(bytes: &[u8; 36]) -> Result<(u64, u64), usize> {
    // SAFETY: All intrinsics are SSSE3 or below, guaranteed available by the
    // module-level `cfg(target_feature = "ssse3")` gate. `bytes` is &[u8; 36]:
    // `_mm_loadu_si128` at offset 0 reads [0..16], at offset 16 reads [16..32]
    // (both within bounds, unaligned-safe). `read_unaligned` at offset 32 reads
    // [32..36] (4 bytes, within the 36-byte array).
    unsafe {
        // Load input: 16 + 16 + 4 bytes
        let reg_a = _mm_loadu_si128(bytes.as_ptr().cast()); // input[0..16]
        let reg_b = _mm_loadu_si128(bytes.as_ptr().add(16).cast()); // input[16..32]
        let tail = core::ptr::read_unaligned(bytes.as_ptr().add(32).cast::<u32>());
        let reg_c = _mm_cvtsi32_si128(tail as i32); // input[32..36]

        // Compact first 16 hex chars:
        // From reg_a[0..16]: positions 0-7, 9-12, 14-15 (14 hex chars, skip dashes at 8,13)
        // From reg_b[0..2]:  positions 16-17 (2 hex chars completing segment 3)
        let mask_a1 = _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, // segment 1: 8 chars
            9, 10, 11, 12, // segment 2: 4 chars (skip dash at index 8)
            14, 15, // segment 3 start: 2 chars (skip dash at index 13)
            -1, -1, // zeros — filled from reg_b
        );
        let first_from_a = _mm_shuffle_epi8(reg_a, mask_a1);

        let mask_b1 = _mm_setr_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,
            1, // reg_b[0,1] = input[16,17] → output[14,15]
        );
        let first_from_b = _mm_shuffle_epi8(reg_b, mask_b1);

        let first_16 = _mm_or_si128(first_from_a, first_from_b);

        // Compact second 16 hex chars:
        // From reg_b[3..7]:  positions 19-22 (4 hex chars, skip dash at rel. index 2 = pos 18)
        // From reg_b[8..16]: positions 24-31 (8 hex chars, skip dash at rel. index 7 = pos 23)
        // From reg_c[0..4]:  positions 32-35 (4 hex chars)
        let mask_b2 = _mm_setr_epi8(
            3, 4, 5, 6, // segment 4: 4 chars
            8, 9, 10, 11, 12, 13, 14, 15, // segment 5 start: 8 chars
            -1, -1, -1, -1, // zeros — filled from reg_c
        );
        let second_from_b = _mm_shuffle_epi8(reg_b, mask_b2);

        let mask_c = _mm_setr_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2,
            3, // reg_c[0..4] = input[32..36] → output[12..16]
        );
        let second_from_c = _mm_shuffle_epi8(reg_c, mask_c);

        let second_16 = _mm_or_si128(second_from_b, second_from_c);

        // Decode both halves using SSE2 register-based decode (ILP between the two)
        let hi = super::sse2::hex_decode_16_reg(first_16)?;
        let lo = super::sse2::hex_decode_16_reg(second_16).map_err(|pos| pos + 16)?;

        Ok((hi, lo))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_u64_matches_scalar() {
        let cases: &[u64] = &[
            0,
            1,
            0xFF,
            0xDEAD_BEEF,
            0xDEAD_BEEF_CAFE_BABE,
            0x0123_4567_89AB_CDEF,
            u64::MAX,
        ];

        for &value in cases {
            let ssse3_result = hex_encode_u64(value);
            let scalar_result = scalar::hex_encode_u64(value);
            assert_eq!(ssse3_result, scalar_result, "mismatch for 0x{:016x}", value);
        }
    }

    #[test]
    fn encode_u128_matches_scalar() {
        let cases: &[(u64, u64)] = &[
            (0, 0),
            (u64::MAX, u64::MAX),
            (0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
            (1, 0),
            (0, 1),
        ];

        for &(hi, lo) in cases {
            let ssse3_result = hex_encode_u128(hi, lo);
            let scalar_result = scalar::hex_encode_u128(hi, lo);
            assert_eq!(
                ssse3_result, scalar_result,
                "mismatch for ({:#x}, {:#x})",
                hi, lo
            );
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        use super::super::sse2;
        for value in [0u64, 1, 42, 0xDEAD_BEEF_CAFE_BABE, u64::MAX] {
            let encoded = hex_encode_u64(value);
            let decoded = sse2::hex_decode_16(&encoded).unwrap();
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn uuid_decode_dashed_valid() {
        let cases: &[(&[u8; 36], u64, u64)] = &[
            (
                b"01234567-89ab-cdef-fedc-ba9876543210",
                0x0123_4567_89AB_CDEF,
                0xFEDC_BA98_7654_3210,
            ),
            (b"00000000-0000-0000-0000-000000000000", 0, 0),
            (b"ffffffff-ffff-ffff-ffff-ffffffffffff", u64::MAX, u64::MAX),
            (
                b"DEADBEEF-CAFE-BABE-0123-456789ABCDEF",
                0xDEAD_BEEF_CAFE_BABE,
                0x0123_4567_89AB_CDEF,
            ),
            (
                b"DeAdBeEf-CaFe-BaBe-0123-456789abcdef",
                0xDEAD_BEEF_CAFE_BABE,
                0x0123_4567_89AB_CDEF,
            ),
        ];

        for &(input, expected_hi, expected_lo) in cases {
            let (hi, lo) = uuid_decode_dashed(input).unwrap();
            assert_eq!(
                hi,
                expected_hi,
                "hi mismatch for {:?}",
                core::str::from_utf8(input)
            );
            assert_eq!(
                lo,
                expected_lo,
                "lo mismatch for {:?}",
                core::str::from_utf8(input)
            );
        }
    }

    #[test]
    fn uuid_decode_dashed_matches_compact() {
        use super::super::sse2;
        // Decode dashed and compact forms should produce the same (hi, lo)
        let dashed = b"01234567-89ab-cdef-fedc-ba9876543210";
        let compact = b"0123456789abcdeffedcba9876543210";
        let (dhi, dlo) = uuid_decode_dashed(dashed).unwrap();
        let (chi, clo) = sse2::hex_decode_32(compact).unwrap();
        assert_eq!((dhi, dlo), (chi, clo));
    }

    #[test]
    fn uuid_decode_dashed_invalid_positions() {
        // Test invalid char in each segment, verify correct error position
        let base = *b"01234567-89ab-cdef-fedc-ba9876543210";

        // Segment 1 (compact pos 0..8): input positions 0..8
        for compact_pos in 0..8 {
            let mut input = base;
            input[compact_pos] = b'x';
            assert_eq!(uuid_decode_dashed(&input), Err(compact_pos));
        }

        // Segment 2 (compact pos 8..12): input positions 9..13
        for i in 0..4 {
            let mut input = base;
            input[9 + i] = b'x';
            assert_eq!(uuid_decode_dashed(&input), Err(8 + i));
        }

        // Segment 3 (compact pos 12..16): input positions 14..18
        for i in 0..4 {
            let mut input = base;
            input[14 + i] = b'x';
            assert_eq!(uuid_decode_dashed(&input), Err(12 + i));
        }

        // Segment 4 (compact pos 16..20): input positions 19..23
        for i in 0..4 {
            let mut input = base;
            input[19 + i] = b'x';
            assert_eq!(uuid_decode_dashed(&input), Err(16 + i));
        }

        // Segment 5 (compact pos 20..32): input positions 24..36
        for i in 0..12 {
            let mut input = base;
            input[24 + i] = b'x';
            assert_eq!(uuid_decode_dashed(&input), Err(20 + i));
        }
    }
}

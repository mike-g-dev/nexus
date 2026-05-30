//! Fast encoding utilities for ID string generation.
//!
//! Provides hex, base36, and base62 encoding optimized for fixed-size
//! integer values. All functions write directly to `AsciiString` buffers
//! with no allocation.
//!
//! Hex encoding dispatches to SIMD (SSSE3 pshufb) when available,
//! falling back to scalar lookup table on other architectures.

use nexus_ascii::AsciiString;

/// Base62 alphabet: 0-9, A-Z, a-z
const BASE62_ALPHABET: &[u8; 62] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Base36 alphabet: 0-9, a-z (lowercase for consistency)
const BASE36_ALPHABET: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

/// Crockford Base32 alphabet: 0-9, A-Z excluding I, L, O, U
/// See: https://www.crockford.com/base32.html
const CROCKFORD32_ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// 62² — used for digit-pair decomposition in base62 encoding.
const BASE62_SQ: u64 = 62 * 62;

/// 36² — used for digit-pair decomposition in base36 encoding.
const BASE36_SQ: u64 = 36 * 36;

/// Encode a u64 as 16-character lowercase hex.
///
/// CAP must be >= 16 and a multiple of 8.
#[inline]
pub(crate) fn hex_u64<const CAP: usize>(value: u64) -> AsciiString<CAP> {
    const { assert!(CAP >= 16, "hex_u64 requires CAP >= 16") };
    const { assert!(CAP.is_multiple_of(8), "CAP must be a multiple of 8") };

    let buf = crate::simd::hex_encode_u64(value);
    // SAFETY: All bytes are valid ASCII hex digits (produced by hex encoder)
    unsafe { AsciiString::from_bytes_unchecked(&buf) }
}

/// Encode two u64s as 32-character lowercase hex.
///
/// CAP must be >= 32 and a multiple of 8.
#[inline]
pub(crate) fn hex_u128<const CAP: usize>(hi: u64, lo: u64) -> AsciiString<CAP> {
    const { assert!(CAP >= 32, "hex_u128 requires CAP >= 32") };
    const { assert!(CAP.is_multiple_of(8), "CAP must be a multiple of 8") };

    let buf = crate::simd::hex_encode_u128(hi, lo);
    // SAFETY: All bytes are valid ASCII hex digits (produced by hex encoder)
    unsafe { AsciiString::from_bytes_unchecked(&buf) }
}

/// Encode a u64 as 11-character base62.
///
/// Base62 uses: 0-9, A-Z, a-z (62 characters).
/// Produces fixed-length output with leading zeros.
///
/// Uses digit-pair decomposition: divmod by 62² (3844) per iteration,
/// halving the serial dependency chain from 11 to 6 divisions.
///
/// CAP must be >= 16 and a multiple of 8.
#[inline]
pub(crate) fn base62_u64<const CAP: usize>(mut value: u64) -> AsciiString<CAP> {
    const { assert!(CAP >= 16, "base62_u64 requires CAP >= 16") };
    const { assert!(CAP.is_multiple_of(8), "CAP must be a multiple of 8") };

    let mut buf = [b'0'; 11];

    // 11 digits = 5 pairs + 1 single.
    // Each pair: value % 3844 gives a 2-digit remainder,
    // decomposed into (r / 62, r % 62). The remainder decomposition
    // is independent of the next value /= 3844.

    let r = (value % BASE62_SQ) as usize;
    value /= BASE62_SQ;
    buf[9] = BASE62_ALPHABET[r / 62];
    buf[10] = BASE62_ALPHABET[r % 62];

    let r = (value % BASE62_SQ) as usize;
    value /= BASE62_SQ;
    buf[7] = BASE62_ALPHABET[r / 62];
    buf[8] = BASE62_ALPHABET[r % 62];

    let r = (value % BASE62_SQ) as usize;
    value /= BASE62_SQ;
    buf[5] = BASE62_ALPHABET[r / 62];
    buf[6] = BASE62_ALPHABET[r % 62];

    let r = (value % BASE62_SQ) as usize;
    value /= BASE62_SQ;
    buf[3] = BASE62_ALPHABET[r / 62];
    buf[4] = BASE62_ALPHABET[r % 62];

    let r = (value % BASE62_SQ) as usize;
    value /= BASE62_SQ;
    buf[1] = BASE62_ALPHABET[r / 62];
    buf[2] = BASE62_ALPHABET[r % 62];

    buf[0] = BASE62_ALPHABET[value as usize];

    // SAFETY: All bytes are valid ASCII alphanumeric
    unsafe { AsciiString::from_bytes_unchecked(&buf) }
}

/// Encode a u64 as 13-character base36.
///
/// Base36 uses: 0-9, a-z (36 characters, case-insensitive).
/// Produces fixed-length output with leading zeros.
///
/// Uses digit-pair decomposition: divmod by 36² (1296) per iteration,
/// halving the serial dependency chain from 13 to 7 divisions.
///
/// CAP must be >= 16 and a multiple of 8.
#[inline]
pub(crate) fn base36_u64<const CAP: usize>(mut value: u64) -> AsciiString<CAP> {
    const { assert!(CAP >= 16, "base36_u64 requires CAP >= 16") };
    const { assert!(CAP.is_multiple_of(8), "CAP must be a multiple of 8") };

    let mut buf = [b'0'; 13];

    // 13 digits = 6 pairs + 1 single.

    let r = (value % BASE36_SQ) as usize;
    value /= BASE36_SQ;
    buf[11] = BASE36_ALPHABET[r / 36];
    buf[12] = BASE36_ALPHABET[r % 36];

    let r = (value % BASE36_SQ) as usize;
    value /= BASE36_SQ;
    buf[9] = BASE36_ALPHABET[r / 36];
    buf[10] = BASE36_ALPHABET[r % 36];

    let r = (value % BASE36_SQ) as usize;
    value /= BASE36_SQ;
    buf[7] = BASE36_ALPHABET[r / 36];
    buf[8] = BASE36_ALPHABET[r % 36];

    let r = (value % BASE36_SQ) as usize;
    value /= BASE36_SQ;
    buf[5] = BASE36_ALPHABET[r / 36];
    buf[6] = BASE36_ALPHABET[r % 36];

    let r = (value % BASE36_SQ) as usize;
    value /= BASE36_SQ;
    buf[3] = BASE36_ALPHABET[r / 36];
    buf[4] = BASE36_ALPHABET[r % 36];

    let r = (value % BASE36_SQ) as usize;
    value /= BASE36_SQ;
    buf[1] = BASE36_ALPHABET[r / 36];
    buf[2] = BASE36_ALPHABET[r % 36];

    buf[0] = BASE36_ALPHABET[value as usize];

    // SAFETY: All bytes are valid ASCII alphanumeric
    unsafe { AsciiString::from_bytes_unchecked(&buf) }
}

/// Format 128-bit value as UUID with dashes.
///
/// Encodes hi and lo as hex, then scatters into dashed format:
/// `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
///
/// CAP must be >= 40 and a multiple of 8.
#[inline]
pub(crate) fn uuid_dashed<const CAP: usize>(hi: u64, lo: u64) -> AsciiString<CAP> {
    const { assert!(CAP >= 40, "uuid_dashed requires CAP >= 40") };
    const { assert!(CAP.is_multiple_of(8), "CAP must be a multiple of 8") };

    let hi_hex = crate::simd::hex_encode_u64(hi);
    let lo_hex = crate::simd::hex_encode_u64(lo);
    let mut buf = [0u8; 36];

    // xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    buf[0..8].copy_from_slice(&hi_hex[0..8]);
    buf[8] = b'-';
    buf[9..13].copy_from_slice(&hi_hex[8..12]);
    buf[13] = b'-';
    buf[14..18].copy_from_slice(&hi_hex[12..16]);
    buf[18] = b'-';
    buf[19..23].copy_from_slice(&lo_hex[0..4]);
    buf[23] = b'-';
    buf[24..36].copy_from_slice(&lo_hex[4..16]);

    // SAFETY: All bytes are valid ASCII (hex digits and dashes)
    unsafe { AsciiString::from_bytes_unchecked(&buf) }
}

/// Encode ULID as 26-character Crockford Base32.
///
/// ULID layout:
/// - Timestamp (48 bits): 10 characters
/// - Random (80 bits): 16 characters
///
/// Input: timestamp_ms (48 bits used), rand_hi (16 bits), rand_lo (64 bits)
///
/// CAP must be >= 32 and a multiple of 8.
#[inline]
pub(crate) fn ulid_encode<const CAP: usize>(
    timestamp_ms: u64,
    rand_hi: u16,
    rand_lo: u64,
) -> AsciiString<CAP> {
    const { assert!(CAP >= 32, "ulid_encode requires CAP >= 32") };
    const { assert!(CAP.is_multiple_of(8), "CAP must be a multiple of 8") };

    let mut buf = [0u8; 26];

    // Encode timestamp (48 bits → 10 chars)
    buf[0] = CROCKFORD32_ALPHABET[((timestamp_ms >> 45) & 0x07) as usize];
    buf[1] = CROCKFORD32_ALPHABET[((timestamp_ms >> 40) & 0x1F) as usize];
    buf[2] = CROCKFORD32_ALPHABET[((timestamp_ms >> 35) & 0x1F) as usize];
    buf[3] = CROCKFORD32_ALPHABET[((timestamp_ms >> 30) & 0x1F) as usize];
    buf[4] = CROCKFORD32_ALPHABET[((timestamp_ms >> 25) & 0x1F) as usize];
    buf[5] = CROCKFORD32_ALPHABET[((timestamp_ms >> 20) & 0x1F) as usize];
    buf[6] = CROCKFORD32_ALPHABET[((timestamp_ms >> 15) & 0x1F) as usize];
    buf[7] = CROCKFORD32_ALPHABET[((timestamp_ms >> 10) & 0x1F) as usize];
    buf[8] = CROCKFORD32_ALPHABET[((timestamp_ms >> 5) & 0x1F) as usize];
    buf[9] = CROCKFORD32_ALPHABET[(timestamp_ms & 0x1F) as usize];

    let rand_hi = rand_hi as u64;

    buf[10] = CROCKFORD32_ALPHABET[((rand_hi >> 11) & 0x1F) as usize];
    buf[11] = CROCKFORD32_ALPHABET[((rand_hi >> 6) & 0x1F) as usize];
    buf[12] = CROCKFORD32_ALPHABET[((rand_hi >> 1) & 0x1F) as usize];

    let combined = ((rand_hi & 0x01) << 4) | ((rand_lo >> 60) & 0x0F);
    buf[13] = CROCKFORD32_ALPHABET[combined as usize];

    buf[14] = CROCKFORD32_ALPHABET[((rand_lo >> 55) & 0x1F) as usize];
    buf[15] = CROCKFORD32_ALPHABET[((rand_lo >> 50) & 0x1F) as usize];
    buf[16] = CROCKFORD32_ALPHABET[((rand_lo >> 45) & 0x1F) as usize];
    buf[17] = CROCKFORD32_ALPHABET[((rand_lo >> 40) & 0x1F) as usize];
    buf[18] = CROCKFORD32_ALPHABET[((rand_lo >> 35) & 0x1F) as usize];
    buf[19] = CROCKFORD32_ALPHABET[((rand_lo >> 30) & 0x1F) as usize];
    buf[20] = CROCKFORD32_ALPHABET[((rand_lo >> 25) & 0x1F) as usize];
    buf[21] = CROCKFORD32_ALPHABET[((rand_lo >> 20) & 0x1F) as usize];
    buf[22] = CROCKFORD32_ALPHABET[((rand_lo >> 15) & 0x1F) as usize];
    buf[23] = CROCKFORD32_ALPHABET[((rand_lo >> 10) & 0x1F) as usize];
    buf[24] = CROCKFORD32_ALPHABET[((rand_lo >> 5) & 0x1F) as usize];
    buf[25] = CROCKFORD32_ALPHABET[(rand_lo & 0x1F) as usize];

    // SAFETY: All bytes are valid ASCII (Crockford base32 characters)
    unsafe { AsciiString::from_bytes_unchecked(&buf) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_u64_zero() {
        assert_eq!(hex_u64::<16>(0).as_str(), "0000000000000000");
    }

    #[test]
    fn hex_u64_max() {
        assert_eq!(hex_u64::<16>(u64::MAX).as_str(), "ffffffffffffffff");
    }

    #[test]
    fn hex_u64_known_value() {
        assert_eq!(
            hex_u64::<16>(0xDEAD_BEEF_CAFE_BABE).as_str(),
            "deadbeefcafebabe"
        );
    }

    #[test]
    fn hex_u64_larger_cap() {
        // Verify larger CAP works and produces same content
        let small: AsciiString<16> = hex_u64(0xDEAD_BEEF);
        let large: AsciiString<32> = hex_u64(0xDEAD_BEEF);
        assert_eq!(small.as_str(), large.as_str());
    }

    #[test]
    fn hex_u128_known_value() {
        let hi = 0x0123_4567_89AB_CDEF;
        let lo = 0xFEDC_BA98_7654_3210;
        assert_eq!(
            hex_u128::<32>(hi, lo).as_str(),
            "0123456789abcdeffedcba9876543210"
        );
    }

    #[test]
    fn base62_zero() {
        assert_eq!(base62_u64::<16>(0).as_str(), "00000000000");
    }

    #[test]
    fn base62_max() {
        let encoded: AsciiString<16> = base62_u64(u64::MAX);
        assert_eq!(encoded.len(), 11);
        for c in encoded.as_str().chars() {
            assert!(c.is_ascii_alphanumeric());
        }
    }

    #[test]
    fn base62_known_values() {
        assert_eq!(base62_u64::<16>(0).as_str(), "00000000000");
        assert_eq!(base62_u64::<16>(1).as_str(), "00000000001");
        assert_eq!(base62_u64::<16>(9).as_str(), "00000000009");
        assert_eq!(base62_u64::<16>(10).as_str(), "0000000000A");
        assert_eq!(base62_u64::<16>(35).as_str(), "0000000000Z");
        assert_eq!(base62_u64::<16>(36).as_str(), "0000000000a");
        assert_eq!(base62_u64::<16>(61).as_str(), "0000000000z");
        assert_eq!(base62_u64::<16>(62).as_str(), "00000000010");
    }

    #[test]
    fn base62_larger_cap() {
        let small: AsciiString<16> = base62_u64(12345);
        let large: AsciiString<32> = base62_u64(12345);
        assert_eq!(small.as_str(), large.as_str());
    }

    #[test]
    fn base36_zero() {
        assert_eq!(base36_u64::<16>(0).as_str(), "0000000000000");
    }

    #[test]
    fn base36_max() {
        let encoded: AsciiString<16> = base36_u64(u64::MAX);
        assert_eq!(encoded.len(), 13);
        for c in encoded.as_str().chars() {
            assert!(c.is_ascii_digit() || c.is_ascii_lowercase());
        }
    }

    #[test]
    fn base36_known_values() {
        assert_eq!(base36_u64::<16>(0).as_str(), "0000000000000");
        assert_eq!(base36_u64::<16>(1).as_str(), "0000000000001");
        assert_eq!(base36_u64::<16>(9).as_str(), "0000000000009");
        assert_eq!(base36_u64::<16>(10).as_str(), "000000000000a");
        assert_eq!(base36_u64::<16>(35).as_str(), "000000000000z");
        assert_eq!(base36_u64::<16>(36).as_str(), "0000000000010");
    }

    #[test]
    fn uuid_dashed_format() {
        let hi = 0x0123_4567_89AB_CDEF;
        let lo = 0xFEDC_BA98_7654_3210;
        let uuid: AsciiString<40> = uuid_dashed(hi, lo);
        assert_eq!(uuid.as_str(), "01234567-89ab-cdef-fedc-ba9876543210");
        assert_eq!(uuid.len(), 36);
    }

    #[test]
    fn uuid_dashed_zeros() {
        let uuid: AsciiString<40> = uuid_dashed(0, 0);
        assert_eq!(uuid.as_str(), "00000000-0000-0000-0000-000000000000");
    }

    #[test]
    fn ulid_encode_larger_cap() {
        let small: AsciiString<32> = ulid_encode(1_234_567_890, 0xABCD, 0x1234_5678_9ABC_DEF0);
        let large: AsciiString<64> = ulid_encode(1_234_567_890, 0xABCD, 0x1234_5678_9ABC_DEF0);
        assert_eq!(small.as_str(), large.as_str());
        assert_eq!(small.len(), 26);
    }
}

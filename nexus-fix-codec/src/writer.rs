//! FIX field writer for encoding `tag=value\x01` fields into a buffer.
//!
//! Provides [`FieldWriter`], a cursor that writes FIX fields into a
//! caller-provided `&mut [u8]`. Generated encoders (from `nexus-fix-codegen`)
//! compose this with framing logic for complete message construction.
//!
//! Also provides [`encode_field`] as a standalone function for cases
//! where the struct overhead isn't needed.

/// FIX field writer.
///
/// Wraps a `&mut [u8]` buffer and tracks the write position as fields
/// are appended. Symmetric with [`FieldReader`](crate::FieldReader)
/// on the read side.
///
/// # Example
///
/// ```
/// use nexus_fix_codec::writer::FieldWriter;
///
/// let mut buf = [0u8; 64];
/// let mut w = FieldWriter::wrap(&mut buf);
/// w.field(35, b"D");
/// w.field(49, b"SENDER");
/// w.field(55, b"BTC-USD");
/// assert_eq!(w.data(), b"35=D\x0149=SENDER\x0155=BTC-USD\x01");
/// ```
pub struct FieldWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> FieldWriter<'a> {
    /// Wrap a mutable buffer for writing FIX fields.
    #[inline]
    pub fn wrap(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Wrap a mutable buffer, starting writes at `offset`.
    #[inline]
    pub fn wrap_at(buf: &'a mut [u8], offset: usize) -> Self {
        Self { buf, pos: offset }
    }

    /// Write a `tag=value\x01` field. Advances position.
    #[inline]
    pub fn field(&mut self, tag: u32, value: &[u8]) {
        self.pos = encode_field(self.buf, self.pos, tag, value);
    }

    /// Current write position (bytes written so far).
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// The written portion of the buffer.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.buf[..self.pos]
    }
}

/// Write a `tag=value\x01` field into `buf` at `pos`. Returns new position.
///
/// This is the standalone version of [`FieldWriter::field`] for use
/// without the struct wrapper.
///
/// # Panics
///
/// Panics if `buf` is too small to hold the encoded field
/// (`pos + tag_digits + 1 + value.len() + 1` bytes). The capacity is
/// checked once up front; on success every byte is written without
/// further per-byte bounds checks.
#[inline]
pub fn encode_field(buf: &mut [u8], pos: usize, tag: u32, value: &[u8]) -> usize {
    let digits = tag_digits(tag);
    // Bytes this field needs: `digits` tag bytes + `=` + value + SOH. Computed
    // without overflow — `digits <= 10` and `value.len() <= isize::MAX`, so the
    // sum stays below `usize::MAX`. The `pos <= buf.len()` guard is checked
    // first so `buf.len() - pos` cannot underflow.
    let need = digits + 2 + value.len();
    assert!(
        pos <= buf.len() && need <= buf.len() - pos,
        "encode_field: buffer too small (need {need} at pos {pos}, have {})",
        buf.len()
    );

    // SAFETY: the assert guarantees `pos + need <= buf.len()`. Every write below
    // lands in `pos..pos + need`: `digits` tag bytes, the `=`, `value.len()`
    // value bytes, then the trailing SOH — so all indices are in bounds.
    // `value` is a `&[u8]` and `buf` a `&mut [u8]`; the borrow checker forbids
    // them from aliasing, so the copy is genuinely non-overlapping.
    unsafe {
        write_tag_unchecked(buf, pos, tag, digits);
        let mut p = pos + digits;
        *buf.get_unchecked_mut(p) = b'=';
        p += 1;
        core::ptr::copy_nonoverlapping(value.as_ptr(), buf.as_mut_ptr().add(p), value.len());
        p += value.len();
        *buf.get_unchecked_mut(p) = 0x01;
        p + 1
    }
}

/// Format a checksum value as 3 zero-padded ASCII digits.
///
/// FIX tag 10 is always a 3-character zero-padded decimal value.
///
/// # Example
///
/// ```
/// use nexus_fix_codec::writer::format_checksum;
///
/// assert_eq!(&format_checksum(42), b"042");
/// assert_eq!(&format_checksum(178), b"178");
/// ```
#[inline]
pub fn format_checksum(sum: u8) -> [u8; 3] {
    [sum / 100 + b'0', (sum / 10) % 10 + b'0', sum % 10 + b'0']
}

// =============================================================================
// Internal: tag number → ASCII digits
// =============================================================================

/// Number of ASCII digits needed to represent `tag`.
///
/// FIX tags are always `>= 1`; tag `0` still reports one digit so the
/// encoding round-trips. When `tag` is a compile-time constant (the
/// generated-encoder case), this folds to a constant and the caller's
/// length math and digit writes collapse to straight-line stores.
#[inline]
fn tag_digits(tag: u32) -> usize {
    match tag {
        0..=9 => 1,
        10..=99 => 2,
        100..=999 => 3,
        1_000..=9_999 => 4,
        10_000..=99_999 => 5,
        100_000..=999_999 => 6,
        1_000_000..=9_999_999 => 7,
        10_000_000..=99_999_999 => 8,
        100_000_000..=999_999_999 => 9,
        _ => 10,
    }
}

/// Write `tag` as exactly `digits` ASCII characters starting at `pos`.
///
/// # Safety
///
/// `buf[pos..pos + digits]` must be in bounds, and `digits` must equal
/// `tag_digits(tag)` so the value fits exactly in the written span.
#[inline]
unsafe fn write_tag_unchecked(buf: &mut [u8], pos: usize, tag: u32, digits: usize) {
    // Internal contract, guaranteed by construction in `encode_field` and
    // proven by its capacity assert. Debug-only: a development tripwire, not a
    // release-time guard (those would belong on external input, not here).
    debug_assert_eq!(digits, tag_digits(tag), "digit count must match tag width");
    debug_assert!(
        pos + digits <= buf.len(),
        "tag write span must be in bounds"
    );

    let mut t = tag;
    let mut i = pos + digits;
    // Least-significant digit first, walking back to `pos`. Trip count is
    // `digits` (constant when `tag` is constant), so no data-dependent exit.
    while i > pos {
        i -= 1;
        // SAFETY: `i` ranges over `pos..pos + digits`, in bounds by precondition.
        unsafe {
            *buf.get_unchecked_mut(i) = b'0' + (t % 10) as u8;
        }
        t /= 10;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_field() {
        let mut buf = [0u8; 32];
        let end = encode_field(&mut buf, 0, 35, b"D");
        assert_eq!(&buf[..end], b"35=D\x01");
    }

    #[test]
    fn multiple_fields() {
        let mut buf = [0u8; 64];
        let mut pos = 0;
        pos = encode_field(&mut buf, pos, 8, b"FIX.4.4");
        pos = encode_field(&mut buf, pos, 35, b"D");
        pos = encode_field(&mut buf, pos, 49, b"SENDER");
        assert_eq!(&buf[..pos], b"8=FIX.4.4\x0135=D\x0149=SENDER\x01");
    }

    #[test]
    fn all_tag_widths() {
        let cases: &[(u32, &[u8])] = &[
            (8, b"8=v\x01"),
            (35, b"35=v\x01"),
            (150, b"150=v\x01"),
            (5592, b"5592=v\x01"),
            (10000, b"10000=v\x01"),
        ];
        for &(tag, expected) in cases {
            let mut buf = [0u8; 16];
            let end = encode_field(&mut buf, 0, tag, b"v");
            assert_eq!(&buf[..end], expected, "tag={}", tag);
        }
    }

    #[test]
    fn empty_value() {
        let mut buf = [0u8; 16];
        let end = encode_field(&mut buf, 0, 35, b"");
        assert_eq!(&buf[..end], b"35=\x01");
    }

    #[test]
    fn exact_fit_buffer() {
        // "35=D\x01" is exactly 5 bytes — no slack.
        let mut buf = [0u8; 5];
        let end = encode_field(&mut buf, 0, 35, b"D");
        assert_eq!(&buf[..end], b"35=D\x01");
    }

    #[test]
    #[should_panic(expected = "buffer too small")]
    fn panics_when_too_small() {
        let mut buf = [0u8; 4]; // needs 5 for "35=D\x01"
        encode_field(&mut buf, 0, 35, b"D");
    }

    #[test]
    #[should_panic(expected = "buffer too small")]
    fn panics_when_pos_past_end() {
        let mut buf = [0u8; 8];
        encode_field(&mut buf, 9, 35, b"D");
    }

    #[test]
    fn wide_tag_widths_roundtrip() {
        // tag_digits boundaries: every width writes the exact digit count.
        for &(tag, expected) in &[
            (9u32, "9"),
            (99, "99"),
            (999, "999"),
            (9_999, "9999"),
            (99_999, "99999"),
            (999_999, "999999"),
            (4_294_967_295, "4294967295"), // u32::MAX, 10 digits
        ] {
            let mut buf = [0u8; 32];
            let end = encode_field(&mut buf, 0, tag, b"v");
            let want = format!("{expected}=v\u{1}");
            assert_eq!(&buf[..end], want.as_bytes(), "tag={tag}");
        }
    }

    #[test]
    fn encode_from_offset() {
        let mut buf = [0u8; 32];
        buf[0..5].copy_from_slice(b"XXXXX");
        let end = encode_field(&mut buf, 5, 35, b"D");
        assert_eq!(&buf[5..end], b"35=D\x01");
    }

    #[test]
    fn format_checksum_values() {
        assert_eq!(&format_checksum(0), b"000");
        assert_eq!(&format_checksum(42), b"042");
        assert_eq!(&format_checksum(178), b"178");
        assert_eq!(&format_checksum(255), b"255");
    }

    #[test]
    fn writer_basic() {
        let mut buf = [0u8; 64];
        let mut w = FieldWriter::wrap(&mut buf);
        w.field(35, b"D");
        w.field(49, b"SENDER");
        assert_eq!(w.pos(), 15);
        assert_eq!(w.data(), b"35=D\x0149=SENDER\x01");
    }

    #[test]
    fn writer_wrap_at() {
        let mut buf = [0u8; 64];
        let mut w = FieldWriter::wrap_at(&mut buf, 10);
        w.field(35, b"D");
        assert_eq!(w.pos(), 15);
        assert_eq!(&buf[10..15], b"35=D\x01");
    }

    #[test]
    fn roundtrip_read_write() {
        let mut buf = [0u8; 128];
        let mut w = FieldWriter::wrap(&mut buf);
        w.field(8, b"FIX.4.4");
        w.field(35, b"D");
        w.field(49, b"SENDER");
        w.field(55, b"BTC-USD");
        let written = w.pos();

        let mut reader = crate::FieldReader::new(&buf[..written], 0);
        let fields: Vec<_> = reader.by_ref().collect();

        assert_eq!(fields.len(), 4);
        assert_eq!(fields[0].tag, 8);
        assert_eq!(fields[0].value.slice(&buf), b"FIX.4.4");
        assert_eq!(fields[1].tag, 35);
        assert_eq!(fields[1].value.slice(&buf), b"D");
        assert_eq!(fields[2].tag, 49);
        assert_eq!(fields[2].value.slice(&buf), b"SENDER");
        assert_eq!(fields[3].tag, 55);
        assert_eq!(fields[3].value.slice(&buf), b"BTC-USD");
    }

    #[test]
    fn writer_with_checksum() {
        let mut buf = [0u8; 128];
        let body_end;
        {
            let mut w = FieldWriter::wrap(&mut buf);
            w.field(35, b"D");
            w.field(49, b"SENDER");
            body_end = w.pos();
        }

        let sum = crate::checksum(&buf[..body_end]);
        let msg_end = encode_field(&mut buf, body_end, 10, &format_checksum(sum));

        assert!(buf[body_end..msg_end].starts_with(b"10="));
        assert_eq!(buf[msg_end - 1], 0x01);
    }
}

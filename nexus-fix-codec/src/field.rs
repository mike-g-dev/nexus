//! Typed field accessor handle for generated FIX codecs.
//!
//! A generated message exposes one accessor per field that returns an
//! `Option<FieldView>` — `None` if the field is absent, otherwise a
//! lightweight handle over the present field's bytes. The handle carries the
//! small fixed method set ([`get`](FieldView::get),
//! [`checked`](FieldView::checked), [`is_valid`](FieldView::is_valid),
//! [`as_bytes`](FieldView::as_bytes)) so the message struct's namespace stays
//! clean — one method per field, not five — and the accessor logic lives here
//! in the codec rather than being regenerated as boilerplate per field.
//!
//! Presence and validity are separate axes: presence is the outer `Option`,
//! validity is the [`checked`](FieldView::checked) `Result`. A `FieldView`
//! therefore always represents a field that is present in the message.

use core::marker::PhantomData;

use nexus_ascii::{AsciiChar, AsciiTextStr};

use crate::error::FixValueError;
use crate::span::FieldSpan;
use crate::types::{
    FixDate, FixDecimal, FixMonthYear, FixTenor, FixTime, FixTimestamp, FixTzTime, FixTzTimestamp,
    parse_fix_bool, parse_fix_char, parse_fix_day_of_month, parse_fix_int, parse_fix_seqnum,
    parse_fix_text, parse_fix_uint,
};

/// Parse a value of `Self` from a single FIX field's bytes.
///
/// Implemented in the codec for every FIX field datatype, so [`FieldView`] can
/// be generic over the value type without the generator emitting per-type
/// parse calls. It monomorphizes and inlines — no `fn`-pointer indirection.
///
/// These parse a *present* field's bytes; absence never reaches here — it is
/// the `None` arm of the accessor's `Option<FieldView>`.
pub trait FromFixValue<'buf>: Copy {
    /// Parse `Self` from a present field's value bytes.
    fn parse_field(bytes: &'buf [u8]) -> Result<Self, FixValueError>;
}

impl<'buf> FromFixValue<'buf> for &'buf [u8] {
    #[inline]
    fn parse_field(bytes: &'buf [u8]) -> Result<Self, FixValueError> {
        Ok(bytes) // DATA / raw bytes: identity, always valid
    }
}

impl<'buf> FromFixValue<'buf> for &'buf AsciiTextStr {
    #[inline]
    fn parse_field(bytes: &'buf [u8]) -> Result<Self, FixValueError> {
        parse_fix_text(bytes)
    }
}

// Owned value types: parser independent of the buffer lifetime.
macro_rules! from_fix_value {
    ($($ty:ty => $parse:expr),+ $(,)?) => {
        $(
            impl FromFixValue<'_> for $ty {
                #[inline]
                fn parse_field(bytes: &[u8]) -> Result<Self, FixValueError> {
                    $parse(bytes)
                }
            }
        )+
    };
}

from_fix_value! {
    AsciiChar => parse_fix_char,
    bool => parse_fix_bool,
    i64 => parse_fix_int,
    u32 => parse_fix_uint,
    u64 => parse_fix_seqnum,
    u8 => parse_fix_day_of_month,
    FixDecimal => FixDecimal::parse,
    FixDate => FixDate::parse,
    FixTime => FixTime::parse,
    FixTimestamp => FixTimestamp::parse,
    FixMonthYear => FixMonthYear::parse,
    FixTenor => FixTenor::parse,
    FixTzTime => FixTzTime::parse,
    FixTzTimestamp => FixTzTimestamp::parse,
}

/// A typed view over a single *present* FIX field in a decoded message.
///
/// Constructed by generated accessors (`msg.price()` returns
/// `Option<FieldView<..>>`); it carries the field's [`FieldSpan`] plus the
/// message buffer and parses on demand. Zero-copy and zero-cost — the handle is
/// a temporary that optimizes away, and the parse inlines via [`FromFixValue`].
///
/// The convention: validate untrusted (counterparty) data at the boundary with
/// [`checked`](Self::checked) / [`is_valid`](Self::is_valid), then use the bare
/// [`get`](Self::get) for clean access once the value is known good.
pub struct FieldView<'buf, T> {
    span: FieldSpan,
    buf: &'buf [u8],
    _t: PhantomData<T>,
}

impl<'buf, T: FromFixValue<'buf>> FieldView<'buf, T> {
    /// Wrap a field span over its message buffer, or `None` if the field is
    /// absent.
    ///
    /// Fallible like [`NonZero::new`](core::num::NonZero): a `FieldView` always
    /// represents a present field, so presence lives in the returned `Option`,
    /// not in the parse `Result`. Called by generated accessors.
    #[inline]
    pub fn new(span: FieldSpan, buf: &'buf [u8]) -> Option<Self> {
        span.is_present().then_some(Self {
            span,
            buf,
            _t: PhantomData,
        })
    }

    /// The raw field bytes (the field is present by construction; no parse).
    #[inline]
    pub fn as_bytes(&self) -> &'buf [u8] {
        self.span.slice(self.buf)
    }

    /// Parse the field's value.
    ///
    /// `Ok` = valid; `Err` = present but malformed. Absence is not represented
    /// here — it is the `None` arm of the accessor's `Option`. Use this at the
    /// trust boundary.
    #[inline]
    pub fn checked(&self) -> Result<T, FixValueError> {
        T::parse_field(self.span.slice(self.buf))
    }

    /// Whether the field parses successfully.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.checked().is_ok()
    }

    /// The parsed value.
    ///
    /// # Panics
    /// Panics if the field is malformed. Use on validated fields; guard
    /// untrusted input with [`checked`](Self::checked) or
    /// [`is_valid`](Self::is_valid) first.
    #[inline]
    pub fn get(&self) -> T {
        self.checked().expect("FieldView::get on a malformed field")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn present(buf: &[u8]) -> FieldSpan {
        FieldSpan::new(0, buf.len() as u32)
    }

    #[test]
    fn present_valid_value() {
        let buf = b"12345";
        let f: FieldView<i64> = FieldView::new(present(buf), buf).unwrap();
        assert!(f.is_valid());
        assert_eq!(f.as_bytes(), &buf[..]);
        assert_eq!(f.checked(), Ok(12345));
        assert_eq!(f.get(), 12345);
    }

    #[test]
    fn absent_is_none() {
        let buf = b"12345";
        let f: Option<FieldView<i64>> = FieldView::new(FieldSpan::EMPTY, buf);
        assert!(f.is_none());
    }

    #[test]
    fn present_but_malformed() {
        let buf = b"12x";
        let f: FieldView<i64> = FieldView::new(present(buf), buf).unwrap();
        assert!(!f.is_valid());
        assert_eq!(f.checked(), Err(FixValueError::NotNumeric));
    }

    #[test]
    #[should_panic(expected = "malformed")]
    fn get_panics_on_malformed() {
        let buf = b"12x";
        let f: FieldView<i64> = FieldView::new(present(buf), buf).unwrap();
        f.get();
    }

    #[test]
    fn text_decimal_and_raw_views() {
        let sym = b"BTC-USD";
        let t: FieldView<&AsciiTextStr> = FieldView::new(present(sym), sym).unwrap();
        assert_eq!(t.get().as_str(), "BTC-USD");

        let price = b"99.50";
        let d: FieldView<FixDecimal> = FieldView::new(present(price), price).unwrap();
        assert_eq!(d.get(), FixDecimal::parse(b"99.50").unwrap());

        let raw: &[u8] = b"\x07\x08\x09";
        let r: FieldView<&[u8]> = FieldView::new(present(raw), raw).unwrap();
        assert_eq!(r.get(), raw); // DATA: identity, always valid
        assert_eq!(r.as_bytes(), raw);
    }
}

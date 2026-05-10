//! Per-(backing, D, IntType) `From` and `TryFrom` impls for `Decimal`.
//!
//! For each `(Backing, D)` pair, every primitive integer type is partitioned
//! into one of two categories:
//!
//! - **Sound** — `IntType::MAX * 10^D` fits the backing (and `IntType::MIN`
//!   side too). These get `impl From<IntType>`, which is infallible by
//!   construction. The standard library's blanket
//!   `impl<T, U: Into<T>> TryFrom<U> for T` then auto-provides `TryFrom`.
//!
//! - **Unsound** — overflow possible. For `i64` and `u64` only (preserving
//!   the existing public API surface), these get `impl TryFrom<IntType>`
//!   returning `ConvertError::Overflow` on overflow. Smaller integer types
//!   (`i8`, `i16`, `i32`, `u8`, `u16`, `u32`) get no impl when unsound —
//!   callers can widen explicitly.
//!
//! Self-backing (`i32 -> Decimal<i32, _>`, etc.) is only sound at `D = 0`
//! (identity conversion, SCALE = 1). `From<backing>` is emitted at the
//! specific (backing, 0) cells; at `D > 0` the conversion would overflow
//! on large values and routes through `TryFrom` instead. `TryFrom<u64>`
//! and cross-backing `TryFrom<i64>` continue to work via the unsound path.
//!
//! The truth table below was derived programmatically — see
//! `tests/from_int.rs::truth_table_matches_verifier`.

use crate::Decimal;
use crate::error::ConvertError;

/// Const-fn version of the runtime `is_sound` helper in
/// `tests/from_int.rs`. Used inside [`impl_from_sound!`] to assert at
/// compile time that every emitted `From<IntType> for Decimal<Backing, D>`
/// is mathematically lossless. Returns `true` iff `IntType::MAX * 10^D`
/// fits `Backing` and `IntType::MIN * 10^D` does too.
///
/// Semantics (rule + checked-arith propagation) mirror the runtime
/// helper exactly. Any future drift is unlikely because both encode the
/// same predicate; the compile-time path catches macro-table typos
/// before they reach the runtime tests.
pub(crate) const fn is_sound_const(
    int_max: i128,
    int_min: i128,
    backing_max: i128,
    backing_min: i128,
    d: u32,
) -> bool {
    let Some(pow) = 10_i128.checked_pow(d) else {
        return false;
    };
    let Some(max_scaled) = int_max.checked_mul(pow) else {
        return false;
    };
    let Some(min_scaled) = int_min.checked_mul(pow) else {
        return false;
    };
    max_scaled <= backing_max && min_scaled >= backing_min
}

macro_rules! impl_from_sound {
    ($backing:ty, $d:literal, [$($int:ty),* $(,)?]) => {
        $(
            // Compile-time soundness check. Fails the build if this
            // (backing, D, int) cell would overflow on
            // IntType::MAX * 10^D or IntType::MIN * 10^D — catches
            // accidentally including an unsound IntType in the list.
            // Note: omissions (a sound IntType missing from the list)
            // still compile silently; this asserts soundness of what
            // is emitted, not coverage of what should be.
            const _: () = assert!(
                $crate::from_int::is_sound_const(
                    <$int>::MAX as i128,
                    <$int>::MIN as i128,
                    <$backing>::MAX as i128,
                    <$backing>::MIN as i128,
                    $d,
                ),
                concat!(
                    "nexus-decimal: impl_from_sound! invoked with unsound combination ",
                    stringify!($int),
                    " -> Decimal<",
                    stringify!($backing),
                    ", ",
                    stringify!($d),
                    ">"
                ),
            );

            impl From<$int> for Decimal<$backing, $d> {
                /// Lossless conversion. The compiler only emits this impl
                /// when `IntType::MAX * 10^D` fits the backing — overflow
                /// is impossible by construction (verified at compile time).
                #[inline(always)]
                fn from(value: $int) -> Self {
                    Self {
                        value: (value as $backing) * Self::SCALE,
                    }
                }
            }
        )*
    };
}

macro_rules! impl_try_from_int {
    ($backing:ty, $d:literal, signed: [$($int:ty),* $(,)?]) => {
        $(
            impl TryFrom<$int> for Decimal<$backing, $d> {
                type Error = ConvertError;
                #[inline]
                fn try_from(value: $int) -> Result<Self, Self::Error> {
                    Self::from_i64(value as i64).ok_or(ConvertError::Overflow)
                }
            }
        )*
    };
    ($backing:ty, $d:literal, unsigned: [$($int:ty),* $(,)?]) => {
        $(
            impl TryFrom<$int> for Decimal<$backing, $d> {
                type Error = ConvertError;
                #[inline]
                fn try_from(value: $int) -> Result<Self, Self::Error> {
                    Self::from_u64(value as u64).ok_or(ConvertError::Overflow)
                }
            }
        )*
    };
}

// ===== Self-backing at D=0 =====
//
// At D=0, SCALE = 1, so conversion is a 1:1 identity mapping with no
// possibility of overflow. These impls are the D=0-only exception to
// the "self-backing goes through TryFrom" rule. Std's blanket
// `impl<T, U: Into<T>> TryFrom<U> for T` auto-derives `TryFrom<backing>`
// from these, so callers who want the fallible API surface still get it.

impl From<i32> for Decimal<i32, 0> {
    /// Identity conversion — at D=0, the backing value is the raw value.
    #[inline(always)]
    fn from(value: i32) -> Self {
        Self { value }
    }
}

impl From<i64> for Decimal<i64, 0> {
    /// Identity conversion — at D=0, the backing value is the raw value.
    #[inline(always)]
    fn from(value: i64) -> Self {
        Self { value }
    }
}

impl From<i128> for Decimal<i128, 0> {
    /// Identity conversion — at D=0, the backing value is the raw value.
    #[inline(always)]
    fn from(value: i128) -> Self {
        Self { value }
    }
}

// ===== i32 backing =====
impl_from_sound!(i32, 0, [i8, i16, u8, u16]);
impl_try_from_int!(i32, 0, signed: [i64]);
impl_try_from_int!(i32, 0, unsigned: [u64]);
impl_from_sound!(i32, 1, [i8, i16, u8, u16]);
impl_try_from_int!(i32, 1, signed: [i64]);
impl_try_from_int!(i32, 1, unsigned: [u64]);
impl_from_sound!(i32, 2, [i8, i16, u8, u16]);
impl_try_from_int!(i32, 2, signed: [i64]);
impl_try_from_int!(i32, 2, unsigned: [u64]);
impl_from_sound!(i32, 3, [i8, i16, u8, u16]);
impl_try_from_int!(i32, 3, signed: [i64]);
impl_try_from_int!(i32, 3, unsigned: [u64]);
impl_from_sound!(i32, 4, [i8, i16, u8, u16]);
impl_try_from_int!(i32, 4, signed: [i64]);
impl_try_from_int!(i32, 4, unsigned: [u64]);
impl_from_sound!(i32, 5, [i8, u8]);
impl_try_from_int!(i32, 5, signed: [i64]);
impl_try_from_int!(i32, 5, unsigned: [u64]);
impl_from_sound!(i32, 6, [i8, u8]);
impl_try_from_int!(i32, 6, signed: [i64]);
impl_try_from_int!(i32, 6, unsigned: [u64]);
impl_from_sound!(i32, 7, [i8]);
impl_try_from_int!(i32, 7, signed: [i64]);
impl_try_from_int!(i32, 7, unsigned: [u64]);
impl_try_from_int!(i32, 8, signed: [i64]);
impl_try_from_int!(i32, 8, unsigned: [u64]);
impl_try_from_int!(i32, 9, signed: [i64]);
impl_try_from_int!(i32, 9, unsigned: [u64]);

// ===== i64 backing =====
impl_from_sound!(i64, 0, [i8, i16, i32, u8, u16, u32]);
// `TryFrom<i64>` auto-derived via std blanket from the `From<i64> for Decimal<i64, 0>`
// impl above. Explicit `impl_try_from_int!(i64, 0, signed: [i64])` removed to
// avoid coherence conflict.
impl_try_from_int!(i64, 0, unsigned: [u64]);
impl_from_sound!(i64, 1, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i64, 1, signed: [i64]);
impl_try_from_int!(i64, 1, unsigned: [u64]);
impl_from_sound!(i64, 2, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i64, 2, signed: [i64]);
impl_try_from_int!(i64, 2, unsigned: [u64]);
impl_from_sound!(i64, 3, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i64, 3, signed: [i64]);
impl_try_from_int!(i64, 3, unsigned: [u64]);
impl_from_sound!(i64, 4, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i64, 4, signed: [i64]);
impl_try_from_int!(i64, 4, unsigned: [u64]);
impl_from_sound!(i64, 5, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i64, 5, signed: [i64]);
impl_try_from_int!(i64, 5, unsigned: [u64]);
impl_from_sound!(i64, 6, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i64, 6, signed: [i64]);
impl_try_from_int!(i64, 6, unsigned: [u64]);
impl_from_sound!(i64, 7, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i64, 7, signed: [i64]);
impl_try_from_int!(i64, 7, unsigned: [u64]);
impl_from_sound!(i64, 8, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i64, 8, signed: [i64]);
impl_try_from_int!(i64, 8, unsigned: [u64]);
impl_from_sound!(i64, 9, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i64, 9, signed: [i64]);
impl_try_from_int!(i64, 9, unsigned: [u64]);
impl_from_sound!(i64, 10, [i8, i16, u8, u16]);
impl_try_from_int!(i64, 10, signed: [i64]);
impl_try_from_int!(i64, 10, unsigned: [u64]);
impl_from_sound!(i64, 11, [i8, i16, u8, u16]);
impl_try_from_int!(i64, 11, signed: [i64]);
impl_try_from_int!(i64, 11, unsigned: [u64]);
impl_from_sound!(i64, 12, [i8, i16, u8, u16]);
impl_try_from_int!(i64, 12, signed: [i64]);
impl_try_from_int!(i64, 12, unsigned: [u64]);
impl_from_sound!(i64, 13, [i8, i16, u8, u16]);
impl_try_from_int!(i64, 13, signed: [i64]);
impl_try_from_int!(i64, 13, unsigned: [u64]);
impl_from_sound!(i64, 14, [i8, i16, u8, u16]);
impl_try_from_int!(i64, 14, signed: [i64]);
impl_try_from_int!(i64, 14, unsigned: [u64]);
impl_from_sound!(i64, 15, [i8, u8]);
impl_try_from_int!(i64, 15, signed: [i64]);
impl_try_from_int!(i64, 15, unsigned: [u64]);
impl_from_sound!(i64, 16, [i8, u8]);
impl_try_from_int!(i64, 16, signed: [i64]);
impl_try_from_int!(i64, 16, unsigned: [u64]);
impl_try_from_int!(i64, 17, signed: [i64]);
impl_try_from_int!(i64, 17, unsigned: [u64]);
impl_try_from_int!(i64, 18, signed: [i64]);
impl_try_from_int!(i64, 18, unsigned: [u64]);

// ===== i128 backing =====
impl_from_sound!(i128, 0, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 1, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 2, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 3, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 4, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 5, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 6, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 7, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 8, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 9, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 10, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 11, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 12, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 13, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 14, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 15, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 16, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 17, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 18, [i8, i16, i32, i64, u8, u16, u32, u64]);
impl_from_sound!(i128, 19, [i8, i16, i32, i64, u8, u16, u32]);
impl_try_from_int!(i128, 19, unsigned: [u64]);
impl_from_sound!(i128, 20, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i128, 20, signed: [i64]);
impl_try_from_int!(i128, 20, unsigned: [u64]);
impl_from_sound!(i128, 21, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i128, 21, signed: [i64]);
impl_try_from_int!(i128, 21, unsigned: [u64]);
impl_from_sound!(i128, 22, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i128, 22, signed: [i64]);
impl_try_from_int!(i128, 22, unsigned: [u64]);
impl_from_sound!(i128, 23, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i128, 23, signed: [i64]);
impl_try_from_int!(i128, 23, unsigned: [u64]);
impl_from_sound!(i128, 24, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i128, 24, signed: [i64]);
impl_try_from_int!(i128, 24, unsigned: [u64]);
impl_from_sound!(i128, 25, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i128, 25, signed: [i64]);
impl_try_from_int!(i128, 25, unsigned: [u64]);
impl_from_sound!(i128, 26, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i128, 26, signed: [i64]);
impl_try_from_int!(i128, 26, unsigned: [u64]);
impl_from_sound!(i128, 27, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i128, 27, signed: [i64]);
impl_try_from_int!(i128, 27, unsigned: [u64]);
impl_from_sound!(i128, 28, [i8, i16, i32, u8, u16, u32]);
impl_try_from_int!(i128, 28, signed: [i64]);
impl_try_from_int!(i128, 28, unsigned: [u64]);
impl_from_sound!(i128, 29, [i8, i16, u8, u16]);
impl_try_from_int!(i128, 29, signed: [i64]);
impl_try_from_int!(i128, 29, unsigned: [u64]);
impl_from_sound!(i128, 30, [i8, i16, u8, u16]);
impl_try_from_int!(i128, 30, signed: [i64]);
impl_try_from_int!(i128, 30, unsigned: [u64]);
impl_from_sound!(i128, 31, [i8, i16, u8, u16]);
impl_try_from_int!(i128, 31, signed: [i64]);
impl_try_from_int!(i128, 31, unsigned: [u64]);
impl_from_sound!(i128, 32, [i8, i16, u8, u16]);
impl_try_from_int!(i128, 32, signed: [i64]);
impl_try_from_int!(i128, 32, unsigned: [u64]);
impl_from_sound!(i128, 33, [i8, i16, u8, u16]);
impl_try_from_int!(i128, 33, signed: [i64]);
impl_try_from_int!(i128, 33, unsigned: [u64]);
impl_from_sound!(i128, 34, [i8, u8]);
impl_try_from_int!(i128, 34, signed: [i64]);
impl_try_from_int!(i128, 34, unsigned: [u64]);
impl_from_sound!(i128, 35, [i8, u8]);
impl_try_from_int!(i128, 35, signed: [i64]);
impl_try_from_int!(i128, 35, unsigned: [u64]);
impl_from_sound!(i128, 36, [i8]);
impl_try_from_int!(i128, 36, signed: [i64]);
impl_try_from_int!(i128, 36, unsigned: [u64]);
impl_try_from_int!(i128, 37, signed: [i64]);
impl_try_from_int!(i128, 37, unsigned: [u64]);
impl_try_from_int!(i128, 38, signed: [i64]);
impl_try_from_int!(i128, 38, unsigned: [u64]);

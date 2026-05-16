//! Property-based tests for the power-of-2 arithmetic and `checked_abs_diff`.
//!
//! Three properties:
//! - Round-trip: when `checked_mul_pow2(v, n)` succeeds, `div_pow2(_, n) == v`.
//! - Symmetry:   `checked_abs_diff(a, b) == checked_abs_diff(b, a)`.
//! - Halve:      `div_pow2(v, 1) == halve(v)` for all `v`.

use nexus_decimal::Decimal;
use proptest::prelude::*;

type D32 = Decimal<i32, 0>;
type D64 = Decimal<i64, 0>;
type D128 = Decimal<i128, 0>;

proptest! {
    // ------- round-trip on every backing -------

    #[test]
    fn mul_div_roundtrip_i32(raw: i32, n in 0u32..i32::BITS) {
        let v = D32::from_raw(raw);
        if let Some(shifted) = v.checked_mul_pow2(n) {
            prop_assert_eq!(shifted.div_pow2(n), v);
        }
    }

    #[test]
    fn mul_div_roundtrip_i64(raw: i64, n in 0u32..i64::BITS) {
        let v = D64::from_raw(raw);
        if let Some(shifted) = v.checked_mul_pow2(n) {
            prop_assert_eq!(shifted.div_pow2(n), v);
        }
    }

    #[test]
    fn mul_div_roundtrip_i128(raw: i128, n in 0u32..i128::BITS) {
        let v = D128::from_raw(raw);
        if let Some(shifted) = v.checked_mul_pow2(n) {
            prop_assert_eq!(shifted.div_pow2(n), v);
        }
    }

    // ------- checked_abs_diff is symmetric -------

    #[test]
    fn abs_diff_symmetric_i32(a: i32, b: i32) {
        let a = D32::from_raw(a);
        let b = D32::from_raw(b);
        prop_assert_eq!(a.checked_abs_diff(b), b.checked_abs_diff(a));
    }

    #[test]
    fn abs_diff_symmetric_i64(a: i64, b: i64) {
        let a = D64::from_raw(a);
        let b = D64::from_raw(b);
        prop_assert_eq!(a.checked_abs_diff(b), b.checked_abs_diff(a));
    }

    #[test]
    fn abs_diff_symmetric_i128(a: i128, b: i128) {
        let a = D128::from_raw(a);
        let b = D128::from_raw(b);
        prop_assert_eq!(a.checked_abs_diff(b), b.checked_abs_diff(a));
    }

    // ------- div_pow2(v, 1) == halve(v) on every backing -------

    #[test]
    fn div_pow2_one_equals_halve_i32(raw: i32) {
        let v = D32::from_raw(raw);
        prop_assert_eq!(v.div_pow2(1), v.halve());
    }

    #[test]
    fn div_pow2_one_equals_halve_i64(raw: i64) {
        let v = D64::from_raw(raw);
        prop_assert_eq!(v.div_pow2(1), v.halve());
    }

    #[test]
    fn div_pow2_one_equals_halve_i128(raw: i128) {
        let v = D128::from_raw(raw);
        prop_assert_eq!(v.div_pow2(1), v.halve());
    }
}

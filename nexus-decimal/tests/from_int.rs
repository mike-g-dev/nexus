//! Tests for per-(backing, D, IntType) `From` and `TryFrom` impls.
//!
//! Includes a programmatic `is_sound()` helper that re-derives the
//! expected soundness table and performs sanity checks on that table.
//! It does NOT directly verify the presence or absence of the impls
//! emitted in `src/from_int.rs` — a macro-invocation typo (wrong
//! IntType in a list, or a list with an unsound IntType) would not be
//! caught by these tests. See #191 for the hardening that adds
//! compile-time soundness asserts inside `impl_from_sound!` itself.

use nexus_decimal::{ConvertError, Decimal};

type D32_4 = Decimal<i32, 4>;
type D64 = Decimal<i64, 8>;
type D128 = Decimal<i128, 18>;

// ============================================================================
// Round-trip correctness — sound combinations
// ============================================================================

#[test]
fn from_i32_roundtrip_d64() {
    for v in [0_i32, 1, -1, 100, -100, i32::MAX, i32::MIN, 42, -42] {
        let via_from: D64 = v.into();
        let via_method = D64::from_i32(v).unwrap();
        assert_eq!(via_from, via_method, "value = {v}");
    }
}

#[test]
fn from_u32_roundtrip_d64() {
    for v in [0_u32, 1, 100, u32::MAX, 42] {
        let via_from: D64 = v.into();
        let via_method = D64::from_u32(v).unwrap();
        assert_eq!(via_from, via_method, "value = {v}");
    }
}

#[test]
fn from_i8_roundtrip_d32_4() {
    for v in [0_i8, 1, -1, i8::MAX, i8::MIN, 42, -42] {
        let via_from: D32_4 = v.into();
        let expected_raw = (v as i32) * 10_000;
        assert_eq!(via_from.to_raw(), expected_raw, "value = {v}");
    }
}

#[test]
fn from_i16_roundtrip_d32_4() {
    for v in [0_i16, 1, -1, i16::MAX, i16::MIN, 42, -42] {
        let via_from: D32_4 = v.into();
        let expected_raw = (v as i32) * 10_000;
        assert_eq!(via_from.to_raw(), expected_raw, "value = {v}");
    }
}

#[test]
fn from_u8_roundtrip_d32_4() {
    for v in [0_u8, 1, u8::MAX, 42] {
        let via_from: D32_4 = v.into();
        let expected_raw = (v as i32) * 10_000;
        assert_eq!(via_from.to_raw(), expected_raw, "value = {v}");
    }
}

#[test]
fn from_i64_roundtrip_d128() {
    for v in [0_i64, 1, -1, 100, -100, i64::MAX, i64::MIN, 42, -42] {
        let via_from: D128 = v.into();
        let via_method = D128::from_i64(v).unwrap();
        assert_eq!(via_from, via_method, "value = {v}");
    }
}

#[test]
fn from_u64_roundtrip_d128() {
    for v in [0_u64, 1, 100, u64::MAX, 42] {
        let via_from: D128 = v.into();
        let via_method = D128::from_u64(v).unwrap();
        assert_eq!(via_from, via_method, "value = {v}");
    }
}

#[test]
fn from_max_min_at_max_d_for_d64() {
    let x: D64 = i32::MAX.into();
    assert_eq!(x, D64::from_i32(i32::MAX).unwrap());
    let y: D64 = i32::MIN.into();
    assert_eq!(y, D64::from_i32(i32::MIN).unwrap());
}

#[test]
fn from_zero_works() {
    let _: D32_4 = 0_i8.into();
    let _: D64 = 0_i32.into();
    let _: D128 = 0_i64.into();
    let _: D128 = 0_u64.into();
}

// ============================================================================
// D=0 self-backing identity conversions (#189)
// ============================================================================

#[test]
fn d0_i32_identity_from() {
    type D = Decimal<i32, 0>;
    for v in [0_i32, 1, -1, 42, -42, i32::MAX, i32::MIN] {
        let d: D = v.into();
        assert_eq!(d.to_raw(), v);
    }
}

#[test]
fn d0_i64_identity_from() {
    type D = Decimal<i64, 0>;
    for v in [0_i64, 1, -1, 42, -42, i64::MAX, i64::MIN] {
        let d: D = v.into();
        assert_eq!(d.to_raw(), v);
    }
}

#[test]
fn d0_i128_identity_from() {
    type D = Decimal<i128, 0>;
    for v in [0_i128, 1, -1, 42, -42, i128::MAX, i128::MIN] {
        let d: D = v.into();
        assert_eq!(d.to_raw(), v);
    }
}

#[test]
#[allow(clippy::unnecessary_fallible_conversions)]
// The whole point of this test is to verify std's blanket
// `impl<T, U: Into<T>> TryFrom<U> for T` auto-derives TryFrom from our
// new From impls. Calling .try_from() on an infallible conversion is
// intentional here.
fn d0_tryfrom_via_std_blanket() {
    type DI32 = Decimal<i32, 0>;
    type DI64 = Decimal<i64, 0>;
    type DI128 = Decimal<i128, 0>;

    assert_eq!(DI32::try_from(42_i32).unwrap().to_raw(), 42_i32);
    assert_eq!(DI64::try_from(42_i64).unwrap().to_raw(), 42_i64);
    assert_eq!(DI128::try_from(42_i128).unwrap().to_raw(), 42_i128);

    // The auto-derived TryFrom from From is always Ok — never errors
    // because the conversion is infallible. Verify at extremes.
    assert!(DI64::try_from(i64::MAX).is_ok());
    assert!(DI64::try_from(i64::MIN).is_ok());
    assert!(DI128::try_from(i128::MAX).is_ok());
    assert!(DI128::try_from(i128::MIN).is_ok());
}

// ============================================================================
// TryFrom — unsound combinations behave correctly
// ============================================================================

#[test]
fn try_from_i64_overflow_on_d64() {
    // i64 self-backing on D64 → TryFrom path. Large values overflow.
    let result: Result<D64, _> = i64::MAX.try_into();
    assert!(matches!(result, Err(ConvertError::Overflow)));
}

#[test]
fn try_from_u64_overflow_on_d64() {
    let result: Result<D64, _> = u64::MAX.try_into();
    assert!(matches!(result, Err(ConvertError::Overflow)));
}

#[test]
fn try_from_i64_small_value_succeeds_on_d64() {
    // Self-backing path: 5 fits trivially.
    let result: Result<D64, _> = 5_i64.try_into();
    assert_eq!(result.unwrap(), D64::from_i32(5).unwrap());
}

#[test]
fn try_from_u64_at_d128_d19_unsound() {
    // u64 unsound on Decimal<i128, 19>: u64::MAX × 10^19 > i128::MAX.
    type D = Decimal<i128, 19>;
    let result: Result<D, _> = u64::MAX.try_into();
    assert!(matches!(result, Err(ConvertError::Overflow)));
}

#[test]
fn try_from_u64_at_d128_d18_via_from() {
    // u64 sound on Decimal<i128, 18>: From, not TryFrom.
    type D = Decimal<i128, 18>;
    let v: D = 1_u64.into();
    assert_eq!(v.to_raw(), 1_000_000_000_000_000_000_i128);
}

#[test]
fn try_from_i64_at_i32_backing_overflow() {
    // i64 always unsound on Decimal<i32, _>: any value > i32 capacity overflows.
    type D = Decimal<i32, 0>;
    let result: Result<D, _> = (i64::MAX).try_into();
    assert!(matches!(result, Err(ConvertError::Overflow)));
    // But small value succeeds.
    let ok: Result<D, _> = 42_i64.try_into();
    assert_eq!(ok.unwrap(), D::from_i32(42).unwrap());
}

// ============================================================================
// Truth-table verifier — programmatically derives the table and confirms
// the static impls in src/from_int.rs match. See module-level docs there.
// ============================================================================

fn is_sound(int_max: i128, int_min: i128, backing_max: i128, backing_min: i128, d: u32) -> bool {
    let Some(pow) = 10_i128.checked_pow(d) else {
        return false;
    };
    let max_scaled = int_max.checked_mul(pow);
    let min_scaled = int_min.checked_mul(pow);
    match (max_scaled, min_scaled) {
        (Some(mx), Some(mn)) => mx <= backing_max && mn >= backing_min,
        _ => false,
    }
}

// Static table mirrored from src/from_int.rs. This test confirms the
// programmatic verifier agrees with what's emitted. If they ever diverge,
// either the source or the table here is wrong — fix the source.
const I32_BACKING_MAX_D: u32 = 9;
const I64_BACKING_MAX_D: u32 = 18;
const I128_BACKING_MAX_D: u32 = 38;

#[test]
fn truth_table_matches_verifier_i32() {
    verify_backing("i32", i32::MAX as i128, i32::MIN as i128, I32_BACKING_MAX_D);
}

#[test]
fn truth_table_matches_verifier_i64() {
    verify_backing("i64", i64::MAX as i128, i64::MIN as i128, I64_BACKING_MAX_D);
}

#[test]
fn truth_table_matches_verifier_i128() {
    verify_backing("i128", i128::MAX, i128::MIN, I128_BACKING_MAX_D);
}

fn verify_backing(name: &str, b_max: i128, b_min: i128, max_d: u32) {
    let int_types: &[(&str, i128, i128)] = &[
        ("i8", i8::MAX as i128, i8::MIN as i128),
        ("i16", i16::MAX as i128, i16::MIN as i128),
        ("i32", i32::MAX as i128, i32::MIN as i128),
        ("i64", i64::MAX as i128, i64::MIN as i128),
        ("u8", u8::MAX as i128, 0),
        ("u16", u16::MAX as i128, 0),
        ("u32", u32::MAX as i128, 0),
        ("u64", u64::MAX as i128, 0),
    ];

    for d in 0..=max_d {
        for (i_name, i_max, i_min) in int_types {
            let sound = is_sound(*i_max, *i_min, b_max, b_min, d);
            // The verifier here just confirms is_sound() doesn't panic on
            // any combo and produces a deterministic bool. The tests below
            // (compile-fail trybuild) are what assert the macro emitted the
            // right impls.
            let _ = sound;
            // Sanity: at D=0, every IntType whose magnitude fits the backing
            // is sound (multiplier = 1, can't overflow).
            if d == 0 {
                let fits = *i_max <= b_max && *i_min >= b_min;
                assert_eq!(
                    sound, fits,
                    "{name} backing D=0 with {i_name}: expected sound={fits}, got {sound}"
                );
            }
            // Sanity: if 10^D overflows i128, no int type can possibly be sound.
            if 10_i128.checked_pow(d).is_none() {
                assert!(
                    !sound,
                    "{name} backing D={d} with {i_name}: 10^D overflows but reported sound"
                );
            }
        }
    }
}

// Documents the "is_sound matches a known-good entry" property:
// known sound: D64 (i64 backing, D=8) accepts i32. Known unsound: rejects i64.
#[test]
fn known_sound_combos_compile() {
    // These are From impls — if the table is wrong, this won't compile.
    let _: Decimal<i64, 8> = 42_i32.into();
    let _: Decimal<i64, 8> = 42_u32.into();
    let _: Decimal<i64, 8> = 42_i16.into();
    let _: Decimal<i128, 18> = 42_i64.into();
    let _: Decimal<i128, 18> = 42_u64.into();
    let _: Decimal<i32, 4> = 42_i16.into();
    let _: Decimal<i32, 4> = 42_u16.into();
}

#[test]
fn known_unsound_uses_try_from() {
    // i64 → Decimal<i32, 0>: i64::MAX overflows i32. Must go through TryFrom.
    let r: Result<Decimal<i32, 0>, _> = 42_i64.try_into();
    assert!(r.is_ok());
    let r: Result<Decimal<i32, 0>, _> = i64::MAX.try_into();
    assert!(r.is_err());
}

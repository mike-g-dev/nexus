//! Arithmetic, rounding, constants, and edge cases.
//!
//! Covers add, sub, neg, abs, floor, ceil, trunc, round, round_dp
//! across all backing types (i32, i64, i128) and custom precisions.

use nexus_decimal::{Decimal, OverflowError};

type D32 = Decimal<i32, 4>;
type D64 = Decimal<i64, 8>;
type D96 = Decimal<i128, 12>;
type D128 = Decimal<i128, 18>;

// ============================================================================
// Constants
// ============================================================================

#[test]
fn zero_is_zero() {
    assert_eq!(D32::ZERO.to_raw(), 0);
    assert_eq!(D64::ZERO.to_raw(), 0);
    assert_eq!(D96::ZERO.to_raw(), 0);
    assert_eq!(D128::ZERO.to_raw(), 0);
}

#[test]
fn one_equals_scale() {
    assert_eq!(D32::ONE.to_raw(), 10_000);
    assert_eq!(D64::ONE.to_raw(), 100_000_000);
    assert_eq!(D96::ONE.to_raw(), 1_000_000_000_000);
    assert_eq!(D128::ONE.to_raw(), 1_000_000_000_000_000_000);
}

#[test]
fn neg_one() {
    assert_eq!(D64::NEG_ONE.to_raw(), -100_000_000);
    assert_eq!(D64::NEG_ONE, -D64::ONE);
}

#[test]
fn max_min_boundaries() {
    assert_eq!(D64::MAX.to_raw(), i64::MAX);
    assert_eq!(D64::MIN.to_raw(), i64::MIN);
    assert_eq!(D32::MAX.to_raw(), i32::MAX);
    assert_eq!(D32::MIN.to_raw(), i32::MIN);
    assert_eq!(D96::MAX.to_raw(), i128::MAX);
    assert_eq!(D128::MAX.to_raw(), i128::MAX);
}

#[test]
fn scale_values() {
    assert_eq!(D32::SCALE, 10_000);
    assert_eq!(D64::SCALE, 100_000_000);
    assert_eq!(D96::SCALE, 1_000_000_000_000);
    assert_eq!(D128::SCALE, 1_000_000_000_000_000_000);
}

#[test]
fn custom_precision() {
    type Usd = Decimal<i64, 2>;
    assert_eq!(Usd::SCALE, 100);
    assert_eq!(Usd::ONE.to_raw(), 100);
    assert_eq!(Usd::new(19, 99).to_raw(), 1999);
}

// ============================================================================
// Constructors
// ============================================================================

#[test]
fn from_raw_roundtrip() {
    let d = D64::from_raw(12345);
    assert_eq!(d.to_raw(), 12345);
}

#[test]
fn new_positive() {
    let d = D64::new(100, 50_000_000); // 100.50
    assert_eq!(d.to_raw(), 10_050_000_000);
}

#[test]
fn new_negative() {
    let d = D64::new(-50, 25_000_000); // -50.25
    assert_eq!(d.to_raw(), -5_025_000_000);
}

#[test]
fn new_zero() {
    let d = D64::new(0, 0);
    assert_eq!(d, D64::ZERO);
}

#[test]
#[should_panic(expected = "overflow")]
fn new_overflow_panics() {
    let _ = D64::new(i64::MAX, 0);
}

#[test]
fn default_is_zero() {
    assert_eq!(D64::default(), D64::ZERO);
    assert_eq!(D32::default(), D32::ZERO);
}

// ============================================================================
// Query methods
// ============================================================================

#[test]
fn is_zero_positive_negative() {
    assert!(D64::ZERO.is_zero());
    assert!(!D64::ONE.is_zero());

    assert!(D64::ONE.is_positive());
    assert!(!D64::ZERO.is_positive());
    assert!(!D64::NEG_ONE.is_positive());

    assert!(D64::NEG_ONE.is_negative());
    assert!(!D64::ZERO.is_negative());
    assert!(!D64::ONE.is_negative());
}

#[test]
fn signum_values() {
    assert_eq!(D64::ONE.signum(), 1);
    assert_eq!(D64::ZERO.signum(), 0);
    assert_eq!(D64::NEG_ONE.signum(), -1);
}

// ============================================================================
// Checked arithmetic
// ============================================================================

#[test]
fn checked_add_basic() {
    let a = D64::new(10, 0);
    let b = D64::new(20, 0);
    assert_eq!(a.checked_add(b).unwrap().to_raw(), D64::new(30, 0).to_raw());
}

#[test]
fn checked_add_overflow_returns_none() {
    assert!(D64::MAX.checked_add(D64::ONE).is_none());
}

#[test]
fn checked_sub_basic() {
    let a = D64::new(30, 0);
    let b = D64::new(10, 0);
    assert_eq!(a.checked_sub(b).unwrap().to_raw(), D64::new(20, 0).to_raw());
}

#[test]
fn checked_sub_overflow_returns_none() {
    assert!(D64::MIN.checked_sub(D64::ONE).is_none());
}

#[test]
fn checked_neg_basic() {
    assert_eq!(D64::ONE.checked_neg().unwrap(), D64::NEG_ONE);
    assert_eq!(D64::NEG_ONE.checked_neg().unwrap(), D64::ONE);
    assert_eq!(D64::ZERO.checked_neg().unwrap(), D64::ZERO);
}

#[test]
fn checked_neg_min_returns_none() {
    assert!(D64::MIN.checked_neg().is_none());
}

#[test]
fn checked_abs_basic() {
    assert_eq!(D64::ONE.checked_abs().unwrap(), D64::ONE);
    assert_eq!(D64::NEG_ONE.checked_abs().unwrap(), D64::ONE);
    assert_eq!(D64::ZERO.checked_abs().unwrap(), D64::ZERO);
}

#[test]
fn checked_abs_min_returns_none() {
    assert!(D64::MIN.checked_abs().is_none());
}

// ============================================================================
// Plain abs() — default semantics (panic in debug on MIN, wrap in release)
// ============================================================================

#[test]
fn abs_positive_unchanged() {
    assert_eq!(D64::ONE.abs(), D64::ONE);
    assert_eq!(D64::from_i32(5).unwrap().abs(), D64::from_i32(5).unwrap());
}

#[test]
fn abs_negative_flipped() {
    assert_eq!(D64::NEG_ONE.abs(), D64::ONE);
    assert_eq!(D64::from_i32(-5).unwrap().abs(), D64::from_i32(5).unwrap());
}

#[test]
fn abs_zero_is_zero() {
    assert_eq!(D64::ZERO.abs(), D64::ZERO);
}

#[test]
fn abs_preserves_fractional() {
    let neg = D64::from_str_exact("-1.23456789").unwrap();
    let pos = D64::from_str_exact("1.23456789").unwrap();
    assert_eq!(neg.abs(), pos);
}

#[test]
fn abs_min_plus_one_does_not_panic() {
    let v = D64::from_raw(i64::MIN + 1);
    assert_eq!(v.abs().to_raw(), i64::MAX);
}

// MIN.abs() panics in debug (i64::MIN.abs() overflows). Skip in release where
// it wraps to MIN — should_panic would fail there.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "attempt to negate with overflow")]
fn abs_min_panics_in_debug() {
    let _ = D64::MIN.abs();
}

#[test]
fn abs_d32() {
    assert_eq!(D32::from_i32(5).unwrap().abs(), D32::from_i32(5).unwrap());
    assert_eq!(D32::from_i32(-5).unwrap().abs(), D32::from_i32(5).unwrap());
    assert_eq!(D32::ZERO.abs(), D32::ZERO);
}

#[test]
fn abs_d128() {
    assert_eq!(D128::from_i32(5).unwrap().abs(), D128::from_i32(5).unwrap());
    assert_eq!(
        D128::from_i32(-5).unwrap().abs(),
        D128::from_i32(5).unwrap()
    );
    assert_eq!(D128::ZERO.abs(), D128::ZERO);
}

#[test]
fn abs_const_evaluable() {
    const NEG: D64 = D64::from_raw(-100_000_000);
    const POS: D64 = NEG.abs();
    assert_eq!(POS.to_raw(), 100_000_000);
}

// ============================================================================
// Saturating arithmetic
// ============================================================================

#[test]
fn saturating_add_clamps() {
    assert_eq!(D64::MAX.saturating_add(D64::ONE), D64::MAX);
    assert_eq!(D64::MIN.saturating_add(D64::NEG_ONE), D64::MIN);
}

#[test]
fn saturating_sub_clamps() {
    assert_eq!(D64::MIN.saturating_sub(D64::ONE), D64::MIN);
    assert_eq!(D64::MAX.saturating_sub(D64::NEG_ONE), D64::MAX);
}

#[test]
fn saturating_neg_min() {
    assert_eq!(D64::MIN.saturating_neg(), D64::MAX);
}

#[test]
fn saturating_abs_min() {
    assert_eq!(D64::MIN.saturating_abs(), D64::MAX);
}

// ============================================================================
// Wrapping arithmetic
// ============================================================================

#[test]
fn wrapping_add_wraps() {
    let result = D64::MAX.wrapping_add(D64::from_raw(1));
    assert_eq!(result, D64::MIN);
}

#[test]
fn wrapping_neg_min() {
    // MIN.wrapping_neg() == MIN for two's complement
    assert_eq!(D64::MIN.wrapping_neg(), D64::MIN);
}

// ============================================================================
// Try variants
// ============================================================================

#[test]
fn try_add_ok() {
    let result = D64::ONE.try_add(D64::ONE);
    assert_eq!(result.unwrap().to_raw(), D64::new(2, 0).to_raw());
}

#[test]
fn try_add_overflow() {
    let result = D64::MAX.try_add(D64::ONE);
    assert_eq!(result, Err(OverflowError));
}

#[test]
fn try_neg_min() {
    assert_eq!(D64::MIN.try_neg(), Err(OverflowError));
}

// ============================================================================
// Operator traits
// ============================================================================

#[test]
fn add_operator() {
    let a = D64::new(10, 0);
    let b = D64::new(20, 0);
    assert_eq!((a + b).to_raw(), D64::new(30, 0).to_raw());
}

#[test]
fn sub_operator() {
    let a = D64::new(30, 0);
    let b = D64::new(10, 0);
    assert_eq!((a - b).to_raw(), D64::new(20, 0).to_raw());
}

#[test]
fn neg_operator() {
    assert_eq!(-D64::ONE, D64::NEG_ONE);
}

#[test]
fn add_assign() {
    let mut a = D64::new(10, 0);
    a += D64::new(5, 0);
    assert_eq!(a.to_raw(), D64::new(15, 0).to_raw());
}

#[test]
fn sub_assign() {
    let mut a = D64::new(10, 0);
    a -= D64::new(3, 0);
    assert_eq!(a.to_raw(), D64::new(7, 0).to_raw());
}

#[test]
#[should_panic(expected = "overflow")]
fn add_operator_overflow_panics() {
    let _ = D64::MAX + D64::ONE;
}

// ============================================================================
// Rounding
// ============================================================================

#[test]
fn floor_positive() {
    assert_eq!(D64::new(1, 75_000_000).floor(), D64::new(1, 0));
    assert_eq!(D64::new(1, 0).floor(), D64::new(1, 0));
}

#[test]
fn floor_negative() {
    assert_eq!(D64::new(-1, 75_000_000).floor(), D64::new(-2, 0));
    assert_eq!(D64::new(-1, 0).floor(), D64::new(-1, 0));
}

#[test]
fn ceil_positive() {
    assert_eq!(D64::new(1, 25_000_000).ceil(), D64::new(2, 0));
    assert_eq!(D64::new(1, 0).ceil(), D64::new(1, 0));
}

#[test]
fn ceil_negative() {
    assert_eq!(D64::new(-1, 25_000_000).ceil(), D64::new(-1, 0));
    assert_eq!(D64::new(-1, 0).ceil(), D64::new(-1, 0));
}

#[test]
fn trunc_positive() {
    assert_eq!(D64::new(1, 99_000_000).trunc(), D64::new(1, 0));
}

#[test]
fn trunc_negative() {
    assert_eq!(D64::new(-1, 99_000_000).trunc(), D64::new(-1, 0));
}

#[test]
fn fract_positive() {
    let d = D64::new(1, 75_000_000); // 1.75
    assert_eq!(d.fract().to_raw(), 75_000_000);
}

#[test]
fn fract_negative() {
    let d = D64::new(-1, 75_000_000); // -1.75
    assert_eq!(d.fract().to_raw(), -75_000_000);
}

#[test]
fn trunc_plus_fract_identity() {
    let values = [
        D64::new(1, 75_000_000),
        D64::new(-1, 75_000_000),
        D64::ZERO,
        D64::new(99, 99_999_999),
    ];
    for d in values {
        assert_eq!(d.trunc() + d.fract(), d, "trunc + fract != self");
    }
}

#[test]
fn to_integer() {
    assert_eq!(D64::new(42, 75_000_000).to_integer(), 42);
    assert_eq!(D64::new(-42, 75_000_000).to_integer(), -42);
    assert_eq!(D64::ZERO.to_integer(), 0);
}

#[test]
fn round_bankers() {
    // 2.5 → 2 (round to even)
    assert_eq!(D64::new(2, 50_000_000).round(), D64::new(2, 0));
    // 3.5 → 4 (round to even)
    assert_eq!(D64::new(3, 50_000_000).round(), D64::new(4, 0));
    // 1.6 → 2
    assert_eq!(D64::new(1, 60_000_000).round(), D64::new(2, 0));
    // 1.4 → 1
    assert_eq!(D64::new(1, 40_000_000).round(), D64::new(1, 0));
}

#[test]
fn round_bankers_negative() {
    // -2.5 → -2 (round to even)
    assert_eq!(D64::new(-2, 50_000_000).round(), D64::new(-2, 0));
    // -3.5 → -4 (round to even)
    assert_eq!(D64::new(-3, 50_000_000).round(), D64::new(-4, 0));
}

#[test]
fn round_dp_basic() {
    let d = D64::new(1, 23_456_789); // 1.23456789
    assert_eq!(d.round_dp(2), D64::new(1, 23_000_000)); // 1.23
    assert_eq!(d.round_dp(4), D64::new(1, 23_460_000)); // 1.2346 (banker's: 5 rounds to even 6)
}

#[test]
fn round_dp_bankers_half() {
    // 1.235 rounded to 2dp: 5 is half, 3 is odd → round up to 1.24
    let d = D64::from_raw(123_500_000); // 1.235
    assert_eq!(d.round_dp(2).to_raw(), 124_000_000); // 1.24

    // 1.225 rounded to 2dp: 5 is half, 2 is even → stay at 1.22
    let d = D64::from_raw(122_500_000); // 1.225
    assert_eq!(d.round_dp(2).to_raw(), 122_000_000); // 1.22
}

#[test]
#[should_panic(expected = "round_dp")]
fn round_dp_panics_if_dp_equals_decimals() {
    let _ = D64::ONE.round_dp(8);
}

// ============================================================================
// Cross-backing-type tests
// ============================================================================

#[test]
fn d32_basic_arithmetic() {
    let a = D32::new(100, 5000); // 100.5
    let b = D32::new(50, 2500); // 50.25
    assert_eq!((a + b).to_raw(), D32::new(150, 7500).to_raw());
}

#[test]
fn d96_basic_arithmetic() {
    let a = D96::new(100, 500_000_000_000); // 100.5
    let b = D96::new(50, 250_000_000_000); // 50.25
    assert_eq!((a + b).to_raw(), D96::new(150, 750_000_000_000).to_raw());
}

#[test]
fn d128_basic_arithmetic() {
    let a = D128::new(100, 500_000_000_000_000_000); // 100.5
    let b = D128::new(50, 250_000_000_000_000_000); // 50.25
    assert_eq!(
        (a + b).to_raw(),
        D128::new(150, 750_000_000_000_000_000).to_raw()
    );
}

// ============================================================================
// Compile-time validation
// ============================================================================

#[test]
fn const_evaluation() {
    // Verify const fn works at compile time
    const A: D64 = D64::new(100, 0);
    const B: D64 = D64::new(50, 0);
    const SUM: D64 = match A.checked_add(B) {
        Some(v) => v,
        None => panic!("overflow"),
    };
    assert_eq!(SUM.to_raw(), D64::new(150, 0).to_raw());
}

#[test]
fn const_rounding() {
    const D: D64 = D64::new(1, 75_000_000);
    const FLOORED: D64 = D.floor();
    const CEILED: D64 = D.ceil();
    const TRUNCATED: D64 = D.trunc();
    assert_eq!(FLOORED, D64::new(1, 0));
    assert_eq!(CEILED, D64::new(2, 0));
    assert_eq!(TRUNCATED, D64::new(1, 0));
}

// ============================================================================
// Ordering and equality
// ============================================================================

#[test]
fn ordering() {
    assert!(D64::ONE > D64::ZERO);
    assert!(D64::ZERO > D64::NEG_ONE);
    assert!(D64::MIN < D64::MAX);
    assert!(D64::new(1, 50_000_000) > D64::new(1, 49_999_999));
}

#[test]
fn equality() {
    assert_eq!(D64::from_raw(100), D64::from_raw(100));
    assert_ne!(D64::from_raw(100), D64::from_raw(101));
}

// ============================================================================
// D96 (Decimal<i128, 12>) rounding
// ============================================================================

mod d96_rounding {
    type D96 = nexus_decimal::Decimal<i128, 12>;

    #[test]
    fn floor_positive() {
        // 1.75 → 1.0
        assert_eq!(D96::new(1, 750_000_000_000).floor(), D96::new(1, 0));
        // Already whole → unchanged
        assert_eq!(D96::new(3, 0).floor(), D96::new(3, 0));
        // 0.001 → 0
        assert_eq!(D96::from_raw(1_000_000_000).floor(), D96::ZERO);
    }

    #[test]
    fn floor_negative() {
        // -1.75 → -2.0
        assert_eq!(D96::new(-1, 750_000_000_000).floor(), D96::new(-2, 0));
        // Already whole → unchanged
        assert_eq!(D96::new(-3, 0).floor(), D96::new(-3, 0));
        // -0.001 → -1.0
        assert_eq!(D96::from_raw(-1_000_000_000).floor(), D96::NEG_ONE);
    }

    #[test]
    fn ceil_positive() {
        // 1.25 → 2.0
        assert_eq!(D96::new(1, 250_000_000_000).ceil(), D96::new(2, 0));
        // Already whole → unchanged
        assert_eq!(D96::new(3, 0).ceil(), D96::new(3, 0));
        // 0.001 → 1.0
        assert_eq!(D96::from_raw(1_000_000_000).ceil(), D96::ONE);
    }

    #[test]
    fn ceil_negative() {
        // -1.25 → -1.0
        assert_eq!(D96::new(-1, 250_000_000_000).ceil(), D96::new(-1, 0));
        // Already whole → unchanged
        assert_eq!(D96::new(-3, 0).ceil(), D96::new(-3, 0));
        // -0.001 → 0.0
        assert_eq!(D96::from_raw(-1_000_000_000).ceil(), D96::ZERO);
    }

    #[test]
    fn trunc_positive() {
        // 1.99 → 1.0
        assert_eq!(D96::new(1, 990_000_000_000).trunc(), D96::new(1, 0));
        // 0.999 → 0.0
        assert_eq!(D96::from_raw(999_000_000_000).trunc(), D96::ZERO);
    }

    #[test]
    fn trunc_negative() {
        // -1.99 → -1.0 (towards zero)
        assert_eq!(D96::new(-1, 990_000_000_000).trunc(), D96::new(-1, 0));
        // -0.999 → 0.0
        assert_eq!(D96::from_raw(-999_000_000_000).trunc(), D96::ZERO);
    }

    #[test]
    fn fract_values() {
        // 1.75 → 0.75
        let d = D96::new(1, 750_000_000_000);
        assert_eq!(d.fract().to_raw(), 750_000_000_000);

        // -1.75 → -0.75
        let d = D96::new(-1, 750_000_000_000);
        assert_eq!(d.fract().to_raw(), -750_000_000_000);

        // Whole number → 0
        assert_eq!(D96::new(5, 0).fract(), D96::ZERO);

        // Zero → 0
        assert_eq!(D96::ZERO.fract(), D96::ZERO);
    }

    #[test]
    fn round_bankers_positive() {
        // 2.5 → 2 (half, even integer → stay)
        assert_eq!(D96::new(2, 500_000_000_000).round(), D96::new(2, 0));
        // 3.5 → 4 (half, odd integer → round up)
        assert_eq!(D96::new(3, 500_000_000_000).round(), D96::new(4, 0));
        // 1.6 → 2 (above half → round up)
        assert_eq!(D96::new(1, 600_000_000_000).round(), D96::new(2, 0));
        // 1.4 → 1 (below half → round down)
        assert_eq!(D96::new(1, 400_000_000_000).round(), D96::new(1, 0));
        // 0.5 → 0 (half, 0 is even → stay)
        assert_eq!(D96::new(0, 500_000_000_000).round(), D96::ZERO);
        // 1.5 → 2 (half, 1 is odd → round up)
        assert_eq!(D96::new(1, 500_000_000_000).round(), D96::new(2, 0));
    }

    #[test]
    fn round_bankers_negative() {
        // -2.5 → -2 (half, even → stay)
        assert_eq!(D96::new(-2, 500_000_000_000).round(), D96::new(-2, 0));
        // -3.5 → -4 (half, odd → round away)
        assert_eq!(D96::new(-3, 500_000_000_000).round(), D96::new(-4, 0));
        // -1.6 → -2 (above half magnitude → round away)
        assert_eq!(D96::new(-1, 600_000_000_000).round(), D96::new(-2, 0));
        // -1.4 → -1 (below half → truncate)
        assert_eq!(D96::new(-1, 400_000_000_000).round(), D96::new(-1, 0));
    }

    #[test]
    fn round_dp_basic() {
        // 1.234567890123 rounded to 2dp → 1.23
        let d = D96::from_raw(1_234_567_890_123);
        assert_eq!(d.round_dp(2), D96::from_raw(1_230_000_000_000));

        // 1.235 rounded to 2dp: half, 3 is odd → round up to 1.24
        let d = D96::from_raw(1_235_000_000_000);
        assert_eq!(d.round_dp(2), D96::from_raw(1_240_000_000_000));

        // 1.225 rounded to 2dp: half, 2 is even → stay at 1.22
        let d = D96::from_raw(1_225_000_000_000);
        assert_eq!(d.round_dp(2), D96::from_raw(1_220_000_000_000));

        // round to 6dp
        let d = D96::from_raw(1_234_567_890_123); // 1.234567890123
        assert_eq!(d.round_dp(6), D96::from_raw(1_234_568_000_000)); // 1.234568
    }

    #[test]
    fn trunc_plus_fract_identity() {
        let values = [
            D96::new(1, 750_000_000_000),
            D96::new(-1, 750_000_000_000),
            D96::ZERO,
            D96::new(99, 999_999_999_999),
            D96::new(-99, 999_999_999_999),
            D96::from_raw(1), // smallest positive fraction
        ];
        for d in values {
            assert_eq!(
                d.trunc() + d.fract(),
                d,
                "trunc + fract != self for {:?}",
                d.to_raw()
            );
        }
    }
}

// ============================================================================
// Power-of-2 multiplication (mul_pow2 family)
// ============================================================================

mod mul_pow2 {
    use super::{D32, D64, D96, D128};
    use nexus_decimal::{Decimal, OverflowError};

    // ------- identity -------

    #[test]
    fn shift_by_zero_is_identity() {
        assert_eq!(D32::ONE.mul_pow2(0), D32::ONE);
        assert_eq!(D64::ONE.mul_pow2(0), D64::ONE);
        assert_eq!(D96::ONE.mul_pow2(0), D96::ONE);
        assert_eq!(D128::ONE.mul_pow2(0), D128::ONE);
        assert_eq!(D64::ZERO.mul_pow2(0), D64::ZERO);
        assert_eq!(D64::NEG_ONE.mul_pow2(0), D64::NEG_ONE);
    }

    #[test]
    fn checked_zero_is_always_some_zero() {
        // 0 << n is always 0 for any n, including n == BITS-1.
        assert_eq!(D32::ZERO.checked_mul_pow2(31), Some(D32::ZERO));
        assert_eq!(D64::ZERO.checked_mul_pow2(63), Some(D64::ZERO));
        assert_eq!(D96::ZERO.checked_mul_pow2(127), Some(D96::ZERO));
    }

    // ------- basic shifts -------

    #[test]
    fn checked_small_shifts() {
        // 1 << 3 == 8 on the backing
        assert_eq!(D64::from_raw(1).checked_mul_pow2(3), Some(D64::from_raw(8)));
        assert_eq!(
            D64::from_raw(-1).checked_mul_pow2(3),
            Some(D64::from_raw(-8))
        );
        // 5 << 4 == 80
        assert_eq!(
            D32::from_raw(5).checked_mul_pow2(4),
            Some(D32::from_raw(80))
        );
    }

    // ------- per-backing boundaries: MAX, MIN, NEG_ONE -------

    #[test]
    fn i32_boundaries() {
        type T = Decimal<i32, 0>;
        let bits = i32::BITS;

        assert_eq!(T::MAX.checked_mul_pow2(1), None);
        assert_eq!(T::MIN.checked_mul_pow2(1), None);

        // -1 << (BITS-1) == MIN, exact, no overflow.
        assert_eq!(T::from_raw(-1).checked_mul_pow2(bits - 1), Some(T::MIN));
        // 1 << (BITS-1) would land on MIN (reinterpreted), overflow.
        assert_eq!(T::from_raw(1).checked_mul_pow2(bits - 1), None);
        // 1 << (BITS-2) fits as positive signed.
        assert_eq!(
            T::from_raw(1).checked_mul_pow2(bits - 2),
            Some(T::from_raw(i32::MAX / 2 + 1))
        );
    }

    #[test]
    fn i64_boundaries() {
        type T = Decimal<i64, 0>;
        let bits = i64::BITS;

        assert_eq!(T::MAX.checked_mul_pow2(1), None);
        assert_eq!(T::MIN.checked_mul_pow2(1), None);
        assert_eq!(T::from_raw(-1).checked_mul_pow2(bits - 1), Some(T::MIN));
        assert_eq!(T::from_raw(1).checked_mul_pow2(bits - 1), None);
        assert_eq!(
            T::from_raw(1).checked_mul_pow2(bits - 2),
            Some(T::from_raw(i64::MAX / 2 + 1))
        );
    }

    #[test]
    fn i128_boundaries() {
        type T = Decimal<i128, 0>;
        let bits = i128::BITS;

        assert_eq!(T::MAX.checked_mul_pow2(1), None);
        assert_eq!(T::MIN.checked_mul_pow2(1), None);
        assert_eq!(T::from_raw(-1).checked_mul_pow2(bits - 1), Some(T::MIN));
        assert_eq!(T::from_raw(1).checked_mul_pow2(bits - 1), None);
        assert_eq!(
            T::from_raw(1).checked_mul_pow2(bits - 2),
            Some(T::from_raw(i128::MAX / 2 + 1))
        );
    }

    // ------- saturating clamps to MAX / MIN -------

    #[test]
    fn saturating_clamps() {
        assert_eq!(D64::MAX.saturating_mul_pow2(1), D64::MAX);
        assert_eq!(D64::MIN.saturating_mul_pow2(1), D64::MIN);
        assert_eq!(D32::MAX.saturating_mul_pow2(1), D32::MAX);
        assert_eq!(D32::MIN.saturating_mul_pow2(1), D32::MIN);
        assert_eq!(D128::MAX.saturating_mul_pow2(1), D128::MAX);
        assert_eq!(D128::MIN.saturating_mul_pow2(1), D128::MIN);

        // Positive non-overflow case unaffected.
        assert_eq!(D64::from_raw(3).saturating_mul_pow2(2), D64::from_raw(12));
    }

    // ------- wrapping: shift-by-BITS is a no-op (n mod BITS == 0) -------

    #[test]
    fn wrapping_at_bits_boundary() {
        // wrapping_shl masks n mod BITS — shifting by BITS is a no-op.
        assert_eq!(D64::from_raw(7).wrapping_mul_pow2(64), D64::from_raw(7));
        assert_eq!(D32::from_raw(5).wrapping_mul_pow2(32), D32::from_raw(5));
    }

    // ------- try_* mirrors checked_* -------

    #[test]
    fn try_form() {
        assert_eq!(D64::from_raw(1).try_mul_pow2(3), Ok(D64::from_raw(8)));
        assert_eq!(D64::MAX.try_mul_pow2(1), Err(OverflowError));
    }
}

// ============================================================================
// Power-of-2 division (div_pow2)
// ============================================================================

mod div_pow2 {
    use super::{D32, D64, D96, D128};
    use nexus_decimal::Decimal;

    // ------- identity -------

    #[test]
    fn shift_by_zero_is_identity() {
        assert_eq!(D32::ONE.div_pow2(0), D32::ONE);
        assert_eq!(D64::ONE.div_pow2(0), D64::ONE);
        assert_eq!(D96::ONE.div_pow2(0), D96::ONE);
        assert_eq!(D128::ONE.div_pow2(0), D128::ONE);
        assert_eq!(D64::NEG_ONE.div_pow2(0), D64::NEG_ONE);
        assert_eq!(D64::ZERO.div_pow2(0), D64::ZERO);
    }

    // ------- invariant: div_pow2(1) == halve() -------

    #[test]
    fn matches_halve() {
        let cases = [
            D64::ZERO,
            D64::ONE,
            D64::NEG_ONE,
            D64::from_raw(7),
            D64::from_raw(-7),
            D64::from_raw(123_456),
            D64::from_raw(-123_456),
            D64::MAX,
            D64::MIN,
        ];
        for d in cases {
            assert_eq!(
                d.div_pow2(1),
                d.halve(),
                "div_pow2(1) != halve() for raw={}",
                d.to_raw()
            );
        }
    }

    // ------- truncate toward zero -------

    #[test]
    fn truncates_toward_zero() {
        // 7 / 2 = 3 (positive truncation)
        assert_eq!(D64::from_raw(7).div_pow2(1), D64::from_raw(3));
        // -7 / 2 = -3 (truncate toward zero, NOT floor: -7/2 == -3.5 → -3)
        assert_eq!(D64::from_raw(-7).div_pow2(1), D64::from_raw(-3));
        // 15 / 8 = 1; -15 / 8 = -1
        assert_eq!(D64::from_raw(15).div_pow2(3), D64::from_raw(1));
        assert_eq!(D64::from_raw(-15).div_pow2(3), D64::from_raw(-1));
    }

    // ------- per-backing boundaries at n == BITS-1 -------

    #[test]
    fn i32_bits_minus_one() {
        type T = Decimal<i32, 0>;
        let n = i32::BITS - 1;
        // MIN / 2^(BITS-1): mathematically -1.0 exactly, truncates to -1.
        assert_eq!(T::MIN.div_pow2(n), T::from_raw(-1));
        // MAX / 2^(BITS-1): mathematically just under 1.0, truncates to 0.
        assert_eq!(T::MAX.div_pow2(n), T::ZERO);
        // Any non-MIN value truncates to 0.
        assert_eq!(T::from_raw(-1).div_pow2(n), T::ZERO);
        assert_eq!(T::from_raw(1).div_pow2(n), T::ZERO);
        assert_eq!(T::ZERO.div_pow2(n), T::ZERO);
    }

    #[test]
    fn i64_bits_minus_one() {
        type T = Decimal<i64, 0>;
        let n = i64::BITS - 1;
        assert_eq!(T::MIN.div_pow2(n), T::from_raw(-1));
        assert_eq!(T::MAX.div_pow2(n), T::ZERO);
        assert_eq!(T::from_raw(-1).div_pow2(n), T::ZERO);
    }

    #[test]
    fn i128_bits_minus_one() {
        type T = Decimal<i128, 0>;
        let n = i128::BITS - 1;
        assert_eq!(T::MIN.div_pow2(n), T::from_raw(-1));
        assert_eq!(T::MAX.div_pow2(n), T::ZERO);
        assert_eq!(T::from_raw(-1).div_pow2(n), T::ZERO);
    }

    // ------- round-trip mul ↔ div -------

    #[test]
    fn mul_div_roundtrip_when_no_overflow() {
        // For each n where checked_mul_pow2(v, n) succeeds, div_pow2(_, n)
        // recovers v.
        let cases = [
            (D64::from_raw(1), 5),
            (D64::from_raw(-1), 5),
            (D64::from_raw(123_456_789), 10),
            (D64::from_raw(-123_456_789), 10),
            (D64::from_raw(3), 30),
        ];
        for (v, n) in cases {
            let shifted = v.checked_mul_pow2(n).expect("setup: no overflow");
            assert_eq!(
                shifted.div_pow2(n),
                v,
                "round-trip failed for raw={}, n={}",
                v.to_raw(),
                n
            );
        }
    }
}

// ============================================================================
// Absolute difference (checked_abs_diff)
// ============================================================================

mod checked_abs_diff {
    use super::{D32, D64, D96, D128};

    // ------- symmetry -------

    #[test]
    fn symmetric() {
        let pairs = [
            (D64::from_raw(100), D64::from_raw(30)),
            (D64::from_raw(-50), D64::from_raw(20)),
            (D64::from_raw(-100), D64::from_raw(-30)),
            (D64::ZERO, D64::from_raw(7)),
            (D64::MAX, D64::from_raw(1)),
            (D64::MIN, D64::from_raw(-1)),
        ];
        for (a, b) in pairs {
            assert_eq!(
                a.checked_abs_diff(b),
                b.checked_abs_diff(a),
                "asymmetric for ({}, {})",
                a.to_raw(),
                b.to_raw()
            );
        }
    }

    // ------- basic correctness -------

    #[test]
    fn basic_values() {
        assert_eq!(
            D64::from_raw(100).checked_abs_diff(D64::from_raw(30)),
            Some(D64::from_raw(70))
        );
        assert_eq!(
            D64::from_raw(30).checked_abs_diff(D64::from_raw(100)),
            Some(D64::from_raw(70))
        );
        // Opposite signs, no overflow.
        assert_eq!(
            D64::from_raw(50).checked_abs_diff(D64::from_raw(-50)),
            Some(D64::from_raw(100))
        );
    }

    // ------- self-diff is zero -------

    #[test]
    fn self_diff_is_zero() {
        assert_eq!(D32::MAX.checked_abs_diff(D32::MAX), Some(D32::ZERO));
        assert_eq!(D64::MAX.checked_abs_diff(D64::MAX), Some(D64::ZERO));
        assert_eq!(D96::MAX.checked_abs_diff(D96::MAX), Some(D96::ZERO));
        assert_eq!(D128::MAX.checked_abs_diff(D128::MAX), Some(D128::ZERO));
        assert_eq!(D64::MIN.checked_abs_diff(D64::MIN), Some(D64::ZERO));
        assert_eq!(D64::ZERO.checked_abs_diff(D64::ZERO), Some(D64::ZERO));
    }

    // ------- overflow at the rails -------

    #[test]
    fn max_minus_min_overflows() {
        // |MAX - MIN| > MAX on every signed type → None.
        assert_eq!(D32::MAX.checked_abs_diff(D32::MIN), None);
        assert_eq!(D64::MAX.checked_abs_diff(D64::MIN), None);
        assert_eq!(D96::MAX.checked_abs_diff(D96::MIN), None);
        assert_eq!(D128::MAX.checked_abs_diff(D128::MIN), None);
        // Symmetric form.
        assert_eq!(D64::MIN.checked_abs_diff(D64::MAX), None);
    }
}

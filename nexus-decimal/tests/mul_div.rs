//! Multiplication, division, operators, overflow, and round-trip tests.

use nexus_decimal::{Decimal, DivError, OverflowError};

type D32 = Decimal<i32, 4>;
type D64 = Decimal<i64, 8>;
type D96 = Decimal<i128, 12>;
type D128 = Decimal<i128, 18>;
type D0_128 = Decimal<i128, 0>;

// ============================================================================
// Basic multiplication
// ============================================================================

#[test]
fn mul_one_identity_d64() {
    let values = [
        D64::ZERO,
        D64::ONE,
        D64::NEG_ONE,
        D64::new(123, 45_600_000),
        D64::new(-99, 99_999_999),
        D64::MAX,
        D64::MIN,
    ];
    for v in values {
        assert_eq!(v.checked_mul(D64::ONE), Some(v), "v * ONE != v for {v:?}");
    }
}

#[test]
fn mul_zero_d64() {
    let values = [D64::ONE, D64::MAX, D64::MIN, D64::new(42, 0)];
    for v in values {
        assert_eq!(
            v.checked_mul(D64::ZERO),
            Some(D64::ZERO),
            "v * ZERO != ZERO"
        );
    }
}

#[test]
fn mul_basic_d64() {
    // 2.0 * 3.0 = 6.0
    let a = D64::new(2, 0);
    let b = D64::new(3, 0);
    assert_eq!(a.checked_mul(b).unwrap(), D64::new(6, 0));
}

#[test]
fn mul_fractional_d64() {
    // 1.5 * 2.0 = 3.0
    let a = D64::new(1, 50_000_000);
    let b = D64::new(2, 0);
    assert_eq!(a.checked_mul(b).unwrap(), D64::new(3, 0));
}

#[test]
fn mul_negative_d64() {
    // -2.0 * 3.0 = -6.0
    let a = D64::new(-2, 0);
    let b = D64::new(3, 0);
    assert_eq!(a.checked_mul(b).unwrap(), D64::new(-6, 0));

    // -2.0 * -3.0 = 6.0
    let c = D64::new(-3, 0);
    assert_eq!(a.checked_mul(c).unwrap(), D64::new(6, 0));
}

#[test]
fn mul_commutativity_d64() {
    let a = D64::new(123, 45_678_900);
    let b = D64::new(987, 65_432_100);
    assert_eq!(a.checked_mul(b), b.checked_mul(a));
}

#[test]
fn mul_overflow_returns_none_d64() {
    assert!(D64::MAX.checked_mul(D64::new(2, 0)).is_none());
}

// ============================================================================
// Basic division
// ============================================================================

#[test]
fn div_one_identity_d64() {
    let values = [
        D64::ONE,
        D64::NEG_ONE,
        D64::new(123, 45_600_000),
        D64::new(-99, 99_999_999),
    ];
    for v in values {
        assert_eq!(v.checked_div(D64::ONE), Some(v), "v / ONE != v for {v:?}");
    }
}

#[test]
fn div_basic_d64() {
    // 6.0 / 2.0 = 3.0
    let a = D64::new(6, 0);
    let b = D64::new(2, 0);
    assert_eq!(a.checked_div(b).unwrap(), D64::new(3, 0));
}

#[test]
fn div_fractional_d64() {
    // 1.0 / 4.0 = 0.25
    let a = D64::new(1, 0);
    let b = D64::new(4, 0);
    assert_eq!(a.checked_div(b).unwrap(), D64::new(0, 25_000_000));
}

#[test]
fn div_by_zero_returns_none() {
    assert!(D64::ONE.checked_div(D64::ZERO).is_none());
    assert!(D32::ONE.checked_div(D32::ZERO).is_none());
    assert!(D96::ONE.checked_div(D96::ZERO).is_none());
}

#[test]
fn div_negative_d64() {
    // -6.0 / 2.0 = -3.0
    let a = D64::new(-6, 0);
    let b = D64::new(2, 0);
    assert_eq!(a.checked_div(b).unwrap(), D64::new(-3, 0));

    // -6.0 / -2.0 = 3.0
    let c = D64::new(-2, 0);
    assert_eq!(a.checked_div(c).unwrap(), D64::new(3, 0));
}

// ============================================================================
// Operator traits
// ============================================================================

#[test]
fn mul_operator() {
    let a = D64::new(3, 0);
    let b = D64::new(4, 0);
    assert_eq!(a * b, D64::new(12, 0));
}

#[test]
fn div_operator() {
    let a = D64::new(12, 0);
    let b = D64::new(4, 0);
    assert_eq!(a / b, D64::new(3, 0));
}

#[test]
fn mul_assign_operator() {
    let mut a = D64::new(3, 0);
    a *= D64::new(4, 0);
    assert_eq!(a, D64::new(12, 0));
}

#[test]
fn div_assign_operator() {
    let mut a = D64::new(12, 0);
    a /= D64::new(4, 0);
    assert_eq!(a, D64::new(3, 0));
}

#[test]
#[should_panic(expected = "overflow")]
fn mul_operator_overflow_panics() {
    let _ = D64::MAX * D64::new(2, 0);
}

#[test]
#[should_panic(expected = "division")]
fn div_operator_zero_panics() {
    let _ = D64::ONE / D64::ZERO;
}

// ============================================================================
// Saturating and wrapping
// ============================================================================

#[test]
fn saturating_mul_clamps() {
    assert_eq!(D64::MAX.saturating_mul(D64::new(2, 0)), D64::MAX);
    assert_eq!(D64::MIN.saturating_mul(D64::new(2, 0)), D64::MIN);
    assert_eq!(D64::MAX.saturating_mul(D64::new(-2, 0)), D64::MIN);
}

#[test]
fn saturating_div_clamps() {
    // Dividing by a very small number → overflow
    let tiny = D64::from_raw(1); // smallest positive
    assert_eq!(D64::MAX.saturating_div(tiny), D64::MAX);
}

#[test]
#[should_panic(expected = "division by zero")]
fn saturating_div_zero_panics() {
    let _ = D64::ONE.saturating_div(D64::ZERO);
}

// ============================================================================
// Try variants
// ============================================================================

#[test]
fn try_mul_overflow() {
    assert_eq!(D64::MAX.try_mul(D64::new(2, 0)), Err(OverflowError));
}

#[test]
fn try_div_by_zero() {
    assert_eq!(D64::ONE.try_div(D64::ZERO), Err(DivError::DivisionByZero));
}

#[test]
fn try_div_overflow() {
    let tiny = D64::from_raw(1);
    assert_eq!(D64::MAX.try_div(tiny), Err(DivError::Overflow));
}

// ============================================================================
// mul_int
// ============================================================================

#[test]
fn mul_int_basic() {
    let price = D64::new(100, 50_000_000); // 100.50
    assert_eq!(price.mul_int(10), Some(D64::new(1005, 0)));
}

#[test]
fn mul_int_overflow() {
    assert!(D64::MAX.mul_int(2).is_none());
}

#[test]
fn mul_int_zero() {
    assert_eq!(D64::new(42, 0).mul_int(0), Some(D64::ZERO));
}

// ============================================================================
// mul_add
// ============================================================================

#[test]
fn mul_add_basic() {
    // 2.0 * 3.0 + 1.0 = 7.0
    let a = D64::new(2, 0);
    let b = D64::new(3, 0);
    let c = D64::new(1, 0);
    assert_eq!(a.mul_add(b, c), Some(D64::new(7, 0)));
}

// ============================================================================
// Cross-backing-type tests
// ============================================================================

#[test]
fn d32_mul_div() {
    let a = D32::new(10, 5000); // 10.5
    let b = D32::new(2, 0);
    assert_eq!(a.checked_mul(b).unwrap(), D32::new(21, 0));
    assert_eq!(a.checked_div(b).unwrap(), D32::new(5, 2500));
}

#[test]
fn d96_mul_div() {
    let a = D96::new(10, 500_000_000_000); // 10.5
    let b = D96::new(2, 0);
    assert_eq!(a.checked_mul(b).unwrap(), D96::new(21, 0));
    assert_eq!(a.checked_div(b).unwrap(), D96::new(5, 250_000_000_000));
}

#[test]
fn d128_mul_div() {
    let a = D128::new(10, 500_000_000_000_000_000); // 10.5
    let b = D128::new(2, 0);
    assert_eq!(a.checked_mul(b).unwrap(), D128::new(21, 0));
    assert_eq!(
        a.checked_div(b).unwrap(),
        D128::new(5, 250_000_000_000_000_000)
    );
}

#[test]
fn custom_precision_mul() {
    type Usd = Decimal<i64, 2>;
    let price = Usd::new(19, 99); // $19.99
    let qty = Usd::new(3, 0);
    assert_eq!(price.checked_mul(qty).unwrap(), Usd::new(59, 97));
}

// ============================================================================
// Mul/div roundtrip (inverse property)
// ============================================================================

#[test]
fn mul_div_roundtrip_d64() {
    let values = [
        D64::new(100, 0),
        D64::new(1, 50_000_000),
        D64::new(42, 12_345_678),
        D64::new(-99, 99_000_000),
    ];
    let divisors = [
        D64::new(2, 0),
        D64::new(3, 0),
        D64::new(7, 0),
        D64::new(0, 50_000_000),
    ];

    for &v in &values {
        for &d in &divisors {
            if let (Some(product), Some(quotient)) = (v.checked_mul(d), v.checked_div(d)) {
                // (v * d) / d ≈ v — truncation in both mul and div
                // can compound up to d raw units of error
                let recovered = product.checked_div(d);
                if let Some(rec) = recovered {
                    let diff = (rec.to_raw() - v.to_raw()).abs();
                    assert!(
                        diff <= d.to_raw().unsigned_abs() as i64,
                        "roundtrip error too large: v={v:?}, d={d:?}, recovered={rec:?}, diff={diff}"
                    );
                }

                // (v / d) * d ≈ v — same truncation bound
                let recovered2 = quotient.checked_mul(d);
                if let Some(rec2) = recovered2 {
                    let diff2 = (rec2.to_raw() - v.to_raw()).abs();
                    assert!(
                        diff2 <= d.to_raw().unsigned_abs() as i64,
                        "roundtrip error too large: v={v:?}, d={d:?}, recovered={rec2:?}, diff={diff2}"
                    );
                }
            }
        }
    }
}

// ============================================================================
// Extreme values
// ============================================================================

#[test]
fn mul_max_by_one_d64() {
    assert_eq!(D64::MAX.checked_mul(D64::ONE), Some(D64::MAX));
}

#[test]
fn mul_min_by_one_d64() {
    assert_eq!(D64::MIN.checked_mul(D64::ONE), Some(D64::MIN));
}

#[test]
fn mul_max_by_neg_one_d64() {
    // MAX * -1 = -MAX (not MIN, because |MIN| > |MAX| in two's complement)
    let result = D64::MAX.checked_mul(D64::NEG_ONE).unwrap();
    assert_eq!(result.to_raw(), -(i64::MAX));
}

#[test]
fn div_max_by_one_d64() {
    assert_eq!(D64::MAX.checked_div(D64::ONE), Some(D64::MAX));
}

#[test]
fn d96_mul_max_by_one() {
    assert_eq!(D96::MAX.checked_mul(D96::ONE), Some(D96::MAX));
}

#[test]
fn d96_mul_zero() {
    assert_eq!(D96::MAX.checked_mul(D96::ZERO), Some(D96::ZERO));
}

// ============================================================================
// Deterministic random sweep (mul correctness)
// ============================================================================

#[test]
fn mul_correctness_sweep_d64() {
    let mut rng = 42u64;
    let mut next_d64 = || -> D64 {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        // Keep values moderate to avoid overflow in multiplication
        let val = (rng as i64) / 1_000_000;
        D64::from_raw(val)
    };

    for _ in 0..10_000 {
        let a = next_d64();
        let b = next_d64();

        // Compare checked_mul against f64 approximation
        let a_f = a.to_raw() as f64 / D64::SCALE as f64;
        let b_f = b.to_raw() as f64 / D64::SCALE as f64;
        let expected_f = a_f * b_f;

        if let Some(result) = a.checked_mul(b) {
            let result_f = result.to_raw() as f64 / D64::SCALE as f64;
            let diff = (result_f - expected_f).abs();
            // Allow 1 ULP of the decimal + f64 rounding
            let tolerance = 2.0 / D64::SCALE as f64;
            assert!(
                diff < expected_f.abs().mul_add(1e-10, tolerance),
                "mul mismatch: {a:?} * {b:?} = {result:?}, expected ≈ {expected_f}"
            );
        }
    }
}

// ============================================================================
// Const evaluation
// ============================================================================

#[test]
fn const_mul_d64() {
    const A: D64 = D64::new(10, 0);
    const B: D64 = D64::new(5, 0);
    const PRODUCT: D64 = match A.checked_mul(B) {
        Some(v) => v,
        None => panic!("overflow"),
    };
    assert_eq!(PRODUCT, D64::new(50, 0));
}

#[test]
fn const_div_d64() {
    const A: D64 = D64::new(10, 0);
    const B: D64 = D64::new(4, 0);
    const QUOTIENT: D64 = match A.checked_div(B) {
        Some(v) => v,
        None => panic!("overflow"),
    };
    assert_eq!(QUOTIENT, D64::new(2, 50_000_000));
}

// ============================================================================
// Regression: #4 — from_unsigned i128 negation panics for i128::MIN
// ============================================================================

#[test]
fn checked_mul_producing_i128_min() {
    // D=0: SCALE=1, so checked_mul is just a * b / 1 = a * b
    // i128::MIN * 1 = i128::MIN — the quotient is exactly (i128::MAX as u128) + 1,
    // which triggered a panic via -(quotient as i128) in debug mode.
    let min = D0_128::from_raw(i128::MIN);
    let one = D0_128::from_raw(1);
    assert_eq!(min.checked_mul(one), Some(min));
}

#[test]
fn checked_mul_i128_min_by_neg_one() {
    // i128::MIN * -1 = 2^127 which exceeds i128::MAX = 2^127-1
    let min = D0_128::from_raw(i128::MIN);
    let neg_one = D0_128::from_raw(-1);
    assert_eq!(min.checked_mul(neg_one), None);
}

// ============================================================================
// Regression: #2/#3 — wide division with large divisors (>= 2^64)
// ============================================================================

#[test]
fn checked_div_large_divisor_d18() {
    // Decimal<i128, 18>: 1000.0 / 20.0
    // b_raw = 20 * 10^18 = 2 * 10^19 > 2^64, exercises the large-divisor path
    let a = D128::new(1000, 0);
    let b = D128::new(20, 0);
    let result = a.checked_div(b);
    assert_eq!(result, Some(D128::new(50, 0)));
}

#[test]
fn checked_div_large_divisor_d18_fractional() {
    // 100.0 / 33.0 ≈ 3.030303...
    let a = D128::new(100, 0);
    let b = D128::new(33, 0);
    let result = a.checked_div(b).unwrap();
    // 100 / 33 * 10^18 = 3030303030303030303 (truncated)
    assert_eq!(result.to_raw(), 3_030_303_030_303_030_303);
}

#[test]
fn checked_div_by_zero_all_types() {
    assert_eq!(D32::from_raw(100).checked_div(D32::ZERO), None);
    assert_eq!(D64::from_raw(100).checked_div(D64::ZERO), None);
    assert_eq!(D128::from_raw(100).checked_div(D128::ZERO), None);
}

// ============================================================================
// Regression: #34 — Rem by zero
// ============================================================================

#[test]
#[should_panic(expected = "division")]
fn rem_by_zero_panics_i32() {
    let a = D32::from_raw(100);
    let _ = a % D32::ZERO;
}

#[test]
#[should_panic(expected = "division")]
fn rem_by_zero_panics_i64() {
    let a = D64::from_raw(100);
    let _ = a % D64::ZERO;
}

#[test]
#[should_panic(expected = "division")]
fn rem_by_zero_panics_i128() {
    let a = D128::from_raw(100);
    let _ = a % D128::ZERO;
}

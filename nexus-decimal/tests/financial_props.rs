//! Property-based tests for financial method invariants.
//!
//! - Shift identity (shift by 0 == self)
//! - Self-diff is zero
//! - within_bps reflexivity
//! - Rounding tick-alignment
//! - Floor/ceil bracket
//! - Floor ≤ round ≤ ceil
//! - add_ticks/tick_diff roundtrip
//! - Cross-backing financial equivalence (i32 vs i64)

use nexus_decimal::Decimal;
use proptest::prelude::*;

type D32 = Decimal<i32, 4>;
type D64 = Decimal<i64, 8>;
type D128 = Decimal<i128, 18>;
type D64_4 = Decimal<i64, 4>;

proptest! {
    // ------- shift identity -------

    #[test]
    fn shift_bps_identity_i32(raw: i32) {
        let a = D32::from_raw(raw);
        prop_assert_eq!(a.shift_bps(0), Some(a));
    }

    #[test]
    fn shift_bps_identity_i64(raw: i64) {
        let a = D64::from_raw(raw);
        prop_assert_eq!(a.shift_bps(0), Some(a));
    }

    #[test]
    fn shift_bps_identity_i128(raw: i128) {
        let a = D128::from_raw(raw);
        prop_assert_eq!(a.shift_bps(0), Some(a));
    }

    #[test]
    fn shift_pct_identity_i32(raw: i32) {
        let a = D32::from_raw(raw);
        prop_assert_eq!(a.shift_pct(0), Some(a));
    }

    #[test]
    fn shift_pct_identity_i64(raw: i64) {
        let a = D64::from_raw(raw);
        prop_assert_eq!(a.shift_pct(0), Some(a));
    }

    #[test]
    fn shift_pct_identity_i128(raw: i128) {
        let a = D128::from_raw(raw);
        prop_assert_eq!(a.shift_pct(0), Some(a));
    }

    // ------- self-diff is zero -------

    #[test]
    fn bps_diff_self_is_zero_i64(raw in 1i64..=i64::MAX) {
        let a = D64::from_raw(raw);
        prop_assert_eq!(a.bps_diff(a), Some(D64::ZERO));
    }

    #[test]
    fn bps_diff_self_is_zero_i128(raw in 1i128..=i128::MAX) {
        let a = D128::from_raw(raw);
        prop_assert_eq!(a.bps_diff(a), Some(D128::ZERO));
    }

    // ------- within_bps reflexive -------

    #[test]
    fn within_bps_reflexive_i64(raw: i64, bps in 0i32..=10000) {
        let a = D64::from_raw(raw);
        prop_assert!(a.within_bps(a, bps));
    }

    #[test]
    fn within_bps_reflexive_i128(raw: i128, bps in 0i32..=10000) {
        let a = D128::from_raw(raw);
        prop_assert!(a.within_bps(a, bps));
    }

    // ------- rounding tick-alignment -------

    #[test]
    fn round_to_tick_aligned_i64(raw: i64, tick_raw in 1i64..=10000i64) {
        let v = D64::from_raw(raw);
        let tick = D64::from_raw(tick_raw);
        if let Some(rounded) = v.round_to_tick(tick) {
            prop_assert_eq!(rounded.to_raw() % tick_raw, 0);
        }
    }

    #[test]
    fn round_to_tick_aligned_i128(raw: i128, tick_raw in 1i128..=10000i128) {
        let v = D128::from_raw(raw);
        let tick = D128::from_raw(tick_raw);
        if let Some(rounded) = v.round_to_tick(tick) {
            prop_assert_eq!(rounded.to_raw() % tick_raw, 0);
        }
    }

    // ------- floor/ceil bracket -------

    #[test]
    fn floor_ceil_bracket_i64(raw: i64, tick_raw in 1i64..=10000i64) {
        let v = D64::from_raw(raw);
        let tick = D64::from_raw(tick_raw);
        if let (Some(floor), Some(ceil)) = (v.floor_to_tick(tick), v.ceil_to_tick(tick)) {
            prop_assert!(floor.to_raw() <= raw, "floor must be <= original");
            prop_assert!(ceil.to_raw() >= raw, "ceil must be >= original");
        }
    }

    #[test]
    fn floor_ceil_bracket_i128(raw: i128, tick_raw in 1i128..=10000i128) {
        let v = D128::from_raw(raw);
        let tick = D128::from_raw(tick_raw);
        if let (Some(floor), Some(ceil)) = (v.floor_to_tick(tick), v.ceil_to_tick(tick)) {
            prop_assert!(floor.to_raw() <= raw, "floor must be <= original");
            prop_assert!(ceil.to_raw() >= raw, "ceil must be >= original");
        }
    }

    // ------- floor ≤ round ≤ ceil -------

    #[test]
    fn floor_le_round_le_ceil_i64(raw: i64, tick_raw in 1i64..=10000i64) {
        let v = D64::from_raw(raw);
        let tick = D64::from_raw(tick_raw);
        if let (Some(floor), Some(round), Some(ceil)) = (
            v.floor_to_tick(tick),
            v.round_to_tick(tick),
            v.ceil_to_tick(tick),
        ) {
            prop_assert!(floor.to_raw() <= round.to_raw(), "floor <= round");
            prop_assert!(round.to_raw() <= ceil.to_raw(), "round <= ceil");
        }
    }

    #[test]
    fn floor_le_round_le_ceil_i128(raw: i128, tick_raw in 1i128..=10000i128) {
        let v = D128::from_raw(raw);
        let tick = D128::from_raw(tick_raw);
        if let (Some(floor), Some(round), Some(ceil)) = (
            v.floor_to_tick(tick),
            v.round_to_tick(tick),
            v.ceil_to_tick(tick),
        ) {
            prop_assert!(floor.to_raw() <= round.to_raw(), "floor <= round");
            prop_assert!(round.to_raw() <= ceil.to_raw(), "round <= ceil");
        }
    }

    // ------- add_ticks/tick_diff roundtrip -------

    #[test]
    fn add_ticks_tick_diff_roundtrip_i64(
        raw: i64,
        n in -1000i64..=1000i64,
        tick_raw in 1i64..=10000i64,
    ) {
        let v = D64::from_raw(raw);
        let tick = D64::from_raw(tick_raw);
        if let Some(shifted) = v.add_ticks(n, tick)
            && let Some(diff) = shifted.tick_diff(v, tick)
        {
            prop_assert_eq!(diff, n);
        }
    }

    #[test]
    fn add_ticks_tick_diff_roundtrip_i128(
        raw: i128,
        n in -1000i64..=1000i64,
        tick_raw in 1i128..=10000i128,
    ) {
        let v = D128::from_raw(raw);
        let tick = D128::from_raw(tick_raw);
        if let Some(shifted) = v.add_ticks(n, tick)
            && let Some(diff) = shifted.tick_diff(v, tick)
        {
            prop_assert_eq!(diff, n);
        }
    }

    // ------- cross-backing financial (i32 vs i64 at D=4) -------

    #[test]
    fn cross_backing_bps_of(raw: i32, bps in -5000i32..=5000i32) {
        let v32 = D32::from_raw(raw);
        let v64 = D64_4::from_raw(raw as i64);
        if let Some(r32) = v32.bps_of(bps) {
            let r64 = v64.bps_of(bps).unwrap();
            prop_assert_eq!(r32.to_raw() as i64, r64.to_raw());
        }
    }

    #[test]
    fn cross_backing_shift_bps(raw: i32, bps in -5000i32..=5000i32) {
        let v32 = D32::from_raw(raw);
        let v64 = D64_4::from_raw(raw as i64);
        if let Some(r32) = v32.shift_bps(bps) {
            let r64 = v64.shift_bps(bps).unwrap();
            prop_assert_eq!(r32.to_raw() as i64, r64.to_raw());
        }
    }

    #[test]
    fn cross_backing_within_bps(a: i32, b: i32, bps in 0i32..=5000i32) {
        let a32 = D32::from_raw(a);
        let b32 = D32::from_raw(b);
        let a64 = D64_4::from_raw(a as i64);
        let b64 = D64_4::from_raw(b as i64);
        prop_assert_eq!(a32.within_bps(b32, bps), a64.within_bps(b64, bps));
    }
}

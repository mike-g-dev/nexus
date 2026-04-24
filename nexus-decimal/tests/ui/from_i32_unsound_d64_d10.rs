// i32 is unsound on Decimal<i64, 10>: i32::MAX × 10^10 > i64::MAX is true,
// and i32::MAX × 10^10 = 2.15e19 > i64::MAX (9.22e18) → unsound.
// `From<i32>` is not emitted at this (backing, D), so .into() must fail.
//
// (`TryFrom<i32>` is also not emitted — i32 is not in the i64/u64 unsound
// list. Caller must widen explicitly via `from_i32` etc.)

use nexus_decimal::Decimal;

type BadD = Decimal<i64, 10>;

fn main() {
    let _: BadD = 5_i32.into();
}

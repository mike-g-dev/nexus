//! Rounding operations for `Decimal`.
//!
//! All methods are `const fn`, generated per backing type via macro.

use crate::Decimal;

macro_rules! impl_decimal_rounding {
    ($backing:ty, $pow10_fn:path) => {
        impl<const D: u8> Decimal<$backing, D> {
            /// Rounds toward negative infinity.
            ///
            /// # Examples
            ///
            /// ```
            /// use nexus_decimal::Decimal;
            /// type D64 = Decimal<i64, 8>;
            ///
            /// let pos = D64::new(1, 75_000_000); // 1.75
            /// assert_eq!(pos.floor().to_raw(), D64::new(1, 0).to_raw());
            ///
            /// let neg = D64::new(-1, 75_000_000); // -1.75
            /// assert_eq!(neg.floor().to_raw(), D64::new(-2, 0).to_raw());
            /// ```
            #[inline(always)]
            pub const fn floor(self) -> Self {
                let remainder = self.value % Self::SCALE;
                if remainder >= 0 {
                    Self {
                        value: self.value - remainder,
                    }
                } else {
                    // self.value - remainder gives next integer toward zero.
                    // Subtract SCALE to go one step negative. Saturate on underflow.
                    let toward_zero = self.value - remainder;
                    match toward_zero.checked_sub(Self::SCALE) {
                        Some(v) => Self { value: v },
                        None => Self::MIN,
                    }
                }
            }

            /// Rounds toward positive infinity.
            ///
            /// # Examples
            ///
            /// ```
            /// use nexus_decimal::Decimal;
            /// type D64 = Decimal<i64, 8>;
            ///
            /// let pos = D64::new(1, 25_000_000); // 1.25
            /// assert_eq!(pos.ceil().to_raw(), D64::new(2, 0).to_raw());
            ///
            /// let neg = D64::new(-1, 25_000_000); // -1.25
            /// assert_eq!(neg.ceil().to_raw(), D64::new(-1, 0).to_raw());
            /// ```
            #[inline(always)]
            pub const fn ceil(self) -> Self {
                let remainder = self.value % Self::SCALE;
                if remainder > 0 {
                    let toward_zero = self.value - remainder;
                    match toward_zero.checked_add(Self::SCALE) {
                        Some(v) => Self { value: v },
                        None => Self::MAX,
                    }
                } else {
                    Self {
                        value: self.value - remainder,
                    }
                }
            }

            /// Truncates toward zero (removes fractional part).
            ///
            /// # Examples
            ///
            /// ```
            /// use nexus_decimal::Decimal;
            /// type D64 = Decimal<i64, 8>;
            ///
            /// let pos = D64::new(1, 99_000_000); // 1.99
            /// assert_eq!(pos.trunc().to_raw(), D64::new(1, 0).to_raw());
            ///
            /// let neg = D64::new(-1, 99_000_000); // -1.99
            /// assert_eq!(neg.trunc().to_raw(), D64::new(-1, 0).to_raw());
            /// ```
            #[inline(always)]
            pub const fn trunc(self) -> Self {
                Self {
                    value: (self.value / Self::SCALE) * Self::SCALE,
                }
            }

            /// Returns the fractional part (same sign as `self`).
            ///
            /// Invariant: `self == self.trunc() + self.fract()`.
            #[inline(always)]
            pub const fn fract(self) -> Self {
                Self {
                    value: self.value % Self::SCALE,
                }
            }

            /// Returns the integer part as the backing type.
            ///
            /// Equivalent to `self.trunc().to_raw() / SCALE`.
            #[inline(always)]
            pub const fn to_integer(self) -> $backing {
                self.value / Self::SCALE
            }

            /// Rounds to the nearest integer using banker's rounding
            /// (round half to even).
            ///
            /// # Examples
            ///
            /// ```
            /// use nexus_decimal::Decimal;
            /// type D64 = Decimal<i64, 8>;
            ///
            /// // Half rounds to even
            /// let half_even = D64::new(2, 50_000_000); // 2.5
            /// assert_eq!(half_even.round().to_raw(), D64::new(2, 0).to_raw());
            ///
            /// let half_odd = D64::new(3, 50_000_000); // 3.5
            /// assert_eq!(half_odd.round().to_raw(), D64::new(4, 0).to_raw());
            /// ```
            #[inline(always)]
            pub const fn round(self) -> Self {
                let quotient = self.value / Self::SCALE;
                let remainder = self.value % Self::SCALE;
                let half = Self::SCALE / 2;

                let rounded = if remainder > half {
                    quotient + 1
                } else if remainder < -half {
                    quotient - 1
                } else if remainder == half {
                    // Banker's rounding: round to even
                    if quotient & 1 != 0 {
                        quotient + 1
                    } else {
                        quotient
                    }
                } else if remainder == -half {
                    if quotient & 1 != 0 {
                        quotient - 1
                    } else {
                        quotient
                    }
                } else {
                    quotient
                };

                match rounded.checked_mul(Self::SCALE) {
                    Some(v) => Self { value: v },
                    None => {
                        if rounded > 0 {
                            Self::MAX
                        } else {
                            Self::MIN
                        }
                    }
                }
            }

            /// Rounds to `dp` decimal places using banker's rounding.
            ///
            /// # Panics
            ///
            /// Panics if `dp >= DECIMALS`.
            ///
            /// # Examples
            ///
            /// ```
            /// use nexus_decimal::Decimal;
            /// type D64 = Decimal<i64, 8>;
            ///
            /// let price = D64::new(1, 23_456_789); // 1.23456789
            /// let rounded = price.round_dp(2);       // 1.23
            /// assert_eq!(rounded.to_raw(), D64::new(1, 23_000_000).to_raw());
            /// ```
            #[inline]
            pub const fn round_dp(self, dp: u8) -> Self {
                assert!(dp < D, "round_dp: dp must be less than DECIMALS");

                let sub_scale = $pow10_fn(D - dp);
                let half = sub_scale / 2;
                let quotient = self.value / sub_scale;
                let remainder = self.value % sub_scale;

                let rounded = if remainder > half {
                    quotient + 1
                } else if remainder < -half {
                    quotient - 1
                } else if remainder == half {
                    if quotient & 1 != 0 {
                        quotient + 1
                    } else {
                        quotient
                    }
                } else if remainder == -half {
                    if quotient & 1 != 0 {
                        quotient - 1
                    } else {
                        quotient
                    }
                } else {
                    quotient
                };

                match rounded.checked_mul(sub_scale) {
                    Some(v) => Self { value: v },
                    None => {
                        if rounded > 0 {
                            Self::MAX
                        } else {
                            Self::MIN
                        }
                    }
                }
            }
        }
    };
}

use crate::pow10::{pow10_i32, pow10_i64, pow10_i128};

impl_decimal_rounding!(i32, pow10_i32);
impl_decimal_rounding!(i64, pow10_i64);
impl_decimal_rounding!(i128, pow10_i128);

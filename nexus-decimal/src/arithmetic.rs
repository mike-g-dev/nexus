//! Checked, saturating, wrapping, and try arithmetic for `Decimal`.
//!
//! Add/Sub/Neg/Abs are shared via macro.
//! Mul/Div differ per backing type:
//! - i32: widen to i64, native division (LLVM magic multiply)
//! - i64: widen to i128, chunked magic division for SCALE < 2^32
//!   (3× u64 magic multiplies, ~14 cycles), native fallback otherwise
//! - i128: 192-bit wide arithmetic (manual limb math)

use crate::Decimal;
use crate::error::{DivError, OverflowError};

// ============================================================================
// Add / Sub / Neg / Abs — shared across all backing types
// ============================================================================

macro_rules! impl_decimal_arithmetic {
    ($backing:ty) => {
        impl<const D: u8> Decimal<$backing, D> {
            // ========================================================
            // Default semantics (panic on overflow in debug, wrap in release)
            // ========================================================

            /// Computes the absolute value of `self`.
            ///
            /// # Overflow behavior
            ///
            /// The absolute value of `Self::MIN` cannot be represented as
            /// a `Self`, and attempting to calculate it will cause an
            /// overflow. This means that code in debug mode will trigger
            /// a panic on this case and optimized code will return
            /// `Self::MIN` without a panic.
            ///
            /// Matches the semantics of `<backing>::abs`. Use
            /// [`checked_abs`](Self::checked_abs),
            /// [`saturating_abs`](Self::saturating_abs), or
            /// [`wrapping_abs`](Self::wrapping_abs) for explicit overflow
            /// policies.
            #[inline(always)]
            pub const fn abs(self) -> Self {
                Self {
                    value: self.value.abs(),
                }
            }

            // ========================================================
            // Checked
            // ========================================================

            /// Checked addition. Returns `None` on overflow.
            #[inline(always)]
            pub const fn checked_add(self, rhs: Self) -> Option<Self> {
                match self.value.checked_add(rhs.value) {
                    Some(v) => Some(Self { value: v }),
                    None => None,
                }
            }

            /// Checked subtraction. Returns `None` on overflow.
            #[inline(always)]
            pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
                match self.value.checked_sub(rhs.value) {
                    Some(v) => Some(Self { value: v }),
                    None => None,
                }
            }

            /// Checked negation. Returns `None` if `self == MIN`.
            #[inline(always)]
            pub const fn checked_neg(self) -> Option<Self> {
                match self.value.checked_neg() {
                    Some(v) => Some(Self { value: v }),
                    None => None,
                }
            }

            /// Checked absolute value. Returns `None` if `self == MIN`.
            #[inline(always)]
            pub const fn checked_abs(self) -> Option<Self> {
                if self.value >= 0 {
                    Some(self)
                } else {
                    self.checked_neg()
                }
            }

            // ========================================================
            // Saturating
            // ========================================================

            /// Saturating addition. Clamps to `MIN`/`MAX` on overflow.
            #[inline(always)]
            pub const fn saturating_add(self, rhs: Self) -> Self {
                Self {
                    value: self.value.saturating_add(rhs.value),
                }
            }

            /// Saturating subtraction.
            #[inline(always)]
            pub const fn saturating_sub(self, rhs: Self) -> Self {
                Self {
                    value: self.value.saturating_sub(rhs.value),
                }
            }

            /// Saturating negation.
            #[inline(always)]
            pub const fn saturating_neg(self) -> Self {
                Self {
                    value: self.value.saturating_neg(),
                }
            }

            /// Saturating absolute value.
            #[inline(always)]
            pub const fn saturating_abs(self) -> Self {
                Self {
                    value: self.value.saturating_abs(),
                }
            }

            // ========================================================
            // Wrapping
            // ========================================================

            /// Wrapping addition.
            #[inline(always)]
            pub const fn wrapping_add(self, rhs: Self) -> Self {
                Self {
                    value: self.value.wrapping_add(rhs.value),
                }
            }

            /// Wrapping subtraction.
            #[inline(always)]
            pub const fn wrapping_sub(self, rhs: Self) -> Self {
                Self {
                    value: self.value.wrapping_sub(rhs.value),
                }
            }

            /// Wrapping negation.
            #[inline(always)]
            pub const fn wrapping_neg(self) -> Self {
                Self {
                    value: self.value.wrapping_neg(),
                }
            }

            /// Wrapping absolute value.
            #[inline(always)]
            pub const fn wrapping_abs(self) -> Self {
                Self {
                    value: self.value.wrapping_abs(),
                }
            }

            // ========================================================
            // Try (Result-returning) — add/sub/neg/abs
            // ========================================================

            /// Addition returning `Result`.
            #[inline(always)]
            pub const fn try_add(self, rhs: Self) -> Result<Self, OverflowError> {
                match self.checked_add(rhs) {
                    Some(v) => Ok(v),
                    None => Err(OverflowError),
                }
            }

            /// Subtraction returning `Result`.
            #[inline(always)]
            pub const fn try_sub(self, rhs: Self) -> Result<Self, OverflowError> {
                match self.checked_sub(rhs) {
                    Some(v) => Ok(v),
                    None => Err(OverflowError),
                }
            }

            /// Negation returning `Result`.
            #[inline(always)]
            pub const fn try_neg(self) -> Result<Self, OverflowError> {
                match self.checked_neg() {
                    Some(v) => Ok(v),
                    None => Err(OverflowError),
                }
            }

            /// Absolute value returning `Result`.
            #[inline(always)]
            pub const fn try_abs(self) -> Result<Self, OverflowError> {
                match self.checked_abs() {
                    Some(v) => Ok(v),
                    None => Err(OverflowError),
                }
            }

            // ========================================================
            // Power-of-2 multiplication
            // ========================================================

            /// Multiply by `2^n` (left shift on the backing value).
            ///
            /// The `10^D` scale factor cancels because the multiplier is
            /// dimensionless — multiplying the represented value by `2^n` is
            /// exactly a left shift on the backing.
            ///
            /// # Overflow behavior
            ///
            /// Matches `<backing>::mul` semantics: debug builds panic on
            /// overflow, release builds wrap (`wrapping_shl`, which masks
            /// `n` to `n mod <backing>::BITS`). Use
            /// [`checked_mul_pow2`](Self::checked_mul_pow2),
            /// [`saturating_mul_pow2`](Self::saturating_mul_pow2), or
            /// [`wrapping_mul_pow2`](Self::wrapping_mul_pow2) for explicit
            /// overflow policies.
            ///
            /// In particular, `mul_pow2(v, BITS)` in release returns `v`
            /// unchanged — the mask makes this a no-op, not a zeroing.
            ///
            /// # Codegen
            ///
            /// Lowers to a single backing-width shift in release builds
            /// (both constant and variable `n`). For `i32` / `i64`
            /// backings this is one instruction (~1 cycle); for `i128`
            /// it expands to a branchless wide-shift sequence (`shld` +
            /// `shl` on x86-64, ~4-5 cycles).
            #[inline(always)]
            pub const fn mul_pow2(self, n: u32) -> Self {
                // `cfg!()` is const-evaluable; the unused branch is removed.
                if cfg!(debug_assertions) {
                    match self.checked_mul_pow2(n) {
                        Some(v) => v,
                        None => panic!("attempt to multiply with overflow"),
                    }
                } else {
                    Self {
                        value: self.value.wrapping_shl(n),
                    }
                }
            }

            /// Checked multiplication by `2^n`. Returns `None` on overflow.
            ///
            /// Uses leading-zero counting to detect overflow without
            /// performing the shift first. For positive `v`, requires
            /// `n < v.leading_zeros()`; for negative `v`, requires
            /// `n < (!v).leading_zeros()`.
            #[inline(always)]
            pub const fn checked_mul_pow2(self, n: u32) -> Option<Self> {
                if self.value == 0 {
                    return Some(self);
                }
                let leading_sign_bits = if self.value >= 0 {
                    self.value.leading_zeros()
                } else {
                    (!self.value).leading_zeros()
                };
                if n < leading_sign_bits {
                    Some(Self {
                        value: self.value.wrapping_shl(n),
                    })
                } else {
                    None
                }
            }

            /// Saturating multiplication by `2^n`. Clamps to
            /// [`MAX`](Self::MAX) / [`MIN`](Self::MIN) on overflow.
            #[inline(always)]
            pub const fn saturating_mul_pow2(self, n: u32) -> Self {
                match self.checked_mul_pow2(n) {
                    Some(v) => v,
                    None => {
                        if self.value >= 0 {
                            Self::MAX
                        } else {
                            Self::MIN
                        }
                    }
                }
            }

            /// Wrapping multiplication by `2^n`.
            ///
            /// Silently wraps on overflow. Note that `wrapping_shl` masks
            /// `n` to `n mod <backing>::BITS`, so e.g. shifting by `BITS`
            /// is a no-op rather than zeroing the value.
            #[inline(always)]
            pub const fn wrapping_mul_pow2(self, n: u32) -> Self {
                Self {
                    value: self.value.wrapping_shl(n),
                }
            }

            /// Multiplication by `2^n` returning `Result`.
            #[inline(always)]
            pub const fn try_mul_pow2(self, n: u32) -> Result<Self, OverflowError> {
                match self.checked_mul_pow2(n) {
                    Some(v) => Ok(v),
                    None => Err(OverflowError),
                }
            }

            // ========================================================
            // Power-of-2 division
            // ========================================================

            /// Divide by `2^n` (truncate toward zero).
            ///
            /// Semantically identical to `/ 2^n`: truncates toward zero,
            /// matching [`halve`](Self::halve), [`div10`](Self::div10),
            /// [`div100`](Self::div100), and the rest of the division
            /// surface. Invariant: `div_pow2(1) == halve()`.
            ///
            /// # Codegen
            ///
            /// Constant `n` folds to a branchless shift + sign-correction
            /// sequence (~2 cycles on modern x86-64). Variable `n`
            /// compiles to a hardware signed division (~8-12 cycles on
            /// Ice Lake+ / Zen 3+) — use a constant when the shift
            /// amount is known.
            ///
            /// # Panics
            ///
            /// Debug builds panic if `n >= <backing>::BITS`. Release
            /// builds return [`ZERO`](Self::ZERO), which is the
            /// mathematically correct result under truncate-toward-zero:
            /// any value divided by `2^n` larger than its magnitude is 0.
            #[inline(always)]
            pub const fn div_pow2(self, n: u32) -> Self {
                debug_assert!(n < <$backing>::BITS, "shift amount out of range");

                if n >= <$backing>::BITS {
                    // Release-mode safety net.
                    return Self { value: 0 };
                }
                if n == <$backing>::BITS - 1 {
                    // 2^(BITS-1) doesn't fit as positive signed.
                    // value / 2^(BITS-1) truncated toward zero:
                    //   value == MIN → -1; otherwise → 0
                    return Self {
                        value: if self.value == <$backing>::MIN { -1 } else { 0 },
                    };
                }
                // n < BITS - 1: (1 << n) fits as positive signed.
                Self {
                    value: self.value / (1 << n),
                }
            }

            // ========================================================
            // Absolute difference
            // ========================================================

            /// Overflow-safe absolute difference: `|self - other|`.
            ///
            /// Returns `None` when the result would exceed
            /// [`MAX`](Self::MAX) — this happens when the operands have
            /// opposite signs near the rails, since `|MIN - MAX|` exceeds
            /// `MAX` on every signed type.
            ///
            /// Named `checked_abs_diff` to match the crate's `checked_*`
            /// convention for `Option`-returning operations. There is no
            /// bare `abs_diff` — every call site must acknowledge the
            /// overflow case. Stdlib's `<backing>::abs_diff` returns an
            /// unsigned type to avoid overflow; since `Decimal` has no
            /// unsigned variant, this returns `Option<Self>` instead.
            #[inline(always)]
            pub const fn checked_abs_diff(self, other: Self) -> Option<Self> {
                let diff = if self.value >= other.value {
                    self.value.checked_sub(other.value)
                } else {
                    other.value.checked_sub(self.value)
                };
                match diff {
                    Some(v) => Some(Self { value: v }),
                    None => None,
                }
            }
        }
    };
}

impl_decimal_arithmetic!(i32);
impl_decimal_arithmetic!(i64);
impl_decimal_arithmetic!(i128);

// ============================================================================
// Mul / Div — i32 (widen to i64, native division)
// ============================================================================

impl<const D: u8> Decimal<i32, D> {
    /// Checked multiplication. Widens to i64, divides by SCALE.
    #[inline(always)]
    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        // i32 * i32 always fits in i64 — no overflow possible
        let product = (self.value as i64) * (rhs.value as i64);
        let result = product / (Self::SCALE as i64);

        if result > i32::MAX as i64 || result < i32::MIN as i64 {
            None
        } else {
            Some(Self {
                value: result as i32,
            })
        }
    }

    /// Checked division. Returns `None` if `rhs` is zero or result overflows.
    #[inline(always)]
    pub const fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.value == 0 {
            return None;
        }
        let a = self.value as i64;
        let b = rhs.value as i64;
        let result = (a * Self::SCALE as i64) / b;

        if result > i32::MAX as i64 || result < i32::MIN as i64 {
            None
        } else {
            Some(Self {
                value: result as i32,
            })
        }
    }

    /// Saturating multiplication.
    #[inline(always)]
    pub const fn saturating_mul(self, rhs: Self) -> Self {
        let product = (self.value as i64) * (rhs.value as i64);
        let result = product / (Self::SCALE as i64);

        if result > i32::MAX as i64 {
            Self::MAX
        } else if result < i32::MIN as i64 {
            Self::MIN
        } else {
            Self {
                value: result as i32,
            }
        }
    }

    /// Wrapping multiplication.
    #[inline(always)]
    pub const fn wrapping_mul(self, rhs: Self) -> Self {
        let product = (self.value as i64) * (rhs.value as i64);
        Self {
            value: (product / (Self::SCALE as i64)) as i32,
        }
    }

    /// Saturating division.
    #[inline(always)]
    pub const fn saturating_div(self, rhs: Self) -> Self {
        assert!(rhs.value != 0, "division by zero");
        match self.checked_div(rhs) {
            Some(v) => v,
            None => {
                if (self.value > 0) == (rhs.value > 0) {
                    Self::MAX
                } else {
                    Self::MIN
                }
            }
        }
    }

    /// Wrapping division.
    #[inline(always)]
    pub const fn wrapping_div(self, rhs: Self) -> Self {
        assert!(rhs.value != 0, "division by zero");
        let a = self.value as i64;
        let b = rhs.value as i64;
        Self {
            value: ((a * Self::SCALE as i64) / b) as i32,
        }
    }

    /// Multiply by a plain integer (no rescaling).
    #[inline(always)]
    pub const fn mul_int(self, rhs: i32) -> Option<Self> {
        match self.value.checked_mul(rhs) {
            Some(v) => Some(Self { value: v }),
            None => None,
        }
    }

    /// Fused multiply-add: `(self * mul) + add` with single rescaling.
    #[inline(always)]
    pub const fn mul_add(self, mul: Self, add: Self) -> Option<Self> {
        let product = (self.value as i64) * (mul.value as i64);
        let rescaled = product / (Self::SCALE as i64);
        let result = rescaled + (add.value as i64);

        if result > i32::MAX as i64 || result < i32::MIN as i64 {
            None
        } else {
            Some(Self {
                value: result as i32,
            })
        }
    }

    /// Multiplication returning `Result`.
    #[inline(always)]
    pub const fn try_mul(self, rhs: Self) -> Result<Self, OverflowError> {
        match self.checked_mul(rhs) {
            Some(v) => Ok(v),
            None => Err(OverflowError),
        }
    }

    /// Division returning `Result` with specific error.
    #[inline(always)]
    pub const fn try_div(self, rhs: Self) -> Result<Self, DivError> {
        if rhs.value == 0 {
            return Err(DivError::DivisionByZero);
        }
        match self.checked_div(rhs) {
            Some(v) => Ok(v),
            None => Err(DivError::Overflow),
        }
    }
}

// ============================================================================
// Mul / Div — i64 (widen to i128, chunked magic division when SCALE < 2^32)
// ============================================================================
//
// When SCALE < 2^32 (covers Decimal<i64, 1..=9>), uses chunked
// u64 division (~14 cycles, 3 magic multiplies). Otherwise falls back to
// native i128 division (__divti3 ~25 cycles). The const branch is
// eliminated by LLVM — zero runtime cost for type selection.

use crate::div_by_scale;

impl<const D: u8> Decimal<i64, D> {
    /// Whether this (i64, D) combination qualifies for chunked fast path.
    const USE_CHUNKED: bool = (Self::SCALE as u64) < div_by_scale::CHUNK_THRESHOLD;

    /// Divide an i128 product by SCALE, using the fast path when available.
    #[inline(always)]
    const fn div_product_by_scale(product: i128) -> Option<i64> {
        div_by_scale::div_i128_by_scale(
            product,
            Self::SCALE as i128,
            Self::SCALE as u64,
            Self::USE_CHUNKED,
        )
    }

    /// Wrapping version of SCALE division.
    #[inline(always)]
    const fn div_product_by_scale_wrapping(product: i128) -> i64 {
        div_by_scale::div_i128_by_scale_wrapping(
            product,
            Self::SCALE as i128,
            Self::SCALE as u64,
            Self::USE_CHUNKED,
        )
    }

    /// Checked multiplication. Widens to i128, divides by SCALE.
    #[inline(always)]
    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        let a = self.value as i128;
        let b = rhs.value as i128;

        let Some(product) = a.checked_mul(b) else {
            return None;
        };

        match Self::div_product_by_scale(product) {
            Some(result) => Some(Self { value: result }),
            None => None,
        }
    }

    /// Checked division. Returns `None` if `rhs` is zero or result overflows.
    ///
    /// Division by a runtime value cannot use the chunked path — the
    /// divisor isn't a compile-time constant. Uses native i128 division.
    #[inline(always)]
    pub const fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.value == 0 {
            return None;
        }
        let a = self.value as i128;
        let b = rhs.value as i128;
        let result = (a * Self::SCALE as i128) / b;

        if result > i64::MAX as i128 || result < i64::MIN as i128 {
            None
        } else {
            Some(Self {
                value: result as i64,
            })
        }
    }

    /// Saturating multiplication.
    #[inline(always)]
    pub const fn saturating_mul(self, rhs: Self) -> Self {
        let product = (self.value as i128) * (rhs.value as i128);
        match Self::div_product_by_scale(product) {
            Some(result) => Self { value: result },
            None => {
                if product > 0 {
                    Self::MAX
                } else {
                    Self::MIN
                }
            }
        }
    }

    /// Wrapping multiplication.
    #[inline(always)]
    pub const fn wrapping_mul(self, rhs: Self) -> Self {
        let product = (self.value as i128).wrapping_mul(rhs.value as i128);
        Self {
            value: Self::div_product_by_scale_wrapping(product),
        }
    }

    /// Saturating division.
    #[inline(always)]
    pub const fn saturating_div(self, rhs: Self) -> Self {
        assert!(rhs.value != 0, "division by zero");
        match self.checked_div(rhs) {
            Some(v) => v,
            None => {
                if (self.value > 0) == (rhs.value > 0) {
                    Self::MAX
                } else {
                    Self::MIN
                }
            }
        }
    }

    /// Wrapping division.
    #[inline(always)]
    pub const fn wrapping_div(self, rhs: Self) -> Self {
        assert!(rhs.value != 0, "division by zero");
        let a = self.value as i128;
        let b = rhs.value as i128;
        Self {
            value: ((a * Self::SCALE as i128) / b) as i64,
        }
    }

    /// Multiply by a plain integer (no rescaling).
    #[inline(always)]
    pub const fn mul_int(self, rhs: i64) -> Option<Self> {
        match self.value.checked_mul(rhs) {
            Some(v) => Some(Self { value: v }),
            None => None,
        }
    }

    /// Fused multiply-add: `(self * mul) + add` with single rescaling.
    #[inline(always)]
    pub const fn mul_add(self, mul: Self, add: Self) -> Option<Self> {
        let a = self.value as i128;
        let b = mul.value as i128;

        let Some(product) = a.checked_mul(b) else {
            return None;
        };

        let Some(rescaled) = Self::div_product_by_scale(product) else {
            return None;
        };
        let rescaled = rescaled as i128;

        let Some(result) = rescaled.checked_add(add.value as i128) else {
            return None;
        };

        if result > i64::MAX as i128 || result < i64::MIN as i128 {
            None
        } else {
            Some(Self {
                value: result as i64,
            })
        }
    }

    /// Multiplication returning `Result`.
    #[inline(always)]
    pub const fn try_mul(self, rhs: Self) -> Result<Self, OverflowError> {
        match self.checked_mul(rhs) {
            Some(v) => Ok(v),
            None => Err(OverflowError),
        }
    }

    /// Division returning `Result` with specific error.
    #[inline(always)]
    pub const fn try_div(self, rhs: Self) -> Result<Self, DivError> {
        if rhs.value == 0 {
            return Err(DivError::DivisionByZero);
        }
        match self.checked_div(rhs) {
            Some(v) => Ok(v),
            None => Err(DivError::Overflow),
        }
    }
}

// ============================================================================
// Mul / Div — i128 (192-bit wide arithmetic, NOT const fn)
// ============================================================================

use crate::wide;

impl<const D: u8> Decimal<i128, D> {
    /// Threshold for fast-path multiplication (both operands < 2^64).
    const FAST_MUL_THRESHOLD: u128 = 1u128 << 64;

    /// Checked multiplication using 192-bit wide arithmetic.
    #[inline(always)]
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        if self.value == 0 || rhs.value == 0 {
            return Some(Self::ZERO);
        }

        let result_negative = (self.value < 0) != (rhs.value < 0);
        let a = self.value.unsigned_abs();
        let b = rhs.value.unsigned_abs();

        // Fast path: both values fit in 64 bits → product fits in 128 bits
        if a < Self::FAST_MUL_THRESHOLD && b < Self::FAST_MUL_THRESHOLD {
            let product = a * b;
            let quotient = product / (Self::SCALE as u128);
            return Self::from_unsigned(quotient, result_negative);
        }

        // Slow path: 192-bit multiplication
        let (prod_low, prod_high) = wide::mul_wide(a, b);
        let quotient = wide::div_192_by_const(prod_low, prod_high, Self::SCALE as u128)?;
        Self::from_unsigned(quotient, result_negative)
    }

    /// Checked division using 192-bit wide arithmetic.
    #[inline(always)]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.value == 0 {
            return None;
        }
        if self.value == 0 {
            return Some(Self::ZERO);
        }

        let result_negative = (self.value < 0) != (rhs.value < 0);
        let a = self.value.unsigned_abs();
        let b = rhs.value.unsigned_abs();
        let scale = Self::SCALE as u128;

        // Widen: a * SCALE (can exceed 128 bits)
        let (prod_low, prod_high) = wide::mul_u128_by_small(a, scale);

        // Divide 192-bit by runtime divisor
        let quotient = wide::div_192_by_u128(prod_low, prod_high, b)?;
        Self::from_unsigned(quotient, result_negative)
    }

    /// Saturating multiplication.
    #[inline(always)]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        self.checked_mul(rhs).unwrap_or({
            if (self.value > 0) == (rhs.value > 0) {
                Self::MAX
            } else {
                Self::MIN
            }
        })
    }

    /// Wrapping multiplication.
    #[inline(always)]
    pub fn wrapping_mul(self, rhs: Self) -> Self {
        if self.value == 0 || rhs.value == 0 {
            return Self::ZERO;
        }

        let result_negative = (self.value < 0) != (rhs.value < 0);
        let a = self.value.unsigned_abs();
        let b = rhs.value.unsigned_abs();

        let (prod_low, prod_high) = wide::mul_wide(a, b);
        let quotient = wide::div_192_by_const_wrapping(prod_low, prod_high, Self::SCALE as u128);

        if result_negative {
            Self {
                value: (quotient as i128).wrapping_neg(),
            }
        } else {
            Self {
                value: quotient as i128,
            }
        }
    }

    /// Saturating division.
    #[inline(always)]
    pub fn saturating_div(self, rhs: Self) -> Self {
        assert!(rhs.value != 0, "division by zero");
        self.checked_div(rhs).unwrap_or({
            if (self.value > 0) == (rhs.value > 0) {
                Self::MAX
            } else {
                Self::MIN
            }
        })
    }

    /// Wrapping division.
    #[inline(always)]
    pub fn wrapping_div(self, rhs: Self) -> Self {
        assert!(rhs.value != 0, "division by zero");

        let result_negative = (self.value < 0) != (rhs.value < 0);
        let a = self.value.unsigned_abs();
        let b = rhs.value.unsigned_abs();
        let scale = Self::SCALE as u128;

        let (prod_low, prod_high) = wide::mul_u128_by_small(a, scale);
        let quotient = wide::div_192_by_u128_wrapping(prod_low, prod_high, b);

        if result_negative {
            Self {
                value: (quotient as i128).wrapping_neg(),
            }
        } else {
            Self {
                value: quotient as i128,
            }
        }
    }

    /// Multiply by a plain integer (no rescaling).
    #[inline(always)]
    pub const fn mul_int(self, rhs: i128) -> Option<Self> {
        match self.value.checked_mul(rhs) {
            Some(v) => Some(Self { value: v }),
            None => None,
        }
    }

    /// Fused multiply-add: `(self * mul) + add` with single rescaling.
    #[inline(always)]
    pub fn mul_add(self, mul: Self, add: Self) -> Option<Self> {
        if self.value == 0 || mul.value == 0 {
            return Some(add);
        }

        let product = self.checked_mul(mul)?;
        product.checked_add(add)
    }

    /// Multiplication returning `Result`.
    #[inline(always)]
    pub fn try_mul(self, rhs: Self) -> Result<Self, OverflowError> {
        self.checked_mul(rhs).ok_or(OverflowError)
    }

    /// Division returning `Result` with specific error.
    #[inline(always)]
    pub fn try_div(self, rhs: Self) -> Result<Self, DivError> {
        if rhs.value == 0 {
            return Err(DivError::DivisionByZero);
        }
        self.checked_div(rhs).ok_or(DivError::Overflow)
    }

    /// Helper: convert unsigned quotient + sign to Decimal, with bounds check.
    #[inline(always)]
    fn from_unsigned(quotient: u128, negative: bool) -> Option<Self> {
        if negative {
            // i128::MIN.unsigned_abs() = i128::MAX + 1
            if quotient > (i128::MAX as u128) + 1 {
                return None;
            }
            Some(Self {
                value: (quotient as i128).wrapping_neg(),
            })
        } else {
            if quotient > i128::MAX as u128 {
                return None;
            }
            Some(Self {
                value: quotient as i128,
            })
        }
    }
}

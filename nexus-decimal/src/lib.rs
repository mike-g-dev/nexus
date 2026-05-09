//! Fixed-point decimal arithmetic with compile-time precision.
//!
//! `nexus-decimal` provides [`Decimal<B, DECIMALS>`] — a generic
//! fixed-point type parameterized by backing integer and decimal
//! places. Operations are `const fn` where possible, zero-allocation,
//! and designed for financial workloads.
//!
//! # Choosing Your Type
//!
//! Define aliases that match your domain:
//!
//! ```
//! use nexus_decimal::Decimal;
//!
//! type Price = Decimal<i64, 8>;       // 8dp, range ±92B — traditional finance
//! type Quantity = Decimal<i64, 4>;    // 4dp, range ±922T
//! type CryptoPrice = Decimal<i128, 12>; // 12dp, range ±39T — DeFi
//! type Usd = Decimal<i64, 2>;         // 2dp cents
//! ```
//!
//! | Backing | Max Decimals | Max Range | Use case |
//! |---------|-------------|-----------|----------|
//! | `i32` | 9 | ±2.1B / SCALE | Embedded, space-constrained |
//! | `i64` | 18 | ±9.2e18 / SCALE | Traditional finance |
//! | `i128` | 38 | ±1.7e38 / SCALE | Cryptocurrency, DeFi |
//!
//! # Quick Start
//!
//! ```
//! use nexus_decimal::Decimal;
//! use core::str::FromStr;
//!
//! type D64 = Decimal<i64, 8>;
//!
//! let price = D64::from_str("123.45").unwrap();
//! let qty = D64::from_i32(10).unwrap();
//!
//! let notional = price * qty;
//! assert_eq!(notional.to_string(), "1234.5");
//!
//! let bid = D64::from_str("100.00").unwrap();
//! let ask = D64::from_str("100.50").unwrap();
//! let mid = bid.midpoint(ask);
//! assert_eq!(mid.to_string(), "100.25");
//! ```
//!
//! # Integer Conversions
//!
//! `From<IntType>` is implemented for primitive integer types whenever the
//! conversion is sound — i.e., `IntType::MAX * 10^DECIMALS` fits the backing.
//! Otherwise, `TryFrom<i64>` and `TryFrom<u64>` provide fallible conversions.
//! Smaller types that don't fit must be widened explicitly.
//!
//! ```
//! use nexus_decimal::Decimal;
//! type D64 = Decimal<i64, 8>;
//!
//! // Sound combinations: infallible.
//! let qty: D64 = 100_i32.into();
//! let count: D64 = 42_u16.into();
//!
//! // Unsound combinations: fallible.
//! let huge: Result<D64, _> = i64::MAX.try_into();
//! assert!(huge.is_err());
//! ```
//!
//! Use [`Decimal::from_scaled`] for tick-size construction:
//!
//! ```
//! use nexus_decimal::Decimal;
//! type D64 = Decimal<i64, 8>;
//!
//! let tick = D64::from_scaled(1, 5).unwrap(); // 0.00001
//! ```
//!
//! # Compile-Time Constants
//!
//! ```
//! use nexus_decimal::Decimal;
//!
//! type D64 = Decimal<i64, 8>;
//!
//! const PRICE: D64 = D64::new(100, 50_000_000); // 100.50
//! const FEE: D64 = D64::from_raw(500_000);       // 0.005
//! const TOTAL: D64 = match PRICE.checked_add(FEE) {
//!     Some(v) => v,
//!     None => panic!("overflow"),
//! };
//! ```
//!
//! # Arithmetic Variants
//!
//! Every arithmetic operation comes in four flavors:
//!
//! | Variant | Returns | On overflow |
//! |---------|---------|-------------|
//! | `checked_*` | `Option<Self>` | `None` |
//! | `try_*` | `Result<Self, SpecificError>` | Typed error |
//! | `saturating_*` | `Self` | Clamps to `MIN`/`MAX` |
//! | `wrapping_*` | `Self` | Wraps around |
//!
//! Operators (`+`, `-`, `*`, `/`, `%`) always panic on overflow
//! in both debug and release builds.
//!
//! # Error Types
//!
//! Errors are scoped per operation — no catch-all enum:
//!
//! | Error | Used by | Variants |
//! |-------|---------|----------|
//! | [`OverflowError`] | `try_add`, `try_mul`, etc. | (unit struct) |
//! | [`DivError`] | `try_div` | `Overflow`, `DivisionByZero` |
//! | [`ParseError`] | `from_str_exact`, `FromStr` | `InvalidFormat`, `Overflow`, `PrecisionLoss` |
//! | [`ConvertError`] | `from_f64`, `TryFrom` | `Overflow`, `PrecisionLoss` |
//!
//! # Feature Flags
//!
//! | Feature | Dependencies | Provides |
//! |---------|-------------|----------|
//! | `std` (default) | — | `Error` trait impls |
//! | `serde` | `serde` | Serialize/Deserialize (string for JSON, raw for binary) |
//! | `num-traits` | `num-traits` | Zero, One, Num, Signed, Bounded, Checked*, ToPrimitive |
//!
//! # `no_std` Support
//!
//! Disable default features for `no_std`:
//! ```toml
//! nexus-decimal = { version = "0.1", default-features = false }
//! ```
//!
//! # Migration from fixdec
//!
//! ```ignore
//! // Before:
//! use fixdec::D64;
//!
//! // After:
//! use nexus_decimal::Decimal;
//! type D64 = Decimal<i64, 8>;
//! ```
//!
//! API differences:
//! - `mul_i64` / `mul_i128` → `mul_int` (takes the backing type)
//! - `DecimalError` → per-method error types ([`OverflowError`], [`DivError`], etc.)
//! - No predefined aliases — define your own (`type Price = Decimal<i64, 8>`)
//! - New: financial methods (`midpoint`, `spread`, `round_to_tick`, etc.)

#![no_std]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

pub mod backing;
pub mod error;

mod arithmetic;
mod bytes;
mod constants;
mod convert;
mod decimal;
mod div_by_scale;
mod financial;
mod format;
mod from_int;
mod ops;
mod pow10;
mod rounding;
mod wide;

#[cfg(feature = "serde")]
mod serde_impl;

#[cfg(feature = "num-traits")]
mod num_traits_impl;

pub use backing::Backing;
pub use decimal::Decimal;
pub use error::{ConvertError, DivError, OverflowError, ParseError};

//! Byte serialization for `Decimal`.
//!
//! Inherent impls per backing type (not trait methods) since the
//! return types differ by size. All methods are const fn.

use crate::Decimal;

// ============================================================================
// i32 — 4 bytes
// ============================================================================

impl<const D: u8> Decimal<i32, D> {
    /// Number of bytes in the serialized representation.
    pub const BYTES: usize = 4;

    /// Returns the underlying value as little-endian bytes.
    #[inline(always)]
    pub const fn to_le_bytes(self) -> [u8; 4] {
        self.value.to_le_bytes()
    }

    /// Returns the underlying value as big-endian bytes.
    #[inline(always)]
    pub const fn to_be_bytes(self) -> [u8; 4] {
        self.value.to_be_bytes()
    }

    /// Returns the underlying value as native-endian bytes.
    #[inline(always)]
    pub const fn to_ne_bytes(self) -> [u8; 4] {
        self.value.to_ne_bytes()
    }

    /// Reconstructs a `Decimal` from its little-endian byte representation.
    #[inline(always)]
    pub const fn from_le_bytes(bytes: [u8; 4]) -> Self {
        Self {
            value: i32::from_le_bytes(bytes),
        }
    }

    /// Reconstructs a `Decimal` from its big-endian byte representation.
    #[inline(always)]
    pub const fn from_be_bytes(bytes: [u8; 4]) -> Self {
        Self {
            value: i32::from_be_bytes(bytes),
        }
    }

    /// Reconstructs a `Decimal` from its native-endian byte representation.
    #[inline(always)]
    pub const fn from_ne_bytes(bytes: [u8; 4]) -> Self {
        Self {
            value: i32::from_ne_bytes(bytes),
        }
    }

    /// Writes little-endian bytes into `buf`. Panics if `buf.len() < 4`.
    #[inline]
    pub fn write_le_bytes(&self, buf: &mut [u8]) {
        buf[..4].copy_from_slice(&self.to_le_bytes());
    }

    /// Writes big-endian bytes into `buf`. Panics if `buf.len() < 4`.
    #[inline]
    pub fn write_be_bytes(&self, buf: &mut [u8]) {
        buf[..4].copy_from_slice(&self.to_be_bytes());
    }

    /// Reads little-endian bytes from `buf`. Panics if `buf.len() < 4`.
    #[inline]
    pub fn read_le_bytes(buf: &[u8]) -> Self {
        let bytes: [u8; 4] = buf[..4].try_into().unwrap();
        Self::from_le_bytes(bytes)
    }

    /// Reads big-endian bytes from `buf`. Panics if `buf.len() < 4`.
    #[inline]
    pub fn read_be_bytes(buf: &[u8]) -> Self {
        let bytes: [u8; 4] = buf[..4].try_into().unwrap();
        Self::from_be_bytes(bytes)
    }
}

// ============================================================================
// i64 — 8 bytes
// ============================================================================

impl<const D: u8> Decimal<i64, D> {
    /// Number of bytes in the serialized representation.
    pub const BYTES: usize = 8;

    /// Returns the underlying value as little-endian bytes.
    #[inline(always)]
    pub const fn to_le_bytes(self) -> [u8; 8] {
        self.value.to_le_bytes()
    }

    /// Returns the underlying value as big-endian bytes.
    #[inline(always)]
    pub const fn to_be_bytes(self) -> [u8; 8] {
        self.value.to_be_bytes()
    }

    /// Returns the underlying value as native-endian bytes.
    #[inline(always)]
    pub const fn to_ne_bytes(self) -> [u8; 8] {
        self.value.to_ne_bytes()
    }

    /// Reconstructs a `Decimal` from its little-endian byte representation.
    #[inline(always)]
    pub const fn from_le_bytes(bytes: [u8; 8]) -> Self {
        Self {
            value: i64::from_le_bytes(bytes),
        }
    }

    /// Reconstructs a `Decimal` from its big-endian byte representation.
    #[inline(always)]
    pub const fn from_be_bytes(bytes: [u8; 8]) -> Self {
        Self {
            value: i64::from_be_bytes(bytes),
        }
    }

    /// Reconstructs a `Decimal` from its native-endian byte representation.
    #[inline(always)]
    pub const fn from_ne_bytes(bytes: [u8; 8]) -> Self {
        Self {
            value: i64::from_ne_bytes(bytes),
        }
    }

    /// Writes little-endian bytes into `buf`. Panics if `buf.len() < 8`.
    #[inline]
    pub fn write_le_bytes(&self, buf: &mut [u8]) {
        buf[..8].copy_from_slice(&self.to_le_bytes());
    }

    /// Writes big-endian bytes into `buf`. Panics if `buf.len() < 8`.
    #[inline]
    pub fn write_be_bytes(&self, buf: &mut [u8]) {
        buf[..8].copy_from_slice(&self.to_be_bytes());
    }

    /// Reads little-endian bytes from `buf`. Panics if `buf.len() < 8`.
    #[inline]
    pub fn read_le_bytes(buf: &[u8]) -> Self {
        let bytes: [u8; 8] = buf[..8].try_into().unwrap();
        Self::from_le_bytes(bytes)
    }

    /// Reads big-endian bytes from `buf`. Panics if `buf.len() < 8`.
    #[inline]
    pub fn read_be_bytes(buf: &[u8]) -> Self {
        let bytes: [u8; 8] = buf[..8].try_into().unwrap();
        Self::from_be_bytes(bytes)
    }
}

// ============================================================================
// i128 — 16 bytes
// ============================================================================

impl<const D: u8> Decimal<i128, D> {
    /// Number of bytes in the serialized representation.
    pub const BYTES: usize = 16;

    /// Returns the underlying value as little-endian bytes.
    #[inline(always)]
    pub const fn to_le_bytes(self) -> [u8; 16] {
        self.value.to_le_bytes()
    }

    /// Returns the underlying value as big-endian bytes.
    #[inline(always)]
    pub const fn to_be_bytes(self) -> [u8; 16] {
        self.value.to_be_bytes()
    }

    /// Returns the underlying value as native-endian bytes.
    #[inline(always)]
    pub const fn to_ne_bytes(self) -> [u8; 16] {
        self.value.to_ne_bytes()
    }

    /// Reconstructs a `Decimal` from its little-endian byte representation.
    #[inline(always)]
    pub const fn from_le_bytes(bytes: [u8; 16]) -> Self {
        Self {
            value: i128::from_le_bytes(bytes),
        }
    }

    /// Reconstructs a `Decimal` from its big-endian byte representation.
    #[inline(always)]
    pub const fn from_be_bytes(bytes: [u8; 16]) -> Self {
        Self {
            value: i128::from_be_bytes(bytes),
        }
    }

    /// Reconstructs a `Decimal` from its native-endian byte representation.
    #[inline(always)]
    pub const fn from_ne_bytes(bytes: [u8; 16]) -> Self {
        Self {
            value: i128::from_ne_bytes(bytes),
        }
    }

    /// Writes little-endian bytes into `buf`. Panics if `buf.len() < 16`.
    #[inline]
    pub fn write_le_bytes(&self, buf: &mut [u8]) {
        buf[..16].copy_from_slice(&self.to_le_bytes());
    }

    /// Writes big-endian bytes into `buf`. Panics if `buf.len() < 16`.
    #[inline]
    pub fn write_be_bytes(&self, buf: &mut [u8]) {
        buf[..16].copy_from_slice(&self.to_be_bytes());
    }

    /// Reads little-endian bytes from `buf`. Panics if `buf.len() < 16`.
    #[inline]
    pub fn read_le_bytes(buf: &[u8]) -> Self {
        let bytes: [u8; 16] = buf[..16].try_into().unwrap();
        Self::from_le_bytes(bytes)
    }

    /// Reads big-endian bytes from `buf`. Panics if `buf.len() < 16`.
    #[inline]
    pub fn read_be_bytes(buf: &[u8]) -> Self {
        let bytes: [u8; 16] = buf[..16].try_into().unwrap();
        Self::from_be_bytes(bytes)
    }
}

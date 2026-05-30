//! Serde implementations for all ID types.
//!
//! String types serialize as strings. Numeric types serialize as numbers.

use core::fmt;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::snowflake_id::{MixedId32, MixedId64, SnowflakeId32, SnowflakeId64};
use crate::typeid::TypeId;
use crate::types::{Base36Id, Base62Id, HexId64, Ulid, Uuid, UuidCompact};

// =============================================================================
// String types: serialize as strings
// =============================================================================

macro_rules! impl_serde_str {
    ($ty:ident, $name:expr, $len:expr) => {
        impl Serialize for $ty {
            #[inline]
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> Deserialize<'de> for $ty {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct IdVisitor;

                impl<'de> Visitor<'de> for IdVisitor {
                    type Value = $ty;

                    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        write!(f, "a {}-character {} string", $len, $name)
                    }

                    fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                        $ty::parse(v).map_err(de::Error::custom)
                    }
                }

                deserializer.deserialize_str(IdVisitor)
            }
        }
    };
}

impl_serde_str!(Uuid, "UUID", 36);
impl_serde_str!(UuidCompact, "compact UUID", 32);
impl_serde_str!(Ulid, "ULID", 26);
impl_serde_str!(HexId64, "hex ID", 16);
impl_serde_str!(Base62Id, "base62 ID", 11);
impl_serde_str!(Base36Id, "base36 ID", 13);

// =============================================================================
// TypeId: serialize as string
// =============================================================================

impl<const CAP: usize> Serialize for TypeId<CAP> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de, const CAP: usize> Deserialize<'de> for TypeId<CAP> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TypeIdVisitor<const C: usize>;

        impl<const C: usize> Visitor<'_> for TypeIdVisitor<C> {
            type Value = TypeId<C>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a TypeId string (prefix_suffix)")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                TypeId::parse(v).map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_str(TypeIdVisitor::<CAP>)
    }
}

// =============================================================================
// Numeric types: serialize as integers
// =============================================================================

impl<const TS: u8, const WK: u8, const SQ: u8> Serialize for SnowflakeId64<TS, WK, SQ> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.raw())
    }
}

impl<'de, const TS: u8, const WK: u8, const SQ: u8> Deserialize<'de> for SnowflakeId64<TS, WK, SQ> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct IdVisitor<const T: u8, const W: u8, const S: u8>;

        impl<const T: u8, const W: u8, const S: u8> Visitor<'_> for IdVisitor<T, W, S> {
            type Value = SnowflakeId64<T, W, S>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a u64 snowflake ID")
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(SnowflakeId64::from_raw(v))
            }
        }

        deserializer.deserialize_u64(IdVisitor::<TS, WK, SQ>)
    }
}

impl<const TS: u8, const WK: u8, const SQ: u8> Serialize for SnowflakeId32<TS, WK, SQ> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.raw())
    }
}

impl<'de, const TS: u8, const WK: u8, const SQ: u8> Deserialize<'de> for SnowflakeId32<TS, WK, SQ> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct IdVisitor<const T: u8, const W: u8, const S: u8>;

        impl<const T: u8, const W: u8, const S: u8> Visitor<'_> for IdVisitor<T, W, S> {
            type Value = SnowflakeId32<T, W, S>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a u32 snowflake ID")
            }

            fn visit_u32<E: de::Error>(self, v: u32) -> Result<Self::Value, E> {
                Ok(SnowflakeId32::from_raw(v))
            }
        }

        deserializer.deserialize_u32(IdVisitor::<TS, WK, SQ>)
    }
}

impl<const TS: u8, const WK: u8, const SQ: u8> Serialize for MixedId64<TS, WK, SQ> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.raw())
    }
}

impl<'de, const TS: u8, const WK: u8, const SQ: u8> Deserialize<'de> for MixedId64<TS, WK, SQ> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct IdVisitor<const T: u8, const W: u8, const S: u8>;

        impl<const T: u8, const W: u8, const S: u8> Visitor<'_> for IdVisitor<T, W, S> {
            type Value = MixedId64<T, W, S>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a u64 mixed ID")
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(MixedId64::from_raw(v))
            }
        }

        deserializer.deserialize_u64(IdVisitor::<TS, WK, SQ>)
    }
}

impl<const TS: u8, const WK: u8, const SQ: u8> Serialize for MixedId32<TS, WK, SQ> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.raw())
    }
}

impl<'de, const TS: u8, const WK: u8, const SQ: u8> Deserialize<'de> for MixedId32<TS, WK, SQ> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct IdVisitor<const T: u8, const W: u8, const S: u8>;

        impl<const T: u8, const W: u8, const S: u8> Visitor<'_> for IdVisitor<T, W, S> {
            type Value = MixedId32<T, W, S>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a u32 mixed ID")
            }

            fn visit_u32<E: de::Error>(self, v: u32) -> Result<Self::Value, E> {
                Ok(MixedId32::from_raw(v))
            }
        }

        deserializer.deserialize_u32(IdVisitor::<TS, WK, SQ>)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_serde_roundtrip() {
        let uuid = Uuid::parse("01234567-89ab-cdef-fedc-ba9876543210").unwrap();
        let json = serde_json::to_string(&uuid).unwrap();
        assert_eq!(json, "\"01234567-89ab-cdef-fedc-ba9876543210\"");
        let restored: Uuid = serde_json::from_str(&json).unwrap();
        assert_eq!(uuid, restored);
    }

    #[test]
    fn uuid_compact_serde_roundtrip() {
        let uuid = UuidCompact::parse("0123456789abcdeffedcba9876543210").unwrap();
        let json = serde_json::to_string(&uuid).unwrap();
        let restored: UuidCompact = serde_json::from_str(&json).unwrap();
        assert_eq!(uuid, restored);
    }

    #[test]
    fn hex_id_serde_roundtrip() {
        let id = HexId64::encode(0xDEAD_BEEF_CAFE_BABE);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"deadbeefcafebabe\"");
        let restored: HexId64 = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn snowflake_id64_serde_roundtrip() {
        let id = SnowflakeId64::<42, 6, 16>::from_raw(123_456_789);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "123456789");
        let restored: SnowflakeId64<42, 6, 16> = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn mixed_id64_serde_roundtrip() {
        let id = MixedId64::<42, 6, 16>::from_raw(987_654_321);
        let json = serde_json::to_string(&id).unwrap();
        let restored: MixedId64<42, 6, 16> = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn base62_serde_roundtrip() {
        let id = Base62Id::encode(12345);
        let json = serde_json::to_string(&id).unwrap();
        let restored: Base62Id = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn base36_serde_roundtrip() {
        let id = Base36Id::encode(12345);
        let json = serde_json::to_string(&id).unwrap();
        let restored: Base36Id = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }
}

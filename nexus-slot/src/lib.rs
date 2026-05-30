//! High-performance conflation slots for latest-value-wins scenarios.
//!
//! Two variants based on reader topology:
//!
//! - [`spsc`] — Single producer, single consumer. Lowest overhead.
//! - [`spmc`] — Single producer, multiple consumers. [`SharedReader`](spmc::SharedReader) is `Clone`.
//!
//! Both use a seqlock internally: the writer increments a sequence counter,
//! copies data via word-at-a-time atomics, and increments again. Readers
//! speculatively copy and retry if the sequence changed.
//!
//! # The `Pod` Trait
//!
//! Types must implement [`Pod`] (Plain Old Data) — no heap allocations,
//! no drop glue, byte-copyable. Any `Copy` type implements `Pod` automatically.
//!
//! ```rust
//! use nexus_slot::Pod;
//!
//! #[repr(C)]
//! struct OrderBook {
//!     bids: [f64; 20],
//!     asks: [f64; 20],
//!     sequence: u64,
//! }
//!
//! // SAFETY: OrderBook is just bytes — no heap allocations
//! unsafe impl Pod for OrderBook {}
//! ```
//!
//! # Examples
//!
//! ```rust
//! #[derive(Copy, Clone, Default)]
//! struct Quote { bid: f64, ask: f64, seq: u64 }
//!
//! // SPSC — single reader
//! let (mut writer, mut reader) = nexus_slot::spsc::slot::<Quote>();
//! writer.write(Quote { bid: 100.0, ask: 100.05, seq: 1 });
//! assert_eq!(reader.read().unwrap().seq, 1);
//! ```
//!
//! ```rust
//! #[derive(Copy, Clone, Default)]
//! struct Quote { bid: f64, ask: f64, seq: u64 }
//!
//! // SPMC — multiple readers
//! let (mut writer, mut reader1) = nexus_slot::spmc::shared_slot::<Quote>();
//! let mut reader2 = reader1.clone();
//!
//! writer.write(Quote { bid: 100.0, ask: 100.05, seq: 1 });
//! assert!(reader1.read().is_some());
//! assert!(reader2.read().is_some()); // independent consumption
//! ```

#![warn(missing_docs)]

pub mod spmc;
pub mod spsc;
pub(crate) mod loom_impl;

#[cfg(not(loom))]
use std::mem::{MaybeUninit, align_of, size_of};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// Marker trait for types safe to use in a conflated slot.
///
/// # Safety
///
/// Implementor guarantees:
///
/// 1. **No heap allocations**: No `Vec`, `String`, `Box`, `Arc`, etc.
/// 2. **No owned resources**: No `File`, `TcpStream`, `Mutex`, etc.
/// 3. **No drop glue**: `std::mem::needs_drop::<Self>()` returns false.
/// 4. **Byte-copyable**: Safe to memcpy without cleanup.
///
/// Essentially: the type could be `Copy`, but chooses not to.
///
/// # Example
///
/// ```rust
/// use nexus_slot::Pod;
///
/// #[repr(C)]
/// struct OrderBook {
///     bids: [f64; 20],
///     asks: [f64; 20],
///     bid_count: u8,
///     ask_count: u8,
///     sequence: u64,
/// }
///
/// // SAFETY: Just bytes, no heap
/// unsafe impl Pod for OrderBook {}
/// ```
pub unsafe trait Pod: Sized {
    /// Compile-time assertion that the implementing type does not require
    /// drop. Forces a build-time error if a `Pod` impl is added for a type
    /// with a non-trivial destructor.
    const _ASSERT_NO_DROP: () = {
        assert!(
            !std::mem::needs_drop::<Self>(),
            "Pod types must not require drop"
        );
    };
}

// SAFETY: Copy types are byte-copyable, have no drop glue, and own no resources.
// This is the canonical set of Pod guarantees.
unsafe impl<T: Copy> Pod for T {}

#[cfg(not(loom))]
/// Atomically stores `size_of::<T>()` bytes into shared memory.
///
/// Word-at-a-time `AtomicUsize` stores when alignment permits,
/// `AtomicU8` fallback for tail bytes or poorly-aligned types.
/// All stores use `Relaxed` ordering — caller provides fences.
///
/// # Safety
///
/// - `dst` must be valid for `size_of::<T>()` bytes
/// - `dst` must be aligned to `align_of::<T>()`
/// - `dst` must be derived from `UnsafeCell` (shared-mutable provenance)
#[inline]
pub(crate) unsafe fn atomic_store<T: Pod>(dst: *mut T, src: &T) {
    // SAFETY: Caller guarantees dst is valid, aligned, and from UnsafeCell.
    // Pod bound ensures T is byte-copyable with no drop glue.
    unsafe {
        let dst = dst.cast::<u8>();
        let src = (src as *const T).cast::<u8>();
        let size = size_of::<T>();

        if align_of::<T>() >= align_of::<usize>() {
            let words = size / size_of::<usize>();
            let tail = size % size_of::<usize>();

            for i in 0..words {
                let atom = &*(dst.add(i * size_of::<usize>()) as *const AtomicUsize);
                let val = src.add(i * size_of::<usize>()).cast::<usize>().read();
                atom.store(val, Ordering::Relaxed);
            }

            let base = words * size_of::<usize>();
            for i in 0..tail {
                let atom = &*(dst.add(base + i) as *const AtomicU8);
                atom.store(*src.add(base + i), Ordering::Relaxed);
            }
        } else {
            for i in 0..size {
                let atom = &*(dst.add(i) as *const AtomicU8);
                atom.store(*src.add(i), Ordering::Relaxed);
            }
        }
    }
}

#[cfg(not(loom))]
/// Atomically loads `size_of::<T>()` bytes from shared memory.
///
/// Word-at-a-time `AtomicUsize` loads when alignment permits,
/// `AtomicU8` fallback for tail bytes or poorly-aligned types.
/// All loads use `Relaxed` ordering — caller provides fences.
///
/// # Safety
///
/// - `src` must be valid for `size_of::<T>()` bytes
/// - `src` must be aligned to `align_of::<T>()`
/// - `src` must be derived from `UnsafeCell` (shared-mutable provenance)
#[inline]
pub(crate) unsafe fn atomic_load<T: Pod>(src: *const T) -> T {
    // SAFETY: Caller guarantees src is valid, aligned, and from UnsafeCell.
    // Pod bound ensures T is byte-copyable; assume_init is sound because all
    // bytes are written by the atomic loads before we return.
    unsafe {
        let mut buf = MaybeUninit::<T>::uninit();
        let dst = buf.as_mut_ptr().cast::<u8>();
        let src = src.cast::<u8>();
        let size = size_of::<T>();

        if align_of::<T>() >= align_of::<usize>() {
            let words = size / size_of::<usize>();
            let tail = size % size_of::<usize>();

            for i in 0..words {
                let atom = &*(src.add(i * size_of::<usize>()) as *const AtomicUsize);
                let val = atom.load(Ordering::Relaxed);
                dst.add(i * size_of::<usize>()).cast::<usize>().write(val);
            }

            let base = words * size_of::<usize>();
            for i in 0..tail {
                let atom = &*(src.add(base + i) as *const AtomicU8);
                *dst.add(base + i) = atom.load(Ordering::Relaxed);
            }
        } else {
            for i in 0..size {
                let atom = &*(src.add(i) as *const AtomicU8);
                *dst.add(i) = atom.load(Ordering::Relaxed);
            }
        }

        buf.assume_init()
    }
}

//! # nexus-pool
//!
//! High-performance object pools for latency-sensitive applications.
//!
//! `nexus-pool` provides two pool implementations optimized for different
//! threading models, both designed to eliminate allocation on hot paths.
//!
//! ## Design Philosophy
//!
//! This crate follows the principle of **predictability over generality**:
//!
//! - **SPSC over MPMC**: Single-writer patterns avoid lock contention entirely
//! - **Pre-allocation over dynamic growth**: Bounded pools have deterministic behavior
//! - **Specialized over general**: Each pool type is optimized for its specific access pattern
//!
//! ## Modules
//!
//! ### [`local`] - Single-threaded pools
//!
//! For use within a single thread. Zero synchronization overhead.
//!
//! - [`local::BoundedPool`] - Fixed capacity, pre-allocated objects (RAII only)
//! - [`local::Pool`] - Growable, creates objects on demand via factory
//!   - RAII: [`acquire()`](local::Pool::acquire) / [`try_acquire()`](local::Pool::try_acquire) → auto-return on drop
//!   - Manual: [`take()`](local::Pool::take) / [`try_take()`](local::Pool::try_take) → [`put()`](local::Pool::put) to return
//!
//! **Performance**: ~26 cycles acquire, ~26-28 cycles release (p50)
//!
//! ### [`sync`] - Thread-safe pools
//!
//! For cross-thread object transfer with single-acquirer semantics.
//!
//! - [`sync::Pool`] - One thread acquires, any thread can return
//!
//! **Performance**: ~42 cycles acquire, ~68 cycles release (p50)
//!
//! ## Why Single-Acquirer?
//!
//! You might ask: why not allow any thread to both acquire and return?
//!
//! **The short answer**: MPMC pools require solving the ABA problem, which adds
//! significant overhead (generation counters, hazard pointers, or epoch-based
//! reclamation). For most high-performance use cases, MPMC is also a design smell.
//!
//! **The architectural answer**: If multiple threads need to acquire from the same
//! pool, you're violating the single-writer principle. This creates contention
//! and unpredictable latency—exactly what you're trying to avoid by using a pool.
//!
//! Better alternatives:
//! - **Per-thread pools**: Each thread owns its own `local::Pool`
//! - **Sharded pools**: Hash to a pool based on thread ID
//! - **Message passing**: Send pre-allocated buffers via channels
//!
//! If you truly need MPMC semantics, consider `crossbeam::ArrayQueue` or
//! `crossbeam::SegQueue` which are well-optimized for that use case.
//!
//! ## Use Cases
//!
//! ### Trading Systems / Market Data
//!
//! ```rust
//! use nexus_pool::sync::Pool;
//!
//! // Hot path thread owns the pool
//! let pool = Pool::new(
//!     1000,
//!     || Vec::<u8>::with_capacity(4096),  // Pre-sized for typical message
//!     |v| v.clear(),                       // Reset for reuse
//! );
//!
//! // Acquire buffer, fill with market data
//! let mut buf = pool.try_acquire().expect("pool exhausted");
//! buf.extend_from_slice(b"market data...");
//!
//! // Send to worker thread for processing
//! // Buffer automatically returns to pool when worker drops it
//! std::thread::spawn(move || {
//!     process(&buf);
//!     // buf drops here, returns to pool
//! });
//! # fn process(_: &[u8]) {}
//! ```
//!
//! ### Single-Threaded Event Loops
//!
//! ```rust
//! use nexus_pool::local::BoundedPool;
//!
//! let pool = BoundedPool::new(
//!     100,
//!     || Box::new([0u8; 1024]),  // Fixed-size buffers
//!     |b| b.fill(0),             // Zero on return
//! );
//!
//! // Event loop - no allocation after startup
//! for _ in 0..1000 {
//!     if let Some(mut buf) = pool.try_acquire() {
//!         // Use buffer...
//!         buf[0] = 42;
//!     }
//!     // buf returns to pool automatically
//! }
//! ```
//!
//! ## Performance Characteristics
//!
//! Measured on Intel Core i9 @ 3.1 GHz (cycles, lower is better):
//!
//! | Pool | Acquire p50 | Release p50 | Release p99 |
//! |------|-------------|-------------|-------------|
//! | `local::BoundedPool` | 26 | 26 | 58 |
//! | `local::Pool` (reuse) | 26 | 26 | 58 |
//! | `local::Pool` (factory) | 32 | 26 | 58 |
//! | `sync::Pool` | 42 | 68 | 86 |
//!
//! The sync pool is ~1.6x slower on acquire due to atomic operations, but
//! still sub-100 cycles for both operations. Release p99 remains stable
//! even under concurrent return from multiple threads.
//!
//! ## Safety
//!
//! The RAII pool types use guards ([`local::Pooled`], [`sync::Pooled`]) that
//! automatically return objects to the pool on drop. If the pool is dropped
//! before all guards, the guards will drop their values directly instead of
//! returning them—no panic, no leak, no use-after-free.
//!
//! [`local::Pool`] also supports manual [`take()`](local::Pool::take) /
//! [`put()`](local::Pool::put) for cases where RAII lifetime doesn't fit
//! (e.g., storing values in structs, passing through pipelines).

#![warn(missing_docs)]

pub mod local;
pub mod sync;

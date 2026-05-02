//! Slab task allocator — optional, power-user feature.
//!
//! By default, tasks are Box-allocated. For zero-alloc hot-path spawning,
//! configure a slab via [`RuntimeBuilder::slab_unbounded`] or
//! [`RuntimeBuilder::slab_bounded`].
//!
//! Three levels of control:
//! - **`spawn_slab(future)`** — allocate and enqueue in one call. Panics if full.
//! - **`claim_slab()`** — reserve a slot, then `.spawn(future)` later. Panics if full.
//! - **`try_claim_slab()`** — reserve if space available. Nothing lost on failure.

use std::cell::Cell;
use std::future::Future;

// Task construction goes through crate::task::{new_joinable_slab, ...}

// =============================================================================
// TLS slots
// =============================================================================

/// Claim a slab slot, copy `size` bytes from `src`, return raw pointer.
/// Returns null if the slab is full (bounded only).
type ClaimFn = unsafe fn(src: *const u8, size: usize) -> *mut u8;

/// Try to claim a vacant slab slot without writing.
/// Returns (ptr, chunk_idx) or (null, 0) if full.
type TryClaimFn = unsafe fn() -> (*mut u8, usize);

/// Free a slab slot (used by task header free_fn).
type FreeFn = unsafe fn(ptr: *mut u8);

/// Free a slab slot with full context (used by SlabClaim::Drop).
type ClaimFreeFn = unsafe fn(slab_ptr: *const u8, ptr: *mut u8, chunk_idx: usize);

thread_local! {
    /// Raw pointer to the slab instance.
    static SLAB_PTR: Cell<*const u8> = const { Cell::new(std::ptr::null()) };

    /// Fn pointer: claim a slot and copy task bytes into it.
    static SLAB_CLAIM: Cell<ClaimFn> = const { Cell::new(no_slab_claim) };

    /// Fn pointer: free a slab slot (task header path).
    static SLAB_FREE: Cell<FreeFn> = const { Cell::new(no_slab_free) };

    /// Fn pointer: try to claim a vacant slot (returns ptr + chunk_idx).
    static SLAB_TRY_CLAIM: Cell<TryClaimFn> = const { Cell::new(no_slab_try_claim) };

    /// Fn pointer: free a claimed slot (SlabClaim::Drop path).
    static SLAB_CLAIM_FREE: Cell<ClaimFreeFn> = const { Cell::new(no_slab_claim_free) };

    /// Configured slot size in bytes.
    static SLAB_SLOT_SIZE: Cell<usize> = const { Cell::new(0) };
}

// -- Panic stubs --

unsafe fn no_slab_claim(_src: *const u8, _size: usize) -> *mut u8 {
    panic!(
        "spawn_slab() called without a slab configured — \
         use Runtime::builder().slab_unbounded(slab) or .slab_bounded(slab)"
    )
}

unsafe fn no_slab_free(_ptr: *mut u8) {
    panic!("slab free called without a slab configured")
}

unsafe fn no_slab_try_claim() -> (*mut u8, usize) {
    panic!(
        "try_claim_slab()/claim_slab() called without a slab configured — \
         use Runtime::builder().slab_unbounded(slab) or .slab_bounded(slab)"
    )
}

unsafe fn no_slab_claim_free(_slab_ptr: *const u8, _ptr: *mut u8, _chunk_idx: usize) {
    panic!("slab claim free called without a slab configured")
}

// =============================================================================
// TLS install/guard
// =============================================================================

/// Configuration for slab TLS installation.
pub(crate) struct SlabTlsConfig {
    pub(crate) slab_ptr: *const u8,
    pub(crate) claim_fn: ClaimFn,
    pub(crate) free_fn: FreeFn,
    pub(crate) try_claim_fn: TryClaimFn,
    pub(crate) claim_free_fn: ClaimFreeFn,
    pub(crate) slot_size: usize,
}

/// Install slab TLS and return an RAII guard that owns the slab.
///
/// The guard restores the previous TLS state on drop (in its manual
/// Drop body), then releases the slab memory (via field drop). Caller
/// is responsible for ensuring the guard outlives any code that might
/// dispatch through TLS — typically by storing it as the LAST field
/// on the type that owns the runtime state. See BUG-1 (#167) for the
/// failure mode this prevents.
pub(crate) fn install_slab(slab: Box<dyn std::any::Any>, config: &SlabTlsConfig) -> SlabGuard {
    let prev_ptr = SLAB_PTR.with(|c| c.replace(config.slab_ptr));
    let prev_claim = SLAB_CLAIM.with(|c| c.replace(config.claim_fn));
    let prev_free = SLAB_FREE.with(|c| c.replace(config.free_fn));
    let prev_try_claim = SLAB_TRY_CLAIM.with(|c| c.replace(config.try_claim_fn));
    let prev_claim_free = SLAB_CLAIM_FREE.with(|c| c.replace(config.claim_free_fn));
    let prev_slot_size = SLAB_SLOT_SIZE.with(|c| c.replace(config.slot_size));
    SlabGuard {
        prev_ptr,
        prev_claim,
        prev_free,
        prev_try_claim,
        prev_claim_free,
        prev_slot_size,
        _slab: slab,
    }
}

#[allow(clippy::struct_field_names)]
pub(crate) struct SlabGuard {
    prev_ptr: *const u8,
    prev_claim: ClaimFn,
    prev_free: FreeFn,
    prev_try_claim: TryClaimFn,
    prev_claim_free: ClaimFreeFn,
    prev_slot_size: usize,

    // Owns the type-erased slab. Drops AFTER the manual Drop body
    // returns, so TLS is already restored when slab memory is released.
    // Slab's own Drop touches its own memory only — never the TLS
    // dispatch path — so this ordering is safe even though prev_* may
    // point to the no-slab panic stubs at that point.
    _slab: Box<dyn std::any::Any>,
}

impl Drop for SlabGuard {
    fn drop(&mut self) {
        // Restore TLS to whatever was there before this install.
        // After this body returns, _slab field drops and the slab is freed.
        SLAB_PTR.with(|c| c.set(self.prev_ptr));
        SLAB_CLAIM.with(|c| c.set(self.prev_claim));
        SLAB_FREE.with(|c| c.set(self.prev_free));
        SLAB_TRY_CLAIM.with(|c| c.set(self.prev_try_claim));
        SLAB_CLAIM_FREE.with(|c| c.set(self.prev_claim_free));
        SLAB_SLOT_SIZE.with(|c| c.set(self.prev_slot_size));
    }
}

// =============================================================================
// spawn_slab — allocate + enqueue in one step
// =============================================================================

/// Allocate a joinable task in the slab and return its raw pointer.
///
/// # Panics
///
/// - If no slab is configured.
/// - If the slab is full (bounded slab).
/// - If the task exceeds the slab's slot size.
pub(crate) fn slab_spawn<F>(future: F, tracker_key: u32) -> *mut u8
where
    F: Future + 'static,
    F::Output: 'static,
{
    let task = crate::task::new_joinable_slab(future, tracker_key, slab_free_task);
    let size = std::mem::size_of_val(&task);
    let src = std::ptr::from_ref(&task).cast::<u8>();

    let claim = SLAB_CLAIM.with(Cell::get);
    // SAFETY: claim copies `size` bytes from `src` into a slab slot.
    let ptr = unsafe { claim(src, size) };
    assert!(!ptr.is_null(), "slab full — spawn_slab failed");

    // Task was copied into the slab. Prevent stack drop.
    std::mem::forget(task);

    ptr
}

/// Free function stored in slab-allocated task headers.
unsafe fn slab_free_task(ptr: *mut u8) {
    let free = SLAB_FREE.with(Cell::get);
    unsafe { free(ptr) };
}

// =============================================================================
// SlabClaim — reserved slot handle (lifetime-free)
// =============================================================================

/// A reserved slab slot for the async runtime.
///
/// Call `.spawn(future)` to write a task and enqueue it, or drop to
/// return the slot to the freelist. Nothing is lost on drop — the
/// future was never constructed.
///
/// Lifetime-free — safe because the runtime owns the slab for the
/// duration of `block_on`, and `SlabClaim` can only be created inside
/// `block_on`.
pub struct SlabClaim {
    ptr: *mut u8,
    slab_ptr: *const u8,
    free: ClaimFreeFn,
    chunk_idx: usize,
    slot_size: usize,
    // !Send + !Sync — must stay on the runtime thread.
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl SlabClaim {
    /// Write a task into the reserved slot and enqueue it.
    ///
    /// Consumes the claim. The future is constructed, placed in the
    /// slab slot, and pushed to the executor's ready queue.
    pub fn spawn<F>(self, future: F) -> crate::task::JoinHandle<F::Output>
    where
        F: Future + 'static,
        F::Output: 'static,
    {
        crate::runtime::with_executor(|exec| {
            let tracker_key = exec.next_tracker_key();
            let task = crate::task::new_joinable_slab(future, tracker_key, slab_free_task);
            let size = std::mem::size_of_val(&task);

            assert!(
                size <= self.slot_size,
                "task size ({size} bytes) exceeds slab slot size ({} bytes)",
                self.slot_size,
            );

            let src = std::ptr::from_ref(&task).cast::<u8>();
            // SAFETY: ptr is a valid vacant slot, src has `size` valid bytes.
            unsafe { std::ptr::copy_nonoverlapping(src, self.ptr, size) };
            std::mem::forget(task);

            let ptr = self.ptr;
            // Don't run Drop — the slot is now occupied.
            std::mem::forget(self);

            exec.spawn_raw(ptr);
            crate::task::JoinHandle::new(ptr)
        })
    }

    /// Raw pointer to the reserved slot.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Slot capacity in bytes.
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }
}

impl Drop for SlabClaim {
    fn drop(&mut self) {
        // Slot claimed but never written — return to freelist.
        // SAFETY: free was captured at claim time from TLS.
        // The slab is alive (runtime owns it for the duration of block_on).
        unsafe { (self.free)(self.slab_ptr, self.ptr, self.chunk_idx) };
    }
}

impl std::fmt::Debug for SlabClaim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlabClaim")
            .field("ptr", &self.ptr)
            .field("slot_size", &self.slot_size)
            .finish()
    }
}

// =============================================================================
// Public claim API
// =============================================================================

/// Try to reserve a slab slot. Returns `None` if the slab is full.
///
/// # Panics
///
/// - If called outside a runtime context.
/// - If no slab is configured.
pub(crate) fn try_claim() -> Option<SlabClaim> {
    let try_claim_fn = SLAB_TRY_CLAIM.with(Cell::get);
    // SAFETY: try_claim_fn claims a slot from the slab.
    let (ptr, chunk_idx) = unsafe { try_claim_fn() };
    if ptr.is_null() {
        return None;
    }

    let slab_ptr = SLAB_PTR.with(Cell::get);
    let free = SLAB_CLAIM_FREE.with(Cell::get);
    let slot_size = SLAB_SLOT_SIZE.with(Cell::get);

    Some(SlabClaim {
        ptr,
        slab_ptr,
        free,
        chunk_idx,
        slot_size,
        _not_send: std::marker::PhantomData,
    })
}

/// Reserve a slab slot. Panics if full.
///
/// # Panics
///
/// - If called outside a runtime context.
/// - If no slab is configured.
/// - If the slab is full (bounded slab).
pub(crate) fn claim() -> SlabClaim {
    try_claim().expect("slab full — claim_slab failed")
}

// =============================================================================
// Monomorphized fn pointers (created at builder time)
// =============================================================================

/// Build TLS config for an unbounded slab.
pub(crate) fn make_unbounded_config<const S: usize>(slab_ptr: *const u8) -> SlabTlsConfig {
    SlabTlsConfig {
        slab_ptr,
        claim_fn: unbounded_claim::<S>,
        free_fn: unbounded_free::<S>,
        try_claim_fn: unbounded_try_claim::<S>,
        claim_free_fn: unbounded_claim_free::<S>,
        slot_size: S,
    }
}

/// Build TLS config for a bounded slab.
pub(crate) fn make_bounded_config<const S: usize>(slab_ptr: *const u8) -> SlabTlsConfig {
    SlabTlsConfig {
        slab_ptr,
        claim_fn: bounded_claim::<S>,
        free_fn: bounded_free::<S>,
        try_claim_fn: bounded_try_claim::<S>,
        claim_free_fn: bounded_claim_free::<S>,
        slot_size: S,
    }
}

// -- Unbounded --

unsafe fn unbounded_claim<const S: usize>(src: *const u8, size: usize) -> *mut u8 {
    let slab_ptr = SLAB_PTR.with(Cell::get);
    let slab = unsafe { &*(slab_ptr as *const nexus_slab::byte::unbounded::Slab<S>) };
    unsafe { slab.alloc_raw(src, size) }
}

unsafe fn unbounded_free<const S: usize>(ptr: *mut u8) {
    let slab_ptr = SLAB_PTR.with(Cell::get);
    let slab = unsafe { &*(slab_ptr as *const nexus_slab::byte::unbounded::Slab<S>) };
    let slot = unsafe { nexus_slab::byte::Slot::<u8>::from_raw(ptr) };
    slab.free(slot);
}

unsafe fn unbounded_try_claim<const S: usize>() -> (*mut u8, usize) {
    let slab_ptr = SLAB_PTR.with(Cell::get);
    let slab = unsafe { &*(slab_ptr as *const nexus_slab::byte::unbounded::Slab<S>) };
    let claim = slab.claim();
    let ptr = claim.as_ptr();
    let chunk_idx = claim.chunk_idx();
    // Consume without running Drop.
    std::mem::forget(claim);
    (ptr, chunk_idx)
}

unsafe fn unbounded_claim_free<const S: usize>(
    slab_ptr: *const u8,
    ptr: *mut u8,
    chunk_idx: usize,
) {
    let slab = unsafe { &*(slab_ptr as *const nexus_slab::byte::unbounded::Slab<S>) };
    // O(1) — goes directly to the correct chunk's freelist.
    unsafe { slab.free_raw_in_chunk(ptr, chunk_idx) };
}

// -- Bounded --

unsafe fn bounded_claim<const S: usize>(src: *const u8, size: usize) -> *mut u8 {
    let slab_ptr = SLAB_PTR.with(Cell::get);
    let slab = unsafe { &*(slab_ptr as *const nexus_slab::byte::bounded::Slab<S>) };
    unsafe { slab.alloc_raw(src, size) }
}

unsafe fn bounded_free<const S: usize>(ptr: *mut u8) {
    let slab_ptr = SLAB_PTR.with(Cell::get);
    let slab = unsafe { &*(slab_ptr as *const nexus_slab::byte::bounded::Slab<S>) };
    let slot = unsafe { nexus_slab::byte::Slot::<u8>::from_raw(ptr) };
    slab.free(slot);
}

unsafe fn bounded_try_claim<const S: usize>() -> (*mut u8, usize) {
    let slab_ptr = SLAB_PTR.with(Cell::get);
    let slab = unsafe { &*(slab_ptr as *const nexus_slab::byte::bounded::Slab<S>) };
    slab.try_claim().map_or((std::ptr::null_mut(), 0), |claim| {
        let ptr = claim.as_ptr();
        std::mem::forget(claim);
        (ptr, 0) // bounded = single chunk
    })
}

unsafe fn bounded_claim_free<const S: usize>(slab_ptr: *const u8, ptr: *mut u8, _chunk_idx: usize) {
    let slab = unsafe { &*(slab_ptr as *const nexus_slab::byte::bounded::Slab<S>) };
    unsafe { slab.free_raw(ptr) };
}

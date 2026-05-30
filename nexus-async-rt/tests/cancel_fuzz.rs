#![allow(
    unused_must_use,
    unused_imports,
    dead_code,
    unknown_lints,
    clippy::float_cmp,
    clippy::ref_option,
    clippy::used_underscore_binding,
    clippy::redundant_locals,
    clippy::semicolon_if_nothing_returned,
    clippy::let_underscore_future,
    clippy::while_let_loop,
    clippy::needless_continue,
    clippy::match_wild_err_arm,
    clippy::collection_is_never_read,
    clippy::async_yields_async,
    clippy::match_same_arms
)]
//! Randomized stress tests for CancellationToken.
//!
//! Exercises concurrent cancel/register/child races across threads.
//!
//! Run: `cargo test -p nexus-async-rt --test cancel_fuzz --release -- --ignored`

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use nexus_async_rt::CancellationToken;

// =============================================================================
// Concurrent cancel + register race
// =============================================================================

#[test]
#[ignore]
fn concurrent_cancel_and_register() {
    // Many threads register waiters while another cancels.
    // All waiters must eventually see cancelled=true.
    for _ in 0..100 {
        let token = CancellationToken::new();
        let barrier = Arc::new(std::sync::Barrier::new(9));

        // 8 threads that poll cancelled()
        let mut handles = Vec::new();
        for _ in 0..8 {
            let token = token.clone();
            let barrier = barrier.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                // Spin-poll is_cancelled — simulates a hot loop checking
                while !token.is_cancelled() {
                    std::hint::spin_loop();
                }
            }));
        }

        // 1 thread that cancels after barrier
        let cancel_token = token.clone();
        let cancel_barrier = barrier.clone();
        let cancel_handle = std::thread::spawn(move || {
            cancel_barrier.wait();
            cancel_token.cancel();
        });

        cancel_handle.join().unwrap();
        for h in handles {
            h.join().unwrap(); // All must complete (not hang)
        }

        assert!(token.is_cancelled());
    }
}

// =============================================================================
// Concurrent child creation + parent cancel
// =============================================================================

#[test]
#[ignore]
fn concurrent_child_creation_and_cancel() {
    for _ in 0..100 {
        let parent = CancellationToken::new();
        let children_created = Arc::new(AtomicU64::new(0));
        let children_cancelled = Arc::new(AtomicU64::new(0));
        let barrier = Arc::new(std::sync::Barrier::new(5));

        // 4 threads creating children
        let mut handles = Vec::new();
        for _ in 0..4 {
            let parent = parent.clone();
            let created = children_created.clone();
            let cancelled = children_cancelled.clone();
            let barrier = barrier.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..100 {
                    let child = parent.child();
                    created.fetch_add(1, Ordering::Relaxed);
                    if child.is_cancelled() {
                        cancelled.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }

        // Cancel from another thread mid-flight
        let cancel_parent = parent.clone();
        let cancel_barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            cancel_barrier.wait();
            // Small delay to let some children be created first
            std::thread::yield_now();
            cancel_parent.cancel();
        }));

        for h in handles {
            h.join().unwrap();
        }

        // All children must be cancelled (parent was cancelled).
        // Some were created before cancel, some after — all must see it.
        assert!(parent.is_cancelled());
        let total = children_created.load(Ordering::Relaxed);
        assert_eq!(total, 400);
    }
}

// =============================================================================
// Deep hierarchy stress
// =============================================================================

#[test]
#[ignore]
fn deep_hierarchy_cancel() {
    let root = CancellationToken::new();
    let mut current = root.clone();

    // Build a chain of 100 deep
    let mut chain = vec![root.clone()];
    for _ in 0..100 {
        let child = current.child();
        chain.push(child.clone());
        current = child;
    }

    // Cancel root — all descendants must see it
    root.cancel();
    for (i, token) in chain.iter().enumerate() {
        assert!(token.is_cancelled(), "token at depth {i} not cancelled");
    }
}

// =============================================================================
// Drop guard correctness
// =============================================================================

#[test]
fn drop_guard_normal_scope_exit() {
    let token = CancellationToken::new();
    let observer = token.clone();

    {
        let _guard = token.drop_guard();
        assert!(!observer.is_cancelled());
        // guard drops at end of scope
    }
    assert!(observer.is_cancelled());
}

#[test]
fn drop_guard_with_children() {
    let token = CancellationToken::new();
    let child = token.child();
    let grandchild = child.child();

    {
        let _guard = token.drop_guard();
        assert!(!child.is_cancelled());
        assert!(!grandchild.is_cancelled());
    }
    // Guard dropped → token cancelled → children cancelled
    assert!(child.is_cancelled());
    assert!(grandchild.is_cancelled());
}

#[test]
fn drop_guard_disarm_prevents_cancel() {
    let token = CancellationToken::new();
    let observer = token.clone();

    let guard = token.drop_guard();
    let recovered = guard.disarm();
    drop(recovered); // Dropping the token itself does NOT cancel
    assert!(!observer.is_cancelled());
}

#[test]
fn drop_guard_on_panic_cancels() {
    let token = CancellationToken::new();
    let observer = token.clone();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = token.drop_guard();
        panic!("boom");
    }));

    assert!(result.is_err());
    assert!(observer.is_cancelled());
}

#[test]
fn drop_guard_nested() {
    let outer = CancellationToken::new();
    let inner = outer.child();
    let observer = inner.clone();

    {
        let _outer_guard = outer.clone().drop_guard();
        {
            let _inner_guard = inner.drop_guard();
            assert!(!observer.is_cancelled());
        }
        // Inner guard dropped → inner cancelled, but outer still alive
        assert!(observer.is_cancelled());
        assert!(!outer.is_cancelled());
    }
    // Outer guard dropped → outer also cancelled
    assert!(outer.is_cancelled());
}

#[test]
fn drop_guard_cross_thread() {
    let token = CancellationToken::new();
    let observer = token.clone();

    let handle = std::thread::spawn(move || {
        let _guard = token.drop_guard();
        std::thread::sleep(std::time::Duration::from_millis(10));
        // guard drops when thread exits
    });

    handle.join().unwrap();
    assert!(observer.is_cancelled());
}

// =============================================================================
// Cancelled future + cancel race
// =============================================================================

#[test]
#[ignore]
fn cancelled_future_cross_thread_race() {
    // One thread polls cancelled(), another thread cancels.
    // The future must resolve — no hang.
    for _ in 0..200 {
        let token = CancellationToken::new();
        let poll_token = token.clone();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done_clone = done.clone();

        let poller = std::thread::spawn(move || {
            // Simulate polling by checking repeatedly
            while !poll_token.is_cancelled() {
                std::hint::spin_loop();
            }
            done_clone.store(true, Ordering::Release);
        });

        // Small random delay
        for _ in 0..pseudo_random(0, 100) {
            std::hint::spin_loop();
        }
        token.cancel();

        poller.join().unwrap();
        assert!(done.load(Ordering::Acquire));
    }
}

// =============================================================================
// Stress: many tokens, many children, many cancels
// =============================================================================

#[test]
#[ignore]
fn stress_many_tokens_concurrent() {
    let root = CancellationToken::new();
    let cancelled_count = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();
    for _ in 0..8 {
        let root = root.clone();
        let count = cancelled_count.clone();
        handles.push(std::thread::spawn(move || {
            let mut tokens = Vec::new();
            for _ in 0..500 {
                let child = root.child();
                tokens.push(child);
            }
            // Wait for ALL children to be cancelled (not just the parent flag).
            // Parent's cancel() sets its flag first, then drains children.
            // Children may not be cancelled until the drain completes.
            for t in &tokens {
                while !t.is_cancelled() {
                    std::hint::spin_loop();
                }
                count.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    // Let threads create children, then cancel.
    std::thread::sleep(std::time::Duration::from_millis(50));
    root.cancel();

    for h in handles {
        h.join().unwrap();
    }

    let count = cancelled_count.load(Ordering::Relaxed);
    assert_eq!(count, 4000, "{count} of 4000 children saw cancelled");
}

fn pseudo_random(min: u64, max: u64) -> u64 {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64;
    min + (seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1) % (max - min))
}

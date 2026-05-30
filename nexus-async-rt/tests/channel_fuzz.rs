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
//! Randomized stress tests for channel correctness.
//!
//! Uses std::thread for concurrent access. No runtime needed for
//! the try_send/try_recv paths. Runtime tests use block_on_busy.
//!
//! Run: `cargo test -p nexus-async-rt --test channel_fuzz --release`

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use nexus_async_rt::Runtime;
use nexus_rt::WorldBuilder;

// =============================================================================
// MPSC typed: concurrent senders + receiver with random drops
// =============================================================================

#[test]
#[ignore]
fn mpsc_concurrent_senders_random_drop() {
    // Many iterations to shake out races.
    for _ in 0..50 {
        let (tx, rx) = make_mpsc_channel::<u64>(64);
        let sent = Arc::new(AtomicU64::new(0));
        let received = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        // Spawn 8 sender threads, each sending random amounts then dropping.
        let mut handles = Vec::new();
        for _ in 0..8 {
            let tx = tx.clone();
            let sent = sent.clone();
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || {
                let count = pseudo_random(50, 500);
                for i in 0..count {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    match tx.try_send(i) {
                        Ok(()) => {
                            sent.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) if e.is_full() => {
                            std::hint::spin_loop();
                        }
                        Err(_) => break, // closed
                    }
                }
                // tx dropped here
            }));
        }
        drop(tx); // drop original

        // Receiver thread
        let recv_handle = {
            let received = received.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                loop {
                    match rx.try_recv() {
                        Ok(_) => {
                            received.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(super_err) => {
                            if matches!(super_err, nexus_async_rt::channel::TryRecvError::Closed) {
                                break;
                            }
                            // Empty — spin
                            std::hint::spin_loop();
                        }
                    }
                }
                stop.store(true, Ordering::Relaxed);
            })
        };

        for h in handles {
            h.join().unwrap();
        }
        recv_handle.join().unwrap();

        // Received should equal sent (no lost messages).
        let s = sent.load(Ordering::Relaxed);
        let r = received.load(Ordering::Relaxed);
        assert_eq!(s, r, "sent {s} != received {r}");
    }
}

#[test]
#[ignore]
fn mpsc_receiver_drop_mid_stream() {
    for _ in 0..50 {
        let (tx, rx) = make_mpsc_channel::<u64>(32);

        let sender_handle = {
            let tx = tx.clone();
            std::thread::spawn(move || {
                for i in 0..10_000u64 {
                    match tx.try_send(i) {
                        Ok(()) => {}
                        Err(e) if e.is_full() => {
                            std::hint::spin_loop();
                        }
                        Err(_) => return, // closed — expected
                    }
                }
            })
        };
        drop(tx);

        // Receive some, then drop receiver mid-stream.
        let drop_after = pseudo_random(10, 500);
        let mut count = 0;
        loop {
            match rx.try_recv() {
                Ok(_) => {
                    count += 1;
                    if count >= drop_after {
                        break;
                    }
                }
                Err(nexus_async_rt::channel::TryRecvError::Empty) => {
                    std::hint::spin_loop();
                }
                Err(nexus_async_rt::channel::TryRecvError::Closed) => break,
            }
        }
        drop(rx); // sender should see closed

        sender_handle.join().unwrap(); // must not hang
    }
}

// =============================================================================
// SPSC typed: fast producer, slow consumer with tiny buffer
// =============================================================================

#[test]
#[ignore]
fn spsc_fast_producer_slow_consumer() {
    for _ in 0..20 {
        let (tx, rx) = make_spsc_channel::<u64>(4);

        let producer = std::thread::spawn(move || {
            for i in 0..5_000u64 {
                loop {
                    match tx.try_send(i) {
                        Ok(()) => break,
                        Err(e) if e.is_full() => std::hint::spin_loop(),
                        Err(_) => return,
                    }
                }
            }
        });

        let consumer = std::thread::spawn(move || {
            let mut last = None;
            let mut count = 0u64;
            loop {
                match rx.try_recv() {
                    Ok(val) => {
                        // Verify ordering.
                        if let Some(prev) = last {
                            assert_eq!(val, prev + 1, "FIFO violation");
                        }
                        last = Some(val);
                        count += 1;
                        if count >= 5_000 {
                            break;
                        }
                    }
                    Err(nexus_async_rt::channel::TryRecvError::Empty) => {
                        std::hint::spin_loop();
                    }
                    Err(nexus_async_rt::channel::TryRecvError::Closed) => break,
                }
            }
            assert_eq!(count, 5_000);
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }
}

// =============================================================================
// SPSC bytes: random message sizes
// =============================================================================

#[test]
#[ignore]
fn spsc_bytes_random_sizes() {
    for _ in 0..20 {
        let (mut tx, mut rx) = make_spsc_bytes_channel(4096);
        let msg_count = 1000;

        let producer = std::thread::spawn(move || {
            for i in 0u32..msg_count {
                let size = 1 + (pseudo_random_seeded(i as u64, 1, 200) as usize);
                let data: Vec<u8> = (0..size).map(|j| ((i as usize + j) & 0xFF) as u8).collect();
                loop {
                    match tx.try_claim(size) {
                        Ok(mut claim) => {
                            claim.copy_from_slice(&data);
                            claim.commit();
                            break;
                        }
                        Err(nexus_logbuf::BufferFull) => {
                            std::hint::spin_loop();
                        }
                    }
                }
            }
        });

        let consumer = std::thread::spawn(move || {
            let mut count = 0u32;
            loop {
                if let Some(msg) = rx.try_recv() {
                    let size = 1 + (pseudo_random_seeded(count as u64, 1, 200) as usize);
                    assert_eq!(msg.len(), size, "size mismatch at msg {count}");
                    count += 1;
                } else {
                    if count >= msg_count {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }
}

// =============================================================================
// MPSC bytes: concurrent senders, random sizes
// =============================================================================

#[test]
#[ignore]
fn mpsc_bytes_concurrent_random_sizes() {
    for _ in 0..10 {
        let (tx, mut rx) = make_mpsc_bytes_channel(8192);
        let total_per_sender = 200u32;
        let num_senders = 4;
        let sent = Arc::new(AtomicU64::new(0));

        let mut handles = Vec::new();
        for sender_id in 0..num_senders {
            let mut tx = tx.clone();
            let sent = sent.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..total_per_sender {
                    let size = 1
                        + (pseudo_random_seeded((sender_id as u64) * 10_000 + i as u64, 1, 100)
                            as usize);
                    let data = vec![(sender_id as u8).wrapping_add(i as u8); size];
                    loop {
                        match tx.try_claim(size) {
                            Ok(mut claim) => {
                                claim.copy_from_slice(&data);
                                claim.commit();
                                sent.fetch_add(1, Ordering::Relaxed);
                                break;
                            }
                            Err(nexus_logbuf::BufferFull) => {
                                std::hint::spin_loop();
                            }
                        }
                    }
                }
            }));
        }
        drop(tx);

        let mut received = 0u64;
        let expected = (num_senders as u64) * (total_per_sender as u64);
        loop {
            if let Some(msg) = rx.try_recv() {
                assert!(!msg.is_empty());
                received += 1;
                continue;
            }
            // Empty.
            if received >= expected {
                break;
            }
            let all_joined = handles.iter().all(std::thread::JoinHandle::is_finished);
            if all_joined {
                // One more drain attempt.
                if rx.try_recv().is_none() {
                    break;
                }
                received += 1;
            } else {
                std::hint::spin_loop();
            }
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(sent.load(Ordering::Relaxed), received, "sent != received");
    }
}

// =============================================================================
// MPSC: rapid clone + send + drop cycle
// =============================================================================

#[test]
#[ignore]
fn mpsc_rapid_clone_drop() {
    for _ in 0..50 {
        let (tx, rx) = make_mpsc_channel::<u64>(128);
        let sent = Arc::new(AtomicU64::new(0));

        let mut handles = Vec::new();
        for _ in 0..16 {
            let tx = tx.clone();
            let sent = sent.clone();
            handles.push(std::thread::spawn(move || {
                // Clone, send a few, drop. Repeat.
                for batch in 0..10 {
                    let tx2 = tx.clone();
                    for i in 0..10 {
                        match tx2.try_send(batch * 10 + i) {
                            Ok(()) => {
                                sent.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(e) if e.is_full() => {}
                            Err(_) => return,
                        }
                    }
                    // tx2 dropped — exercises sender_count + cancelled nodes
                }
            }));
        }
        drop(tx);

        for h in handles {
            h.join().unwrap();
        }

        // Drain.
        let mut received = 0u64;
        while rx.try_recv().is_ok() {
            received += 1;
        }

        assert_eq!(sent.load(Ordering::Relaxed), received);
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Cheap deterministic pseudo-random in [min, max).
fn pseudo_random(min: u64, max: u64) -> u64 {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64;
    min + (seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1) % (max - min))
}

fn pseudo_random_seeded(seed: u64, min: u64, max: u64) -> u64 {
    min + (seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1) % (max - min))
}

/// Create MPSC channel without runtime context (for cross-thread tests).
fn make_mpsc_channel<T: Send + 'static>(
    capacity: usize,
) -> (
    nexus_async_rt::channel::mpsc::Sender<T>,
    nexus_async_rt::channel::mpsc::Receiver<T>,
) {
    // We need runtime context for channel creation. Create a minimal runtime,
    // create the channel inside block_on, extract via Rc.
    use std::cell::Cell;
    use std::rc::Rc;

    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let result = Rc::new(Cell::new(None));
    let r = result.clone();

    rt.block_on(async move {
        let pair = nexus_async_rt::channel::mpsc::channel::<T>(capacity);
        r.set(Some(pair));
    });

    result.take().unwrap()
}

fn make_spsc_channel<T: Send + 'static>(
    capacity: usize,
) -> (
    nexus_async_rt::channel::spsc::Sender<T>,
    nexus_async_rt::channel::spsc::Receiver<T>,
) {
    use std::cell::Cell;
    use std::rc::Rc;

    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let result = Rc::new(Cell::new(None));
    let r = result.clone();

    rt.block_on(async move {
        let pair = nexus_async_rt::channel::spsc::channel::<T>(capacity);
        r.set(Some(pair));
    });

    result.take().unwrap()
}

fn make_spsc_bytes_channel(
    capacity: usize,
) -> (
    nexus_async_rt::channel::spsc_bytes::Sender,
    nexus_async_rt::channel::spsc_bytes::Receiver,
) {
    use std::cell::Cell;
    use std::rc::Rc;

    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let result = Rc::new(Cell::new(None));
    let r = result.clone();

    rt.block_on(async move {
        let pair = nexus_async_rt::channel::spsc_bytes::channel(capacity);
        r.set(Some(pair));
    });

    result.take().unwrap()
}

fn make_mpsc_bytes_channel(
    capacity: usize,
) -> (
    nexus_async_rt::channel::mpsc_bytes::Sender,
    nexus_async_rt::channel::mpsc_bytes::Receiver,
) {
    use std::cell::Cell;
    use std::rc::Rc;

    let wb = WorldBuilder::new();
    let mut world = wb.build();
    let mut rt = Runtime::new(&mut world);

    let result = Rc::new(Cell::new(None));
    let r = result.clone();

    rt.block_on(async move {
        let pair = nexus_async_rt::channel::mpsc_bytes::channel(capacity);
        r.set(Some(pair));
    });

    result.take().unwrap()
}

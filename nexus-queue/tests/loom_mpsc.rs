#![cfg(loom)]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::thread;
use nexus_queue::mpsc;

#[test]
fn no_lost_items_two_producers() {
    loom::model(|| {
        let (tx, rx) = mpsc::ring_buffer::<u32>(2);
        let tx2 = tx.clone();

        let t1 = thread::spawn(move || {
            while tx.push(1).is_err() {
                thread::yield_now();
            }
        });

        let t2 = thread::spawn(move || {
            while tx2.push(2).is_err() {
                thread::yield_now();
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        let mut received = Vec::new();
        while let Some(v) = rx.pop() {
            received.push(v);
        }

        received.sort_unstable();
        assert_eq!(received, vec![1, 2]);
    });
}

#[test]
fn no_duplicates() {
    loom::model(|| {
        let (tx, rx) = mpsc::ring_buffer::<u32>(4);
        let tx2 = tx.clone();

        let t1 = thread::spawn(move || {
            while tx.push(10).is_err() {
                thread::yield_now();
            }
            while tx.push(11).is_err() {
                thread::yield_now();
            }
        });

        let t2 = thread::spawn(move || {
            while tx2.push(20).is_err() {
                thread::yield_now();
            }
            while tx2.push(21).is_err() {
                thread::yield_now();
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        let mut received = Vec::new();
        while let Some(v) = rx.pop() {
            received.push(v);
        }

        assert_eq!(received.len(), 4);
        received.sort_unstable();
        assert_eq!(received, vec![10, 11, 20, 21]);
    });
}

#[test]
fn concurrent_push_pop() {
    loom::model(|| {
        let (tx, rx) = mpsc::ring_buffer::<u32>(2);
        let tx2 = tx.clone();

        let t1 = thread::spawn(move || {
            while tx.push(1).is_err() {
                thread::yield_now();
            }
        });

        let t2 = thread::spawn(move || {
            while tx2.push(2).is_err() {
                thread::yield_now();
            }
        });

        let mut received = Vec::new();
        while received.len() < 2 {
            if let Some(v) = rx.pop() {
                received.push(v);
            } else {
                thread::yield_now();
            }
        }

        t1.join().unwrap();
        t2.join().unwrap();

        received.sort_unstable();
        assert_eq!(received, vec![1, 2]);
    });
}

#[test]
fn drop_with_pending_items() {
    loom::model(|| {
        let (tx, rx) = mpsc::ring_buffer::<Arc<u32>>(2);
        let val = Arc::new(42);

        let tx2 = tx.clone();
        tx.push(Arc::clone(&val)).unwrap();
        tx2.push(Arc::clone(&val)).unwrap();

        drop(tx);
        drop(tx2);
        drop(rx);

        assert_eq!(Arc::strong_count(&val), 1);
    });
}

#[test]
fn single_slot_two_producers() {
    loom::model(|| {
        let (tx, rx) = mpsc::ring_buffer::<u32>(1);
        let tx2 = tx.clone();
        let count = Arc::new(AtomicUsize::new(0));
        let count1 = Arc::clone(&count);
        let count2 = Arc::clone(&count);

        let t1 = thread::spawn(move || {
            if tx.push(1).is_ok() {
                count1.fetch_add(1, Ordering::Relaxed);
            }
        });

        let t2 = thread::spawn(move || {
            if tx2.push(2).is_ok() {
                count2.fetch_add(1, Ordering::Relaxed);
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        let pushed = count.load(Ordering::SeqCst);
        let mut popped = 0;
        while rx.pop().is_some() {
            popped += 1;
        }

        assert!(popped >= 1);
        assert_eq!(popped, pushed);
    });
}

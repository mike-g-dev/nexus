#![cfg(loom)]

use loom::thread;
use nexus_slot::spsc;

#[test]
fn write_then_read() {
    loom::model(|| {
        let (mut writer, mut reader) = spsc::slot::<usize>();

        let w = thread::spawn(move || {
            writer.write(42);
        });

        w.join().unwrap();
        assert_eq!(reader.read(), Some(42));
        assert!(reader.read().is_none());
    });
}

#[test]
fn no_torn_reads() {
    loom::model(|| {
        let (mut writer, mut reader) = spsc::slot::<usize>();

        let w = thread::spawn(move || {
            writer.write(42);
        });

        loop {
            if let Some(v) = reader.read() {
                assert_eq!(v, 42);
                break;
            }
            if reader.is_disconnected() {
                if let Some(v) = reader.read() {
                    assert_eq!(v, 42);
                }
                break;
            }
            thread::yield_now();
        }

        w.join().unwrap();
    });
}

#[test]
fn two_writes_no_torn_reads() {
    loom::model(|| {
        let (mut writer, mut reader) = spsc::slot::<usize>();

        let w = thread::spawn(move || {
            writer.write(0xAAAA);
            writer.write(0xBBBB);
        });

        loop {
            if let Some(v) = reader.read() {
                assert!(v == 0xAAAA || v == 0xBBBB, "torn read: {v:#x}");
            }
            if reader.is_disconnected() && !reader.has_update() {
                break;
            }
            thread::yield_now();
        }

        w.join().unwrap();
    });
}

#[test]
fn conflation() {
    loom::model(|| {
        let (mut writer, mut reader) = spsc::slot::<usize>();

        writer.write(1);
        writer.write(2);
        writer.write(3);

        assert_eq!(reader.read(), Some(3));
        assert!(reader.read().is_none());
    });
}

#[test]
fn disconnect_writer() {
    loom::model(|| {
        let (writer, reader) = spsc::slot::<usize>();
        assert!(!reader.is_disconnected());
        drop(writer);
        assert!(reader.is_disconnected());
    });
}

#[test]
fn disconnect_reader() {
    loom::model(|| {
        let (writer, reader) = spsc::slot::<usize>();
        assert!(!writer.is_disconnected());
        drop(reader);
        assert!(writer.is_disconnected());
    });
}

#[test]
fn read_versioned_consistent() {
    loom::model(|| {
        let (mut writer, mut reader) = spsc::slot::<usize>();

        let w = thread::spawn(move || {
            writer.write(42);
        });

        loop {
            if let Some((v, _ver)) = reader.read_versioned() {
                assert_eq!(v, 42);
                break;
            }
            if reader.is_disconnected() {
                if let Some((v, _ver)) = reader.read_versioned() {
                    assert_eq!(v, 42);
                }
                break;
            }
            thread::yield_now();
        }

        w.join().unwrap();
    });
}

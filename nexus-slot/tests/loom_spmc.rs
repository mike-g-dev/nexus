#![cfg(loom)]

use loom::thread;
use nexus_slot::spmc;

#[test]
fn write_then_read() {
    loom::model(|| {
        let (mut writer, mut reader) = spmc::shared_slot::<usize>();

        let w = thread::spawn(move || {
            writer.write(42);
        });

        w.join().unwrap();
        assert_eq!(reader.read(), Some(42));
    });
}

#[test]
fn no_torn_reads() {
    loom::model(|| {
        let (mut writer, mut reader) = spmc::shared_slot::<usize>();

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
fn writer_disconnect_ordering() {
    // Writer's Drop stores writer_alive=false with Release.
    // Reader's is_disconnected loads with Acquire.
    // If the reader sees disconnected, the writer's final write must be visible.
    loom::model(|| {
        let (mut writer, mut reader) = spmc::shared_slot::<usize>();

        let w = thread::spawn(move || {
            writer.write(42);
        });

        let mut seen = false;
        loop {
            if reader.is_disconnected() {
                if reader.has_update() {
                    let v = reader.read().unwrap();
                    assert_eq!(v, 42);
                    seen = true;
                }
                break;
            }
            if let Some(v) = reader.read() {
                assert_eq!(v, 42);
                seen = true;
            }
            thread::yield_now();
        }

        assert!(seen, "writer wrote but reader never observed the value");

        w.join().unwrap();
    });
}

#[test]
fn disconnect_all_readers() {
    loom::model(|| {
        let (writer, reader1) = spmc::shared_slot::<usize>();
        let reader2 = reader1.clone();

        assert!(!writer.is_disconnected());
        drop(reader1);
        assert!(!writer.is_disconnected());
        drop(reader2);
        assert!(writer.is_disconnected());
    });
}

#[test]
fn cloned_reader_disconnect() {
    loom::model(|| {
        let (writer, reader) = spmc::shared_slot::<usize>();
        let reader2 = reader.clone();

        assert!(!reader.is_disconnected());
        assert!(!reader2.is_disconnected());

        drop(writer);

        assert!(reader.is_disconnected());
        assert!(reader2.is_disconnected());
    });
}

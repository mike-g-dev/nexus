//! FrameReader benchmarks — inbound path: read() + next() → Message.
//!
//! Baseline before optimizations. Run with:
//!   cargo bench -p nexus-web --bench frame_reader

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use nexus_web::ws::{FrameReader, Role, apply_mask};

// =============================================================================
// Helpers
// =============================================================================

fn make_frame(fin: bool, opcode: u8, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::new();
    let byte0 = if fin { 0x80 } else { 0x00 } | opcode;
    frame.push(byte0);
    if payload.len() <= 125 {
        frame.push(payload.len() as u8);
    } else if payload.len() <= 65535 {
        frame.push(126);
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    frame.extend_from_slice(payload);
    frame
}

fn make_masked_frame(fin: bool, opcode: u8, payload: &[u8], mask: [u8; 4]) -> Vec<u8> {
    let mut frame = Vec::new();
    let byte0 = if fin { 0x80 } else { 0x00 } | opcode;
    frame.push(byte0);
    let len_byte = if payload.len() <= 125 {
        payload.len() as u8
    } else if payload.len() <= 65535 {
        126
    } else {
        127
    };
    frame.push(0x80 | len_byte);
    if payload.len() > 125 && payload.len() <= 65535 {
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else if payload.len() > 65535 {
        frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    frame.extend_from_slice(&mask);
    let mut masked = payload.to_vec();
    apply_mask(&mut masked, mask);
    frame.extend_from_slice(&masked);
    frame
}

// =============================================================================
// Group 1: Single-frame by size (hot path)
// =============================================================================

fn bench_text_unmasked(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_unmasked");
    for size in [32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let payload = vec![b'x'; size];
        let frame = make_frame(true, 0x1, &payload);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &frame, |b, frame| {
            let mut reader = FrameReader::builder()
                .role(Role::Client)
                .buffer_capacity(64 * 1024)
                .build();
            b.iter(|| {
                reader.read(frame).unwrap();
                let msg = reader.next().unwrap().unwrap();
                black_box(&msg);
            });
        });
    }
    group.finish();
}

fn bench_binary_unmasked(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_unmasked");
    for size in [32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let payload = vec![0x42u8; size];
        let frame = make_frame(true, 0x2, &payload);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &frame, |b, frame| {
            let mut reader = FrameReader::builder()
                .role(Role::Client)
                .buffer_capacity(64 * 1024)
                .build();
            b.iter(|| {
                reader.read(frame).unwrap();
                let msg = reader.next().unwrap().unwrap();
                black_box(&msg);
            });
        });
    }
    group.finish();
}

fn bench_text_masked(c: &mut Criterion) {
    let mask = [0x37, 0xFA, 0x21, 0x3D];
    let mut group = c.benchmark_group("text_masked");
    for size in [32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let payload = vec![b'x'; size];
        let frame = make_masked_frame(true, 0x1, &payload, mask);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &frame, |b, frame| {
            let mut reader = FrameReader::builder()
                .role(Role::Server)
                .buffer_capacity(64 * 1024)
                .build();
            b.iter(|| {
                reader.read(frame).unwrap();
                let msg = reader.next().unwrap().unwrap();
                black_box(&msg);
            });
        });
    }
    group.finish();
}

// =============================================================================
// Group 2: Fragment assembly
// =============================================================================

fn bench_assembly_2_fragments(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly_2frag");
    for size in [128, 512, 2048] {
        let half = size / 2;
        let payload = vec![b'x'; size];
        let mut wire = make_frame(false, 0x2, &payload[..half]);
        wire.extend(&make_frame(true, 0x0, &payload[half..]));
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &wire, |b, wire| {
            let mut reader = FrameReader::builder()
                .role(Role::Client)
                .buffer_capacity(64 * 1024)
                .build();
            b.iter(|| {
                reader.read(wire).unwrap();
                let msg = reader.next().unwrap().unwrap();
                black_box(&msg);
            });
        });
    }
    group.finish();
}

// =============================================================================
// Group 3: Throughput (messages per second)
// =============================================================================

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    for msg_count in [10, 100, 1000] {
        let payload = vec![0x42u8; 128];
        let frame = make_frame(true, 0x2, &payload);
        let mut wire = Vec::new();
        for _ in 0..msg_count {
            wire.extend(&frame);
        }
        group.throughput(Throughput::Elements(msg_count as u64));
        group.bench_with_input(
            BenchmarkId::new("binary_128B", msg_count),
            &wire,
            |b, wire| {
                let mut reader = FrameReader::builder()
                    .role(Role::Client)
                    .buffer_capacity(wire.len() + 4096)
                    .build();
                b.iter(|| {
                    reader.read(wire).unwrap();
                    let mut count = 0;
                    while let Some(msg) = reader.next().unwrap() {
                        black_box(&msg);
                        count += 1;
                    }
                    assert_eq!(count, msg_count);
                });
            },
        );
    }
    group.finish();
}

// =============================================================================
// Group 4: Component isolation
// =============================================================================

fn bench_unmask(c: &mut Criterion) {
    let mut group = c.benchmark_group("unmask");
    for size in [64, 128, 256, 512, 1024, 4096] {
        let mut data = vec![0x42u8; size];
        let mask = [0x37, 0xFA, 0x21, 0x3D];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                apply_mask(&mut data, mask);
                black_box(&data);
            });
        });
    }
    group.finish();
}

fn bench_utf8_validate(c: &mut Criterion) {
    let mut group = c.benchmark_group("utf8_validate");
    for size in [64, 128, 256, 512, 1024, 4096] {
        let data = vec![b'x'; size]; // valid ASCII
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                black_box(std::str::from_utf8(data).unwrap());
            });
        });
    }
    group.finish();
}

// =============================================================================
// Criterion setup
// =============================================================================

criterion_group!(
    single_frame,
    bench_text_unmasked,
    bench_binary_unmasked,
    bench_text_masked,
);

criterion_group!(assembly, bench_assembly_2_fragments);
criterion_group!(throughput, bench_throughput);
criterion_group!(components, bench_unmask, bench_utf8_validate);

criterion_main!(single_frame, assembly, throughput, components);

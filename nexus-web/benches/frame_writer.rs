//! FrameWriter benchmarks — outbound path: encode → wire bytes.
//!
//! Run with:
//!   cargo bench -p nexus-web --bench frame_writer

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use nexus_web::buf::WriteBuf;
use nexus_web::ws::{FrameWriter, Role};

fn bench_encode_text_server(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_text_server");
    for size in [32, 128, 512, 2048, 4096] {
        let mut writer = FrameWriter::new(Role::Server);
        let payload = vec![b'x'; size];
        let mut dst = vec![0u8; writer.max_encoded_len(size)];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &payload, |b, payload| {
            b.iter(|| {
                let n = writer.encode_text(payload, &mut dst);
                black_box(n);
            });
        });
    }
    group.finish();
}

fn bench_encode_text_client(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_text_client");
    for size in [32, 128, 512, 2048, 4096] {
        let mut writer = FrameWriter::new(Role::Client);
        let payload = vec![b'x'; size];
        let mut dst = vec![0u8; writer.max_encoded_len(size)];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &payload, |b, payload| {
            b.iter(|| {
                let n = writer.encode_text(payload, &mut dst);
                black_box(n);
            });
        });
    }
    group.finish();
}

fn bench_encode_into_writebuf(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_into_writebuf");
    for size in [32, 128, 512, 2048, 4096] {
        let mut writer = FrameWriter::new(Role::Server);
        let payload = vec![b'x'; size];
        let mut wbuf = WriteBuf::new(size + 14, 14);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &payload, |b, payload| {
            b.iter(|| {
                writer.encode_text_into(payload, &mut wbuf);
                black_box(wbuf.data());
            });
        });
    }
    group.finish();
}

fn bench_encode_writer(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_text_writer");
    for size in [32, 128, 512, 2048, 4096] {
        let mut writer = FrameWriter::new(Role::Server);
        let payload = vec![b'x'; size];
        let mut wbuf = WriteBuf::new(size + 14, 14);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &payload, |b, payload| {
            b.iter(|| {
                writer
                    .encode_text_writer(&mut wbuf, |w| {
                        use std::io::Write;
                        w.write_all(payload)
                    })
                    .unwrap();
                black_box(wbuf.data());
            });
        });
    }
    group.finish();
}

fn bench_encode_fixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_text_fixed");
    for size in [32, 128, 512, 2048, 4096] {
        let mut writer = FrameWriter::new(Role::Server);
        let payload = vec![b'x'; size];
        let mut wbuf = WriteBuf::new(size + 14, 14);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &payload, |b, payload| {
            b.iter(|| {
                writer.encode_text_fixed(&mut wbuf, size, |buf| {
                    buf.copy_from_slice(payload);
                });
                black_box(wbuf.data());
            });
        });
    }
    group.finish();
}

criterion_group!(
    encode,
    bench_encode_text_server,
    bench_encode_text_client,
    bench_encode_into_writebuf,
    bench_encode_writer,
    bench_encode_fixed,
);

criterion_main!(encode);

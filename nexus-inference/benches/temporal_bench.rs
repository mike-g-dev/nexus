use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nexus_inference::{Activation, Causal1dConvF32, TinyGruF32, TinyLstmF32};

fn make_lstm(input: usize, hidden: usize, output: usize) -> TinyLstmF32 {
    let gc = 4 * hidden;
    let weight_ih = vec![0.1_f32; gc * input];
    let weight_hh = vec![0.1_f32; gc * hidden];
    let bias_ih = vec![0.0_f32; gc];
    let bias_hh = vec![0.0_f32; gc];
    let w_out = vec![0.1_f32; output * hidden];
    let b_out = vec![0.0_f32; output];
    TinyLstmF32::from_parts(
        input, hidden, output, &weight_ih, &weight_hh, &bias_ih, &bias_hh, &w_out, &b_out,
    )
    .unwrap()
}

fn make_gru(input: usize, hidden: usize, output: usize) -> TinyGruF32 {
    let gc = 3 * hidden;
    let weight_ih = vec![0.1_f32; gc * input];
    let weight_hh = vec![0.1_f32; gc * hidden];
    let bias_ih = vec![0.0_f32; gc];
    let bias_hh = vec![0.0_f32; gc];
    let w_out = vec![0.1_f32; output * hidden];
    let b_out = vec![0.0_f32; output];
    TinyGruF32::from_parts(
        input, hidden, output, &weight_ih, &weight_hh, &bias_ih, &bias_hh, &w_out, &b_out,
    )
    .unwrap()
}

fn make_conv(input_ch: usize, kernel: usize, filters: usize, output: usize) -> Causal1dConvF32 {
    let w_conv = vec![0.1_f32; filters * kernel * input_ch];
    let b_conv = vec![0.0_f32; filters];
    let w_out = vec![0.1_f32; output * filters];
    let b_out = vec![0.0_f32; output];
    Causal1dConvF32::from_parts(
        input_ch,
        kernel,
        filters,
        output,
        &w_conv,
        &b_conv,
        &w_out,
        &b_out,
        Activation::Relu,
    )
    .unwrap()
}

fn bench_lstm(c: &mut Criterion) {
    let input_4 = vec![0.5_f32; 4];
    let input_8 = vec![0.5_f32; 8];
    let input_16 = vec![0.5_f32; 16];

    let mut m = make_lstm(4, 8, 1);
    for _ in 0..100 {
        m.step(&input_4);
    } // warm hidden state
    c.bench_function("LSTM 4→8→1", |b| {
        b.iter(|| m.step(black_box(&input_4)));
    });

    let mut m = make_lstm(8, 16, 1);
    for _ in 0..100 {
        m.step(&input_8);
    }
    c.bench_function("LSTM 8→16→1", |b| {
        b.iter(|| m.step(black_box(&input_8)));
    });

    let mut m = make_lstm(8, 32, 1);
    for _ in 0..100 {
        m.step(&input_8);
    }
    c.bench_function("LSTM 8→32→1", |b| {
        b.iter(|| m.step(black_box(&input_8)));
    });

    let mut m = make_lstm(16, 64, 1);
    for _ in 0..100 {
        m.step(&input_16);
    }
    c.bench_function("LSTM 16→64→1", |b| {
        b.iter(|| m.step(black_box(&input_16)));
    });
}

fn bench_gru(c: &mut Criterion) {
    let input_8 = vec![0.5_f32; 8];
    let input_16 = vec![0.5_f32; 16];

    let mut m = make_gru(8, 16, 1);
    for _ in 0..100 {
        m.step(&input_8);
    }
    c.bench_function("GRU 8→16→1", |b| {
        b.iter(|| m.step(black_box(&input_8)));
    });

    let mut m = make_gru(8, 32, 1);
    for _ in 0..100 {
        m.step(&input_8);
    }
    c.bench_function("GRU 8→32→1", |b| {
        b.iter(|| m.step(black_box(&input_8)));
    });

    let mut m = make_gru(16, 64, 1);
    for _ in 0..100 {
        m.step(&input_16);
    }
    c.bench_function("GRU 16→64→1", |b| {
        b.iter(|| m.step(black_box(&input_16)));
    });
}

fn bench_conv(c: &mut Criterion) {
    let input_4 = vec![0.5_f32; 4];
    let input_8 = vec![0.5_f32; 8];

    let mut m = make_conv(4, 4, 8, 1);
    for _ in 0..10 {
        m.step(&input_4);
    } // prime buffer
    c.bench_function("Conv 4ch×4k×8f→1", |b| {
        b.iter(|| m.step(black_box(&input_4)));
    });

    let mut m = make_conv(4, 8, 16, 1);
    for _ in 0..10 {
        m.step(&input_4);
    }
    c.bench_function("Conv 4ch×8k×16f→1", |b| {
        b.iter(|| m.step(black_box(&input_4)));
    });

    let mut m = make_conv(8, 8, 32, 1);
    for _ in 0..10 {
        m.step(&input_8);
    }
    c.bench_function("Conv 8ch×8k×32f→1", |b| {
        b.iter(|| m.step(black_box(&input_8)));
    });
}

criterion_group!(benches, bench_lstm, bench_gru, bench_conv);
criterion_main!(benches);

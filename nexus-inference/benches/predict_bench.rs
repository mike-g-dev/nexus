// Benchmark code prioritizes legibility over lint-cleanliness.
#![allow(
    clippy::format_push_string,
    clippy::many_single_char_names,
    clippy::redundant_closure_for_method_calls,
    clippy::suboptimal_flops,
    clippy::unreadable_literal,
    unused_mut
)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nexus_inference::{Activation, Bnn, Gbdt, Lut, Mlp, QuantizedMlp, TinyTcn};

const LIGHTGBM_HEADER: &str = "\
tree
version=v4
num_class=1
num_tree_per_iteration=1
";

fn build_lightgbm_model(n_trees: usize, depth: usize, n_features: usize) -> String {
    let mut s = String::from(LIGHTGBM_HEADER);
    s.push_str(&format!("max_feature_idx={}\n", n_features - 1));
    s.push_str("average_output=0.0\n\n");

    for t in 0..n_trees {
        let num_leaves = 1usize << depth;
        let num_internal = num_leaves - 1;

        s.push_str(&format!("Tree={t}\n"));
        s.push_str(&format!("num_leaves={num_leaves}\n"));
        s.push_str("num_cat=0\n");

        // split_feature: cycle through features
        s.push_str("split_feature=");
        for i in 0..num_internal {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&format!("{}", (t + i) % n_features));
        }
        s.push('\n');

        // threshold: spread evenly
        s.push_str("threshold=");
        for i in 0..num_internal {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&format!("{:.1}", (i as f64 + 1.0) * 0.5));
        }
        s.push('\n');

        // decision_type: all 0
        s.push_str("decision_type=");
        for i in 0..num_internal {
            if i > 0 {
                s.push(' ');
            }
            s.push('0');
        }
        s.push('\n');

        // BFS-indexed children for a complete binary tree
        s.push_str("left_child=");
        for i in 0..num_internal {
            if i > 0 {
                s.push(' ');
            }
            let left = 2 * i + 1;
            if left < num_internal {
                s.push_str(&format!("{left}"));
            } else {
                let leaf_idx = left - num_internal;
                s.push_str(&format!("-{}", leaf_idx + 1));
            }
        }
        s.push('\n');

        s.push_str("right_child=");
        for i in 0..num_internal {
            if i > 0 {
                s.push(' ');
            }
            let right = 2 * i + 2;
            if right < num_internal {
                s.push_str(&format!("{right}"));
            } else {
                let leaf_idx = right - num_internal;
                s.push_str(&format!("-{}", leaf_idx + 1));
            }
        }
        s.push('\n');

        // leaf_value: small values
        s.push_str("leaf_value=");
        for i in 0..num_leaves {
            if i > 0 {
                s.push(' ');
            }
            let val = (i as f64 - num_leaves as f64 / 2.0) * 0.01;
            s.push_str(&format!("{val:.4}"));
        }
        s.push('\n');

        s.push('\n');
    }

    s.push_str("end of trees\n");
    s
}

fn build_mlp_weights(layer_sizes: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    let mut seed = 42u64;
    for i in 0..layer_sizes.len() - 1 {
        let n = layer_sizes[i] * layer_sizes[i + 1];
        for _ in 0..n {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            weights.push((seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0);
        }
        for _ in 0..layer_sizes[i + 1] {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            biases.push((seed >> 33) as f64 / (1u64 << 31) as f64 * 0.1);
        }
    }
    (weights, biases)
}

fn bench_gbdt(c: &mut Criterion) {
    let features_8 = vec![0.5_f32; 8];
    let features_16 = vec![0.5_f32; 16];

    let text_50x6 = build_lightgbm_model(50, 6, 8);
    let model_50x6 = Gbdt::from_lightgbm(text_50x6.as_bytes()).unwrap();
    c.bench_function("Gbdt::predict 50x6 8feat", |b| {
        b.iter(|| model_50x6.predict(black_box(&features_8)));
    });

    let text_100x6 = build_lightgbm_model(100, 6, 8);
    let model_100x6 = Gbdt::from_lightgbm(text_100x6.as_bytes()).unwrap();
    c.bench_function("Gbdt::predict 100x6 8feat", |b| {
        b.iter(|| model_100x6.predict(black_box(&features_8)));
    });

    let text_200x8 = build_lightgbm_model(200, 8, 16);
    let model_200x8 = Gbdt::from_lightgbm(text_200x8.as_bytes()).unwrap();
    c.bench_function("Gbdt::predict 200x8 16feat", |b| {
        b.iter(|| model_200x8.predict(black_box(&features_16)));
    });

    let text_100x6b = build_lightgbm_model(100, 6, 8);
    let model_100x6b = Gbdt::from_lightgbm(text_100x6b.as_bytes()).unwrap();
    c.bench_function("Gbdt::predict (NaN-aware) 100x6 8feat", |b| {
        b.iter(|| model_100x6b.predict_nan_aware(black_box(&features_8)));
    });
}

fn bench_gbdt_random(c: &mut Criterion) {
    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    fn random_features(n: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut batches = Vec::with_capacity(1024);
        for _ in 0..1024 {
            let mut feats = Vec::with_capacity(n);
            for _ in 0..n {
                let bits = xorshift64(&mut state);
                feats.push((bits as f64 / u64::MAX as f64) as f32);
            }
            batches.push(feats);
        }
        batches
    }

    let batches_8 = random_features(8, 0xDEAD_BEEF_CAFE_F00D);
    let batches_16 = random_features(16, 0xCAFE_BABE_1234_5678);

    let text_50x6 = build_lightgbm_model(50, 6, 8);
    let model_50x6 = Gbdt::from_lightgbm(text_50x6.as_bytes()).unwrap();
    c.bench_function("Gbdt::predict 50x6 8feat (random)", |b| {
        let mut i = 0;
        b.iter(|| {
            let result = model_50x6.predict(black_box(&batches_8[i % 1024]));
            i += 1;
            result
        });
    });

    let text_100x6 = build_lightgbm_model(100, 6, 8);
    let model_100x6 = Gbdt::from_lightgbm(text_100x6.as_bytes()).unwrap();
    c.bench_function("Gbdt::predict 100x6 8feat (random)", |b| {
        let mut i = 0;
        b.iter(|| {
            let result = model_100x6.predict(black_box(&batches_8[i % 1024]));
            i += 1;
            result
        });
    });

    let text_200x8 = build_lightgbm_model(200, 8, 16);
    let model_200x8 = Gbdt::from_lightgbm(text_200x8.as_bytes()).unwrap();
    c.bench_function("Gbdt::predict 200x8 16feat (random)", |b| {
        let mut i = 0;
        b.iter(|| {
            let result = model_200x8.predict(black_box(&batches_16[i % 1024]));
            i += 1;
            result
        });
    });

    let text_100x6b = build_lightgbm_model(100, 6, 8);
    let model_100x6b = Gbdt::from_lightgbm(text_100x6b.as_bytes()).unwrap();
    c.bench_function("Gbdt::predict (NaN-aware) 100x6 8feat (random)", |b| {
        let mut i = 0;
        b.iter(|| {
            let result = model_100x6b.predict_nan_aware(black_box(&batches_8[i % 1024]));
            i += 1;
            result
        });
    });
}

fn bench_mlp_f32(c: &mut Criterion) {
    let features_8: Vec<f32> = vec![0.5; 8];
    let features_16: Vec<f32> = vec![0.5; 16];
    let features_64: Vec<f32> = vec![0.5; 64];

    let (w, b) = build_mlp_weights(&[8, 16, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = Mlp::from_parts(&[8, 16, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("Mlp::predict 8→16→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_8)));
    });

    let (w, b) = build_mlp_weights(&[16, 32, 8, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = Mlp::from_parts(&[16, 32, 8, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("Mlp::predict 16→32→8→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_16)));
    });

    let (w, b) = build_mlp_weights(&[64, 64, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = Mlp::from_parts(&[64, 64, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("Mlp::predict 64→64→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_64)));
    });

    // Deep stacked configs
    let features_32: Vec<f32> = vec![0.5; 32];

    // 32 → 32 → 32 → 32 → 1 (4 layers)
    let (w, b) = build_mlp_weights(&[32, 32, 32, 32, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = Mlp::from_parts(&[32, 32, 32, 32, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("Mlp::predict 32→32→32→32→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_32)));
    });

    // 64 → 64 → 64 → 1 (3 layers)
    let (w, b) = build_mlp_weights(&[64, 64, 64, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = Mlp::from_parts(&[64, 64, 64, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("Mlp::predict 64→64→64→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_64)));
    });
}

fn bench_mlp_f32_activations(c: &mut Criterion) {
    let features_64: Vec<f32> = vec![0.5; 64];
    let features_32: Vec<f32> = vec![0.5; 32];

    let (w, b) = build_mlp_weights(&[64, 64, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();

    let mut model = Mlp::from_parts(&[64, 64, 1], &w, &b, Activation::Tanh).unwrap();
    c.bench_function("Mlp::predict 64→64→1 tanh", |b| {
        b.iter(|| model.predict(black_box(&features_64)));
    });

    let (w, b) = build_mlp_weights(&[64, 64, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = Mlp::from_parts(&[64, 64, 1], &w, &b, Activation::Gelu).unwrap();
    c.bench_function("Mlp::predict 64→64→1 gelu", |b| {
        b.iter(|| model.predict(black_box(&features_64)));
    });

    let (w, b) = build_mlp_weights(&[64, 64, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = Mlp::from_parts(&[64, 64, 1], &w, &b, Activation::Swish).unwrap();
    c.bench_function("Mlp::predict 64→64→1 swish", |b| {
        b.iter(|| model.predict(black_box(&features_64)));
    });

    let (w, b) = build_mlp_weights(&[32, 32, 32, 32, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = Mlp::from_parts(&[32, 32, 32, 32, 1], &w, &b, Activation::Tanh).unwrap();
    c.bench_function("Mlp::predict 32→32→32→32→1 tanh", |b| {
        b.iter(|| model.predict(black_box(&features_32)));
    });

    let (w, b) = build_mlp_weights(&[32, 32, 32, 32, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = Mlp::from_parts(&[32, 32, 32, 32, 1], &w, &b, Activation::Gelu).unwrap();
    c.bench_function("Mlp::predict 32→32→32→32→1 gelu", |b| {
        b.iter(|| model.predict(black_box(&features_32)));
    });
}

fn bench_lut(c: &mut Criterion) {
    // 2 features × 10 bins
    let table_2x10: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let model = Lut::from_parts(2, 10, &[0.0, 0.0], &[1.0, 1.0], &table_2x10).unwrap();
    c.bench_function("Lut::predict 2feat×10bins", |b| {
        b.iter(|| model.predict(black_box(&[0.35_f32, 0.72])));
    });

    // 3 features × 20 bins
    let table_3x20: Vec<f32> = (0..8000).map(|i| i as f32 * 0.001).collect();
    let model = Lut::from_parts(3, 20, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], &table_3x20).unwrap();
    c.bench_function("Lut::predict 3feat×20bins", |b| {
        b.iter(|| model.predict(black_box(&[0.35_f32, 0.72, 0.15])));
    });
}

fn bench_mlp_f32_layernorm(c: &mut Criterion) {
    let features_16: Vec<f32> = vec![0.5; 16];
    let features_32: Vec<f32> = vec![0.5; 32];
    let features_64: Vec<f32> = vec![0.5; 64];

    // 16 → 32 → 8 → 1: hidden layers are 32 and 8, so ln params = 40
    let (w, b) = build_mlp_weights(&[16, 32, 8, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let ln_gamma: Vec<f32> = vec![1.0; 40];
    let ln_beta: Vec<f32> = vec![0.0; 40];
    let mut model = Mlp::from_parts_with_layer_norm(
        &[16, 32, 8, 1],
        &w,
        &b,
        &ln_gamma,
        &ln_beta,
        Activation::Relu,
    )
    .unwrap();
    c.bench_function("Mlp::predict 16→32→8→1 relu+LN", |b| {
        b.iter(|| model.predict(black_box(&features_16)));
    });

    // 32 → 32 → 32 → 32 → 1: hidden layers are 32,32,32, so ln params = 96
    let (w, b) = build_mlp_weights(&[32, 32, 32, 32, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let ln_gamma: Vec<f32> = vec![1.0; 96];
    let ln_beta: Vec<f32> = vec![0.0; 96];
    let mut model = Mlp::from_parts_with_layer_norm(
        &[32, 32, 32, 32, 1],
        &w,
        &b,
        &ln_gamma,
        &ln_beta,
        Activation::Relu,
    )
    .unwrap();
    c.bench_function("Mlp::predict 32→32→32→32→1 relu+LN", |b| {
        b.iter(|| model.predict(black_box(&features_32)));
    });

    // 64 → 64 → 64 → 1: hidden layers are 64,64, so ln params = 128
    let (w, b) = build_mlp_weights(&[64, 64, 64, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let ln_gamma: Vec<f32> = vec![1.0; 128];
    let ln_beta: Vec<f32> = vec![0.0; 128];
    let mut model = Mlp::from_parts_with_layer_norm(
        &[64, 64, 64, 1],
        &w,
        &b,
        &ln_gamma,
        &ln_beta,
        Activation::Relu,
    )
    .unwrap();
    c.bench_function("Mlp::predict 64→64→64→1 relu+LN", |b| {
        b.iter(|| model.predict(black_box(&features_64)));
    });
}

fn make_bnn(input: usize, hidden: usize, output: usize, n_binary: usize) -> Bnn {
    let wpr = hidden / 64;
    let w_input = vec![0.1_f32; hidden * input];
    let b_input = vec![0.0_f32; hidden];

    let bin_w = vec![0xAAAA_AAAA_AAAA_AAAA_u64; hidden * wpr];
    let bin_b = vec![0.0_f32; hidden];

    let bin_weights: Vec<&[u64]> = (0..n_binary).map(|_| bin_w.as_slice()).collect();
    let bin_biases: Vec<&[f32]> = (0..n_binary).map(|_| bin_b.as_slice()).collect();

    let w_output = vec![0.1_f32; output * hidden];
    let b_output = vec![0.0_f32; output];

    Bnn::from_parts(
        &w_input,
        &b_input,
        &bin_weights,
        &bin_biases,
        &w_output,
        &b_output,
        output,
    )
    .unwrap()
}

fn bench_bnn(c: &mut Criterion) {
    let input_8 = vec![0.5_f32; 8];

    let mut m = make_bnn(8, 64, 1, 0);
    c.bench_function("BNN 8→64→1 (0 binary)", |b| {
        b.iter(|| m.predict(black_box(&input_8)));
    });

    let mut m = make_bnn(8, 64, 1, 1);
    c.bench_function("BNN 8→64→1 (1 binary)", |b| {
        b.iter(|| m.predict(black_box(&input_8)));
    });

    let mut m = make_bnn(8, 64, 1, 2);
    c.bench_function("BNN 8→64→1 (2 binary)", |b| {
        b.iter(|| m.predict(black_box(&input_8)));
    });

    let mut m = make_bnn(8, 128, 1, 2);
    c.bench_function("BNN 8→128→1 (2 binary)", |b| {
        b.iter(|| m.predict(black_box(&input_8)));
    });
}

fn make_tcn(
    input: usize,
    filters: usize,
    kernel: usize,
    num_layers: usize,
    output: usize,
    residual: bool,
) -> TinyTcn {
    let mut seed = 42u64;
    let mut next_f32 = || -> f32 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (seed >> 33) as f32 / (1u64 << 31) as f32 * 0.2 - 0.1
    };

    let mut w_convs = Vec::new();
    let mut b_convs = Vec::new();
    for k in 0..num_layers {
        let in_ch = if k == 0 { input } else { filters };
        let n = filters * kernel * in_ch;
        w_convs.push((0..n).map(|_| next_f32()).collect::<Vec<_>>());
        b_convs.push(vec![0.01_f32; filters]);
    }

    let w_refs: Vec<&[f32]> = w_convs.iter().map(|v| v.as_slice()).collect();
    let b_refs: Vec<&[f32]> = b_convs.iter().map(|v| v.as_slice()).collect();

    let w_out: Vec<f32> = (0..output * filters).map(|_| next_f32()).collect();
    let b_out = vec![0.0_f32; output];

    TinyTcn::from_parts(
        input,
        filters,
        kernel,
        output,
        residual,
        &w_refs,
        &b_refs,
        &w_out,
        &b_out,
        Activation::Relu,
    )
    .unwrap()
}

fn bench_tcn(c: &mut Criterion) {
    let input_4 = vec![0.5_f32; 4];

    // 4→16, K=3, 2 layers, no residual
    let mut m = make_tcn(4, 16, 3, 2, 1, false);
    // Prime the model
    for _ in 0..10 {
        m.predict(&input_4);
    }
    c.bench_function("TCN I=4 F=16 K=3 L=2", |b| {
        b.iter(|| m.predict(black_box(&input_4)));
    });

    // 4→16, K=3, 4 layers, residual
    let mut m = make_tcn(4, 16, 3, 4, 1, true);
    for _ in 0..40 {
        m.predict(&input_4);
    }
    c.bench_function("TCN I=4 F=16 K=3 L=4 res", |b| {
        b.iter(|| m.predict(black_box(&input_4)));
    });

    // 4→32, K=3, 3 layers
    let mut m = make_tcn(4, 32, 3, 3, 1, false);
    for _ in 0..20 {
        m.predict(&input_4);
    }
    c.bench_function("TCN I=4 F=32 K=3 L=3", |b| {
        b.iter(|| m.predict(black_box(&input_4)));
    });

    let input_8 = vec![0.5_f32; 8];

    // 8→16, K=3, 4 layers, residual
    let mut m = make_tcn(8, 16, 3, 4, 1, true);
    for _ in 0..40 {
        m.predict(&input_8);
    }
    c.bench_function("TCN I=8 F=16 K=3 L=4 res", |b| {
        b.iter(|| m.predict(black_box(&input_8)));
    });
}

fn make_quantized_mlp(sizes: &[usize]) -> QuantizedMlp {
    let mut layers_w = Vec::new();
    let mut layers_b = Vec::new();
    let mut w_scales = Vec::new();
    let mut w_zero_points = Vec::new();
    let mut input_scales = Vec::new();
    let mut input_zero_points = Vec::new();

    for k in 0..sizes.len() - 1 {
        let in_size = sizes[k];
        let out_size = sizes[k + 1];
        let w: Vec<i8> = (0..out_size * in_size)
            .map(|i| ((i % 255) as i8).wrapping_sub(64))
            .collect();
        let b: Vec<f32> = vec![0.01; out_size];
        layers_w.push(w);
        layers_b.push(b);
        w_scales.push(0.02_f32);
        w_zero_points.push(0_i8);
        input_scales.push(0.01_f32);
        input_zero_points.push(0_i8);
    }

    let w_refs: Vec<&[i8]> = layers_w.iter().map(Vec::as_slice).collect();
    let b_refs: Vec<&[f32]> = layers_b.iter().map(Vec::as_slice).collect();
    QuantizedMlp::from_parts(
        &w_refs,
        &b_refs,
        &w_scales,
        &w_zero_points,
        &input_scales,
        &input_zero_points,
        Activation::Relu,
    )
    .unwrap()
}

fn bench_quantized_mlp(c: &mut Criterion) {
    let features_8: Vec<f32> = vec![0.5; 8];
    let features_16: Vec<f32> = vec![0.5; 16];
    let features_64: Vec<f32> = vec![0.5; 64];

    let mut model = make_quantized_mlp(&[8, 16, 1]);
    c.bench_function("QuantizedMlp::predict 8→16→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_8)));
    });

    let mut model = make_quantized_mlp(&[16, 32, 8, 1]);
    c.bench_function("QuantizedMlp::predict 16→32→8→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_16)));
    });

    let mut model = make_quantized_mlp(&[64, 64, 1]);
    c.bench_function("QuantizedMlp::predict 64→64→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_64)));
    });

    let features_32: Vec<f32> = vec![0.5; 32];
    let mut model = make_quantized_mlp(&[32, 32, 32, 32, 1]);
    c.bench_function("QuantizedMlp::predict 32→32→32→32→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_32)));
    });
}

criterion_group!(
    benches,
    bench_gbdt,
    bench_gbdt_random,
    bench_mlp_f32,
    bench_mlp_f32_activations,
    bench_mlp_f32_layernorm,
    bench_lut,
    bench_bnn,
    bench_tcn,
    bench_quantized_mlp
);
criterion_main!(benches);

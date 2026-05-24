use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nexus_inference::{Activation, GbdtF64, LutF64, MlpF32, MlpF64};

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
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            weights.push((seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0);
        }
        for _ in 0..layer_sizes[i + 1] {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            biases.push((seed >> 33) as f64 / (1u64 << 31) as f64 * 0.1);
        }
    }
    (weights, biases)
}

fn bench_gbdt(c: &mut Criterion) {
    let features_8 = vec![0.5_f64; 8];
    let features_16 = vec![0.5_f64; 16];

    let text_50x6 = build_lightgbm_model(50, 6, 8);
    let model_50x6 = GbdtF64::from_lightgbm(text_50x6.as_bytes()).unwrap();
    c.bench_function("GbdtF64::predict 50x6 8feat", |b| {
        b.iter(|| model_50x6.predict(black_box(&features_8)));
    });

    let text_100x6 = build_lightgbm_model(100, 6, 8);
    let model_100x6 = GbdtF64::from_lightgbm(text_100x6.as_bytes()).unwrap();
    c.bench_function("GbdtF64::predict 100x6 8feat", |b| {
        b.iter(|| model_100x6.predict(black_box(&features_8)));
    });

    let text_200x8 = build_lightgbm_model(200, 8, 16);
    let model_200x8 = GbdtF64::from_lightgbm(text_200x8.as_bytes()).unwrap();
    c.bench_function("GbdtF64::predict 200x8 16feat", |b| {
        b.iter(|| model_200x8.predict(black_box(&features_16)));
    });

    let text_100x6b = build_lightgbm_model(100, 6, 8);
    let model_100x6b = GbdtF64::from_lightgbm(text_100x6b.as_bytes()).unwrap();
    c.bench_function("GbdtF64::predict (NaN-aware) 100x6 8feat", |b| {
        b.iter(|| model_100x6b.predict(black_box(&features_8)));
    });
}

fn bench_mlp(c: &mut Criterion) {
    let features_8 = vec![0.5_f64; 8];
    let features_16 = vec![0.5_f64; 16];
    let features_64 = vec![0.5_f64; 64];

    // 8 → 16 → 1
    let (w, b) = build_mlp_weights(&[8, 16, 1]);
    let mut model = MlpF64::from_parts(&[8, 16, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("MlpF64::predict 8→16→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_8)));
    });

    // 16 → 32 → 8 → 1
    let (w, b) = build_mlp_weights(&[16, 32, 8, 1]);
    let mut model = MlpF64::from_parts(&[16, 32, 8, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("MlpF64::predict 16→32→8→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_16)));
    });

    // 64 → 64 → 1
    let (w, b) = build_mlp_weights(&[64, 64, 1]);
    let mut model = MlpF64::from_parts(&[64, 64, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("MlpF64::predict 64→64→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_64)));
    });
}

fn bench_mlp_f32(c: &mut Criterion) {
    let features_8: Vec<f32> = vec![0.5; 8];
    let features_16: Vec<f32> = vec![0.5; 16];
    let features_64: Vec<f32> = vec![0.5; 64];

    let (w, b) = build_mlp_weights(&[8, 16, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = MlpF32::from_parts(&[8, 16, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("MlpF32::predict 8→16→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_8)));
    });

    let (w, b) = build_mlp_weights(&[16, 32, 8, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = MlpF32::from_parts(&[16, 32, 8, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("MlpF32::predict 16→32→8→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_16)));
    });

    let (w, b) = build_mlp_weights(&[64, 64, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = MlpF32::from_parts(&[64, 64, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("MlpF32::predict 64→64→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_64)));
    });

    // Deep stacked configs
    let features_32: Vec<f32> = vec![0.5; 32];

    // 32 → 32 → 32 → 32 → 1 (4 layers)
    let (w, b) = build_mlp_weights(&[32, 32, 32, 32, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = MlpF32::from_parts(&[32, 32, 32, 32, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("MlpF32::predict 32→32→32→32→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_32)));
    });

    // 64 → 64 → 64 → 1 (3 layers)
    let (w, b) = build_mlp_weights(&[64, 64, 64, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let mut model = MlpF32::from_parts(&[64, 64, 64, 1], &w, &b, Activation::Relu).unwrap();
    c.bench_function("MlpF32::predict 64→64→64→1 relu", |b| {
        b.iter(|| model.predict(black_box(&features_64)));
    });
}

fn bench_lut(c: &mut Criterion) {
    // 2 features × 10 bins
    let table_2x10: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    let model = LutF64::from_parts(2, 10, &[0.0, 0.0], &[1.0, 1.0], &table_2x10).unwrap();
    c.bench_function("LutF64::predict 2feat×10bins", |b| {
        b.iter(|| model.predict(black_box(&[0.35, 0.72])));
    });

    // 3 features × 20 bins
    let table_3x20: Vec<f64> = (0..8000).map(|i| i as f64 * 0.001).collect();
    let model = LutF64::from_parts(3, 20, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], &table_3x20).unwrap();
    c.bench_function("LutF64::predict 3feat×20bins", |b| {
        b.iter(|| model.predict(black_box(&[0.35, 0.72, 0.15])));
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
    let mut model = MlpF32::from_parts_with_layer_norm(
        &[16, 32, 8, 1], &w, &b, &ln_gamma, &ln_beta, Activation::Relu,
    ).unwrap();
    c.bench_function("MlpF32::predict 16→32→8→1 relu+LN", |b| {
        b.iter(|| model.predict(black_box(&features_16)));
    });

    // 32 → 32 → 32 → 32 → 1: hidden layers are 32,32,32, so ln params = 96
    let (w, b) = build_mlp_weights(&[32, 32, 32, 32, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let ln_gamma: Vec<f32> = vec![1.0; 96];
    let ln_beta: Vec<f32> = vec![0.0; 96];
    let mut model = MlpF32::from_parts_with_layer_norm(
        &[32, 32, 32, 32, 1], &w, &b, &ln_gamma, &ln_beta, Activation::Relu,
    ).unwrap();
    c.bench_function("MlpF32::predict 32→32→32→32→1 relu+LN", |b| {
        b.iter(|| model.predict(black_box(&features_32)));
    });

    // 64 → 64 → 64 → 1: hidden layers are 64,64, so ln params = 128
    let (w, b) = build_mlp_weights(&[64, 64, 64, 1]);
    let w: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    let b: Vec<f32> = b.into_iter().map(|x| x as f32).collect();
    let ln_gamma: Vec<f32> = vec![1.0; 128];
    let ln_beta: Vec<f32> = vec![0.0; 128];
    let mut model = MlpF32::from_parts_with_layer_norm(
        &[64, 64, 64, 1], &w, &b, &ln_gamma, &ln_beta, Activation::Relu,
    ).unwrap();
    c.bench_function("MlpF32::predict 64→64→64→1 relu+LN", |b| {
        b.iter(|| model.predict(black_box(&features_64)));
    });
}

criterion_group!(benches, bench_gbdt, bench_mlp, bench_mlp_f32, bench_mlp_f32_layernorm, bench_lut);
criterion_main!(benches);

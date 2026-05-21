use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nexus_inference::GbdtF64;

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

fn bench_predict(c: &mut Criterion) {
    let features_8 = vec![0.5_f64; 8];
    let features_16 = vec![0.5_f64; 16];

    let text_50x6 = build_lightgbm_model(50, 6, 8);
    let model_50x6 = GbdtF64::from_lightgbm(text_50x6.as_bytes()).unwrap();
    c.bench_function("GbdtF64::predict_unchecked 50x6 8feat", |b| {
        b.iter(|| model_50x6.predict_unchecked(black_box(&features_8)));
    });

    let text_100x6 = build_lightgbm_model(100, 6, 8);
    let model_100x6 = GbdtF64::from_lightgbm(text_100x6.as_bytes()).unwrap();
    c.bench_function("GbdtF64::predict_unchecked 100x6 8feat", |b| {
        b.iter(|| model_100x6.predict_unchecked(black_box(&features_8)));
    });

    let text_200x8 = build_lightgbm_model(200, 8, 16);
    let model_200x8 = GbdtF64::from_lightgbm(text_200x8.as_bytes()).unwrap();
    c.bench_function("GbdtF64::predict_unchecked 200x8 16feat", |b| {
        b.iter(|| model_200x8.predict_unchecked(black_box(&features_16)));
    });

    let text_100x6b = build_lightgbm_model(100, 6, 8);
    let model_100x6b = GbdtF64::from_lightgbm(text_100x6b.as_bytes()).unwrap();
    c.bench_function("GbdtF64::predict (NaN-aware) 100x6 8feat", |b| {
        b.iter(|| model_100x6b.predict(black_box(&features_8)));
    });
}

criterion_group!(benches, bench_predict);
criterion_main!(benches);

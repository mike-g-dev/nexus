//! Measures the full latency distribution of GBDT predict.
//! Run with: taskset -c 0 cargo run --example gbdt_tail --release --features loader-lightgbm

use nexus_inference::Gbdt;
use std::time::Instant;

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

        s.push_str("split_feature=");
        for i in 0..num_internal {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&format!("{}", (t + i) % n_features));
        }
        s.push('\n');

        s.push_str("threshold=");
        for i in 0..num_internal {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&format!("{:.1}", (i as f64 + 1.0) * 0.5));
        }
        s.push('\n');

        s.push_str("decision_type=");
        for i in 0..num_internal {
            if i > 0 {
                s.push(' ');
            }
            s.push('0');
        }
        s.push('\n');

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

        s.push_str("leaf_value=");
        for i in 0..num_leaves {
            if i > 0 {
                s.push(' ');
            }
            let val = (i as f64 - num_leaves as f64 / 2.0) * 0.01;
            s.push_str(&format!("{val:.4}"));
        }
        s.push_str("\n\n");
    }

    s.push_str("end of trees\n");
    s
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn random_features(n: usize, count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            (0..n)
                .map(|_| (xorshift64(&mut state) as f64 / u64::MAX as f64) as f32)
                .collect()
        })
        .collect()
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    let idx = ((sorted.len() as f64) * p / 100.0) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn run_distribution(name: &str, model: &Gbdt, features: &[Vec<f32>], n_samples: usize) {
    let n_feat = features.len();
    let mut latencies = Vec::with_capacity(n_samples);

    // Warmup
    for i in 0..1000 {
        std::hint::black_box(model.predict(&features[i % n_feat]));
    }

    // Collect
    for i in 0..n_samples {
        let f = &features[i % n_feat];
        let start = Instant::now();
        std::hint::black_box(model.predict(std::hint::black_box(f)));
        let elapsed = start.elapsed().as_nanos() as u64;
        latencies.push(elapsed);
    }

    latencies.sort_unstable();

    let mean: f64 = latencies.iter().sum::<u64>() as f64 / n_samples as f64;
    let variance: f64 = latencies
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n_samples as f64;
    let stddev = variance.sqrt();

    println!("{name}");
    println!("  samples: {n_samples}");
    println!("  mean:    {mean:.1} ns");
    println!(
        "  stddev:  {stddev:.1} ns  (CV: {:.1}%)",
        stddev / mean * 100.0
    );
    println!("  p50:     {} ns", percentile(&latencies, 50.0));
    println!("  p90:     {} ns", percentile(&latencies, 90.0));
    println!("  p95:     {} ns", percentile(&latencies, 95.0));
    println!("  p99:     {} ns", percentile(&latencies, 99.0));
    println!("  p99.9:   {} ns", percentile(&latencies, 99.9));
    println!("  p99.99:  {} ns", percentile(&latencies, 99.99));
    println!("  min:     {} ns", latencies[0]);
    println!("  max:     {} ns", latencies[latencies.len() - 1]);
    println!();
}

fn run_distribution_nan(name: &str, model: &Gbdt, features: &[Vec<f32>], n_samples: usize) {
    let n_feat = features.len();
    let mut latencies = Vec::with_capacity(n_samples);

    for i in 0..1000 {
        std::hint::black_box(model.predict_nan_aware(&features[i % n_feat]));
    }

    for i in 0..n_samples {
        let f = &features[i % n_feat];
        let start = Instant::now();
        std::hint::black_box(model.predict_nan_aware(std::hint::black_box(f)));
        let elapsed = start.elapsed().as_nanos() as u64;
        latencies.push(elapsed);
    }

    latencies.sort_unstable();

    let mean: f64 = latencies.iter().sum::<u64>() as f64 / n_samples as f64;
    let variance: f64 = latencies
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n_samples as f64;
    let stddev = variance.sqrt();

    println!("{name}");
    println!("  samples: {n_samples}");
    println!("  mean:    {mean:.1} ns");
    println!(
        "  stddev:  {stddev:.1} ns  (CV: {:.1}%)",
        stddev / mean * 100.0
    );
    println!("  p50:     {} ns", percentile(&latencies, 50.0));
    println!("  p90:     {} ns", percentile(&latencies, 90.0));
    println!("  p95:     {} ns", percentile(&latencies, 95.0));
    println!("  p99:     {} ns", percentile(&latencies, 99.0));
    println!("  p99.9:   {} ns", percentile(&latencies, 99.9));
    println!("  p99.99:  {} ns", percentile(&latencies, 99.99));
    println!("  min:     {} ns", latencies[0]);
    println!("  max:     {} ns", latencies[latencies.len() - 1]);
    println!();
}

fn main() {
    let n_samples = 100_000;

    let features_const = vec![vec![0.5_f32; 8]; 1];
    let features_random = random_features(8, 4096, 0xDEAD_BEEF_CAFE_F00D);

    let text = build_lightgbm_model(100, 6, 8);
    let model = Gbdt::from_lightgbm(text.as_bytes()).unwrap();

    println!("=== GBDT 100x6, 8 features — Latency Distribution ===\n");

    run_distribution(
        "predict (constant features)",
        &model,
        &features_const,
        n_samples,
    );
    run_distribution(
        "predict (random features)",
        &model,
        &features_random,
        n_samples,
    );
    run_distribution_nan(
        "predict_nan_aware (constant features)",
        &model,
        &features_const,
        n_samples,
    );
    run_distribution_nan(
        "predict_nan_aware (random features)",
        &model,
        &features_random,
        n_samples,
    );
}

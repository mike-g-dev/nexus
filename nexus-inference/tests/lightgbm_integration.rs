#![cfg(feature = "loader-lightgbm")]

use nexus_inference::*;

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn load_model_txt(name: &str) -> Vec<u8> {
    let path = fixture_path(&format!("lgb_{name}.txt"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn load_expected(name: &str) -> serde_json::Value {
    let path = fixture_path(&format!("lgb_{name}_expected.json"));
    let data = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&data).unwrap()
}

fn parse_inputs(v: &serde_json::Value) -> Vec<Vec<f64>> {
    v["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|inp| {
            inp.as_array()
                .unwrap()
                .iter()
                .map(|x| {
                    if x.is_null() {
                        f64::NAN
                    } else {
                        x.as_f64().unwrap()
                    }
                })
                .collect()
        })
        .collect()
}

fn parse_outputs(v: &serde_json::Value) -> Vec<f64> {
    v["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn run_test(name: &str) {
    let model_bytes = load_model_txt(name);
    let exp = load_expected(name);
    let tol = exp["tolerance"].as_f64().unwrap().max(1e-4);
    let expected_n_features = exp["n_features"].as_u64().unwrap() as usize;
    let expected_n_trees = exp["n_trees"].as_u64().unwrap() as usize;
    let has_nan = exp
        .get("has_nan")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let model = Gbdt::from_lightgbm(&model_bytes).unwrap();
    assert_eq!(
        model.n_features(),
        expected_n_features,
        "{name}: n_features mismatch"
    );
    assert_eq!(
        model.n_trees(),
        expected_n_trees,
        "{name}: n_trees mismatch"
    );

    let inputs = parse_inputs(&exp);
    let outputs = parse_outputs(&exp);
    assert_eq!(
        inputs.len(),
        outputs.len(),
        "{name}: input/output count mismatch"
    );

    for (i, (inp, &expected)) in inputs.iter().zip(outputs.iter()).enumerate() {
        let inp_f32: Vec<f32> = inp.iter().map(|&x| x as f32).collect();
        let actual = if has_nan {
            model.predict_nan_aware(&inp_f32)
        } else {
            model.predict(&inp_f32)
        };
        let err = (actual as f64 - expected).abs();
        assert!(
            err < tol,
            "{name} f32 input {i}: got {actual}, expected {expected}, err={err}",
        );
    }
}

// ---- regression ----

#[test]
fn regression_small() {
    run_test("regression_small");
}

#[test]
fn regression_deep() {
    run_test("regression_deep");
}

#[test]
fn regression_large() {
    run_test("regression_large");
}

// ---- binary classification (raw logit) ----

#[test]
fn binary_small() {
    run_test("binary_small");
}

#[test]
fn binary_deep() {
    run_test("binary_deep");
}

// ---- NaN routing ----

#[test]
fn nan_regression() {
    run_test("nan_regression");
}

// ---- edge cases ----

#[test]
fn stump() {
    run_test("stump");
}

#[test]
fn many_features() {
    run_test("many_features");
}

// ---- alternative objectives ----

#[test]
fn huber() {
    run_test("huber");
}

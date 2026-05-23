#![cfg(feature = "loader-safetensors")]

use nexus_inference::*;

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn load_model(name: &str) -> Vec<u8> {
    let path = fixture_path(&format!("{name}.safetensors"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn load_expected(name: &str) -> serde_json::Value {
    let path = fixture_path(&format!("{name}_expected.json"));
    let data = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&data).unwrap()
}

fn inputs_f32(v: &serde_json::Value) -> Vec<Vec<f32>> {
    v["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|inp| {
            inp.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        })
        .collect()
}

fn inputs_f64(v: &serde_json::Value) -> Vec<Vec<f64>> {
    v["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|inp| {
            inp.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect()
}

fn expected_outputs(v: &serde_json::Value) -> Vec<Vec<f64>> {
    v["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|out| match out {
            serde_json::Value::Array(arr) => arr.iter().map(|x| x.as_f64().unwrap()).collect(),
            serde_json::Value::Number(n) => vec![n.as_f64().unwrap()],
            _ => panic!("unexpected output format"),
        })
        .collect()
}

fn assert_close(model: &str, step: usize, idx: usize, actual: f64, expected: f64, tol: f64) {
    let err = (actual - expected).abs();
    assert!(
        err < tol,
        "{model} step {step} output {idx}: got {actual}, expected {expected}, err={err}"
    );
}

#[test]
fn lstm_matches_pytorch() {
    let data = load_model("lstm");
    let exp = load_expected("lstm");
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut lstm = TinyLstmF32::from_safetensors(
        &data,
        exp["rnn_prefix"].as_str().unwrap(),
        exp["proj_prefix"].as_str().unwrap(),
    )
    .unwrap();

    for (i, (inp, exp_out)) in inputs_f32(&exp)
        .iter()
        .zip(expected_outputs(&exp).iter())
        .enumerate()
    {
        let mut out = vec![0.0_f32; exp_out.len()];
        lstm.step_into(inp, &mut out);
        for (j, (&actual, &expected)) in out.iter().zip(exp_out.iter()).enumerate() {
            assert_close("lstm", i, j, actual as f64, expected, tol);
        }
    }
}

#[test]
fn gru_matches_pytorch() {
    let data = load_model("gru");
    let exp = load_expected("gru");
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut gru = TinyGruF32::from_safetensors(
        &data,
        exp["rnn_prefix"].as_str().unwrap(),
        exp["proj_prefix"].as_str().unwrap(),
    )
    .unwrap();

    for (i, (inp, exp_out)) in inputs_f32(&exp)
        .iter()
        .zip(expected_outputs(&exp).iter())
        .enumerate()
    {
        let mut out = vec![0.0_f32; exp_out.len()];
        gru.step_into(inp, &mut out);
        for (j, (&actual, &expected)) in out.iter().zip(exp_out.iter()).enumerate() {
            assert_close("gru", i, j, actual as f64, expected, tol);
        }
    }
}

#[test]
fn mlp_f32_matches_pytorch() {
    let data = load_model("mlp_f32");
    let exp = load_expected("mlp_f32");
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut mlp =
        MlpF32::from_safetensors(&data, exp["prefix"].as_str().unwrap(), Activation::Relu).unwrap();

    for (i, (inp, exp_out)) in inputs_f32(&exp)
        .iter()
        .zip(expected_outputs(&exp).iter())
        .enumerate()
    {
        let mut out = vec![0.0_f32; exp_out.len()];
        mlp.predict_into(inp, &mut out);
        for (j, (&actual, &expected)) in out.iter().zip(exp_out.iter()).enumerate() {
            assert_close("mlp_f32", i, j, actual as f64, expected, tol);
        }
    }
}

#[test]
fn mlp_f64_matches_pytorch() {
    let data = load_model("mlp_f64");
    let exp = load_expected("mlp_f64");
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut mlp =
        MlpF64::from_safetensors(&data, exp["prefix"].as_str().unwrap(), Activation::Relu).unwrap();

    for (i, (inp, exp_out)) in inputs_f64(&exp)
        .iter()
        .zip(expected_outputs(&exp).iter())
        .enumerate()
    {
        let mut out = vec![0.0_f64; exp_out.len()];
        mlp.predict_into(inp, &mut out);
        for (j, (&actual, &expected)) in out.iter().zip(exp_out.iter()).enumerate() {
            assert_close("mlp_f64", i, j, actual, expected, tol);
        }
    }
}

#[test]
fn conv1d_matches_pytorch() {
    let data = load_model("conv1d");
    let exp = load_expected("conv1d");
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut conv = Causal1dConvF32::from_safetensors(
        &data,
        exp["conv_prefix"].as_str().unwrap(),
        exp["proj_prefix"].as_str().unwrap(),
        Activation::Relu,
    )
    .unwrap();

    for (i, (inp, exp_out)) in inputs_f32(&exp)
        .iter()
        .zip(expected_outputs(&exp).iter())
        .enumerate()
    {
        let mut out = vec![0.0_f32; exp_out.len()];
        conv.step_into(inp, &mut out);
        for (j, (&actual, &expected)) in out.iter().zip(exp_out.iter()).enumerate() {
            assert_close("conv1d", i, j, actual as f64, expected, tol);
        }
    }
}

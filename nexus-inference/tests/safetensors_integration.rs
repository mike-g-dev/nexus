#![cfg(feature = "safetensors")]

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

fn parse_activation(v: &serde_json::Value) -> Activation {
    let param = v
        .get("activation_param")
        .and_then(serde_json::Value::as_f64)
        .map(|p| p as f32);
    match v["activation"].as_str().unwrap() {
        "relu" => Activation::Relu,
        "tanh" => Activation::Tanh,
        "sigmoid" => Activation::Sigmoid,
        "gelu" => Activation::Gelu,
        "identity" => Activation::Identity,
        "swish" => Activation::Swish,
        "elu" => Activation::Elu(param.unwrap_or(1.0)),
        "leaky_relu" => Activation::LeakyRelu(param.unwrap_or(0.01)),
        other => panic!("unknown activation: {other}"),
    }
}

fn assert_close(model: &str, step: usize, idx: usize, actual: f64, expected: f64, tol: f64) {
    let err = (actual - expected).abs();
    assert!(
        err < tol,
        "{model} step {step} output {idx}: got {actual}, expected {expected}, err={err}"
    );
}

// ---- test runners ----

fn run_stacked_lstm_test(name: &str) {
    let data = load_model(name);
    let exp = load_expected(name);
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut lstm = StackedLstm::from_safetensors(
        &data,
        exp["rnn_prefix"].as_str().unwrap(),
        exp["proj_prefix"].as_str().unwrap(),
    )
    .unwrap();

    let expected_layers = exp["num_layers"].as_u64().unwrap() as usize;
    assert_eq!(lstm.n_layers(), expected_layers);

    for (i, (inp, exp_out)) in inputs_f32(&exp)
        .iter()
        .zip(expected_outputs(&exp).iter())
        .enumerate()
    {
        let mut out = vec![0.0_f32; exp_out.len()];
        lstm.step_into(inp, &mut out);
        for (j, (&actual, &expected)) in out.iter().zip(exp_out.iter()).enumerate() {
            assert_close(name, i, j, actual as f64, expected, tol);
        }
    }
}

fn run_stacked_gru_test(name: &str) {
    let data = load_model(name);
    let exp = load_expected(name);
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut gru = StackedGru::from_safetensors(
        &data,
        exp["rnn_prefix"].as_str().unwrap(),
        exp["proj_prefix"].as_str().unwrap(),
    )
    .unwrap();

    let expected_layers = exp["num_layers"].as_u64().unwrap() as usize;
    assert_eq!(gru.n_layers(), expected_layers);

    for (i, (inp, exp_out)) in inputs_f32(&exp)
        .iter()
        .zip(expected_outputs(&exp).iter())
        .enumerate()
    {
        let mut out = vec![0.0_f32; exp_out.len()];
        gru.step_into(inp, &mut out);
        for (j, (&actual, &expected)) in out.iter().zip(exp_out.iter()).enumerate() {
            assert_close(name, i, j, actual as f64, expected, tol);
        }
    }
}

fn run_lstm_test(name: &str) {
    let data = load_model(name);
    let exp = load_expected(name);
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut lstm = TinyLstm::from_safetensors(
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
            assert_close(name, i, j, actual as f64, expected, tol);
        }
    }
}

fn run_gru_test(name: &str) {
    let data = load_model(name);
    let exp = load_expected(name);
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut gru = TinyGru::from_safetensors(
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
            assert_close(name, i, j, actual as f64, expected, tol);
        }
    }
}

fn run_mlp_f32_test(name: &str) {
    let data = load_model(name);
    let exp = load_expected(name);
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut mlp = Mlp::from_safetensors(
        &data,
        exp["prefix"].as_str().unwrap(),
        parse_activation(&exp),
    )
    .unwrap();

    for (i, (inp, exp_out)) in inputs_f32(&exp)
        .iter()
        .zip(expected_outputs(&exp).iter())
        .enumerate()
    {
        let mut out = vec![0.0_f32; exp_out.len()];
        mlp.predict_into(inp, &mut out);
        for (j, (&actual, &expected)) in out.iter().zip(exp_out.iter()).enumerate() {
            assert_close(name, i, j, actual as f64, expected, tol);
        }
    }
}

fn run_conv1d_test(name: &str) {
    let data = load_model(name);
    let exp = load_expected(name);
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut conv = Causal1dConv::from_safetensors(
        &data,
        exp["conv_prefix"].as_str().unwrap(),
        exp["proj_prefix"].as_str().unwrap(),
        parse_activation(&exp),
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
            assert_close(name, i, j, actual as f64, expected, tol);
        }
    }
}

// ---- LSTM tests ----

#[test]
fn lstm() {
    run_lstm_test("lstm");
}

#[test]
fn lstm_large() {
    run_lstm_test("lstm_large");
}

#[test]
fn lstm_single_output() {
    run_lstm_test("lstm_single_output");
}

// ---- GRU tests ----

#[test]
fn gru() {
    run_gru_test("gru");
}

#[test]
fn gru_large() {
    run_gru_test("gru_large");
}

#[test]
fn gru_multi_output() {
    run_gru_test("gru_multi_output");
}

// ---- MLP f32 tests ----

#[test]
fn mlp_f32() {
    run_mlp_f32_test("mlp_f32");
}

#[test]
fn mlp_f32_tanh() {
    run_mlp_f32_test("mlp_f32_tanh");
}

#[test]
fn mlp_f32_sigmoid() {
    run_mlp_f32_test("mlp_f32_sigmoid");
}

#[test]
fn mlp_f32_gelu() {
    run_mlp_f32_test("mlp_f32_gelu");
}

#[test]
fn mlp_f32_single_layer() {
    run_mlp_f32_test("mlp_f32_single_layer");
}

#[test]
fn mlp_f32_deep() {
    run_mlp_f32_test("mlp_f32_deep");
}

#[test]
fn mlp_f32_swish() {
    run_mlp_f32_test("mlp_f32_swish");
}

#[test]
fn mlp_f32_elu() {
    run_mlp_f32_test("mlp_f32_elu");
}

#[test]
fn mlp_f32_leaky_relu() {
    run_mlp_f32_test("mlp_f32_leaky_relu");
}

#[test]
fn mlp_f32_no_bias() {
    run_mlp_f32_test("mlp_f32_no_bias");
}

#[test]
fn mlp_f32_batchnorm() {
    run_mlp_f32_test("mlp_f32_batchnorm");
}

#[test]
fn mlp_f32_batchnorm_no_bias() {
    run_mlp_f32_test("mlp_f32_batchnorm_no_bias");
}

#[test]
fn mlp_f32_layernorm() {
    run_mlp_f32_test("mlp_f32_layernorm");
}

#[test]
fn mlp_f32_layernorm_no_bias() {
    run_mlp_f32_test("mlp_f32_layernorm_no_bias");
}

// ---- Conv1d tests ----

#[test]
fn conv1d() {
    run_conv1d_test("conv1d");
}

#[test]
fn conv1d_tanh() {
    run_conv1d_test("conv1d_tanh");
}

#[test]
fn conv1d_identity() {
    run_conv1d_test("conv1d_identity");
}

#[test]
fn conv1d_large() {
    run_conv1d_test("conv1d_large");
}

#[test]
fn conv1d_sigmoid() {
    run_conv1d_test("conv1d_sigmoid");
}

#[test]
fn conv1d_swish() {
    run_conv1d_test("conv1d_swish");
}

#[test]
fn conv1d_elu() {
    run_conv1d_test("conv1d_elu");
}

#[test]
fn conv1d_leaky_relu() {
    run_conv1d_test("conv1d_leaky_relu");
}

// ---- SSM tests ----

fn run_ssm_test(name: &str) {
    let data = load_model(name);
    let exp = load_expected(name);
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut ssm = LinearSsm::from_safetensors(&data, exp["prefix"].as_str().unwrap()).unwrap();

    for (i, (inp, exp_out)) in inputs_f32(&exp)
        .iter()
        .zip(expected_outputs(&exp).iter())
        .enumerate()
    {
        let mut out = vec![0.0_f32; exp_out.len()];
        ssm.step_into(inp, &mut out);
        for (j, (&actual, &expected)) in out.iter().zip(exp_out.iter()).enumerate() {
            assert_close(name, i, j, actual as f64, expected, tol);
        }
    }
}

#[test]
fn ssm() {
    run_ssm_test("ssm");
}

#[test]
fn ssm_no_skip() {
    run_ssm_test("ssm_no_skip");
}

#[test]
fn ssm_multi_output() {
    run_ssm_test("ssm_multi_output");
}

#[test]
fn ssm_large() {
    run_ssm_test("ssm_large");
}

// ---- Stacked LSTM tests ----

#[test]
fn stacked_lstm_2layer() {
    run_stacked_lstm_test("stacked_lstm_2layer");
}

#[test]
fn stacked_lstm_3layer() {
    run_stacked_lstm_test("stacked_lstm_3layer");
}

#[test]
fn stacked_lstm_large() {
    run_stacked_lstm_test("stacked_lstm_large");
}

// ---- Stacked GRU tests ----

#[test]
fn stacked_gru_2layer() {
    run_stacked_gru_test("stacked_gru_2layer");
}

#[test]
fn stacked_gru_3layer() {
    run_stacked_gru_test("stacked_gru_3layer");
}

#[test]
fn stacked_gru_large() {
    run_stacked_gru_test("stacked_gru_large");
}

// ---- Fuzz tests (seeded random configs) ----

macro_rules! fuzz_tests {
    ($runner:ident, $($name:ident),+ $(,)?) => {
        $(
            #[test]
            fn $name() {
                $runner(stringify!($name));
            }
        )+
    };
}

fuzz_tests!(
    run_lstm_test,
    fuzz_lstm_0,
    fuzz_lstm_1,
    fuzz_lstm_2,
    fuzz_lstm_3
);
fuzz_tests!(run_gru_test, fuzz_gru_0, fuzz_gru_1, fuzz_gru_2, fuzz_gru_3);
fuzz_tests!(
    run_mlp_f32_test,
    fuzz_mlp_f32_0,
    fuzz_mlp_f32_1,
    fuzz_mlp_f32_2,
    fuzz_mlp_f32_3,
);
fuzz_tests!(
    run_conv1d_test,
    fuzz_conv1d_0,
    fuzz_conv1d_1,
    fuzz_conv1d_2,
    fuzz_conv1d_3,
);
fuzz_tests!(
    run_stacked_lstm_test,
    fuzz_stacked_lstm_0,
    fuzz_stacked_lstm_1,
    fuzz_stacked_lstm_2,
    fuzz_stacked_lstm_3,
);
fuzz_tests!(
    run_stacked_gru_test,
    fuzz_stacked_gru_0,
    fuzz_stacked_gru_1,
    fuzz_stacked_gru_2,
    fuzz_stacked_gru_3,
);
fuzz_tests!(run_ssm_test, fuzz_ssm_0, fuzz_ssm_1, fuzz_ssm_2, fuzz_ssm_3,);

// ---- BNN tests ----

fn run_bnn_test(name: &str) {
    let data = load_model(name);
    let exp = load_expected(name);
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut bnn = Bnn::from_safetensors(&data, exp["prefix"].as_str().unwrap()).unwrap();

    for (i, (inp, exp_out)) in inputs_f32(&exp)
        .iter()
        .zip(expected_outputs(&exp).iter())
        .enumerate()
    {
        let mut out = vec![0.0_f32; exp_out.len()];
        bnn.predict_into(inp, &mut out);
        for (j, (&actual, &expected)) in out.iter().zip(exp_out.iter()).enumerate() {
            assert_close(name, i, j, actual as f64, expected, tol);
        }
    }
}

#[test]
fn bnn() {
    run_bnn_test("bnn");
}

#[test]
fn bnn_one_binary() {
    run_bnn_test("bnn_one_binary");
}

#[test]
fn bnn_two_binary() {
    run_bnn_test("bnn_two_binary");
}

#[test]
fn bnn_large() {
    run_bnn_test("bnn_large");
}

fuzz_tests!(run_bnn_test, fuzz_bnn_0, fuzz_bnn_1, fuzz_bnn_2, fuzz_bnn_3,);

// ---- TCN tests ----

fn run_tcn_test(name: &str) {
    let data = load_model(name);
    let exp = load_expected(name);
    let tol = exp["tolerance"].as_f64().unwrap();
    let residual = exp["residual"].as_bool().unwrap();

    let mut tcn = TinyTcn::from_safetensors(
        &data,
        exp["prefix"].as_str().unwrap(),
        parse_activation(&exp),
        residual,
    )
    .unwrap();

    for (i, (inp, exp_out)) in inputs_f32(&exp)
        .iter()
        .zip(expected_outputs(&exp).iter())
        .enumerate()
    {
        let mut out = vec![0.0_f32; exp_out.len()];
        tcn.step_into(inp, &mut out);
        for (j, (&actual, &expected)) in out.iter().zip(exp_out.iter()).enumerate() {
            assert_close(name, i, j, actual as f64, expected, tol);
        }
    }
}

#[test]
fn tcn() {
    run_tcn_test("tcn");
}

#[test]
fn tcn_residual() {
    run_tcn_test("tcn_residual");
}

#[test]
fn tcn_identity() {
    run_tcn_test("tcn_identity");
}

#[test]
fn tcn_large() {
    run_tcn_test("tcn_large");
}

fuzz_tests!(run_tcn_test, fuzz_tcn_0, fuzz_tcn_1, fuzz_tcn_2, fuzz_tcn_3,);

// ---- Quantized MLP tests ----

fn run_quantized_mlp_test(name: &str) {
    let data = load_model(name);
    let exp = load_expected(name);
    let tol = exp["tolerance"].as_f64().unwrap();

    let mut qmlp = QuantizedMlp::from_safetensors(
        &data,
        exp["prefix"].as_str().unwrap(),
        parse_activation(&exp),
    )
    .unwrap();

    for (i, (inp, exp_out)) in inputs_f32(&exp)
        .iter()
        .zip(expected_outputs(&exp).iter())
        .enumerate()
    {
        let mut out = vec![0.0_f32; exp_out.len()];
        qmlp.predict_into(inp, &mut out);
        for (j, (&actual, &expected)) in out.iter().zip(exp_out.iter()).enumerate() {
            assert_close(name, i, j, actual as f64, expected, tol);
        }
    }
}

#[test]
fn quantized_mlp_basic() {
    run_quantized_mlp_test("quantized_mlp_basic");
}

#[test]
fn quantized_mlp_identity() {
    run_quantized_mlp_test("quantized_mlp_identity");
}

#[test]
fn quantized_mlp_deep() {
    run_quantized_mlp_test("quantized_mlp_deep");
}

#[test]
fn quantized_mlp_asymmetric() {
    run_quantized_mlp_test("quantized_mlp_asymmetric");
}

fuzz_tests!(
    run_quantized_mlp_test,
    fuzz_quantized_mlp_0,
    fuzz_quantized_mlp_1,
    fuzz_quantized_mlp_2,
    fuzz_quantized_mlp_3,
);

use safetensors::{Dtype, SafeTensors};

use crate::LoadError;

// ---- helpers ----

fn prefixed(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        String::from(name)
    } else {
        format!("{prefix}.{name}")
    }
}

fn parse(data: &[u8]) -> Result<SafeTensors<'_>, LoadError> {
    SafeTensors::deserialize(data).map_err(|_| LoadError::Parse("invalid safetensors data"))
}

fn extract_f32_1d(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>, LoadError> {
    let tv = st
        .tensor(name)
        .map_err(|_| LoadError::TensorNotFound(String::from(name)))?;
    if tv.dtype() != Dtype::F32 {
        return Err(LoadError::Validation("expected F32 tensor"));
    }
    if tv.shape().len() != 1 {
        return Err(LoadError::Validation("expected 1D tensor"));
    }
    let bytes = tv.data();
    if bytes.len() % 4 != 0 {
        return Err(LoadError::Parse("F32 tensor data not aligned"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn extract_f32_2d(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<f32>, [usize; 2]), LoadError> {
    let tv = st
        .tensor(name)
        .map_err(|_| LoadError::TensorNotFound(String::from(name)))?;
    if tv.dtype() != Dtype::F32 {
        return Err(LoadError::Validation("expected F32 tensor"));
    }
    let shape = tv.shape();
    if shape.len() != 2 {
        return Err(LoadError::Validation("expected 2D tensor"));
    }
    let dims = [shape[0], shape[1]];
    let bytes = tv.data();
    if bytes.len() % 4 != 0 {
        return Err(LoadError::Parse("F32 tensor data not aligned"));
    }
    let data = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((data, dims))
}

fn extract_f32_3d(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<f32>, [usize; 3]), LoadError> {
    let tv = st
        .tensor(name)
        .map_err(|_| LoadError::TensorNotFound(String::from(name)))?;
    if tv.dtype() != Dtype::F32 {
        return Err(LoadError::Validation("expected F32 tensor"));
    }
    let shape = tv.shape();
    if shape.len() != 3 {
        return Err(LoadError::Validation("expected 3D tensor"));
    }
    let dims = [shape[0], shape[1], shape[2]];
    let bytes = tv.data();
    if bytes.len() % 4 != 0 {
        return Err(LoadError::Parse("F32 tensor data not aligned"));
    }
    let data = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((data, dims))
}

fn extract_f32_scalar(st: &SafeTensors<'_>, name: &str) -> Result<f32, LoadError> {
    let tv = st
        .tensor(name)
        .map_err(|_| LoadError::TensorNotFound(String::from(name)))?;
    if tv.dtype() != Dtype::F32 {
        return Err(LoadError::Validation("expected F32 tensor"));
    }
    let n: usize = tv.shape().iter().product();
    if n != 1 {
        return Err(LoadError::Validation(
            "expected scalar (single-element) tensor",
        ));
    }
    let bytes = tv.data();
    if bytes.len() != 4 {
        return Err(LoadError::Parse("F32 scalar data not 4 bytes"));
    }
    Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn extract_i8_scalar(st: &SafeTensors<'_>, name: &str) -> Result<i8, LoadError> {
    let tv = st
        .tensor(name)
        .map_err(|_| LoadError::TensorNotFound(String::from(name)))?;
    if tv.dtype() != Dtype::I8 {
        return Err(LoadError::Validation("expected I8 tensor"));
    }
    let n: usize = tv.shape().iter().product();
    if n != 1 {
        return Err(LoadError::Validation(
            "expected scalar (single-element) tensor",
        ));
    }
    let bytes = tv.data();
    if bytes.len() != 1 {
        return Err(LoadError::Parse("I8 scalar data not 1 byte"));
    }
    Ok(bytes[0] as i8)
}

fn extract_i8_2d(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<i8>, [usize; 2]), LoadError> {
    let tv = st
        .tensor(name)
        .map_err(|_| LoadError::TensorNotFound(String::from(name)))?;
    if tv.dtype() != Dtype::I8 {
        return Err(LoadError::Validation("expected I8 tensor"));
    }
    let shape = tv.shape();
    if shape.len() != 2 {
        return Err(LoadError::Validation("expected 2D tensor"));
    }
    let dims = [shape[0], shape[1]];
    let data: Vec<i8> = tv.data().iter().map(|&b| b as i8).collect();
    Ok((data, dims))
}

// ---- sqrt helper for BatchNorm fusion ----

fn sqrt_f64(x: f64) -> f64 {
    x.sqrt()
}

// ---- RNN loaders ----

impl crate::TinyLstm {
    /// Load from safetensors data.
    ///
    /// `rnn_prefix` resolves PyTorch `nn.LSTM` tensors:
    /// `weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0`.
    ///
    /// `proj_prefix` resolves the output projection `nn.Linear`:
    /// `weight`, `bias`.
    ///
    /// Dimensions are inferred from tensor shapes. Only single-layer,
    /// unidirectional LSTM is supported (`num_layers=1`,
    /// `bidirectional=False`).
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::TensorNotFound`] if any required tensor is
    /// missing, or [`LoadError::Validation`] if shapes are inconsistent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let bytes = std::fs::read("model.safetensors")?;
    /// let lstm = TinyLstm::from_safetensors(&bytes, "encoder.lstm", "encoder.fc")?;
    /// ```
    pub fn from_safetensors(
        data: &[u8],
        rnn_prefix: &str,
        proj_prefix: &str,
    ) -> Result<Self, LoadError> {
        let st = parse(data)?;

        let wih_name = prefixed(rnn_prefix, "weight_ih_l0");
        let whh_name = prefixed(rnn_prefix, "weight_hh_l0");
        let bih_name = prefixed(rnn_prefix, "bias_ih_l0");
        let bhh_name = prefixed(rnn_prefix, "bias_hh_l0");
        let wo_name = prefixed(proj_prefix, "weight");
        let bo_name = prefixed(proj_prefix, "bias");

        let (weight_ih, wih_shape) = extract_f32_2d(&st, &wih_name)?;

        if wih_shape[0] % 4 != 0 {
            return Err(LoadError::Validation(
                "weight_ih_l0 rows not divisible by 4 (expected 4*hidden_size)",
            ));
        }
        let hidden_size = wih_shape[0] / 4;
        let input_size = wih_shape[1];

        let (weight_hh, whh_shape) = extract_f32_2d(&st, &whh_name)?;
        if whh_shape != [4 * hidden_size, hidden_size] {
            return Err(LoadError::Validation(
                "weight_hh_l0 shape mismatch (expected [4*hidden, hidden])",
            ));
        }
        let bias_ih = extract_f32_1d(&st, &bih_name)?;
        let bias_hh = extract_f32_1d(&st, &bhh_name)?;
        let (w_out, wo_shape) = extract_f32_2d(&st, &wo_name)?;
        let b_out = extract_f32_1d(&st, &bo_name)?;
        let output_size = wo_shape[0];

        Self::from_parts(
            input_size,
            hidden_size,
            output_size,
            &weight_ih,
            &weight_hh,
            &bias_ih,
            &bias_hh,
            &w_out,
            &b_out,
        )
    }
}

impl crate::TinyGru {
    /// Load from safetensors data.
    ///
    /// `rnn_prefix` resolves PyTorch `nn.GRU` tensors:
    /// `weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0`.
    ///
    /// `proj_prefix` resolves the output projection `nn.Linear`:
    /// `weight`, `bias`.
    ///
    /// Only single-layer, unidirectional GRU is supported
    /// (`num_layers=1`, `bidirectional=False`).
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::TensorNotFound`] if any required tensor is
    /// missing, or [`LoadError::Validation`] if shapes are inconsistent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let bytes = std::fs::read("model.safetensors")?;
    /// let gru = TinyGru::from_safetensors(&bytes, "gru", "fc")?;
    /// ```
    pub fn from_safetensors(
        data: &[u8],
        rnn_prefix: &str,
        proj_prefix: &str,
    ) -> Result<Self, LoadError> {
        let st = parse(data)?;

        let wih_name = prefixed(rnn_prefix, "weight_ih_l0");
        let whh_name = prefixed(rnn_prefix, "weight_hh_l0");
        let bih_name = prefixed(rnn_prefix, "bias_ih_l0");
        let bhh_name = prefixed(rnn_prefix, "bias_hh_l0");
        let wo_name = prefixed(proj_prefix, "weight");
        let bo_name = prefixed(proj_prefix, "bias");

        let (weight_ih, wih_shape) = extract_f32_2d(&st, &wih_name)?;

        if wih_shape[0] % 3 != 0 {
            return Err(LoadError::Validation(
                "weight_ih_l0 rows not divisible by 3 (expected 3*hidden_size)",
            ));
        }
        let hidden_size = wih_shape[0] / 3;
        let input_size = wih_shape[1];

        let (weight_hh, whh_shape) = extract_f32_2d(&st, &whh_name)?;
        if whh_shape != [3 * hidden_size, hidden_size] {
            return Err(LoadError::Validation(
                "weight_hh_l0 shape mismatch (expected [3*hidden, hidden])",
            ));
        }
        let bias_ih = extract_f32_1d(&st, &bih_name)?;
        let bias_hh = extract_f32_1d(&st, &bhh_name)?;
        let (w_out, wo_shape) = extract_f32_2d(&st, &wo_name)?;
        let b_out = extract_f32_1d(&st, &bo_name)?;
        let output_size = wo_shape[0];

        Self::from_parts(
            input_size,
            hidden_size,
            output_size,
            &weight_ih,
            &weight_hh,
            &bias_ih,
            &bias_hh,
            &w_out,
            &b_out,
        )
    }
}

// ---- Stacked RNN loaders ----

impl crate::StackedLstm {
    /// Load from safetensors data.
    ///
    /// Auto-detects `num_layers` by scanning for `weight_ih_l0`,
    /// `weight_ih_l1`, ... tensors under `rnn_prefix`. All layers must
    /// share the same `hidden_size`.
    ///
    /// `proj_prefix` resolves the output projection `nn.Linear`:
    /// `weight`, `bias`.
    ///
    /// **Divergence from PyTorch:** PyTorch's `nn.LSTM` requires
    /// `num_layers` as an explicit constructor argument. We infer it
    /// from the safetensors file by counting consecutive layer weights.
    /// This prevents split-brain errors where the declared layer count
    /// doesn't match the actual weights in the file.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::TensorNotFound`] if required tensors are
    /// missing, or [`LoadError::Validation`] if shapes are inconsistent
    /// or shapes are inconsistent between layers.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let bytes = std::fs::read("model.safetensors")?;
    /// let lstm = StackedLstm::from_safetensors(&bytes, "encoder.lstm", "encoder.fc")?;
    /// ```
    pub fn from_safetensors(
        data: &[u8],
        rnn_prefix: &str,
        proj_prefix: &str,
    ) -> Result<Self, LoadError> {
        let st = parse(data)?;

        let num_layers = count_rnn_layers(&st, rnn_prefix, 4)?;

        let mut layers_wih = Vec::with_capacity(num_layers);
        let mut layers_whh = Vec::with_capacity(num_layers);
        let mut layers_bih = Vec::with_capacity(num_layers);
        let mut layers_bhh = Vec::with_capacity(num_layers);

        let mut hidden_size = 0;
        let mut input_size = 0;

        for k in 0..num_layers {
            let wih_name = prefixed(rnn_prefix, &format!("weight_ih_l{k}"));
            let whh_name = prefixed(rnn_prefix, &format!("weight_hh_l{k}"));
            let bih_name = prefixed(rnn_prefix, &format!("bias_ih_l{k}"));
            let bhh_name = prefixed(rnn_prefix, &format!("bias_hh_l{k}"));

            let (wih, wih_shape) = extract_f32_2d(&st, &wih_name)?;
            if wih_shape[0] % 4 != 0 {
                return Err(LoadError::Validation(
                    "weight_ih rows not divisible by 4 (expected 4*hidden_size)",
                ));
            }
            let h = wih_shape[0] / 4;

            if k == 0 {
                hidden_size = h;
                input_size = wih_shape[1];
            } else {
                if h != hidden_size {
                    return Err(LoadError::Validation(
                        "all layers must have the same hidden_size",
                    ));
                }
                if wih_shape[1] != hidden_size {
                    return Err(LoadError::Validation(
                        "layer 1+ weight_ih columns must equal hidden_size",
                    ));
                }
            }

            let (whh, whh_shape) = extract_f32_2d(&st, &whh_name)?;
            if whh_shape != [4 * hidden_size, hidden_size] {
                return Err(LoadError::Validation(
                    "weight_hh shape mismatch (expected [4*hidden, hidden])",
                ));
            }

            let bih = extract_f32_1d(&st, &bih_name)?;
            let bhh = extract_f32_1d(&st, &bhh_name)?;

            layers_wih.push(wih);
            layers_whh.push(whh);
            layers_bih.push(bih);
            layers_bhh.push(bhh);
        }

        let wo_name = prefixed(proj_prefix, "weight");
        let bo_name = prefixed(proj_prefix, "bias");
        let (w_out, wo_shape) = extract_f32_2d(&st, &wo_name)?;
        let b_out = extract_f32_1d(&st, &bo_name)?;
        let output_size = wo_shape[0];

        let wih_refs: Vec<&[f32]> = layers_wih.iter().map(Vec::as_slice).collect();
        let whh_refs: Vec<&[f32]> = layers_whh.iter().map(Vec::as_slice).collect();
        let bih_refs: Vec<&[f32]> = layers_bih.iter().map(Vec::as_slice).collect();
        let bhh_refs: Vec<&[f32]> = layers_bhh.iter().map(Vec::as_slice).collect();

        Self::from_parts(
            input_size,
            hidden_size,
            output_size,
            &wih_refs,
            &whh_refs,
            &bih_refs,
            &bhh_refs,
            &w_out,
            &b_out,
        )
    }
}

impl crate::StackedGru {
    /// Load from safetensors data.
    ///
    /// Auto-detects `num_layers` by scanning for `weight_ih_l0`,
    /// `weight_ih_l1`, ... tensors under `rnn_prefix`. All layers must
    /// share the same `hidden_size`.
    ///
    /// `proj_prefix` resolves the output projection `nn.Linear`:
    /// `weight`, `bias`.
    ///
    /// **Divergence from PyTorch:** PyTorch's `nn.GRU` requires
    /// `num_layers` as an explicit constructor argument. We infer it
    /// from the safetensors file by counting consecutive layer weights.
    /// This prevents split-brain errors where the declared layer count
    /// doesn't match the actual weights in the file.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::TensorNotFound`] if required tensors are
    /// missing, or [`LoadError::Validation`] if shapes are inconsistent
    /// or shapes are inconsistent between layers.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let bytes = std::fs::read("model.safetensors")?;
    /// let gru = StackedGru::from_safetensors(&bytes, "encoder.gru", "encoder.fc")?;
    /// ```
    pub fn from_safetensors(
        data: &[u8],
        rnn_prefix: &str,
        proj_prefix: &str,
    ) -> Result<Self, LoadError> {
        let st = parse(data)?;

        let num_layers = count_rnn_layers(&st, rnn_prefix, 3)?;

        let mut layers_wih = Vec::with_capacity(num_layers);
        let mut layers_whh = Vec::with_capacity(num_layers);
        let mut layers_bih = Vec::with_capacity(num_layers);
        let mut layers_bhh = Vec::with_capacity(num_layers);

        let mut hidden_size = 0;
        let mut input_size = 0;

        for k in 0..num_layers {
            let wih_name = prefixed(rnn_prefix, &format!("weight_ih_l{k}"));
            let whh_name = prefixed(rnn_prefix, &format!("weight_hh_l{k}"));
            let bih_name = prefixed(rnn_prefix, &format!("bias_ih_l{k}"));
            let bhh_name = prefixed(rnn_prefix, &format!("bias_hh_l{k}"));

            let (wih, wih_shape) = extract_f32_2d(&st, &wih_name)?;
            if wih_shape[0] % 3 != 0 {
                return Err(LoadError::Validation(
                    "weight_ih rows not divisible by 3 (expected 3*hidden_size)",
                ));
            }
            let h = wih_shape[0] / 3;

            if k == 0 {
                hidden_size = h;
                input_size = wih_shape[1];
            } else {
                if h != hidden_size {
                    return Err(LoadError::Validation(
                        "all layers must have the same hidden_size",
                    ));
                }
                if wih_shape[1] != hidden_size {
                    return Err(LoadError::Validation(
                        "layer 1+ weight_ih columns must equal hidden_size",
                    ));
                }
            }

            let (whh, whh_shape) = extract_f32_2d(&st, &whh_name)?;
            if whh_shape != [3 * hidden_size, hidden_size] {
                return Err(LoadError::Validation(
                    "weight_hh shape mismatch (expected [3*hidden, hidden])",
                ));
            }

            let bih = extract_f32_1d(&st, &bih_name)?;
            let bhh = extract_f32_1d(&st, &bhh_name)?;

            layers_wih.push(wih);
            layers_whh.push(whh);
            layers_bih.push(bih);
            layers_bhh.push(bhh);
        }

        let wo_name = prefixed(proj_prefix, "weight");
        let bo_name = prefixed(proj_prefix, "bias");
        let (w_out, wo_shape) = extract_f32_2d(&st, &wo_name)?;
        let b_out = extract_f32_1d(&st, &bo_name)?;
        let output_size = wo_shape[0];

        let wih_refs: Vec<&[f32]> = layers_wih.iter().map(Vec::as_slice).collect();
        let whh_refs: Vec<&[f32]> = layers_whh.iter().map(Vec::as_slice).collect();
        let bih_refs: Vec<&[f32]> = layers_bih.iter().map(Vec::as_slice).collect();
        let bhh_refs: Vec<&[f32]> = layers_bhh.iter().map(Vec::as_slice).collect();

        Self::from_parts(
            input_size,
            hidden_size,
            output_size,
            &wih_refs,
            &whh_refs,
            &bih_refs,
            &bhh_refs,
            &w_out,
            &b_out,
        )
    }
}

/// Count consecutive `weight_ih_l{k}` tensors to detect num_layers.
/// `gate_mult` is 4 for LSTM, 3 for GRU.
///
/// PyTorch emits layer weights as consecutive `weight_ih_l0..l{N-1}`.
/// A gap in the sequence (e.g. `l0` and `l2` present but `l1` missing)
/// means a malformed file or a wrong `rnn_prefix`. We reject it rather
/// than silently load a truncated model from the consecutive prefix.
fn count_rnn_layers(
    st: &SafeTensors<'_>,
    rnn_prefix: &str,
    gate_mult: usize,
) -> Result<usize, LoadError> {
    let mut num_layers = 0;
    loop {
        let name = prefixed(rnn_prefix, &format!("weight_ih_l{num_layers}"));
        match st.tensor(&name) {
            Ok(tv) => {
                let shape = tv.shape();
                if shape.len() != 2 || shape[0] % gate_mult != 0 {
                    return Err(LoadError::Validation(
                        "unexpected weight_ih shape for RNN layer",
                    ));
                }
                num_layers += 1;
            }
            Err(_) => break,
        }
    }
    if num_layers == 0 {
        return Err(LoadError::TensorNotFound(prefixed(
            rnn_prefix,
            "weight_ih_l0",
        )));
    }

    // Reject orphaned higher-index layers past the consecutive run.
    let stem = prefixed(rnn_prefix, "weight_ih_l");
    for name in st.names() {
        if let Some(suffix) = name.strip_prefix(stem.as_str())
            && let Ok(idx) = suffix.parse::<usize>()
            && idx >= num_layers
        {
            return Err(LoadError::Validation(
                "non-consecutive RNN layer indices (gap in weight_ih_l{k})",
            ));
        }
    }

    Ok(num_layers)
}

// ---- MLP loaders ----

macro_rules! impl_mlp_safetensors {
    ($name:ident, $ty:ty, $extract_2d:ident, $extract_1d:ident) => {
        impl crate::$name {
            /// Load from safetensors data.
            ///
            /// Discovers linear layers by scanning for `{prefix}.{N}.weight`
            /// tensors where `N` is a numeric index (PyTorch `nn.Sequential`
            /// default naming). Non-contiguous indices are handled — activation
            /// layers in Sequential skip numbering.
            ///
            /// `activation` applies to all hidden layers. The final layer has
            /// no activation (same as [`from_parts`](Self::from_parts)).
            ///
            /// Layers trained with `bias=False` are handled automatically —
            /// missing bias tensors are treated as zero bias.
            ///
            /// `BatchNorm1d` layers between linear layers are detected by
            /// `running_mean` presence and fused into the preceding linear
            /// layer at load time (requires `std` or `libm` for `sqrt`).
            /// Both `affine=True` (default) and `affine=False` are supported.
            ///
            /// `LayerNorm` layers are detected by 1D `.weight` tensors
            /// between linear layers (without `running_mean`). LayerNorm
            /// is applied at inference time with eps=1e-5 (PyTorch default).
            /// Requires `std` or `libm`.
            ///
            /// # Errors
            ///
            /// Returns [`LoadError::Validation`] if layer dimensions are
            /// inconsistent, or [`LoadError::Parse`] if no linear layers
            /// are found.
            ///
            /// # Examples
            ///
            /// ```ignore
            /// let bytes = std::fs::read("model.safetensors")?;
            /// let mlp = Mlp::from_safetensors(&bytes, "fc", Activation::Relu)?;
            /// ```
            pub fn from_safetensors(
                data: &[u8],
                prefix: &str,
                activation: crate::Activation,
            ) -> Result<Self, LoadError> {
                let st = parse(data)?;
                let prefix_dot = if prefix.is_empty() {
                    String::new()
                } else {
                    format!("{prefix}.")
                };

                let mut layer_indices: Vec<usize> = Vec::new();
                let mut batchnorm_indices: Vec<usize> = Vec::new();
                let mut onedim_weight_indices: Vec<usize> = Vec::new();
                for name in st.names() {
                    let suffix = if prefix.is_empty() {
                        name.as_ref()
                    } else {
                        match name.strip_prefix(prefix_dot.as_str()) {
                            Some(s) => s,
                            None => continue,
                        }
                    };
                    if let Some(idx_str) = suffix.strip_suffix(".weight") {
                        if let Ok(idx) = idx_str.parse::<usize>() {
                            if let Ok(tv) = st.tensor(name) {
                                if tv.shape().len() == 2 {
                                    layer_indices.push(idx);
                                } else if tv.shape().len() == 1 {
                                    onedim_weight_indices.push(idx);
                                }
                            }
                        }
                    }
                    if let Some(idx_str) = suffix.strip_suffix(".running_mean") {
                        if let Ok(idx) = idx_str.parse::<usize>() {
                            batchnorm_indices.push(idx);
                        }
                    }
                }
                layer_indices.sort_unstable();
                batchnorm_indices.sort_unstable();
                let mut layernorm_indices: Vec<usize> = onedim_weight_indices
                    .into_iter()
                    .filter(|i| !batchnorm_indices.contains(i))
                    .collect();
                layernorm_indices.sort_unstable();

                if layer_indices.is_empty() {
                    return Err(LoadError::Parse("no linear layers found in safetensors"));
                }

                let mut layer_sizes: Vec<usize> = Vec::new();
                let mut all_weights: Vec<$ty> = Vec::new();
                let mut all_biases: Vec<$ty> = Vec::new();
                let mut ln_gamma_data: Vec<$ty> = Vec::new();
                let mut ln_beta_data: Vec<$ty> = Vec::new();
                let mut has_layernorm = false;
                let n_linear = layer_indices.len();

                for (i, &idx) in layer_indices.iter().enumerate() {
                    let w_name = format!("{prefix_dot}{idx}.weight");
                    let b_name = format!("{prefix_dot}{idx}.bias");

                    let (mut w_data, w_shape) = $extract_2d(&st, &w_name)?;
                    let mut b_data = match $extract_1d(&st, &b_name) {
                        Ok(b) => {
                            if w_shape[0] != b.len() {
                                return Err(LoadError::Validation("weight rows != bias length"));
                            }
                            b
                        }
                        Err(LoadError::TensorNotFound(_)) => vec![0.0 as $ty; w_shape[0]],
                        Err(e) => return Err(e),
                    };
                    if i == 0 {
                        layer_sizes.push(w_shape[1]);
                    } else if *layer_sizes.last().unwrap() != w_shape[1] {
                        return Err(LoadError::Validation(
                            "layer input size doesn't match previous output",
                        ));
                    }
                    layer_sizes.push(w_shape[0]);

                    // Fuse BatchNorm if one exists between this linear and the next
                    let next_linear = layer_indices.get(i + 1).copied().unwrap_or(usize::MAX);
                    if let Some(&bn_idx) = batchnorm_indices
                        .iter()
                        .find(|&&bi| bi > idx && bi < next_linear)
                    {
                        let bn_mean =
                            $extract_1d(&st, &format!("{prefix_dot}{bn_idx}.running_mean"))?;
                        let bn_var =
                            $extract_1d(&st, &format!("{prefix_dot}{bn_idx}.running_var"))?;
                        let out_features = w_shape[0];
                        let in_features = w_shape[1];
                        if bn_mean.len() != out_features || bn_var.len() != out_features {
                            return Err(LoadError::Validation(
                                "BatchNorm size mismatch with linear output",
                            ));
                        }
                        let bn_gamma: Vec<$ty> =
                            match $extract_1d(&st, &format!("{prefix_dot}{bn_idx}.weight")) {
                                Ok(g) => {
                                    if g.len() != out_features {
                                        return Err(LoadError::Validation(
                                            "BatchNorm gamma size mismatch",
                                        ));
                                    }
                                    g
                                }
                                Err(LoadError::TensorNotFound(_)) => {
                                    vec![1.0 as $ty; out_features]
                                }
                                Err(e) => return Err(e),
                            };
                        let bn_beta: Vec<$ty> =
                            match $extract_1d(&st, &format!("{prefix_dot}{bn_idx}.bias")) {
                                Ok(b) => {
                                    if b.len() != out_features {
                                        return Err(LoadError::Validation(
                                            "BatchNorm beta size mismatch",
                                        ));
                                    }
                                    b
                                }
                                Err(LoadError::TensorNotFound(_)) => {
                                    vec![0.0 as $ty; out_features]
                                }
                                Err(e) => return Err(e),
                            };
                        let eps = 1e-5_f64;
                        for row in 0..out_features {
                            let scale =
                                bn_gamma[row] as f64 / sqrt_f64(bn_var[row] as f64 + eps);
                            for col in 0..in_features {
                                let wi = row * in_features + col;
                                w_data[wi] = (w_data[wi] as f64 * scale) as $ty;
                            }
                            b_data[row] = scale.mul_add(
                                b_data[row] as f64 - bn_mean[row] as f64,
                                bn_beta[row] as f64,
                            ) as $ty;
                        }
                    }

                    // Detect LayerNorm for hidden layers
                    let is_last_layer = i == n_linear - 1;
                    if !is_last_layer {
                        if let Some(&ln_idx) = layernorm_indices
                            .iter()
                            .find(|&&li| li > idx && li < next_linear)
                        {
                            has_layernorm = true;
                            let ln_g = $extract_1d(&st, &format!("{prefix_dot}{ln_idx}.weight"))?;
                            let ln_b = match $extract_1d(&st, &format!("{prefix_dot}{ln_idx}.bias"))
                            {
                                Ok(b) => b,
                                Err(LoadError::TensorNotFound(_)) => {
                                    vec![0.0 as $ty; w_shape[0]]
                                }
                                Err(e) => return Err(e),
                            };
                            if ln_g.len() != w_shape[0] || ln_b.len() != w_shape[0] {
                                return Err(LoadError::Validation(
                                    "LayerNorm size mismatch with linear output",
                                ));
                            }
                            ln_gamma_data.extend_from_slice(&ln_g);
                            ln_beta_data.extend_from_slice(&ln_b);
                        }
                    }

                    all_weights.extend_from_slice(&w_data);
                    all_biases.extend_from_slice(&b_data);
                }

                if has_layernorm {
                    let n_hidden = n_linear - 1;
                    let expected_ln: usize = (0..n_hidden).map(|l| layer_sizes[l + 1]).sum();
                    if ln_gamma_data.len() != expected_ln {
                        return Err(LoadError::Validation(
                            "LayerNorm must be present on all hidden layers or none",
                        ));
                    }
                    return Self::from_parts_with_layer_norm(
                        &layer_sizes,
                        &all_weights,
                        &all_biases,
                        &ln_gamma_data,
                        &ln_beta_data,
                        activation,
                    );
                }

                Self::from_parts(&layer_sizes, &all_weights, &all_biases, activation)
            }
        }
    };
}

impl_mlp_safetensors!(Mlp, f32, extract_f32_2d, extract_f32_1d);

// ---- Conv1d loader ----

impl crate::Causal1dConv {
    /// Load from safetensors data.
    ///
    /// `conv_prefix` resolves PyTorch `nn.Conv1d` tensors:
    /// `weight` (shape `[filters, input_ch, kernel_size]`), `bias`.
    ///
    /// `proj_prefix` resolves the output projection `nn.Linear`:
    /// `weight`, `bias`.
    ///
    /// PyTorch stores Conv1d weights as `(out_ch, in_ch, kernel)` where
    /// kernel position 0 corresponds to the oldest input. Our layout is
    /// `(filters, kernel, in_ch)` where position 0 is the newest
    /// (current) input. This loader transposes and reverses the kernel
    /// dimension.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::TensorNotFound`] if any required tensor is
    /// missing, or [`LoadError::Validation`] if shapes are inconsistent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let bytes = std::fs::read("model.safetensors")?;
    /// let conv = Causal1dConv::from_safetensors(
    ///     &bytes, "conv", "fc", Activation::Relu,
    /// )?;
    /// ```
    pub fn from_safetensors(
        data: &[u8],
        conv_prefix: &str,
        proj_prefix: &str,
        activation: crate::Activation,
    ) -> Result<Self, LoadError> {
        let st = parse(data)?;

        let wc_name = prefixed(conv_prefix, "weight");
        let bc_name = prefixed(conv_prefix, "bias");
        let wo_name = prefixed(proj_prefix, "weight");
        let bo_name = prefixed(proj_prefix, "bias");

        let (wc_pt, wc_shape) = extract_f32_3d(&st, &wc_name)?;
        let b_conv = extract_f32_1d(&st, &bc_name)?;
        let (w_out, wo_shape) = extract_f32_2d(&st, &wo_name)?;
        let b_out = extract_f32_1d(&st, &bo_name)?;

        // PyTorch shape: (out_channels, in_channels, kernel_size)
        let filters = wc_shape[0];
        let input_ch = wc_shape[1];
        let kernel_size = wc_shape[2];
        let output_size = wo_shape[0];

        // Transpose + reverse: PyTorch (F, C, K) → our (F, K, C)
        // PyTorch k=0 is oldest, k=K-1 is newest.
        // Our k=0 is newest (current), k=K-1 is oldest.
        let mut w_conv = vec![0.0_f32; filters * kernel_size * input_ch];
        for f in 0..filters {
            for k in 0..kernel_size {
                let pt_k = kernel_size - 1 - k;
                for c in 0..input_ch {
                    w_conv[f * kernel_size * input_ch + k * input_ch + c] =
                        wc_pt[f * input_ch * kernel_size + c * kernel_size + pt_k];
                }
            }
        }

        Self::from_parts(
            input_ch,
            kernel_size,
            filters,
            output_size,
            &w_conv,
            &b_conv,
            &w_out,
            &b_out,
            activation,
        )
    }
}

// ---- SSM loader ----

impl crate::LinearSsm {
    /// Load from safetensors data.
    ///
    /// Expected tensors under `prefix`:
    /// - `a_diag`: 1D `[H]` — diagonal of state transition matrix (pre-discretized)
    /// - `b`: 2D `[H, I]` — input-to-state matrix
    /// - `c`: 2D `[O, H]` — state-to-output matrix
    /// - `d`: 2D `[O, I]` — skip connection (optional, defaults to zeros)
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::TensorNotFound`] if `a_diag`, `b`, or `c` are
    /// missing, or [`LoadError::Validation`] if shapes are inconsistent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let bytes = std::fs::read("ssm.safetensors")?;
    /// let ssm = LinearSsm::from_safetensors(&bytes, "ssm")?;
    /// ```
    pub fn from_safetensors(data: &[u8], prefix: &str) -> Result<Self, LoadError> {
        let st = parse(data)?;

        let a_name = prefixed(prefix, "a_diag");
        let b_name = prefixed(prefix, "b");
        let c_name = prefixed(prefix, "c");
        let d_name = prefixed(prefix, "d");

        let a_diag = extract_f32_1d(&st, &a_name)?;
        let (b, b_shape) = extract_f32_2d(&st, &b_name)?;
        let (c, c_shape) = extract_f32_2d(&st, &c_name)?;

        let hidden_size = a_diag.len();
        let input_size = b_shape[1];
        let output_size = c_shape[0];

        if b_shape[0] != hidden_size {
            return Err(LoadError::Validation(
                "b rows must equal hidden_size (a_diag length)",
            ));
        }
        if c_shape[1] != hidden_size {
            return Err(LoadError::Validation(
                "c columns must equal hidden_size (a_diag length)",
            ));
        }

        let d = match extract_f32_2d(&st, &d_name) {
            Ok((d_data, d_shape)) => {
                if d_shape != [output_size, input_size] {
                    return Err(LoadError::Validation(
                        "d shape must be [output_size, input_size]",
                    ));
                }
                d_data
            }
            Err(LoadError::TensorNotFound(_)) => {
                vec![0.0_f32; output_size * input_size]
            }
            Err(e) => return Err(e),
        };

        Self::from_parts(&a_diag, &b, &c, &d, output_size)
    }
}

// ---- TCN loader ----

fn count_tcn_layers(st: &SafeTensors<'_>, prefix: &str) -> Result<usize, LoadError> {
    let mut n = 0;
    loop {
        let name = format!("{prefix}.conv_{n}.weight");
        if st.tensor(&name).is_ok() {
            n += 1;
        } else {
            break;
        }
    }
    if n == 0 {
        return Err(LoadError::TensorNotFound(format!("{prefix}.conv_0.weight")));
    }

    // Reject orphaned per-layer tensors (weight or bias) past the
    // consecutive run — a stray conv_{k}.bias signals a malformed file.
    let stem = format!("{prefix}.conv_");
    for name in st.names() {
        let idx_str = name.strip_prefix(stem.as_str()).and_then(|s| {
            s.strip_suffix(".weight")
                .or_else(|| s.strip_suffix(".bias"))
        });
        if let Some(idx_str) = idx_str
            && let Ok(idx) = idx_str.parse::<usize>()
            && idx >= n
        {
            return Err(LoadError::Validation(
                "non-consecutive TCN layer indices (gap in conv_{k})",
            ));
        }
    }

    Ok(n)
}

impl crate::TinyTcn {
    /// Load from safetensors data.
    ///
    /// Tensor naming convention:
    /// - `{prefix}.conv_0.weight` — F32 `[filters, input_ch, kernel_size]`
    /// - `{prefix}.conv_0.bias` — F32 `[filters]`
    /// - `{prefix}.conv_1.weight` — F32 `[filters, filters, kernel_size]`
    /// - `{prefix}.conv_1.bias` — F32 `[filters]` ...
    /// - `{prefix}.output.weight` — F32 `[output_size, filters]`
    /// - `{prefix}.output.bias` — F32 `[output_size]`
    ///
    /// Layer count is auto-detected by scanning for consecutive
    /// `conv_{k}` tensors. PyTorch stores Conv1d weights as
    /// `(out_ch, in_ch, kernel)` where kernel position 0 is the oldest
    /// input; this loader transposes and reverses the kernel dimension
    /// to match our `(filters, kernel, in_ch)` layout where position 0
    /// is the newest (current) input.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::TensorNotFound`] if required tensors are
    /// missing, or [`LoadError::Validation`] if shapes are inconsistent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let bytes = std::fs::read("tcn.safetensors")?;
    /// let tcn = TinyTcn::from_safetensors(
    ///     &bytes, "tcn", Activation::Relu, false,
    /// )?;
    /// ```
    pub fn from_safetensors(
        data: &[u8],
        prefix: &str,
        activation: crate::Activation,
        residual: bool,
    ) -> Result<Self, LoadError> {
        let st = parse(data)?;
        let num_layers = count_tcn_layers(&st, prefix)?;

        let mut layers_w_conv = Vec::with_capacity(num_layers);
        let mut layers_b_conv = Vec::with_capacity(num_layers);

        let mut input_size = 0;
        let mut filters = 0;
        let mut kernel_size = 0;

        for k in 0..num_layers {
            let w_name = format!("{prefix}.conv_{k}.weight");
            let b_name = format!("{prefix}.conv_{k}.bias");

            let (wc_pt, wc_shape) = extract_f32_3d(&st, &w_name)?;
            let b_conv = extract_f32_1d(&st, &b_name)?;

            let f = wc_shape[0];
            let ic = wc_shape[1];
            let ks = wc_shape[2];

            if k == 0 {
                input_size = ic;
                filters = f;
                kernel_size = ks;
            } else {
                if f != filters {
                    return Err(LoadError::Validation(
                        "inconsistent filter count across TCN layers",
                    ));
                }
                if ic != filters {
                    return Err(LoadError::Validation(
                        "TCN layer input channels must equal filters for layer > 0",
                    ));
                }
                if ks != kernel_size {
                    return Err(LoadError::Validation(
                        "inconsistent kernel size across TCN layers",
                    ));
                }
            }

            // Transpose + reverse: PyTorch (F, C, K) → our (F, K, C) with kernel reversed
            let mut w_conv = vec![0.0_f32; f * ks * ic];
            for fi in 0..f {
                for ki in 0..ks {
                    let pt_k = ks - 1 - ki;
                    for ci in 0..ic {
                        w_conv[fi * ks * ic + ki * ic + ci] = wc_pt[fi * ic * ks + ci * ks + pt_k];
                    }
                }
            }

            layers_w_conv.push(w_conv);
            layers_b_conv.push(b_conv);
        }

        let wo_name = format!("{prefix}.output.weight");
        let bo_name = format!("{prefix}.output.bias");
        let (w_out, wo_shape) = extract_f32_2d(&st, &wo_name)?;
        let b_out = extract_f32_1d(&st, &bo_name)?;
        let output_size = wo_shape[0];

        let w_refs: Vec<&[f32]> = layers_w_conv.iter().map(Vec::as_slice).collect();
        let b_refs: Vec<&[f32]> = layers_b_conv.iter().map(Vec::as_slice).collect();

        Self::from_parts(
            input_size,
            filters,
            kernel_size,
            output_size,
            residual,
            &w_refs,
            &b_refs,
            &w_out,
            &b_out,
            activation,
        )
    }
}

fn pack_i8_to_u64(weights: &[i8], rows: usize, cols: usize) -> Vec<u64> {
    debug_assert_eq!(weights.len(), rows * cols);
    debug_assert_eq!(cols % 64, 0);
    let wpr = cols / 64;
    let mut packed = vec![0_u64; rows * wpr];
    for r in 0..rows {
        for c in 0..cols {
            if weights[r * cols + c] == 1 {
                packed[r * wpr + c / 64] |= 1 << (c % 64);
            }
        }
    }
    packed
}

/// Count consecutive `binary_weight_{k}` tensors. Zero is valid (a BNN
/// with no binary layers is a single fp32 layer). A gap in the sequence
/// (e.g. `binary_weight_0` and `_2` present but `_1` missing) means a
/// malformed file or wrong `prefix`; reject it rather than silently load
/// a truncated network.
fn count_bnn_binary_layers(st: &SafeTensors<'_>, prefix: &str) -> Result<usize, LoadError> {
    let mut n = 0;
    loop {
        let name = prefixed(prefix, &format!("binary_weight_{n}"));
        if st.tensor(&name).is_ok() {
            n += 1;
        } else {
            break;
        }
    }

    // Reject orphaned higher-index layers past the consecutive run.
    let stem = prefixed(prefix, "binary_weight_");
    for name in st.names() {
        if let Some(suffix) = name.strip_prefix(stem.as_str())
            && let Ok(idx) = suffix.parse::<usize>()
            && idx >= n
        {
            return Err(LoadError::Validation(
                "non-consecutive BNN binary layer indices (gap in binary_weight_{k})",
            ));
        }
    }

    Ok(n)
}

impl crate::Bnn {
    /// Load from safetensors data.
    ///
    /// Tensor naming convention:
    /// - `{prefix}.input_weight` — F32 `[H, I]`
    /// - `{prefix}.input_bias` — F32 `[H]`
    /// - `{prefix}.binary_weight_0` — I8 `[H, H]` (±1 values)
    /// - `{prefix}.binary_bias_0` — F32 `[H]`
    /// - `{prefix}.binary_weight_1`, `binary_bias_1`, ... (optional)
    /// - `{prefix}.output_weight` — F32 `[O, H]`
    /// - `{prefix}.output_bias` — F32 `[O]`
    ///
    /// Binary layer count is auto-detected by scanning for consecutive
    /// `binary_weight_{k}` tensors. Binary weights are stored as I8 ±1
    /// and packed to u64 at load time (bit 1 = +1, bit 0 = −1).
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::TensorNotFound`] if required tensors are
    /// missing, or [`LoadError::Validation`] if shapes are inconsistent
    /// or hidden size is not a multiple of 64.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let bytes = std::fs::read("bnn.safetensors")?;
    /// let bnn = Bnn::from_safetensors(&bytes, "bnn")?;
    /// ```
    pub fn from_safetensors(data: &[u8], prefix: &str) -> Result<Self, LoadError> {
        let st = parse(data)?;

        let wi_name = prefixed(prefix, "input_weight");
        let bi_name = prefixed(prefix, "input_bias");
        let wo_name = prefixed(prefix, "output_weight");
        let bo_name = prefixed(prefix, "output_bias");

        let (w_input, wi_shape) = extract_f32_2d(&st, &wi_name)?;
        let b_input = extract_f32_1d(&st, &bi_name)?;
        let (w_output, wo_shape) = extract_f32_2d(&st, &wo_name)?;
        let b_output = extract_f32_1d(&st, &bo_name)?;

        let hidden_size = wi_shape[0];
        let output_size = wo_shape[0];

        if b_input.len() != hidden_size {
            return Err(LoadError::Validation(
                "input_bias length must equal hidden_size",
            ));
        }
        if wo_shape[1] != hidden_size {
            return Err(LoadError::Validation(
                "output_weight columns must equal hidden_size",
            ));
        }
        // Validate before packing: pack_i8_to_u64 assumes H % 64 == 0 and
        // would index out of bounds otherwise (from_parts checks this too,
        // but it runs after packing).
        if !hidden_size.is_multiple_of(64) {
            return Err(LoadError::Validation(
                "hidden_size must be a multiple of 64",
            ));
        }

        let num_binary = count_bnn_binary_layers(&st, prefix)?;

        let mut binary_weights_packed = Vec::with_capacity(num_binary);
        let mut binary_biases = Vec::with_capacity(num_binary);

        for k in 0..num_binary {
            let bw_name = prefixed(prefix, &format!("binary_weight_{k}"));
            let bb_name = prefixed(prefix, &format!("binary_bias_{k}"));

            let (bw_i8, bw_shape) = extract_i8_2d(&st, &bw_name)?;
            let bb = extract_f32_1d(&st, &bb_name)?;

            if bw_shape != [hidden_size, hidden_size] {
                return Err(LoadError::Validation("binary_weight shape must be [H, H]"));
            }
            if bb.len() != hidden_size {
                return Err(LoadError::Validation(
                    "binary_bias length must equal hidden_size",
                ));
            }
            if bw_i8.iter().any(|&v| v != -1 && v != 1) {
                return Err(LoadError::Validation("binary weights must be -1 or 1"));
            }

            let packed = pack_i8_to_u64(&bw_i8, hidden_size, hidden_size);
            binary_weights_packed.push(packed);
            binary_biases.push(bb);
        }

        let bw_refs: Vec<&[u64]> = binary_weights_packed.iter().map(Vec::as_slice).collect();
        let bb_refs: Vec<&[f32]> = binary_biases.iter().map(Vec::as_slice).collect();

        Self::from_parts(
            &w_input,
            &b_input,
            &bw_refs,
            &bb_refs,
            &w_output,
            &b_output,
            output_size,
        )
    }
}

// ---- Quantized MLP loader ----

fn count_quantized_mlp_layers(st: &SafeTensors<'_>, prefix: &str) -> Result<usize, LoadError> {
    let mut n = 0;
    loop {
        let name = prefixed(prefix, &format!("layer_{n}.weight"));
        match st.tensor(&name) {
            Ok(tv) if tv.dtype() == Dtype::I8 && tv.shape().len() == 2 => n += 1,
            _ => break,
        }
    }

    if n == 0 {
        return Ok(0);
    }

    let stem = prefixed(prefix, "layer_");
    for name in st.names() {
        let Some(suffix) = name.strip_prefix(stem.as_str()) else {
            continue;
        };
        if let Some(idx_str) = suffix
            .strip_suffix(".weight")
            .or_else(|| suffix.strip_suffix(".bias"))
            .or_else(|| suffix.strip_suffix(".weight_scale"))
            .or_else(|| suffix.strip_suffix(".weight_zero_point"))
            .or_else(|| suffix.strip_suffix(".input_scale"))
            .or_else(|| suffix.strip_suffix(".input_zero_point"))
            && let Ok(idx) = idx_str.parse::<usize>()
            && idx >= n
        {
            return Err(LoadError::Validation(
                "non-consecutive quantized MLP layer indices",
            ));
        }
    }

    Ok(n)
}

impl crate::QuantizedMlp {
    /// Load from safetensors data.
    ///
    /// Tensor naming convention (per layer `k`):
    /// - `{prefix}.layer_{k}.weight` — I8 `[out, in]`
    /// - `{prefix}.layer_{k}.bias` — F32 `[out]`
    /// - `{prefix}.layer_{k}.weight_scale` — F32 scalar
    /// - `{prefix}.layer_{k}.weight_zero_point` — I8 scalar
    /// - `{prefix}.layer_{k}.input_scale` — F32 scalar
    /// - `{prefix}.layer_{k}.input_zero_point` — I8 scalar
    ///
    /// Layer count is auto-detected by scanning for consecutive
    /// `layer_{k}.weight` I8 tensors starting from `k=0`.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::TensorNotFound`] if required tensors are
    /// missing, or [`LoadError::Validation`] if shapes or quantization
    /// parameters are invalid.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let bytes = std::fs::read("quantized_mlp.safetensors")?;
    /// let qmlp = QuantizedMlp::from_safetensors(&bytes, "qmlp", Activation::Relu)?;
    /// ```
    pub fn from_safetensors(
        data: &[u8],
        prefix: &str,
        activation: crate::Activation,
    ) -> Result<Self, LoadError> {
        let st = parse(data)?;
        let n_layers = count_quantized_mlp_layers(&st, prefix)?;
        if n_layers == 0 {
            return Err(LoadError::TensorNotFound(prefixed(
                prefix,
                "layer_0.weight",
            )));
        }

        let mut layers_w: Vec<Vec<i8>> = Vec::with_capacity(n_layers);
        let mut layers_b: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        let mut w_scales: Vec<f32> = Vec::with_capacity(n_layers);
        let mut w_zero_points: Vec<i8> = Vec::with_capacity(n_layers);
        let mut input_scales: Vec<f32> = Vec::with_capacity(n_layers);
        let mut input_zero_points: Vec<i8> = Vec::with_capacity(n_layers);

        let mut prev_out: Option<usize> = None;

        for k in 0..n_layers {
            let w_name = prefixed(prefix, &format!("layer_{k}.weight"));
            let b_name = prefixed(prefix, &format!("layer_{k}.bias"));
            let ws_name = prefixed(prefix, &format!("layer_{k}.weight_scale"));
            let wzp_name = prefixed(prefix, &format!("layer_{k}.weight_zero_point"));
            let is_name = prefixed(prefix, &format!("layer_{k}.input_scale"));
            let izp_name = prefixed(prefix, &format!("layer_{k}.input_zero_point"));

            let (w_data, w_shape) = extract_i8_2d(&st, &w_name)?;
            let b_data = extract_f32_1d(&st, &b_name)?;
            let ws = extract_f32_scalar(&st, &ws_name)?;
            let wzp = extract_i8_scalar(&st, &wzp_name)?;
            let is_val = extract_f32_scalar(&st, &is_name)?;
            let izp = extract_i8_scalar(&st, &izp_name)?;

            let out_size = w_shape[0];
            let in_size = w_shape[1];

            if b_data.len() != out_size {
                return Err(LoadError::Validation("bias length must match output size"));
            }
            if let Some(prev) = prev_out
                && in_size != prev
            {
                return Err(LoadError::Validation(
                    "layer input size must match previous layer output size",
                ));
            }
            prev_out = Some(out_size);

            layers_w.push(w_data);
            layers_b.push(b_data);
            w_scales.push(ws);
            w_zero_points.push(wzp);
            input_scales.push(is_val);
            input_zero_points.push(izp);
        }

        let w_refs: Vec<&[i8]> = layers_w.iter().map(Vec::as_slice).collect();
        let b_refs: Vec<&[f32]> = layers_b.iter().map(Vec::as_slice).collect();

        Self::from_parts(
            &w_refs,
            &b_refs,
            &w_scales,
            &w_zero_points,
            &input_scales,
            &input_zero_points,
            activation,
        )
    }
}

#[cfg(test)]
mod tests {
    use safetensors::Dtype;

    use crate::LoadError;

    fn f32_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn make_view<'a>(
        dtype: Dtype,
        shape: &[usize],
        bytes: &'a [u8],
    ) -> safetensors::tensor::TensorView<'a> {
        safetensors::tensor::TensorView::new(dtype, shape.to_vec(), bytes).unwrap()
    }

    fn serialize_tensors(tensors: Vec<(&str, safetensors::tensor::TensorView<'_>)>) -> Vec<u8> {
        safetensors::tensor::serialize(tensors, None).unwrap()
    }

    // ---- LSTM ----

    #[test]
    fn lstm_from_safetensors() {
        let i = 4_usize;
        let h = 8_usize;
        let o = 2_usize;
        let gc = 4 * h;

        let wih = vec![0.1_f32; gc * i];
        let whh = vec![0.1_f32; gc * h];
        let bih = vec![0.0_f32; gc];
        let bhh = vec![0.0_f32; gc];
        let wo = vec![0.1_f32; o * h];
        let bo = vec![0.0_f32; o];

        let wih_b = f32_bytes(&wih);
        let whh_b = f32_bytes(&whh);
        let bih_b = f32_bytes(&bih);
        let bhh_b = f32_bytes(&bhh);
        let wo_b = f32_bytes(&wo);
        let bo_b = f32_bytes(&bo);

        let data = serialize_tensors(vec![
            ("lstm.weight_ih_l0", make_view(Dtype::F32, &[gc, i], &wih_b)),
            ("lstm.weight_hh_l0", make_view(Dtype::F32, &[gc, h], &whh_b)),
            ("lstm.bias_ih_l0", make_view(Dtype::F32, &[gc], &bih_b)),
            ("lstm.bias_hh_l0", make_view(Dtype::F32, &[gc], &bhh_b)),
            ("fc.weight", make_view(Dtype::F32, &[o, h], &wo_b)),
            ("fc.bias", make_view(Dtype::F32, &[o], &bo_b)),
        ]);

        let lstm = crate::TinyLstm::from_safetensors(&data, "lstm", "fc").unwrap();
        assert_eq!(lstm.n_inputs(), i);
        assert_eq!(lstm.n_hidden(), h);
        assert_eq!(lstm.n_outputs(), o);
    }

    #[test]
    fn lstm_matches_from_parts() {
        let i = 2_usize;
        let h = 4_usize;
        let o = 1_usize;
        let gc = 4 * h;

        let wih = vec![0.1_f32; gc * i];
        let whh = vec![0.05_f32; gc * h];
        let bih = vec![0.01_f32; gc];
        let bhh = vec![-0.01_f32; gc];
        let wo = vec![0.2_f32; o * h];
        let bo = vec![0.1_f32; o];

        let mut reference =
            crate::TinyLstm::from_parts(i, h, o, &wih, &whh, &bih, &bhh, &wo, &bo).unwrap();

        let wih_b = f32_bytes(&wih);
        let whh_b = f32_bytes(&whh);
        let bih_b = f32_bytes(&bih);
        let bhh_b = f32_bytes(&bhh);
        let wo_b = f32_bytes(&wo);
        let bo_b = f32_bytes(&bo);

        let data = serialize_tensors(vec![
            ("rnn.weight_ih_l0", make_view(Dtype::F32, &[gc, i], &wih_b)),
            ("rnn.weight_hh_l0", make_view(Dtype::F32, &[gc, h], &whh_b)),
            ("rnn.bias_ih_l0", make_view(Dtype::F32, &[gc], &bih_b)),
            ("rnn.bias_hh_l0", make_view(Dtype::F32, &[gc], &bhh_b)),
            ("out.weight", make_view(Dtype::F32, &[o, h], &wo_b)),
            ("out.bias", make_view(Dtype::F32, &[o], &bo_b)),
        ]);

        let mut loaded = crate::TinyLstm::from_safetensors(&data, "rnn", "out").unwrap();

        let input = [0.5_f32, -0.3];
        let ref_out = reference.predict(&input);
        let load_out = loaded.predict(&input);
        assert!(
            (ref_out - load_out).abs() < 1e-7,
            "ref={ref_out}, loaded={load_out}"
        );
    }

    // ---- GRU ----

    #[test]
    fn gru_from_safetensors() {
        let i = 4_usize;
        let h = 8_usize;
        let o = 1_usize;
        let gc = 3 * h;

        let wih = vec![0.1_f32; gc * i];
        let whh = vec![0.1_f32; gc * h];
        let bih = vec![0.0_f32; gc];
        let bhh = vec![0.0_f32; gc];
        let wo = vec![0.1_f32; o * h];
        let bo = vec![0.0_f32; o];

        let wih_b = f32_bytes(&wih);
        let whh_b = f32_bytes(&whh);
        let bih_b = f32_bytes(&bih);
        let bhh_b = f32_bytes(&bhh);
        let wo_b = f32_bytes(&wo);
        let bo_b = f32_bytes(&bo);

        let data = serialize_tensors(vec![
            ("gru.weight_ih_l0", make_view(Dtype::F32, &[gc, i], &wih_b)),
            ("gru.weight_hh_l0", make_view(Dtype::F32, &[gc, h], &whh_b)),
            ("gru.bias_ih_l0", make_view(Dtype::F32, &[gc], &bih_b)),
            ("gru.bias_hh_l0", make_view(Dtype::F32, &[gc], &bhh_b)),
            ("fc.weight", make_view(Dtype::F32, &[o, h], &wo_b)),
            ("fc.bias", make_view(Dtype::F32, &[o], &bo_b)),
        ]);

        let gru = crate::TinyGru::from_safetensors(&data, "gru", "fc").unwrap();
        assert_eq!(gru.n_inputs(), i);
        assert_eq!(gru.n_hidden(), h);
        assert_eq!(gru.n_outputs(), o);
    }

    #[test]
    fn gru_matches_from_parts() {
        let i = 2_usize;
        let h = 4_usize;
        let o = 1_usize;
        let gc = 3 * h;

        let wih = vec![0.1_f32; gc * i];
        let whh = vec![0.05_f32; gc * h];
        let bih = vec![0.01_f32; gc];
        let bhh = vec![-0.01_f32; gc];
        let wo = vec![0.2_f32; o * h];
        let bo = vec![0.1_f32; o];

        let mut reference =
            crate::TinyGru::from_parts(i, h, o, &wih, &whh, &bih, &bhh, &wo, &bo).unwrap();

        let wih_b = f32_bytes(&wih);
        let whh_b = f32_bytes(&whh);
        let bih_b = f32_bytes(&bih);
        let bhh_b = f32_bytes(&bhh);
        let wo_b = f32_bytes(&wo);
        let bo_b = f32_bytes(&bo);

        let data = serialize_tensors(vec![
            ("gru.weight_ih_l0", make_view(Dtype::F32, &[gc, i], &wih_b)),
            ("gru.weight_hh_l0", make_view(Dtype::F32, &[gc, h], &whh_b)),
            ("gru.bias_ih_l0", make_view(Dtype::F32, &[gc], &bih_b)),
            ("gru.bias_hh_l0", make_view(Dtype::F32, &[gc], &bhh_b)),
            ("fc.weight", make_view(Dtype::F32, &[o, h], &wo_b)),
            ("fc.bias", make_view(Dtype::F32, &[o], &bo_b)),
        ]);

        let mut loaded = crate::TinyGru::from_safetensors(&data, "gru", "fc").unwrap();

        let input = [0.5_f32, -0.3];
        let ref_out = reference.predict(&input);
        let load_out = loaded.predict(&input);
        assert!(
            (ref_out - load_out).abs() < 1e-7,
            "ref={ref_out}, loaded={load_out}"
        );
    }

    // ---- MLP ----

    #[test]
    fn mlp_f32_from_safetensors() {
        // 2 inputs → 4 hidden (relu) → 1 output
        // Sequential indices: 0 (Linear), 1 (ReLU — no params), 2 (Linear)
        let w0 = vec![0.1_f32; 4 * 2]; // (4, 2)
        let b0 = vec![0.0_f32; 4];
        let w1 = vec![0.1_f32; 1 * 4]; // (1, 4)
        let b1 = vec![0.0_f32; 1];

        let w0_b = f32_bytes(&w0);
        let b0_b = f32_bytes(&b0);
        let w1_b = f32_bytes(&w1);
        let b1_b = f32_bytes(&b1);

        let data = serialize_tensors(vec![
            ("fc.0.weight", make_view(Dtype::F32, &[4, 2], &w0_b)),
            ("fc.0.bias", make_view(Dtype::F32, &[4], &b0_b)),
            ("fc.2.weight", make_view(Dtype::F32, &[1, 4], &w1_b)),
            ("fc.2.bias", make_view(Dtype::F32, &[1], &b1_b)),
        ]);

        let mlp = crate::Mlp::from_safetensors(&data, "fc", crate::Activation::Relu).unwrap();
        assert_eq!(mlp.n_inputs(), 2);
        assert_eq!(mlp.n_outputs(), 1);
        assert_eq!(mlp.n_layers(), 2);
    }

    #[test]
    fn mlp_f32_matches_from_parts() {
        let w0: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0];
        let b0: Vec<f32> = vec![0.0; 4];
        let w1: Vec<f32> = vec![1.0, 1.0, 0.0, 0.0];
        let b1: Vec<f32> = vec![0.0];

        let reference = crate::Mlp::from_parts(
            &[2, 4, 1],
            &[w0.as_slice(), w1.as_slice()].concat(),
            &[b0.as_slice(), b1.as_slice()].concat(),
            crate::Activation::Relu,
        )
        .unwrap();

        let w0_b = f32_bytes(&w0);
        let b0_b = f32_bytes(&b0);
        let w1_b = f32_bytes(&w1);
        let b1_b = f32_bytes(&b1);

        let data = serialize_tensors(vec![
            ("0.weight", make_view(Dtype::F32, &[4, 2], &w0_b)),
            ("0.bias", make_view(Dtype::F32, &[4], &b0_b)),
            ("1.weight", make_view(Dtype::F32, &[1, 4], &w1_b)),
            ("1.bias", make_view(Dtype::F32, &[1], &b1_b)),
        ]);

        let loaded = crate::Mlp::from_safetensors(&data, "", crate::Activation::Relu).unwrap();

        let input = [3.0_f32, 4.0];
        let ref_out = reference.predict(&input);
        let load_out = loaded.predict(&input);
        assert!(
            (ref_out - load_out).abs() < 1e-6,
            "ref={ref_out}, loaded={load_out}"
        );
    }

    // ---- BatchNorm fusion ----

    #[test]
    fn mlp_f32_batchnorm_fusion() {
        // 2 → 4 (Linear+BN) → ReLU → 1 (Linear)
        // Sequential: 0=Linear, 1=BatchNorm, 2=ReLU, 3=Linear
        let w0: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5]; // (4, 2)
        let b0: Vec<f32> = vec![0.0; 4];
        let bn_gamma: Vec<f32> = vec![2.0; 4];
        let bn_beta: Vec<f32> = vec![0.1; 4];
        let bn_mean: Vec<f32> = vec![1.0; 4];
        let bn_var: Vec<f32> = vec![4.0; 4];
        let w1: Vec<f32> = vec![1.0; 4]; // (1, 4)
        let b1: Vec<f32> = vec![0.0];

        let w0_b = f32_bytes(&w0);
        let b0_b = f32_bytes(&b0);
        let g_b = f32_bytes(&bn_gamma);
        let beta_b = f32_bytes(&bn_beta);
        let mean_b = f32_bytes(&bn_mean);
        let var_b = f32_bytes(&bn_var);
        let w1_b = f32_bytes(&w1);
        let b1_b = f32_bytes(&b1);

        let data = serialize_tensors(vec![
            ("fc.0.weight", make_view(Dtype::F32, &[4, 2], &w0_b)),
            ("fc.0.bias", make_view(Dtype::F32, &[4], &b0_b)),
            ("fc.1.weight", make_view(Dtype::F32, &[4], &g_b)),
            ("fc.1.bias", make_view(Dtype::F32, &[4], &beta_b)),
            ("fc.1.running_mean", make_view(Dtype::F32, &[4], &mean_b)),
            ("fc.1.running_var", make_view(Dtype::F32, &[4], &var_b)),
            ("fc.3.weight", make_view(Dtype::F32, &[1, 4], &w1_b)),
            ("fc.3.bias", make_view(Dtype::F32, &[1], &b1_b)),
        ]);

        let mlp = crate::Mlp::from_safetensors(&data, "fc", crate::Activation::Relu)
            .expect("should load with BN fusion");

        // Verify: input [3, 5]
        // W0 rows: [1,0], [0,1], [0.5,0.5], [-0.5,0.5]
        // Linear: W0 @ [3,5] + b0 = [3, 5, 4, 1]
        // BN: scale = gamma/sqrt(var+eps) = 2/sqrt(4+1e-5) ≈ 1.0
        //     y = scale*(x - mean) + beta ≈ [2.1, 4.1, 3.1, 0.1]
        // ReLU: [2.1, 4.1, 3.1, 0.1]
        // Linear: sum ≈ 9.4
        let input = [3.0_f32, 5.0];
        let out = mlp.predict(&input);

        let eps = 1e-5_f64;
        let scale = 2.0_f64 / (4.0_f64 + eps).sqrt();
        let linear_out = [3.0_f64, 5.0, 4.0, 1.0];
        let bn_out: Vec<f64> = linear_out
            .iter()
            .map(|&x| scale * (x - 1.0) + 0.1)
            .collect();
        let relu_out: Vec<f64> = bn_out.iter().map(|&x| x.max(0.0)).collect();
        let expected: f64 = relu_out.iter().sum();

        assert!(
            (out as f64 - expected).abs() < 1e-4,
            "got {out}, expected {expected}"
        );
    }

    #[test]
    fn mlp_f32_batchnorm_affine_false() {
        // BatchNorm with affine=False — no gamma/beta tensors
        let w0: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0]; // (2, 2) identity
        let b0: Vec<f32> = vec![0.0; 2];
        let bn_mean: Vec<f32> = vec![1.0, 2.0];
        let bn_var: Vec<f32> = vec![1.0, 1.0];
        let w1: Vec<f32> = vec![1.0, 1.0]; // (1, 2) sum
        let b1: Vec<f32> = vec![0.0];

        let w0_b = f32_bytes(&w0);
        let b0_b = f32_bytes(&b0);
        let mean_b = f32_bytes(&bn_mean);
        let var_b = f32_bytes(&bn_var);
        let w1_b = f32_bytes(&w1);
        let b1_b = f32_bytes(&b1);

        let data = serialize_tensors(vec![
            ("0.weight", make_view(Dtype::F32, &[2, 2], &w0_b)),
            ("0.bias", make_view(Dtype::F32, &[2], &b0_b)),
            ("1.running_mean", make_view(Dtype::F32, &[2], &mean_b)),
            ("1.running_var", make_view(Dtype::F32, &[2], &var_b)),
            ("2.weight", make_view(Dtype::F32, &[1, 2], &w1_b)),
            ("2.bias", make_view(Dtype::F32, &[1], &b1_b)),
        ]);

        let mlp = crate::Mlp::from_safetensors(&data, "", crate::Activation::Relu).unwrap();

        // Input [3, 4]: Linear=[3,4], BN with gamma=1,beta=0:
        //   scale = 1/sqrt(1+1e-5) ≈ 1.0
        //   y = [3-1, 4-2] = [2, 2]
        // ReLU: [2, 2], sum=4
        let out = mlp.predict(&[3.0, 4.0]);

        let eps = 1e-5_f64;
        let s = 1.0 / (1.0_f64 + eps).sqrt();
        let expected = s * (3.0 - 1.0) + s * (4.0 - 2.0);
        assert!(
            (out as f64 - expected).abs() < 1e-4,
            "got {out}, expected {expected}"
        );
    }

    // ---- Conv1d ----

    #[test]
    fn conv_from_safetensors() {
        // 1 channel, kernel 3, 2 filters, 1 output, identity
        let w_conv = vec![0.1_f32; 2 * 1 * 3]; // PyTorch: (F=2, C=1, K=3)
        let b_conv = vec![0.0_f32; 2];
        let w_out = vec![0.1_f32; 1 * 2]; // (O=1, F=2)
        let b_out = vec![0.0_f32; 1];

        let wc_b = f32_bytes(&w_conv);
        let bc_b = f32_bytes(&b_conv);
        let wo_b = f32_bytes(&w_out);
        let bo_b = f32_bytes(&b_out);

        let data = serialize_tensors(vec![
            ("conv.weight", make_view(Dtype::F32, &[2, 1, 3], &wc_b)),
            ("conv.bias", make_view(Dtype::F32, &[2], &bc_b)),
            ("fc.weight", make_view(Dtype::F32, &[1, 2], &wo_b)),
            ("fc.bias", make_view(Dtype::F32, &[1], &bo_b)),
        ]);

        let conv =
            crate::Causal1dConv::from_safetensors(&data, "conv", "fc", crate::Activation::Identity)
                .unwrap();
        assert_eq!(conv.n_inputs(), 1);
        assert_eq!(conv.kernel_size(), 3);
        assert_eq!(conv.n_filters(), 2);
        assert_eq!(conv.n_outputs(), 1);
    }

    #[test]
    fn conv_transpose_and_reverse() {
        // Verify PyTorch (F, C, K) → our (F, K, C) transpose+reverse
        // 2 channels, kernel 2, 1 filter
        //
        // PyTorch layout (1, 2, 2): [w(f0,c0,k_pt=0), w(f0,c0,k_pt=1), w(f0,c1,k_pt=0), w(f0,c1,k_pt=1)]
        //                         = [0.1,              0.2,              0.3,              0.4]
        // PyTorch: k_pt=0 multiplies oldest input, k_pt=1 multiplies newest.
        //
        // Our layout (1, 2, 2): k=0 is newest (current), k=1 is oldest.
        //   our[k=0,c=0] = pt[c=0,k_pt=1] = 0.2
        //   our[k=0,c=1] = pt[c=1,k_pt=1] = 0.4
        //   our[k=1,c=0] = pt[c=0,k_pt=0] = 0.1
        //   our[k=1,c=1] = pt[c=1,k_pt=0] = 0.3
        // Our layout = [0.2, 0.4, 0.1, 0.3]
        let w_conv_pt = [0.1_f32, 0.2, 0.3, 0.4];
        let b_conv = [0.0_f32];
        let w_out = [1.0_f32];
        let b_out = [0.0_f32];

        let wc_b = f32_bytes(&w_conv_pt);
        let bc_b = f32_bytes(&b_conv);
        let wo_b = f32_bytes(&w_out);
        let bo_b = f32_bytes(&b_out);

        let data = serialize_tensors(vec![
            ("conv.weight", make_view(Dtype::F32, &[1, 2, 2], &wc_b)),
            ("conv.bias", make_view(Dtype::F32, &[1], &bc_b)),
            ("fc.weight", make_view(Dtype::F32, &[1, 1], &wo_b)),
            ("fc.bias", make_view(Dtype::F32, &[1], &bo_b)),
        ]);

        let mut conv =
            crate::Causal1dConv::from_safetensors(&data, "conv", "fc", crate::Activation::Identity)
                .unwrap();

        let w_conv_ours = [0.2_f32, 0.4, 0.1, 0.3];
        let mut reference = crate::Causal1dConv::from_parts(
            2,
            2,
            1,
            1,
            &w_conv_ours,
            &b_conv,
            &w_out,
            &b_out,
            crate::Activation::Identity,
        )
        .unwrap();

        let input = [1.0_f32, 2.0];
        let ref_out = reference.predict(&input);
        let load_out = conv.predict(&input);
        assert!(
            (ref_out - load_out).abs() < 1e-6,
            "ref={ref_out}, loaded={load_out}"
        );

        let input2 = [3.0_f32, 4.0];
        let ref_out2 = reference.predict(&input2);
        let load_out2 = conv.predict(&input2);
        assert!(
            (ref_out2 - load_out2).abs() < 1e-6,
            "ref={ref_out2}, loaded={load_out2}"
        );
    }

    // ---- error cases ----

    #[test]
    fn missing_tensor() {
        let w = vec![0.1_f32; 4];
        let w_b = f32_bytes(&w);
        let data = serialize_tensors(vec![(
            "wrong.weight_ih_l0",
            make_view(Dtype::F32, &[4, 1], &w_b),
        )]);

        let err = crate::Mlp::from_safetensors(&data, "fc", crate::Activation::Relu);
        assert!(matches!(err, Err(LoadError::Parse(_))));
    }

    #[test]
    fn lstm_missing_tensor() {
        let w = vec![0.1_f32; 8];
        let w_b = f32_bytes(&w);
        let data = serialize_tensors(vec![(
            "lstm.weight_ih_l0",
            make_view(Dtype::F32, &[4, 2], &w_b),
        )]);

        let err = crate::TinyLstm::from_safetensors(&data, "lstm", "fc");
        match err {
            Err(LoadError::TensorNotFound(name)) => {
                assert_eq!(name, "lstm.weight_hh_l0".to_string());
            }
            other => panic!("expected TensorNotFound, got {other:?}"),
        }
    }

    #[test]
    fn invalid_safetensors() {
        let err =
            crate::Mlp::from_safetensors(b"not valid safetensors", "fc", crate::Activation::Relu);
        assert!(matches!(err, Err(LoadError::Parse(_))));
    }

    // ---- Stacked LSTM ----

    #[test]
    fn stacked_lstm_from_safetensors() {
        let i = 4_usize;
        let h = 8_usize;
        let o = 2_usize;
        let gc = 4 * h;

        let wih_l0 = vec![0.1_f32; gc * i];
        let whh_l0 = vec![0.1_f32; gc * h];
        let wih_l1 = vec![0.1_f32; gc * h];
        let whh_l1 = vec![0.1_f32; gc * h];
        let bih = vec![0.0_f32; gc];
        let bhh = vec![0.0_f32; gc];
        let wo = vec![0.1_f32; o * h];
        let bo = vec![0.0_f32; o];

        let wih_l0_b = f32_bytes(&wih_l0);
        let whh_l0_b = f32_bytes(&whh_l0);
        let wih_l1_b = f32_bytes(&wih_l1);
        let whh_l1_b = f32_bytes(&whh_l1);
        let bih_b = f32_bytes(&bih);
        let bhh_b = f32_bytes(&bhh);
        let wo_b = f32_bytes(&wo);
        let bo_b = f32_bytes(&bo);

        let data = serialize_tensors(vec![
            (
                "lstm.weight_ih_l0",
                make_view(Dtype::F32, &[gc, i], &wih_l0_b),
            ),
            (
                "lstm.weight_hh_l0",
                make_view(Dtype::F32, &[gc, h], &whh_l0_b),
            ),
            ("lstm.bias_ih_l0", make_view(Dtype::F32, &[gc], &bih_b)),
            ("lstm.bias_hh_l0", make_view(Dtype::F32, &[gc], &bhh_b)),
            (
                "lstm.weight_ih_l1",
                make_view(Dtype::F32, &[gc, h], &wih_l1_b),
            ),
            (
                "lstm.weight_hh_l1",
                make_view(Dtype::F32, &[gc, h], &whh_l1_b),
            ),
            ("lstm.bias_ih_l1", make_view(Dtype::F32, &[gc], &bih_b)),
            ("lstm.bias_hh_l1", make_view(Dtype::F32, &[gc], &bhh_b)),
            ("fc.weight", make_view(Dtype::F32, &[o, h], &wo_b)),
            ("fc.bias", make_view(Dtype::F32, &[o], &bo_b)),
        ]);

        let lstm = crate::StackedLstm::from_safetensors(&data, "lstm", "fc").unwrap();
        assert_eq!(lstm.n_inputs(), i);
        assert_eq!(lstm.n_hidden(), h);
        assert_eq!(lstm.n_outputs(), o);
        assert_eq!(lstm.n_layers(), 2);
    }

    #[test]
    fn stacked_lstm_matches_from_parts() {
        let i = 2_usize;
        let h = 4_usize;
        let o = 1_usize;
        let gc = 4 * h;

        let wih_l0 = vec![0.1_f32; gc * i];
        let whh_l0 = vec![0.05_f32; gc * h];
        let wih_l1 = vec![0.08_f32; gc * h];
        let whh_l1 = vec![0.03_f32; gc * h];
        let bih_l0 = vec![0.01_f32; gc];
        let bhh_l0 = vec![-0.01_f32; gc];
        let bih_l1 = vec![0.02_f32; gc];
        let bhh_l1 = vec![-0.02_f32; gc];
        let wo = vec![0.2_f32; o * h];
        let bo = vec![0.1_f32; o];

        let mut reference = crate::StackedLstm::from_parts(
            i,
            h,
            o,
            &[&wih_l0, &wih_l1],
            &[&whh_l0, &whh_l1],
            &[&bih_l0, &bih_l1],
            &[&bhh_l0, &bhh_l1],
            &wo,
            &bo,
        )
        .unwrap();

        let wih_l0_b = f32_bytes(&wih_l0);
        let whh_l0_b = f32_bytes(&whh_l0);
        let wih_l1_b = f32_bytes(&wih_l1);
        let whh_l1_b = f32_bytes(&whh_l1);
        let bih_l0_b = f32_bytes(&bih_l0);
        let bhh_l0_b = f32_bytes(&bhh_l0);
        let bih_l1_b = f32_bytes(&bih_l1);
        let bhh_l1_b = f32_bytes(&bhh_l1);
        let wo_b = f32_bytes(&wo);
        let bo_b = f32_bytes(&bo);

        let data = serialize_tensors(vec![
            (
                "rnn.weight_ih_l0",
                make_view(Dtype::F32, &[gc, i], &wih_l0_b),
            ),
            (
                "rnn.weight_hh_l0",
                make_view(Dtype::F32, &[gc, h], &whh_l0_b),
            ),
            ("rnn.bias_ih_l0", make_view(Dtype::F32, &[gc], &bih_l0_b)),
            ("rnn.bias_hh_l0", make_view(Dtype::F32, &[gc], &bhh_l0_b)),
            (
                "rnn.weight_ih_l1",
                make_view(Dtype::F32, &[gc, h], &wih_l1_b),
            ),
            (
                "rnn.weight_hh_l1",
                make_view(Dtype::F32, &[gc, h], &whh_l1_b),
            ),
            ("rnn.bias_ih_l1", make_view(Dtype::F32, &[gc], &bih_l1_b)),
            ("rnn.bias_hh_l1", make_view(Dtype::F32, &[gc], &bhh_l1_b)),
            ("out.weight", make_view(Dtype::F32, &[o, h], &wo_b)),
            ("out.bias", make_view(Dtype::F32, &[o], &bo_b)),
        ]);

        let mut loaded = crate::StackedLstm::from_safetensors(&data, "rnn", "out").unwrap();

        let input = [0.5_f32, -0.3];
        let ref_out = reference.predict(&input);
        let load_out = loaded.predict(&input);
        assert!(
            (ref_out - load_out).abs() < 1e-7,
            "ref={ref_out}, loaded={load_out}"
        );
    }

    #[test]
    fn stacked_lstm_single_layer_auto_detect() {
        let i = 4_usize;
        let h = 8_usize;
        let o = 1_usize;
        let gc = 4 * h;

        let wih = vec![0.1_f32; gc * i];
        let whh = vec![0.1_f32; gc * h];
        let bih = vec![0.0_f32; gc];
        let bhh = vec![0.0_f32; gc];
        let wo = vec![0.1_f32; o * h];
        let bo = vec![0.0_f32; o];

        let wih_b = f32_bytes(&wih);
        let whh_b = f32_bytes(&whh);
        let bih_b = f32_bytes(&bih);
        let bhh_b = f32_bytes(&bhh);
        let wo_b = f32_bytes(&wo);
        let bo_b = f32_bytes(&bo);

        let data = serialize_tensors(vec![
            ("lstm.weight_ih_l0", make_view(Dtype::F32, &[gc, i], &wih_b)),
            ("lstm.weight_hh_l0", make_view(Dtype::F32, &[gc, h], &whh_b)),
            ("lstm.bias_ih_l0", make_view(Dtype::F32, &[gc], &bih_b)),
            ("lstm.bias_hh_l0", make_view(Dtype::F32, &[gc], &bhh_b)),
            ("fc.weight", make_view(Dtype::F32, &[o, h], &wo_b)),
            ("fc.bias", make_view(Dtype::F32, &[o], &bo_b)),
        ]);

        let lstm = crate::StackedLstm::from_safetensors(&data, "lstm", "fc").unwrap();
        assert_eq!(lstm.n_layers(), 1);
    }

    #[test]
    fn stacked_lstm_rejects_non_consecutive_layers() {
        // l0 and l2 present, l1 missing. A gap means a malformed file or
        // wrong prefix; must error rather than silently load 1 layer.
        let i = 4_usize;
        let h = 8_usize;
        let o = 1_usize;
        let gc = 4 * h;

        let wih_l0 = vec![0.1_f32; gc * i];
        let whh = vec![0.1_f32; gc * h];
        let bih = vec![0.0_f32; gc];
        let bhh = vec![0.0_f32; gc];
        let wo = vec![0.1_f32; o * h];
        let bo = vec![0.0_f32; o];

        let wih_l0_b = f32_bytes(&wih_l0);
        let whh_b = f32_bytes(&whh);
        let bih_b = f32_bytes(&bih);
        let bhh_b = f32_bytes(&bhh);
        let wo_b = f32_bytes(&wo);
        let bo_b = f32_bytes(&bo);

        let data = serialize_tensors(vec![
            (
                "lstm.weight_ih_l0",
                make_view(Dtype::F32, &[gc, i], &wih_l0_b),
            ),
            ("lstm.weight_hh_l0", make_view(Dtype::F32, &[gc, h], &whh_b)),
            ("lstm.bias_ih_l0", make_view(Dtype::F32, &[gc], &bih_b)),
            ("lstm.bias_hh_l0", make_view(Dtype::F32, &[gc], &bhh_b)),
            ("lstm.weight_ih_l2", make_view(Dtype::F32, &[gc, h], &whh_b)),
            ("lstm.weight_hh_l2", make_view(Dtype::F32, &[gc, h], &whh_b)),
            ("lstm.bias_ih_l2", make_view(Dtype::F32, &[gc], &bih_b)),
            ("lstm.bias_hh_l2", make_view(Dtype::F32, &[gc], &bhh_b)),
            ("fc.weight", make_view(Dtype::F32, &[o, h], &wo_b)),
            ("fc.bias", make_view(Dtype::F32, &[o], &bo_b)),
        ]);

        let err = crate::StackedLstm::from_safetensors(&data, "lstm", "fc");
        assert!(matches!(err, Err(LoadError::Validation(_))), "got {err:?}");
    }

    // ---- Stacked GRU ----

    #[test]
    fn stacked_gru_from_safetensors() {
        let i = 4_usize;
        let h = 8_usize;
        let o = 1_usize;
        let gc = 3 * h;

        let wih_l0 = vec![0.1_f32; gc * i];
        let whh_l0 = vec![0.1_f32; gc * h];
        let wih_l1 = vec![0.1_f32; gc * h];
        let whh_l1 = vec![0.1_f32; gc * h];
        let bih = vec![0.0_f32; gc];
        let bhh = vec![0.0_f32; gc];
        let wo = vec![0.1_f32; o * h];
        let bo = vec![0.0_f32; o];

        let wih_l0_b = f32_bytes(&wih_l0);
        let whh_l0_b = f32_bytes(&whh_l0);
        let wih_l1_b = f32_bytes(&wih_l1);
        let whh_l1_b = f32_bytes(&whh_l1);
        let bih_b = f32_bytes(&bih);
        let bhh_b = f32_bytes(&bhh);
        let wo_b = f32_bytes(&wo);
        let bo_b = f32_bytes(&bo);

        let data = serialize_tensors(vec![
            (
                "gru.weight_ih_l0",
                make_view(Dtype::F32, &[gc, i], &wih_l0_b),
            ),
            (
                "gru.weight_hh_l0",
                make_view(Dtype::F32, &[gc, h], &whh_l0_b),
            ),
            ("gru.bias_ih_l0", make_view(Dtype::F32, &[gc], &bih_b)),
            ("gru.bias_hh_l0", make_view(Dtype::F32, &[gc], &bhh_b)),
            (
                "gru.weight_ih_l1",
                make_view(Dtype::F32, &[gc, h], &wih_l1_b),
            ),
            (
                "gru.weight_hh_l1",
                make_view(Dtype::F32, &[gc, h], &whh_l1_b),
            ),
            ("gru.bias_ih_l1", make_view(Dtype::F32, &[gc], &bih_b)),
            ("gru.bias_hh_l1", make_view(Dtype::F32, &[gc], &bhh_b)),
            ("fc.weight", make_view(Dtype::F32, &[o, h], &wo_b)),
            ("fc.bias", make_view(Dtype::F32, &[o], &bo_b)),
        ]);

        let gru = crate::StackedGru::from_safetensors(&data, "gru", "fc").unwrap();
        assert_eq!(gru.n_inputs(), i);
        assert_eq!(gru.n_hidden(), h);
        assert_eq!(gru.n_outputs(), o);
        assert_eq!(gru.n_layers(), 2);
    }

    #[test]
    fn stacked_gru_matches_from_parts() {
        let i = 2_usize;
        let h = 4_usize;
        let o = 1_usize;
        let gc = 3 * h;

        let wih_l0 = vec![0.1_f32; gc * i];
        let whh_l0 = vec![0.05_f32; gc * h];
        let wih_l1 = vec![0.08_f32; gc * h];
        let whh_l1 = vec![0.03_f32; gc * h];
        let bih_l0 = vec![0.01_f32; gc];
        let bhh_l0 = vec![-0.01_f32; gc];
        let bih_l1 = vec![0.02_f32; gc];
        let bhh_l1 = vec![-0.02_f32; gc];
        let wo = vec![0.2_f32; o * h];
        let bo = vec![0.1_f32; o];

        let mut reference = crate::StackedGru::from_parts(
            i,
            h,
            o,
            &[&wih_l0, &wih_l1],
            &[&whh_l0, &whh_l1],
            &[&bih_l0, &bih_l1],
            &[&bhh_l0, &bhh_l1],
            &wo,
            &bo,
        )
        .unwrap();

        let wih_l0_b = f32_bytes(&wih_l0);
        let whh_l0_b = f32_bytes(&whh_l0);
        let wih_l1_b = f32_bytes(&wih_l1);
        let whh_l1_b = f32_bytes(&whh_l1);
        let bih_l0_b = f32_bytes(&bih_l0);
        let bhh_l0_b = f32_bytes(&bhh_l0);
        let bih_l1_b = f32_bytes(&bih_l1);
        let bhh_l1_b = f32_bytes(&bhh_l1);
        let wo_b = f32_bytes(&wo);
        let bo_b = f32_bytes(&bo);

        let data = serialize_tensors(vec![
            (
                "gru.weight_ih_l0",
                make_view(Dtype::F32, &[gc, i], &wih_l0_b),
            ),
            (
                "gru.weight_hh_l0",
                make_view(Dtype::F32, &[gc, h], &whh_l0_b),
            ),
            ("gru.bias_ih_l0", make_view(Dtype::F32, &[gc], &bih_l0_b)),
            ("gru.bias_hh_l0", make_view(Dtype::F32, &[gc], &bhh_l0_b)),
            (
                "gru.weight_ih_l1",
                make_view(Dtype::F32, &[gc, h], &wih_l1_b),
            ),
            (
                "gru.weight_hh_l1",
                make_view(Dtype::F32, &[gc, h], &whh_l1_b),
            ),
            ("gru.bias_ih_l1", make_view(Dtype::F32, &[gc], &bih_l1_b)),
            ("gru.bias_hh_l1", make_view(Dtype::F32, &[gc], &bhh_l1_b)),
            ("fc.weight", make_view(Dtype::F32, &[o, h], &wo_b)),
            ("fc.bias", make_view(Dtype::F32, &[o], &bo_b)),
        ]);

        let mut loaded = crate::StackedGru::from_safetensors(&data, "gru", "fc").unwrap();

        let input = [0.5_f32, -0.3];
        let ref_out = reference.predict(&input);
        let load_out = loaded.predict(&input);
        assert!(
            (ref_out - load_out).abs() < 1e-7,
            "ref={ref_out}, loaded={load_out}"
        );
    }

    #[test]
    fn stacked_lstm_missing_l0() {
        let data = serialize_tensors(vec![]);
        let err = crate::StackedLstm::from_safetensors(&data, "lstm", "fc");
        assert!(matches!(err, Err(LoadError::TensorNotFound(_))));
    }

    // ---- SSM ----

    #[test]
    fn ssm_from_safetensors() {
        let h = 4_usize;
        let i = 2_usize;
        let o = 1_usize;

        let a = vec![0.9_f32; h];
        let b = vec![0.1_f32; h * i];
        let c = vec![0.1_f32; o * h];
        let d = vec![0.01_f32; o * i];

        let a_b = f32_bytes(&a);
        let b_b = f32_bytes(&b);
        let c_b = f32_bytes(&c);
        let d_b = f32_bytes(&d);

        let data = serialize_tensors(vec![
            ("ssm.a_diag", make_view(Dtype::F32, &[h], &a_b)),
            ("ssm.b", make_view(Dtype::F32, &[h, i], &b_b)),
            ("ssm.c", make_view(Dtype::F32, &[o, h], &c_b)),
            ("ssm.d", make_view(Dtype::F32, &[o, i], &d_b)),
        ]);

        let ssm = crate::LinearSsm::from_safetensors(&data, "ssm").unwrap();
        assert_eq!(ssm.n_inputs(), i);
        assert_eq!(ssm.n_hidden(), h);
        assert_eq!(ssm.n_outputs(), o);
    }

    #[test]
    fn ssm_missing_d_defaults_to_zeros() {
        let h = 2_usize;
        let i = 1_usize;
        let o = 1_usize;

        let a = vec![0.5_f32; h];
        let b = vec![1.0_f32; h * i];
        let c = vec![1.0_f32; o * h];

        let a_b = f32_bytes(&a);
        let b_b = f32_bytes(&b);
        let c_b = f32_bytes(&c);

        let data = serialize_tensors(vec![
            ("m.a_diag", make_view(Dtype::F32, &[h], &a_b)),
            ("m.b", make_view(Dtype::F32, &[h, i], &b_b)),
            ("m.c", make_view(Dtype::F32, &[o, h], &c_b)),
        ]);

        let mut ssm = crate::LinearSsm::from_safetensors(&data, "m").unwrap();
        // h = [0,0]*0.5 + [5,5] = [5, 5]; y = 1*5 + 1*5 + 0 = 10
        let y = ssm.predict(&[5.0]);
        assert!((y - 10.0).abs() < 1e-6);
    }

    #[test]
    fn ssm_matches_from_parts() {
        let a = [0.9_f32, 0.8];
        let b = [0.1_f32, 0.2, 0.3, 0.4];
        let c = [0.5_f32, 0.6];
        let d = [0.01_f32, 0.02];

        let a_b = f32_bytes(&a);
        let b_b = f32_bytes(&b);
        let c_b = f32_bytes(&c);
        let d_b = f32_bytes(&d);

        let data = serialize_tensors(vec![
            ("s.a_diag", make_view(Dtype::F32, &[2], &a_b)),
            ("s.b", make_view(Dtype::F32, &[2, 2], &b_b)),
            ("s.c", make_view(Dtype::F32, &[1, 2], &c_b)),
            ("s.d", make_view(Dtype::F32, &[1, 2], &d_b)),
        ]);

        let mut st = crate::LinearSsm::from_safetensors(&data, "s").unwrap();
        let mut fp = crate::LinearSsm::from_parts(&a, &b, &c, &d, 1).unwrap();

        let input = [1.0_f32, 2.0];
        let y_st = st.predict(&input);
        let y_fp = fp.predict(&input);
        assert!((y_st - y_fp).abs() < 1e-7, "step 1: st={y_st} fp={y_fp}");

        let y_st2 = st.predict(&input);
        let y_fp2 = fp.predict(&input);
        assert!(
            (y_st2 - y_fp2).abs() < 1e-7,
            "step 2: st={y_st2} fp={y_fp2}"
        );
    }

    // ---- BNN ----

    fn i8_bytes(data: &[i8]) -> Vec<u8> {
        data.iter().map(|&v| v as u8).collect()
    }

    #[test]
    fn bnn_from_safetensors() {
        let h = 64_usize;
        let i = 2_usize;
        let o = 1_usize;

        let w_in = vec![0.1_f32; h * i];
        let b_in = vec![0.0_f32; h];
        let w_out = vec![0.1_f32; o * h];
        let b_out = vec![0.0_f32; o];
        let bin_w = vec![1_i8; h * h];
        let bin_b = vec![0.0_f32; h];

        let w_in_b = f32_bytes(&w_in);
        let b_in_b = f32_bytes(&b_in);
        let w_out_b = f32_bytes(&w_out);
        let b_out_b = f32_bytes(&b_out);
        let bin_w_b = i8_bytes(&bin_w);
        let bin_b_b = f32_bytes(&bin_b);

        let data = serialize_tensors(vec![
            ("bnn.input_weight", make_view(Dtype::F32, &[h, i], &w_in_b)),
            ("bnn.input_bias", make_view(Dtype::F32, &[h], &b_in_b)),
            (
                "bnn.binary_weight_0",
                make_view(Dtype::I8, &[h, h], &bin_w_b),
            ),
            ("bnn.binary_bias_0", make_view(Dtype::F32, &[h], &bin_b_b)),
            (
                "bnn.output_weight",
                make_view(Dtype::F32, &[o, h], &w_out_b),
            ),
            ("bnn.output_bias", make_view(Dtype::F32, &[o], &b_out_b)),
        ]);

        let bnn = crate::Bnn::from_safetensors(&data, "bnn").unwrap();
        assert_eq!(bnn.n_inputs(), i);
        assert_eq!(bnn.n_hidden(), h);
        assert_eq!(bnn.n_outputs(), o);
        assert_eq!(bnn.n_layers(), 1);
    }

    #[test]
    fn bnn_no_binary_layers() {
        let h = 64_usize;
        let i = 4_usize;
        let o = 1_usize;

        let w_in = vec![0.1_f32; h * i];
        let b_in = vec![0.0_f32; h];
        let w_out = vec![0.1_f32; o * h];
        let b_out = vec![0.0_f32; o];

        let w_in_b = f32_bytes(&w_in);
        let b_in_b = f32_bytes(&b_in);
        let w_out_b = f32_bytes(&w_out);
        let b_out_b = f32_bytes(&b_out);

        let data = serialize_tensors(vec![
            ("net.input_weight", make_view(Dtype::F32, &[h, i], &w_in_b)),
            ("net.input_bias", make_view(Dtype::F32, &[h], &b_in_b)),
            (
                "net.output_weight",
                make_view(Dtype::F32, &[o, h], &w_out_b),
            ),
            ("net.output_bias", make_view(Dtype::F32, &[o], &b_out_b)),
        ]);

        let bnn = crate::Bnn::from_safetensors(&data, "net").unwrap();
        assert_eq!(bnn.n_layers(), 0);
        assert_eq!(bnn.n_inputs(), i);
    }

    #[test]
    fn bnn_rejects_non_consecutive_binary_layers() {
        // binary_weight_0 and _2 present, _1 missing. A gap means a
        // malformed file; must error rather than silently load 1 layer.
        let h = 64_usize;
        let i = 2_usize;
        let o = 1_usize;

        let w_in = vec![0.1_f32; h * i];
        let b_in = vec![0.0_f32; h];
        let w_out = vec![0.1_f32; o * h];
        let b_out = vec![0.0_f32; o];
        let bin_w = vec![1_i8; h * h];
        let bin_b = vec![0.0_f32; h];

        let w_in_b = f32_bytes(&w_in);
        let b_in_b = f32_bytes(&b_in);
        let w_out_b = f32_bytes(&w_out);
        let b_out_b = f32_bytes(&b_out);
        let bin_w_b = i8_bytes(&bin_w);
        let bin_b_b = f32_bytes(&bin_b);

        let data = serialize_tensors(vec![
            ("bnn.input_weight", make_view(Dtype::F32, &[h, i], &w_in_b)),
            ("bnn.input_bias", make_view(Dtype::F32, &[h], &b_in_b)),
            (
                "bnn.binary_weight_0",
                make_view(Dtype::I8, &[h, h], &bin_w_b),
            ),
            ("bnn.binary_bias_0", make_view(Dtype::F32, &[h], &bin_b_b)),
            (
                "bnn.binary_weight_2",
                make_view(Dtype::I8, &[h, h], &bin_w_b),
            ),
            ("bnn.binary_bias_2", make_view(Dtype::F32, &[h], &bin_b_b)),
            (
                "bnn.output_weight",
                make_view(Dtype::F32, &[o, h], &w_out_b),
            ),
            ("bnn.output_bias", make_view(Dtype::F32, &[o], &b_out_b)),
        ]);

        let err = crate::Bnn::from_safetensors(&data, "bnn");
        assert!(matches!(err, Err(LoadError::Validation(_))), "got {err:?}");
    }

    #[test]
    fn bnn_rejects_hidden_size_not_multiple_of_64() {
        // H=32 is not a multiple of 64. Must error before packing rather
        // than panic / corrupt (pack_i8_to_u64 indexes by H/64).
        let h = 32_usize;
        let i = 2_usize;
        let o = 1_usize;

        let w_in_b = f32_bytes(&vec![0.1_f32; h * i]);
        let b_in_b = f32_bytes(&vec![0.0_f32; h]);
        let w_out_b = f32_bytes(&vec![0.1_f32; o * h]);
        let b_out_b = f32_bytes(&vec![0.0_f32; o]);
        let bin_w_b = i8_bytes(&vec![1_i8; h * h]);
        let bin_b_b = f32_bytes(&vec![0.0_f32; h]);

        let data = serialize_tensors(vec![
            ("bnn.input_weight", make_view(Dtype::F32, &[h, i], &w_in_b)),
            ("bnn.input_bias", make_view(Dtype::F32, &[h], &b_in_b)),
            (
                "bnn.binary_weight_0",
                make_view(Dtype::I8, &[h, h], &bin_w_b),
            ),
            ("bnn.binary_bias_0", make_view(Dtype::F32, &[h], &bin_b_b)),
            (
                "bnn.output_weight",
                make_view(Dtype::F32, &[o, h], &w_out_b),
            ),
            ("bnn.output_bias", make_view(Dtype::F32, &[o], &b_out_b)),
        ]);

        let err = crate::Bnn::from_safetensors(&data, "bnn");
        assert!(matches!(err, Err(LoadError::Validation(_))), "got {err:?}");
    }

    #[test]
    fn bnn_rejects_non_pm1_binary_weights() {
        // A binary weight of 0 (neither -1 nor 1) signals a corrupted or
        // mis-exported model; reject rather than silently treat as -1.
        let h = 64_usize;
        let i = 2_usize;
        let o = 1_usize;

        let mut bin_w = vec![1_i8; h * h];
        bin_w[0] = 0; // not ±1

        let w_in_b = f32_bytes(&vec![0.1_f32; h * i]);
        let b_in_b = f32_bytes(&vec![0.0_f32; h]);
        let w_out_b = f32_bytes(&vec![0.1_f32; o * h]);
        let b_out_b = f32_bytes(&vec![0.0_f32; o]);
        let bin_w_b = i8_bytes(&bin_w);
        let bin_b_b = f32_bytes(&vec![0.0_f32; h]);

        let data = serialize_tensors(vec![
            ("bnn.input_weight", make_view(Dtype::F32, &[h, i], &w_in_b)),
            ("bnn.input_bias", make_view(Dtype::F32, &[h], &b_in_b)),
            (
                "bnn.binary_weight_0",
                make_view(Dtype::I8, &[h, h], &bin_w_b),
            ),
            ("bnn.binary_bias_0", make_view(Dtype::F32, &[h], &bin_b_b)),
            (
                "bnn.output_weight",
                make_view(Dtype::F32, &[o, h], &w_out_b),
            ),
            ("bnn.output_bias", make_view(Dtype::F32, &[o], &b_out_b)),
        ]);

        let err = crate::Bnn::from_safetensors(&data, "bnn");
        assert!(matches!(err, Err(LoadError::Validation(_))), "got {err:?}");
    }

    #[test]
    fn bnn_matches_from_parts() {
        let h = 64_usize;
        let i = 2_usize;
        let o = 1_usize;

        let w_in: Vec<f32> = (0..h * i)
            .map(|k| 0.1 * (k as f32 + 1.0) / (h * i) as f32)
            .collect();
        let b_in: Vec<f32> = (0..h).map(|k| 0.01 * k as f32).collect();
        let w_out: Vec<f32> = (0..o * h)
            .map(|k| -0.1 + 0.2 * k as f32 / (o * h) as f32)
            .collect();
        let b_out = vec![0.05_f32; o];

        // Binary weights: alternating +1/-1 pattern
        let bin_w_i8: Vec<i8> = (0..h * h)
            .map(|k| if k % 2 == 0 { 1 } else { -1 })
            .collect();
        let bin_b: Vec<f32> = (0..h).map(|k| 0.5 - 0.01 * k as f32).collect();

        // Pack i8 → u64 (same logic as loader)
        let wpr = h / 64;
        let mut bin_w_u64 = vec![0_u64; h * wpr];
        for r in 0..h {
            for c in 0..h {
                if bin_w_i8[r * h + c] == 1 {
                    bin_w_u64[r * wpr + c / 64] |= 1 << (c % 64);
                }
            }
        }

        let bw_refs: Vec<&[u64]> = vec![bin_w_u64.as_slice()];
        let bb_refs: Vec<&[f32]> = vec![bin_b.as_slice()];
        let fp =
            crate::Bnn::from_parts(&w_in, &b_in, &bw_refs, &bb_refs, &w_out, &b_out, o).unwrap();

        // Build safetensors
        let w_in_b = f32_bytes(&w_in);
        let b_in_b = f32_bytes(&b_in);
        let w_out_b = f32_bytes(&w_out);
        let b_out_b = f32_bytes(&b_out);
        let bin_w_b = i8_bytes(&bin_w_i8);
        let bin_b_b = f32_bytes(&bin_b);

        let data = serialize_tensors(vec![
            ("b.input_weight", make_view(Dtype::F32, &[h, i], &w_in_b)),
            ("b.input_bias", make_view(Dtype::F32, &[h], &b_in_b)),
            ("b.binary_weight_0", make_view(Dtype::I8, &[h, h], &bin_w_b)),
            ("b.binary_bias_0", make_view(Dtype::F32, &[h], &bin_b_b)),
            ("b.output_weight", make_view(Dtype::F32, &[o, h], &w_out_b)),
            ("b.output_bias", make_view(Dtype::F32, &[o], &b_out_b)),
        ]);

        let st = crate::Bnn::from_safetensors(&data, "b").unwrap();

        let input = [1.0_f32, -0.5];
        let y_st = st.predict(&input);
        let y_fp = fp.predict(&input);
        assert!((y_st - y_fp).abs() < 1e-7, "st={y_st} fp={y_fp}");
    }

    // ---- TCN ----

    #[test]
    fn tcn_rejects_orphan_bias() {
        // conv_0 complete + a stray conv_2.bias (no conv_2.weight, conv_1
        // missing). The orphan bias past the consecutive run must be
        // rejected, not silently ignored.
        let f = 4_usize;
        let ic = 2_usize;
        let ks = 3_usize;
        let o = 1_usize;

        let w0 = f32_bytes(&vec![0.1_f32; f * ic * ks]);
        let b0 = f32_bytes(&vec![0.0_f32; f]);
        let wo = f32_bytes(&vec![0.1_f32; o * f]);
        let bo = f32_bytes(&vec![0.0_f32; o]);
        let orphan_bias = f32_bytes(&vec![0.0_f32; f]);

        let data = serialize_tensors(vec![
            ("t.conv_0.weight", make_view(Dtype::F32, &[f, ic, ks], &w0)),
            ("t.conv_0.bias", make_view(Dtype::F32, &[f], &b0)),
            ("t.conv_2.bias", make_view(Dtype::F32, &[f], &orphan_bias)),
            ("t.output.weight", make_view(Dtype::F32, &[o, f], &wo)),
            ("t.output.bias", make_view(Dtype::F32, &[o], &bo)),
        ]);

        let err = crate::TinyTcn::from_safetensors(&data, "t", crate::Activation::Relu, false);
        assert!(matches!(err, Err(LoadError::Validation(_))), "got {err:?}");
    }

    // ---- Quantized MLP ----

    #[test]
    fn quantized_mlp_loads_single_layer() {
        let w: Vec<i8> = vec![10, 20, 30, 40, 50, 60];
        let b: Vec<f32> = vec![0.1, 0.2];
        let w_s: f32 = 0.5;
        let w_zp: i8 = 0;
        let i_s: f32 = 0.25;
        let i_zp: i8 = 0;

        let wb = i8_bytes(&w);
        let bb = f32_bytes(&b);
        let wsb = f32_bytes(&[w_s]);
        let wzb = i8_bytes(&[w_zp]);
        let isb = f32_bytes(&[i_s]);
        let izb = i8_bytes(&[i_zp]);

        let data = serialize_tensors(vec![
            ("q.layer_0.weight", make_view(Dtype::I8, &[2, 3], &wb)),
            ("q.layer_0.bias", make_view(Dtype::F32, &[2], &bb)),
            ("q.layer_0.weight_scale", make_view(Dtype::F32, &[1], &wsb)),
            (
                "q.layer_0.weight_zero_point",
                make_view(Dtype::I8, &[1], &wzb),
            ),
            ("q.layer_0.input_scale", make_view(Dtype::F32, &[1], &isb)),
            (
                "q.layer_0.input_zero_point",
                make_view(Dtype::I8, &[1], &izb),
            ),
        ]);

        let qmlp =
            crate::QuantizedMlp::from_safetensors(&data, "q", crate::Activation::Identity).unwrap();
        assert_eq!(qmlp.n_inputs(), 3);
        assert_eq!(qmlp.n_outputs(), 2);
    }

    #[test]
    fn quantized_mlp_rejects_orphan_layer() {
        let w: Vec<i8> = vec![10, 20, 30, 40, 50, 60];
        let b: Vec<f32> = vec![0.1, 0.2];
        let wb = i8_bytes(&w);
        let bb = f32_bytes(&b);
        let wsb = f32_bytes(&[0.5]);
        let wzb = i8_bytes(&[0]);
        let isb = f32_bytes(&[0.25]);
        let izb = i8_bytes(&[0]);

        let orphan_w = i8_bytes(&vec![1_i8; 4]);

        let data = serialize_tensors(vec![
            ("q.layer_0.weight", make_view(Dtype::I8, &[2, 3], &wb)),
            ("q.layer_0.bias", make_view(Dtype::F32, &[2], &bb)),
            ("q.layer_0.weight_scale", make_view(Dtype::F32, &[1], &wsb)),
            (
                "q.layer_0.weight_zero_point",
                make_view(Dtype::I8, &[1], &wzb),
            ),
            ("q.layer_0.input_scale", make_view(Dtype::F32, &[1], &isb)),
            (
                "q.layer_0.input_zero_point",
                make_view(Dtype::I8, &[1], &izb),
            ),
            ("q.layer_2.weight", make_view(Dtype::I8, &[2, 2], &orphan_w)),
        ]);

        let err = crate::QuantizedMlp::from_safetensors(&data, "q", crate::Activation::Relu);
        assert!(matches!(err, Err(LoadError::Validation(_))), "got {err:?}");
    }
}

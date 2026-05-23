extern crate alloc;

use alloc::{format, string::String, vec, vec::Vec};
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

fn extract_f64_1d(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f64>, LoadError> {
    let tv = st
        .tensor(name)
        .map_err(|_| LoadError::TensorNotFound(String::from(name)))?;
    if tv.dtype() != Dtype::F64 {
        return Err(LoadError::Validation("expected F64 tensor"));
    }
    if tv.shape().len() != 1 {
        return Err(LoadError::Validation("expected 1D tensor"));
    }
    let bytes = tv.data();
    if bytes.len() % 8 != 0 {
        return Err(LoadError::Parse("F64 tensor data not aligned"));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

fn extract_f64_2d(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<f64>, [usize; 2]), LoadError> {
    let tv = st
        .tensor(name)
        .map_err(|_| LoadError::TensorNotFound(String::from(name)))?;
    if tv.dtype() != Dtype::F64 {
        return Err(LoadError::Validation("expected F64 tensor"));
    }
    let shape = tv.shape();
    if shape.len() != 2 {
        return Err(LoadError::Validation("expected 2D tensor"));
    }
    let dims = [shape[0], shape[1]];
    let bytes = tv.data();
    if bytes.len() % 8 != 0 {
        return Err(LoadError::Parse("F64 tensor data not aligned"));
    }
    let data = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    Ok((data, dims))
}

// ---- sqrt helper for BatchNorm fusion ----

#[cfg(feature = "std")]
fn sqrt_f64(x: f64) -> f64 {
    x.sqrt()
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
fn sqrt_f64(x: f64) -> f64 {
    libm::sqrt(x)
}

// ---- RNN loaders (require tanh/sigmoid from std or libm) ----

#[cfg(any(feature = "std", feature = "libm"))]
impl crate::TinyLstmF32 {
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
    /// let lstm = TinyLstmF32::from_safetensors(&bytes, "encoder.lstm", "encoder.fc")?;
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

#[cfg(any(feature = "std", feature = "libm"))]
impl crate::TinyGruF32 {
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
    /// let gru = TinyGruF32::from_safetensors(&bytes, "gru", "fc")?;
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
            /// let mlp = MlpF32::from_safetensors(&bytes, "fc", Activation::Relu)?;
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
                        #[cfg(not(any(feature = "std", feature = "libm")))]
                        {
                            return Err(LoadError::Validation(
                                "BatchNorm fusion requires 'std' or 'libm' feature",
                            ));
                        }
                        #[cfg(any(feature = "std", feature = "libm"))]
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
                    }

                    // Detect LayerNorm for hidden layers
                    let is_last_layer = i == n_linear - 1;
                    if !is_last_layer {
                        if let Some(&ln_idx) = layernorm_indices
                            .iter()
                            .find(|&&li| li > idx && li < next_linear)
                        {
                            has_layernorm = true;
                            let ln_g =
                                $extract_1d(&st, &format!("{prefix_dot}{ln_idx}.weight"))?;
                            let ln_b =
                                match $extract_1d(&st, &format!("{prefix_dot}{ln_idx}.bias")) {
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
                    let expected_ln: usize = (0..n_hidden)
                        .map(|l| layer_sizes[l + 1])
                        .sum();
                    if ln_gamma_data.len() != expected_ln {
                        return Err(LoadError::Validation(
                            "LayerNorm must be present on all hidden layers or none",
                        ));
                    }
                    #[cfg(not(any(feature = "std", feature = "libm")))]
                    {
                        return Err(LoadError::Validation(
                            "LayerNorm requires 'std' or 'libm' feature",
                        ));
                    }
                    #[cfg(any(feature = "std", feature = "libm"))]
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

impl_mlp_safetensors!(MlpF32, f32, extract_f32_2d, extract_f32_1d);
impl_mlp_safetensors!(MlpF64, f64, extract_f64_2d, extract_f64_1d);

// ---- Conv1d loader ----

impl crate::Causal1dConvF32 {
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
    /// let conv = Causal1dConvF32::from_safetensors(
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

#[cfg(test)]
mod tests {
    use alloc::{string::ToString, vec, vec::Vec};
    use safetensors::Dtype;

    use crate::LoadError;

    fn f32_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn f64_bytes(data: &[f64]) -> Vec<u8> {
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
    #[cfg(any(feature = "std", feature = "libm"))]
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

        let lstm = crate::TinyLstmF32::from_safetensors(&data, "lstm", "fc").unwrap();
        assert_eq!(lstm.input_size(), i);
        assert_eq!(lstm.hidden_size(), h);
        assert_eq!(lstm.output_size(), o);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
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
            crate::TinyLstmF32::from_parts(i, h, o, &wih, &whh, &bih, &bhh, &wo, &bo).unwrap();

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

        let mut loaded = crate::TinyLstmF32::from_safetensors(&data, "rnn", "out").unwrap();

        let input = [0.5_f32, -0.3];
        let ref_out = reference.step(&input);
        let load_out = loaded.step(&input);
        assert!(
            (ref_out - load_out).abs() < 1e-7,
            "ref={ref_out}, loaded={load_out}"
        );
    }

    // ---- GRU ----

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
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

        let gru = crate::TinyGruF32::from_safetensors(&data, "gru", "fc").unwrap();
        assert_eq!(gru.input_size(), i);
        assert_eq!(gru.hidden_size(), h);
        assert_eq!(gru.output_size(), o);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
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
            crate::TinyGruF32::from_parts(i, h, o, &wih, &whh, &bih, &bhh, &wo, &bo).unwrap();

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

        let mut loaded = crate::TinyGruF32::from_safetensors(&data, "gru", "fc").unwrap();

        let input = [0.5_f32, -0.3];
        let ref_out = reference.step(&input);
        let load_out = loaded.step(&input);
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

        let mlp = crate::MlpF32::from_safetensors(&data, "fc", crate::Activation::Relu).unwrap();
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

        let mut reference = crate::MlpF32::from_parts(
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

        let mut loaded =
            crate::MlpF32::from_safetensors(&data, "", crate::Activation::Relu).unwrap();

        let input = [3.0_f32, 4.0];
        let ref_out = reference.predict(&input);
        let load_out = loaded.predict(&input);
        assert!(
            (ref_out - load_out).abs() < 1e-6,
            "ref={ref_out}, loaded={load_out}"
        );
    }

    #[test]
    fn mlp_f64_from_safetensors() {
        let w0: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
        let b0: Vec<f64> = vec![0.0; 2];
        let w1: Vec<f64> = vec![1.0, 1.0];
        let b1: Vec<f64> = vec![0.0];

        let w0_b = f64_bytes(&w0);
        let b0_b = f64_bytes(&b0);
        let w1_b = f64_bytes(&w1);
        let b1_b = f64_bytes(&b1);

        let data = serialize_tensors(vec![
            ("net.0.weight", make_view(Dtype::F64, &[2, 2], &w0_b)),
            ("net.0.bias", make_view(Dtype::F64, &[2], &b0_b)),
            ("net.1.weight", make_view(Dtype::F64, &[1, 2], &w1_b)),
            ("net.1.bias", make_view(Dtype::F64, &[1], &b1_b)),
        ]);

        let mut mlp =
            crate::MlpF64::from_safetensors(&data, "net", crate::Activation::Relu).unwrap();
        let out = mlp.predict(&[3.0, 4.0]);
        assert!((out - 7.0).abs() < 1e-12);
    }

    // ---- BatchNorm fusion ----

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
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

        let mut mlp = crate::MlpF32::from_safetensors(&data, "fc", crate::Activation::Relu)
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
    #[cfg(any(feature = "std", feature = "libm"))]
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

        let mut mlp = crate::MlpF32::from_safetensors(&data, "", crate::Activation::Relu).unwrap();

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

        let conv = crate::Causal1dConvF32::from_safetensors(
            &data,
            "conv",
            "fc",
            crate::Activation::Identity,
        )
        .unwrap();
        assert_eq!(conv.input_ch(), 1);
        assert_eq!(conv.kernel_size(), 3);
        assert_eq!(conv.filters(), 2);
        assert_eq!(conv.output_size(), 1);
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

        let mut conv = crate::Causal1dConvF32::from_safetensors(
            &data,
            "conv",
            "fc",
            crate::Activation::Identity,
        )
        .unwrap();

        let w_conv_ours = [0.2_f32, 0.4, 0.1, 0.3];
        let mut reference = crate::Causal1dConvF32::from_parts(
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
        let ref_out = reference.step(&input);
        let load_out = conv.step(&input);
        assert!(
            (ref_out - load_out).abs() < 1e-6,
            "ref={ref_out}, loaded={load_out}"
        );

        let input2 = [3.0_f32, 4.0];
        let ref_out2 = reference.step(&input2);
        let load_out2 = conv.step(&input2);
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

        let err = crate::MlpF32::from_safetensors(&data, "fc", crate::Activation::Relu);
        assert!(matches!(err, Err(LoadError::Parse(_))));
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn lstm_missing_tensor() {
        let w = vec![0.1_f32; 8];
        let w_b = f32_bytes(&w);
        let data = serialize_tensors(vec![(
            "lstm.weight_ih_l0",
            make_view(Dtype::F32, &[4, 2], &w_b),
        )]);

        let err = crate::TinyLstmF32::from_safetensors(&data, "lstm", "fc");
        match err {
            Err(LoadError::TensorNotFound(name)) => {
                assert_eq!(name, "lstm.weight_hh_l0".to_string());
            }
            other => panic!("expected TensorNotFound, got {other:?}"),
        }
    }

    #[test]
    fn invalid_safetensors() {
        let err = crate::MlpF32::from_safetensors(
            b"not valid safetensors",
            "fc",
            crate::Activation::Relu,
        );
        assert!(matches!(err, Err(LoadError::Parse(_))));
    }
}

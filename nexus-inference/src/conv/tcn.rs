use crate::LoadError;
use crate::activation::{Activation, activate_f32};
use crate::dot::{dot_f32, dot4_f32, matvec_bias_f32};

#[derive(Debug, Clone)]
struct TcnConvLayer {
    w_conv: Box<[f32]>,
    b_conv: Box<[f32]>,
    buffer: Box<[f32]>,
    write_idx: u16,
    buf_len: u16,
    dilation: u16,
    in_ch: u16,
}

/// Temporal convolutional network for streaming inference.
///
/// Stack of dilated causal 1D convolutions with exponentially growing
/// receptive field. Layer L has dilation `2^L`. Optional residual
/// connections add the layer input to the layer output where
/// dimensions match (layers 1+, and layer 0 when
/// `input_size == filters`).
///
/// # Architecture
///
/// ```text
/// input → conv_0(d=1) → act → conv_1(d=2) → act [+res] → ... → output_proj
/// ```
///
/// Receptive field: `1 + (kernel_size - 1) * (2^num_layers - 1)`.
///
/// # Examples
///
/// ```
/// use nexus_inference::{Activation, TinyTcn};
///
/// let filters = 4;
/// let kernel_size = 3;
///
/// // Layer 0: input_size=2 → filters=4
/// let w0 = vec![0.1_f32; filters * kernel_size * 2];
/// let b0 = vec![0.0_f32; filters];
/// // Layer 1: filters → filters, dilation=2
/// let w1 = vec![0.1_f32; filters * kernel_size * filters];
/// let b1 = vec![0.0_f32; filters];
/// let w_out = vec![0.1_f32; 1 * filters];
/// let b_out = vec![0.0_f32; 1];
///
/// let mut tcn = TinyTcn::from_parts(
///     2, filters, kernel_size, 1, false,
///     &[&w0, &w1], &[&b0, &b1],
///     &w_out, &b_out,
///     Activation::Relu,
/// ).unwrap();
///
/// let output = tcn.predict(&[0.5, 1.0]);
/// ```
#[derive(Debug, Clone)]
pub struct TinyTcn {
    layers: Box<[TcnConvLayer]>,
    w_out: Box<[f32]>,
    b_out: Box<[f32]>,
    filter_scratch: Box<[f32]>,
    lin_buf: Box<[f32]>,
    input_size: u16,
    filters: u16,
    kernel_size: u16,
    output_size: u16,
    step_count: u32,
    receptive_field: u32,
    residual: bool,
    activation: Activation,
}

impl TinyTcn {
    /// Construct from pre-trained per-layer weights.
    ///
    /// - `input_size`: features per timestep.
    /// - `filters`: channels per conv layer (all layers share the same count).
    /// - `kernel_size`: temporal kernel width.
    /// - `output_size`: final output dimension after projection.
    /// - `residual`: add skip connections where dimensions match.
    /// - `layers_w_conv`: per-layer conv weights, `(filters, kernel_size, in_ch)`
    ///   row-major. Layer 0 has `in_ch = input_size`; layers 1+ have
    ///   `in_ch = filters`. Kernel position 0 is the current (newest) input.
    /// - `layers_b_conv`: per-layer conv biases, `filters` elements each.
    /// - `w_out`: output projection, `(output_size, filters)` row-major.
    /// - `b_out`: output projection bias, `output_size` elements.
    /// - `activation`: applied to each conv layer's output before residual.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        input_size: usize,
        filters: usize,
        kernel_size: usize,
        output_size: usize,
        residual: bool,
        layers_w_conv: &[&[f32]],
        layers_b_conv: &[&[f32]],
        w_out: &[f32],
        b_out: &[f32],
        activation: Activation,
    ) -> Result<Self, LoadError> {
        let num_layers = layers_w_conv.len();
        if num_layers == 0 {
            return Err(LoadError::Validation("num_layers must be >= 1"));
        }
        if layers_b_conv.len() != num_layers {
            return Err(LoadError::Validation(
                "layers_w_conv and layers_b_conv must have the same length",
            ));
        }
        if input_size == 0 || filters == 0 || kernel_size == 0 || output_size == 0 {
            return Err(LoadError::Validation("sizes must be > 0"));
        }
        if input_size > u16::MAX as usize
            || filters > u16::MAX as usize
            || kernel_size > u16::MAX as usize
            || output_size > u16::MAX as usize
        {
            return Err(LoadError::Validation("size exceeds u16::MAX"));
        }
        if w_out.len() != output_size * filters {
            return Err(LoadError::Validation("w_out length mismatch"));
        }
        if b_out.len() != output_size {
            return Err(LoadError::Validation("b_out length mismatch"));
        }

        for &w in w_out.iter().chain(b_out) {
            if !w.is_finite() {
                return Err(LoadError::Validation("non-finite weight"));
            }
        }

        let mut layers = Vec::with_capacity(num_layers);
        let mut max_conv_len = 0;

        for k in 0..num_layers {
            let in_ch = if k == 0 { input_size } else { filters };
            let dilation = 1_usize
                .checked_shl(k as u32)
                .ok_or(LoadError::Validation("dilation overflow (too many layers)"))?;
            let buf_len = (kernel_size - 1)
                .checked_mul(dilation)
                .and_then(|v| v.checked_add(1))
                .ok_or(LoadError::Validation("buffer size overflow"))?;

            if dilation > u16::MAX as usize || buf_len > u16::MAX as usize {
                return Err(LoadError::Validation(
                    "dilation or buffer size exceeds u16::MAX",
                ));
            }

            let conv_len = kernel_size * in_ch;
            if conv_len > max_conv_len {
                max_conv_len = conv_len;
            }

            if layers_w_conv[k].len() != filters * conv_len {
                return Err(LoadError::Validation("layer w_conv length mismatch"));
            }
            if layers_b_conv[k].len() != filters {
                return Err(LoadError::Validation("layer b_conv length mismatch"));
            }

            for &w in layers_w_conv[k].iter().chain(layers_b_conv[k]) {
                if !w.is_finite() {
                    return Err(LoadError::Validation("non-finite weight"));
                }
            }

            layers.push(TcnConvLayer {
                w_conv: layers_w_conv[k].into(),
                b_conv: layers_b_conv[k].into(),
                buffer: vec![0.0_f32; buf_len * in_ch].into_boxed_slice(),
                write_idx: 0,
                buf_len: buf_len as u16,
                dilation: dilation as u16,
                in_ch: in_ch as u16,
            });
        }

        let rf = 1 + (kernel_size - 1) * ((1_usize << num_layers) - 1);

        Ok(Self {
            layers: layers.into_boxed_slice(),
            w_out: w_out.into(),
            b_out: b_out.into(),
            filter_scratch: vec![0.0_f32; filters].into_boxed_slice(),
            lin_buf: vec![0.0_f32; max_conv_len].into_boxed_slice(),
            input_size: input_size as u16,
            filters: filters as u16,
            kernel_size: kernel_size as u16,
            output_size: output_size as u16,
            step_count: 0,
            receptive_field: rf as u32,
            residual,
            activation,
        })
    }

    /// Process one timestep and return a single scalar output.
    ///
    /// Panics if `output_size != 1` or `input.len() != input_size`.
    pub fn predict(&mut self, input: &[f32]) -> f32 {
        assert_eq!(
            self.output_size, 1,
            "predict() requires output_size == 1, use predict_into()"
        );
        let mut out = [0.0_f32];
        self.predict_into(input, &mut out);
        out[0]
    }

    /// Process one timestep, writing output into caller's buffer.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != input_size` or
    /// `output.len() != output_size`.
    pub fn predict_into(&mut self, input: &[f32], output: &mut [f32]) {
        let k_size = self.kernel_size as usize;
        let n_filters = self.filters as usize;
        let n_layers = self.layers.len();
        assert_eq!(input.len(), self.input_size as usize);
        assert_eq!(output.len(), self.output_size as usize);

        let filters_4 = n_filters & !3;
        let act = self.activation;
        let res = self.residual;

        // Layer 0: write caller's input, then convolve
        buffer_write(&mut self.layers[0], input);
        conv_from_buffer(
            &mut self.layers[0],
            &mut self.lin_buf,
            &mut self.filter_scratch,
            k_size,
            n_filters,
            filters_4,
            act,
            res,
        );

        // Layers 1+: write filter_scratch (prev output) into buffer, then convolve.
        // Split API avoids aliasing — filter_scratch is read into buffer first,
        // then safely overwritten by the convolution.
        for k in 1..n_layers {
            buffer_write(&mut self.layers[k], &self.filter_scratch[..n_filters]);
            conv_from_buffer(
                &mut self.layers[k],
                &mut self.lin_buf,
                &mut self.filter_scratch,
                k_size,
                n_filters,
                filters_4,
                act,
                res,
            );
        }

        matvec_bias_f32(
            &self.w_out,
            &self.filter_scratch[..n_filters],
            &self.b_out,
            output,
            self.output_size as usize,
            n_filters,
        );

        self.step_count = self.step_count.saturating_add(1);
    }

    /// Reset all circular buffers and step counter.
    pub fn reset(&mut self) {
        for layer in &mut *self.layers {
            layer.buffer.fill(0.0);
            layer.write_idx = 0;
        }
        self.step_count = 0;
    }

    /// Whether enough steps have been taken to fill the receptive field.
    ///
    /// Before priming, early outputs are computed from zero-padded
    /// history (standard causal conv behavior).
    pub fn is_primed(&self) -> bool {
        self.step_count >= self.receptive_field
    }

    /// Number of past timesteps that influence the current output.
    ///
    /// Equal to `1 + (kernel_size - 1) * (2^num_layers - 1)`.
    pub fn receptive_field(&self) -> usize {
        self.receptive_field as usize
    }

    /// Number of input features per timestep.
    pub fn n_inputs(&self) -> usize {
        self.input_size as usize
    }

    /// Number of convolution filters (channels per layer).
    pub fn n_filters(&self) -> usize {
        self.filters as usize
    }

    /// Temporal kernel width.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size as usize
    }

    /// Number of output values per timestep.
    pub fn n_outputs(&self) -> usize {
        self.output_size as usize
    }

    /// Number of dilated conv layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Activation function applied after each conv layer.
    pub fn activation(&self) -> Activation {
        self.activation
    }

    /// Whether residual connections are enabled.
    pub fn residual(&self) -> bool {
        self.residual
    }
}

/// Write input into the layer's circular buffer. Must be called before
/// `conv_from_buffer` — separating these allows the caller to pass
/// `filter_scratch` as input without borrowing conflicts.
fn buffer_write(layer: &mut TcnConvLayer, input: &[f32]) {
    let in_ch = layer.in_ch as usize;
    let wi = layer.write_idx as usize;
    layer.buffer[wi * in_ch..(wi + 1) * in_ch].copy_from_slice(input);
}

#[allow(clippy::too_many_arguments)]
fn conv_from_buffer(
    layer: &mut TcnConvLayer,
    lin_buf: &mut [f32],
    filter_scratch: &mut [f32],
    kernel_size: usize,
    n_filters: usize,
    filters_4: usize,
    activation: Activation,
    residual: bool,
) {
    let in_ch = layer.in_ch as usize;
    let buf_len = layer.buf_len as usize;
    let dilation = layer.dilation as usize;
    let wi = layer.write_idx as usize;
    let conv_len = kernel_size * in_ch;

    // kk=0: position is always wi (current write position)
    lin_buf[..in_ch].copy_from_slice(&layer.buffer[wi * in_ch..(wi + 1) * in_ch]);

    // kk>0: conditional subtract replaces integer division (% buf_len).
    // Value before mod is in [1, 2*buf_len-1], so a single compare+subtract suffices.
    for kk in 1..kernel_size {
        let raw = wi + buf_len - kk * dilation;
        let buf_pos = if raw >= buf_len { raw - buf_len } else { raw };
        lin_buf[kk * in_ch..(kk + 1) * in_ch]
            .copy_from_slice(&layer.buffer[buf_pos * in_ch..(buf_pos + 1) * in_ch]);
    }

    let lin = &lin_buf[..conv_len];

    #[cfg(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma"),
        )
    ))]
    let mut f = super::causal1d::conv_tiled_simd(
        &layer.w_conv,
        &layer.b_conv,
        lin,
        filter_scratch,
        conv_len,
        filters_4,
        activation,
    );
    #[cfg(not(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma"),
        )
    )))]
    let mut f = 0usize;
    let _ = filters_4;

    while f < filters_4 {
        let rows = &layer.w_conv[f * conv_len..(f + 4) * conv_len];
        let dots = dot4_f32(rows, lin);
        filter_scratch[f] = activate_f32(layer.b_conv[f] + dots[0], activation);
        filter_scratch[f + 1] = activate_f32(layer.b_conv[f + 1] + dots[1], activation);
        filter_scratch[f + 2] = activate_f32(layer.b_conv[f + 2] + dots[2], activation);
        filter_scratch[f + 3] = activate_f32(layer.b_conv[f + 3] + dots[3], activation);
        f += 4;
    }
    while f < n_filters {
        let row = &layer.w_conv[f * conv_len..(f + 1) * conv_len];
        filter_scratch[f] = activate_f32(layer.b_conv[f] + dot_f32(row, lin), activation);
        f += 1;
    }

    if residual && in_ch == n_filters {
        for i in 0..n_filters {
            filter_scratch[i] += layer.buffer[wi * in_ch + i];
        }
    }

    let next = wi + 1;
    layer.write_idx = if next >= buf_len { 0 } else { next as u16 };
}

impl crate::Model for TinyTcn {
    fn predict(&mut self, input: &[f32]) -> f32 {
        TinyTcn::predict(self, input)
    }
    fn predict_into(&mut self, input: &[f32], output: &mut [f32]) {
        TinyTcn::predict_into(self, input, output);
    }
    fn n_outputs(&self) -> usize {
        TinyTcn::n_outputs(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tcn(
        input_size: usize,
        filters: usize,
        kernel: usize,
        num_layers: usize,
        output: usize,
        residual: bool,
        w_val: f32,
    ) -> TinyTcn {
        let mut w_convs = Vec::new();
        let mut b_convs = Vec::new();

        for k in 0..num_layers {
            let in_ch = if k == 0 { input_size } else { filters };
            w_convs.push(vec![w_val; filters * kernel * in_ch]);
            b_convs.push(vec![0.0_f32; filters]);
        }

        let w_refs: Vec<&[f32]> = w_convs.iter().map(|v| v.as_slice()).collect();
        let b_refs: Vec<&[f32]> = b_convs.iter().map(|v| v.as_slice()).collect();

        let w_out = vec![w_val; output * filters];
        let b_out = vec![0.0_f32; output];

        TinyTcn::from_parts(
            input_size,
            filters,
            kernel,
            output,
            residual,
            &w_refs,
            &b_refs,
            &w_out,
            &b_out,
            Activation::Identity,
        )
        .unwrap()
    }

    #[test]
    fn single_layer_known_output() {
        // 1 layer, 1 ch, K=2, 1 filter, 1 output, identity, no residual
        // w_conv = [0.5, 0.3] → k=0 (current): 0.5, k=1 (prev): 0.3
        let w_conv = [0.5_f32, 0.3];
        let b_conv = [0.0_f32];
        let w_out = [1.0_f32]; // pass-through
        let b_out = [0.0_f32];
        let mut tcn = TinyTcn::from_parts(
            1,
            1,
            2,
            1,
            false,
            &[&w_conv],
            &[&b_conv],
            &w_out,
            &b_out,
            Activation::Identity,
        )
        .unwrap();

        // Step 1: buffer=[1, 0], lin=[1, 0], out = 0.5*1 + 0.3*0 = 0.5
        let o1 = tcn.predict(&[1.0]);
        assert!((o1 - 0.5).abs() < 1e-6, "step1: {o1}");

        // Step 2: buffer=[1, 2], lin=[2, 1], out = 0.5*2 + 0.3*1 = 1.3
        let o2 = tcn.predict(&[2.0]);
        assert!((o2 - 1.3).abs() < 1e-6, "step2: {o2}");

        // Step 3: buffer=[3, 2], lin=[3, 2], out = 0.5*3 + 0.3*2 = 2.1
        let o3 = tcn.predict(&[3.0]);
        assert!((o3 - 2.1).abs() < 1e-6, "step3: {o3}");
    }

    #[test]
    fn two_layers_differs_from_single() {
        let mut one = make_tcn(2, 4, 3, 1, 1, false, 0.1);
        let mut two = make_tcn(2, 4, 3, 2, 1, false, 0.1);

        let input = [1.0_f32, 0.5];
        let o1 = one.predict(&input);
        let o2 = two.predict(&input);
        assert!(
            (o1 - o2).abs() > 1e-6,
            "single and double should differ: {o1} vs {o2}"
        );
    }

    #[test]
    fn residual_changes_output() {
        let mut no_res = make_tcn(4, 4, 2, 2, 1, false, 0.1);
        let mut with_res = make_tcn(4, 4, 2, 2, 1, true, 0.1);

        let input = [1.0_f32, 0.5, -0.3, 0.8];
        let o1 = no_res.predict(&input);
        let o2 = with_res.predict(&input);
        assert!(
            (o1 - o2).abs() > 1e-6,
            "residual should change output: {o1} vs {o2}"
        );
    }

    #[test]
    fn residual_layer0_when_dims_match() {
        // input_size == filters → layer 0 gets residual
        let mut tcn = make_tcn(4, 4, 2, 1, 1, true, 0.1);
        let mut tcn_no = make_tcn(4, 4, 2, 1, 1, false, 0.1);

        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let o_res = tcn.predict(&input);
        let o_no = tcn_no.predict(&input);
        assert!(
            (o_res - o_no).abs() > 1e-6,
            "layer0 residual when dims match: {o_res} vs {o_no}"
        );
    }

    #[test]
    fn residual_skips_layer0_dim_mismatch() {
        // input_size=2, filters=4 → layer 0 skips residual
        // With only 1 layer, residual flag has no effect when dims don't match
        let mut tcn_res = make_tcn(2, 4, 2, 1, 1, true, 0.1);
        let mut tcn_no = make_tcn(2, 4, 2, 1, 1, false, 0.1);

        let input = [1.0_f32, 0.5];
        let o1 = tcn_res.predict(&input);
        let o2 = tcn_no.predict(&input);
        assert!(
            (o1 - o2).abs() < 1e-6,
            "layer0 residual should be skipped when dims don't match: {o1} vs {o2}"
        );
    }

    #[test]
    fn priming_sequence() {
        // K=3, 2 layers → RF = 1 + 2*(4-1) = 7
        let mut tcn = make_tcn(1, 2, 3, 2, 1, false, 0.1);
        assert_eq!(tcn.receptive_field(), 7);
        assert!(!tcn.is_primed());

        for _ in 0..6 {
            tcn.predict(&[1.0]);
            assert!(!tcn.is_primed());
        }
        tcn.predict(&[1.0]);
        assert!(tcn.is_primed()); // step 7
    }

    #[test]
    fn receptive_field_formula() {
        // RF = 1 + (K-1) * (2^L - 1)
        let tcn = make_tcn(1, 2, 3, 1, 1, false, 0.1);
        assert_eq!(tcn.receptive_field(), 3); // 1 + 2*(2-1)

        let tcn = make_tcn(1, 2, 3, 3, 1, false, 0.1);
        assert_eq!(tcn.receptive_field(), 15); // 1 + 2*(8-1)

        let tcn = make_tcn(1, 2, 2, 4, 1, false, 0.1);
        assert_eq!(tcn.receptive_field(), 16); // 1 + 1*(16-1)
    }

    #[test]
    fn reset_reproduces_first_output() {
        let mut tcn = make_tcn(2, 4, 3, 2, 1, false, 0.1);
        let first = tcn.predict(&[1.0, -0.5]);
        tcn.predict(&[0.3, 0.8]);
        tcn.predict(&[0.0, 1.0]);
        tcn.reset();
        assert!(!tcn.is_primed());
        let after_reset = tcn.predict(&[1.0, -0.5]);
        assert!(
            (first - after_reset).abs() < 1e-6,
            "first={first}, after_reset={after_reset}"
        );
    }

    #[test]
    fn multi_output() {
        let mut tcn = make_tcn(2, 4, 2, 2, 3, false, 0.1);
        let mut out = [0.0_f32; 3];
        tcn.predict_into(&[1.0, 0.5], &mut out);
        // All weights identical → all outputs equal
        assert!((out[0] - out[1]).abs() < 1e-6);
        assert!((out[1] - out[2]).abs() < 1e-6);
    }

    #[test]
    fn state_carries_between_steps() {
        let mut tcn = make_tcn(2, 4, 3, 2, 1, false, 0.1);
        let out1 = tcn.predict(&[1.0, 0.5]);
        let out2 = tcn.predict(&[1.0, 0.5]);
        assert!(
            (out1 - out2).abs() > 1e-6,
            "state should carry: {out1} vs {out2}"
        );
    }

    #[test]
    fn relu_activation() {
        let w_conv = [-1.0_f32]; // K=1, ch=1, 1 filter
        let b_conv = [0.0_f32];
        let w_out = [1.0_f32];
        let b_out = [0.0_f32];
        let mut tcn = TinyTcn::from_parts(
            1,
            1,
            1,
            1,
            false,
            &[&w_conv],
            &[&b_conv],
            &w_out,
            &b_out,
            Activation::Relu,
        )
        .unwrap();

        // conv = -1 * 5 = -5, relu(-5) = 0
        let out = tcn.predict(&[5.0]);
        assert!((out - 0.0).abs() < 1e-6, "{out}");

        // conv = -1 * (-3) = 3, relu(3) = 3
        let out2 = tcn.predict(&[-3.0]);
        assert!((out2 - 3.0).abs() < 1e-6, "{out2}");
    }

    #[test]
    fn dilation_offsets() {
        // 2 layers: d=1 (buf_len=3), d=2 (buf_len=5)
        // K=3, 1 ch, 1 filter, identity
        // Layer 0 weights: k0=1, k1=0, k2=0 → just passes current input
        let w0 = [1.0_f32, 0.0, 0.0];
        let b0 = [0.0_f32];
        // Layer 1 weights: k0=0, k1=0, k2=1 → reads 2*dilation=4 steps back
        let w1 = [0.0_f32, 0.0, 1.0];
        let b1 = [0.0_f32];
        let w_out = [1.0_f32];
        let b_out = [0.0_f32];
        let mut tcn = TinyTcn::from_parts(
            1,
            1,
            3,
            1,
            false,
            &[&w0, &w1],
            &[&b0, &b1],
            &w_out,
            &b_out,
            Activation::Identity,
        )
        .unwrap();

        // Feed known sequence, verify dilation picks up the right step
        tcn.predict(&[10.0]); // t=0
        tcn.predict(&[20.0]); // t=1
        tcn.predict(&[30.0]); // t=2
        tcn.predict(&[40.0]); // t=3

        // t=4: layer 0 passes 50.0 through (k0=1, current)
        // layer 1 k2=1 reads 2*2=4 steps back in its buffer
        // layer 1's buffer at t=4 has received: 10,20,30,40,50
        // k2 at dilation=2: position (wi+5-2*2)%5 = 4 steps back = 10.0
        let o = tcn.predict(&[50.0]);
        assert!(
            (o - 10.0).abs() < 1e-5,
            "dilation should reach 4 steps back: {o}"
        );
    }

    #[test]
    fn accessors() {
        let tcn = make_tcn(3, 8, 4, 3, 2, true, 0.1);
        assert_eq!(tcn.n_inputs(), 3);
        assert_eq!(tcn.n_filters(), 8);
        assert_eq!(tcn.kernel_size(), 4);
        assert_eq!(tcn.n_outputs(), 2);
        assert_eq!(tcn.n_layers(), 3);
        assert!(tcn.residual());
        assert!(matches!(tcn.activation(), Activation::Identity));
        // RF = 1 + 3*(8-1) = 22
        assert_eq!(tcn.receptive_field(), 22);
    }

    #[test]
    fn validation_rejects_zero_layers() {
        let r: Result<TinyTcn, _> = TinyTcn::from_parts(
            1,
            4,
            3,
            1,
            false,
            &[],
            &[],
            &[0.0; 4],
            &[0.0; 1],
            Activation::Relu,
        );
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_zero_size() {
        let w = [0.0_f32; 4];
        let b = [0.0_f32; 4];
        let r = TinyTcn::from_parts(
            0,
            4,
            2,
            1,
            false,
            &[&w],
            &[&b],
            &[0.0; 4],
            &[0.0; 1],
            Activation::Relu,
        );
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_weight_mismatch() {
        let r = TinyTcn::from_parts(
            2,
            4,
            3,
            1,
            false,
            &[&[0.0; 23]], // wrong: should be 4*3*2=24
            &[&[0.0; 4]],
            &[0.0; 4],
            &[0.0; 1],
            Activation::Relu,
        );
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_non_finite() {
        let mut w = vec![0.1_f32; 8]; // 2*2*2
        w[3] = f32::NAN;
        let r = TinyTcn::from_parts(
            2,
            2,
            2,
            1,
            false,
            &[&w],
            &[&[0.0; 2]],
            &[0.0; 2],
            &[0.0; 1],
            Activation::Relu,
        );
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_mismatched_layer_counts() {
        let w0 = [0.0_f32; 8]; // 2*2*2
        let b0 = [0.0_f32; 2];
        let r = TinyTcn::from_parts(
            2,
            2,
            2,
            1,
            false,
            &[&w0, &w0], // 2 weight layers
            &[&b0],      // 1 bias layer
            &[0.0; 2],
            &[0.0; 1],
            Activation::Relu,
        );
        assert!(r.is_err());
    }

    #[test]
    #[should_panic(expected = "output_size == 1")]
    fn predict_panics_multi_output() {
        let mut tcn = make_tcn(2, 4, 2, 1, 3, false, 0.1);
        tcn.predict(&[1.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn predict_panics_wrong_input_len() {
        let mut tcn = make_tcn(2, 4, 2, 1, 1, false, 0.1);
        tcn.predict(&[1.0]);
    }
}

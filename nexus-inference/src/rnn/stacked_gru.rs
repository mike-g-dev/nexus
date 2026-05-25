use crate::LoadError;
use crate::dot::{matvec_bias_f32, matvec_f32};

/// Multi-layer GRU for streaming temporal inference.
///
/// Stacks N single-layer GRUs where each layer's hidden state feeds
/// as input to the next. Output projection applies only to the final
/// layer's hidden state. Matches PyTorch's `nn.GRU(num_layers=N)`.
///
/// All layers share the same `hidden_size`. Layer 0 takes `input_size`
/// features; layers 1..N take `hidden_size` features from the previous
/// layer's hidden state. ~75% of LSTM compute per layer (3 gates
/// instead of 4, no separate cell state).
///
/// # Examples
///
/// ```
/// use nexus_inference::StackedGru;
///
/// let input_size = 4;
/// let hidden_size = 8;
///
/// // Layer 0
/// let wih_l0 = vec![0.1_f32; 3 * hidden_size * input_size];
/// let whh_l0 = vec![0.1_f32; 3 * hidden_size * hidden_size];
/// let bih_l0 = vec![0.0_f32; 3 * hidden_size];
/// let bhh_l0 = vec![0.0_f32; 3 * hidden_size];
///
/// // Layer 1
/// let wih_l1 = vec![0.1_f32; 3 * hidden_size * hidden_size];
/// let whh_l1 = vec![0.1_f32; 3 * hidden_size * hidden_size];
/// let bih_l1 = vec![0.0_f32; 3 * hidden_size];
/// let bhh_l1 = vec![0.0_f32; 3 * hidden_size];
///
/// let w_out = vec![0.1_f32; 1 * hidden_size];
/// let b_out = vec![0.0_f32; 1];
///
/// let mut gru = StackedGru::from_parts(
///     input_size, hidden_size, 1,
///     &[&wih_l0, &wih_l1],
///     &[&whh_l0, &whh_l1],
///     &[&bih_l0, &bih_l1],
///     &[&bhh_l0, &bhh_l1],
///     &w_out, &b_out,
/// ).unwrap();
///
/// let output = gru.predict(&[0.5, 1.2, -0.3, 0.8]);
/// ```
#[derive(Debug, Clone)]
pub struct StackedGru {
    layers: Box<[GruLayer]>,
    w_out: Box<[f32]>,
    b_out: Box<[f32]>,
    input_size: u16,
    hidden_size: u16,
    output_size: u16,
}

#[derive(Debug, Clone)]
struct GruLayer {
    weight_ih: Box<[f32]>,
    weight_hh: Box<[f32]>,
    bias_ih: Box<[f32]>,
    bias_hh: Box<[f32]>,
    h: Box<[f32]>,
    ih_scratch: Box<[f32]>,
    hh_scratch: Box<[f32]>,
}

impl GruLayer {
    fn new(
        layer_input_size: usize,
        hidden_size: usize,
        weight_ih: &[f32],
        weight_hh: &[f32],
        bias_ih: &[f32],
        bias_hh: &[f32],
    ) -> Result<Self, LoadError> {
        let gate_count = 3 * hidden_size;

        if weight_ih.len() != gate_count * layer_input_size {
            return Err(LoadError::Validation("weight_ih length mismatch"));
        }
        if weight_hh.len() != gate_count * hidden_size {
            return Err(LoadError::Validation("weight_hh length mismatch"));
        }
        if bias_ih.len() != gate_count {
            return Err(LoadError::Validation("bias_ih length mismatch"));
        }
        if bias_hh.len() != gate_count {
            return Err(LoadError::Validation("bias_hh length mismatch"));
        }

        for &w in weight_ih
            .iter()
            .chain(weight_hh)
            .chain(bias_ih)
            .chain(bias_hh)
        {
            if !w.is_finite() {
                return Err(LoadError::Validation("non-finite weight"));
            }
        }

        Ok(Self {
            weight_ih: weight_ih.into(),
            weight_hh: weight_hh.into(),
            bias_ih: bias_ih.into(),
            bias_hh: bias_hh.into(),
            h: vec![0.0_f32; hidden_size].into_boxed_slice(),
            ih_scratch: vec![0.0_f32; gate_count].into_boxed_slice(),
            hh_scratch: vec![0.0_f32; gate_count].into_boxed_slice(),
        })
    }

    fn step(&mut self, input: &[f32], hidden_size: usize) {
        let gate_count = 3 * hidden_size;

        matvec_f32(
            &self.weight_ih,
            input,
            &mut self.ih_scratch,
            gate_count,
            input.len(),
        );

        matvec_f32(
            &self.weight_hh,
            &self.h[..hidden_size],
            &mut self.hh_scratch,
            gate_count,
            hidden_size,
        );

        super::apply_gru_gates(
            &self.ih_scratch,
            &self.hh_scratch,
            &self.bias_ih,
            &self.bias_hh,
            &mut self.h,
            hidden_size,
        );
    }
}

impl StackedGru {
    /// Construct from pre-trained per-layer weights.
    ///
    /// Each slice in `layers_weight_ih`, `layers_weight_hh`,
    /// `layers_bias_ih`, `layers_bias_hh` corresponds to one GRU
    /// layer. The number of layers is determined by the slice lengths
    /// (must all be equal and >= 1).
    ///
    /// Layer 0 expects `weight_ih` shape `(3*hidden, input_size)`.
    /// Layers 1+ expect `weight_ih` shape `(3*hidden, hidden_size)`.
    /// All layers expect `weight_hh` shape `(3*hidden, hidden_size)`.
    ///
    /// `w_out` and `b_out` are the output projection applied to the
    /// final layer's hidden state.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        layers_weight_ih: &[&[f32]],
        layers_weight_hh: &[&[f32]],
        layers_bias_ih: &[&[f32]],
        layers_bias_hh: &[&[f32]],
        w_out: &[f32],
        b_out: &[f32],
    ) -> Result<Self, LoadError> {
        let num_layers = layers_weight_ih.len();
        if num_layers == 0 {
            return Err(LoadError::Validation("num_layers must be >= 1"));
        }
        if layers_weight_hh.len() != num_layers
            || layers_bias_ih.len() != num_layers
            || layers_bias_hh.len() != num_layers
        {
            return Err(LoadError::Validation(
                "all per-layer weight slices must have the same length",
            ));
        }
        if input_size == 0 || hidden_size == 0 || output_size == 0 {
            return Err(LoadError::Validation("sizes must be > 0"));
        }
        if input_size > u16::MAX as usize
            || hidden_size > u16::MAX as usize
            || output_size > u16::MAX as usize
        {
            return Err(LoadError::Validation("size exceeds u16::MAX"));
        }
        if w_out.len() != output_size * hidden_size {
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
        for k in 0..num_layers {
            let layer_input = if k == 0 { input_size } else { hidden_size };
            layers.push(GruLayer::new(
                layer_input,
                hidden_size,
                layers_weight_ih[k],
                layers_weight_hh[k],
                layers_bias_ih[k],
                layers_bias_hh[k],
            )?);
        }

        Ok(Self {
            layers: layers.into_boxed_slice(),
            w_out: w_out.into(),
            b_out: b_out.into(),
            input_size: input_size as u16,
            hidden_size: hidden_size as u16,
            output_size: output_size as u16,
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
    /// Each layer processes the input through its GRU gates, updating
    /// its own hidden state. Layer 0 receives the user's input;
    /// subsequent layers receive the previous layer's hidden state.
    /// The output projection is applied only to the final layer's
    /// hidden state.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != input_size` or
    /// `output.len() != output_size`.
    pub fn predict_into(&mut self, input: &[f32], output: &mut [f32]) {
        let h = self.hidden_size as usize;
        let n = self.layers.len();
        assert_eq!(
            input.len(),
            self.input_size as usize,
            "input length must equal input_size"
        );
        assert_eq!(
            output.len(),
            self.output_size as usize,
            "output length must equal output_size"
        );

        self.layers[0].step(input, h);

        for k in 1..n {
            let (prev, rest) = self.layers.split_at_mut(k);
            let prev_h: &[f32] = &prev[k - 1].h;
            rest[0].step(prev_h, h);
        }

        let last_h = &self.layers[n - 1].h;
        matvec_bias_f32(
            &self.w_out,
            &last_h[..h],
            &self.b_out,
            output,
            self.output_size as usize,
            h,
        );
    }

    /// Reset all layers' hidden state to zeros.
    pub fn reset(&mut self) {
        for layer in &mut *self.layers {
            layer.h.fill(0.0);
        }
    }

    /// Hidden state of a specific layer.
    ///
    /// # Panics
    ///
    /// Panics if `layer >= n_layers()`.
    pub fn hidden_state(&self, layer: usize) -> &[f32] {
        &self.layers[layer].h
    }

    /// Number of stacked GRU layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Number of input features per timestep.
    pub fn n_inputs(&self) -> usize {
        self.input_size as usize
    }

    /// Number of hidden units (same for all layers).
    pub fn n_hidden(&self) -> usize {
        self.hidden_size as usize
    }

    /// Number of output values per timestep.
    pub fn n_outputs(&self) -> usize {
        self.output_size as usize
    }
}

impl crate::Model for StackedGru {
    fn predict(&mut self, input: &[f32]) -> f32 {
        StackedGru::predict(self, input)
    }
    fn predict_into(&mut self, input: &[f32], output: &mut [f32]) {
        StackedGru::predict_into(self, input, output);
    }
    fn n_outputs(&self) -> usize {
        StackedGru::n_outputs(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stacked_gru(
        input: usize,
        hidden: usize,
        output: usize,
        num_layers: usize,
        val: f32,
    ) -> StackedGru {
        let gc = 3 * hidden;
        let wih_l0 = vec![val; gc * input];
        let whh_l0 = vec![val; gc * hidden];
        let bih = vec![0.0_f32; gc];
        let bhh = vec![0.0_f32; gc];

        let wih_rest = vec![val; gc * hidden];

        let mut layers_wih: Vec<&[f32]> = Vec::new();
        let mut layers_whh: Vec<&[f32]> = Vec::new();
        let mut layers_bih: Vec<&[f32]> = Vec::new();
        let mut layers_bhh: Vec<&[f32]> = Vec::new();

        layers_wih.push(&wih_l0);
        layers_whh.push(&whh_l0);
        layers_bih.push(&bih);
        layers_bhh.push(&bhh);

        for _ in 1..num_layers {
            layers_wih.push(&wih_rest);
            layers_whh.push(&whh_l0);
            layers_bih.push(&bih);
            layers_bhh.push(&bhh);
        }

        let w_out = vec![val; output * hidden];
        let b_out = vec![0.0_f32; output];

        StackedGru::from_parts(
            input,
            hidden,
            output,
            &layers_wih,
            &layers_whh,
            &layers_bih,
            &layers_bhh,
            &w_out,
            &b_out,
        )
        .unwrap()
    }

    #[test]
    fn single_layer_matches_tiny() {
        let input_size = 4;
        let hidden_size = 8;
        let output_size = 2;
        let gc = 3 * hidden_size;

        let wih = vec![0.1_f32; gc * input_size];
        let whh = vec![0.05_f32; gc * hidden_size];
        let bih = vec![0.01_f32; gc];
        let bhh = vec![-0.01_f32; gc];
        let w_out = vec![0.2_f32; output_size * hidden_size];
        let b_out = vec![0.1_f32; output_size];

        let mut tiny = crate::TinyGru::from_parts(
            input_size,
            hidden_size,
            output_size,
            &wih,
            &whh,
            &bih,
            &bhh,
            &w_out,
            &b_out,
        )
        .unwrap();

        let mut stacked = StackedGru::from_parts(
            input_size,
            hidden_size,
            output_size,
            &[&wih],
            &[&whh],
            &[&bih],
            &[&bhh],
            &w_out,
            &b_out,
        )
        .unwrap();

        assert_eq!(stacked.n_layers(), 1);

        let input = [0.5_f32, -0.3, 1.2, 0.0];
        let mut tiny_out = [0.0_f32; 2];
        let mut stacked_out = [0.0_f32; 2];
        tiny.predict_into(&input, &mut tiny_out);
        stacked.predict_into(&input, &mut stacked_out);

        for i in 0..output_size {
            assert!(
                (tiny_out[i] - stacked_out[i]).abs() < 1e-6,
                "output {i}: tiny={}, stacked={}",
                tiny_out[i],
                stacked_out[i]
            );
        }
    }

    #[test]
    fn two_layer_differs_from_single() {
        let mut single = make_stacked_gru(4, 8, 1, 1, 0.1);
        let mut double = make_stacked_gru(4, 8, 1, 2, 0.1);

        let input = [1.0_f32, 0.5, -0.3, 0.8];
        let out1 = single.predict(&input);
        let out2 = double.predict(&input);

        assert!(
            (out1 - out2).abs() > 1e-6,
            "single and double should differ: {out1} vs {out2}"
        );
    }

    #[test]
    fn state_carries_between_steps() {
        let mut gru = make_stacked_gru(2, 4, 1, 2, 0.1);
        let out1 = gru.predict(&[1.0, 0.0]);
        let out2 = gru.predict(&[1.0, 0.0]);
        assert!((out1 - out2).abs() > 1e-6);
    }

    #[test]
    fn reset_clears_all_layers() {
        let mut gru = make_stacked_gru(2, 4, 1, 3, 0.1);
        gru.predict(&[1.0, 0.5]);

        for k in 0..3 {
            assert!(gru.hidden_state(k).iter().any(|&v| v != 0.0));
        }

        gru.reset();

        for k in 0..3 {
            assert!(gru.hidden_state(k).iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn reset_reproduces_first_output() {
        let mut gru = make_stacked_gru(2, 4, 1, 2, 0.1);
        let first = gru.predict(&[1.0, -1.0]);
        gru.predict(&[0.5, 0.5]);
        gru.predict(&[0.0, 1.0]);
        gru.reset();
        let after_reset = gru.predict(&[1.0, -1.0]);
        assert!(
            (first - after_reset).abs() < 1e-6,
            "first={first}, after_reset={after_reset}"
        );
    }

    #[test]
    fn multi_output() {
        let mut gru = make_stacked_gru(2, 4, 3, 2, 0.1);
        let mut out = [0.0_f32; 3];
        gru.predict_into(&[1.0, 0.5], &mut out);
        assert!((out[0] - out[1]).abs() < 1e-6);
        assert!((out[1] - out[2]).abs() < 1e-6);
    }

    #[test]
    fn accessors() {
        let gru = make_stacked_gru(4, 8, 2, 3, 0.1);
        assert_eq!(gru.n_inputs(), 4);
        assert_eq!(gru.n_hidden(), 8);
        assert_eq!(gru.n_outputs(), 2);
        assert_eq!(gru.n_layers(), 3);
        assert_eq!(gru.hidden_state(0).len(), 8);
        assert_eq!(gru.hidden_state(2).len(), 8);
    }

    #[test]
    fn validation_rejects_zero_layers() {
        let r = StackedGru::from_parts(2, 4, 1, &[], &[], &[], &[], &[0.0; 4], &[0.0; 1]);
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_mismatched_layer_counts() {
        let gc = 3 * 4;
        let wih = vec![0.1_f32; gc * 2];
        let whh = vec![0.1_f32; gc * 4];
        let bih = vec![0.0_f32; gc];
        let bhh = vec![0.0_f32; gc];
        let r = StackedGru::from_parts(
            2,
            4,
            1,
            &[&wih],
            &[&whh, &whh],
            &[&bih],
            &[&bhh],
            &[0.0; 4],
            &[0.0; 1],
        );
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_zero_size() {
        let r = StackedGru::from_parts(0, 4, 1, &[&[]], &[&[]], &[&[]], &[&[]], &[], &[]);
        assert!(r.is_err());
    }

    #[test]
    #[should_panic(expected = "output_size == 1")]
    fn predict_panics_multi_output() {
        let mut gru = make_stacked_gru(2, 4, 3, 2, 0.1);
        gru.predict(&[1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "input length")]
    fn predict_panics_wrong_input_len() {
        let mut gru = make_stacked_gru(2, 4, 1, 2, 0.1);
        gru.predict(&[1.0]);
    }
}

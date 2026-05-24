extern crate alloc;

use alloc::{boxed::Box, vec};

use crate::LoadError;
use crate::dot::{matvec_bias_f32, matvec_f32};

#[cfg(not(all(
    target_arch = "x86_64",
    any(
        target_feature = "avx512f",
        all(target_feature = "avx2", target_feature = "fma"),
    )
)))]
#[allow(unused_imports)]
use super::{sigmoid_f32, tanh_f32};

/// Single-layer GRU for streaming temporal inference.
///
/// Three gates (reset, update, candidate) with hidden state carried
/// between steps. ~75% of LSTM cost for comparable quality on many
/// tasks. No separate cell state — simpler memory model.
///
/// Gate activations are hardcoded: sigmoid for reset/update gates,
/// tanh for candidate. Matches PyTorch's `nn.GRU` formulation
/// (reset applied after hidden-to-hidden matmul).
///
/// Weight parameters map directly to PyTorch's `nn.GRU` tensors
/// (gate order: reset, update, candidate). The output projection
/// is a separate linear layer (`nn.Linear`).
///
/// # Examples
///
/// ```
/// use nexus_inference::TinyGruF32;
///
/// let weight_ih = vec![0.1_f32; 3 * 8 * 4];
/// let weight_hh = vec![0.1_f32; 3 * 8 * 8];
/// let bias_ih = vec![0.0_f32; 3 * 8];
/// let bias_hh = vec![0.0_f32; 3 * 8];
/// let w_out = vec![0.1_f32; 1 * 8];
/// let b_out = vec![0.0_f32; 1];
///
/// let mut gru = TinyGruF32::from_parts(
///     4, 8, 1,
///     &weight_ih, &weight_hh,
///     &bias_ih, &bias_hh,
///     &w_out, &b_out,
/// ).unwrap();
///
/// let output = gru.step(&[0.5, 1.2, -0.3, 0.8]);
/// ```
#[derive(Debug, Clone)]
pub struct TinyGruF32 {
    weight_ih: Box<[f32]>,
    weight_hh: Box<[f32]>,
    bias_ih: Box<[f32]>,
    bias_hh: Box<[f32]>,
    w_out: Box<[f32]>,
    b_out: Box<[f32]>,
    h: Box<[f32]>,
    ih_scratch: Box<[f32]>,
    hh_scratch: Box<[f32]>,
    input_size: u16,
    hidden_size: u16,
    output_size: u16,
}

impl TinyGruF32 {
    /// Construct from pre-trained weights.
    ///
    /// Parameters map to PyTorch's `nn.GRU` + `nn.Linear`:
    ///
    /// - `weight_ih`: input-to-hidden weights, `(3*hidden, input)` row-major.
    ///   Gate order: reset, update, candidate.
    /// - `weight_hh`: hidden-to-hidden weights, `(3*hidden, hidden)` row-major.
    ///   Same gate order.
    /// - `bias_ih`, `bias_hh`: gate biases, `3*hidden` each.
    /// - `w_out`: output projection, `(output, hidden)` row-major.
    /// - `b_out`: output bias, `output` elements.
    ///
    /// Unlike LSTM, GRU weights are stored separately because the
    /// candidate gate applies the reset gate between the two matmul
    /// halves (PyTorch's `reset_after` formulation).
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight_ih: &[f32],
        weight_hh: &[f32],
        bias_ih: &[f32],
        bias_hh: &[f32],
        w_out: &[f32],
        b_out: &[f32],
    ) -> Result<Self, LoadError> {
        if input_size == 0 || hidden_size == 0 || output_size == 0 {
            return Err(LoadError::Validation("sizes must be > 0"));
        }
        if input_size > u16::MAX as usize
            || hidden_size > u16::MAX as usize
            || output_size > u16::MAX as usize
        {
            return Err(LoadError::Validation("size exceeds u16::MAX"));
        }

        let gate_count = 3 * hidden_size;

        if weight_ih.len() != gate_count * input_size {
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
        if w_out.len() != output_size * hidden_size {
            return Err(LoadError::Validation("w_out length mismatch"));
        }
        if b_out.len() != output_size {
            return Err(LoadError::Validation("b_out length mismatch"));
        }

        for &w in weight_ih
            .iter()
            .chain(weight_hh)
            .chain(bias_ih)
            .chain(bias_hh)
            .chain(w_out)
            .chain(b_out)
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
            w_out: w_out.into(),
            b_out: b_out.into(),
            h: vec![0.0_f32; hidden_size].into_boxed_slice(),
            ih_scratch: vec![0.0_f32; gate_count].into_boxed_slice(),
            hh_scratch: vec![0.0_f32; gate_count].into_boxed_slice(),
            input_size: input_size as u16,
            hidden_size: hidden_size as u16,
            output_size: output_size as u16,
        })
    }

    /// Process one timestep and return a single scalar output.
    ///
    /// Panics if `output_size != 1` or `input.len() != input_size`.
    pub fn step(&mut self, input: &[f32]) -> f32 {
        assert_eq!(
            self.output_size, 1,
            "step() requires output_size == 1, use step_into()"
        );
        let mut out = [0.0_f32];
        self.step_into(input, &mut out);
        out[0]
    }

    /// Process one timestep, writing output into caller's buffer.
    ///
    /// GRU equations (PyTorch formulation):
    /// ```text
    /// r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
    /// z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
    /// n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
    /// h' = (1 - z) * n + z * h
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != input_size` or
    /// `output.len() != output_size`.
    #[allow(clippy::many_single_char_names)]
    pub fn step_into(&mut self, input: &[f32], output: &mut [f32]) {
        let in_sz = self.input_size as usize;
        let hi = self.hidden_size as usize;
        let gate_count = 3 * hi;
        assert_eq!(input.len(), in_sz, "input length must equal input_size");
        assert_eq!(
            output.len(),
            self.output_size as usize,
            "output length must equal output_size"
        );

        // ih_scratch = weight_ih @ x
        matvec_f32(
            &self.weight_ih,
            input,
            &mut self.ih_scratch,
            gate_count,
            in_sz,
        );

        // hh_scratch = weight_hh @ h
        matvec_f32(
            &self.weight_hh,
            &self.h[..hi],
            &mut self.hh_scratch,
            gate_count,
            hi,
        );

        super::apply_gru_gates(
            &self.ih_scratch,
            &self.hh_scratch,
            &self.bias_ih,
            &self.bias_hh,
            &mut self.h,
            hi,
        );

        matvec_bias_f32(
            &self.w_out,
            &self.h[..hi],
            &self.b_out,
            output,
            self.output_size as usize,
            hi,
        );
    }

    /// Reset hidden state to zeros.
    pub fn reset_state(&mut self) {
        self.h.fill(0.0);
    }

    /// Current hidden state.
    pub fn hidden_state(&self) -> &[f32] {
        &self.h
    }

    /// Number of input features per timestep.
    pub fn input_size(&self) -> usize {
        self.input_size as usize
    }

    /// Number of hidden units.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size as usize
    }

    /// Number of output values per timestep.
    pub fn output_size(&self) -> usize {
        self.output_size as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{sigmoid_f32, tanh_f32};

    fn make_gru(
        input: usize,
        hidden: usize,
        output: usize,
        w_ih_val: f32,
        w_hh_val: f32,
        b_val: f32,
        w_out_val: f32,
    ) -> TinyGruF32 {
        let gate_count = 3 * hidden;
        let weight_ih = vec![w_ih_val; gate_count * input];
        let weight_hh = vec![w_hh_val; gate_count * hidden];
        let bias_ih = vec![b_val; gate_count];
        let bias_hh = vec![0.0_f32; gate_count];
        let w_out = vec![w_out_val; output * hidden];
        let b_out = vec![0.0_f32; output];
        TinyGruF32::from_parts(
            input, hidden, output, &weight_ih, &weight_hh, &bias_ih, &bias_hh, &w_out, &b_out,
        )
        .unwrap()
    }

    #[test]
    fn basic_forward_pass() {
        let mut gru = make_gru(2, 2, 1, 0.1, 0.1, 0.0, 0.1);
        let out = gru.step(&[1.0, 0.0]);

        // h_0 = [0, 0], x = [1, 0]
        // ih = W_ih @ x: each row dot [1, 0] = 0.1
        // hh = W_hh @ h: each row dot [0, 0] = 0.0
        // r = sigmoid(0.1 + 0 + 0 + 0) = sigmoid(0.1)
        // z = sigmoid(0.1 + 0 + 0 + 0) = sigmoid(0.1)
        // n = tanh(0.1 + 0 + r * (0 + 0)) = tanh(0.1)
        // h' = (1-z)*n + z*0 = (1-sigmoid(0.1))*tanh(0.1)
        let z = sigmoid_f32(0.1);
        let n = tanh_f32(0.1);
        let h1 = (1.0 - z) * n;
        let expected = 0.1 * h1 + 0.1 * h1;
        assert!(
            (out - expected).abs() < 1e-6,
            "got {out}, expected {expected}"
        );
    }

    #[test]
    fn state_carries_between_steps() {
        let mut gru = make_gru(2, 2, 1, 0.1, 0.1, 0.0, 0.1);
        let out1 = gru.step(&[1.0, 0.0]);
        let out2 = gru.step(&[1.0, 0.0]);
        assert!((out1 - out2).abs() > 1e-6);
    }

    #[test]
    fn reset_clears_state() {
        let mut gru = make_gru(2, 2, 1, 0.1, 0.1, 0.0, 0.1);
        gru.step(&[1.0, 0.5]);
        assert!(gru.hidden_state().iter().any(|&v| v != 0.0));
        gru.reset_state();
        assert!(gru.hidden_state().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn reset_reproduces_first_output() {
        let mut gru = make_gru(2, 4, 1, 0.1, 0.2, 0.0, 0.1);
        let first = gru.step(&[1.0, -1.0]);
        gru.step(&[0.5, 0.5]);
        gru.step(&[0.0, 1.0]);
        gru.reset_state();
        let after_reset = gru.step(&[1.0, -1.0]);
        assert!(
            (first - after_reset).abs() < 1e-6,
            "first={first}, after_reset={after_reset}"
        );
    }

    #[test]
    fn multi_output() {
        let mut gru = make_gru(2, 4, 3, 0.1, 0.1, 0.0, 0.1);
        let mut out = [0.0_f32; 3];
        gru.step_into(&[1.0, 0.5], &mut out);
        assert!((out[0] - out[1]).abs() < 1e-6);
        assert!((out[1] - out[2]).abs() < 1e-6);
    }

    #[test]
    fn nan_propagates() {
        let mut gru = make_gru(2, 2, 1, 0.1, 0.1, 0.0, 0.1);
        let out = gru.step(&[f32::NAN, 1.0]);
        assert!(out.is_nan());
    }

    #[test]
    fn gru_cheaper_than_lstm() {
        // GRU has 3 gates vs LSTM's 4. At same hidden size, GRU stores
        // 3*H*(I+H) gate weights vs LSTM's 4*H*(I+H). Verify by size.
        let i = 4;
        let h = 8;
        let gru = make_gru(i, h, 1, 0.1, 0.1, 0.0, 0.1);
        assert_eq!(gru.weight_ih.len(), 3 * h * i);
        assert_eq!(gru.weight_hh.len(), 3 * h * h);
    }

    #[test]
    fn update_gate_controls_memory() {
        // High update gate bias → z ≈ 1 → h' ≈ h (ignores new input).
        // Low update gate bias → z ≈ 0 → h' ≈ n (forgets old state).
        let h = 2;
        let i = 2;
        let gc = 3 * h;

        // z-biased model (z gate bias = 5.0 → sigmoid ≈ 1)
        let mut bias_z = vec![0.0_f32; gc];
        for k in h..2 * h {
            bias_z[k] = 5.0;
        }
        let mut gru_z = TinyGruF32::from_parts(
            i,
            h,
            1,
            &vec![0.1; gc * i],
            &vec![0.1; gc * h],
            &bias_z,
            &vec![0.0; gc],
            &vec![0.1; h],
            &vec![0.0; 1],
        )
        .unwrap();

        let mut gru_normal = make_gru(i, h, 1, 0.1, 0.1, 0.0, 0.1);

        // Inject state via a step, then feed zeros
        gru_z.step(&[1.0, 1.0]);
        gru_normal.step(&[1.0, 1.0]);

        for _ in 0..10 {
            gru_z.step(&[0.0, 0.0]);
            gru_normal.step(&[0.0, 0.0]);
        }

        // z-biased model should retain more hidden state
        let h_z: f32 = gru_z.hidden_state().iter().map(|v| v.abs()).sum();
        let h_n: f32 = gru_normal.hidden_state().iter().map(|v| v.abs()).sum();
        assert!(
            h_z > h_n,
            "z-biased GRU should retain more state: {h_z} vs {h_n}"
        );
    }

    #[test]
    fn accessors() {
        let gru = make_gru(4, 8, 2, 0.1, 0.1, 0.0, 0.1);
        assert_eq!(gru.input_size(), 4);
        assert_eq!(gru.hidden_size(), 8);
        assert_eq!(gru.output_size(), 2);
        assert_eq!(gru.hidden_state().len(), 8);
    }

    #[test]
    fn validation_rejects_zero_size() {
        let r = TinyGruF32::from_parts(0, 2, 1, &[], &[], &[], &[], &[], &[]);
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_weight_mismatch() {
        let r = TinyGruF32::from_parts(
            2, 2, 1, &[0.0; 11], // wrong: should be 3*2*2 = 12
            &[0.0; 12], &[0.0; 6], &[0.0; 6], &[0.0; 2], &[0.0; 1],
        );
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_non_finite() {
        let mut w = vec![0.1_f32; 12];
        w[5] = f32::INFINITY;
        let r = TinyGruF32::from_parts(
            2, 2, 1, &w, &[0.0; 12], &[0.0; 6], &[0.0; 6], &[0.0; 2], &[0.0; 1],
        );
        assert!(r.is_err());
    }

    #[test]
    #[should_panic(expected = "output_size == 1")]
    fn step_panics_multi_output() {
        let mut gru = make_gru(2, 2, 3, 0.1, 0.1, 0.0, 0.1);
        gru.step(&[1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "input length")]
    fn step_panics_wrong_input_len() {
        let mut gru = make_gru(2, 2, 1, 0.1, 0.1, 0.0, 0.1);
        gru.step(&[1.0]);
    }
}

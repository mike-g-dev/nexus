extern crate alloc;

use alloc::{boxed::Box, vec};

use crate::LoadError;
use crate::dot::matvec_bias_f32;

#[allow(unused_imports)]
use super::{sigmoid_f32, tanh_f32};

/// Single-layer LSTM for streaming temporal inference.
///
/// Four gates (input, forget, cell candidate, output) with hidden and
/// cell state carried between steps. Trained externally (PyTorch),
/// loaded via [`from_parts`](Self::from_parts), one timestep per
/// [`step`](Self::step) call.
///
/// Gate activations are hardcoded: sigmoid for input/forget/output
/// gates, tanh for cell candidate and output nonlinearity. This
/// matches the standard LSTM formulation used by PyTorch's `nn.LSTM`.
///
/// Weight parameters map directly to PyTorch's `nn.LSTM` tensors
/// (gate order: input, forget, cell candidate, output). The output
/// projection is a separate linear layer (`nn.Linear`).
///
/// # Examples
///
/// ```
/// use nexus_inference::TinyLstmF32;
///
/// let weight_ih = vec![0.1_f32; 4 * 8 * 4];
/// let weight_hh = vec![0.1_f32; 4 * 8 * 8];
/// let bias_ih = vec![0.0_f32; 4 * 8];
/// let bias_hh = vec![0.0_f32; 4 * 8];
/// let w_out = vec![0.1_f32; 1 * 8];
/// let b_out = vec![0.0_f32; 1];
///
/// let mut lstm = TinyLstmF32::from_parts(
///     4, 8, 1,
///     &weight_ih, &weight_hh,
///     &bias_ih, &bias_hh,
///     &w_out, &b_out,
/// ).unwrap();
///
/// let output = lstm.step(&[0.5, 1.2, -0.3, 0.8]);
/// ```
#[derive(Debug, Clone)]
pub struct TinyLstmF32 {
    w_gates: Box<[f32]>,
    b_gates: Box<[f32]>,
    w_out: Box<[f32]>,
    b_out: Box<[f32]>,
    h: Box<[f32]>,
    c: Box<[f32]>,
    concat: Box<[f32]>,
    gates: Box<[f32]>,
    input_size: u16,
    hidden_size: u16,
    output_size: u16,
}

impl TinyLstmF32 {
    /// Construct from pre-trained weights.
    ///
    /// Parameters map to PyTorch's `nn.LSTM` + `nn.Linear`:
    ///
    /// - `weight_ih`: input-to-hidden weights, `(4*hidden, input)` row-major.
    ///   Gate order: input, forget, cell candidate, output.
    /// - `weight_hh`: hidden-to-hidden weights, `(4*hidden, hidden)` row-major.
    ///   Same gate order.
    /// - `bias_ih`, `bias_hh`: gate biases, `4*hidden` each.
    /// - `w_out`: output projection, `(output, hidden)` row-major.
    /// - `b_out`: output bias, `output` elements.
    ///
    /// Internally fuses `weight_ih` and `weight_hh` into a single
    /// `(4*hidden, input+hidden)` matrix and pre-sums biases for a
    /// single matrix-vector product per step.
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

        let gate_count = 4 * hidden_size;
        let concat_size = input_size + hidden_size;

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

        let mut w_gates = vec![0.0_f32; gate_count * concat_size];
        for j in 0..gate_count {
            w_gates[j * concat_size..j * concat_size + input_size]
                .copy_from_slice(&weight_ih[j * input_size..(j + 1) * input_size]);
            w_gates[j * concat_size + input_size..(j + 1) * concat_size]
                .copy_from_slice(&weight_hh[j * hidden_size..(j + 1) * hidden_size]);
        }

        let mut b_gates = vec![0.0_f32; gate_count];
        for j in 0..gate_count {
            b_gates[j] = bias_ih[j] + bias_hh[j];
        }

        Ok(Self {
            w_gates: w_gates.into_boxed_slice(),
            b_gates: b_gates.into_boxed_slice(),
            w_out: w_out.into(),
            b_out: b_out.into(),
            h: vec![0.0_f32; hidden_size].into_boxed_slice(),
            c: vec![0.0_f32; hidden_size].into_boxed_slice(),
            concat: vec![0.0_f32; concat_size].into_boxed_slice(),
            gates: vec![0.0_f32; gate_count].into_boxed_slice(),
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
    /// LSTM equations (PyTorch formulation, fused weights):
    /// ```text
    /// gates = W_fused @ [x; h] + b_fused
    /// i = sigmoid(gates[0..H])
    /// f = sigmoid(gates[H..2H])
    /// g = tanh(gates[2H..3H])
    /// o = sigmoid(gates[3H..4H])
    /// c' = f * c + i * g
    /// h' = o * tanh(c')
    /// output = W_out @ h' + b_out
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != input_size` or
    /// `output.len() != output_size`.
    pub fn step_into(&mut self, input: &[f32], output: &mut [f32]) {
        let i = self.input_size as usize;
        let h = self.hidden_size as usize;
        let concat_size = i + h;
        let gate_count = 4 * h;
        assert_eq!(input.len(), i);
        assert_eq!(output.len(), self.output_size as usize);

        self.concat[..i].copy_from_slice(input);
        self.concat[i..concat_size].copy_from_slice(&self.h);

        matvec_bias_f32(
            &self.w_gates,
            &self.concat[..concat_size],
            &self.b_gates,
            &mut self.gates,
            gate_count,
            concat_size,
        );

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            super::avx512_gates::lstm_gates_avx512(&self.gates, &mut self.c, &mut self.h, h);
        }

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma",
            not(target_feature = "avx512f"),
        ))]
        {
            super::avx2_gates::lstm_gates_avx2(&self.gates, &mut self.c, &mut self.h, h);
        }

        #[cfg(not(all(
            target_arch = "x86_64",
            any(
                target_feature = "avx512f",
                all(target_feature = "avx2", target_feature = "fma"),
            )
        )))]
        {
            for k in 0..h {
                let ig = sigmoid_f32(self.gates[k]);
                let fg = sigmoid_f32(self.gates[h + k]);
                let cg = tanh_f32(self.gates[2 * h + k]);
                let og = sigmoid_f32(self.gates[3 * h + k]);

                self.c[k] = fg.mul_add(self.c[k], ig * cg);
                self.h[k] = og * tanh_f32(self.c[k]);
            }
        }

        matvec_bias_f32(
            &self.w_out,
            &self.h[..h],
            &self.b_out,
            output,
            self.output_size as usize,
            h,
        );
    }

    /// Reset hidden and cell state to zeros.
    pub fn reset_state(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }

    /// Current hidden state.
    pub fn hidden_state(&self) -> &[f32] {
        &self.h
    }

    /// Current cell state.
    pub fn cell_state(&self) -> &[f32] {
        &self.c
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

    fn make_lstm(
        input: usize,
        hidden: usize,
        output: usize,
        w_ih_val: f32,
        w_hh_val: f32,
        b_val: f32,
        w_out_val: f32,
    ) -> TinyLstmF32 {
        let gate_count = 4 * hidden;
        let weight_ih = vec![w_ih_val; gate_count * input];
        let weight_hh = vec![w_hh_val; gate_count * hidden];
        let bias_ih = vec![b_val; gate_count];
        let bias_hh = vec![0.0_f32; gate_count];
        let w_out = vec![w_out_val; output * hidden];
        let b_out = vec![0.0_f32; output];
        TinyLstmF32::from_parts(
            input, hidden, output, &weight_ih, &weight_hh, &bias_ih, &bias_hh, &w_out, &b_out,
        )
        .unwrap()
    }

    #[test]
    fn basic_forward_pass() {
        let mut lstm = make_lstm(2, 2, 1, 0.1, 0.1, 0.0, 0.1);
        let out = lstm.step(&[1.0, 0.0]);

        // h_0 = [0, 0], so concat = [1, 0, 0, 0]
        // Each gate row dot concat = 0.1*1 = 0.1 (all rows identical)
        // i = sigmoid(0.1), f = sigmoid(0.1), g = tanh(0.1), o = sigmoid(0.1)
        let sig01 = sigmoid_f32(0.1);
        let tanh01 = tanh_f32(0.1);
        let c1 = sig01 * 0.0 + sig01 * tanh01; // f*c0 + i*g
        let h1 = sig01 * tanh_f32(c1);
        let expected = 0.1 * h1 + 0.1 * h1; // w_out @ h
        assert!(
            (out - expected).abs() < 1e-6,
            "got {out}, expected {expected}"
        );
    }

    #[test]
    fn state_carries_between_steps() {
        let mut lstm = make_lstm(2, 2, 1, 0.1, 0.1, 0.0, 0.1);
        let out1 = lstm.step(&[1.0, 0.0]);
        let out2 = lstm.step(&[1.0, 0.0]);
        // Second step sees h != 0, so output differs from first
        assert!((out1 - out2).abs() > 1e-6);
    }

    #[test]
    fn reset_clears_state() {
        let mut lstm = make_lstm(2, 2, 1, 0.1, 0.1, 0.0, 0.1);
        lstm.step(&[1.0, 0.0]);
        assert!(lstm.hidden_state().iter().any(|&v| v != 0.0));
        assert!(lstm.cell_state().iter().any(|&v| v != 0.0));

        lstm.reset_state();
        assert!(lstm.hidden_state().iter().all(|&v| v == 0.0));
        assert!(lstm.cell_state().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn reset_reproduces_first_output() {
        let mut lstm = make_lstm(2, 4, 1, 0.1, 0.2, 0.0, 0.1);
        let first = lstm.step(&[1.0, -1.0]);
        lstm.step(&[0.5, 0.5]);
        lstm.step(&[0.0, 1.0]);
        lstm.reset_state();
        let after_reset = lstm.step(&[1.0, -1.0]);
        assert!(
            (first - after_reset).abs() < 1e-6,
            "first={first}, after_reset={after_reset}"
        );
    }

    #[test]
    fn multi_output() {
        let mut lstm = make_lstm(2, 4, 3, 0.1, 0.1, 0.0, 0.1);
        let mut out = [0.0_f32; 3];
        lstm.step_into(&[1.0, 0.5], &mut out);
        // All output neurons see the same h with uniform w_out
        assert!((out[0] - out[1]).abs() < 1e-6);
        assert!((out[1] - out[2]).abs() < 1e-6);
    }

    #[test]
    fn nan_propagates() {
        let mut lstm = make_lstm(2, 2, 1, 0.1, 0.1, 0.0, 0.1);
        let out = lstm.step(&[f32::NAN, 1.0]);
        assert!(out.is_nan());
    }

    #[test]
    fn accessors() {
        let lstm = make_lstm(4, 8, 2, 0.1, 0.1, 0.0, 0.1);
        assert_eq!(lstm.input_size(), 4);
        assert_eq!(lstm.hidden_size(), 8);
        assert_eq!(lstm.output_size(), 2);
        assert_eq!(lstm.hidden_state().len(), 8);
        assert_eq!(lstm.cell_state().len(), 8);
    }

    #[test]
    fn validation_rejects_zero_size() {
        let r = TinyLstmF32::from_parts(0, 2, 1, &[], &[], &[], &[], &[], &[]);
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_weight_mismatch() {
        let r = TinyLstmF32::from_parts(
            2, 2, 1, &[0.0; 15], // wrong: should be 4*2*2 = 16
            &[0.0; 16], &[0.0; 8], &[0.0; 8], &[0.0; 2], &[0.0; 1],
        );
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_non_finite() {
        let mut w = vec![0.1_f32; 16];
        w[5] = f32::INFINITY;
        let r = TinyLstmF32::from_parts(
            2, 2, 1, &w, &[0.0; 16], &[0.0; 8], &[0.0; 8], &[0.0; 2], &[0.0; 1],
        );
        assert!(r.is_err());
    }

    #[test]
    fn forget_bias_one_preserves_cell() {
        // With forget gate bias = 1.0, sigmoid(1.0) ≈ 0.731 — cell
        // state decays slowly. With bias = 0, sigmoid(0) = 0.5 — cell
        // decays faster. Verify the bias-1 model retains more cell state.
        let mut lstm_bias1 = {
            let h = 2;
            let i = 2;
            let gc = 4 * h;
            let weight_ih = vec![0.1_f32; gc * i];
            let weight_hh = vec![0.1_f32; gc * h];
            let mut bias_ih = vec![0.0_f32; gc];
            // Set forget gate bias to 1.0 (indices h..2*h)
            for k in h..2 * h {
                bias_ih[k] = 1.0;
            }
            let bias_hh = vec![0.0_f32; gc];
            let w_out = vec![0.1_f32; h];
            let b_out = vec![0.0_f32; 1];
            TinyLstmF32::from_parts(
                i, h, 1, &weight_ih, &weight_hh, &bias_ih, &bias_hh, &w_out, &b_out,
            )
            .unwrap()
        };
        let mut lstm_bias0 = make_lstm(2, 2, 1, 0.1, 0.1, 0.0, 0.1);

        lstm_bias1.step(&[1.0, 0.5]);
        lstm_bias0.step(&[1.0, 0.5]);
        // After one step with input, run several steps with zero input
        for _ in 0..5 {
            lstm_bias1.step(&[0.0, 0.0]);
            lstm_bias0.step(&[0.0, 0.0]);
        }
        let cell_bias1: f32 = lstm_bias1.cell_state().iter().map(|v| v.abs()).sum();
        let cell_bias0: f32 = lstm_bias0.cell_state().iter().map(|v| v.abs()).sum();
        assert!(
            cell_bias1 > cell_bias0,
            "forget bias=1 should retain more cell: {cell_bias1} vs {cell_bias0}"
        );
    }

    #[test]
    #[should_panic(expected = "output_size == 1")]
    fn step_panics_multi_output() {
        let mut lstm = make_lstm(2, 2, 3, 0.1, 0.1, 0.0, 0.1);
        lstm.step(&[1.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn step_panics_wrong_input_len() {
        let mut lstm = make_lstm(2, 2, 1, 0.1, 0.1, 0.0, 0.1);
        lstm.step(&[1.0]);
    }
}

extern crate alloc;

use alloc::{boxed::Box, vec};

use crate::LoadError;
use crate::dot::{dot_f32, matvec_f32};

/// Linear state-space model with diagonal state transition.
///
/// Pre-discretized dynamics applied once per [`step`](Self::step) call:
///
/// ```text
/// h_t = A ⊙ h_{t-1} + B @ u_t
/// y_t = C @ h_t + D @ u_t
/// ```
///
/// A is diagonal (element-wise multiply), so per-step cost is
/// `O(H*I + H + H*O + I*O)` with no transcendentals — purely linear.
/// Fastest temporal model in the crate.
///
/// Use case: long-range regime detection (vol regime over hours,
/// correlation over days) where LSTM forget gates leak signal.
/// Diagonal A makes each state dimension an independent first-order
/// recurrence with its own decay rate.
///
/// Users train in Python (S4, S4D, or custom SSM), discretize to
/// obtain `A_d`, `B_d`, `C`, `D`, and export via safetensors.
/// Missing `D` is treated as zeros (no skip connection).
///
/// # Examples
///
/// ```
/// use nexus_inference::LinearSsmF32;
///
/// let a_diag = vec![0.9_f32; 4];
/// let b = vec![0.1_f32; 4 * 2];
/// let c = vec![0.1_f32; 1 * 4];
/// let d = vec![0.0_f32; 1 * 2];
///
/// let mut ssm = LinearSsmF32::from_parts(
///     &a_diag, &b, &c, &d, 1,
/// ).unwrap();
///
/// let output = ssm.step(&[0.5, 1.0]);
/// ```
#[derive(Debug, Clone)]
pub struct LinearSsmF32 {
    a_diag: Box<[f32]>,
    b: Box<[f32]>,
    c: Box<[f32]>,
    d: Box<[f32]>,
    state: Box<[f32]>,
    scratch: Box<[f32]>,
    input_size: u16,
    hidden_size: u16,
    output_size: u16,
}

impl LinearSsmF32 {
    /// Construct from pre-discretized parameters.
    ///
    /// - `a_diag`: diagonal of A, `[H]`.
    /// - `b`: input-to-state matrix, `[H, I]` row-major.
    /// - `c`: state-to-output matrix, `[O, H]` row-major.
    /// - `d`: skip connection, `[O, I]` row-major. Pass all zeros for no skip.
    /// - `output_size`: number of outputs (`c` and `d` lengths validated against this).
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::Validation`] if dimensions are inconsistent or
    /// any weight is non-finite.
    pub fn from_parts(
        a_diag: &[f32],
        b: &[f32],
        c: &[f32],
        d: &[f32],
        output_size: usize,
    ) -> Result<Self, LoadError> {
        let hidden_size = a_diag.len();
        if hidden_size == 0 || output_size == 0 {
            return Err(LoadError::Validation("sizes must be > 0"));
        }
        if b.is_empty() {
            return Err(LoadError::Validation("b must not be empty"));
        }
        if !b.len().is_multiple_of(hidden_size) {
            return Err(LoadError::Validation("b length must be H * I"));
        }
        let input_size = b.len() / hidden_size;
        if input_size == 0 {
            return Err(LoadError::Validation("sizes must be > 0"));
        }
        if input_size > u16::MAX as usize
            || hidden_size > u16::MAX as usize
            || output_size > u16::MAX as usize
        {
            return Err(LoadError::Validation("size exceeds u16::MAX"));
        }
        if c.len() != output_size * hidden_size {
            return Err(LoadError::Validation("c length must be O * H"));
        }
        if d.len() != output_size * input_size {
            return Err(LoadError::Validation("d length must be O * I"));
        }

        for &w in a_diag.iter().chain(b).chain(c).chain(d) {
            if !w.is_finite() {
                return Err(LoadError::Validation("non-finite weight"));
            }
        }

        Ok(Self {
            a_diag: a_diag.into(),
            b: b.into(),
            c: c.into(),
            d: d.into(),
            state: vec![0.0_f32; hidden_size].into_boxed_slice(),
            scratch: vec![0.0_f32; hidden_size].into_boxed_slice(),
            input_size: input_size as u16,
            hidden_size: hidden_size as u16,
            output_size: output_size as u16,
        })
    }

    /// Process one timestep and return a single scalar output.
    ///
    /// # Panics
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
    /// SSM equations (pre-discretized, diagonal A):
    /// ```text
    /// h = A ⊙ h + B @ u
    /// y = C @ h + D @ u
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != input_size` or
    /// `output.len() != output_size`.
    pub fn step_into(&mut self, input: &[f32], output: &mut [f32]) {
        let i_sz = self.input_size as usize;
        let h_sz = self.hidden_size as usize;
        let o_sz = self.output_size as usize;
        assert_eq!(input.len(), i_sz, "input length must equal input_size");
        assert_eq!(output.len(), o_sz, "output length must equal output_size");

        // h = B @ u (into scratch)
        matvec_f32(&self.b, input, &mut self.scratch, h_sz, i_sz);

        // h = A ⊙ h_prev + scratch
        for k in 0..h_sz {
            self.state[k] = self.a_diag[k].mul_add(self.state[k], self.scratch[k]);
        }

        // y = C @ h + D @ u
        matvec_f32(&self.c, &self.state, output, o_sz, h_sz);
        for j in 0..o_sz {
            let d_row = &self.d[j * i_sz..(j + 1) * i_sz];
            output[j] += dot_f32(d_row, input);
        }
    }

    /// Reset hidden state to zero.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Number of input features per timestep.
    pub fn input_size(&self) -> usize {
        self.input_size as usize
    }

    /// Hidden state dimension.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size as usize
    }

    /// Number of outputs per timestep.
    pub fn output_size(&self) -> usize {
        self.output_size as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ssm_1x2x1() -> LinearSsmF32 {
        // I=2, H=2, O=1
        // A = diag(0.9, 0.8)
        // B = [[0.1, 0.2],
        //      [0.3, 0.4]]  (H=2, I=2)
        // C = [[0.5, 0.6]]  (O=1, H=2)
        // D = [[0.01, 0.02]] (O=1, I=2)
        LinearSsmF32::from_parts(
            &[0.9, 0.8],
            &[0.1, 0.2, 0.3, 0.4],
            &[0.5, 0.6],
            &[0.01, 0.02],
            1,
        )
        .unwrap()
    }

    #[test]
    fn first_step_from_zero_state() {
        let mut ssm = ssm_1x2x1();
        // h_0 = [0, 0]
        // h_1 = A ⊙ [0, 0] + B @ [1, 2] = [0, 0] + [0.5, 1.1] = [0.5, 1.1]
        // y = C @ [0.5, 1.1] + D @ [1, 2]
        //   = 0.5*0.5 + 0.6*1.1 + 0.01*1 + 0.02*2
        //   = 0.25 + 0.66 + 0.01 + 0.04 = 0.96
        let y = ssm.step(&[1.0, 2.0]);
        assert!((y - 0.96).abs() < 1e-6, "got {y}");
    }

    #[test]
    fn second_step_carries_state() {
        let mut ssm = ssm_1x2x1();
        ssm.step(&[1.0, 2.0]); // h = [0.5, 1.1]
        // h_2 = A ⊙ [0.5, 1.1] + B @ [0, 0]
        //      = [0.9*0.5, 0.8*1.1] = [0.45, 0.88]
        // y = C @ [0.45, 0.88] + D @ [0, 0]
        //   = 0.5*0.45 + 0.6*0.88 = 0.225 + 0.528 = 0.753
        let y = ssm.step(&[0.0, 0.0]);
        assert!((y - 0.753).abs() < 1e-5, "got {y}");
    }

    #[test]
    fn reset_clears_state() {
        let mut ssm = ssm_1x2x1();
        let y1 = ssm.step(&[1.0, 2.0]);
        ssm.reset();
        let y2 = ssm.step(&[1.0, 2.0]);
        assert!((y1 - y2).abs() < 1e-7);
    }

    #[test]
    fn zero_d_means_no_skip() {
        let mut ssm = LinearSsmF32::from_parts(
            &[0.9, 0.8],
            &[0.1, 0.2, 0.3, 0.4],
            &[0.5, 0.6],
            &[0.0, 0.0],
            1,
        )
        .unwrap();
        // h_1 = B @ [1, 2] = [0.5, 1.1]
        // y = C @ [0.5, 1.1] + 0 = 0.25 + 0.66 = 0.91
        let y = ssm.step(&[1.0, 2.0]);
        assert!((y - 0.91).abs() < 1e-6, "got {y}");
    }

    #[test]
    fn multi_output() {
        // I=1, H=2, O=2
        let mut ssm = LinearSsmF32::from_parts(
            &[0.5, 0.5],
            &[1.0, 1.0],           // B: H=2, I=1
            &[1.0, 0.0, 0.0, 1.0], // C: O=2, H=2 (identity)
            &[0.0, 0.0],           // D: O=2, I=1
            2,
        )
        .unwrap();
        let mut out = [0.0_f32; 2];
        ssm.step_into(&[3.0], &mut out);
        // h = [0, 0]*0.5 + [3, 3] = [3, 3]
        // y = I @ [3, 3] = [3, 3]
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn state_decays_without_input() {
        let mut ssm = LinearSsmF32::from_parts(
            &[0.5],        // fast decay
            &[1.0],        // I=1, H=1
            &[1.0],        // O=1, H=1 (identity)
            &[0.0],        // no skip
            1,
        )
        .unwrap();
        ssm.step(&[10.0]); // h = 10
        let y1 = ssm.step(&[0.0]); // h = 5
        let y2 = ssm.step(&[0.0]); // h = 2.5
        let y3 = ssm.step(&[0.0]); // h = 1.25
        assert!((y1 - 5.0).abs() < 1e-6);
        assert!((y2 - 2.5).abs() < 1e-6);
        assert!((y3 - 1.25).abs() < 1e-6);
    }

    #[test]
    fn accessors() {
        let ssm = ssm_1x2x1();
        assert_eq!(ssm.input_size(), 2);
        assert_eq!(ssm.hidden_size(), 2);
        assert_eq!(ssm.output_size(), 1);
    }

    #[test]
    fn rejects_empty() {
        assert!(LinearSsmF32::from_parts(&[], &[1.0], &[1.0], &[0.0], 1).is_err());
    }

    #[test]
    fn rejects_mismatched_c() {
        assert!(LinearSsmF32::from_parts(
            &[0.9],
            &[0.1],        // H=1, I=1
            &[0.5, 0.6],  // expects O*H=1, got 2
            &[0.0],
            1,
        )
        .is_err());
    }

    #[test]
    fn rejects_non_finite() {
        assert!(LinearSsmF32::from_parts(
            &[f32::NAN],
            &[0.1],
            &[0.5],
            &[0.0],
            1,
        )
        .is_err());
    }

    #[test]
    #[should_panic(expected = "input length must equal input_size")]
    fn wrong_input_panics() {
        let mut ssm = ssm_1x2x1();
        ssm.step(&[1.0]); // expects 2 inputs
    }

    #[test]
    #[should_panic(expected = "output length must equal output_size")]
    fn wrong_output_panics() {
        let mut ssm = ssm_1x2x1();
        let mut out = [0.0_f32; 3];
        ssm.step_into(&[1.0, 2.0], &mut out);
    }

    #[test]
    #[should_panic(expected = "step() requires output_size == 1")]
    fn step_multi_output_panics() {
        let mut ssm = LinearSsmF32::from_parts(
            &[0.5, 0.5],
            &[1.0, 1.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[0.0, 0.0],
            2,
        )
        .unwrap();
        ssm.step(&[1.0]);
    }
}

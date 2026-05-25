use crate::LoadError;
use crate::activation::{Activation, activate_f32};
use crate::dot::{dot_f32, dot4_f32, matvec_bias_f32};

#[cfg(all(
    target_arch = "x86_64",
    any(
        target_feature = "avx512f",
        all(target_feature = "avx2", target_feature = "fma"),
    )
))]
#[inline(never)]
pub(super) fn conv_tiled_simd(
    w_conv: &[f32],
    b_conv: &[f32],
    lin: &[f32],
    filter_scratch: &mut [f32],
    conv_len: usize,
    filters_4: usize,
    activation: Activation,
) -> usize {
    use crate::activation::simd::{activate_4wide, activate_8wide};
    use crate::dot::{dot4_f32_m128, dot8_f32_m256};
    use core::arch::x86_64::*;

    let filters_8 = filters_4 & !7;
    let mut f = 0;
    // SAFETY: cfg guarantees SIMD availability.
    // f + N <= filters_4 within respective loops; bias/scratch accesses are in bounds.
    unsafe {
        if conv_len >= 32 {
            while f < filters_8 {
                let rows = &w_conv[f * conv_len..(f + 8) * conv_len];
                let dots = dot8_f32_m256(rows, lin);
                let bias_v = _mm256_loadu_ps(b_conv.as_ptr().add(f));
                let with_bias = _mm256_add_ps(dots, bias_v);
                match activate_8wide(with_bias, activation) {
                    Some(activated) => {
                        _mm256_storeu_ps(filter_scratch.as_mut_ptr().add(f), activated)
                    }
                    None => return f,
                }
                f += 8;
            }
        }

        while f < filters_4 {
            let rows = &w_conv[f * conv_len..(f + 4) * conv_len];
            let dots = dot4_f32_m128(rows, lin);
            let bias_v = _mm_loadu_ps(b_conv.as_ptr().add(f));
            let with_bias = _mm_add_ps(dots, bias_v);
            match activate_4wide(with_bias, activation) {
                Some(activated) => _mm_storeu_ps(filter_scratch.as_mut_ptr().add(f), activated),
                None => return f,
            }
            f += 4;
        }
    }
    f
}

/// Streaming causal 1D convolution.
///
/// Maintains a circular buffer of the last `kernel_size` inputs. Each
/// step convolves the buffer with learned filters, applies an
/// activation, and projects to output. Causal: only past and current
/// inputs are used, no future leakage.
///
/// # Examples
///
/// ```
/// use nexus_inference::{Activation, Causal1dConv};
///
/// // 2 input channels, kernel 3, 4 filters, 1 output
/// let w_conv = vec![0.1_f32; 4 * 3 * 2];
/// let b_conv = vec![0.0_f32; 4];
/// let w_out = vec![0.1_f32; 1 * 4];
/// let b_out = vec![0.0_f32; 1];
///
/// let mut conv = Causal1dConv::from_parts(
///     2, 3, 4, 1,
///     &w_conv, &b_conv,
///     &w_out, &b_out,
///     Activation::Relu,
/// ).unwrap();
///
/// let output = conv.predict(&[0.5, 1.0]);
/// assert!(!conv.is_primed()); // needs 3 steps to fill kernel buffer
/// conv.predict(&[0.2, 0.3]);
/// conv.predict(&[0.1, 0.4]);
/// assert!(conv.is_primed());
/// ```
#[derive(Debug, Clone)]
pub struct Causal1dConv {
    w_conv: Box<[f32]>,
    b_conv: Box<[f32]>,
    w_out: Box<[f32]>,
    b_out: Box<[f32]>,
    buffer: Box<[f32]>,
    lin_buf: Box<[f32]>,
    filter_scratch: Box<[f32]>,
    write_idx: u16,
    step_count: u32,
    input_ch: u16,
    kernel_size: u16,
    filters: u16,
    output_size: u16,
    activation: Activation,
}

impl Causal1dConv {
    /// Construct from pre-trained weights.
    ///
    /// - `input_ch`: number of input channels per timestep.
    /// - `kernel_size`: temporal kernel width (number of past timesteps).
    /// - `filters`: number of convolution filters (output channels).
    /// - `output_size`: final output dimension after projection.
    /// - `w_conv`: convolution weights, `(filters, kernel_size, input_ch)`
    ///   stored as a flat row-major array. For filter `f`, kernel position
    ///   `k`, channel `c`: index is `f * kernel_size * input_ch + k * input_ch + c`.
    /// - `b_conv`: convolution bias, `filters` elements.
    /// - `w_out`: output projection, `(output_size, filters)` row-major.
    /// - `b_out`: output projection bias, `output_size` elements.
    /// - `activation`: applied to convolution output before projection.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        input_ch: usize,
        kernel_size: usize,
        filters: usize,
        output_size: usize,
        w_conv: &[f32],
        b_conv: &[f32],
        w_out: &[f32],
        b_out: &[f32],
        activation: Activation,
    ) -> Result<Self, LoadError> {
        if input_ch == 0 || kernel_size == 0 || filters == 0 || output_size == 0 {
            return Err(LoadError::Validation("sizes must be > 0"));
        }
        if input_ch > u16::MAX as usize
            || kernel_size > u16::MAX as usize
            || filters > u16::MAX as usize
            || output_size > u16::MAX as usize
        {
            return Err(LoadError::Validation("size exceeds u16::MAX"));
        }

        if w_conv.len() != filters * kernel_size * input_ch {
            return Err(LoadError::Validation("w_conv length mismatch"));
        }
        if b_conv.len() != filters {
            return Err(LoadError::Validation("b_conv length mismatch"));
        }
        if w_out.len() != output_size * filters {
            return Err(LoadError::Validation("w_out length mismatch"));
        }
        if b_out.len() != output_size {
            return Err(LoadError::Validation("b_out length mismatch"));
        }

        for &w in w_conv.iter().chain(b_conv).chain(w_out).chain(b_out) {
            if !w.is_finite() {
                return Err(LoadError::Validation("non-finite weight"));
            }
        }

        Ok(Self {
            w_conv: w_conv.into(),
            b_conv: b_conv.into(),
            w_out: w_out.into(),
            b_out: b_out.into(),
            buffer: vec![0.0_f32; kernel_size * input_ch].into_boxed_slice(),
            lin_buf: vec![0.0_f32; kernel_size * input_ch].into_boxed_slice(),
            filter_scratch: vec![0.0_f32; filters].into_boxed_slice(),
            write_idx: 0,
            step_count: 0,
            input_ch: input_ch as u16,
            kernel_size: kernel_size as u16,
            filters: filters as u16,
            output_size: output_size as u16,
            activation,
        })
    }

    /// Process one timestep and return a single scalar output.
    ///
    /// Panics if `output_size != 1` or `input.len() != input_ch`.
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
    /// Panics if `input.len() != input_ch` or
    /// `output.len() != output_size`.
    pub fn predict_into(&mut self, input: &[f32], output: &mut [f32]) {
        let ch = self.input_ch as usize;
        let k_size = self.kernel_size as usize;
        let n_filters = self.filters as usize;
        let conv_len = k_size * ch;
        assert_eq!(input.len(), ch);
        assert_eq!(output.len(), self.output_size as usize);

        let wi = self.write_idx as usize;
        self.buffer[wi * ch..(wi + 1) * ch].copy_from_slice(input);

        // Linearize circular buffer: position kk=0 is current (wi),
        // kk=1 is previous, etc. Matches w_conv layout per filter.
        for kk in 0..k_size {
            let buf_pos = (wi + k_size - kk) % k_size;
            self.lin_buf[kk * ch..(kk + 1) * ch]
                .copy_from_slice(&self.buffer[buf_pos * ch..(buf_pos + 1) * ch]);
        }

        // Tiled convolution: 4 filters at a time.
        let lin = &self.lin_buf[..conv_len];
        let filters_4 = n_filters & !3;

        #[cfg(all(
            target_arch = "x86_64",
            any(
                target_feature = "avx512f",
                all(target_feature = "avx2", target_feature = "fma"),
            )
        ))]
        let mut f = conv_tiled_simd(
            &self.w_conv,
            &self.b_conv,
            lin,
            &mut self.filter_scratch,
            conv_len,
            filters_4,
            self.activation,
        );
        #[cfg(not(all(
            target_arch = "x86_64",
            any(
                target_feature = "avx512f",
                all(target_feature = "avx2", target_feature = "fma"),
            )
        )))]
        let mut f = 0usize;

        while f < filters_4 {
            let rows = &self.w_conv[f * conv_len..(f + 4) * conv_len];
            let dots = dot4_f32(rows, lin);
            self.filter_scratch[f] = activate_f32(self.b_conv[f] + dots[0], self.activation);
            self.filter_scratch[f + 1] =
                activate_f32(self.b_conv[f + 1] + dots[1], self.activation);
            self.filter_scratch[f + 2] =
                activate_f32(self.b_conv[f + 2] + dots[2], self.activation);
            self.filter_scratch[f + 3] =
                activate_f32(self.b_conv[f + 3] + dots[3], self.activation);
            f += 4;
        }
        while f < n_filters {
            let row = &self.w_conv[f * conv_len..(f + 1) * conv_len];
            self.filter_scratch[f] =
                activate_f32(self.b_conv[f] + dot_f32(row, lin), self.activation);
            f += 1;
        }

        matvec_bias_f32(
            &self.w_out,
            &self.filter_scratch[..n_filters],
            &self.b_out,
            output,
            self.output_size as usize,
            n_filters,
        );

        self.write_idx = ((wi + 1) % k_size) as u16;
        self.step_count = self.step_count.saturating_add(1);
    }

    /// Reset circular buffer and step counter.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_idx = 0;
        self.step_count = 0;
    }

    /// Whether the buffer has received at least `kernel_size` inputs.
    ///
    /// Before priming, early outputs are computed from zero-padded
    /// history (standard causal conv behavior).
    pub fn is_primed(&self) -> bool {
        self.step_count >= self.kernel_size as u32
    }

    /// Number of input channels per timestep.
    pub fn n_inputs(&self) -> usize {
        self.input_ch as usize
    }

    /// Temporal kernel width.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size as usize
    }

    /// Number of convolution filters.
    pub fn n_filters(&self) -> usize {
        self.filters as usize
    }

    /// Number of output values per timestep.
    pub fn n_outputs(&self) -> usize {
        self.output_size as usize
    }

    /// Activation function applied to convolution outputs.
    pub fn activation(&self) -> Activation {
        self.activation
    }
}

impl crate::Model for Causal1dConv {
    fn predict(&mut self, input: &[f32]) -> f32 {
        Causal1dConv::predict(self, input)
    }
    fn predict_into(&mut self, input: &[f32], output: &mut [f32]) {
        Causal1dConv::predict_into(self, input, output);
    }
    fn n_outputs(&self) -> usize {
        Causal1dConv::n_outputs(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_conv(
        input_ch: usize,
        kernel: usize,
        filters: usize,
        output: usize,
        w_val: f32,
    ) -> Causal1dConv {
        let w_conv = vec![w_val; filters * kernel * input_ch];
        let b_conv = vec![0.0_f32; filters];
        let w_out = vec![w_val; output * filters];
        let b_out = vec![0.0_f32; output];
        Causal1dConv::from_parts(
            input_ch,
            kernel,
            filters,
            output,
            &w_conv,
            &b_conv,
            &w_out,
            &b_out,
            Activation::Identity,
        )
        .unwrap()
    }

    #[test]
    fn priming_sequence() {
        let mut conv = make_conv(1, 3, 1, 1, 1.0);
        assert!(!conv.is_primed());

        conv.predict(&[1.0]);
        assert!(!conv.is_primed()); // 1 of 3

        conv.predict(&[1.0]);
        assert!(!conv.is_primed()); // 2 of 3

        conv.predict(&[1.0]);
        assert!(conv.is_primed()); // 3 of 3
    }

    #[test]
    fn known_convolution() {
        // 1 channel, kernel 3, 1 filter, 1 output, identity activation
        // w_conv = [0.5, 0.3, 0.1] (kernel positions 0, 1, 2)
        // k=0 is current input, k=1 is previous, k=2 is two steps ago
        let w_conv = [0.5_f32, 0.3, 0.1];
        let b_conv = [0.0_f32];
        let w_out = [1.0_f32]; // pass-through
        let b_out = [0.0_f32];
        let mut conv = Causal1dConv::from_parts(
            1,
            3,
            1,
            1,
            &w_conv,
            &b_conv,
            &w_out,
            &b_out,
            Activation::Identity,
        )
        .unwrap();

        // Step 1: buffer = [1, 0, 0] (zero-padded)
        let out1 = conv.predict(&[1.0]);
        // k=0: w[0]*buf[current]=0.5*1=0.5, k=1: w[1]*buf[prev]=0.3*0=0, k=2: w[2]*0=0
        assert!((out1 - 0.5).abs() < 1e-6, "step1: {out1}");

        // Step 2: buffer = [1, 2, 0]
        let out2 = conv.predict(&[2.0]);
        // k=0: 0.5*2=1.0, k=1: 0.3*1=0.3, k=2: 0.1*0=0
        assert!((out2 - 1.3).abs() < 1e-6, "step2: {out2}");

        // Step 3: buffer = [1, 2, 3], primed
        let out3 = conv.predict(&[3.0]);
        // k=0: 0.5*3=1.5, k=1: 0.3*2=0.6, k=2: 0.1*1=0.1
        assert!((out3 - 2.2).abs() < 1e-6, "step3: {out3}");

        // Step 4: buffer overwrites oldest → [4, 2, 3]
        let out4 = conv.predict(&[4.0]);
        // k=0: 0.5*4=2.0, k=1: 0.3*3=0.9, k=2: 0.1*2=0.2
        assert!((out4 - 3.1).abs() < 1e-6, "step4: {out4}");
    }

    #[test]
    fn multi_channel() {
        // 2 channels, kernel 2, 1 filter, 1 output
        // w_conv = [0.1, 0.2, 0.3, 0.4] → k=0: [0.1, 0.2], k=1: [0.3, 0.4]
        let w_conv = [0.1_f32, 0.2, 0.3, 0.4];
        let b_conv = [0.0_f32];
        let w_out = [1.0_f32];
        let b_out = [0.0_f32];
        let mut conv = Causal1dConv::from_parts(
            2,
            2,
            1,
            1,
            &w_conv,
            &b_conv,
            &w_out,
            &b_out,
            Activation::Identity,
        )
        .unwrap();

        // Step 1: x=[1, 2], buffer=[[1,2],[0,0]]
        let out1 = conv.predict(&[1.0, 2.0]);
        // k=0: 0.1*1+0.2*2=0.5, k=1: 0.3*0+0.4*0=0
        assert!((out1 - 0.5).abs() < 1e-6, "step1: {out1}");

        // Step 2: x=[3, 4], buffer=[[1,2],[3,4]]
        let out2 = conv.predict(&[3.0, 4.0]);
        // k=0: 0.1*3+0.2*4=1.1, k=1: 0.3*1+0.4*2=1.1
        assert!((out2 - 2.2).abs() < 1e-6, "step2: {out2}");
    }

    #[test]
    fn relu_activation() {
        // Negative conv output should be zeroed by relu
        let w_conv = [-1.0_f32]; // kernel=1, ch=1, filter=1
        let b_conv = [0.0_f32];
        let w_out = [1.0_f32];
        let b_out = [0.0_f32];
        let mut conv = Causal1dConv::from_parts(
            1,
            1,
            1,
            1,
            &w_conv,
            &b_conv,
            &w_out,
            &b_out,
            Activation::Relu,
        )
        .unwrap();

        let out = conv.predict(&[5.0]);
        // conv = -1*5 = -5, relu(-5) = 0, output = 1*0 = 0
        assert!((out - 0.0).abs() < 1e-6, "{out}");

        let out2 = conv.predict(&[-3.0]);
        // conv = -1*(-3) = 3, relu(3) = 3, output = 1*3 = 3
        assert!((out2 - 3.0).abs() < 1e-6, "{out2}");
    }

    #[test]
    fn reset_clears_buffer() {
        let mut conv = make_conv(1, 3, 1, 1, 0.5);
        conv.predict(&[1.0]);
        conv.predict(&[2.0]);
        assert!(!conv.is_primed());

        conv.reset();
        assert!(!conv.is_primed());

        // After reset, same input should produce same output as fresh
        let mut fresh = make_conv(1, 3, 1, 1, 0.5);
        let out_reset = conv.predict(&[1.0]);
        let out_fresh = fresh.predict(&[1.0]);
        assert!(
            (out_reset - out_fresh).abs() < 1e-6,
            "reset={out_reset}, fresh={out_fresh}"
        );
    }

    #[test]
    fn multi_filter_multi_output() {
        let mut conv = make_conv(1, 2, 4, 2, 0.1);
        let mut out = [0.0_f32; 2];
        conv.predict_into(&[1.0], &mut out);
        // All filters have same weights, so filter outputs are equal.
        // All output neurons have same weights, so outputs are equal.
        assert!((out[0] - out[1]).abs() < 1e-6);
    }

    #[test]
    fn nan_propagates() {
        let mut conv = make_conv(1, 2, 1, 1, 0.1);
        let out = conv.predict(&[f32::NAN]);
        assert!(out.is_nan());
    }

    #[test]
    fn accessors() {
        let conv = make_conv(3, 5, 8, 2, 0.1);
        assert_eq!(conv.n_inputs(), 3);
        assert_eq!(conv.kernel_size(), 5);
        assert_eq!(conv.n_filters(), 8);
        assert_eq!(conv.n_outputs(), 2);
        assert!(matches!(conv.activation(), Activation::Identity));
    }

    #[test]
    fn validation_rejects_zero_size() {
        let r = Causal1dConv::from_parts(0, 3, 4, 1, &[], &[], &[], &[], Activation::Relu);
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_weight_mismatch() {
        let r = Causal1dConv::from_parts(
            2,
            3,
            4,
            1,
            &[0.0; 23], // wrong: should be 4*3*2 = 24
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 1],
            Activation::Relu,
        );
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_non_finite() {
        let mut w = vec![0.1_f32; 4];
        w[2] = f32::NAN;
        let r = Causal1dConv::from_parts(
            1,
            2,
            2,
            1,
            &w,
            &[0.0; 2],
            &[0.0; 2],
            &[0.0; 1],
            Activation::Relu,
        );
        assert!(r.is_err());
    }

    #[test]
    #[should_panic(expected = "output_size == 1")]
    fn predict_panics_multi_output() {
        let mut conv = make_conv(1, 2, 2, 3, 0.1);
        conv.predict(&[1.0]);
    }

    #[test]
    #[should_panic]
    fn predict_panics_wrong_input_len() {
        let mut conv = make_conv(2, 2, 1, 1, 0.1);
        conv.predict(&[1.0]);
    }
}

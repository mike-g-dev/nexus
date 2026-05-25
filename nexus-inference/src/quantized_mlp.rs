#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, vec, vec::Vec};

#[cfg(feature = "alloc")]
use crate::LoadError;
#[cfg(feature = "alloc")]
use crate::activation::{Activation, activate_f32};

#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
struct QuantLayer {
    w_i8: Box<[i8]>,
    bias_f32: Box<[f32]>,
    w_scale: f32,
    w_zero_point: i8,
    input_scale: f32,
    input_zero_point: i8,
    row_sum: Box<[i32]>,
    in_size: u16,
    out_size: u16,
}

/// Int8 quantized multi-layer perceptron for fast inference.
///
/// Weights are stored as `i8` with per-layer affine quantization
/// (scale + zero_point). The forward pass quantizes f32 inputs to i8,
/// performs integer matmul with i32 accumulation, dequantizes back to
/// f32, applies activation, and repeats for each layer. The final
/// layer outputs f32 directly.
///
/// Supports both symmetric (zero_point = 0) and asymmetric
/// quantization, matching PyTorch's `torch.ao.quantization` output.
///
/// # Examples
///
/// ```
/// use nexus_inference::{Activation, QuantizedMlpI8};
///
/// // 2-input, 4-hidden, 1-output with symmetric quantization
/// let w0 = vec![10_i8; 4 * 2];  // layer 0: [4, 2]
/// let b0 = vec![0.0_f32; 4];
/// let w1 = vec![5_i8; 1 * 4];   // layer 1: [1, 4]
/// let b1 = vec![0.0_f32; 1];
///
/// let mut model = QuantizedMlpI8::from_parts(
///     &[&w0, &w1],
///     &[&b0, &b1],
///     &[0.01, 0.01],  // weight scales
///     &[0, 0],         // weight zero points (symmetric)
///     &[0.05, 0.05],   // input scales
///     &[0, 0],          // input zero points (symmetric)
///     Activation::Relu,
/// ).unwrap();
///
/// let output = model.predict(&[1.0, 2.0]);
/// ```
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct QuantizedMlpI8 {
    layers: Box<[QuantLayer]>,
    scratch_f32: Box<[f32]>,
    scratch_i8: Box<[i8]>,
    scratch_i32: Box<[i32]>,
    input_size: u16,
    output_size: u16,
    activation: Activation,
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn dot_i8_i32(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s0 = 0_i32;
    let mut s1 = 0_i32;
    let mut s2 = 0_i32;
    let mut s3 = 0_i32;
    let n4 = a.len() & !3;
    for i in (0..n4).step_by(4) {
        s0 += a[i] as i32 * b[i] as i32;
        s1 += a[i + 1] as i32 * b[i + 1] as i32;
        s2 += a[i + 2] as i32 * b[i + 2] as i32;
        s3 += a[i + 3] as i32 * b[i + 3] as i32;
    }
    for i in n4..a.len() {
        s0 += a[i] as i32 * b[i] as i32;
    }
    (s0 + s2) + (s1 + s3)
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn dot4_i8_i32(rows: &[i8], input: &[i8]) -> [i32; 4] {
    let n = input.len();
    [
        dot_i8_i32(&rows[..n], input),
        dot_i8_i32(&rows[n..2 * n], input),
        dot_i8_i32(&rows[2 * n..3 * n], input),
        dot_i8_i32(&rows[3 * n..4 * n], input),
    ]
}

// `_mm256_maddubs_epi16` saturates its i16 pairwise sums. With the XOR trick
// (i8→u8 via +128), two adjacent large products can exceed i16 range. This is
// accepted — matches FBGEMM/oneDNN/TVM behavior. Quantized inference is
// inherently approximate; the saturation delta is negligible vs quantization
// error for well-calibrated models (PyTorch torch.ao.quantization output).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(never)]
fn matvec_i8_i32_simd(
    weights: &[i8],
    input: &[i8],
    output: &mut [i32],
    out_size: usize,
    in_size: usize,
) -> usize {
    use core::arch::x86_64::*;
    let mut j = 0;
    let in_32 = in_size & !31;

    // SAFETY: cfg guarantees AVX2 availability.
    // All pointer accesses are bounded by out_size * in_size (weights),
    // in_size (input), and out_size (output).
    unsafe {
        while j < out_size {
            let row = &weights[j * in_size..];
            let mut acc = _mm256_setzero_si256();
            let ones_16 = _mm256_set1_epi16(1);

            let mut i = 0;
            while i < in_32 {
                // Load 32 bytes of weights (i8) and input (i8).
                // _mm256_maddubs_epi16 requires first arg unsigned, second signed.
                // We treat input as unsigned: i8 → u8 by XOR with 0x80.
                // Correction: sum(w * (x+128)) = sum(w*x) + 128*sum(w)
                // The 128*sum(w) correction is applied via row_sum after the loop.
                let w = _mm256_loadu_si256(row.as_ptr().add(i) as *const _);
                let x_i8 = _mm256_loadu_si256(input.as_ptr().add(i) as *const _);
                let x_u8 = _mm256_xor_si256(x_i8, _mm256_set1_epi8(-128));

                // maddubs: u8*i8 → i16 with pairwise horizontal add
                let prod16 = _mm256_maddubs_epi16(x_u8, w);
                // madd: i16*1 → i32 with pairwise horizontal add
                let prod32 = _mm256_madd_epi16(prod16, ones_16);
                acc = _mm256_add_epi32(acc, prod32);

                i += 32;
            }

            // Horizontal sum of 8 i32 lanes
            let hi128 = _mm256_extracti128_si256(acc, 1);
            let lo128 = _mm256_castsi256_si128(acc);
            let sum128 = _mm_add_epi32(lo128, hi128);
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            let sum64 = _mm_add_epi32(sum128, hi64);
            let hi32 = _mm_shuffle_epi32(sum64, 0x01);
            let sum32 = _mm_add_epi32(sum64, hi32);
            let mut dot = _mm_cvtsi128_si32(sum32);

            // Scalar remainder — must match SIMD path: multiply (input+128) * weight
            // so the 128*row_sum correction in predict_into applies uniformly.
            while i < in_size {
                dot += (input[i] as i32 + 128) * row[i] as i32;
                i += 1;
            }

            output[j] = dot;
            j += 1;
        }
    }
    j
}

#[inline]
fn quantize_f32_to_i8(src: &[f32], dst: &mut [i8], inv_scale: f32, zero_point: i8) {
    let zp = zero_point as i32;
    for (x, q) in src.iter().zip(dst.iter_mut()) {
        let v = (*x * inv_scale).round() as i32 + zp;
        *q = v.clamp(-128, 127) as i8;
    }
}

#[cfg(feature = "alloc")]
impl QuantizedMlpI8 {
    /// Construct from pre-trained quantized weights.
    ///
    /// Each layer k has:
    /// - `layers_w[k]`: i8 weights, `[out_size, in_size]` row-major
    /// - `layers_b[k]`: f32 biases, `[out_size]`
    /// - `w_scales[k]`: weight quantization scale
    /// - `w_zero_points[k]`: weight zero point (0 for symmetric)
    /// - `input_scales[k]`: input/activation quantization scale
    /// - `input_zero_points[k]`: input/activation zero point
    ///
    /// Layer sizes are inferred from weight dimensions. Layer 0 input
    /// size is `layers_w[0].len() / layers_b[0].len()`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        layers_w: &[&[i8]],
        layers_b: &[&[f32]],
        w_scales: &[f32],
        w_zero_points: &[i8],
        input_scales: &[f32],
        input_zero_points: &[i8],
        activation: Activation,
    ) -> Result<Self, LoadError> {
        let num_layers = layers_w.len();
        if num_layers == 0 {
            return Err(LoadError::Validation("must have at least 1 layer"));
        }
        if layers_b.len() != num_layers
            || w_scales.len() != num_layers
            || w_zero_points.len() != num_layers
            || input_scales.len() != num_layers
            || input_zero_points.len() != num_layers
        {
            return Err(LoadError::Validation(
                "all per-layer arrays must have the same length",
            ));
        }

        #[cfg(not(any(feature = "std", feature = "libm")))]
        match activation {
            Activation::Tanh
            | Activation::Sigmoid
            | Activation::Elu(_)
            | Activation::Gelu
            | Activation::Swish => {
                return Err(LoadError::Validation(
                    "Tanh/Sigmoid/Elu/Gelu/Swish require std or libm feature",
                ));
            }
            _ => {}
        }

        let mut layers = Vec::with_capacity(num_layers);
        let mut prev_out_size: Option<usize> = None;
        let mut max_dim = 0_usize;

        for k in 0..num_layers {
            let out_size = layers_b[k].len();
            if out_size == 0 {
                return Err(LoadError::Validation("layer output size must be > 0"));
            }
            if layers_w[k].is_empty() || !layers_w[k].len().is_multiple_of(out_size) {
                return Err(LoadError::Validation(
                    "weight length not divisible by output size",
                ));
            }
            let in_size = layers_w[k].len() / out_size;
            if in_size == 0 {
                return Err(LoadError::Validation("layer input size must be > 0"));
            }

            if let Some(prev) = prev_out_size
                && in_size != prev
            {
                return Err(LoadError::Validation(
                    "layer input size must match previous layer output size",
                ));
            }

            if in_size > u16::MAX as usize || out_size > u16::MAX as usize {
                return Err(LoadError::Validation("layer size exceeds u16::MAX"));
            }

            if !w_scales[k].is_finite() || w_scales[k] <= 0.0 {
                return Err(LoadError::Validation("w_scale must be finite and positive"));
            }
            if !input_scales[k].is_finite() || input_scales[k] <= 0.0 {
                return Err(LoadError::Validation(
                    "input_scale must be finite and positive",
                ));
            }

            for &b in layers_b[k] {
                if !b.is_finite() {
                    return Err(LoadError::Validation("non-finite bias"));
                }
            }

            // Precompute row sums for zero-point correction.
            // When input x_i8 is XOR'd with 0x80 to produce u8 for maddubs,
            // the result includes an extra 128 * sum(w_row) term.
            // Also, for asymmetric input: sum(w * (x - x_zp)) = sum(w*x) - x_zp * sum(w).
            // Both corrections use the same row_sum precomputation.
            let mut row_sum = vec![0_i32; out_size];
            for j in 0..out_size {
                let mut s = 0_i32;
                for i in 0..in_size {
                    s += layers_w[k][j * in_size + i] as i32;
                }
                row_sum[j] = s;
            }

            if in_size > max_dim {
                max_dim = in_size;
            }
            if out_size > max_dim {
                max_dim = out_size;
            }

            layers.push(QuantLayer {
                w_i8: layers_w[k].into(),
                bias_f32: layers_b[k].into(),
                w_scale: w_scales[k],
                w_zero_point: w_zero_points[k],
                input_scale: input_scales[k],
                input_zero_point: input_zero_points[k],
                row_sum: row_sum.into_boxed_slice(),
                in_size: in_size as u16,
                out_size: out_size as u16,
            });

            prev_out_size = Some(out_size);
        }

        let input_size = layers[0].in_size;
        let output_size = layers[num_layers - 1].out_size;

        Ok(Self {
            layers: layers.into_boxed_slice(),
            scratch_f32: vec![0.0_f32; max_dim].into_boxed_slice(),
            scratch_i8: vec![0_i8; max_dim].into_boxed_slice(),
            scratch_i32: vec![0_i32; max_dim].into_boxed_slice(),
            input_size,
            output_size,
            activation,
        })
    }

    /// Single-output prediction.
    ///
    /// Panics if `output_size != 1`.
    pub fn predict(&mut self, input: &[f32]) -> f32 {
        assert_eq!(
            self.output_size, 1,
            "predict() requires output_size == 1, use predict_into()"
        );
        let mut out = [0.0_f32];
        self.predict_into(input, &mut out);
        out[0]
    }

    /// General prediction (multi-output).
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != input_size()` or
    /// `output.len() != output_size()`.
    pub fn predict_into(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.input_size as usize);
        assert_eq!(output.len(), self.output_size as usize);

        let n_layers = self.layers.len();

        // First layer reads from caller's input
        self.scratch_f32[..input.len()].copy_from_slice(input);

        for layer_idx in 0..n_layers {
            let in_size = self.layers[layer_idx].in_size as usize;
            let out_size = self.layers[layer_idx].out_size as usize;
            let is_last = layer_idx == n_layers - 1;

            let inv_input_scale = 1.0 / self.layers[layer_idx].input_scale;
            let input_zp = self.layers[layer_idx].input_zero_point;
            let w_zp = self.layers[layer_idx].w_zero_point as i32;
            let combined_scale =
                self.layers[layer_idx].w_scale * self.layers[layer_idx].input_scale;

            // Step 1: Quantize f32 → i8
            quantize_f32_to_i8(
                &self.scratch_f32[..in_size],
                &mut self.scratch_i8[..in_size],
                inv_input_scale,
                input_zp,
            );

            // Step 2: Integer matmul → i32 accumulation
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                matvec_i8_i32_simd(
                    &self.layers[layer_idx].w_i8,
                    &self.scratch_i8[..in_size],
                    &mut self.scratch_i32[..out_size],
                    out_size,
                    in_size,
                );
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            {
                let out_4 = out_size & !3;
                let mut j = 0;
                while j < out_4 {
                    let rows = &self.layers[layer_idx].w_i8[j * in_size..(j + 4) * in_size];
                    let dots = dot4_i8_i32(rows, &self.scratch_i8[..in_size]);
                    self.scratch_i32[j] = dots[0];
                    self.scratch_i32[j + 1] = dots[1];
                    self.scratch_i32[j + 2] = dots[2];
                    self.scratch_i32[j + 3] = dots[3];
                    j += 4;
                }
                while j < out_size {
                    let row = &self.layers[layer_idx].w_i8[j * in_size..(j + 1) * in_size];
                    self.scratch_i32[j] = dot_i8_i32(row, &self.scratch_i8[..in_size]);
                    j += 1;
                }
            }

            // Step 3: Zero-point correction + dequantize + bias + activation
            let input_zp_i32 = input_zp as i32;
            let dst = if is_last {
                &mut *output
            } else {
                &mut self.scratch_f32[..out_size]
            };

            for j in 0..out_size {
                let mut acc = self.scratch_i32[j];

                // Correct for SIMD u8 trick: subtract 128 * row_sum
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                {
                    acc -= 128 * self.layers[layer_idx].row_sum[j];
                }

                // Correct for input zero point: subtract input_zp * row_sum
                if input_zp_i32 != 0 {
                    acc -= input_zp_i32 * self.layers[layer_idx].row_sum[j];
                }

                // Correct for weight zero point: subtract w_zp * sum(input)
                if w_zp != 0 {
                    let mut input_sum = 0_i32;
                    for i in 0..in_size {
                        input_sum += self.scratch_i8[i] as i32;
                    }
                    acc -= w_zp * input_sum;
                    // Also correct the cross-term: + N * w_zp * input_zp
                    acc += (in_size as i32) * w_zp * input_zp_i32;
                }

                // Dequantize: y = acc * (w_scale * input_scale) + bias
                let mut y =
                    (acc as f32).mul_add(combined_scale, self.layers[layer_idx].bias_f32[j]);

                // Activation (not on last layer)
                if !is_last {
                    y = activate_f32(y, self.activation);
                }

                dst[j] = y;
            }
        }
    }

    /// Number of input features.
    pub fn input_size(&self) -> usize {
        self.input_size as usize
    }

    /// Number of output values.
    pub fn output_size(&self) -> usize {
        self.output_size as usize
    }

    /// Number of layers (weight matrices).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Activation function applied to hidden layers.
    pub fn activation(&self) -> Activation {
        self.activation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_identity_passthrough() {
        // 1 layer, 2→1, identity, scale=1.0, zp=0
        // weights=[1, 1], bias=0
        // input=[3.0, 4.0] → quantized=[3, 4] → dot=7 → dequant=7*1*1+0=7.0
        let w = [1_i8, 1];
        let b = [0.0_f32];
        let mut m = QuantizedMlpI8::from_parts(
            &[&w],
            &[&b],
            &[1.0],
            &[0],
            &[1.0],
            &[0],
            Activation::Identity,
        )
        .unwrap();
        let out = m.predict(&[3.0, 4.0]);
        assert!((out - 7.0).abs() < 1e-6, "got {out}");
    }

    #[test]
    fn symmetric_with_scale() {
        // 1 layer, 2→1, identity
        // w=[10, 20], w_scale=0.1 (real weights: 1.0, 2.0)
        // input_scale=0.5 → input [2.0, 3.0] quantizes to [4, 6]
        // dot = 10*4 + 20*6 = 160
        // dequant = 160 * 0.1 * 0.5 = 8.0
        // real: 1.0*2.0 + 2.0*3.0 = 8.0
        let w = [10_i8, 20];
        let b = [0.0_f32];
        let mut m = QuantizedMlpI8::from_parts(
            &[&w],
            &[&b],
            &[0.1],
            &[0],
            &[0.5],
            &[0],
            Activation::Identity,
        )
        .unwrap();
        let out = m.predict(&[2.0, 3.0]);
        assert!((out - 8.0).abs() < 0.5, "got {out}");
    }

    #[test]
    fn bias_applied() {
        let w = [1_i8, 1];
        let b = [5.0_f32];
        let mut m = QuantizedMlpI8::from_parts(
            &[&w],
            &[&b],
            &[1.0],
            &[0],
            &[1.0],
            &[0],
            Activation::Identity,
        )
        .unwrap();
        let out = m.predict(&[3.0, 4.0]);
        assert!((out - 12.0).abs() < 1e-6, "got {out}");
    }

    #[test]
    fn relu_activation() {
        // 2 layers: 2→2→1, relu
        // Layer 0: w=[1,0, -1,0], b=[0, 0] → [x0, -x0]
        // After relu: [max(0,x0), 0]
        // Layer 1: w=[1, 1], b=[0] → max(0,x0)
        let w0 = [1_i8, 0, -1, 0];
        let b0 = [0.0_f32, 0.0];
        let w1 = [1_i8, 1];
        let b1 = [0.0_f32];
        let mut m = QuantizedMlpI8::from_parts(
            &[&w0, &w1],
            &[&b0, &b1],
            &[1.0, 1.0],
            &[0, 0],
            &[1.0, 1.0],
            &[0, 0],
            Activation::Relu,
        )
        .unwrap();
        // input=[5, 0] → layer0=[5, -5] → relu=[5, 0] → layer1=5
        let out = m.predict(&[5.0, 0.0]);
        assert!((out - 5.0).abs() < 1e-4, "positive: got {out}");

        // input=[-3, 0] → layer0=[-3, 3] → relu=[0, 3] → layer1=3
        let out2 = m.predict(&[-3.0, 0.0]);
        assert!((out2 - 3.0).abs() < 1e-4, "negative: got {out2}");
    }

    #[test]
    fn asymmetric_weight_zero_point() {
        // w_zp=10 means real weight = (w_i8 - 10) * w_scale
        // w_i8=[20, 30], w_scale=0.1, w_zp=10
        // real weights: (20-10)*0.1=1.0, (30-10)*0.1=2.0
        // input=[2.0, 3.0], input_scale=1.0, input_zp=0
        // expected: 1.0*2.0 + 2.0*3.0 = 8.0
        let w = [20_i8, 30];
        let b = [0.0_f32];
        let mut m = QuantizedMlpI8::from_parts(
            &[&w],
            &[&b],
            &[0.1],
            &[10],
            &[1.0],
            &[0],
            Activation::Identity,
        )
        .unwrap();
        let out = m.predict(&[2.0, 3.0]);
        assert!((out - 8.0).abs() < 0.5, "got {out}");
    }

    #[test]
    fn asymmetric_input_zero_point() {
        // input_zp=5 means real input = (x_i8 - 5) * input_scale
        // input_scale=1.0, so input [7.0] → quantized i8=12 → real=(12-5)*1=7.0
        // w=[1], w_scale=1.0, w_zp=0
        // dot=1*12=12, correction: -5*sum(w)=-5*1=-5 → acc=12-5=7
        // dequant=7*1.0*1.0=7.0
        let w = [1_i8];
        let b = [0.0_f32];
        let mut m = QuantizedMlpI8::from_parts(
            &[&w],
            &[&b],
            &[1.0],
            &[0],
            &[1.0],
            &[5],
            Activation::Identity,
        )
        .unwrap();
        let out = m.predict(&[7.0]);
        assert!((out - 7.0).abs() < 1e-4, "got {out}");
    }

    #[test]
    fn multi_output() {
        let w = [1_i8, 0, 0, 1];
        let b = [10.0_f32, 20.0];
        let mut m = QuantizedMlpI8::from_parts(
            &[&w],
            &[&b],
            &[1.0],
            &[0],
            &[1.0],
            &[0],
            Activation::Identity,
        )
        .unwrap();
        let mut out = [0.0_f32; 2];
        m.predict_into(&[3.0, 7.0], &mut out);
        assert!((out[0] - 13.0).abs() < 1e-4, "out[0]={}", out[0]);
        assert!((out[1] - 27.0).abs() < 1e-4, "out[1]={}", out[1]);
    }

    #[test]
    fn two_layer_with_scale() {
        // Layer 0: 2→2, w_scale=0.1, i_scale=0.5
        // Layer 1: 2→1, w_scale=0.2, i_scale=0.1
        let w0 = [10_i8, 0, 0, 10]; // identity (real: 1.0, 1.0 diagonal)
        let b0 = [0.0_f32, 0.0];
        let w1 = [5_i8, 5]; // real: 1.0, 1.0
        let b1 = [0.0_f32];
        let mut m = QuantizedMlpI8::from_parts(
            &[&w0, &w1],
            &[&b0, &b1],
            &[0.1, 0.2],
            &[0, 0],
            &[0.5, 0.1],
            &[0, 0],
            Activation::Identity,
        )
        .unwrap();
        // input=[2.0, 3.0]
        // L0: quantize(2.0/0.5)=4, (3.0/0.5)=6
        //     dot0=10*4=40, dot1=10*6=60
        //     dequant: 40*0.1*0.5=2.0, 60*0.1*0.5=3.0 → [2.0, 3.0]
        // L1: quantize(2.0/0.1)=20, (3.0/0.1)=30
        //     dot=5*20+5*30=250
        //     dequant: 250*0.2*0.1=5.0
        let out = m.predict(&[2.0, 3.0]);
        assert!((out - 5.0).abs() < 0.5, "got {out}");
    }

    #[test]
    fn accessors() {
        let w0 = [0_i8; 3 * 2];
        let b0 = [0.0_f32; 3];
        let w1 = [0_i8; 1 * 3];
        let b1 = [0.0_f32; 1];
        let m = QuantizedMlpI8::from_parts(
            &[&w0, &w1],
            &[&b0, &b1],
            &[1.0, 1.0],
            &[0, 0],
            &[1.0, 1.0],
            &[0, 0],
            Activation::Relu,
        )
        .unwrap();
        assert_eq!(m.input_size(), 2);
        assert_eq!(m.output_size(), 1);
        assert_eq!(m.num_layers(), 2);
        assert!(matches!(m.activation(), Activation::Relu));
    }

    #[test]
    fn validation_rejects_empty() {
        let r: Result<QuantizedMlpI8, _> =
            QuantizedMlpI8::from_parts(&[], &[], &[], &[], &[], &[], Activation::Relu);
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_mismatched_arrays() {
        let w = [0_i8; 4];
        let b = [0.0_f32; 2];
        let r = QuantizedMlpI8::from_parts(
            &[&w],
            &[&b],
            &[1.0, 1.0], // 2 scales but only 1 layer
            &[0],
            &[1.0],
            &[0],
            Activation::Relu,
        );
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_bad_scale() {
        let w = [0_i8; 4];
        let b = [0.0_f32; 2];
        let r = QuantizedMlpI8::from_parts(
            &[&w],
            &[&b],
            &[-1.0], // negative scale
            &[0],
            &[1.0],
            &[0],
            Activation::Relu,
        );
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_non_finite_bias() {
        let w = [0_i8; 4];
        let b = [f32::NAN, 0.0];
        let r =
            QuantizedMlpI8::from_parts(&[&w], &[&b], &[1.0], &[0], &[1.0], &[0], Activation::Relu);
        assert!(r.is_err());
    }

    #[test]
    fn validation_rejects_layer_size_mismatch() {
        // Layer 0: 2→3, Layer 1: 4→1 (should be 3→1)
        let w0 = [0_i8; 3 * 2];
        let b0 = [0.0_f32; 3];
        let w1 = [0_i8; 1 * 4]; // wrong: in_size=4 != prev out_size=3
        let b1 = [0.0_f32; 1];
        let r = QuantizedMlpI8::from_parts(
            &[&w0, &w1],
            &[&b0, &b1],
            &[1.0, 1.0],
            &[0, 0],
            &[1.0, 1.0],
            &[0, 0],
            Activation::Relu,
        );
        assert!(r.is_err());
    }

    #[test]
    #[should_panic(expected = "output_size == 1")]
    fn predict_panics_multi_output() {
        let w = [0_i8; 4];
        let b = [0.0_f32; 2];
        let mut m =
            QuantizedMlpI8::from_parts(&[&w], &[&b], &[1.0], &[0], &[1.0], &[0], Activation::Relu)
                .unwrap();
        m.predict(&[1.0, 2.0]);
    }

    #[test]
    #[should_panic]
    fn predict_panics_wrong_input_len() {
        let w = [0_i8; 4];
        let b = [0.0_f32; 2];
        let mut m =
            QuantizedMlpI8::from_parts(&[&w], &[&b], &[1.0], &[0], &[1.0], &[0], Activation::Relu)
                .unwrap();
        m.predict(&[1.0]); // expects 2 inputs
    }

    #[test]
    fn clamps_to_i8_range() {
        // Large input that would overflow i8 range after quantization
        let w = [1_i8];
        let b = [0.0_f32];
        let mut m = QuantizedMlpI8::from_parts(
            &[&w],
            &[&b],
            &[1.0],
            &[0],
            &[1.0],
            &[0],
            Activation::Identity,
        )
        .unwrap();
        // 200.0 / 1.0 = 200 → clamped to 127
        let out = m.predict(&[200.0]);
        assert!((out - 127.0).abs() < 1e-4, "got {out}");

        // -200.0 → clamped to -128
        let out2 = m.predict(&[-200.0]);
        assert!((out2 - (-128.0)).abs() < 1e-4, "got {out2}");
    }
}

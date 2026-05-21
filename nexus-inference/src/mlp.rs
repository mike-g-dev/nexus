#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, vec::Vec};

#[cfg(feature = "alloc")]
use crate::{LoadError, NanInput};

/// Hidden-layer activation function.
///
/// Applied to all hidden layers. The output layer always produces raw
/// linear scores (caller applies sigmoid/softmax if needed).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    /// max(0, x)
    Relu,
    /// x if x >= 0, alpha * x otherwise.
    LeakyRelu(f64),
    /// Hyperbolic tangent. Requires `std` or `libm`.
    Tanh,
    /// 1 / (1 + exp(-x)). Requires `std` or `libm`.
    Sigmoid,
}

#[cfg(feature = "alloc")]
macro_rules! impl_mlp {
    ($name:ident, $ty:ty) => {
        /// Feedforward neural network (multi-layer perceptron).
        ///
        /// Immutable after construction. All prediction methods take `&self`.
        /// Weights are row-major (output-major): each row of a weight matrix
        /// contains the weights for one output neuron. This matches PyTorch's
        /// `nn.Linear.weight` layout.
        ///
        /// # Examples
        ///
        /// ```
        /// use nexus_inference::{MlpF64, Activation};
        ///
        /// let model = MlpF64::from_parts(
        ///     &[2, 3, 1],
        ///     &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ///     &[0.0, 0.0, 0.0, 0.0],
        ///     Activation::Relu,
        /// ).unwrap();
        /// let score = model.predict(&[1.0, 2.0]).unwrap();
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            weights: Box<[$ty]>,
            biases: Box<[$ty]>,
            layer_sizes: Box<[u16]>,
            activation: Activation,
        }

        impl $name {
            /// Construct from pre-trained weights.
            ///
            /// `layer_sizes` defines the full topology: `[n_inputs, h1, h2, ..., n_outputs]`.
            /// Minimum length 2 (input + output).
            ///
            /// Weight layout is row-major (output-major). For layer `l` connecting
            /// `layer_sizes[l]` inputs to `layer_sizes[l+1]` outputs, the weight
            /// matrix has `layer_sizes[l+1]` rows of `layer_sizes[l]` columns.
            pub fn from_parts(
                layer_sizes: &[usize],
                weights: &[$ty],
                biases: &[$ty],
                activation: Activation,
            ) -> Result<Self, LoadError> {
                if layer_sizes.len() < 2 {
                    return Err(LoadError::Validation("layer_sizes must have at least 2 elements"));
                }
                for &sz in layer_sizes.iter() {
                    if sz == 0 {
                        return Err(LoadError::Validation("layer size must be > 0"));
                    }
                    if sz > u16::MAX as usize {
                        return Err(LoadError::Validation("layer size exceeds u16::MAX"));
                    }
                }

                let n_layers = layer_sizes.len() - 1;
                let expected_weights: usize = (0..n_layers)
                    .map(|i| layer_sizes[i] * layer_sizes[i + 1])
                    .sum();
                let expected_biases: usize = (0..n_layers).map(|i| layer_sizes[i + 1]).sum();

                if weights.len() != expected_weights {
                    return Err(LoadError::Validation("weights length mismatch"));
                }
                if biases.len() != expected_biases {
                    return Err(LoadError::Validation("biases length mismatch"));
                }

                for &w in weights {
                    if !w.is_finite() {
                        return Err(LoadError::Validation("non-finite weight"));
                    }
                }
                for &b in biases {
                    if !b.is_finite() {
                        return Err(LoadError::Validation("non-finite bias"));
                    }
                }

                #[cfg(not(any(feature = "std", feature = "libm")))]
                match activation {
                    Activation::Tanh | Activation::Sigmoid => {
                        return Err(LoadError::Validation(
                            "Tanh/Sigmoid require std or libm feature",
                        ));
                    }
                    _ => {}
                }

                let layer_sizes_u16: Box<[u16]> = layer_sizes
                    .iter()
                    .map(|&s| s as u16)
                    .collect::<Vec<u16>>()
                    .into_boxed_slice();

                Ok(Self {
                    weights: weights.into(),
                    biases: biases.into(),
                    layer_sizes: layer_sizes_u16,
                    activation,
                })
            }

            /// Single-output prediction with NaN input check.
            ///
            /// Returns `Err(NanInput)` if any input is NaN.
            /// Panics if `n_outputs() != 1`.
            pub fn predict(&self, input: &[$ty]) -> Result<$ty, NanInput> {
                if input.iter().any(|x| x.is_nan()) {
                    return Err(NanInput);
                }
                Ok(self.predict_unchecked(input))
            }

            /// Single-output prediction without NaN check.
            ///
            /// NaN inputs propagate through the computation (including
            /// through relu). Panics if `n_outputs() != 1`.
            pub fn predict_unchecked(&self, input: &[$ty]) -> $ty {
                assert_eq!(
                    self.n_outputs(),
                    1,
                    "predict_unchecked() requires n_outputs == 1, use predict_into_unchecked()"
                );
                let mut out = [0.0 as $ty];
                self.predict_into_unchecked(input, &mut out);
                out[0]
            }

            /// General prediction with NaN input check.
            ///
            /// Returns `Err(NanInput)` if any input is NaN.
            ///
            /// # Panics
            ///
            /// Panics if `input.len() != self.n_inputs()` or
            /// `output.len() != self.n_outputs()`.
            pub fn predict_into(&self, input: &[$ty], output: &mut [$ty]) -> Result<(), NanInput> {
                if input.iter().any(|x| x.is_nan()) {
                    return Err(NanInput);
                }
                self.predict_into_unchecked(input, output);
                Ok(())
            }

            /// General prediction without NaN check.
            ///
            /// NaN inputs propagate through the computation.
            ///
            /// # Panics
            ///
            /// Panics if `input.len() != self.n_inputs()` or
            /// `output.len() != self.n_outputs()`.
            pub fn predict_into_unchecked(&self, input: &[$ty], output: &mut [$ty]) {
                assert_eq!(input.len(), self.n_inputs());
                assert_eq!(output.len(), self.n_outputs());

                let n_layers = self.layer_sizes.len() - 1;
                let max_dim = self.layer_sizes.iter().map(|&s| s as usize).max().unwrap_or(0);
                let mut buf_a: Vec<$ty> = alloc::vec![0.0 as $ty; max_dim];
                let mut buf_b: Vec<$ty> = alloc::vec![0.0 as $ty; max_dim];

                // Copy input into buf_a so we can ping-pong without
                // holding a reference to the caller's slice.
                buf_a[..input.len()].copy_from_slice(input);
                // src_is_a tracks which buffer holds the current layer's input.
                let mut src_is_a = true;
                let mut w_offset = 0usize;
                let mut b_offset = 0usize;

                for layer in 0..n_layers {
                    let in_size = self.layer_sizes[layer] as usize;
                    let out_size = self.layer_sizes[layer + 1] as usize;
                    let is_last = layer == n_layers - 1;

                    for j in 0..out_size {
                        let row = &self.weights[w_offset + j * in_size..w_offset + (j + 1) * in_size];
                        let mut sum = self.biases[b_offset + j];
                        let src = if src_is_a { &buf_a } else { &buf_b };
                        for k in 0..in_size {
                            sum += row[k] * src[k];
                        }
                        if !is_last {
                            sum = Self::activate(sum, self.activation);
                        }
                        if is_last {
                            output[j] = sum;
                        } else if src_is_a {
                            buf_b[j] = sum;
                        } else {
                            buf_a[j] = sum;
                        }
                    }

                    w_offset += in_size * out_size;
                    b_offset += out_size;
                    src_is_a = !src_is_a;
                }
            }

            /// Number of input features.
            pub fn n_inputs(&self) -> usize {
                self.layer_sizes[0] as usize
            }

            /// Number of output values.
            pub fn n_outputs(&self) -> usize {
                *self.layer_sizes.last().unwrap() as usize
            }

            /// Number of weight matrices (layers).
            pub fn n_layers(&self) -> usize {
                self.layer_sizes.len() - 1
            }

            /// Activation function used for hidden layers.
            pub fn activation(&self) -> Activation {
                self.activation
            }

            #[inline(always)]
            fn activate(x: $ty, activation: Activation) -> $ty {
                match activation {
                    Activation::Relu => {
                        if x > 0.0 as $ty { x } else if x <= 0.0 as $ty { 0.0 as $ty } else { x }
                    }
                    Activation::LeakyRelu(alpha) => {
                        if x >= 0.0 as $ty { x } else { x * alpha as $ty }
                    }
                    Activation::Tanh => {
                        #[cfg(feature = "std")]
                        { (x as f64).tanh() as $ty }
                        #[cfg(all(not(feature = "std"), feature = "libm"))]
                        { libm::tanh(x as f64) as $ty }
                        #[cfg(not(any(feature = "std", feature = "libm")))]
                        { let _ = x; unreachable!() }
                    }
                    Activation::Sigmoid => {
                        #[cfg(feature = "std")]
                        { (1.0 / (1.0 + (-(x as f64)).exp())) as $ty }
                        #[cfg(all(not(feature = "std"), feature = "libm"))]
                        { (1.0 / (1.0 + libm::exp(-(x as f64)))) as $ty }
                        #[cfg(not(any(feature = "std", feature = "libm")))]
                        { let _ = x; unreachable!() }
                    }
                }
            }
        }
    };
}

#[cfg(feature = "alloc")]
impl_mlp!(MlpF64, f64);
#[cfg(feature = "alloc")]
impl_mlp!(MlpF32, f32);

#[cfg(test)]
mod tests {
    #[cfg(feature = "alloc")]
    use super::*;
    #[cfg(feature = "alloc")]
    use alloc::vec;

    #[test]
    #[cfg(feature = "alloc")]
    fn single_neuron_no_hidden() {
        // 1 input → 1 output, w=2.0, b=0.5 → 2*x + 0.5
        let model = MlpF64::from_parts(&[1, 1], &[2.0], &[0.5], Activation::Relu).unwrap();
        assert!((model.predict(&[3.0]).unwrap() - 6.5).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn two_layer_relu() {
        // 2 inputs → 2 hidden (relu) → 1 output
        // Hidden weights (2×2, row-major):
        //   h0 = relu(1.0*x0 + 0.0*x1 + 0.0) = relu(x0)
        //   h1 = relu(0.0*x0 + 1.0*x1 + 0.0) = relu(x1)
        // Output weights (1×2):
        //   o0 = 1.0*h0 + 1.0*h1 + 0.0
        let weights = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let biases = vec![0.0, 0.0, 0.0];
        let model = MlpF64::from_parts(&[2, 2, 1], &weights, &biases, Activation::Relu).unwrap();
        assert!((model.predict(&[3.0, 4.0]).unwrap() - 7.0).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn relu_clips_negative() {
        // 1 input → 1 hidden (relu) → 1 output
        // h0 = relu(1.0*x + (-5.0)) → relu(x - 5)
        // o0 = 1.0 * h0 + 0.0
        let model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[-5.0, 0.0], Activation::Relu).unwrap();
        assert!((model.predict(&[3.0]).unwrap() - 0.0).abs() < 1e-12); // relu(3 - 5) = 0
        assert!((model.predict(&[7.0]).unwrap() - 2.0).abs() < 1e-12); // relu(7 - 5) = 2
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn leaky_relu() {
        // 1 input → 1 hidden (leaky_relu 0.1) → 1 output
        // h0 = leaky_relu(1.0*x + 0.0)
        // o0 = 1.0*h0 + 0.0
        let model = MlpF64::from_parts(
            &[1, 1, 1],
            &[1.0, 1.0],
            &[0.0, 0.0],
            Activation::LeakyRelu(0.1),
        )
        .unwrap();
        assert!((model.predict(&[2.0]).unwrap() - 2.0).abs() < 1e-12);
        assert!((model.predict(&[-3.0]).unwrap() - (-0.3)).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn tanh_activation() {
        // 1 input → 1 hidden (tanh) → 1 output
        // h0 = tanh(1.0*x + 0.0)
        // o0 = 1.0*h0 + 0.0
        let model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, 0.0], Activation::Tanh).unwrap();
        let expected = 2.0_f64.tanh();
        assert!((model.predict(&[2.0]).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn sigmoid_activation() {
        // 1 input → 1 hidden (sigmoid) → 1 output
        // h0 = sigmoid(1.0*x + 0.0)
        // o0 = 1.0*h0 + 0.0
        let model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, 0.0], Activation::Sigmoid).unwrap();
        let expected = 1.0 / (1.0 + (-2.0_f64).exp());
        assert!((model.predict(&[2.0]).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn three_layer() {
        // 3 inputs → 4 hidden → 2 hidden → 1 output (relu)
        //
        // Layer 0 weights (4×3): identity-ish mapping + bias
        //   h0 = relu(1*x0 + 0*x1 + 0*x2 + 0) = relu(x0)
        //   h1 = relu(0*x0 + 1*x1 + 0*x2 + 0) = relu(x1)
        //   h2 = relu(0*x0 + 0*x1 + 1*x2 + 0) = relu(x2)
        //   h3 = relu(1*x0 + 1*x1 + 1*x2 + 0) = relu(x0+x1+x2)
        let w0: Vec<f64> = vec![
            1.0, 0.0, 0.0, // h0
            0.0, 1.0, 0.0, // h1
            0.0, 0.0, 1.0, // h2
            1.0, 1.0, 1.0, // h3
        ];
        let b0: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0];

        // Layer 1 weights (2×4):
        //   g0 = relu(1*h0 + 1*h1 + 0*h2 + 0*h3 + 0) = relu(h0 + h1)
        //   g1 = relu(0*h0 + 0*h1 + 1*h2 + 1*h3 + 0) = relu(h2 + h3)
        let w1: Vec<f64> = vec![
            1.0, 1.0, 0.0, 0.0, // g0
            0.0, 0.0, 1.0, 1.0, // g1
        ];
        let b1: Vec<f64> = vec![0.0, 0.0];

        // Layer 2 weights (1×2):
        //   o0 = 1*g0 + 1*g1 + 0
        let w2: Vec<f64> = vec![1.0, 1.0];
        let b2: Vec<f64> = vec![0.0];

        let mut weights = Vec::new();
        weights.extend_from_slice(&w0);
        weights.extend_from_slice(&w1);
        weights.extend_from_slice(&w2);
        let mut biases = Vec::new();
        biases.extend_from_slice(&b0);
        biases.extend_from_slice(&b1);
        biases.extend_from_slice(&b2);

        let model = MlpF64::from_parts(&[3, 4, 2, 1], &weights, &biases, Activation::Relu).unwrap();

        // x = [1, 2, 3]
        // h = [1, 2, 3, 6], g = [1+2, 3+6] = [3, 9], o = 3+9 = 12
        assert!((model.predict(&[1.0, 2.0, 3.0]).unwrap() - 12.0).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn output_layer_no_activation() {
        // 1 input → 1 hidden (relu) → 1 output
        // Hidden: h = relu(1.0*x + 0.0) = relu(x)
        // Output: o = 1.0*h + (-10.0)
        // If activation applied to output, negative output would be clipped.
        let model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, -10.0], Activation::Relu).unwrap();
        // x=5 → h=relu(5)=5 → o=5-10=-5 (NOT relu'd)
        assert!((model.predict(&[5.0]).unwrap() - (-5.0)).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[should_panic]
    fn wrong_input_panics() {
        let model = MlpF64::from_parts(&[2, 1], &[1.0, 1.0], &[0.0], Activation::Relu).unwrap();
        model.predict_unchecked(&[1.0]); // expects 2 inputs
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_parts_validates_sizes() {
        // Wrong weight count
        let err = MlpF64::from_parts(&[2, 3, 1], &[1.0; 5], &[0.0; 4], Activation::Relu);
        assert!(err.is_err());
        // Wrong bias count
        let err = MlpF64::from_parts(&[2, 3, 1], &[1.0; 9], &[0.0; 3], Activation::Relu);
        assert!(err.is_err());
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_parts_validates_layer_sizes() {
        // Empty
        let err = MlpF64::from_parts(&[], &[], &[], Activation::Relu);
        assert!(err.is_err());
        // Single element
        let err = MlpF64::from_parts(&[5], &[], &[], Activation::Relu);
        assert!(err.is_err());
        // Zero-sized layer
        let err = MlpF64::from_parts(&[2, 0, 1], &[], &[], Activation::Relu);
        assert!(err.is_err());
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn f32_variant() {
        let model = MlpF32::from_parts(
            &[2, 2, 1],
            &[1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[0.0_f32, 0.0, 0.0],
            Activation::Relu,
        )
        .unwrap();
        assert!((model.predict(&[3.0_f32, 4.0]).unwrap() - 7.0_f32).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn nan_through_relu_propagates_unchecked() {
        // 1 input → 1 hidden (relu) → 1 output
        // NaN goes through relu hidden layer — must come out as NaN
        let model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, 0.0], Activation::Relu).unwrap();
        assert!(model.predict_unchecked(&[f64::NAN]).is_nan());
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn nan_input_returns_error() {
        let model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, 0.0], Activation::Relu).unwrap();
        assert!(model.predict(&[f64::NAN]).is_err());
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn nan_input_predict_into_returns_error() {
        let model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, 0.0], Activation::Relu).unwrap();
        let mut out = [0.0_f64];
        assert!(model.predict_into(&[f64::NAN], &mut out).is_err());
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn multi_output() {
        // 2 inputs → 4 hidden (relu) → 3 outputs
        // Hidden: identity-ish
        //   h0=x0, h1=x1, h2=x0+x1, h3=x0-x1 (clipped by relu)
        let w0: Vec<f64> = vec![
            1.0, 0.0, // h0
            0.0, 1.0, // h1
            1.0, 1.0, // h2
            1.0, -1.0, // h3
        ];
        let b0: Vec<f64> = vec![0.0; 4];
        // Output: 3 outputs, each picks one hidden
        //   o0 = h0, o1 = h1, o2 = h2
        let w1: Vec<f64> = vec![
            1.0, 0.0, 0.0, 0.0, // o0
            0.0, 1.0, 0.0, 0.0, // o1
            0.0, 0.0, 1.0, 0.0, // o2
        ];
        let b1: Vec<f64> = vec![0.0; 3];

        let mut weights = Vec::new();
        weights.extend_from_slice(&w0);
        weights.extend_from_slice(&w1);
        let mut biases = Vec::new();
        biases.extend_from_slice(&b0);
        biases.extend_from_slice(&b1);

        let model = MlpF64::from_parts(&[2, 4, 3], &weights, &biases, Activation::Relu).unwrap();
        assert_eq!(model.n_outputs(), 3);

        let mut out = [0.0_f64; 3];
        model.predict_into(&[5.0, 3.0], &mut out).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-12);
        assert!((out[1] - 3.0).abs() < 1e-12);
        assert!((out[2] - 8.0).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[should_panic]
    fn predict_panics_multi_output() {
        let model = MlpF64::from_parts(&[2, 3], &[1.0; 6], &[0.0; 3], Activation::Relu).unwrap();
        model.predict_unchecked(&[1.0, 2.0]); // n_outputs=3, should panic
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[should_panic]
    fn predict_into_wrong_output_len() {
        let model = MlpF64::from_parts(&[1, 1], &[1.0], &[0.0], Activation::Relu).unwrap();
        let mut out = [0.0_f64; 2];
        model.predict_into_unchecked(&[1.0], &mut out);
    }
}

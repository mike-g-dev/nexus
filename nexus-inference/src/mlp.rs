#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, vec::Vec};

#[cfg(feature = "alloc")]
use crate::LoadError;
#[cfg(feature = "alloc")]
use crate::activation::Activation;

#[cfg(feature = "alloc")]
macro_rules! impl_mlp {
    ($name:ident, $ty:ty, $dot_fn:path, $dot4_fn:path, $activate_fn:path) => {
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
        /// let mut model = MlpF64::from_parts(
        ///     &[2, 3, 1],
        ///     &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ///     &[0.0, 0.0, 0.0, 0.0],
        ///     Activation::Relu,
        /// ).unwrap();
        /// let score = model.predict(&[1.0, 2.0]);
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            weights: Box<[$ty]>,
            biases: Box<[$ty]>,
            layer_sizes: Box<[u16]>,
            activation: Activation,
            scratch_a: Vec<$ty>,
            scratch_b: Vec<$ty>,
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

                let layer_sizes_u16: Box<[u16]> = layer_sizes
                    .iter()
                    .map(|&s| s as u16)
                    .collect::<Vec<u16>>()
                    .into_boxed_slice();

                let max_dim = layer_sizes.iter().copied().max().unwrap();

                Ok(Self {
                    weights: weights.into(),
                    biases: biases.into(),
                    layer_sizes: layer_sizes_u16,
                    activation,
                    scratch_a: alloc::vec![0.0 as $ty; max_dim],
                    scratch_b: alloc::vec![0.0 as $ty; max_dim],
                })
            }

            /// Single-output prediction.
            ///
            /// NaN inputs propagate through the computation.
            /// Panics if `n_outputs() != 1`.
            pub fn predict(&mut self, input: &[$ty]) -> $ty {
                assert_eq!(
                    self.n_outputs(),
                    1,
                    "predict() requires n_outputs == 1, use predict_into()"
                );
                let mut out = [0.0 as $ty];
                self.predict_into(input, &mut out);
                out[0]
            }

            /// General prediction (multi-output).
            ///
            /// NaN inputs propagate through the computation.
            ///
            /// # Panics
            ///
            /// Panics if `input.len() != self.n_inputs()` or
            /// `output.len() != self.n_outputs()`.
            pub fn predict_into(&mut self, input: &[$ty], output: &mut [$ty]) {
                assert_eq!(input.len(), self.n_inputs());
                assert_eq!(output.len(), self.n_outputs());

                let n_layers = self.layer_sizes.len() - 1;

                self.scratch_a[..input.len()].copy_from_slice(input);
                let mut src_is_a = true;
                let mut w_offset = 0usize;
                let mut b_offset = 0usize;

                for layer in 0..n_layers {
                    let in_size = self.layer_sizes[layer] as usize;
                    let out_size = self.layer_sizes[layer + 1] as usize;
                    let is_last = layer == n_layers - 1;
                    let out_size_4 = out_size & !3;

                    let mut j = 0;
                    while j < out_size_4 {
                        let rows = &self.weights[w_offset + j * in_size..w_offset + (j + 4) * in_size];
                        let src = if src_is_a { &self.scratch_a[..in_size] } else { &self.scratch_b[..in_size] };
                        let dots = $dot4_fn(rows, src);
                        for k in 0..4 {
                            let mut sum = self.biases[b_offset + j + k] + dots[k];
                            if !is_last {
                                sum = $activate_fn(sum, self.activation);
                            }
                            if is_last {
                                output[j + k] = sum;
                            } else if src_is_a {
                                self.scratch_b[j + k] = sum;
                            } else {
                                self.scratch_a[j + k] = sum;
                            }
                        }
                        j += 4;
                    }
                    while j < out_size {
                        let row = &self.weights[w_offset + j * in_size..w_offset + (j + 1) * in_size];
                        let src = if src_is_a { &self.scratch_a[..in_size] } else { &self.scratch_b[..in_size] };
                        let mut sum = self.biases[b_offset + j] + $dot_fn(row, src);
                        if !is_last {
                            sum = $activate_fn(sum, self.activation);
                        }
                        if is_last {
                            output[j] = sum;
                        } else if src_is_a {
                            self.scratch_b[j] = sum;
                        } else {
                            self.scratch_a[j] = sum;
                        }
                        j += 1;
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
        }
    };
}

#[cfg(feature = "alloc")]
impl_mlp!(
    MlpF64,
    f64,
    crate::dot::dot_f64,
    crate::dot::dot4_f64,
    crate::activation::activate_f64
);
#[cfg(feature = "alloc")]
impl_mlp!(
    MlpF32,
    f32,
    crate::dot::dot_f32,
    crate::dot::dot4_f32,
    crate::activation::activate_f32
);

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
        let mut model = MlpF64::from_parts(&[1, 1], &[2.0], &[0.5], Activation::Relu).unwrap();
        assert!((model.predict(&[3.0]) - 6.5).abs() < 1e-12);
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
        let mut model =
            MlpF64::from_parts(&[2, 2, 1], &weights, &biases, Activation::Relu).unwrap();
        assert!((model.predict(&[3.0, 4.0]) - 7.0).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn relu_clips_negative() {
        // 1 input → 1 hidden (relu) → 1 output
        // h0 = relu(1.0*x + (-5.0)) → relu(x - 5)
        // o0 = 1.0 * h0 + 0.0
        let mut model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[-5.0, 0.0], Activation::Relu).unwrap();
        assert!((model.predict(&[3.0]) - 0.0).abs() < 1e-12); // relu(3 - 5) = 0
        assert!((model.predict(&[7.0]) - 2.0).abs() < 1e-12); // relu(7 - 5) = 2
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn leaky_relu() {
        // 1 input → 1 hidden (leaky_relu 0.1) → 1 output
        // h0 = leaky_relu(1.0*x + 0.0)
        // o0 = 1.0*h0 + 0.0
        let mut model = MlpF64::from_parts(
            &[1, 1, 1],
            &[1.0, 1.0],
            &[0.0, 0.0],
            Activation::LeakyRelu(0.1),
        )
        .unwrap();
        assert!((model.predict(&[2.0]) - 2.0).abs() < 1e-12);
        assert!((model.predict(&[-3.0]) - (-0.3)).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn tanh_activation() {
        // 1 input → 1 hidden (tanh) → 1 output
        // h0 = tanh(1.0*x + 0.0)
        // o0 = 1.0*h0 + 0.0
        let mut model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, 0.0], Activation::Tanh).unwrap();
        let expected = 2.0_f64.tanh();
        assert!((model.predict(&[2.0]) - expected).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn sigmoid_activation() {
        // 1 input → 1 hidden (sigmoid) → 1 output
        // h0 = sigmoid(1.0*x + 0.0)
        // o0 = 1.0*h0 + 0.0
        let mut model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, 0.0], Activation::Sigmoid).unwrap();
        let expected = 1.0 / (1.0 + (-2.0_f64).exp());
        assert!((model.predict(&[2.0]) - expected).abs() < 1e-12);
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

        let mut model =
            MlpF64::from_parts(&[3, 4, 2, 1], &weights, &biases, Activation::Relu).unwrap();

        // x = [1, 2, 3]
        // h = [1, 2, 3, 6], g = [1+2, 3+6] = [3, 9], o = 3+9 = 12
        assert!((model.predict(&[1.0, 2.0, 3.0]) - 12.0).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn output_layer_no_activation() {
        // 1 input → 1 hidden (relu) → 1 output
        // Hidden: h = relu(1.0*x + 0.0) = relu(x)
        // Output: o = 1.0*h + (-10.0)
        // If activation applied to output, negative output would be clipped.
        let mut model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, -10.0], Activation::Relu).unwrap();
        // x=5 → h=relu(5)=5 → o=5-10=-5 (NOT relu'd)
        assert!((model.predict(&[5.0]) - (-5.0)).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[should_panic]
    fn wrong_input_panics() {
        let mut model = MlpF64::from_parts(&[2, 1], &[1.0, 1.0], &[0.0], Activation::Relu).unwrap();
        model.predict(&[1.0]); // expects 2 inputs
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
        let mut model = MlpF32::from_parts(
            &[2, 2, 1],
            &[1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[0.0_f32, 0.0, 0.0],
            Activation::Relu,
        )
        .unwrap();
        assert!((model.predict(&[3.0_f32, 4.0]) - 7.0_f32).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn nan_through_relu_propagates() {
        // 1 input → 1 hidden (relu) → 1 output
        // NaN goes through relu hidden layer — must come out as NaN
        let mut model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, 0.0], Activation::Relu).unwrap();
        assert!(model.predict(&[f64::NAN]).is_nan());
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

        let mut model =
            MlpF64::from_parts(&[2, 4, 3], &weights, &biases, Activation::Relu).unwrap();
        assert_eq!(model.n_outputs(), 3);

        let mut out = [0.0_f64; 3];
        model.predict_into(&[5.0, 3.0], &mut out);
        assert!((out[0] - 5.0).abs() < 1e-12);
        assert!((out[1] - 3.0).abs() < 1e-12);
        assert!((out[2] - 8.0).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[should_panic]
    fn predict_panics_multi_output() {
        let mut model =
            MlpF64::from_parts(&[2, 3], &[1.0; 6], &[0.0; 3], Activation::Relu).unwrap();
        model.predict(&[1.0, 2.0]); // n_outputs=3, should panic
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[should_panic]
    fn predict_into_wrong_output_len() {
        let mut model = MlpF64::from_parts(&[1, 1], &[1.0], &[0.0], Activation::Relu).unwrap();
        let mut out = [0.0_f64; 2];
        model.predict_into(&[1.0], &mut out);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn identity_activation() {
        // 1 input → 1 hidden (identity) → 1 output
        // h0 = identity(1.0*x + 0.0) = x (no clipping)
        // o0 = 1.0*h0 + 0.0
        let mut model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, 0.0], Activation::Identity).unwrap();
        assert!((model.predict(&[5.0]) - 5.0).abs() < 1e-12);
        assert!((model.predict(&[-3.0]) - (-3.0)).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn elu_activation() {
        // 1 input → 1 hidden (elu alpha=1.0) → 1 output
        // h0 = elu(1.0*x + 0.0)
        // o0 = 1.0*h0 + 0.0
        let mut model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, 0.0], Activation::Elu(1.0)).unwrap();
        // Positive: passthrough
        assert!((model.predict(&[2.0]) - 2.0).abs() < 1e-12);
        // Negative: alpha * (exp(x) - 1)
        let expected = 1.0 * ((-1.0_f64).exp() - 1.0);
        assert!((model.predict(&[-1.0]) - expected).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn gelu_activation() {
        // 1 input → 1 hidden (gelu) → 1 output
        let mut model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, 0.0], Activation::Gelu).unwrap();
        // GELU(1.0) ≈ 0.8411920 (tanh approximation)
        let x = 1.0_f64;
        let expected =
            0.5 * x * (1.0 + (0.7978845608028654 * (0.044715 * x * x).mul_add(x, x)).tanh());
        assert!((model.predict(&[1.0]) - expected).abs() < 1e-12);
        // GELU(0) = 0
        assert!((model.predict(&[0.0]) - 0.0).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn swish_activation() {
        // 1 input → 1 hidden (swish) → 1 output
        let mut model =
            MlpF64::from_parts(&[1, 1, 1], &[1.0, 1.0], &[0.0, 0.0], Activation::Swish).unwrap();
        // Swish(2.0) = 2.0 * sigmoid(2.0) = 2.0 / (1 + exp(-2))
        let expected = 2.0 / (1.0 + (-2.0_f64).exp());
        assert!((model.predict(&[2.0]) - expected).abs() < 1e-12);
        // Swish(0) = 0
        assert!((model.predict(&[0.0]) - 0.0).abs() < 1e-12);
    }
}

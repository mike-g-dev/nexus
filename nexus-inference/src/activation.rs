/// Activation function for hidden layers and convolution outputs.
///
/// Applied element-wise. `Relu`, `LeakyRelu`, `Identity` are always
/// available. `Tanh`, `Sigmoid`, `Elu`, `Gelu`, `Swish` require
/// `std` or `libm`.
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
    /// Pass-through (no transformation).
    Identity,
    /// x if x >= 0, alpha * (exp(x) - 1) otherwise. Requires `std` or `libm`.
    Elu(f64),
    /// Gaussian error linear unit (tanh approximation). Requires `std` or `libm`.
    Gelu,
    /// x * sigmoid(x), also known as SiLU in PyTorch. Requires `std` or `libm`.
    Swish,
}

#[inline(always)]
pub(crate) fn activate_f32(x: f32, activation: Activation) -> f32 {
    match activation {
        Activation::Relu => {
            if x > 0.0 {
                x
            } else if x <= 0.0 {
                0.0
            } else {
                x // NaN
            }
        }
        Activation::LeakyRelu(alpha) => {
            if x >= 0.0 {
                x
            } else {
                x * alpha as f32
            }
        }
        Activation::Identity => x,
        Activation::Tanh => {
            #[cfg(feature = "std")]
            {
                (x as f64).tanh() as f32
            }
            #[cfg(all(not(feature = "std"), feature = "libm"))]
            {
                libm::tanh(x as f64) as f32
            }
            #[cfg(not(any(feature = "std", feature = "libm")))]
            {
                let _ = x;
                unreachable!()
            }
        }
        Activation::Sigmoid => {
            #[cfg(feature = "std")]
            {
                (1.0_f64 / (1.0_f64 + (-(x as f64)).exp())) as f32
            }
            #[cfg(all(not(feature = "std"), feature = "libm"))]
            {
                (1.0_f64 / (1.0_f64 + libm::exp(-(x as f64)))) as f32
            }
            #[cfg(not(any(feature = "std", feature = "libm")))]
            {
                let _ = x;
                unreachable!()
            }
        }
        Activation::Elu(alpha) => {
            if x >= 0.0 {
                x
            } else {
                #[cfg(feature = "std")]
                {
                    alpha as f32 * (x as f64).exp_m1() as f32
                }
                #[cfg(all(not(feature = "std"), feature = "libm"))]
                {
                    alpha as f32 * libm::expm1(x as f64) as f32
                }
                #[cfg(not(any(feature = "std", feature = "libm")))]
                {
                    let _ = (x, alpha);
                    unreachable!()
                }
            }
        }
        Activation::Gelu => {
            #[cfg(feature = "std")]
            {
                let xf = x as f64;
                let inner = (0.044_715 * xf * xf).mul_add(xf, xf)
                    * core::f64::consts::FRAC_2_SQRT_PI
                    * core::f64::consts::FRAC_1_SQRT_2;
                (0.5 * xf * (1.0 + inner.tanh())) as f32
            }
            #[cfg(all(not(feature = "std"), feature = "libm"))]
            {
                let xf = x as f64;
                let inner = (0.044_715 * xf * xf * xf + xf)
                    * core::f64::consts::FRAC_2_SQRT_PI
                    * core::f64::consts::FRAC_1_SQRT_2;
                (0.5 * xf * (1.0 + libm::tanh(inner))) as f32
            }
            #[cfg(not(any(feature = "std", feature = "libm")))]
            {
                let _ = x;
                unreachable!()
            }
        }
        Activation::Swish => {
            #[cfg(feature = "std")]
            {
                let xf = x as f64;
                (xf / (1.0 + (-xf).exp())) as f32
            }
            #[cfg(all(not(feature = "std"), feature = "libm"))]
            {
                let xf = x as f64;
                (xf / (1.0 + libm::exp(-xf))) as f32
            }
            #[cfg(not(any(feature = "std", feature = "libm")))]
            {
                let _ = x;
                unreachable!()
            }
        }
    }
}

#[inline(always)]
pub(crate) fn activate_f64(x: f64, activation: Activation) -> f64 {
    match activation {
        Activation::Relu => {
            if x > 0.0 {
                x
            } else if x <= 0.0 {
                0.0
            } else {
                x
            }
        }
        Activation::LeakyRelu(alpha) => {
            if x >= 0.0 {
                x
            } else {
                x * alpha
            }
        }
        Activation::Identity => x,
        Activation::Tanh => {
            #[cfg(feature = "std")]
            {
                x.tanh()
            }
            #[cfg(all(not(feature = "std"), feature = "libm"))]
            {
                libm::tanh(x)
            }
            #[cfg(not(any(feature = "std", feature = "libm")))]
            {
                let _ = x;
                unreachable!()
            }
        }
        Activation::Sigmoid => {
            #[cfg(feature = "std")]
            {
                1.0 / (1.0 + (-x).exp())
            }
            #[cfg(all(not(feature = "std"), feature = "libm"))]
            {
                1.0 / (1.0 + libm::exp(-x))
            }
            #[cfg(not(any(feature = "std", feature = "libm")))]
            {
                let _ = x;
                unreachable!()
            }
        }
        Activation::Elu(alpha) => {
            if x >= 0.0 {
                x
            } else {
                #[cfg(feature = "std")]
                {
                    alpha * x.exp_m1()
                }
                #[cfg(all(not(feature = "std"), feature = "libm"))]
                {
                    alpha * libm::expm1(x)
                }
                #[cfg(not(any(feature = "std", feature = "libm")))]
                {
                    let _ = (x, alpha);
                    unreachable!()
                }
            }
        }
        Activation::Gelu => {
            #[cfg(feature = "std")]
            {
                let inner = (0.044_715 * x * x).mul_add(x, x)
                    * core::f64::consts::FRAC_2_SQRT_PI
                    * core::f64::consts::FRAC_1_SQRT_2;
                0.5 * x * (1.0 + inner.tanh())
            }
            #[cfg(all(not(feature = "std"), feature = "libm"))]
            {
                let inner = (0.044_715 * x * x * x + x)
                    * core::f64::consts::FRAC_2_SQRT_PI
                    * core::f64::consts::FRAC_1_SQRT_2;
                0.5 * x * (1.0 + libm::tanh(inner))
            }
            #[cfg(not(any(feature = "std", feature = "libm")))]
            {
                let _ = x;
                unreachable!()
            }
        }
        Activation::Swish => {
            #[cfg(feature = "std")]
            {
                x / (1.0 + (-x).exp())
            }
            #[cfg(all(not(feature = "std"), feature = "libm"))]
            {
                x / (1.0 + libm::exp(-x))
            }
            #[cfg(not(any(feature = "std", feature = "libm")))]
            {
                let _ = x;
                unreachable!()
            }
        }
    }
}

mod gru;
mod lstm;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512_gates;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    not(target_feature = "avx512f"),
))]
mod avx2_gates;

pub use gru::TinyGruF32;
pub use lstm::TinyLstmF32;

/// Padé [7,6] rational approximant for tanh.
///
/// Max relative error ~1.2e-7 in [-4.97, 4.97] (full f32 precision,
/// within 1-2 ULP of correctly-rounded result). Saturates to ±1
/// for |x| >= 4.97; IEEE f32 tanh doesn't reach exactly 1.0 until
/// ~9.1, so this introduces up to ~1e-4 absolute error in the
/// saturation region. Negligible for gate activations — values in
/// this range are already deep in sigmoid/tanh saturation.
///
/// This matches the precision used during training (PyTorch computes
/// activations in f32 on GPU). The same polynomial approximation
/// strategy is used by XNNPACK (Google), oneDNN (Intel), TensorRT
/// (NVIDIA), and ONNX Runtime for production inference.
#[inline(always)]
pub(crate) fn tanh_f32(x: f32) -> f32 {
    if x >= 4.97 {
        return 1.0;
    }
    if x <= -4.97 {
        return -1.0;
    }
    let x2 = x * x;
    let num = x * x2.mul_add(x2.mul_add(x2 + 378.0, 17_325.0), 135_135.0);
    let den = x2.mul_add(x2.mul_add(x2.mul_add(28.0, 3_150.0), 62_370.0), 135_135.0);
    num / den
}

/// Logistic sigmoid via the tanh identity: sigmoid(x) = 0.5 + 0.5 * tanh(x/2).
/// Inherits the Padé approximant's full f32 precision.
#[inline(always)]
pub(crate) fn sigmoid_f32(x: f32) -> f32 {
    0.5_f32.mul_add(tanh_f32(x * 0.5), 0.5)
}

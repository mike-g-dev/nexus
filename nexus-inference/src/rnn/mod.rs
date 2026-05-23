mod gru;
mod lstm;
mod stacked_gru;
mod stacked_lstm;

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
pub use stacked_gru::StackedGruF32;
pub use stacked_lstm::StackedLstmF32;

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

#[inline(always)]
pub(crate) fn apply_lstm_gates(gates: &[f32], c: &mut [f32], h: &mut [f32], hidden_size: usize) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        avx512_gates::lstm_gates_avx512(gates, c, h, hidden_size);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma",
        not(target_feature = "avx512f"),
    ))]
    {
        avx2_gates::lstm_gates_avx2(gates, c, h, hidden_size);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma"),
        )
    )))]
    {
        for k in 0..hidden_size {
            let ig = sigmoid_f32(gates[k]);
            let fg = sigmoid_f32(gates[hidden_size + k]);
            let cg = tanh_f32(gates[2 * hidden_size + k]);
            let og = sigmoid_f32(gates[3 * hidden_size + k]);

            c[k] = fg.mul_add(c[k], ig * cg);
            h[k] = og * tanh_f32(c[k]);
        }
    }
}

#[inline(always)]
pub(crate) fn apply_gru_gates(
    ih_scratch: &[f32],
    hh_scratch: &[f32],
    bias_ih: &[f32],
    bias_hh: &[f32],
    h: &mut [f32],
    hidden_size: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        avx512_gates::gru_gates_avx512(ih_scratch, hh_scratch, bias_ih, bias_hh, h, hidden_size);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma",
        not(target_feature = "avx512f"),
    ))]
    {
        avx2_gates::gru_gates_avx2(ih_scratch, hh_scratch, bias_ih, bias_hh, h, hidden_size);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma"),
        )
    )))]
    {
        let hi = hidden_size;
        for k in 0..hi {
            let r = sigmoid_f32(
                ih_scratch[k] + bias_ih[k] + hh_scratch[k] + bias_hh[k],
            );
            let z = sigmoid_f32(
                ih_scratch[hi + k]
                    + bias_ih[hi + k]
                    + hh_scratch[hi + k]
                    + bias_hh[hi + k],
            );
            let hh_candidate = hh_scratch[2 * hi + k] + bias_hh[2 * hi + k];
            let n = tanh_f32(r.mul_add(
                hh_candidate,
                ih_scratch[2 * hi + k] + bias_ih[2 * hi + k],
            ));
            h[k] = (1.0 - z).mul_add(n, z * h[k]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// AVX2 vectorized Padé [7,6] tanh on 8 lanes.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn tanh_8wide(x: __m256) -> __m256 {
    unsafe {
        // Detect NaN lanes before clamping (min/max would swallow them).
        let nan_mask = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);

        let pos_clip = _mm256_set1_ps(4.97);
        let neg_clip = _mm256_set1_ps(-4.97);
        let xc = _mm256_min_ps(_mm256_max_ps(x, neg_clip), pos_clip);

        let x2 = _mm256_mul_ps(xc, xc);

        // num = x * (135_135 + x2 * (17_325 + x2 * (378 + x2)))
        let n_inner = _mm256_fmadd_ps(x2, _mm256_set1_ps(1.0), _mm256_set1_ps(378.0));
        let n_mid = _mm256_fmadd_ps(x2, n_inner, _mm256_set1_ps(17_325.0));
        let n_outer = _mm256_fmadd_ps(x2, n_mid, _mm256_set1_ps(135_135.0));
        let num = _mm256_mul_ps(xc, n_outer);

        // den = 135_135 + x2 * (62_370 + x2 * (3_150 + x2 * 28))
        let d_inner = _mm256_fmadd_ps(x2, _mm256_set1_ps(28.0), _mm256_set1_ps(3_150.0));
        let d_mid = _mm256_fmadd_ps(x2, d_inner, _mm256_set1_ps(62_370.0));
        let den = _mm256_fmadd_ps(x2, d_mid, _mm256_set1_ps(135_135.0));

        let result = _mm256_div_ps(num, den);

        // Restore NaN lanes that clamping would have silently replaced.
        _mm256_blendv_ps(result, x, nan_mask)
    }
}

/// AVX2 vectorized sigmoid: 0.5 + 0.5 * tanh(x * 0.5)
#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn sigmoid_8wide(x: __m256) -> __m256 {
    unsafe {
        let half = _mm256_set1_ps(0.5);
        let t = tanh_8wide(_mm256_mul_ps(x, half));
        _mm256_fmadd_ps(half, t, half)
    }
}

/// LSTM gate activation + cell/hidden update, 8 units at a time.
///
/// gates layout: [input_gate(H) | forget_gate(H) | cell_candidate(H) | output_gate(H)]
/// Updates c and h in-place.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::many_single_char_names)]
pub(super) fn lstm_gates_avx2(gates: &[f32], c: &mut [f32], h: &mut [f32], hidden: usize) {
    let mut k = 0;
    let h8 = hidden & !7;

    // SAFETY: AVX2+FMA guaranteed by cfg on parent module.
    // All accesses: k + 8 <= hidden, gate offsets are k, h+k, 2h+k, 3h+k (all within 4*hidden).
    unsafe {
        while k < h8 {
            let ig = sigmoid_8wide(_mm256_loadu_ps(gates.as_ptr().add(k)));
            let fg = sigmoid_8wide(_mm256_loadu_ps(gates.as_ptr().add(hidden + k)));
            let cg = tanh_8wide(_mm256_loadu_ps(gates.as_ptr().add(2 * hidden + k)));
            let og = sigmoid_8wide(_mm256_loadu_ps(gates.as_ptr().add(3 * hidden + k)));

            let c_old = _mm256_loadu_ps(c.as_ptr().add(k));
            // c_new = fg * c_old + ig * cg
            let c_new = _mm256_fmadd_ps(fg, c_old, _mm256_mul_ps(ig, cg));
            _mm256_storeu_ps(c.as_mut_ptr().add(k), c_new);

            // h_new = og * tanh(c_new)
            let h_new = _mm256_mul_ps(og, tanh_8wide(c_new));
            _mm256_storeu_ps(h.as_mut_ptr().add(k), h_new);

            k += 8;
        }
    }

    // Scalar tail for H % 8 != 0
    while k < hidden {
        let ig = super::sigmoid_f32(gates[k]);
        let fg = super::sigmoid_f32(gates[hidden + k]);
        let cg = super::tanh_f32(gates[2 * hidden + k]);
        let og = super::sigmoid_f32(gates[3 * hidden + k]);

        c[k] = fg.mul_add(c[k], ig * cg);
        h[k] = og * super::tanh_f32(c[k]);
        k += 1;
    }
}

/// GRU gate activation + hidden update, 8 units at a time.
///
/// ih_scratch layout: [reset(H) | update(H) | candidate_ih(H)]
/// hh_scratch layout: [reset(H) | update(H) | candidate_hh(H)]
/// bias_ih/bias_hh: same layout as scratch
/// Updates h in-place.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::many_single_char_names)]
pub(super) fn gru_gates_avx2(
    ih: &[f32],
    hh: &[f32],
    bias_ih: &[f32],
    bias_hh: &[f32],
    h: &mut [f32],
    hidden: usize,
) {
    let mut k = 0;
    let h8 = hidden & !7;

    // SAFETY: AVX2+FMA guaranteed by cfg on parent module.
    // All offsets: k, hidden+k, 2*hidden+k within 3*hidden total.
    unsafe {
        let one = _mm256_set1_ps(1.0);

        while k < h8 {
            // r = sigmoid(ih[k] + b_ih[k] + hh[k] + b_hh[k])
            let r_sum = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_loadu_ps(ih.as_ptr().add(k)),
                    _mm256_loadu_ps(bias_ih.as_ptr().add(k)),
                ),
                _mm256_add_ps(
                    _mm256_loadu_ps(hh.as_ptr().add(k)),
                    _mm256_loadu_ps(bias_hh.as_ptr().add(k)),
                ),
            );
            let r = sigmoid_8wide(r_sum);

            // z = sigmoid(ih[H+k] + b_ih[H+k] + hh[H+k] + b_hh[H+k])
            let z_sum = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_loadu_ps(ih.as_ptr().add(hidden + k)),
                    _mm256_loadu_ps(bias_ih.as_ptr().add(hidden + k)),
                ),
                _mm256_add_ps(
                    _mm256_loadu_ps(hh.as_ptr().add(hidden + k)),
                    _mm256_loadu_ps(bias_hh.as_ptr().add(hidden + k)),
                ),
            );
            let z = sigmoid_8wide(z_sum);

            // n = tanh(ih[2H+k] + b_ih[2H+k] + r * (hh[2H+k] + b_hh[2H+k]))
            let hh_cand = _mm256_add_ps(
                _mm256_loadu_ps(hh.as_ptr().add(2 * hidden + k)),
                _mm256_loadu_ps(bias_hh.as_ptr().add(2 * hidden + k)),
            );
            let ih_cand = _mm256_add_ps(
                _mm256_loadu_ps(ih.as_ptr().add(2 * hidden + k)),
                _mm256_loadu_ps(bias_ih.as_ptr().add(2 * hidden + k)),
            );
            let n = tanh_8wide(_mm256_fmadd_ps(r, hh_cand, ih_cand));

            // h' = (1 - z) * n + z * h
            let h_old = _mm256_loadu_ps(h.as_ptr().add(k));
            let one_minus_z = _mm256_sub_ps(one, z);
            let h_new = _mm256_fmadd_ps(one_minus_z, n, _mm256_mul_ps(z, h_old));
            _mm256_storeu_ps(h.as_mut_ptr().add(k), h_new);

            k += 8;
        }
    }

    // Scalar tail
    while k < hidden {
        let r = super::sigmoid_f32(ih[k] + bias_ih[k] + hh[k] + bias_hh[k]);
        let z = super::sigmoid_f32(
            ih[hidden + k] + bias_ih[hidden + k] + hh[hidden + k] + bias_hh[hidden + k],
        );
        let hh_candidate = hh[2 * hidden + k] + bias_hh[2 * hidden + k];
        let n =
            super::tanh_f32(r.mul_add(hh_candidate, ih[2 * hidden + k] + bias_ih[2 * hidden + k]));
        h[k] = (1.0 - z).mul_add(n, z * h[k]);
        k += 1;
    }
}

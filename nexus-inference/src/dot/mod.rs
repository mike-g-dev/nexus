#[allow(dead_code)]
mod scalar;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    not(target_feature = "avx512f"),
))]
mod avx2;

#[inline]
pub(crate) fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        avx512::dot_f64(a, b)
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma",
        not(target_feature = "avx512f"),
    ))]
    {
        avx2::dot_f64(a, b)
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma"),
        )
    )))]
    {
        scalar::dot_f64(a, b)
    }
}

#[inline]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        avx512::dot_f32(a, b)
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma",
        not(target_feature = "avx512f"),
    ))]
    {
        avx2::dot_f32(a, b)
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma"),
        )
    )))]
    {
        scalar::dot_f32(a, b)
    }
}

/// Compute 4 dot products simultaneously: dot(rows[k*n..], input) for k in 0..4.
/// `rows` layout: [row0 | row1 | row2 | row3], each row has `input.len()` elements.
#[inline]
pub(crate) fn dot4_f64(rows: &[f64], input: &[f64]) -> [f64; 4] {
    debug_assert_eq!(rows.len(), 4 * input.len());

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        avx512::dot4_f64(rows, input)
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma",
        not(target_feature = "avx512f"),
    ))]
    {
        avx2::dot4_f64(rows, input)
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma"),
        )
    )))]
    {
        scalar::dot4_f64(rows, input)
    }
}

/// Matrix-vector product: output[j] = bias[j] + dot(weight[j], input).
///
/// `weight` is `(out_size, in_size)` row-major.
#[inline]
pub(crate) fn matvec_bias_f32(
    weight: &[f32],
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
    out_size: usize,
    in_size: usize,
) {
    let out_8 = out_size & !7;
    let out_4 = out_size & !3;
    let mut j = 0;

    #[cfg(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma"),
        )
    ))]
    {
        use core::arch::x86_64::*;
        // SAFETY: cfg guarantees SIMD availability.
        // All j + N <= out_size within respective loops; bias/output in bounds.
        unsafe {
            // dot8 reduction is heavier than dot4; only worthwhile when the
            // inner dimension is long enough to amortize it.
            if in_size >= 32 {
                while j < out_8 {
                    let rows = &weight[j * in_size..(j + 8) * in_size];
                    let dots = dot8_f32_m256(rows, &input[..in_size]);
                    let bias_v = _mm256_loadu_ps(bias.as_ptr().add(j));
                    _mm256_storeu_ps(output.as_mut_ptr().add(j), _mm256_add_ps(dots, bias_v));
                    j += 8;
                }
            }
            while j < out_4 {
                let rows = &weight[j * in_size..(j + 4) * in_size];
                let dots = dot4_f32_m128(rows, &input[..in_size]);
                let bias_v = _mm_loadu_ps(bias.as_ptr().add(j));
                _mm_storeu_ps(output.as_mut_ptr().add(j), _mm_add_ps(dots, bias_v));
                j += 4;
            }
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma"),
        )
    )))]
    {
        while j < out_4 {
            let rows = &weight[j * in_size..(j + 4) * in_size];
            let dots = dot4_f32(rows, &input[..in_size]);
            output[j] = bias[j] + dots[0];
            output[j + 1] = bias[j + 1] + dots[1];
            output[j + 2] = bias[j + 2] + dots[2];
            output[j + 3] = bias[j + 3] + dots[3];
            j += 4;
        }
    }

    while j < out_size {
        let row = &weight[j * in_size..(j + 1) * in_size];
        output[j] = bias[j] + dot_f32(row, &input[..in_size]);
        j += 1;
    }
}

/// Matrix-vector product without bias: output[j] = dot(weight[j], input).
///
/// `weight` is `(out_size, in_size)` row-major.
#[inline]
// Used by GRU (gated on std|libm), but dot module compiles under alloc alone.
#[allow(dead_code)]
pub(crate) fn matvec_f32(
    weight: &[f32],
    input: &[f32],
    output: &mut [f32],
    out_size: usize,
    in_size: usize,
) {
    let out_8 = out_size & !7;
    let out_4 = out_size & !3;
    let mut j = 0;

    #[cfg(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma"),
        )
    ))]
    {
        use core::arch::x86_64::*;
        // SAFETY: cfg guarantees SIMD availability.
        // All j + N <= out_size within respective loops; output in bounds.
        unsafe {
            if in_size >= 32 {
                while j < out_8 {
                    let rows = &weight[j * in_size..(j + 8) * in_size];
                    let dots = dot8_f32_m256(rows, &input[..in_size]);
                    _mm256_storeu_ps(output.as_mut_ptr().add(j), dots);
                    j += 8;
                }
            }
            while j < out_4 {
                let rows = &weight[j * in_size..(j + 4) * in_size];
                let dots = dot4_f32_m128(rows, &input[..in_size]);
                _mm_storeu_ps(output.as_mut_ptr().add(j), dots);
                j += 4;
            }
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma"),
        )
    )))]
    {
        while j < out_4 {
            let rows = &weight[j * in_size..(j + 4) * in_size];
            let dots = dot4_f32(rows, &input[..in_size]);
            output[j] = dots[0];
            output[j + 1] = dots[1];
            output[j + 2] = dots[2];
            output[j + 3] = dots[3];
            j += 4;
        }
    }

    while j < out_size {
        let row = &weight[j * in_size..(j + 1) * in_size];
        output[j] = dot_f32(row, &input[..in_size]);
        j += 1;
    }
}

/// Compute 4 dot products simultaneously: dot(rows[k*n..], input) for k in 0..4.
/// `rows` layout: [row0 | row1 | row2 | row3], each row has `input.len()` elements.
#[inline]
pub(crate) fn dot4_f32(rows: &[f32], input: &[f32]) -> [f32; 4] {
    debug_assert_eq!(rows.len(), 4 * input.len());

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        avx512::dot4_f32(rows, input)
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma",
        not(target_feature = "avx512f"),
    ))]
    {
        avx2::dot4_f32(rows, input)
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        any(
            target_feature = "avx512f",
            all(target_feature = "avx2", target_feature = "fma"),
        )
    )))]
    {
        scalar::dot4_f32(rows, input)
    }
}

/// 8 simultaneous f32 dot products packed in `__m256`.
/// `rows` layout: [row0 | row1 | ... | row7], each row has `input.len()` elements.
#[cfg(all(
    target_arch = "x86_64",
    any(
        target_feature = "avx512f",
        all(target_feature = "avx2", target_feature = "fma"),
    )
))]
#[inline]
pub(crate) fn dot8_f32_m256(
    rows: &[f32],
    input: &[f32],
) -> core::arch::x86_64::__m256 {
    debug_assert_eq!(rows.len(), 8 * input.len());

    #[cfg(target_feature = "avx512f")]
    {
        avx512::dot8_f32_m256(rows, input)
    }

    #[cfg(all(
        target_feature = "avx2",
        target_feature = "fma",
        not(target_feature = "avx512f"),
    ))]
    {
        avx2::dot8_f32_m256(rows, input)
    }
}

/// Like [`dot4_f32`] but returns results packed in `__m128` to avoid
/// per-lane horizontal sum overhead. Only available on x86_64 with
/// AVX2+FMA or AVX-512F.
#[cfg(all(
    target_arch = "x86_64",
    any(
        target_feature = "avx512f",
        all(target_feature = "avx2", target_feature = "fma"),
    )
))]
#[inline]
pub(crate) fn dot4_f32_m128(
    rows: &[f32],
    input: &[f32],
) -> core::arch::x86_64::__m128 {
    debug_assert_eq!(rows.len(), 4 * input.len());

    #[cfg(target_feature = "avx512f")]
    {
        avx512::dot4_f32_m128(rows, input)
    }

    #[cfg(all(
        target_feature = "avx2",
        target_feature = "fma",
        not(target_feature = "avx512f"),
    ))]
    {
        avx2::dot4_f32_m128(rows, input)
    }
}

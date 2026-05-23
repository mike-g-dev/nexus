#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::scalar;

#[inline]
#[cfg(target_arch = "x86_64")]
unsafe fn hsum_f64(v: __m512d) -> f64 {
    unsafe {
        let hi = _mm512_extractf64x4_pd(v, 1);
        let lo = _mm512_castpd512_pd256(v);
        let sum = _mm256_add_pd(lo, hi);
        let hi128 = _mm256_extractf128_pd(sum, 1);
        let lo128 = _mm256_castpd256_pd128(sum);
        let pair = _mm_add_pd(lo128, hi128);
        let high_lane = _mm_unpackhi_pd(pair, pair);
        _mm_cvtsd_f64(_mm_add_sd(pair, high_lane))
    }
}

#[inline]
#[cfg(target_arch = "x86_64")]
unsafe fn hsum_f32(v: __m512) -> f32 {
    unsafe {
        let lo = _mm512_castps512_ps256(v);
        let hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
        let sum = _mm256_add_ps(lo, hi);
        let hi128 = _mm256_extractf128_ps(sum, 1);
        let lo128 = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    }
}

#[inline]
#[cfg(target_arch = "x86_64")]
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut i = 0;

    // SAFETY: AVX-512F guaranteed by target_feature cfg on parent module.
    // All pointer offsets satisfy i + N <= len before access.
    let sum = unsafe {
        let mut acc0 = _mm512_setzero_pd();
        let mut acc1 = _mm512_setzero_pd();
        let mut acc2 = _mm512_setzero_pd();
        let mut acc3 = _mm512_setzero_pd();

        while i + 32 <= len {
            let a0 = _mm512_loadu_pd(a.as_ptr().add(i));
            let b0 = _mm512_loadu_pd(b.as_ptr().add(i));
            let a1 = _mm512_loadu_pd(a.as_ptr().add(i + 8));
            let b1 = _mm512_loadu_pd(b.as_ptr().add(i + 8));
            let a2 = _mm512_loadu_pd(a.as_ptr().add(i + 16));
            let b2 = _mm512_loadu_pd(b.as_ptr().add(i + 16));
            let a3 = _mm512_loadu_pd(a.as_ptr().add(i + 24));
            let b3 = _mm512_loadu_pd(b.as_ptr().add(i + 24));
            acc0 = _mm512_fmadd_pd(a0, b0, acc0);
            acc1 = _mm512_fmadd_pd(a1, b1, acc1);
            acc2 = _mm512_fmadd_pd(a2, b2, acc2);
            acc3 = _mm512_fmadd_pd(a3, b3, acc3);
            i += 32;
        }

        while i + 8 <= len {
            let av = _mm512_loadu_pd(a.as_ptr().add(i));
            let bv = _mm512_loadu_pd(b.as_ptr().add(i));
            acc0 = _mm512_fmadd_pd(av, bv, acc0);
            i += 8;
        }

        acc0 = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));

        hsum_f64(acc0)
    };

    sum + scalar::dot_f64(&a[i..], &b[i..])
}

#[inline]
#[cfg(target_arch = "x86_64")]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;

    // SAFETY: AVX-512F guaranteed by target_feature cfg on parent module.
    // All pointer offsets satisfy i + N <= len before access.
    let sum = unsafe {
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        while i + 64 <= len {
            let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
            let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
            let a1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
            let b1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
            let a2 = _mm512_loadu_ps(a.as_ptr().add(i + 32));
            let b2 = _mm512_loadu_ps(b.as_ptr().add(i + 32));
            let a3 = _mm512_loadu_ps(a.as_ptr().add(i + 48));
            let b3 = _mm512_loadu_ps(b.as_ptr().add(i + 48));
            acc0 = _mm512_fmadd_ps(a0, b0, acc0);
            acc1 = _mm512_fmadd_ps(a1, b1, acc1);
            acc2 = _mm512_fmadd_ps(a2, b2, acc2);
            acc3 = _mm512_fmadd_ps(a3, b3, acc3);
            i += 64;
        }

        while i + 16 <= len {
            let av = _mm512_loadu_ps(a.as_ptr().add(i));
            let bv = _mm512_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm512_fmadd_ps(av, bv, acc0);
            i += 16;
        }

        acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));

        hsum_f32(acc0)
    };

    sum + scalar::dot_f32(&a[i..], &b[i..])
}

/// 4 simultaneous f64 dot products sharing input loads.
/// 2 accumulators per neuron (8 total) to hide FMA latency.
#[inline]
#[cfg(target_arch = "x86_64")]
pub fn dot4_f64(rows: &[f64], input: &[f64]) -> [f64; 4] {
    let in_size = input.len();
    let mut i = 0;

    // SAFETY: AVX-512F guaranteed by target_feature cfg on parent module.
    // Row pointers: rows.len() == 4 * in_size, offsets k * in_size + i where k < 4.
    // Input pointer: i + N <= in_size before every access.
    let sums = unsafe {
        let mut a0a = _mm512_setzero_pd();
        let mut a0b = _mm512_setzero_pd();
        let mut a1a = _mm512_setzero_pd();
        let mut a1b = _mm512_setzero_pd();
        let mut a2a = _mm512_setzero_pd();
        let mut a2b = _mm512_setzero_pd();
        let mut a3a = _mm512_setzero_pd();
        let mut a3b = _mm512_setzero_pd();

        let r0 = rows.as_ptr();
        let r1 = r0.add(in_size);
        let r2 = r1.add(in_size);
        let r3 = r2.add(in_size);
        let inp = input.as_ptr();

        while i + 16 <= in_size {
            let x0 = _mm512_loadu_pd(inp.add(i));
            let x1 = _mm512_loadu_pd(inp.add(i + 8));

            a0a = _mm512_fmadd_pd(_mm512_loadu_pd(r0.add(i)), x0, a0a);
            a0b = _mm512_fmadd_pd(_mm512_loadu_pd(r0.add(i + 8)), x1, a0b);
            a1a = _mm512_fmadd_pd(_mm512_loadu_pd(r1.add(i)), x0, a1a);
            a1b = _mm512_fmadd_pd(_mm512_loadu_pd(r1.add(i + 8)), x1, a1b);
            a2a = _mm512_fmadd_pd(_mm512_loadu_pd(r2.add(i)), x0, a2a);
            a2b = _mm512_fmadd_pd(_mm512_loadu_pd(r2.add(i + 8)), x1, a2b);
            a3a = _mm512_fmadd_pd(_mm512_loadu_pd(r3.add(i)), x0, a3a);
            a3b = _mm512_fmadd_pd(_mm512_loadu_pd(r3.add(i + 8)), x1, a3b);

            i += 16;
        }

        if i + 8 <= in_size {
            let x = _mm512_loadu_pd(inp.add(i));
            a0a = _mm512_fmadd_pd(_mm512_loadu_pd(r0.add(i)), x, a0a);
            a1a = _mm512_fmadd_pd(_mm512_loadu_pd(r1.add(i)), x, a1a);
            a2a = _mm512_fmadd_pd(_mm512_loadu_pd(r2.add(i)), x, a2a);
            a3a = _mm512_fmadd_pd(_mm512_loadu_pd(r3.add(i)), x, a3a);
            i += 8;
        }

        a0a = _mm512_add_pd(a0a, a0b);
        a1a = _mm512_add_pd(a1a, a1b);
        a2a = _mm512_add_pd(a2a, a2b);
        a3a = _mm512_add_pd(a3a, a3b);

        [hsum_f64(a0a), hsum_f64(a1a), hsum_f64(a2a), hsum_f64(a3a)]
    };

    let mut out = sums;
    for k in i..in_size {
        let x = input[k];
        out[0] += rows[k] * x;
        out[1] += rows[in_size + k] * x;
        out[2] += rows[2 * in_size + k] * x;
        out[3] += rows[3 * in_size + k] * x;
    }
    out
}

/// 4 simultaneous f32 dot products sharing input loads.
/// 2 accumulators per neuron (8 total) to hide FMA latency.
#[inline]
#[cfg(target_arch = "x86_64")]
pub fn dot4_f32(rows: &[f32], input: &[f32]) -> [f32; 4] {
    let in_size = input.len();
    let mut i = 0;

    // SAFETY: AVX-512F guaranteed by target_feature cfg on parent module.
    // Row pointers: rows.len() == 4 * in_size, offsets k * in_size + i where k < 4.
    // Input pointer: i + N <= in_size before every access.
    let sums = unsafe {
        let mut a0a = _mm512_setzero_ps();
        let mut a0b = _mm512_setzero_ps();
        let mut a1a = _mm512_setzero_ps();
        let mut a1b = _mm512_setzero_ps();
        let mut a2a = _mm512_setzero_ps();
        let mut a2b = _mm512_setzero_ps();
        let mut a3a = _mm512_setzero_ps();
        let mut a3b = _mm512_setzero_ps();

        let r0 = rows.as_ptr();
        let r1 = r0.add(in_size);
        let r2 = r1.add(in_size);
        let r3 = r2.add(in_size);
        let inp = input.as_ptr();

        while i + 32 <= in_size {
            let x0 = _mm512_loadu_ps(inp.add(i));
            let x1 = _mm512_loadu_ps(inp.add(i + 16));

            a0a = _mm512_fmadd_ps(_mm512_loadu_ps(r0.add(i)), x0, a0a);
            a0b = _mm512_fmadd_ps(_mm512_loadu_ps(r0.add(i + 16)), x1, a0b);
            a1a = _mm512_fmadd_ps(_mm512_loadu_ps(r1.add(i)), x0, a1a);
            a1b = _mm512_fmadd_ps(_mm512_loadu_ps(r1.add(i + 16)), x1, a1b);
            a2a = _mm512_fmadd_ps(_mm512_loadu_ps(r2.add(i)), x0, a2a);
            a2b = _mm512_fmadd_ps(_mm512_loadu_ps(r2.add(i + 16)), x1, a2b);
            a3a = _mm512_fmadd_ps(_mm512_loadu_ps(r3.add(i)), x0, a3a);
            a3b = _mm512_fmadd_ps(_mm512_loadu_ps(r3.add(i + 16)), x1, a3b);

            i += 32;
        }

        if i + 16 <= in_size {
            let x = _mm512_loadu_ps(inp.add(i));
            a0a = _mm512_fmadd_ps(_mm512_loadu_ps(r0.add(i)), x, a0a);
            a1a = _mm512_fmadd_ps(_mm512_loadu_ps(r1.add(i)), x, a1a);
            a2a = _mm512_fmadd_ps(_mm512_loadu_ps(r2.add(i)), x, a2a);
            a3a = _mm512_fmadd_ps(_mm512_loadu_ps(r3.add(i)), x, a3a);
            i += 16;
        }

        a0a = _mm512_add_ps(a0a, a0b);
        a1a = _mm512_add_ps(a1a, a1b);
        a2a = _mm512_add_ps(a2a, a2b);
        a3a = _mm512_add_ps(a3a, a3b);

        [hsum_f32(a0a), hsum_f32(a1a), hsum_f32(a2a), hsum_f32(a3a)]
    };

    let mut out = sums;
    for k in i..in_size {
        let x = input[k];
        out[0] += rows[k] * x;
        out[1] += rows[in_size + k] * x;
        out[2] += rows[2 * in_size + k] * x;
        out[3] += rows[3 * in_size + k] * x;
    }
    out
}

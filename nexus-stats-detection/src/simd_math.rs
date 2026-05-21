//! Vectorized math for BOCPD inner loops.
//!
//! Dispatches to AVX2 (4 f64 at a time) when compiled with `+avx2`,
//! otherwise falls back to scalar. Same fdlibm/musl coefficients,
//! same ~15-digit precision, just in packed form.

#[allow(dead_code)]
mod scalar;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod avx2;

/// Compute ln(x) in-place for each element. All values must be positive.
#[inline]
pub fn ln_inplace(buf: &mut [f64]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        avx2::ln_inplace(buf);
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        scalar::ln_inplace(buf);
    }
}

/// Compute sum of exp(x - offset) for each element.
#[inline]
pub fn exp_sum(buf: &[f64], offset: f64) -> f64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        avx2::exp_sum(buf, offset)
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        scalar::exp_sum(buf, offset)
    }
}

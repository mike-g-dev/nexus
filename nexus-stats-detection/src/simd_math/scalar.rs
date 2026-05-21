use nexus_stats_core::math::{exp, ln};

#[inline]
pub fn ln_inplace(buf: &mut [f64]) {
    for v in buf.iter_mut() {
        *v = ln(*v);
    }
}

#[inline]
pub fn exp_sum(buf: &[f64], offset: f64) -> f64 {
    let mut sum = 0.0;
    for &v in buf {
        sum += exp(v - offset);
    }
    sum
}

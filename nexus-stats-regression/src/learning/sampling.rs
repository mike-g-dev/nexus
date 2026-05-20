macro_rules! impl_sampling {
    ($mod_name:ident, $ty:ty) => {
        pub(super) mod $mod_name {
            /// Marsaglia polar method. Returns one standard normal sample.
            /// Consumes at least two calls to `rng` per sample.
            #[allow(clippy::cast_possible_truncation, clippy::suboptimal_flops)]
            pub fn normal_sample(rng: &mut impl FnMut() -> $ty) -> $ty {
                loop {
                    let u = 2.0 as $ty * rng() - 1.0 as $ty;
                    let v = 2.0 as $ty * rng() - 1.0 as $ty;
                    let s = u * u + v * v;
                    if s > 0.0 as $ty && s < 1.0 as $ty {
                        let ln_s = nexus_stats_core::math::ln(s as f64) as $ty;
                        let factor =
                            nexus_stats_core::math::sqrt((-(2.0 as $ty) * ln_s / s) as f64) as $ty;
                        return u * factor;
                    }
                }
            }

            /// Marsaglia-Tsang method for Gamma(shape, 1). Shape must be > 0.
            /// For shape < 1: Gamma(shape) = Gamma(shape+1) * U^(1/shape).
            #[allow(clippy::cast_possible_truncation, clippy::suboptimal_flops)]
            pub fn gamma_sample(shape: $ty, rng: &mut impl FnMut() -> $ty) -> $ty {
                if shape < 1.0 as $ty {
                    let g = gamma_sample(shape + 1.0 as $ty, rng);
                    let u = rng();
                    let pow = nexus_stats_core::math::exp(
                        nexus_stats_core::math::ln(u as f64) / shape as f64,
                    ) as $ty;
                    return g * pow;
                }

                let d = shape - 1.0 as $ty / 3.0 as $ty;
                let c = (1.0 / nexus_stats_core::math::sqrt((9.0 as $ty * d) as f64)) as $ty;

                loop {
                    let x = normal_sample(rng);
                    let v_base = 1.0 as $ty + c * x;
                    if v_base <= 0.0 as $ty {
                        continue;
                    }
                    let v = v_base * v_base * v_base;
                    let u = rng();
                    let x2 = x * x;

                    if u < 1.0 as $ty - 0.0331 as $ty * x2 * x2 {
                        return d * v;
                    }

                    let ln_u = nexus_stats_core::math::ln(u as f64) as $ty;
                    let ln_v = nexus_stats_core::math::ln(v as f64) as $ty;
                    if ln_u < 0.5 as $ty * x2 + d * (1.0 as $ty - v + ln_v) {
                        return d * v;
                    }
                }
            }

            /// Beta(alpha, beta) via ratio of two Gamma samples.
            #[allow(clippy::float_cmp)]
            pub fn beta_sample(alpha: $ty, beta: $ty, rng: &mut impl FnMut() -> $ty) -> $ty {
                let x = gamma_sample(alpha, rng);
                let y = gamma_sample(beta, rng);
                let sum = x + y;
                if sum == 0.0 as $ty {
                    return 0.5 as $ty;
                }
                x / sum
            }
        }
    };
}

impl_sampling!(f64_impl, f64);
impl_sampling!(f32_impl, f32);

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng_f64(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        }
    }

    #[test]
    fn normal_sample_distribution() {
        let mut rng = make_rng_f64(42);
        let n = 10_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let x = f64_impl::normal_sample(&mut rng);
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean={mean}, expected ~0");
        assert!((var - 1.0).abs() < 0.1, "var={var}, expected ~1");
    }

    #[test]
    fn gamma_sample_mean() {
        let mut rng = make_rng_f64(123);
        let shape = 5.0;
        let n = 10_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let x = f64_impl::gamma_sample(shape, &mut rng);
            assert!(x > 0.0, "gamma sample must be positive");
            sum += x;
        }
        let mean = sum / n as f64;
        assert!((mean - shape).abs() < 0.3, "mean={mean}, expected ~{shape}");
    }

    #[test]
    fn gamma_sample_shape_less_than_one() {
        let mut rng = make_rng_f64(77);
        let shape = 0.5;
        let n = 5_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let x = f64_impl::gamma_sample(shape, &mut rng);
            assert!(x >= 0.0, "gamma sample must be non-negative");
            sum += x;
        }
        let mean = sum / n as f64;
        assert!((mean - shape).abs() < 0.1, "mean={mean}, expected ~{shape}");
    }

    #[test]
    fn beta_sample_range() {
        let mut rng = make_rng_f64(99);
        for _ in 0..1_000 {
            let x = f64_impl::beta_sample(2.0, 5.0, &mut rng);
            assert!((0.0..=1.0).contains(&x), "beta sample {x} out of [0,1]");
        }
    }

    #[test]
    fn beta_sample_mean() {
        let mut rng = make_rng_f64(55);
        let alpha = 2.0;
        let beta = 5.0;
        let n = 10_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += f64_impl::beta_sample(alpha, beta, &mut rng);
        }
        let mean = sum / n as f64;
        let expected = alpha / (alpha + beta);
        assert!(
            (mean - expected).abs() < 0.02,
            "mean={mean}, expected ~{expected}"
        );
    }

    #[test]
    fn f32_normal_sample() {
        let mut state: u64 = 42;
        let mut rng = || -> f32 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f32 / (1u64 << 31) as f32
        };
        let mut sum = 0.0_f32;
        for _ in 0..1_000 {
            let x = f32_impl::normal_sample(&mut rng);
            assert!(x.is_finite());
            sum += x;
        }
        let mean = sum / 1000.0;
        assert!(mean.abs() < 0.2, "f32 mean={mean}");
    }
}

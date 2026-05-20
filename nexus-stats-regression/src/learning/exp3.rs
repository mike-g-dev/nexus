extern crate alloc;
use alloc::boxed::Box;
use alloc::vec;

macro_rules! impl_exp3 {
    ($name:ident, $builder:ident, $ty:ty) => {
        /// EXP3 adversarial bandit.
        ///
        /// Exponential-weight algorithm for Exploration and Exploitation.
        /// Unlike UCB1/Thompson which assume stochastic rewards, EXP3
        /// makes no assumptions about how rewards are generated — they
        /// can be adversarial.
        ///
        /// Selection probability: `p_i = (1 - gamma) * w_i / sum(w) + gamma / K`
        ///
        /// Weight update: `w_i *= exp(eta * reward / (K * p_i))`
        ///
        /// Auer, Cesa-Bianchi, Freund, Schapire (2002).
        ///
        /// # Parameters
        ///
        /// - `arms` — number of arms K (>= 2)
        /// - `gamma` — exploration mixing rate, in (0, 1]. Higher = more uniform.
        /// - `eta` — learning rate (default: gamma / K)
        ///
        /// # Reward range
        ///
        /// Rewards must be in [0, 1]. Normalize before feeding.
        ///
        /// # Examples
        ///
        /// ```
        /// use nexus_stats_regression::learning::Exp3F64;
        ///
        /// let mut bandit = Exp3F64::builder()
        ///     .arms(3)
        ///     .gamma(0.1)
        ///     .build()
        ///     .unwrap();
        ///
        /// let mut s: u64 = 42;
        /// let mut rng = || -> f64 {
        ///     s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ///     (s >> 33) as f64 / (1u64 << 31) as f64
        /// };
        /// let (arm, prob) = bandit.select(&mut rng);
        /// bandit.update(arm, 0.8, prob).unwrap();
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            weights: Box<[$ty]>,
            gamma: $ty,
            eta: $ty,
            total_pulls: u64,
            num_arms: usize,
            min_samples: u64,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            arms: Option<usize>,
            gamma: Option<$ty>,
            eta: Option<$ty>,
            min_samples: Option<u64>,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    arms: Option::None,
                    gamma: Option::None,
                    eta: Option::None,
                    min_samples: Option::None,
                }
            }

            /// Samples an arm proportional to mixed weights.
            ///
            /// Returns `(arm_index, selection_probability)`. The caller
            /// must pass the probability back to `update()`.
            #[must_use]
            #[allow(clippy::suboptimal_flops)]
            pub fn select(&self, rng: &mut impl FnMut() -> $ty) -> (usize, $ty) {
                let sum_w: $ty = self.weights.iter().copied().sum();
                let k = self.num_arms as $ty;
                let u = rng();
                let mut cumulative = 0.0 as $ty;

                for i in 0..self.num_arms {
                    let p_i = (1.0 as $ty - self.gamma) * self.weights[i] / sum_w + self.gamma / k;
                    cumulative += p_i;
                    if u < cumulative {
                        return (i, p_i);
                    }
                }

                // Numerical safety: return last arm
                let last = self.num_arms - 1;
                let p_last =
                    (1.0 as $ty - self.gamma) * self.weights[last] / sum_w + self.gamma / k;
                (last, p_last)
            }

            /// Records a reward for the selected arm.
            ///
            /// `prob` is the selection probability returned by `select()`.
            ///
            /// # Errors
            ///
            /// Returns `DataError` if reward is NaN, infinite, or outside [0, 1],
            /// or if prob is NaN, infinite, or <= 0.
            ///
            /// # Panics
            ///
            /// Panics if `arm >= num_arms`.
            #[inline]
            #[allow(clippy::cast_possible_truncation)]
            pub fn update(
                &mut self,
                arm: usize,
                reward: $ty,
                prob: $ty,
            ) -> Result<(), nexus_stats_core::DataError> {
                check_finite!(reward);
                if reward < 0.0 as $ty || reward > 1.0 as $ty {
                    return Err(nexus_stats_core::DataError::Negative);
                }
                check_finite!(prob);
                if prob <= 0.0 as $ty {
                    return Err(nexus_stats_core::DataError::Negative);
                }
                assert!(
                    arm < self.num_arms,
                    "arm {arm} >= num_arms {}",
                    self.num_arms
                );

                let k = self.num_arms as $ty;
                let estimated_reward = reward / prob;
                let exponent = self.eta * estimated_reward / k;
                self.weights[arm] *= nexus_stats_core::math::exp(exponent as f64) as $ty;

                // Prevent weight overflow: normalize if max weight exceeds threshold
                let threshold = if core::mem::size_of::<$ty>() >= 8 {
                    1e30 as $ty
                } else {
                    1e15 as $ty
                };
                let max_w = self
                    .weights
                    .iter()
                    .copied()
                    .fold(0.0 as $ty, |a, b| if a > b { a } else { b });
                if max_w > threshold {
                    for w in self.weights.iter_mut() {
                        *w /= max_w;
                    }
                }

                self.total_pulls += 1;
                Ok(())
            }

            /// Writes the current probability distribution into `out`.
            ///
            /// # Panics
            ///
            /// Panics if `out.len() != num_arms`.
            #[allow(clippy::suboptimal_flops)]
            pub fn probabilities(&self, out: &mut [$ty]) {
                assert_eq!(
                    out.len(),
                    self.num_arms,
                    "output length {} != num_arms {}",
                    out.len(),
                    self.num_arms,
                );
                let sum_w: $ty = self.weights.iter().copied().sum();
                let k = self.num_arms as $ty;
                for i in 0..self.num_arms {
                    out[i] = (1.0 as $ty - self.gamma) * self.weights[i] / sum_w + self.gamma / k;
                }
            }

            /// Total pulls across all arms.
            #[inline]
            #[must_use]
            pub fn total_pulls(&self) -> u64 {
                self.total_pulls
            }

            /// Number of arms.
            #[inline]
            #[must_use]
            pub fn num_arms(&self) -> usize {
                self.num_arms
            }

            /// Whether total pulls >= min_samples.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                self.total_pulls >= self.min_samples
            }

            /// Returns the number of updates performed.
            #[inline]
            #[must_use]
            pub fn count(&self) -> u64 {
                self.total_pulls
            }

            /// Resets weights to uniform.
            #[inline]
            pub fn reset(&mut self) {
                self.weights.fill(1.0 as $ty);
                self.total_pulls = 0;
            }
        }

        impl $builder {
            /// Sets the number of arms (required, >= 2).
            #[inline]
            #[must_use]
            pub fn arms(mut self, n: usize) -> Self {
                self.arms = Option::Some(n);
                self
            }

            /// Sets the exploration mixing rate (required, in (0, 1]).
            #[inline]
            #[must_use]
            pub fn gamma(mut self, g: $ty) -> Self {
                self.gamma = Option::Some(g);
                self
            }

            /// Sets the learning rate (default: gamma / K, must be > 0).
            #[inline]
            #[must_use]
            pub fn eta(mut self, e: $ty) -> Self {
                self.eta = Option::Some(e);
                self
            }

            /// Sets the minimum samples before `is_primed()` returns true.
            #[inline]
            #[must_use]
            pub fn min_samples(mut self, n: u64) -> Self {
                self.min_samples = Option::Some(n);
                self
            }

            /// Builds the bandit.
            #[inline]
            pub fn build(self) -> Result<$name, nexus_stats_core::ConfigError> {
                let arms = self
                    .arms
                    .ok_or(nexus_stats_core::ConfigError::Missing("arms"))?;
                let gamma = self
                    .gamma
                    .ok_or(nexus_stats_core::ConfigError::Missing("gamma"))?;
                if arms < 2 {
                    return Err(nexus_stats_core::ConfigError::Invalid("arms must be >= 2"));
                }
                if gamma <= 0.0 as $ty || gamma > 1.0 as $ty || !gamma.is_finite() {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "gamma must be in (0, 1]",
                    ));
                }
                let eta = self.eta.unwrap_or(gamma / arms as $ty);
                if eta <= 0.0 as $ty || !eta.is_finite() {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "eta must be positive and finite",
                    ));
                }
                let min_samples = self.min_samples.unwrap_or(arms as u64);
                Ok($name {
                    weights: vec![1.0 as $ty; arms].into_boxed_slice(),
                    gamma,
                    eta,
                    total_pulls: 0,
                    num_arms: arms,
                    min_samples,
                })
            }
        }
    };
}

impl_exp3!(Exp3F64, Exp3F64Builder, f64);
impl_exp3!(Exp3F32, Exp3F32Builder, f32);

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        }
    }

    #[test]
    fn uniform_start() {
        let bandit = Exp3F64::builder().arms(3).gamma(0.1).build().unwrap();

        let mut probs = vec![0.0; 3];
        bandit.probabilities(&mut probs);
        let expected = 1.0 / 3.0;
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                (p - expected).abs() < 1e-10,
                "prob[{i}]={p}, expected {expected}"
            );
        }
    }

    #[test]
    fn adapts_to_best() {
        let mut bandit = Exp3F64::builder().arms(3).gamma(0.1).build().unwrap();
        let mut rng = make_rng(42);

        for _ in 0..500 {
            let (arm, prob) = bandit.select(&mut rng);
            let reward = if arm == 1 { 0.9 } else { 0.1 };
            bandit.update(arm, reward, prob).unwrap();
        }

        let mut probs = vec![0.0; 3];
        bandit.probabilities(&mut probs);
        assert!(
            probs[1] > probs[0] && probs[1] > probs[2],
            "arm 1 should have highest prob: {probs:?}"
        );
    }

    #[test]
    fn adversarial_robustness() {
        let mut bandit = Exp3F64::builder().arms(2).gamma(0.2).build().unwrap();
        let mut rng = make_rng(42);

        // Alternating which arm pays
        for round in 0..200 {
            let (arm, prob) = bandit.select(&mut rng);
            let best_arm = round % 2;
            let reward = if arm == best_arm { 1.0 } else { 0.0 };
            bandit.update(arm, reward, prob).unwrap();
        }

        // Both arms should maintain reasonable probability
        let mut probs = vec![0.0; 2];
        bandit.probabilities(&mut probs);
        assert!(probs[0] > 0.1, "arm 0 prob={}", probs[0]);
        assert!(probs[1] > 0.1, "arm 1 prob={}", probs[1]);
    }

    #[test]
    fn prob_sums_to_one() {
        let mut bandit = Exp3F64::builder().arms(4).gamma(0.15).build().unwrap();
        let mut rng = make_rng(42);

        for _ in 0..50 {
            let (arm, prob) = bandit.select(&mut rng);
            bandit.update(arm, 0.5, prob).unwrap();
        }

        let mut probs = vec![0.0; 4];
        bandit.probabilities(&mut probs);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "prob sum={sum}, expected 1.0");
    }

    #[test]
    fn weight_normalization() {
        let mut bandit = Exp3F64::builder()
            .arms(2)
            .gamma(0.1)
            .eta(1.0) // aggressive learning rate
            .build()
            .unwrap();

        // Many updates with high reward to trigger normalization
        for _ in 0..500 {
            bandit.update(0, 1.0, 0.5).unwrap();
        }

        // Weights should be finite after normalization
        assert!(
            bandit.weights.iter().all(|w| w.is_finite()),
            "weights should be finite: {:?}",
            &*bandit.weights,
        );
    }

    #[test]
    fn reward_out_of_range() {
        let mut bandit = Exp3F64::builder().arms(2).gamma(0.1).build().unwrap();
        assert!(bandit.update(0, -0.1, 0.5).is_err());
        assert!(bandit.update(0, 1.1, 0.5).is_err());
        assert!(bandit.update(0, 0.5, 0.0).is_err());
        assert!(bandit.update(0, 0.5, -0.1).is_err());
        assert!(bandit.update(0, f64::NAN, 0.5).is_err());
        assert_eq!(bandit.total_pulls(), 0);
    }

    #[test]
    fn reset_clears() {
        let mut bandit = Exp3F64::builder().arms(3).gamma(0.1).build().unwrap();

        bandit.update(0, 0.9, 0.5).unwrap();
        bandit.update(1, 0.1, 0.4).unwrap();
        bandit.reset();

        assert_eq!(bandit.total_pulls(), 0);
        let mut probs = vec![0.0; 3];
        bandit.probabilities(&mut probs);
        let expected = 1.0 / 3.0;
        for &p in &probs {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn builder_validation() {
        assert!(Exp3F64::builder().arms(1).gamma(0.1).build().is_err());
        assert!(Exp3F64::builder().arms(2).gamma(0.0).build().is_err());
        assert!(Exp3F64::builder().arms(2).gamma(1.5).build().is_err());
        assert!(
            Exp3F64::builder()
                .arms(2)
                .gamma(0.1)
                .eta(0.0)
                .build()
                .is_err()
        );
        assert!(Exp3F64::builder().arms(2).build().is_err()); // missing gamma
    }
}

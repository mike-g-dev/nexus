# nexus-stats-regression

Online regression, learning, and estimation for [nexus-stats](https://crates.io/crates/nexus-stats).

## Regression Types

- **LinearRegressionF64** — Online linear regression
- **PolynomialRegressionF64** — Online polynomial regression
- **EwPolynomialRegressionF64** — Exponentially weighted polynomial regression
- **TransformedRegressionF64** — Log/exp/power/reciprocal transforms (requires `std` or `libm`)
- **LogisticRegressionF64** — Online logistic regression (requires `alloc` + (`std` or `libm`))

## Learning Types

- **LmsFilterF64** — Least Mean Squares adaptive filter (requires `alloc`)
- **RlsFilterF64** — Recursive Least Squares adaptive filter (requires `alloc`)
- **OnlineKMeansF64** — Online K-Means clustering (requires `alloc`)
- **OnlineGdF64** — Online gradient descent (requires `alloc`)
- **AdaGradF64** — AdaGrad optimizer (requires `alloc` + (`std` or `libm`))
- **AdamF64** — Adam optimizer (requires `alloc` + (`std` or `libm`))
- **Ucb1F64** — UCB1 multi-armed bandit (requires `alloc` + (`std` or `libm`))
- **ThompsonBetaF64** — Thompson Sampling with Beta prior (requires `alloc` + (`std` or `libm`))
- **ThompsonGammaF64** — Thompson Sampling with Gamma prior (requires `alloc` + (`std` or `libm`))
- **EpsilonGreedyF64** — Epsilon-greedy bandit (requires `alloc` + (`std` or `libm`))
- **Exp3F64** — EXP3 adversarial bandit (requires `alloc` + (`std` or `libm`))

## Estimation Types

- **Kalman2dF64 / Kalman3dF64** — 2D/3D Kalman filters
- **BetaBinomialF64** — Beta-Binomial conjugate estimator
- **GammaPoissonF64** — Gamma-Poisson conjugate estimator

## Usage

Enable the `regression` feature on `nexus-stats` for unified import paths:

```rust
use nexus_stats::regression::LinearRegressionF64;
use nexus_stats::learning::AdamF64;
use nexus_stats::estimation::Kalman2dF64;
```

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT license](../LICENSE-MIT) at your option.

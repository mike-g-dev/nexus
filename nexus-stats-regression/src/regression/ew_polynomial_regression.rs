// Exponentially-Weighted Polynomial Regression
//
// Same sufficient-statistics approach as OLS polynomial regression,
// but with exponential decay on accumulators. Recent data dominates.
// Degree and intercept are runtime-configured via builder.

use super::polynomial_regression::{CoefficientsF32, CoefficientsF64};
use nexus_stats_core::math::MulAdd;

macro_rules! impl_ew_polynomial_regression {
    ($name:ident, $builder:ident, $coeff:ident, $solve_fn:path, $ty:ty) => {
        /// Exponentially-weighted online polynomial regression.
        ///
        /// Same accumulator structure as OLS polynomial regression but with
        /// exponential decay. Recent observations are weighted more heavily,
        /// making the fit adaptive to trend changes.
        ///
        /// `alpha` is the weight on the new observation (same convention as EMA).
        ///
        /// # Complexity
        /// - O(degree) per update, O(degree³) per coefficient query.
        /// - ~240 bytes state (f64), zero allocation.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("use nexus_stats_regression::regression::", stringify!($name), ";")]
        ///
        #[doc = concat!("let mut r = ", stringify!($name), "::builder().degree(2).alpha(0.05 as ", stringify!($ty), ").build().unwrap();")]
        #[doc = concat!("for x in 0..200u64 { r.update(x as ", stringify!($ty), ", 2.0 as ", stringify!($ty), " * x as ", stringify!($ty), "); }")]
        /// assert!(r.is_primed());
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            sum_x: [$ty; 17],
            sum_xy: [$ty; 9],
            sum_y2: $ty,
            alpha: $ty,
            one_minus_alpha: $ty,
            effective_n: $ty,
            count: u64,
            degree: usize,
            intercept: bool,
        }

        /// Builder for [`
        #[doc = stringify!($name)]
        /// `].
        #[derive(Debug, Clone)]
        pub struct $builder {
            degree: Option<usize>,
            intercept: bool,
            alpha: Option<$ty>,
        }

        impl $name {
            /// Creates a builder.
            #[inline]
            #[must_use]
            pub fn builder() -> $builder {
                $builder {
                    degree: Option::None,
                    intercept: true,
                    alpha: Option::None,
                }
            }

            fn dim(&self) -> usize {
                self.degree + self.intercept as usize
            }

            /// Feeds an (x, y) observation.
            ///
            /// # Errors
            ///
            /// Returns `DataError::NotANumber` if either value is NaN, or
            /// `DataError::Infinite` if either value is infinite.
            #[inline]
            pub fn update(&mut self, x: $ty, y: $ty) -> Result<(), nexus_stats_core::DataError> {
                check_finite!(x);
                check_finite!(y);
                self.count += 1;
                self.effective_n = self.one_minus_alpha.fma(self.effective_n, 1.0 as $ty);
                self.sum_y2 = self.one_minus_alpha.fma(self.sum_y2, y * y);

                let mut x_pow = 1.0 as $ty;
                let max_pow = 2 * self.degree;
                for j in 0..=max_pow {
                    self.sum_x[j] = self.one_minus_alpha.fma(self.sum_x[j], x_pow);
                    if j <= self.degree {
                        self.sum_xy[j] = self.one_minus_alpha.fma(self.sum_xy[j], x_pow * y);
                    }
                    x_pow *= x;
                }
                Ok(())
            }

            /// Solve for polynomial coefficients.
            #[must_use]
            pub fn coefficients(&self) -> Option<$coeff> {
                let dim = self.dim();
                if (self.effective_n as usize) < dim {
                    return Option::None;
                }

                let mut a = [[0.0 as $ty; 9]; 9];
                let mut b = [0.0 as $ty; 9];
                let offset: usize = if self.intercept { 0 } else { 1 };

                for i in 0..dim {
                    for j in 0..dim {
                        a[i][j] = self.sum_x[i + j + 2 * offset];
                    }
                    b[i] = self.sum_xy[i + offset];
                }

                if !$solve_fn(dim, &mut a, &mut b) {
                    return Option::None;
                }

                let mut coeffs = $coeff {
                    values: [0.0 as $ty; 9],
                    len: dim,
                };
                for i in 0..dim {
                    coeffs.values[i] = b[i];
                }
                Option::Some(coeffs)
            }

            /// R² goodness of fit.
            #[must_use]
            pub fn r_squared(&self) -> Option<$ty> {
                let coeffs = self.coefficients()?;
                let dim = self.dim();
                let offset: usize = if self.intercept { 0 } else { 1 };

                let mut beta_dot_rhs = 0.0 as $ty;
                for i in 0..dim {
                    beta_dot_rhs += coeffs.values[i] * self.sum_xy[i + offset];
                }
                let ss_res = self.sum_y2 - beta_dot_rhs;

                let ss_tot = if self.intercept {
                    let sum_y = self.sum_xy[0];
                    self.sum_y2 - sum_y * sum_y / self.effective_n
                } else {
                    self.sum_y2
                };

                if ss_tot <= 0.0 as $ty {
                    return Option::None;
                }

                Option::Some(1.0 as $ty - ss_res / ss_tot)
            }

            /// Predict y for a given x.
            #[must_use]
            pub fn predict(&self, x: $ty) -> Option<$ty> {
                let coeffs = self.coefficients()?;
                let dim = self.dim();

                let mut y = 0.0 as $ty;
                let mut x_pow = if self.intercept { 1.0 as $ty } else { x };
                for i in 0..dim {
                    y += coeffs.values[i] * x_pow;
                    x_pow *= x;
                }
                Option::Some(y)
            }

            /// Number of observations processed.
            #[inline]
            #[must_use]
            pub fn count(&self) -> u64 {
                self.count
            }

            /// Effective sample count (converges to 1/alpha as n → ∞).
            #[inline]
            #[must_use]
            pub fn effective_count(&self) -> $ty {
                self.effective_n
            }

            /// Alpha (weight on new observation).
            #[inline]
            #[must_use]
            pub fn alpha(&self) -> $ty {
                self.alpha
            }

            /// Configured polynomial degree.
            #[inline]
            #[must_use]
            pub fn degree(&self) -> usize {
                self.degree
            }

            /// Whether primed.
            #[inline]
            #[must_use]
            pub fn is_primed(&self) -> bool {
                (self.effective_n as usize) >= self.dim()
            }

            /// Resets to empty state. Config unchanged.
            #[inline]
            pub fn reset(&mut self) {
                self.sum_x = [0.0 as $ty; 17];
                self.sum_xy = [0.0 as $ty; 9];
                self.sum_y2 = 0.0 as $ty;
                self.effective_n = 0.0 as $ty;
                self.count = 0;
            }
        }

        impl $builder {
            /// Polynomial degree (1..=8). Required.
            #[inline]
            #[must_use]
            pub fn degree(mut self, degree: usize) -> Self {
                self.degree = Option::Some(degree);
                self
            }

            /// Whether to include the constant term. Default: `true`.
            #[inline]
            #[must_use]
            pub fn intercept(mut self, intercept: bool) -> Self {
                self.intercept = intercept;
                self
            }

            /// Weight on new observation, in (0, 1). Required.
            #[inline]
            #[must_use]
            pub fn alpha(mut self, alpha: $ty) -> Self {
                self.alpha = Option::Some(alpha);
                self
            }

            /// Builds the regression estimator.
            ///
            /// # Errors
            ///
            /// Returns errors if degree or alpha not set, or values out of range.
            pub fn build(self) -> Result<$name, nexus_stats_core::ConfigError> {
                let degree = self.degree
                    .ok_or(nexus_stats_core::ConfigError::Missing("degree"))?;
                let alpha = self.alpha
                    .ok_or(nexus_stats_core::ConfigError::Missing("alpha"))?;

                if degree < 1 || degree > 8 {
                    return Err(nexus_stats_core::ConfigError::Invalid("degree must be in 1..=8"));
                }
                if !(alpha > 0.0 as $ty && alpha < 1.0 as $ty) {
                    return Err(nexus_stats_core::ConfigError::Invalid(
                        "alpha must be in (0, 1) exclusive",
                    ));
                }

                Ok($name {
                    sum_x: [0.0 as $ty; 17],
                    sum_xy: [0.0 as $ty; 9],
                    sum_y2: 0.0 as $ty,
                    alpha,
                    one_minus_alpha: 1.0 as $ty - alpha,
                    effective_n: 0.0 as $ty,
                    count: 0,
                    degree,
                    intercept: self.intercept,
                })
            }
        }
    };
}

impl_ew_polynomial_regression!(
    EwPolynomialRegressionF64,
    EwPolynomialRegressionF64Builder,
    CoefficientsF64,
    super::polynomial_regression::gauss_solve_f64,
    f64
);
impl_ew_polynomial_regression!(
    EwPolynomialRegressionF32,
    EwPolynomialRegressionF32Builder,
    CoefficientsF32,
    super::polynomial_regression::gauss_solve_f32,
    f32
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ew_linear_basic() {
        let mut r = EwPolynomialRegressionF64::builder()
            .degree(1)
            .alpha(0.05)
            .build()
            .unwrap();
        for x in 0..500 {
            r.update(x as f64, 2.0 * x as f64 + 3.0).unwrap();
        }
        let c = r.coefficients().unwrap();
        assert!(
            (c.as_slice()[1] - 2.0).abs() < 0.5,
            "ew slope = {}",
            c.as_slice()[1]
        );
    }

    #[test]
    fn ew_adapts_to_trend_change() {
        let mut r = EwPolynomialRegressionF64::builder()
            .degree(1)
            .alpha(0.05)
            .build()
            .unwrap();
        for x in 0..200 {
            r.update(x as f64, x as f64).unwrap();
        }
        for x in 200..500 {
            r.update(x as f64, -(x as f64) + 400.0).unwrap();
        }
        let slope = r.coefficients().unwrap().values[1];
        assert!(
            slope < 0.0,
            "slope should be negative after trend change, got {slope}"
        );
    }

    #[test]
    fn ew_rejects_invalid_alpha() {
        assert!(
            EwPolynomialRegressionF64::builder()
                .degree(1)
                .alpha(0.0)
                .build()
                .is_err()
        );
        assert!(
            EwPolynomialRegressionF64::builder()
                .degree(1)
                .alpha(1.0)
                .build()
                .is_err()
        );
    }

    #[test]
    fn ew_rejects_missing() {
        assert!(
            EwPolynomialRegressionF64::builder()
                .alpha(0.05)
                .build()
                .is_err()
        ); // missing degree
        assert!(
            EwPolynomialRegressionF64::builder()
                .degree(1)
                .build()
                .is_err()
        ); // missing alpha
    }

    #[test]
    fn ew_predict() {
        let mut r = EwPolynomialRegressionF64::builder()
            .degree(1)
            .alpha(0.05)
            .build()
            .unwrap();
        for x in 0..300 {
            r.update(x as f64, 3.0 * x as f64).unwrap();
        }
        let y = r.predict(100.0).unwrap();
        assert!((y - 300.0).abs() < 50.0, "predict(100) = {y}");
    }

    #[test]
    fn ew_reset() {
        let mut r = EwPolynomialRegressionF64::builder()
            .degree(1)
            .alpha(0.05)
            .build()
            .unwrap();
        for x in 0..100 {
            r.update(x as f64, x as f64).unwrap();
        }
        r.reset();
        assert_eq!(r.count(), 0);
        assert!(r.coefficients().is_none());
        assert_eq!(r.degree(), 1);
    }

    #[test]
    fn ew_f32_basic() {
        let mut r = EwPolynomialRegressionF32::builder()
            .degree(1)
            .alpha(0.05)
            .build()
            .unwrap();
        for x in 0..200u32 {
            r.update(x as f32, 2.0 * x as f32 + 1.0).unwrap();
        }
        assert!(r.is_primed());
    }

    #[test]
    fn ew_effective_count() {
        let mut r = EwPolynomialRegressionF64::builder()
            .degree(1)
            .alpha(0.05)
            .build()
            .unwrap();
        for x in 0..1000 {
            r.update(x as f64, x as f64).unwrap();
        }
        assert!(
            (r.effective_count() - 20.0).abs() < 1.0,
            "effective_n = {}",
            r.effective_count()
        );
    }

    #[test]
    fn ew_quadratic() {
        let r = EwPolynomialRegressionF64::builder()
            .degree(2)
            .alpha(0.05)
            .build()
            .unwrap();
        assert_eq!(r.degree(), 2);
    }

    #[test]
    fn rejects_nan_and_inf() {
        let mut r = EwPolynomialRegressionF64::builder()
            .degree(1)
            .alpha(0.05)
            .build()
            .unwrap();
        assert_eq!(
            r.update(f64::NAN, 1.0),
            Err(nexus_stats_core::DataError::NotANumber)
        );
        assert_eq!(
            r.update(1.0, f64::INFINITY),
            Err(nexus_stats_core::DataError::Infinite)
        );
        assert_eq!(r.count(), 0);
    }
}

// Transformed Regression — Linearized Fits via ln
//
// Thin wrappers around linear regression that apply ln at the API boundary:
// - Exponential: y = a * e^(bx)  ->  ln(y) = ln(a) + bx
// - Logarithmic: y = a * ln(x) + b  ->  y = a * ln(x) + b  (already linear in ln(x))
// - Power law: y = a * x^b  ->  ln(y) = ln(a) + b * ln(x)

use super::linear_regression::{EwLinearRegressionF64, LinearRegressionF64};

// ============================================================================
// Exponential: y = a * e^(bx)
// ============================================================================

/// Online exponential regression: `y = a * e^(bx)`.
///
/// Linearized as `ln(y) = ln(a) + bx` and solved via linear regression.
/// Observations with `y <= 0` are silently skipped (ln undefined).
///
/// R² is measured in log-space (goodness of fit of `ln(y)` vs `x`).
///
/// # Examples
///
/// ```
/// use nexus_stats_regression::regression::ExponentialRegressionF64;
///
/// let mut r = ExponentialRegressionF64::new();
/// for x in 0..100 {
///     let y = 2.0_f64 * (0.05_f64 * x as f64).exp();
///     r.update(x as _, y);
/// }
/// let rate = r.growth_rate().unwrap();
/// assert!((rate - 0.05).abs() < 0.01);
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialRegressionF64 {
    inner: LinearRegressionF64,
}

impl ExponentialRegressionF64 {
    /// Creates a new empty exponential regression.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: LinearRegressionF64::new(),
        }
    }

    /// Feeds (x, y). Silently skips if `y <= 0`.
    ///
    /// # Errors
    ///
    /// Returns `DataError::NotANumber` if x or y is NaN, or
    /// `DataError::Infinite` if x or y is infinite.
    #[inline]
    pub fn update(&mut self, x: f64, y: f64) -> Result<(), nexus_stats_core::DataError> {
        check_finite!(x);
        check_finite!(y);
        if y > 0.0 {
            let ln_y = nexus_stats_core::math::ln(y);
            self.inner.update(x, ln_y)?;
        }
        Ok(())
    }

    /// Growth/decay rate (the exponent b), or `None` if not primed.
    #[must_use]
    pub fn growth_rate(&self) -> Option<f64> {
        self.inner.slope()
    }

    /// Scale factor `a = e^(intercept)`, or `None` if not primed.
    #[must_use]
    pub fn scale(&self) -> Option<f64> {
        self.inner
            .intercept_value()
            .map(nexus_stats_core::math::exp)
    }

    /// R² in log-space.
    #[must_use]
    pub fn r_squared(&self) -> Option<f64> {
        self.inner.r_squared()
    }

    /// Predict `y = a * e^(bx)`.
    #[must_use]
    pub fn predict(&self, x: f64) -> Option<f64> {
        self.inner.predict(x).map(nexus_stats_core::math::exp)
    }

    /// Number of accepted observations (y > 0).
    #[inline]
    #[must_use]
    pub fn count(&self) -> u64 {
        self.inner.count()
    }

    /// Whether enough data for a fit (>= 2 observations with y > 0).
    #[inline]
    #[must_use]
    pub fn is_primed(&self) -> bool {
        self.inner.is_primed()
    }

    /// Resets to empty state.
    #[inline]
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

impl Default for ExponentialRegressionF64 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Logarithmic: y = a * ln(x) + b
// ============================================================================

/// Online logarithmic regression: `y = a * ln(x) + b`.
///
/// Linearized by substituting `u = ln(x)`, solving `y = a*u + b`.
/// Observations with `x <= 0` are silently skipped (ln undefined).
///
/// # Examples
///
/// ```
/// use nexus_stats_regression::regression::LogarithmicRegressionF64;
///
/// let mut r = LogarithmicRegressionF64::new();
/// for x in 1..200 {
///     let y = 3.0_f64 * (x as f64).ln() + 1.0_f64;
///     r.update(x as _, y);
/// }
/// let slope = r.slope().unwrap();
/// assert!((slope - 3.0).abs() < 0.01);
/// ```
#[derive(Debug, Clone)]
pub struct LogarithmicRegressionF64 {
    inner: LinearRegressionF64,
}

impl LogarithmicRegressionF64 {
    /// Creates a new empty logarithmic regression.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: LinearRegressionF64::new(),
        }
    }

    /// Feeds (x, y). Silently skips if `x <= 0`.
    ///
    /// # Errors
    ///
    /// Returns `DataError::NotANumber` if x or y is NaN, or
    /// `DataError::Infinite` if x or y is infinite.
    #[inline]
    pub fn update(&mut self, x: f64, y: f64) -> Result<(), nexus_stats_core::DataError> {
        check_finite!(x);
        check_finite!(y);
        if x > 0.0 {
            let ln_x = nexus_stats_core::math::ln(x);
            self.inner.update(ln_x, y)?;
        }
        Ok(())
    }

    /// Slope (coefficient of ln(x)), or `None` if not primed.
    #[must_use]
    pub fn slope(&self) -> Option<f64> {
        self.inner.slope()
    }

    /// Intercept (constant term b), or `None` if not primed.
    #[must_use]
    pub fn intercept_value(&self) -> Option<f64> {
        self.inner.intercept_value()
    }

    /// R² goodness of fit.
    #[must_use]
    pub fn r_squared(&self) -> Option<f64> {
        self.inner.r_squared()
    }

    /// Predict `y = a * ln(x) + b`. Returns `None` if not primed or `x <= 0`.
    #[must_use]
    pub fn predict(&self, x: f64) -> Option<f64> {
        if x <= 0.0 {
            return None;
        }
        let ln_x = nexus_stats_core::math::ln(x);
        self.inner.predict(ln_x)
    }

    /// Number of accepted observations (x > 0).
    #[inline]
    #[must_use]
    pub fn count(&self) -> u64 {
        self.inner.count()
    }

    #[inline]
    #[must_use]
    /// Whether enough data for a fit.
    pub fn is_primed(&self) -> bool {
        self.inner.is_primed()
    }

    /// Resets to empty state.
    #[inline]
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

impl Default for LogarithmicRegressionF64 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Power Law: y = a * x^b
// ============================================================================

/// Online power law regression: `y = a * x^b`.
///
/// Linearized as `ln(y) = ln(a) + b * ln(x)`. Observations with
/// `x <= 0` or `y <= 0` are silently skipped.
///
/// R² is measured in log-log space.
///
/// # Examples
///
/// ```
/// use nexus_stats_regression::regression::PowerRegressionF64;
///
/// let mut r = PowerRegressionF64::new();
/// for x in 1..200 {
///     let y = 4.0_f64 * (x as f64).powf(2.5);
///     r.update(x as _, y);
/// }
/// let exp = r.exponent().unwrap();
/// assert!((exp - 2.5).abs() < 0.01);
/// ```
#[derive(Debug, Clone)]
pub struct PowerRegressionF64 {
    inner: LinearRegressionF64,
}

impl PowerRegressionF64 {
    /// Creates a new empty power law regression.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: LinearRegressionF64::new(),
        }
    }

    /// Feeds (x, y). Silently skips if `x <= 0` or `y <= 0`.
    ///
    /// # Errors
    ///
    /// Returns `DataError::NotANumber` if x or y is NaN, or
    /// `DataError::Infinite` if x or y is infinite.
    #[inline]
    pub fn update(&mut self, x: f64, y: f64) -> Result<(), nexus_stats_core::DataError> {
        check_finite!(x);
        check_finite!(y);
        if x > 0.0 && y > 0.0 {
            let ln_x = nexus_stats_core::math::ln(x);
            let ln_y = nexus_stats_core::math::ln(y);
            self.inner.update(ln_x, ln_y)?;
        }
        Ok(())
    }

    /// Exponent (the power b), or `None` if not primed.
    #[must_use]
    pub fn exponent(&self) -> Option<f64> {
        self.inner.slope()
    }

    /// Scale factor `a = e^(intercept)`, or `None` if not primed.
    #[must_use]
    pub fn scale(&self) -> Option<f64> {
        self.inner
            .intercept_value()
            .map(nexus_stats_core::math::exp)
    }

    /// R² in log-log space.
    #[must_use]
    pub fn r_squared(&self) -> Option<f64> {
        self.inner.r_squared()
    }

    /// Predict `y = a * x^b`. Returns `None` if not primed or `x <= 0`.
    #[must_use]
    pub fn predict(&self, x: f64) -> Option<f64> {
        if x <= 0.0 {
            return None;
        }
        let intercept = self.inner.intercept_value()?;
        let slope = self.inner.slope()?;
        let a = nexus_stats_core::math::exp(intercept);
        let result = a * x.powf(slope);
        Some(result)
    }

    /// Number of accepted observations.
    #[inline]
    #[must_use]
    pub fn count(&self) -> u64 {
        self.inner.count()
    }

    #[inline]
    #[must_use]
    /// Whether enough data for a fit.
    pub fn is_primed(&self) -> bool {
        self.inner.is_primed()
    }

    /// Resets to empty state.
    #[inline]
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

impl Default for PowerRegressionF64 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// EW Transformed Variants
// ============================================================================

/// Exponentially-weighted exponential regression: `y = a * e^(bx)`.
#[derive(Debug, Clone)]
pub struct EwExponentialRegressionF64 {
    inner: EwLinearRegressionF64,
}

/// Builder for [`EwExponentialRegressionF64`].
#[derive(Debug, Clone)]
pub struct EwExponentialRegressionF64Builder {
    alpha: Option<f64>,
}

impl EwExponentialRegressionF64 {
    /// Creates a builder.
    #[inline]
    #[must_use]
    pub fn builder() -> EwExponentialRegressionF64Builder {
        EwExponentialRegressionF64Builder { alpha: None }
    }

    /// Feeds (x, y). Silently skips if `y <= 0`.
    ///
    /// # Errors
    ///
    /// Returns `DataError::NotANumber` if x or y is NaN, or
    /// `DataError::Infinite` if x or y is infinite.
    #[inline]
    pub fn update(&mut self, x: f64, y: f64) -> Result<(), nexus_stats_core::DataError> {
        check_finite!(x);
        check_finite!(y);
        if y > 0.0 {
            let ln_y = nexus_stats_core::math::ln(y);
            self.inner.update(x, ln_y)?;
        }
        Ok(())
    }

    /// Growth/decay rate.
    #[must_use]
    pub fn growth_rate(&self) -> Option<f64> {
        self.inner.slope()
    }

    /// Scale factor `a = e^(intercept)`.
    #[must_use]
    pub fn scale(&self) -> Option<f64> {
        self.inner
            .intercept_value()
            .map(nexus_stats_core::math::exp)
    }

    /// R² in log-space.
    #[must_use]
    pub fn r_squared(&self) -> Option<f64> {
        self.inner.r_squared()
    }

    /// Predict `y = a * e^(bx)`.
    #[must_use]
    pub fn predict(&self, x: f64) -> Option<f64> {
        self.inner.predict(x).map(nexus_stats_core::math::exp)
    }

    /// Number of accepted observations.
    #[inline]
    #[must_use]
    pub fn count(&self) -> u64 {
        self.inner.count()
    }

    #[inline]
    #[must_use]
    /// Whether primed.
    pub fn is_primed(&self) -> bool {
        self.inner.is_primed()
    }

    /// Reset.
    #[inline]
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

impl EwExponentialRegressionF64Builder {
    /// Weight on new observation, in (0, 1).
    #[inline]
    #[must_use]
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Builds the estimator.
    pub fn build(self) -> Result<EwExponentialRegressionF64, nexus_stats_core::ConfigError> {
        let alpha = self
            .alpha
            .ok_or(nexus_stats_core::ConfigError::Missing("alpha"))?;
        let inner = EwLinearRegressionF64::builder().alpha(alpha).build()?;
        Ok(EwExponentialRegressionF64 { inner })
    }
}

/// Exponentially-weighted logarithmic regression: `y = a * ln(x) + b`.
#[derive(Debug, Clone)]
pub struct EwLogarithmicRegressionF64 {
    inner: EwLinearRegressionF64,
}

/// Builder for [`EwLogarithmicRegressionF64`].
#[derive(Debug, Clone)]
pub struct EwLogarithmicRegressionF64Builder {
    alpha: Option<f64>,
}

impl EwLogarithmicRegressionF64 {
    /// Creates a builder.
    #[inline]
    #[must_use]
    pub fn builder() -> EwLogarithmicRegressionF64Builder {
        EwLogarithmicRegressionF64Builder { alpha: None }
    }

    /// Feeds (x, y). Silently skips if `x <= 0`.
    ///
    /// # Errors
    ///
    /// Returns `DataError::NotANumber` if x or y is NaN, or
    /// `DataError::Infinite` if x or y is infinite.
    #[inline]
    pub fn update(&mut self, x: f64, y: f64) -> Result<(), nexus_stats_core::DataError> {
        check_finite!(x);
        check_finite!(y);
        if x > 0.0 {
            let ln_x = nexus_stats_core::math::ln(x);
            self.inner.update(ln_x, y)?;
        }
        Ok(())
    }

    /// Slope (coefficient of ln(x)).
    #[must_use]
    pub fn slope(&self) -> Option<f64> {
        self.inner.slope()
    }

    /// Intercept.
    #[must_use]
    pub fn intercept_value(&self) -> Option<f64> {
        self.inner.intercept_value()
    }

    /// R².
    #[must_use]
    pub fn r_squared(&self) -> Option<f64> {
        self.inner.r_squared()
    }

    /// Predict `y = a * ln(x) + b`.
    #[must_use]
    pub fn predict(&self, x: f64) -> Option<f64> {
        if x <= 0.0 {
            return None;
        }
        let ln_x = nexus_stats_core::math::ln(x);
        self.inner.predict(ln_x)
    }

    #[inline]
    #[must_use]
    /// Count.
    pub fn count(&self) -> u64 {
        self.inner.count()
    }
    #[inline]
    #[must_use]
    /// Primed.
    pub fn is_primed(&self) -> bool {
        self.inner.is_primed()
    }
    /// Reset.
    #[inline]
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

impl EwLogarithmicRegressionF64Builder {
    /// Alpha.
    #[inline]
    #[must_use]
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Build.
    pub fn build(self) -> Result<EwLogarithmicRegressionF64, nexus_stats_core::ConfigError> {
        let alpha = self
            .alpha
            .ok_or(nexus_stats_core::ConfigError::Missing("alpha"))?;
        let inner = EwLinearRegressionF64::builder().alpha(alpha).build()?;
        Ok(EwLogarithmicRegressionF64 { inner })
    }
}

/// Exponentially-weighted power law regression: `y = a * x^b`.
#[derive(Debug, Clone)]
pub struct EwPowerRegressionF64 {
    inner: EwLinearRegressionF64,
}

/// Builder for [`EwPowerRegressionF64`].
#[derive(Debug, Clone)]
pub struct EwPowerRegressionF64Builder {
    alpha: Option<f64>,
}

impl EwPowerRegressionF64 {
    /// Creates a builder.
    #[inline]
    #[must_use]
    pub fn builder() -> EwPowerRegressionF64Builder {
        EwPowerRegressionF64Builder { alpha: None }
    }

    /// Feeds (x, y). Silently skips if `x <= 0` or `y <= 0`.
    ///
    /// # Errors
    ///
    /// Returns `DataError::NotANumber` if x or y is NaN, or
    /// `DataError::Infinite` if x or y is infinite.
    #[inline]
    pub fn update(&mut self, x: f64, y: f64) -> Result<(), nexus_stats_core::DataError> {
        check_finite!(x);
        check_finite!(y);
        if x > 0.0 && y > 0.0 {
            let ln_x = nexus_stats_core::math::ln(x);
            let ln_y = nexus_stats_core::math::ln(y);
            self.inner.update(ln_x, ln_y)?;
        }
        Ok(())
    }

    /// Exponent b.
    #[must_use]
    pub fn exponent(&self) -> Option<f64> {
        self.inner.slope()
    }

    /// Scale `a = e^(intercept)`.
    #[must_use]
    pub fn scale(&self) -> Option<f64> {
        self.inner
            .intercept_value()
            .map(nexus_stats_core::math::exp)
    }

    /// R² in log-log space.
    #[must_use]
    pub fn r_squared(&self) -> Option<f64> {
        self.inner.r_squared()
    }

    /// Predict `y = a * x^b`.
    #[must_use]
    pub fn predict(&self, x: f64) -> Option<f64> {
        if x <= 0.0 {
            return None;
        }
        let intercept = self.inner.intercept_value()?;
        let slope = self.inner.slope()?;
        let a = nexus_stats_core::math::exp(intercept);
        let result = a * x.powf(slope);
        Some(result)
    }

    #[inline]
    #[must_use]
    /// Count.
    pub fn count(&self) -> u64 {
        self.inner.count()
    }
    #[inline]
    #[must_use]
    /// Primed.
    pub fn is_primed(&self) -> bool {
        self.inner.is_primed()
    }
    /// Reset.
    #[inline]
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

impl EwPowerRegressionF64Builder {
    /// Alpha.
    #[inline]
    #[must_use]
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Build.
    pub fn build(self) -> Result<EwPowerRegressionF64, nexus_stats_core::ConfigError> {
        let alpha = self
            .alpha
            .ok_or(nexus_stats_core::ConfigError::Missing("alpha"))?;
        let inner = EwLinearRegressionF64::builder().alpha(alpha).build()?;
        Ok(EwPowerRegressionF64 { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Exponential: y = a * e^(bx)
    // =========================================================================

    #[test]
    fn exponential_exact_fit() {
        let mut r = ExponentialRegressionF64::new();
        for x in 0..100 {
            let xf = x as f64;
            let y = 2.0 * (0.05 * xf).exp();
            r.update(xf, y).unwrap();
        }
        let rate = r.growth_rate().unwrap();
        assert!((rate - 0.05).abs() < 1e-8, "growth rate = {rate}");
        let scale = r.scale().unwrap();
        assert!((scale - 2.0).abs() < 1e-6, "scale = {scale}");
        assert!((r.r_squared().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn exponential_predict() {
        let mut r = ExponentialRegressionF64::new();
        for x in 0..100 {
            let xf = x as f64;
            r.update(xf, 2.0 * (0.05 * xf).exp()).unwrap();
        }
        let y = r.predict(10.0).unwrap();
        let expected = 2.0 * (0.05 * 10.0_f64).exp();
        assert!((y - expected).abs() < 1e-4, "predict(10) = {y}");
    }

    #[test]
    fn exponential_skips_negative_y() {
        let mut r = ExponentialRegressionF64::new();
        r.update(1.0, -5.0).unwrap(); // skipped
        r.update(2.0, 0.0).unwrap(); // skipped
        assert_eq!(r.count(), 0);
    }

    // =========================================================================
    // Logarithmic: y = a * ln(x) + b
    // =========================================================================

    #[test]
    fn logarithmic_exact_fit() {
        let mut r = LogarithmicRegressionF64::new();
        for x in 1..200 {
            let xf = x as f64;
            r.update(xf, 3.0 * xf.ln() + 1.0).unwrap();
        }
        let slope = r.slope().unwrap();
        assert!((slope - 3.0).abs() < 1e-6, "slope = {slope}");
        let intercept = r.intercept_value().unwrap();
        assert!((intercept - 1.0).abs() < 1e-6, "intercept = {intercept}");
    }

    #[test]
    fn logarithmic_skips_negative_x() {
        let mut r = LogarithmicRegressionF64::new();
        r.update(-1.0, 5.0).unwrap();
        r.update(0.0, 5.0).unwrap();
        assert_eq!(r.count(), 0);
    }

    #[test]
    fn logarithmic_predict_negative_x_returns_none() {
        let mut r = LogarithmicRegressionF64::new();
        for x in 1..100 {
            r.update(x as f64, (x as f64).ln()).unwrap();
        }
        assert!(r.predict(-1.0).is_none());
    }

    // =========================================================================
    // Power law: y = a * x^b
    // =========================================================================

    #[test]
    fn power_exact_fit() {
        let mut r = PowerRegressionF64::new();
        for x in 1..200 {
            let xf = x as f64;
            r.update(xf, 4.0 * xf.powf(2.5)).unwrap();
        }
        let exp = r.exponent().unwrap();
        assert!((exp - 2.5).abs() < 1e-4, "exponent = {exp}");
        let scale = r.scale().unwrap();
        assert!((scale - 4.0).abs() < 0.1, "scale = {scale}");
    }

    #[test]
    fn power_skips_nonpositive() {
        let mut r = PowerRegressionF64::new();
        r.update(0.0, 5.0).unwrap();
        r.update(1.0, -5.0).unwrap();
        let _ = r.update(-1.0, 5.0);
        assert_eq!(r.count(), 0);
    }

    // =========================================================================
    // EW transformed variants
    // =========================================================================

    #[test]
    fn ew_exponential_basic() {
        let mut r = EwExponentialRegressionF64::builder()
            .alpha(0.05)
            .build()
            .unwrap();
        for x in 0..300 {
            let xf = x as f64;
            r.update(xf, 2.0 * (0.01 * xf).exp()).unwrap();
        }
        assert!(r.is_primed());
        assert!(r.growth_rate().is_some());
    }

    #[test]
    fn ew_logarithmic_basic() {
        let mut r = EwLogarithmicRegressionF64::builder()
            .alpha(0.05)
            .build()
            .unwrap();
        for x in 1..300 {
            r.update(x as f64, 2.0 * (x as f64).ln() + 5.0).unwrap();
        }
        assert!(r.is_primed());
    }

    #[test]
    fn ew_power_basic() {
        let mut r = EwPowerRegressionF64::builder().alpha(0.05).build().unwrap();
        for x in 1..300 {
            r.update(x as f64, 3.0 * (x as f64).powf(1.5)).unwrap();
        }
        assert!(r.is_primed());
    }

    // =========================================================================
    // Reset / Default
    // =========================================================================

    #[test]
    fn reset_all_transforms() {
        let mut exp = ExponentialRegressionF64::new();
        let mut log = LogarithmicRegressionF64::new();
        let mut pow = PowerRegressionF64::new();
        for x in 1..100 {
            exp.update(x as f64, (x as f64).exp()).unwrap();
            log.update(x as f64, (x as f64).ln()).unwrap();
            pow.update(x as f64, (x as f64).powi(2)).unwrap();
        }
        exp.reset();
        log.reset();
        pow.reset();
        assert_eq!(exp.count(), 0);
        assert_eq!(log.count(), 0);
        assert_eq!(pow.count(), 0);
    }

    #[test]
    fn defaults_are_empty() {
        assert_eq!(ExponentialRegressionF64::default().count(), 0);
        assert_eq!(LogarithmicRegressionF64::default().count(), 0);
        assert_eq!(PowerRegressionF64::default().count(), 0);
    }
}

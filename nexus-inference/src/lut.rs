#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, vec::Vec};

#[cfg(feature = "alloc")]
use crate::{LoadError, NanInput};

#[cfg(feature = "alloc")]
macro_rules! impl_lut {
    ($name:ident, $ty:ty) => {
        /// Lookup table predictor with uniform bin spacing.
        ///
        /// Discretizes continuous features into bins and indexes a flat
        /// pre-computed table. O(1) prediction — no arithmetic beyond
        /// a division and an array index per feature.
        ///
        /// Out-of-range features are clamped to the first/last bin.
        /// NaN features map to bin 0 (Rust's saturating float-to-int cast).
        ///
        /// # Examples
        ///
        /// ```
        /// use nexus_inference::LutF64;
        ///
        /// let model = LutF64::from_parts(
        ///     1, 4,
        ///     &[0.0], &[1.0],
        ///     &[10.0, 20.0, 30.0, 40.0],
        /// ).unwrap();
        /// assert_eq!(model.predict(&[0.1]).unwrap(), 10.0);
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            table: Box<[$ty]>,
            mins: Box<[$ty]>,
            steps: Box<[$ty]>,
            n_features: u8,
            n_bins: u16,
        }

        impl $name {
            /// Construct from pre-computed table.
            ///
            /// `mins` and `maxs` define the per-feature range. Features outside
            /// the range are clamped. `table` is a flat array of `n_bins^n_features`
            /// entries in row-major order (first feature varies slowest).
            pub fn from_parts(
                n_features: usize,
                n_bins: usize,
                mins: &[$ty],
                maxs: &[$ty],
                table: &[$ty],
            ) -> Result<Self, LoadError> {
                if n_features == 0 {
                    return Err(LoadError::Validation("n_features must be >= 1"));
                }
                if n_features > u8::MAX as usize {
                    return Err(LoadError::Validation("n_features exceeds u8::MAX"));
                }
                if n_bins < 2 {
                    return Err(LoadError::Validation("n_bins must be >= 2"));
                }
                if n_bins > u16::MAX as usize {
                    return Err(LoadError::Validation("n_bins exceeds u16::MAX"));
                }
                if mins.len() != n_features {
                    return Err(LoadError::Validation("mins length mismatch"));
                }
                if maxs.len() != n_features {
                    return Err(LoadError::Validation("maxs length mismatch"));
                }

                let expected_len = checked_pow(n_bins, n_features)
                    .ok_or(LoadError::Validation("table size overflow"))?;
                if table.len() != expected_len {
                    return Err(LoadError::Validation("table length mismatch"));
                }

                for i in 0..n_features {
                    if !mins[i].is_finite() || !maxs[i].is_finite() {
                        return Err(LoadError::Validation("non-finite min or max"));
                    }
                    if maxs[i] <= mins[i] {
                        return Err(LoadError::Validation("max must be > min"));
                    }
                }
                for &v in table {
                    if !v.is_finite() {
                        return Err(LoadError::Validation("non-finite table value"));
                    }
                }

                let steps: Vec<$ty> = (0..n_features)
                    .map(|i| (maxs[i] - mins[i]) / n_bins as $ty)
                    .collect();

                Ok(Self {
                    table: table.into(),
                    mins: mins.into(),
                    steps: steps.into_boxed_slice(),
                    n_features: n_features as u8,
                    n_bins: n_bins as u16,
                })
            }

            /// Predict with NaN input check.
            ///
            /// Returns `Err(NanInput)` if any feature is NaN.
            ///
            /// # Panics
            ///
            /// Panics if `features.len() != self.n_features()`.
            pub fn predict(&self, features: &[$ty]) -> Result<$ty, NanInput> {
                if features.iter().any(|x| x.is_nan()) {
                    return Err(NanInput);
                }
                Ok(self.predict_unchecked(features))
            }

            /// Predict without NaN check.
            ///
            /// NaN features map to bin 0 (Rust's saturating float-to-int cast).
            ///
            /// # Panics
            ///
            /// Panics if `features.len() != self.n_features()`.
            pub fn predict_unchecked(&self, features: &[$ty]) -> $ty {
                assert_eq!(features.len(), self.n_features as usize);
                let nb = self.n_bins as usize;
                let max_bin = nb - 1;
                let mut idx = 0usize;
                for i in 0..self.n_features as usize {
                    let bin = ((features[i] - self.mins[i]) / self.steps[i]) as usize;
                    let bin = if bin > max_bin { max_bin } else { bin };
                    idx = idx * nb + bin;
                }
                self.table[idx]
            }

            /// Write prediction to output buffer with NaN input check.
            ///
            /// Returns `Err(NanInput)` if any feature is NaN.
            ///
            /// # Panics
            ///
            /// Panics if `features.len() != self.n_features()` or
            /// `output.len() != 1`.
            pub fn predict_into(&self, features: &[$ty], output: &mut [$ty]) -> Result<(), NanInput> {
                if features.iter().any(|x| x.is_nan()) {
                    return Err(NanInput);
                }
                self.predict_into_unchecked(features, output);
                Ok(())
            }

            /// Write prediction to output buffer without NaN check.
            ///
            /// # Panics
            ///
            /// Panics if `features.len() != self.n_features()` or
            /// `output.len() != 1`.
            pub fn predict_into_unchecked(&self, features: &[$ty], output: &mut [$ty]) {
                assert_eq!(output.len(), 1);
                output[0] = self.predict_unchecked(features);
            }

            /// Number of input features.
            pub fn n_features(&self) -> usize {
                self.n_features as usize
            }

            /// Number of bins per feature.
            pub fn n_bins(&self) -> usize {
                self.n_bins as usize
            }

            /// Number of outputs. Always 1 for LUT.
            pub fn n_outputs(&self) -> usize {
                1
            }
        }
    };
}

#[cfg(feature = "alloc")]
fn checked_pow(base: usize, exp: usize) -> Option<usize> {
    let mut result = 1usize;
    for _ in 0..exp {
        result = result.checked_mul(base)?;
    }
    Some(result)
}

#[cfg(feature = "alloc")]
impl_lut!(LutF64, f64);
#[cfg(feature = "alloc")]
impl_lut!(LutF32, f32);

#[cfg(test)]
mod tests {
    #[cfg(feature = "alloc")]
    use super::*;

    #[test]
    #[cfg(feature = "alloc")]
    fn single_feature() {
        // 1 feature, 4 bins over [0, 1)
        // bins: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0)
        let model = LutF64::from_parts(1, 4, &[0.0], &[1.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(model.predict(&[0.1]).unwrap(), 10.0);
        assert_eq!(model.predict(&[0.3]).unwrap(), 20.0);
        assert_eq!(model.predict(&[0.6]).unwrap(), 30.0);
        assert_eq!(model.predict(&[0.8]).unwrap(), 40.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn two_features() {
        // 2 features, 3 bins each → 9 entries
        // Feature 0: [0, 3), bins at width 1.0
        // Feature 1: [0, 3), bins at width 1.0
        // table[f0*3 + f1]
        let table: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let model = LutF64::from_parts(2, 3, &[0.0, 0.0], &[3.0, 3.0], &table).unwrap();
        // f0=0.5 → bin 0, f1=1.5 → bin 1 → idx = 0*3 + 1 = 1
        assert_eq!(model.predict(&[0.5, 1.5]).unwrap(), 1.0);
        // f0=2.5 → bin 2, f1=0.5 → bin 0 → idx = 2*3 + 0 = 6
        assert_eq!(model.predict(&[2.5, 0.5]).unwrap(), 6.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn clamp_low() {
        let model = LutF64::from_parts(1, 4, &[0.0], &[1.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(model.predict(&[-5.0]).unwrap(), 10.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn clamp_high() {
        let model = LutF64::from_parts(1, 4, &[0.0], &[1.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(model.predict(&[99.0]).unwrap(), 40.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn boundary_values() {
        // 1 feature, 4 bins over [0, 4), step=1.0
        // bins: [0,1), [1,2), [2,3), [3,4)
        let model = LutF64::from_parts(1, 4, &[0.0], &[4.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        // Exactly at bin boundary → floor to that bin
        assert_eq!(model.predict(&[0.0]).unwrap(), 10.0);
        assert_eq!(model.predict(&[1.0]).unwrap(), 20.0);
        assert_eq!(model.predict(&[2.0]).unwrap(), 30.0);
        assert_eq!(model.predict(&[3.0]).unwrap(), 40.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[should_panic]
    fn wrong_feature_count_panics() {
        let model = LutF64::from_parts(1, 4, &[0.0], &[1.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        model.predict_unchecked(&[1.0, 2.0]); // expects 1 feature
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_parts_validates() {
        // Wrong table size
        assert!(LutF64::from_parts(1, 4, &[0.0], &[1.0], &[1.0; 3]).is_err());
        // Zero bins
        assert!(LutF64::from_parts(1, 0, &[0.0], &[1.0], &[]).is_err());
        // One bin (minimum is 2)
        assert!(LutF64::from_parts(1, 1, &[0.0], &[1.0], &[1.0]).is_err());
        // Zero features
        assert!(LutF64::from_parts(0, 4, &[], &[], &[]).is_err());
        // max <= min
        assert!(LutF64::from_parts(1, 4, &[5.0], &[5.0], &[1.0; 4]).is_err());
        assert!(LutF64::from_parts(1, 4, &[5.0], &[3.0], &[1.0; 4]).is_err());
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn f32_variant() {
        let model =
            LutF32::from_parts(1, 4, &[0.0_f32], &[1.0], &[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(model.predict(&[0.3_f32]).unwrap(), 20.0_f32);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn three_features() {
        // 3 features × 5 bins → 125 entries
        // table[f0*25 + f1*5 + f2]
        let table: Vec<f64> = (0..125).map(|i| i as f64).collect();
        let model = LutF64::from_parts(3, 5, &[0.0, 0.0, 0.0], &[5.0, 5.0, 5.0], &table).unwrap();
        // f0=0.5→bin0, f1=2.5→bin2, f2=4.5→bin4 → idx = 0*25 + 2*5 + 4 = 14
        assert_eq!(model.predict(&[0.5, 2.5, 4.5]).unwrap(), 14.0);
        // f0=3.5→bin3, f1=1.5→bin1, f2=0.5→bin0 → idx = 3*25 + 1*5 + 0 = 80
        assert_eq!(model.predict(&[3.5, 1.5, 0.5]).unwrap(), 80.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn nan_input_returns_error() {
        let model = LutF64::from_parts(1, 4, &[0.0], &[1.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert!(model.predict(&[f64::NAN]).is_err());
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn nan_unchecked_maps_to_bin_zero() {
        let model = LutF64::from_parts(1, 4, &[0.0], &[1.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(model.predict_unchecked(&[f64::NAN]), 10.0);
    }
}

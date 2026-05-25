use crate::LoadError;

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
/// use nexus_inference::Lut;
///
/// let model = Lut::from_parts(
///     1, 4,
///     &[0.0], &[1.0],
///     &[10.0, 20.0, 30.0, 40.0],
/// ).unwrap();
/// assert_eq!(model.predict(&[0.1]), 10.0);
/// ```
#[derive(Debug, Clone)]
pub struct Lut {
    table: Box<[f32]>,
    mins: Box<[f32]>,
    steps: Box<[f32]>,
    n_features: u8,
    n_bins: u16,
}

impl Lut {
    /// Construct from pre-computed table.
    ///
    /// `mins` and `maxs` define the per-feature range. Features outside
    /// the range are clamped. `table` is a flat array of `n_bins^n_features`
    /// entries in row-major order (first feature varies slowest).
    pub fn from_parts(
        n_features: usize,
        n_bins: usize,
        mins: &[f32],
        maxs: &[f32],
        table: &[f32],
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

        let expected_len =
            checked_pow(n_bins, n_features).ok_or(LoadError::Validation("table size overflow"))?;
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

        let steps: Vec<f32> = (0..n_features)
            .map(|i| (maxs[i] - mins[i]) / n_bins as f32)
            .collect();

        Ok(Self {
            table: table.into(),
            mins: mins.into(),
            steps: steps.into_boxed_slice(),
            n_features: n_features as u8,
            n_bins: n_bins as u16,
        })
    }

    /// Predict a single output value.
    ///
    /// NaN features map to bin 0 (Rust's saturating float-to-int cast).
    ///
    /// # Panics
    ///
    /// Panics if `features.len() != self.n_features()`.
    pub fn predict(&self, features: &[f32]) -> f32 {
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

    /// Write prediction to output buffer.
    ///
    /// # Panics
    ///
    /// Panics if `features.len() != self.n_features()` or
    /// `output.len() != 1`.
    pub fn predict_into(&self, features: &[f32], output: &mut [f32]) {
        assert_eq!(output.len(), 1);
        output[0] = self.predict(features);
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

impl crate::Model for Lut {
    fn predict(&mut self, input: &[f32]) -> f32 {
        Lut::predict(self, input)
    }
    fn predict_into(&mut self, input: &[f32], output: &mut [f32]) {
        Lut::predict_into(self, input, output);
    }
    fn n_outputs(&self) -> usize {
        Lut::n_outputs(self)
    }
}

impl crate::StatelessModel for Lut {}

fn checked_pow(base: usize, exp: usize) -> Option<usize> {
    let mut result = 1usize;
    for _ in 0..exp {
        result = result.checked_mul(base)?;
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_feature() {
        // 1 feature, 4 bins over [0, 1)
        // bins: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0)
        let model = Lut::from_parts(1, 4, &[0.0], &[1.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(model.predict(&[0.1]), 10.0);
        assert_eq!(model.predict(&[0.3]), 20.0);
        assert_eq!(model.predict(&[0.6]), 30.0);
        assert_eq!(model.predict(&[0.8]), 40.0);
    }

    #[test]
    fn two_features() {
        // 2 features, 3 bins each → 9 entries
        // Feature 0: [0, 3), bins at width 1.0
        // Feature 1: [0, 3), bins at width 1.0
        // table[f0*3 + f1]
        let table: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let model = Lut::from_parts(2, 3, &[0.0, 0.0], &[3.0, 3.0], &table).unwrap();
        // f0=0.5 → bin 0, f1=1.5 → bin 1 → idx = 0*3 + 1 = 1
        assert_eq!(model.predict(&[0.5, 1.5]), 1.0);
        // f0=2.5 → bin 2, f1=0.5 → bin 0 → idx = 2*3 + 0 = 6
        assert_eq!(model.predict(&[2.5, 0.5]), 6.0);
    }

    #[test]
    fn clamp_low() {
        let model = Lut::from_parts(1, 4, &[0.0], &[1.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(model.predict(&[-5.0]), 10.0);
    }

    #[test]
    fn clamp_high() {
        let model = Lut::from_parts(1, 4, &[0.0], &[1.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(model.predict(&[99.0]), 40.0);
    }

    #[test]
    fn boundary_values() {
        // 1 feature, 4 bins over [0, 4), step=1.0
        // bins: [0,1), [1,2), [2,3), [3,4)
        let model = Lut::from_parts(1, 4, &[0.0], &[4.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        // Exactly at bin boundary → floor to that bin
        assert_eq!(model.predict(&[0.0]), 10.0);
        assert_eq!(model.predict(&[1.0]), 20.0);
        assert_eq!(model.predict(&[2.0]), 30.0);
        assert_eq!(model.predict(&[3.0]), 40.0);
    }

    #[test]
    #[should_panic]
    fn wrong_feature_count_panics() {
        let model = Lut::from_parts(1, 4, &[0.0], &[1.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        model.predict(&[1.0, 2.0]); // expects 1 feature
    }

    #[test]
    fn from_parts_validates() {
        // Wrong table size
        assert!(Lut::from_parts(1, 4, &[0.0], &[1.0], &[1.0; 3]).is_err());
        // Zero bins
        assert!(Lut::from_parts(1, 0, &[0.0], &[1.0], &[]).is_err());
        // One bin (minimum is 2)
        assert!(Lut::from_parts(1, 1, &[0.0], &[1.0], &[1.0]).is_err());
        // Zero features
        assert!(Lut::from_parts(0, 4, &[], &[], &[]).is_err());
        // max <= min
        assert!(Lut::from_parts(1, 4, &[5.0], &[5.0], &[1.0; 4]).is_err());
        assert!(Lut::from_parts(1, 4, &[5.0], &[3.0], &[1.0; 4]).is_err());
    }

    #[test]
    fn three_features() {
        // 3 features × 5 bins → 125 entries
        // table[f0*25 + f1*5 + f2]
        let table: Vec<f32> = (0..125).map(|i| i as f32).collect();
        let model = Lut::from_parts(3, 5, &[0.0, 0.0, 0.0], &[5.0, 5.0, 5.0], &table).unwrap();
        // f0=0.5→bin0, f1=2.5→bin2, f2=4.5→bin4 → idx = 0*25 + 2*5 + 4 = 14
        assert_eq!(model.predict(&[0.5, 2.5, 4.5]), 14.0);
        // f0=3.5→bin3, f1=1.5→bin1, f2=0.5→bin0 → idx = 3*25 + 1*5 + 0 = 80
        assert_eq!(model.predict(&[3.5, 1.5, 0.5]), 80.0);
    }

    #[test]
    fn nan_maps_to_bin_zero() {
        let model = Lut::from_parts(1, 4, &[0.0], &[1.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(model.predict(&[f32::NAN]), 10.0);
    }
}

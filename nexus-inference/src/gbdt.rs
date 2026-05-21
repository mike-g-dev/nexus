#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, vec::Vec};

#[cfg(feature = "alloc")]
pub(crate) const LEAF_SENTINEL: u16 = u16::MAX;

/// Decision tree node. 16 bytes, cache-line friendly.
///
/// Internal nodes: `feature_idx < LEAF_SENTINEL`, `value` = threshold.
/// Leaf nodes: `feature_idx == LEAF_SENTINEL`, `value` = leaf prediction.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct Node {
    pub(crate) feature_idx: u16,
    pub(crate) left: u16,
    pub(crate) right: u16,
    // bit 0: default_left (NaN routing)
    pub(crate) flags: u16,
    pub(crate) value: f64,
}

#[cfg(feature = "alloc")]
macro_rules! impl_gbdt {
    ($name:ident, $ty:ty) => {
        /// Gradient-boosted decision tree ensemble.
        ///
        /// Immutable after construction. All prediction methods take `&self`.
        /// Thread-safe: `Send + Sync` by construction (no interior mutability).
        ///
        /// # Examples
        ///
        /// ```
        /// # #[cfg(feature = "loader-lightgbm")] {
        /// use nexus_inference::GbdtF64;
        ///
        /// // Load a LightGBM model from text format bytes
        /// // let model = GbdtF64::from_lightgbm(model_bytes).unwrap();
        /// // let prediction = model.predict(&features);
        /// # }
        /// ```
        #[derive(Debug, Clone)]
        pub struct $name {
            nodes: Box<[Node]>,
            tree_offsets: Box<[u32]>,
            n_features: usize,
            base_score: $ty,
        }

        impl $name {
            /// Predict with NaN routing (LightGBM-compatible).
            ///
            /// Returns the raw ensemble score (base score + sum of leaf values).
            /// For classification objectives, apply the appropriate link function
            /// (e.g., sigmoid for binary classification).
            ///
            /// NaN features are routed via the learned default direction at each
            /// split node. Matches LightGBM's inference behavior.
            ///
            /// # Panics
            ///
            /// Panics if `features.len() != self.n_features()`.
            pub fn predict(&self, features: &[$ty]) -> $ty {
                assert_eq!(features.len(), self.n_features);
                let base = self.nodes.as_ptr();
                let mut score = self.base_score;
                for &offset in &*self.tree_offsets {
                    // SAFETY: offset is the validated start of a tree within self.nodes.
                    score += Self::walk_tree(unsafe { base.add(offset as usize) }, features, true);
                }
                score
            }

            /// Predict without NaN checks. Caller guarantees all features are finite.
            ///
            /// Returns the raw ensemble score (base score + sum of leaf values).
            /// For classification objectives, apply the appropriate link function
            /// (e.g., sigmoid for binary classification).
            ///
            /// NaN features produce undefined output (IEEE 754: `NaN <= threshold`
            /// is always false, so NaN always routes right).
            ///
            /// # Panics
            ///
            /// Panics if `features.len() != self.n_features()`.
            pub fn predict_unchecked(&self, features: &[$ty]) -> $ty {
                assert_eq!(features.len(), self.n_features);
                let base = self.nodes.as_ptr();
                let mut score = self.base_score;
                for &offset in &*self.tree_offsets {
                    // SAFETY: offset is the validated start of a tree within self.nodes.
                    score += Self::walk_tree(unsafe { base.add(offset as usize) }, features, false);
                }
                score
            }

            /// Evaluate only the first `n_trees` trees with NaN routing.
            ///
            /// Clamped to `self.n_trees()` if `n_trees` exceeds the ensemble size.
            pub fn predict_n(&self, features: &[$ty], n_trees: usize) -> $ty {
                assert_eq!(features.len(), self.n_features);
                let n = n_trees.min(self.tree_offsets.len());
                let base = self.nodes.as_ptr();
                let mut score = self.base_score;
                for &offset in &self.tree_offsets[..n] {
                    // SAFETY: offset is the validated start of a tree within self.nodes.
                    score += Self::walk_tree(unsafe { base.add(offset as usize) }, features, true);
                }
                score
            }

            /// Evaluate only the first `n_trees` trees without NaN checks.
            ///
            /// Clamped to `self.n_trees()` if `n_trees` exceeds the ensemble size.
            /// Caller guarantees all features are finite.
            pub fn predict_n_unchecked(&self, features: &[$ty], n_trees: usize) -> $ty {
                assert_eq!(features.len(), self.n_features);
                let n = n_trees.min(self.tree_offsets.len());
                let base = self.nodes.as_ptr();
                let mut score = self.base_score;
                for &offset in &self.tree_offsets[..n] {
                    // SAFETY: offset is the validated start of a tree within self.nodes.
                    score += Self::walk_tree(unsafe { base.add(offset as usize) }, features, false);
                }
                score
            }

            /// Number of trees in the ensemble.
            pub fn n_trees(&self) -> usize {
                self.tree_offsets.len()
            }

            /// Number of features expected by the model.
            pub fn n_features(&self) -> usize {
                self.n_features
            }

            /// Base score (initial prediction before tree contributions).
            pub fn base_score(&self) -> $ty {
                self.base_score
            }

            /// # Safety
            ///
            /// `tree` must point to the root of a valid tree within `self.nodes`.
            /// Node indices (left/right) are tree-relative, validated by
            /// `remap_child` during loading.
            fn walk_tree(tree: *const Node, features: &[$ty], nan_aware: bool) -> $ty {
                let mut idx = 0usize;
                loop {
                    // SAFETY: idx=0 at entry. Subsequent values from node.left/right,
                    // validated by remap_child during loading.
                    let node = unsafe { *tree.add(idx) };
                    if node.feature_idx == LEAF_SENTINEL {
                        return node.value as $ty;
                    }
                    // SAFETY: feature_idx < n_features validated in convert_tree.
                    // Caller asserts features.len() == n_features.
                    let feat = unsafe { *features.get_unchecked(node.feature_idx as usize) };
                    let go_left = if nan_aware {
                        match feat.partial_cmp(&(node.value as $ty)) {
                            Some(core::cmp::Ordering::Greater) => false,
                            None => node.flags & 1 != 0,
                            _ => true,
                        }
                    } else {
                        feat <= node.value as $ty
                    };
                    idx = if go_left {
                        node.left as usize
                    } else {
                        node.right as usize
                    };
                }
            }

            #[allow(dead_code)]
            pub(crate) fn from_parts(
                trees: Vec<Vec<Node>>,
                n_features: usize,
                base_score: $ty,
            ) -> Self {
                let total: usize = trees.iter().map(|t| t.len()).sum();
                let mut nodes = Vec::with_capacity(total);
                let mut tree_offsets = Vec::with_capacity(trees.len());
                for tree in trees {
                    tree_offsets.push(nodes.len() as u32);
                    nodes.extend_from_slice(&tree);
                }
                Self {
                    nodes: nodes.into_boxed_slice(),
                    tree_offsets: tree_offsets.into_boxed_slice(),
                    n_features,
                    base_score,
                }
            }
        }
    };
}

#[cfg(feature = "alloc")]
impl_gbdt!(GbdtF64, f64);
#[cfg(feature = "alloc")]
impl_gbdt!(GbdtF32, f32);

#[cfg(test)]
mod tests {
    #[cfg(feature = "alloc")]
    use super::*;
    #[cfg(feature = "alloc")]
    use alloc::vec;

    #[cfg(feature = "alloc")]
    fn single_stump(base_score: f64) -> GbdtF64 {
        // feature[0] <= 0.5 → leaf -1.0, else → leaf 1.0
        let nodes = vec![
            Node {
                feature_idx: 0,
                left: 1,
                right: 2,
                flags: 0,
                value: 0.5,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: -1.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: 1.0,
            },
        ];
        GbdtF64::from_parts(vec![nodes], 1, base_score)
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn single_stump_left() {
        let model = single_stump(0.0);
        assert_eq!(model.predict(&[0.3]), -1.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn single_stump_right() {
        let model = single_stump(0.0);
        assert_eq!(model.predict(&[0.8]), 1.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn single_stump_boundary() {
        let model = single_stump(0.0);
        // 0.5 <= 0.5 is true → goes left
        assert_eq!(model.predict(&[0.5]), -1.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn base_score_added() {
        let model = single_stump(10.0);
        assert_eq!(model.predict(&[0.3]), 10.0 + -1.0);
        assert_eq!(model.predict(&[0.8]), 10.0 + 1.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn multi_tree_sums() {
        // 3 identical stumps: each contributes ±1.0
        let stump = vec![
            Node {
                feature_idx: 0,
                left: 1,
                right: 2,
                flags: 0,
                value: 0.5,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: -1.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: 1.0,
            },
        ];
        let model = GbdtF64::from_parts(vec![stump.clone(), stump.clone(), stump], 1, 0.0);
        assert_eq!(model.predict(&[0.3]), -3.0);
        assert_eq!(model.predict(&[0.8]), 3.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn predict_n_partial() {
        let stump = vec![
            Node {
                feature_idx: 0,
                left: 1,
                right: 2,
                flags: 0,
                value: 0.5,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: -1.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: 1.0,
            },
        ];
        let model = GbdtF64::from_parts(vec![stump.clone(), stump.clone(), stump], 1, 5.0);
        // 2 of 3 trees, feature goes left
        assert_eq!(model.predict_n(&[0.3], 2), 5.0 + -2.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn predict_n_exceeds_count() {
        let stump = vec![
            Node {
                feature_idx: 0,
                left: 1,
                right: 2,
                flags: 0,
                value: 0.5,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: -1.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: 1.0,
            },
        ];
        let model = GbdtF64::from_parts(vec![stump.clone(), stump.clone(), stump], 1, 0.0);
        assert_eq!(model.predict_n(&[0.3], 100), model.predict(&[0.3]));
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn predict_n_unchecked_partial() {
        let stump = vec![
            Node {
                feature_idx: 0,
                left: 1,
                right: 2,
                flags: 0,
                value: 0.5,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: -1.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: 1.0,
            },
        ];
        let model = GbdtF64::from_parts(vec![stump.clone(), stump.clone(), stump], 1, 5.0);
        assert_eq!(model.predict_n_unchecked(&[0.3], 2), 5.0 + -2.0);
        assert_eq!(
            model.predict_n_unchecked(&[0.3], 100),
            model.predict_unchecked(&[0.3])
        );
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn deeper_tree() {
        // depth-3 tree on 2 features:
        //        node0: f[0] <= 5.0
        //       /              \
        //   node1: f[1] <= 2.0   node2: f[1] <= 8.0
        //   /     \              /      \
        // leaf0   leaf1       leaf2    leaf3
        // -4.0    -2.0         2.0      4.0
        let nodes = vec![
            Node {
                feature_idx: 0,
                left: 1,
                right: 2,
                flags: 0,
                value: 5.0,
            },
            Node {
                feature_idx: 1,
                left: 3,
                right: 4,
                flags: 0,
                value: 2.0,
            },
            Node {
                feature_idx: 1,
                left: 5,
                right: 6,
                flags: 0,
                value: 8.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: -4.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: -2.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: 2.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: 4.0,
            },
        ];
        let model = GbdtF64::from_parts(vec![nodes], 2, 0.0);
        // f[0]=3 <= 5 → left, f[1]=1 <= 2 → left → leaf0 = -4.0
        assert_eq!(model.predict(&[3.0, 1.0]), -4.0);
        // f[0]=3 <= 5 → left, f[1]=3 > 2 → right → leaf1 = -2.0
        assert_eq!(model.predict(&[3.0, 3.0]), -2.0);
        // f[0]=7 > 5 → right, f[1]=5 <= 8 → left → leaf2 = 2.0
        assert_eq!(model.predict(&[7.0, 5.0]), 2.0);
        // f[0]=7 > 5 → right, f[1]=9 > 8 → right → leaf3 = 4.0
        assert_eq!(model.predict(&[7.0, 9.0]), 4.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn nan_routing_default_left() {
        // flags bit 0 = 1 → NaN goes left
        let nodes = vec![
            Node {
                feature_idx: 0,
                left: 1,
                right: 2,
                flags: 1,
                value: 0.5,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: -1.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: 1.0,
            },
        ];
        let model = GbdtF64::from_parts(vec![nodes], 1, 0.0);
        assert_eq!(model.predict(&[f64::NAN]), -1.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn nan_routing_default_right() {
        // flags bit 0 = 0 → NaN goes right
        let nodes = vec![
            Node {
                feature_idx: 0,
                left: 1,
                right: 2,
                flags: 0,
                value: 0.5,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: -1.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: 1.0,
            },
        ];
        let model = GbdtF64::from_parts(vec![nodes], 1, 0.0);
        assert_eq!(model.predict(&[f64::NAN]), 1.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn nan_unchecked_goes_right() {
        // predict_unchecked: NaN <= threshold is false → always right
        let nodes = vec![
            Node {
                feature_idx: 0,
                left: 1,
                right: 2,
                flags: 1, // default_left set, but ignored by unchecked
                value: 0.5,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: -1.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: 1.0,
            },
        ];
        let model = GbdtF64::from_parts(vec![nodes], 1, 0.0);
        assert_eq!(model.predict_unchecked(&[f64::NAN]), 1.0);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[should_panic]
    fn wrong_feature_count_panics() {
        let model = single_stump(0.0);
        model.predict(&[1.0, 2.0]); // expects 1 feature, got 2
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn f32_variant() {
        let nodes = vec![
            Node {
                feature_idx: 0,
                left: 1,
                right: 2,
                flags: 0,
                value: 0.5,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: -1.0,
            },
            Node {
                feature_idx: LEAF_SENTINEL,
                left: 0,
                right: 0,
                flags: 0,
                value: 1.0,
            },
        ];
        let model = GbdtF32::from_parts(vec![nodes], 1, 0.0_f32);
        assert_eq!(model.predict(&[0.3_f32]), -1.0_f32);
        assert_eq!(model.predict(&[0.8_f32]), 1.0_f32);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn accessors() {
        let model = single_stump(2.5);
        assert_eq!(model.n_trees(), 1);
        assert_eq!(model.n_features(), 1);
        assert_eq!(model.base_score(), 2.5);
    }
}

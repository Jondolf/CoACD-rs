//! Parameters for the CoACD algorithm.

/// Parameters for tuning the [CoACD](crate::Coacd) convex decomposition algorithm.
///
/// In most cases, one of the predefined presets such as [`FAST`](Self::FAST) or
/// [`MEDIUM`](Self::MEDIUM) should be sufficient.
///
/// For more control, use the [`new`](Self::new) constructor with a given
/// [`concavity_threshold`], or modify individual fields directly.
///
/// [`concavity_threshold`]: Self::concavity_threshold
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoacdParaneters {
    /// The concavity threshold for terminating the decomposition.
    ///
    /// This is the main parameter for balancing the level of detail and the number of decomposed
    /// components. Lower values yield more detailed decompositions with more components,
    /// while higher values yield coarser decompositions with fewer components.
    ///
    /// **Range**: 0.01 ~ 1.0
    pub concavity_threshold: f32,
    /// The sampling resolution used for Hausdorff distance calculations.
    ///
    /// Higher values yield more accurate concavity measurements but increase computation time.
    ///
    /// **Range**: 500 ~ 10000
    pub resolution: u32,
    /// Whether to merge adjacent convex hulls after decomposition, potentially reducing
    /// the number of final components.
    pub merge_convex_hulls: bool,
    /// The maximum number of convex hulls in the final decomposition.
    ///
    /// If `None`, there is no limit on the number of convex hulls.
    /// Otherwise, if `merge_convex_hulls` is `true`, the algorithm will
    /// attempt to merge convex hulls until the specified limit is reached.
    ///
    /// This may introduce convex hulls with a concavity greater than the specified
    /// `concavity_threshold`.
    pub max_convex_hulls: Option<u32>,
    /// Whether to use [Principal Component Analysis (PCA)](crate::point_cloud::Pca)
    /// to orient the input mesh with its principal axes before decomposition.
    ///
    /// This can help with choosing better cutting planes, leading to improved decompositions
    /// with fewer components.
    pub pca: bool,
    /// A coefficient `k` used for scaling the estimated interior Hausdorff distance `r_v`
    /// when computing the [concavity metric](crate::cost).
    ///
    /// **Range**: 0.0 ~ 1.0
    pub rv_k: f32,
    /// The number of search iterations for the [`MonteCarloTree`](crate::mcts::MonteCarloTree).
    ///
    /// **Range**: 60 ~ 2000
    pub mcts_iterations: u32,
    /// The maximum search depth for the [`MonteCarloTree`](crate::mcts::MonteCarloTree).
    ///
    /// If set too low, the search may not explore enough cutting planes to find
    /// good decompositions. If set too high, the search may take longer without
    /// significant improvements.
    ///
    /// **Range**: 1 ~ 7
    pub mcts_max_depth: u32,
    /// The maximum number of child nodes in the [`MonteCarloTree`](crate::mcts::MonteCarloTree).
    ///
    /// **Range**: 5 ~ 40
    pub mcts_max_nodes: u32,
    /// The seed used for random number generation during sampling.
    pub seed: u64,
}

impl CoacdParaneters {
    /// Very fast parameters for quick decomposition.
    ///
    /// This is suitable for very low-detail convex decompositions where speed is prioritized
    /// over accuracy. This can be a good choice for simple dynamic objects in physics simulations.
    pub const VERY_FAST: Self = Self {
        concavity_threshold: 0.1,
        resolution: 500,
        merge_convex_hulls: false,
        max_convex_hulls: None,
        pca: false,
        rv_k: 0.3,
        mcts_iterations: 100,
        mcts_max_depth: 1,
        mcts_max_nodes: 5,
        seed: 0,
    };

    /// Fast parameters for decomposition.
    ///
    /// This is suitable for low-detail convex decompositions where efficiency is prioritized
    /// over accuracy. This can be a good choice for dynamic objects in physics simulations.
    pub const FAST: Self = Self {
        concavity_threshold: 0.075,
        resolution: 1000,
        merge_convex_hulls: false,
        max_convex_hulls: None,
        pca: false,
        rv_k: 0.3,
        mcts_iterations: 150,
        mcts_max_depth: 2,
        mcts_max_nodes: 10,
        seed: 0,
    };

    /// Medium quality parameters for decomposition.
    ///
    /// This provides a balance between speed and detail for many meshes.
    /// It can produce decently accurate decompositions in a reasonable time.
    pub const MEDIUM: Self = Self {
        concavity_threshold: 0.05,
        resolution: 2000,
        merge_convex_hulls: false,
        max_convex_hulls: None,
        pca: false,
        rv_k: 0.3,
        mcts_iterations: 200,
        mcts_max_depth: 3,
        mcts_max_nodes: 20,
        seed: 0,
    };

    /// Slow but high quality parameters for decomposition.
    ///
    /// This is suitable for high-detail convex decompositions where accuracy is prioritized
    /// over speed. This can be a good choice for static objects or when precomputing collision meshes
    /// with high fidelity.
    pub const SLOW: Self = Self {
        concavity_threshold: 0.025,
        resolution: 5000,
        merge_convex_hulls: false,
        max_convex_hulls: None,
        pca: false,
        rv_k: 0.3,
        mcts_iterations: 500,
        mcts_max_depth: 5,
        mcts_max_nodes: 30,
        seed: 0,
    };

    /// Very slow but very high quality parameters for decomposition.
    ///
    /// This is suitable for very high-detail convex decompositions where maximum accuracy is prioritized
    /// over speed. This can be a good choice for static objects or when precomputing collision meshes
    /// with the highest fidelity.
    pub const VERY_SLOW: Self = Self {
        concavity_threshold: 0.01,
        resolution: 10000,
        merge_convex_hulls: false,
        max_convex_hulls: None,
        pca: false,
        rv_k: 0.3,
        mcts_iterations: 2000,
        mcts_max_depth: 7,
        mcts_max_nodes: 40,
        seed: 0,
    };

    /// Creates new [`CoacdParaneters`] with the specified concavity threshold,
    /// using the rest of the values from the [`FAST`](Self::FAST) preset.
    ///
    /// The `concavity_threshold` controls the level of detail in the decomposition.
    /// Lower values yield more detailed decompositions with more components,
    /// while higher values yield coarser decompositions with fewer components.
    ///
    /// The expected range for `concavity_threshold` is between 0.01 and 1.0.
    #[inline]
    pub const fn new(concavity_threshold: f32) -> Self {
        Self {
            concavity_threshold,
            // This shouldn't be too low for low concavity thresholds to work well.
            mcts_max_nodes: 20,
            ..Self::FAST
        }
    }
}

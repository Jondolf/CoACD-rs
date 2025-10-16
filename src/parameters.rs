//! Parameters for the CoACD algorithm.

/// Parameters for the CoACD algorithm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoacdParaneters {
    pub mcts_nodes: u32,
    pub threshold: f32,
    pub resolution: u32,
    pub seed: u64,
    pub rv_k: f32,
    pub pca: bool,
    pub mcts_iterations: u32,
    pub mcts_max_depth: u32,
}

impl Default for CoacdParaneters {
    fn default() -> Self {
        Self::new()
    }
}

impl CoacdParaneters {
    /// Creates a new set of parameters with default values.
    #[inline]
    pub const fn new() -> Self {
        Self {
            mcts_nodes: 20,
            threshold: 0.05,
            resolution: 2000,
            seed: 42,
            rv_k: 0.3,
            pca: true,
            mcts_iterations: 150,
            mcts_max_depth: 3,
        }
    }
}

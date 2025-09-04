//! Approximate Convex Decomposition for 3D Meshes
//! with Collision-Aware Concavity and Tree Search.
//!
//! # References
//!
//! - [Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search](https://colin97.github.io/CoACD/) by Xinyue Wei, Minghua Liu, Zhan Ling, Hao Su

#![warn(missing_docs)]

pub mod clip;

pub(crate) mod collections;
pub(crate) mod math;

pub use math::{Plane, PlaneSide};
pub use obvhs::aabb::Aabb;

use glam::Vec3A;

/// A 3D triangle mesh represented by vertices and indices.
#[derive(Clone, Debug, Default)]
pub struct IndexedMesh {
    /// The vertices of the mesh.
    pub vertices: Vec<Vec3A>,
    /// The indices of the mesh.
    pub indices: Vec<[usize; 3]>,
}

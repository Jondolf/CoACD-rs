//! Approximate Convex Decomposition for 3D Meshes
//! with Collision-Aware Concavity and Tree Search.
//!
//! # References
//!
//! - [Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search](https://colin97.github.io/CoACD/) by Xinyue Wei, Minghua Liu, Zhan Ling, Hao Su

#![warn(missing_docs)]

pub mod clip;
pub mod cost;
pub mod hausdorff;
pub mod mesh;

pub(crate) mod collections;
pub(crate) mod math;

pub use math::{Plane, PlaneSide};
pub use obvhs::aabb::Aabb;

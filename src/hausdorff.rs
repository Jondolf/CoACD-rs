//! [Hausdorff distances] between two meshes.
//!
//! The Hausdorff distance is a measure of how far two subsets of a metric space are from each other.
//! In this context, the two subsets are the surfaces of two 3D meshes.
//!
//! [Hausdorff distances]: https://en.wikipedia.org/wiki/Hausdorff_distance

use std::num::NonZero;

use glam::Vec3A;
use kiddo::{ImmutableKdTree, SquaredEuclidean};

use crate::mesh::IndexedMesh;

/// Computes the [Hausdorff distance] between two meshes using sampled points and nearest neighbor search.
///
/// See the [module-level documentation](crate::hausdorff) for more details.
///
/// [Hausdorff distance]: https://en.wikipedia.org/wiki/Hausdorff_distance
pub fn face_hausdorff_distance(
    mesh1: &IndexedMesh,
    mesh2: &IndexedMesh,
    samples1: &[[f32; 3]],
    samples2: &[[f32; 3]],
    indices1: &[usize],
    indices2: &[usize],
) -> f32 {
    let tree1: ImmutableKdTree<f32, 3> = ImmutableKdTree::new_from_slice(samples1);
    let tree2: ImmutableKdTree<f32, 3> = ImmutableKdTree::new_from_slice(samples2);

    const NUM_NEIGHBORS: NonZero<usize> = NonZero::new(10).unwrap();
    const FAR_AWAY_NEAREST_NEIGHBOR: f32 = 1e2;
    const DISTANCE_TOLERANCE: f32 = 1e-14;

    // The squared maximum of the minimum distances.
    let mut max_dist_sq = 0.0;

    // For each sample point in `mesh2`, compute the minimum distance to `mesh1`.
    // The maximum of these distances is the one-sided Hausdorff distance from `mesh2` to `mesh1`.
    for point in samples2 {
        // Find the nearest neighbors in `mesh1`.
        let results = tree1.nearest_n::<SquaredEuclidean>(point, NUM_NEIGHBORS);
        let first_distance_squared = results[0].distance;

        let p = Vec3A::from_array(*point);

        // The squared distance to the closest point on the triangles.
        let mut min_dist_sq = f32::INFINITY;

        for nearest_neighbor in results {
            // Get the triangle corresponding to the nearest neighbor.
            let idx = nearest_neighbor.item as usize;
            let tri = &mesh1.indices[indices1[idx]];
            let v0 = mesh1.vertices[tri[0]];
            let v1 = mesh1.vertices[tri[1]];
            let v2 = mesh1.vertices[tri[2]];

            // Project the point onto the triangle and compute the squared distance.
            let projection = crate::math::project_on_triangle_3d(v0, v1, v2, p);
            let dist_sq = p.distance_squared(projection);

            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;

                // TODO: Since the distance is squared, should this threshold be squared too?
                if min_dist_sq < DISTANCE_TOLERANCE {
                    // Early out if we're very close.
                    break;
                }
            }
        }

        // TODO: Do we need this?
        if min_dist_sq > FAR_AWAY_NEAREST_NEIGHBOR {
            // If the minimum distance is much larger than the distance to the nearest neighbor,
            // we can use the distance to the nearest neighbor as a lower bound.
            min_dist_sq = first_distance_squared;
        }

        if min_dist_sq >= max_dist_sq && min_dist_sq != f32::INFINITY {
            max_dist_sq = min_dist_sq;
        }
    }

    // Repeat the process for sample points in `mesh1` to `mesh2`.
    for point in samples1 {
        // Find the nearest neighbors in `mesh2`.
        let results = tree2.nearest_n::<SquaredEuclidean>(point, NUM_NEIGHBORS);
        let first_distance_squared = results[0].distance;

        let p = Vec3A::from_array(*point);

        // The squared distance to the closest point on the triangles.
        let mut min_dist_sq = f32::INFINITY;

        for nearest_neighbor in results {
            // Get the triangle corresponding to the nearest neighbor.
            let idx = nearest_neighbor.item as usize;
            let tri = &mesh2.indices[indices2[idx]];
            let v0 = mesh2.vertices[tri[0]];
            let v1 = mesh2.vertices[tri[1]];
            let v2 = mesh2.vertices[tri[2]];

            // Project the point onto the triangle and compute the squared distance.
            let projection = crate::math::project_on_triangle_3d(v0, v1, v2, p);
            let dist_sq = p.distance_squared(projection);

            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;

                if min_dist_sq < DISTANCE_TOLERANCE {
                    // Early out if we're very close.
                    break;
                }
            }
        }

        if min_dist_sq > FAR_AWAY_NEAREST_NEIGHBOR {
            // If the minimum distance is much larger than the distance to the nearest neighbor,
            // we can use the distance to the nearest neighbor as a lower bound.
            min_dist_sq = first_distance_squared;
        }

        if min_dist_sq >= max_dist_sq && min_dist_sq != f32::INFINITY {
            max_dist_sq = min_dist_sq;
        }
    }

    // TODO: Can we return the squared distance instead?
    max_dist_sq.sqrt()
}

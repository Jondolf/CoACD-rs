//! Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search.
//!
//! # References
//!
//! - [Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search](https://colin97.github.io/CoACD/) by Xinyue Wei, Minghua Liu, Zhan Ling, Hao Su

#![warn(missing_docs)]

pub mod clip;
pub mod cost;
pub mod hausdorff;
pub mod mcts;
pub mod mesh;
pub mod parameters;
pub mod point_cloud;

pub(crate) mod collections;
pub(crate) mod hashable_partial_eq;
pub(crate) mod math;

pub use math::{Plane, PlaneSide};
pub use obvhs::aabb::Aabb;
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    mesh::{ConvexHullExt, IndexedMesh},
    parameters::CoacdParaneters,
};

pub struct Coacd {
    pub parameters: CoacdParaneters,
}

impl Coacd {
    /// Merges convex hulls based on the concavity metric until the concavity threshold or the maximum number of convex hulls is reached.
    ///
    /// Returns the maximum concavity of the merged hulls.
    #[inline]
    pub fn merge_convex_hulls(
        &self,
        meshes: &[IndexedMesh],
        hulls: &mut Vec<IndexedMesh>,
        threshold: f32,
    ) -> f32 {
        let num_hulls = hulls.len();
        let mut h = 0.0f32;

        if num_hulls > 1 {
            let bound = ((num_hulls - 1) * num_hulls) >> 1;

            // Only keep the top half of the matrices.
            let mut cost_matrix = vec![f32::INFINITY; bound];
            let mut precost_matrix = vec![f32::INFINITY; bound];

            cost_matrix
                .par_iter_mut()
                .enumerate()
                .zip(precost_matrix.par_iter_mut())
                .for_each(|((i, cost), precost)| {
                    // Nearest triangle number index
                    let mut p1 = ((8.0 * i as f32 + 1.0).sqrt() as usize - 1) >> 1;
                    // Nearest triangle number from index
                    let sum = (p1 * (p1 + 1)) >> 1;
                    // Modular arithmetic to find the other triangle index
                    let p2 = i - sum;
                    p1 += 1;

                    let hull1 = &hulls[p1];
                    let hull2 = &hulls[p2];

                    let distance = hull1.distance_to_mesh(hull2);
                    if distance < threshold {
                        let merged_mesh = hull1.merged_with(hull2);
                        let merged_hull = merged_mesh.compute_convex_hull().unwrap().to_mesh();

                        *cost = cost::compute_concavity_hulls(
                            hull1,
                            hull2,
                            &merged_hull,
                            self.parameters.seed,
                            self.parameters.resolution,
                            self.parameters.rv_k,
                        );
                        *precost = f32::max(
                            cost::compute_concavity(
                                &meshes[p1],
                                hull1,
                                self.parameters.seed,
                                // TODDO: This should probably not be hardcoded.
                                3000,
                                self.parameters.rv_k,
                            ),
                            cost::compute_concavity(
                                &meshes[p2],
                                hull2,
                                self.parameters.seed,
                                // TODDO: This should probably not be hardcoded.
                                3000,
                                self.parameters.rv_k,
                            ),
                        );
                    }
                });

            let mut cost_size = hulls.len();

            loop {
                // Find the minimum cost in the cost matrix.
                let Some((min_cost_i, &min_cost)) = cost_matrix
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                else {
                    // No more hulls to merge.
                    break;
                };

                if let Some(max_hulls) = self.parameters.max_convex_hulls {
                    // Stop merging if the maximum number of hulls is reached and the minimum cost is above the threshold.
                    if hulls.len() as u32 <= max_hulls && min_cost > self.parameters.threshold {
                        break;
                    }
                    // Avoid merging parts that have already used up the threshold.
                    if hulls.len() as u32 <= max_hulls
                        && min_cost
                            > (self.parameters.threshold - precost_matrix[min_cost_i]).max(0.01)
                    {
                        cost_matrix[min_cost_i] = f32::INFINITY;
                        continue;
                    }
                } else {
                    // Stop merging if the minimum cost is above the threshold.
                    if min_cost > self.parameters.threshold {
                        break;
                    }
                    // Avoid merging parts that have already used up the threshold.
                    if min_cost > (self.parameters.threshold - precost_matrix[min_cost_i]).max(0.01)
                    {
                        cost_matrix[min_cost_i] = f32::INFINITY;
                        continue;
                    }
                }

                h = h.max(min_cost);

                // Find the indices of the two hulls to merge.
                let index = (((1 + 8 * min_cost_i) as f32).sqrt() as usize - 1) >> 1;
                let p1 = index + 1;
                let p2 = min_cost_i - ((index * (index + 1)) >> 1);

                debug_assert!(p1 < cost_size);
                debug_assert!(p2 < cost_size);

                // Construct a new hull from the lowest cost row and column.
                let merged_mesh = hulls[p1].merged_with(&hulls[p2]);
                let merged_hull = merged_mesh.compute_convex_hull().unwrap().to_mesh();
                hulls[p2] = merged_hull;
                hulls.swap_remove(p1);

                cost_size -= 1;

                // Update the cost matrix.
                let hull2 = &hulls[p2];
                let mut row_index = ((p2 - 1) * p2) >> 1;

                // Update the costs for the affected rows.
                for i in 0..p2 {
                    let hull_i = &hulls[i];
                    let distance = hull2.distance_to_mesh(hull_i);
                    if distance < threshold {
                        let merged_mesh = hull2.merged_with(hull_i);
                        let merged_hull = merged_mesh.compute_convex_hull().unwrap().to_mesh();
                        cost_matrix[row_index] = cost::compute_concavity_hulls(
                            hull2,
                            hull_i,
                            &merged_hull,
                            self.parameters.seed,
                            self.parameters.resolution,
                            self.parameters.rv_k,
                        );
                        precost_matrix[row_index] =
                            f32::max(precost_matrix[p2] + min_cost, precost_matrix[i]);
                    } else {
                        cost_matrix[row_index] = f32::INFINITY;
                    }
                    row_index += 1;
                }

                row_index += p2;

                // Update the costs for the affected columns.
                for i in (p2 + 1)..cost_size {
                    let hull_i = &hulls[i];
                    let distance = hull2.distance_to_mesh(hull_i);
                    if distance < threshold {
                        let merged_mesh = hull2.merged_with(hull_i);
                        let merged_hull = merged_mesh.compute_convex_hull().unwrap().to_mesh();
                        cost_matrix[row_index] = cost::compute_concavity_hulls(
                            hull2,
                            hull_i,
                            &merged_hull,
                            self.parameters.seed,
                            self.parameters.resolution,
                            self.parameters.rv_k,
                        );
                        precost_matrix[row_index] =
                            f32::max(precost_matrix[p2] + min_cost, precost_matrix[i]);
                    } else {
                        cost_matrix[row_index] = f32::INFINITY;
                    }
                    row_index += 1;
                }

                // Remove the row and column corresponding to the merged hull.
                let remove_index = ((cost_size - 1) * cost_size) >> 1;
                if p1 < cost_size {
                    row_index = (index * p1) >> 1;
                    let mut top_row = remove_index;
                    for i in 0..p1 {
                        if i != p2 {
                            cost_matrix[row_index] = cost_matrix[top_row];
                            precost_matrix[row_index] = precost_matrix[top_row];
                        }
                        row_index += 1;
                        top_row += 1;
                    }

                    row_index += p1;
                    top_row += 1;

                    for i in (p1 + 1)..cost_size {
                        cost_matrix[row_index] = cost_matrix[top_row];
                        precost_matrix[row_index] = precost_matrix[top_row];
                        row_index += i;
                        top_row += 1;
                    }
                }

                // Remove the last row and column.
                cost_matrix.truncate(remove_index);
                precost_matrix.truncate(remove_index);
            }
        }

        h
    }

    /// Decomposes a mesh into approximately convex parts using the CoACD algorithm.
    ///
    /// Returns a vector of convex hulls representing the decomposed parts.
    pub fn decompose(&self, mesh: &IndexedMesh) -> Vec<IndexedMesh> {
        let mut mesh = mesh.clone();

        mesh.merge_duplicate_vertices();

        // Normalize the mesh to improve numerical stability.
        let original_aabb = point_cloud::normalize_point_cloud(&mut mesh.vertices);

        // Align the mesh with its principal axes if PCA is enabled.
        // TODO: This doesn't really work?
        let mut pca = self
            .parameters
            .pca
            .then(|| point_cloud::Pca::try_from_points(&mesh.vertices, 1e-6).ok())
            .flatten();
        if let Some(pca) = &mut pca {
            pca.apply_transform(&mut mesh.vertices);
        }

        let mut input_parts = vec![mesh];
        let mut hull_parts: Vec<IndexedMesh> = vec![];
        let mut mesh_parts: Vec<IndexedMesh> = vec![];

        while !input_parts.is_empty() {
            let results = core::mem::take(&mut input_parts)
                .into_par_iter()
                .map(|part| {
                    // Compute the convex hull.
                    let hull = part.compute_convex_hull().unwrap().to_mesh();

                    let mut rng = StdRng::seed_from_u64(self.parameters.seed);
                    let seed_dist = Uniform::<u64>::new(0, 1000).unwrap();
                    // Compute the concavity.
                    let concavity = cost::compute_concavity(
                        &part,
                        &hull,
                        // TODO: Random seed per part?
                        seed_dist.sample(&mut rng),
                        self.parameters.resolution,
                        self.parameters.rv_k,
                    );

                    if concavity > self.parameters.threshold {
                        // Find cutting plane using MCTS.
                        let mut tree = mcts::MonteCarloTree::new(part.clone(), &self.parameters);

                        if let Some(best_plane) = tree.search(&self.parameters) {
                            // Split the part using the best plane.
                            let clip_result =
                                clip::clip(&part, &best_plane).expect("Failed to clip mesh");

                            // Add the resulting parts to the temporary list if they are non-empty.
                            let mut temp = vec![];
                            if !clip_result.positive_mesh.indices.is_empty() {
                                temp.push(clip_result.positive_mesh);
                            }
                            if !clip_result.negative_mesh.indices.is_empty() {
                                temp.push(clip_result.negative_mesh);
                            }
                            (Some(temp), vec![], vec![])
                        } else {
                            // If no plane was found, add the part to the output as is.
                            (None, vec![hull], vec![part])
                        }
                    } else {
                        // Part is sufficiently convex, add to output.
                        (None, vec![hull], vec![part])
                    }
                })
                .collect::<Vec<_>>();

            for (new_parts, new_hulls, new_meshes) in results {
                if let Some(new_parts) = new_parts {
                    input_parts.extend(new_parts);
                }
                hull_parts.extend(new_hulls);
                mesh_parts.extend(new_meshes);
            }
        }

        if self.parameters.merge_convex_hulls {
            // TODO: Don't hardcode the threshold here.
            self.merge_convex_hulls(&mesh_parts, &mut hull_parts, 0.1);
        }

        // Recover the original scale and position of the hulls.
        for hull in &mut hull_parts {
            // Revert the PCA transformation if applied.
            if let Some(pca) = &mut pca {
                pca.revert_transform(&mut hull.vertices);
            }
            point_cloud::recover_point_cloud(&mut hull.vertices, &original_aabb);
        }

        hull_parts
    }
}

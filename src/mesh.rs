//! Indexed 3D triangle mesh.

use glam::Vec3A;
use hashbrown::hash_map::Entry;
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use obvhs::aabb::Aabb;
use quickhull::ConvexHull3dError;
use rand::{Rng, SeedableRng, distr::Uniform, rngs::StdRng};

use crate::{Plane, PlaneSide, collections::HashMap, hashable_partial_eq::HashablePartialEq, math};

/// A 3D triangle mesh represented by vertices and indices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IndexedMesh {
    /// The vertices of the mesh.
    pub vertices: Vec<Vec3A>,
    /// The indices of the mesh.
    pub indices: Vec<[u32; 3]>,
}

impl IndexedMesh {
    /// Clears the mesh data.
    #[inline]
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// Returns `true` if the mesh has no vertices or no indices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }

    /// Computes the axis-aligned bounding box (AABB) of the mesh.
    #[inline]
    pub fn compute_aabb(&self) -> Aabb {
        Aabb::from_points(&self.vertices)
    }

    /// Computes the convex hull of the mesh.
    #[inline]
    pub fn compute_convex_hull(&self) -> Result<IndexedMesh, ConvexHull3dError> {
        if self.vertices.len() < 4 {
            return Ok(IndexedMesh::default());
        }
        quickhull::ConvexHull3d::try_from_points(&self.vertices, None).map(|hull| {
            let (vertices, indices) = hull.vertices_indices();
            IndexedMesh { vertices, indices }
        })
    }

    /// Merges this mesh with another mesh, returning a new mesh.
    ///
    /// The vertices and indices of the other mesh are appended to this mesh.
    /// The indices of the other mesh are offset by the number of vertices in this mesh.
    #[inline]
    pub fn merged_with(&self, other: &IndexedMesh) -> IndexedMesh {
        let mut merged = self.clone();
        let base_index = merged.vertices.len() as u32;
        merged.vertices.extend_from_slice(&other.vertices);
        merged.indices.extend(other.indices.iter().map(|tri| {
            [
                tri[0] + base_index,
                tri[1] + base_index,
                tri[2] + base_index,
            ]
        }));
        merged
    }

    /// Merges all duplicate vertices and adjusts the index buffer accordingly.
    pub fn merge_duplicate_vertices(&mut self) {
        let mut vtx_to_id = HashMap::default();
        let mut new_vertices = Vec::with_capacity(self.vertices.len());
        let mut new_indices = Vec::with_capacity(self.indices.len());

        fn resolve_coord_id(
            coord: &Vec3A,
            vtx_to_id: &mut HashMap<HashablePartialEq<Vec3A>, u32>,
            new_vertices: &mut Vec<Vec3A>,
        ) -> u32 {
            let key = HashablePartialEq::new(*coord);
            let id = match vtx_to_id.entry(key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(new_vertices.len() as u32),
            };

            if *id == new_vertices.len() as u32 {
                new_vertices.push(*coord);
            }

            *id
        }

        for [t0, t1, t2] in self.indices.iter().copied() {
            let va = resolve_coord_id(
                &self.vertices[t0 as usize],
                &mut vtx_to_id,
                &mut new_vertices,
            );
            let vb = resolve_coord_id(
                &self.vertices[t1 as usize],
                &mut vtx_to_id,
                &mut new_vertices,
            );
            let vc = resolve_coord_id(
                &self.vertices[t2 as usize],
                &mut vtx_to_id,
                &mut new_vertices,
            );
            new_indices.push([va, vb, vc]);
        }

        new_vertices.shrink_to_fit();

        self.vertices = new_vertices;
        self.indices = new_indices;
    }

    /// Computes the surface area of the mesh.
    #[inline]
    pub fn surface_area(&self) -> f32 {
        self.indices
            .iter()
            .map(|tri| {
                let v0 = self.vertices[tri[0] as usize];
                let v1 = self.vertices[tri[1] as usize];
                let v2 = self.vertices[tri[2] as usize];
                math::triangle_area(v0, v1, v2)
            })
            .sum()
    }

    /// Computes the signed volume of the mesh.
    ///
    /// Returns a positive value if the mesh is oriented counter-clockwise,
    /// and a negative value if it is oriented clockwise.
    #[inline]
    pub fn signed_volume(&self) -> f32 {
        // Compute the signed volume of the mesh using the divergence theorem.
        self.indices
            .iter()
            .map(|tri| {
                let v0 = self.vertices[tri[0] as usize];
                let v1 = self.vertices[tri[1] as usize];
                let v2 = self.vertices[tri[2] as usize];
                math::tetrahedron_signed_volume(v0, v1, v2)
            })
            .sum()
    }

    /// Finds a separating plane between this convex mesh and another convex mesh, if one exists.
    /// Only the faces of `self` are considered as potential separating planes.
    ///
    /// A separating plane is a plane that divides the space into two half-spaces,
    /// such that all vertices of this mesh are on one side of the plane,
    /// and all vertices of the other mesh are on the opposite side.
    pub fn find_convex_separating_face_plane(&self, other: &IndexedMesh) -> Option<Plane> {
        let mut plane_found = false;

        // Check all triangles in `self` as potential separating planes.
        // TODO: Should we pick the mesh with fewer faces to iterate over?
        for tri in &self.indices {
            let v0 = self.vertices[tri[0] as usize];
            let v1 = self.vertices[tri[1] as usize];
            let v2 = self.vertices[tri[2] as usize];
            let normal = math::triangle_normal(v0, v1, v2);
            let plane = Plane::from_point_and_normal(v0, normal);

            // Find the side of the first vertex in `self` that is not on the plane.
            let mut side1 = PlaneSide::OnPlane;
            for v in self.vertices.iter().copied() {
                let side = plane.side(v, 1e-8);
                if side != PlaneSide::OnPlane {
                    side1 = side;
                    plane_found = true;
                    break;
                }
            }

            // Check if all vertices in `other` are on the opposite side of the plane.
            for v in other.vertices.iter().copied() {
                let side = plane.side(v, 1e-8);
                if !plane_found || side == side1 {
                    plane_found = false;
                    break;
                }
            }

            // If all vertices in `other` are on the opposite side of the plane,
            // we have found a separating plane.
            if plane_found {
                return Some(plane);
            }
        }

        None
    }

    /// Samples points from the surface of the mesh.
    ///
    /// The sampling uses a low-discrepancy [`r2_sequence`] to ensure even coverage of the mesh surface.
    ///
    /// # Parameters
    ///
    /// - `seed`: Seed for the random number generator to ensure reproducibility.
    /// - `resolution`: Approximate number of points to sample from the mesh surface.
    /// - `base`: Base surface area used to adjust the number of samples based on the mesh's surface area.
    ///   If `base` is 0, the number of samples is not adjusted.
    /// - `min_samples`: Minimum number of samples to generate, regardless of surface area.
    /// - `plane`: Optional plane to exclude triangles that lie exactly on the plane.
    /// - `samples`: Output vector to store the sampled points.
    /// - `tri_indices`: Output vector to store the indices of the triangles from which the points were sampled.
    #[expect(clippy::too_many_arguments)]
    pub fn sample_boundary(
        &self,
        seed: u64,
        mut resolution: u32,
        base: f32,
        min_samples: u32,
        plane: Option<Plane>,
        samples: &mut Vec<[f32; 3]>,
        tri_indices: &mut Vec<usize>,
    ) {
        debug_assert_eq!(samples.len(), tri_indices.len());

        if resolution == 0 {
            return;
        }
        let surface_area = self.surface_area();

        if base != 0.0 {
            // Adjust the resolution based on the surface area to maintain a consistent density of samples.
            resolution = ((resolution as f32 * (surface_area / base)) as u32).max(min_samples);
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let seed_dist = Uniform::<u32>::new(0, 1000).unwrap();

        for (i, tri) in self.indices.iter().enumerate() {
            let v0 = self.vertices[tri[0] as usize];
            let v1 = self.vertices[tri[1] as usize];
            let v2 = self.vertices[tri[2] as usize];

            // Skip triangles that lie exactly on the plane (if provided).
            if plane.is_some_and(|p| {
                p.side(v0, 1e-3) == PlaneSide::OnPlane
                    && p.side(v1, 1e-3) == PlaneSide::OnPlane
                    && p.side(v2, 1e-3) == PlaneSide::OnPlane
            }) {
                continue;
            }

            let area = math::triangle_area(v0, v1, v2);

            // Determine the number of samples for this triangle based on its area.
            let num_samples = if self.indices.len() as u32 > resolution {
                u32::max(
                    (i as u32).is_multiple_of(self.indices.len() as u32 / resolution) as u32,
                    (resolution as f32 * area / surface_area) as u32,
                )
            } else {
                u32::max(
                    (i % 2 == 0) as u32,
                    (resolution as f32 * area / surface_area) as u32,
                )
            };

            let seed = rng.sample(seed_dist);

            for k in 0..num_samples {
                // Here, the C++ implementation uses quasirandom i4_sobol samples for even coverage,
                // and uniform random samples for every third sample to add some noise.
                //
                // We deviate from that by using a low-discrepancy R2 sequence for all samples.
                // This gives us even coverage with minimal noise or artifacts, while being
                // much simpler and faster to compute.
                let [a, b] = r2_sequence(k, seed);

                // Barycentric coordinates
                let sqrt_a = a.sqrt();
                let p = (1.0 - sqrt_a) * v0 + (sqrt_a * (1.0 - b)) * v1 + (sqrt_a * b) * v2;

                samples.push(p.to_array());
                tri_indices.push(i);
            }
        }
    }

    /// Computes the minimum distance from the points of this mesh to another mesh.
    #[inline]
    pub fn distance_to_mesh(&self, mesh: &IndexedMesh) -> f32 {
        let points1: Vec<[f32; 3]> = self.vertices.iter().map(|v| v.to_array()).collect();
        let points2: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.to_array()).collect();

        let tree2: ImmutableKdTree<f32, 3> = ImmutableKdTree::new_from_slice(&points2);

        // The squared minimum of the minimum distances.
        let mut min_dist_sq = f32::INFINITY;

        // For each point in `self`, compute the minimum distance to `mesh`.
        for &p in &points1 {
            let nearest = tree2.nearest_one::<SquaredEuclidean>(&p);
            min_dist_sq = min_dist_sq.min(nearest.distance);
        }

        min_dist_sq.sqrt()
    }
}

/// R2 sequence in 2D, as described in "The Unreasonable Effectiveness of Quasirandom Sequences"
/// https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
fn r2_sequence(n: u32, seed: u32) -> [f32; 2] {
    // Inverse of the plastic constant.
    const FRAC_1_PLASTIC: f32 = 0.754_877_7;
    /// Square of the inverse of the plastic constant.
    const FRAC_1_PLASTIC_SQUARED: f32 = 0.569_840_3;
    [
        (seed as f32 + FRAC_1_PLASTIC * n as f32) % 1.0,
        (seed as f32 + FRAC_1_PLASTIC_SQUARED * n as f32) % 1.0,
    ]
}

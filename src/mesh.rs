//! Indexed 3D triangle mesh.

use glam::{DVec3, Vec3A};
use obvhs::aabb::Aabb;
use quickhull::{ConvexHull3d, ConvexHull3dError};
use rand::{Rng, SeedableRng, distr::Uniform, rngs::StdRng};

use crate::{Plane, PlaneSide, math};

/// A 3D triangle mesh represented by vertices and indices.
#[derive(Clone, Debug, Default)]
pub struct IndexedMesh {
    /// The vertices of the mesh.
    pub vertices: Vec<Vec3A>,
    /// The indices of the mesh.
    pub indices: Vec<[usize; 3]>,
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
    pub fn compute_convex_hull(&self) -> Result<ConvexHull3d, ConvexHull3dError> {
        // Convert the vertices to `DVec3` for higher precision during hull computation.
        let points = self
            .vertices
            .iter()
            .map(|v| DVec3::new(v.x as f64, v.y as f64, v.z as f64))
            .collect::<Vec<_>>();

        // Use the `quickhull` crate to compute the convex hull.
        // TODO: Could we make the Quickhull algorithm generic over the vector type
        //       so that we don't have to convert between `Vec3A` and `DVec3`?
        //       Would the loss in precision be acceptable?
        // TODO: This can sometimes fail. Could we have a slower, more robust fallback?
        quickhull::ConvexHull3d::try_from_points(&points, None)
    }

    /// Merges this mesh with another mesh, returning a new mesh.
    ///
    /// The vertices and indices of the other mesh are appended to this mesh.
    /// The indices of the other mesh are offset by the number of vertices in this mesh.
    #[inline]
    pub fn merged_with(&self, other: &IndexedMesh) -> IndexedMesh {
        let mut merged = self.clone();
        let base_index = merged.vertices.len();
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

    /// Computes the surface area of the mesh.
    #[inline]
    pub fn surface_area(&self) -> f32 {
        self.indices.iter().fold(0.0, |acc, tri| {
            let v0 = self.vertices[tri[0]];
            let v1 = self.vertices[tri[1]];
            let v2 = self.vertices[tri[2]];
            acc + math::triangle_area(v0, v1, v2)
        })
    }

    /// Computes the signed volume of the mesh.
    ///
    /// Returns a positive value if the mesh is oriented counter-clockwise,
    /// and a negative value if it is oriented clockwise.
    #[inline]
    pub fn signed_volume(&self) -> f32 {
        // Compute the signed volume of the mesh using the divergence theorem.
        self.indices.iter().fold(0.0, |acc, tri| {
            let v0 = self.vertices[tri[0]];
            let v1 = self.vertices[tri[1]];
            let v2 = self.vertices[tri[2]];
            acc + math::tetrahedron_signed_volume(v0, v1, v2)
        })
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
            let v0 = self.vertices[tri[0]];
            let v1 = self.vertices[tri[1]];
            let v2 = self.vertices[tri[2]];
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
            let v0 = self.vertices[tri[0]];
            let v1 = self.vertices[tri[1]];
            let v2 = self.vertices[tri[2]];

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

/// An extension trait for convex hulls.
pub trait ConvexHullExt {
    /// Converts the convex hull to an [`IndexedMesh`].
    fn to_mesh(&self) -> IndexedMesh;
}

impl ConvexHullExt for ConvexHull3d {
    fn to_mesh(&self) -> IndexedMesh {
        let indices = self
            .triangles()
            .map(|tri| tri.indices())
            .collect::<Vec<_>>();
        let vertices = self
            .points_ref()
            .iter()
            .map(|v| Vec3A::new(v.x as f32, v.y as f32, v.z as f32))
            .collect::<Vec<_>>();

        IndexedMesh { vertices, indices }
    }
}

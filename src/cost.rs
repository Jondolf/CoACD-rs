//! Concavity cost functions.
//!
//! The concavity metric of a solid shape is `concavity = max(h_b, h_i)` (Eq. 4)
//!
//! The terms `h_b` and `h_i` are the Hausdorff distances between sampled point sets of the shape
//! and its convex hull:
//!
//! ```text
//! h_b = hausdorff_distance(sample_boundary(shape), sample_boumdary(convex_hull)) (Eq. 2)
//! h_i = hausdorff_distance(sample_interior(shape), sample_interior(convex_hull)) (Eq. 3)
//! ```
//!
//! Computing an accurate estimate for `h_b` is feasible, but computing `h_i` accurately
//! would require too many samples to be practical. Instead, we use a surrogate term `r_v`:
//!
//! ```text
//! r_v = cbrt(3 * abs(volume(shape) - volume(convex_hull)) / (4 * PI)) (Eq. 5)
//! ```
//!
//! The geometric interpretation of `r_v` is the radius of a sphere with the same volume
//! as the absolute difference in volume between the shape and its convex hull,
//! which is potentially the largest inscribed sphere that could fit inside the concavity.
//!
//! We can prove a theoretical guarantee for `r_v`:
//!
//! ```text
//! sqrt(2) * max(h_b, r_v) >= max(h_b, h_i) (Theorem 1)
//! ```
//!
//! The theorem means that we can use `h_b` and `r_v` to bound the true concavity `max(h_b, h_i)`
//! and be able to recognize any unreasonable approximations. In practice, `r_v` tends to overestimate `h_i`.
//! Thus, we scale it by a coefficient `k` in the range `[0.0, 1.0)`:
//!
//! ```text
//! concavity ≈ max(h_b, k * r_v) (Eq. 6)
//! ```
//!
//! The above equation is our final concavity metric.

use crate::{Plane, hausdorff::face_hausdorff_distance, mesh::IndexedMesh};

/// `3.0 / (PI * 4.0)`
const FRAC_3_PI_4: f32 = 3.0 * std::f32::consts::FRAC_1_PI / 4.0;

/// Computes the surrogate term `r_v` (Eq. 5) used to estimate the interior Hausdorff distance
/// between a shape and its convex hull.
///
/// The geometric interpretation of `r_v` is the radius of a sphere with the same volume
/// as the absolute difference in volume between the shape and its convex hull,
/// which is potentially the largest inscribed sphere that could fit inside the concavity.
///
/// See the [module-level documentation](crate::cost) for more details.
#[inline]
// TODO: Rename to `compute_rv_k` and multiply by `k` here, to prevent accidentally omitting `k`
pub(crate) fn compute_rv(mesh1: &IndexedMesh, mesh2: &IndexedMesh) -> f32 {
    let v1 = mesh1.signed_volume();
    let v2 = mesh2.signed_volume();
    (FRAC_3_PI_4 * (v1 - v2).abs()).cbrt()
}

/// Computes the surrogate term `r_v` (Eq. 5) used to estimate the interior Hausdorff distance
/// between a shape and its convex hull.
///
/// This version takes the combined convex hull as an argument to avoid redundant computations.
/// The geometric interpretation of `r_v` is the radius of a sphere with the same volume
/// as the absolute difference in volume between the shape and its convex hull,
/// which is potentially the largest inscribed sphere that could fit inside the concavity.
///
/// See the [module-level documentation](crate::cost) for more details.
#[inline]
pub(crate) fn compute_rv_hulls(
    hull1: &IndexedMesh,
    hull2: &IndexedMesh,
    combined_hull: &IndexedMesh,
) -> f32 {
    let v1 = hull1.signed_volume();
    let v2 = hull2.signed_volume();
    let v3 = combined_hull.signed_volume();
    (FRAC_3_PI_4 * (v1 + v2 - v3).abs()).cbrt()
}

/// Computes the Hausdorff distance `h_b` (Eq. 2) between two meshes.
///
/// See the [module-level documentation](crate::cost) for more details.
#[inline]
pub(crate) fn compute_hb(
    mesh1: &IndexedMesh,
    mesh2: &IndexedMesh,
    seed: u64,
    resolution: u32,
) -> f32 {
    let mut samples1 = Vec::new();
    let mut indices1 = Vec::new();
    let mut samples2 = Vec::new();
    let mut indices2 = Vec::new();

    // Sample points from both meshes.
    mesh1.sample_boundary(
        seed,
        resolution,
        1.0,
        1000,
        None,
        &mut samples1,
        &mut indices1,
    );
    mesh2.sample_boundary(
        seed,
        resolution,
        1.0,
        1000,
        None,
        &mut samples2,
        &mut indices2,
    );

    if samples1.is_empty() || samples2.is_empty() {
        return f32::INFINITY;
    }

    face_hausdorff_distance(mesh1, mesh2, &samples1, &samples2, &indices1, &indices2)
}

/// Computes the Hausdorff distance `h_b` (Eq. 2) between two meshes, given their convex hull.
///
/// See the [module-level documentation](crate::cost) for more details.
#[inline]
pub(crate) fn compute_hb_hulls(
    hull1: &IndexedMesh,
    hull2: &IndexedMesh,
    combined_hull: &IndexedMesh,
    seed: u64,
    resolution: u32,
) -> f32 {
    if hull1.vertices.is_empty() || hull2.vertices.is_empty() || combined_hull.vertices.is_empty() {
        return 0.0;
    }

    let mut samples1 = Vec::new();
    let mut indices1 = Vec::new();
    let mut samples2 = Vec::new();
    let mut indices2 = Vec::new();

    // Create the merged hull to use for sampling.
    let merged_hull = hull1.merged_with(hull2);

    // Sample points from both meshes.
    extract_point_set(hull1, hull2, seed, resolution, &mut samples1, &mut indices1);
    combined_hull.sample_boundary(
        seed,
        resolution,
        1.0,
        1000,
        None,
        &mut samples2,
        &mut indices2,
    );

    if samples1.is_empty() || samples2.is_empty() {
        return f32::INFINITY;
    }

    face_hausdorff_distance(
        &merged_hull,
        combined_hull,
        &samples1,
        &samples2,
        &indices1,
        &indices2,
    )
}

/// Calls [`compute_rv`] for two meshes and returns the maximum value.
#[inline]
pub(crate) fn compute_total_rv(
    mesh1: &IndexedMesh,
    volume_hull1: &IndexedMesh,
    mesh2: &IndexedMesh,
    volume_hull2: &IndexedMesh,
) -> f32 {
    let h_pos = compute_rv(mesh1, volume_hull1);
    let h_neg = compute_rv(mesh2, volume_hull2);
    h_pos.max(h_neg)
}

/// Computes the concavity metric (Eq. 6) for two meshes.
///
/// See the [module-level documentation](crate::cost) for more details.
#[inline]
pub(crate) fn compute_concavity(
    mesh1: &IndexedMesh,
    mesh2: &IndexedMesh,
    seed: u64,
    resolution: u32,
    k: f32,
) -> f32 {
    let rv = compute_rv(mesh1, mesh2);
    let hb = compute_hb(mesh1, mesh2, seed, resolution);
    hb.max(k * rv)
}

/// Computes the concavity metric (Eq. 6) for two meshes, given their convex hulls.
///
/// See the [module-level documentation](crate::cost) for more details.
#[inline]
pub(crate) fn compute_concavity_hulls(
    hull1: &IndexedMesh,
    hull2: &IndexedMesh,
    combined_hull: &IndexedMesh,
    seed: u64,
    resolution: u32,
    k: f32,
) -> f32 {
    let rv = compute_rv_hulls(hull1, hull2, combined_hull);
    // TODO: Why does the C++ implementation use `resolution + 2000` here?
    let hb = compute_hb_hulls(hull1, hull2, combined_hull, seed, resolution + 2000);
    hb.max(k * rv)
}

/// Computes the concavity metric (Eq. 6) for a positive and negative mesh given their convex hulls.
/// The final concavity is the maximum of the two concavities.
///
/// See the [module-level documentation](crate::cost) for more details.
#[inline]
pub(crate) fn compute_energy(
    positive_mesh: &IndexedMesh,
    positive_hull: &IndexedMesh,
    negative_mesh: &IndexedMesh,
    negative_hull: &IndexedMesh,
    seed: u64,
    resolution: u32,
    k: f32,
) -> f32 {
    let positive_concavity = compute_concavity(positive_mesh, positive_hull, seed, resolution, k);
    let negative_concavity = compute_concavity(negative_mesh, negative_hull, seed, resolution, k);
    positive_concavity.max(negative_concavity)
}

/// Extracts a set of points sampled from the surfaces of two convex hulls.
/// The number of points sampled from each hull is proportional to its surface area.
///
/// `resolution` specifies the (rough) total number of points to sample from both hulls.
/// `samples` is populated with the sampled points, while `indices` is populated with the
/// corresponding triangle indices from which the points were sampled.
pub(crate) fn extract_point_set(
    hull1: &IndexedMesh,
    hull2: &IndexedMesh,
    seed: u64,
    resolution: u32,
    samples: &mut Vec<[f32; 3]>,
    indices: &mut Vec<usize>,
) {
    const MIN_SAMPLES: u32 = 1000;

    let area1 = hull1.surface_area();
    let area2 = hull2.surface_area();
    let total_area_recip = (area1 + area2).recip();

    // Find a separating plane between the two convex hulls, if one exists.
    let plane: Option<Plane> = hull1.find_convex_separating_face_plane(hull2);

    // Distribute the resolution based on surface area.
    let res1 = (area1 * total_area_recip * resolution as f32) as u32;
    let res2 = (area2 * total_area_recip * resolution as f32) as u32;

    // Sample points from both hulls.
    hull1.sample_boundary(seed, res1, 1.0, MIN_SAMPLES, plane, samples, indices);
    let num_indices_before = indices.len();
    hull2.sample_boundary(seed, res2, 1.0, MIN_SAMPLES, plane, samples, indices);

    // Adjust triangle indices for the second hull.
    for tri_index in &mut indices[num_indices_before..] {
        *tri_index += hull1.indices.len();
    }
}

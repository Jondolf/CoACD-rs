//! Mesh clipping with a plane.

use std::collections::VecDeque;

use glam::{Affine3A, Mat3A, Vec2, Vec3A, Vec3Swizzles};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use crate::{
    Aabb,
    collections::{HashMap, HashSet},
    math::{Plane, PlaneSide, triangle_area},
    mesh::IndexedMesh,
};

// TODO: Scaling factor to tune thresholds to different mesh sizes?
const DISTANCE_EPSILON_SQUARED: f32 = 1e-4;
const AREA_EPSILON_SQUARED: f32 = 1e-12;
const COINCIDENT_EPSILON: f32 = 1e-5;
const ON_PLANE_EPSILON: f32 = 1e-6;
const SEGMENT_INTERSECTION_TOLERANCE: f32 = 1e-6;

/// Computes the translation and rotation of a plane defined by three points on the given border.
///
/// Returns `None` if a valid transformation cannot be computed, for example if the points are
/// coincident or collinear.
// TODO: Errors
pub fn compute_plane_transform(plane: &Plane, border: &[Vec3A]) -> Option<Affine3A> {
    let i0 = 0;
    let mut i1 = 0;
    let mut i2 = 0;
    let mut flag = false;

    // Find a point that is not too close to the first point.
    for i in 1..border.len() {
        let distance_squared = border[i0].distance_squared(border[i]);
        if distance_squared > DISTANCE_EPSILON_SQUARED {
            i1 = i;
            flag = true;
            break;
        }
    }

    if !flag {
        return None;
    }

    debug_assert_ne!(i0, i1, "Failed to find a valid second point");

    flag = false;

    // Find another point such that the triangle area is not too small.
    for i in 2..border.len() {
        if i == i1 {
            continue;
        }

        // If cross(AB, BC) is zero, AB and BC are collinear,
        // and the area of the triangle (or parallelogram here) is zero.
        // NOTE: We differ from the C++ implementation here.
        let ab = border[i1] - border[i0];
        let bc = border[i] - border[i0];
        let area_squared = ab.cross(bc).length_squared();

        if area_squared > AREA_EPSILON_SQUARED {
            i2 = i;
            flag = true;
            break;
        }
    }

    if !flag {
        return None;
    }

    debug_assert_ne!(i0, i2, "Failed to find a valid third point");

    let p0 = border[i0];
    let mut p1 = border[i1];
    let mut p2 = border[i2];

    if plane.cut_side(p0, p1, p2) == PlaneSide::Back {
        // Change the winding order.
        core::mem::swap(&mut p1, &mut p2);
    }

    // Translate to the origin.
    let translation = p0;

    // Compute the rotation matrix
    // TODO: Is the normalization for `y_axis` useful here?
    //       Mathematically, it should already be normalized.
    let x_axis = (p0 - p1).normalize();
    let z_axis = (p2 - p0).cross(x_axis).normalize();
    let y_axis = z_axis.cross(x_axis).normalize();
    let rotation = Mat3A::from_cols(x_axis, y_axis, z_axis);

    Some(Affine3A {
        matrix3: rotation,
        translation,
    })
}

/// Errors that can occur during border triangulation.
pub enum TriangulationError {
    /// Computing the plane transformation failed.
    PlaneTransformFailed,
    /// A point's coordinates were too large, too small, or NaN.
    InvalidCoordinate,
}

/// Triangulates a border defined by a set of points and edges on a plane.
fn triangulation(
    plane: &Plane,
    border: &[Vec3A],
    border_edges: Vec<[usize; 2]>,
) -> Result<Vec<[usize; 3]>, TriangulationError> {
    let affine =
        compute_plane_transform(plane, border).ok_or(TriangulationError::PlaneTransformFailed)?;

    let mut points: Vec<Point2<f32>> = Vec::with_capacity(border.len());
    let mut min = Vec2::MAX;
    let mut max = Vec2::MIN;

    // Project the border points onto the plane and operate in local space.
    for p in border.iter().copied() {
        // It's enough to compute the `x` and `y` coordinates.
        // Would a custom transformation be more efficient?
        let local_p = affine.inverse().transform_point3a(p).xy();
        points.push(Point2::new(local_p.x, local_p.y));

        min = min.min(local_p);
        max = max.max(local_p);
    }

    // TODO: What to do with conflicting edges?
    let mut conflicting_edges = Vec::new();
    let triangulation = ConstrainedDelaunayTriangulation::<Point2<f32>>::try_bulk_load_cdt(
        points,
        border_edges
            .iter()
            // TODO: This is silly
            .map(|&[v0, v1]| [v0 - 1, v1 - 1])
            .collect(),
        |e| conflicting_edges.push(e),
    )
    .map_err(|_| TriangulationError::InvalidCoordinate)?;

    let mut border_triangles = Vec::with_capacity(triangulation.num_inner_faces());
    for face in triangulation.inner_faces() {
        border_triangles.push(face.vertices().map(|v| v.index() + 1));
    }

    // TODO: Fix this
    for vertex in triangulation.vertices().skip(border.len()) {
        let p = vertex.position();
        let local_p = Vec3A::new(p.x, p.y, 0.0);
        let _world_p = affine.inverse().transform_point3a(local_p);
        todo!("Not implemented");
    }

    debug_assert_eq!(border.len(), triangulation.num_vertices());

    Ok(border_triangles)
}

/// The result of removing outlier triangles from a border triangulation.
struct RemoveOutliersResult {
    border: Vec<Vec3A>,
    border_triangles: Vec<[usize; 3]>,
}

// TODO: Reuse allocations
fn remove_outlier_triangles(
    border: &[Vec3A],
    overlap: &[Vec3A],
    border_edges: &[[usize; 2]],
    border_triangles: &mut [[usize; 3]],
    vertex_map: &mut HashMap<usize, usize>,
) -> RemoveOutliersResult {
    let mut new_border = Vec::with_capacity(border.len());
    let mut new_border_triangles = Vec::with_capacity(border_triangles.len());

    let mut bfs_edges: VecDeque<[usize; 2]> = VecDeque::from_iter(border_edges.iter().copied());
    let mut edge_map: HashMap<[usize; 2], [isize; 2]> = HashMap::with_capacity(border_edges.len());
    let mut border_set: HashSet<[usize; 2]> = HashSet::with_capacity(border_edges.len());
    let same_edge_set: HashSet<[usize; 2]> = HashSet::from_iter(border_edges.iter().copied());
    let mut overlap_set: HashSet<usize> = HashSet::with_capacity(overlap.len());

    // TODO: Bit vectors?
    let mut add_vertex = vec![false; border.len()];
    let mut remove_map = vec![false; border_triangles.len()];

    // Mark coincident points.
    for p1 in overlap.iter().copied() {
        for (j, p0) in border.iter().enumerate() {
            if p0.abs_diff_eq(p1, COINCIDENT_EPSILON) {
                overlap_set.insert(j + 1);
            }
        }
    }

    // Find true border edges that have no duplicates with reversed direction.
    for [v0, v1] in border_edges.iter().copied() {
        if !same_edge_set.contains(&[v1, v0]) {
            border_set.insert([v0, v1]);
            border_set.insert([v1, v0]);
        }
    }

    for (i, &[v0, v1, v2]) in border_triangles.iter().enumerate() {
        // Ignore points added by triangle
        if !(v0 >= 1
            && v0 <= border.len()
            && v1 >= 1
            && v1 <= border.len()
            && v2 >= 1
            && v2 <= border.len())
        {
            continue;
        }

        let e01 = [v0, v1];
        let e10 = [v1, v0];
        let e12 = [v1, v2];
        let e21 = [v2, v1];
        let e20 = [v2, v0];
        let e02 = [v0, v2];

        if !(same_edge_set.contains(&e10) && same_edge_set.contains(&e01)) {
            if let Some(edge) = edge_map.get_mut(&e10) {
                edge[1] = i as isize;
            } else {
                edge_map.insert(e01, [i as isize, -1]);
            }
        }

        if !(same_edge_set.contains(&e21) && same_edge_set.contains(&e12)) {
            if let Some(edge) = edge_map.get_mut(&e21) {
                edge[1] = i as isize;
            } else {
                edge_map.insert(e12, [i as isize, -1]);
            }
        }

        if !(same_edge_set.contains(&e02) && same_edge_set.contains(&e20)) {
            if let Some(edge) = edge_map.get_mut(&e02) {
                edge[1] = i as isize;
            } else {
                edge_map.insert(e20, [i as isize, -1]);
            }
        }
    }

    let mut i = 0;
    while let Some([v0, v1]) = bfs_edges.pop_front() {
        let mut idx: isize;
        let e01 = [v0, v1];
        let e10 = [v1, v0];
        if i < border_edges.len()
            && let Some(edge) = edge_map.get(&e10)
        {
            idx = edge[1];
            if idx != -1 {
                remove_map[idx as usize] = true;
            }
            idx = edge[0];
            if idx != -1
                && !remove_map[idx as usize]
                && !face_overlap(&overlap_set, border_triangles[idx as usize])
            {
                remove_map[idx as usize] = true;
                new_border_triangles.push(border_triangles[idx as usize]);
                for k in 0..3 {
                    add_vertex[border_triangles[idx as usize][k] - 1] = true;
                }
                let [p0, p1, p2] = border_triangles[idx as usize];
                if p2 != v0 && p2 != v1 {
                    let pt12 = [p1, p2];
                    let pt20 = [p2, p0];
                    if !border_set.contains(&pt12) {
                        bfs_edges.push_back(pt12);
                    }
                    if !border_set.contains(&pt20) {
                        bfs_edges.push_back(pt20);
                    }
                } else if p1 != v0 && p1 != v1 {
                    let pt12 = [p1, p2];
                    let pt01 = [p0, p1];
                    if !border_set.contains(&pt12) {
                        bfs_edges.push_back(pt12);
                    }
                    if !border_set.contains(&pt01) {
                        bfs_edges.push_back(pt01);
                    }
                } else if p0 != v0 && p0 != v1 {
                    let pt01 = [p0, p1];
                    let pt20 = [p2, p0];
                    if !border_set.contains(&pt01) {
                        bfs_edges.push_back(pt01);
                    }
                    if !border_set.contains(&pt20) {
                        bfs_edges.push_back(pt20);
                    }
                }
            }
        } else if i < border_edges.len()
            && let Some(edge) = edge_map.get(&e01)
        {
            idx = edge[0];
            if idx != -1 {
                remove_map[idx as usize] = true;
            }
            idx = edge[1];
            if idx != -1
                && !remove_map[idx as usize]
                && !face_overlap(&overlap_set, border_triangles[idx as usize])
            {
                remove_map[idx as usize] = true;
                new_border_triangles.push(border_triangles[idx as usize]);
                for k in 0..3 {
                    add_vertex[border_triangles[idx as usize][k] - 1] = true;
                }
                let [p0, p1, p2] = border_triangles[idx as usize];
                if p2 != v0 && p2 != v1 {
                    let pt21 = [p2, p1];
                    let pt02 = [p0, p2];
                    if !border_set.contains(&pt21) {
                        bfs_edges.push_back(pt21);
                    }
                    if !border_set.contains(&pt02) {
                        bfs_edges.push_back(pt02);
                    }
                } else if p1 != v0 && p1 != v1 {
                    let pt21 = [p2, p1];
                    let pt10 = [p1, p0];
                    if !border_set.contains(&pt21) {
                        bfs_edges.push_back(pt21);
                    }
                    if !border_set.contains(&pt10) {
                        bfs_edges.push_back(pt10);
                    }
                } else if p0 != v0 && p0 != v1 {
                    let pt10 = [p1, p0];
                    let pt02 = [p0, p2];
                    if !border_set.contains(&pt10) {
                        bfs_edges.push_back(pt10);
                    }
                    if !border_set.contains(&pt02) {
                        bfs_edges.push_back(pt02);
                    }
                }
            }
        } else if i >= border_edges.len()
            && (edge_map.contains_key(&e10) || edge_map.contains_key(&e01))
        {
            for j in 0..2 {
                if j == 0 {
                    if let Some(edge) = edge_map.get(&e01) {
                        idx = edge[0];
                    } else {
                        idx = edge_map.get(&e10).unwrap()[0];
                    }
                } else if let Some(edge) = edge_map.get(&e01) {
                    idx = edge[1];
                } else {
                    idx = edge_map.get(&e10).unwrap()[1];
                }
                if idx != -1 && !remove_map[idx as usize] {
                    remove_map[idx as usize] = true;
                    new_border_triangles.push(border_triangles[idx as usize]);
                    for k in 0..3 {
                        add_vertex[border_triangles[idx as usize][k] - 1] = true;
                    }
                    let [p0, p1, p2] = border_triangles[idx as usize];
                    if p2 != v0 && p2 != v1 {
                        bfs_edges.push_back([p1, p2]);
                        bfs_edges.push_back([p2, p0]);
                    } else if p1 != v0 && p1 != v1 {
                        bfs_edges.push_back([p1, p2]);
                        bfs_edges.push_back([p0, p1]);
                    } else if p0 != v0 && p0 != v1 {
                        bfs_edges.push_back([p0, p1]);
                        bfs_edges.push_back([p2, p0]);
                    }
                }
            }
        }
        i += 1;
    }

    let mut index = 0;
    for (i, point) in border.iter().enumerate() {
        if i < border.len() || add_vertex[i] {
            new_border.push(*point);
            index += 1;
            vertex_map.insert(i + 1, index);
        }
    }

    RemoveOutliersResult {
        border: new_border,
        border_triangles: new_border_triangles,
    }
}

/// Check if a face overlaps with any points in the overlap set.
#[inline(always)]
fn face_overlap(overlap_set: &HashSet<usize>, face: [usize; 3]) -> bool {
    overlap_set.contains(&face[0])
        || overlap_set.contains(&face[1])
        || overlap_set.contains(&face[2])
}

/// The result of clipping a mesh with a plane.
pub struct ClipResult {
    /// The mesh on the positive side of the plane.
    pub positive_mesh: IndexedMesh,
    /// The axis-aligned bounding box of the positive mesh.
    pub positive_aabb: Aabb,
    /// The mesh on the negative side of the plane.
    pub negative_mesh: IndexedMesh,
    /// The axis-aligned bounding box of the negative mesh.
    pub negative_aabb: Aabb,
    /// The area of the cut surface.
    pub cut_area: f32,
}

/// Clips a mesh with a plane, producing two output meshes on either side of the plane.
///
/// # Panics
///
/// Panics if the input `mesh` has more vertices than [`i32::MAX`].
pub fn clip(mesh: &IndexedMesh, plane: &Plane) -> Option<ClipResult> {
    assert!(mesh.vertices.len() <= i32::MAX as usize);

    // Four steps:
    // 1. Find triangles on either side of the plane, and group them into positive and negative sets.
    // 2. Split triangles that intersect the plane, and add them into the two sets.
    // 3. Add new surfaces overlapping with the plane to form solid meshes, using constrained Delaunay triangulation.
    // 4. Remove redundant triangles (if any) introduced by step 3.

    let mut border: Vec<Vec3A> = Vec::new();
    let mut border_edges: Vec<[usize; 2]> = Vec::new();
    let mut overlap: Vec<Vec3A> = Vec::new();
    let mut border_map: HashMap<usize, usize> = HashMap::new();

    let mut idx = 0;
    let mut positive_map = vec![false; mesh.vertices.len()];
    let mut negative_map = vec![false; mesh.vertices.len()];

    let mut positive_vertices = Vec::with_capacity(mesh.vertices.len() / 2);
    let mut negative_vertices = Vec::with_capacity(mesh.vertices.len() / 2);

    // The algorithm currently uses signed indices. For the output meshes,
    // we reinterpret these vectors as unsigned without unnecessary allocations.
    // The caveat is that indices are limited to `i32::MAX` vertices.
    let mut positive_indices: Vec<[isize; 3]> = Vec::with_capacity(mesh.indices.len() / 2);
    let mut negative_indices: Vec<[isize; 3]> = Vec::with_capacity(mesh.indices.len() / 2);

    let mut edge_map: HashMap<[usize; 2], isize> = HashMap::new();
    let mut vertex_map: HashMap<usize, usize> = HashMap::new();

    for [id0, id1, id2] in mesh.indices.iter().copied() {
        let p0 = mesh.vertices[id0];
        let p1 = mesh.vertices[id1];
        let p2 = mesh.vertices[id2];

        let mut side0 = plane.side(p0, ON_PLANE_EPSILON);
        let mut side1 = plane.side(p1, ON_PLANE_EPSILON);
        let mut side2 = plane.side(p2, ON_PLANE_EPSILON);
        let mut sum = (side0 as i8) + (side1 as i8) + (side2 as i8);

        if side0 == PlaneSide::OnPlane && side1 == PlaneSide::OnPlane && side2 == PlaneSide::OnPlane
        {
            let cut_side = plane.cut_side(p0, p1, p2);
            side0 = cut_side;
            side1 = cut_side;
            side2 = cut_side;
            sum = (side0 as i8) + (side1 as i8) + (side2 as i8);
            overlap.push(p0);
            overlap.push(p1);
            overlap.push(p2);
        }

        if sum >= 2
            || (sum == 1
                // TODO: Double-check that this works. The C++ version uses a more complex check
                //       that should be equivalent.
                && (side0 == PlaneSide::OnPlane
                    || side1 == PlaneSide::OnPlane
                    || side2 == PlaneSide::OnPlane))
        {
            // Positive side
            positive_map[id0] = true;
            positive_map[id1] = true;
            positive_map[id2] = true;
            positive_indices.push([id0 as isize, id1 as isize, id2 as isize]);

            // The plane crosses the triangle edge.
            if sum == 1 {
                if side0 == PlaneSide::Front && side1 == PlaneSide::OnPlane {
                    add_point(&mut vertex_map, &mut border, p1, id1, &mut idx);
                    add_point(&mut vertex_map, &mut border, p2, id2, &mut idx);
                    if let Some(v1) = vertex_map.get(&id1)
                        && let Some(v2) = vertex_map.get(&id2)
                        && v1 != v2
                    {
                        border_edges.push([*v1 + 1, *v2 + 1]);
                    }
                } else if side1 == PlaneSide::Front && side0 == PlaneSide::OnPlane {
                    add_point(&mut vertex_map, &mut border, p2, id2, &mut idx);
                    add_point(&mut vertex_map, &mut border, p0, id0, &mut idx);
                    if let Some(v2) = vertex_map.get(&id2)
                        && let Some(v0) = vertex_map.get(&id0)
                        && v2 != v0
                    {
                        border_edges.push([*v2 + 1, *v0 + 1]);
                    }
                } else if side2 == PlaneSide::Front && side1 == PlaneSide::OnPlane {
                    add_point(&mut vertex_map, &mut border, p0, id0, &mut idx);
                    add_point(&mut vertex_map, &mut border, p1, id1, &mut idx);
                    if let Some(v0) = vertex_map.get(&id0)
                        && let Some(v1) = vertex_map.get(&id1)
                        && v0 != v1
                    {
                        border_edges.push([*v0 + 1, *v1 + 1]);
                    }
                }
            }
        } else if sum <= -2
            || (sum == -1
                && (side0 == PlaneSide::OnPlane
                    || side1 == PlaneSide::OnPlane
                    || side2 == PlaneSide::OnPlane))
        {
            // Negative side
            negative_map[id0] = true;
            negative_map[id1] = true;
            negative_map[id2] = true;
            negative_indices.push([id0 as isize, id1 as isize, id2 as isize]);

            // The plane crosses the triangle edge.
            if sum == -1 {
                if side0 == PlaneSide::Back && side1 == PlaneSide::OnPlane {
                    add_point(&mut vertex_map, &mut border, p2, id2, &mut idx);
                    add_point(&mut vertex_map, &mut border, p1, id1, &mut idx);
                    if let Some(v2) = vertex_map.get(&id2)
                        && let Some(v1) = vertex_map.get(&id1)
                        && v2 != v1
                    {
                        border_edges.push([*v2 + 1, *v1 + 1]);
                    }
                } else if side1 == PlaneSide::Back && side0 == PlaneSide::OnPlane {
                    add_point(&mut vertex_map, &mut border, p0, id0, &mut idx);
                    add_point(&mut vertex_map, &mut border, p2, id2, &mut idx);
                    if let Some(v0) = vertex_map.get(&id0)
                        && let Some(v2) = vertex_map.get(&id2)
                        && v0 != v2
                    {
                        border_edges.push([*v0 + 1, *v2 + 1]);
                    }
                } else if side2 == PlaneSide::Back && side0 == PlaneSide::OnPlane {
                    add_point(&mut vertex_map, &mut border, p1, id1, &mut idx);
                    add_point(&mut vertex_map, &mut border, p0, id0, &mut idx);
                    if let Some(v1) = vertex_map.get(&id1)
                        && let Some(v0) = vertex_map.get(&id0)
                        && v1 != v0
                    {
                        border_edges.push([*v1 + 1, *v0 + 1]);
                    }
                }
            }
        } else {
            // Different side
            let pi0 = plane.intersect_segment(p0, p1, SEGMENT_INTERSECTION_TOLERANCE);
            let pi1 = plane.intersect_segment(p1, p2, SEGMENT_INTERSECTION_TOLERANCE);
            let pi2 = plane.intersect_segment(p2, p0, SEGMENT_INTERSECTION_TOLERANCE);

            if let Some(pi0) = pi0
                && let Some(pi1) = pi1
                && pi2.is_none()
            {
                // Record the points.
                add_edge_point(&mut edge_map, &mut border, pi0, id0, id1, &mut idx);
                add_edge_point(&mut edge_map, &mut border, pi1, id1, id2, &mut idx);

                // Record the edges.
                let id_pi0 = *edge_map.get(&[id0, id1]).unwrap();
                let id_pi1 = *edge_map.get(&[id1, id2]).unwrap();
                if side1 == PlaneSide::Front {
                    if id_pi1 != id_pi0 {
                        border_edges.push([id_pi1 as usize + 1, id_pi0 as usize + 1]);
                        positive_map[id1] = true;
                        negative_map[id0] = true;
                        negative_map[id2] = true;
                        positive_indices.push([id1 as isize, -id_pi1 - 1, -id_pi0 - 1]);
                        negative_indices.push([id0 as isize, -id_pi0 - 1, -id_pi1 - 1]);
                        negative_indices.push([-id_pi1 - 1, id2 as isize, id0 as isize]);
                    } else {
                        negative_map[id0] = true;
                        negative_map[id2] = true;
                        negative_indices.push([-id_pi1 - 1, id2 as isize, id0 as isize]);
                    }
                } else if id_pi0 != id_pi1 {
                    border_edges.push([id_pi0 as usize + 1, id_pi1 as usize + 1]);
                    negative_map[id1] = true;
                    positive_map[id0] = true;
                    positive_map[id2] = true;
                    negative_indices.push([id1 as isize, -id_pi1 - 1, -id_pi0 - 1]);
                    positive_indices.push([id0 as isize, -id_pi0 - 1, -id_pi1 - 1]);
                    positive_indices.push([-id_pi1 - 1, id2 as isize, id0 as isize]);
                } else {
                    positive_map[id0] = true;
                    positive_map[id2] = true;
                    positive_indices.push([-id_pi1 - 1, id2 as isize, id0 as isize]);
                }
            } else if let Some(pi1) = pi1
                && let Some(pi2) = pi2
                && pi0.is_none()
            {
                // Record the points.
                add_edge_point(&mut edge_map, &mut border, pi1, id1, id2, &mut idx);
                add_edge_point(&mut edge_map, &mut border, pi2, id2, id0, &mut idx);

                // Record the edges.
                let id_pi1 = *edge_map.get(&[id1, id2]).unwrap();
                let id_pi2 = *edge_map.get(&[id2, id0]).unwrap();
                if side2 == PlaneSide::Front {
                    if id_pi2 != id_pi1 {
                        border_edges.push([id_pi2 as usize + 1, id_pi1 as usize + 1]);
                        positive_map[id2] = true;
                        negative_map[id0] = true;
                        negative_map[id1] = true;
                        positive_indices.push([id2 as isize, -id_pi2 - 1, -id_pi1 - 1]);
                        negative_indices.push([id0 as isize, -id_pi1 - 1, -id_pi2 - 1]);
                        negative_indices.push([-id_pi1 - 1, id0 as isize, id1 as isize]);
                    } else {
                        negative_map[id0] = true;
                        negative_map[id1] = true;
                        negative_indices.push([-id_pi1 - 1, id0 as isize, id1 as isize]);
                    }
                } else if id_pi1 != id_pi2 {
                    border_edges.push([id_pi1 as usize + 1, id_pi2 as usize + 1]);
                    negative_map[id2] = true;
                    positive_map[id0] = true;
                    positive_map[id1] = true;
                    negative_indices.push([id2 as isize, -id_pi2 - 1, -id_pi1 - 1]);
                    positive_indices.push([id0 as isize, -id_pi1 - 1, -id_pi2 - 1]);
                    positive_indices.push([-id_pi1 - 1, id0 as isize, id1 as isize]);
                } else {
                    positive_map[id0] = true;
                    positive_map[id1] = true;
                    positive_indices.push([-id_pi1 - 1, id0 as isize, id1 as isize]);
                }
            } else if let Some(pi2) = pi2
                && let Some(pi0) = pi0
                && pi1.is_none()
            {
                // Record the points.
                add_edge_point(&mut edge_map, &mut border, pi2, id2, id0, &mut idx);
                add_edge_point(&mut edge_map, &mut border, pi0, id0, id1, &mut idx);

                // Record the edges.
                let id_pi0 = *edge_map.get(&[id0, id1]).unwrap();
                let id_pi2 = *edge_map.get(&[id2, id0]).unwrap();
                if side0 == PlaneSide::Front {
                    if id_pi0 != id_pi2 {
                        border_edges.push([id_pi0 as usize + 1, id_pi2 as usize + 1]);
                        positive_map[id0] = true;
                        negative_map[id1] = true;
                        negative_map[id2] = true;
                        positive_indices.push([id0 as isize, -id_pi0 - 1, -id_pi2 - 1]);
                        negative_indices.push([id1 as isize, -id_pi2 - 1, -id_pi0 - 1]);
                        negative_indices.push([-id_pi2 - 1, id1 as isize, id2 as isize]);
                    } else {
                        negative_map[id1] = true;
                        negative_map[id2] = true;
                        negative_indices.push([-id_pi2 - 1, id1 as isize, id2 as isize]);
                    }
                } else if id_pi2 != id_pi0 {
                    border_edges.push([id_pi2 as usize + 1, id_pi0 as usize + 1]);
                    negative_map[id0] = true;
                    positive_map[id1] = true;
                    positive_map[id2] = true;
                    negative_indices.push([id0 as isize, -id_pi0 - 1, -id_pi2 - 1]);
                    positive_indices.push([id1 as isize, -id_pi2 - 1, -id_pi0 - 1]);
                    positive_indices.push([-id_pi2 - 1, id1 as isize, id2 as isize]);
                } else {
                    positive_map[id1] = true;
                    positive_map[id2] = true;
                    positive_indices.push([-id_pi2 - 1, id1 as isize, id2 as isize]);
                }
            } else if let Some(pi0) = pi0
                && let Some(pi1) = pi1
                && let Some(pi2) = pi2
            {
                if side0 == PlaneSide::OnPlane
                    || (side1 != PlaneSide::OnPlane
                        && side2 != PlaneSide::OnPlane
                        && pi0.abs_diff_eq(pi2, COINCIDENT_EPSILON))
                {
                    add_point(&mut vertex_map, &mut border, p0, id0, &mut idx);
                    let id_pi0 = *vertex_map.get(&id0).unwrap() as isize;
                    edge_map.insert([id0, id1], id_pi0);
                    edge_map.insert([id1, id0], id_pi0);
                    edge_map.insert([id2, id0], id_pi0);
                    edge_map.insert([id0, id2], id_pi0);

                    add_edge_point(&mut edge_map, &mut border, pi1, id1, id2, &mut idx);
                    let id_pi1 = *edge_map.get(&[id1, id2]).unwrap();
                    if side1 == PlaneSide::Front {
                        if id_pi1 != id_pi0 {
                            border_edges.push([id_pi1 as usize + 1, id_pi0 as usize + 1]);
                            positive_map[id1] = true;
                            negative_map[id2] = true;
                            positive_indices.push([id1 as isize, -id_pi1 - 1, -id_pi0 - 1]);
                            negative_indices.push([id2 as isize, -id_pi0 - 1, -id_pi1 - 1]);
                        }
                    } else if id_pi0 != id_pi1 {
                        border_edges.push([id_pi0 as usize + 1, id_pi1 as usize + 1]);
                        negative_map[id1] = true;
                        positive_map[id2] = true;
                        negative_indices.push([id1 as isize, -id_pi1 - 1, -id_pi0 - 1]);
                        positive_indices.push([id2 as isize, -id_pi0 - 1, -id_pi1 - 1]);
                    }
                } else if side1 == PlaneSide::OnPlane
                    || (side0 != PlaneSide::OnPlane
                        && side2 != PlaneSide::OnPlane
                        && pi0.abs_diff_eq(pi1, COINCIDENT_EPSILON))
                {
                    add_point(&mut vertex_map, &mut border, p1, id1, &mut idx);
                    let id_pi1 = *vertex_map.get(&id1).unwrap() as isize;
                    edge_map.insert([id0, id1], id_pi1);
                    edge_map.insert([id1, id0], id_pi1);
                    edge_map.insert([id1, id2], id_pi1);
                    edge_map.insert([id2, id1], id_pi1);

                    add_edge_point(&mut edge_map, &mut border, pi2, id2, id0, &mut idx);
                    let id_pi2 = *edge_map.get(&[id2, id0]).unwrap();
                    if side0 == PlaneSide::Front {
                        if id_pi1 != id_pi2 {
                            border_edges.push([id_pi1 as usize + 1, id_pi2 as usize + 1]);
                            positive_map[id0] = true;
                            negative_map[id2] = true;
                            positive_indices.push([id0 as isize, -id_pi1 - 1, -id_pi2 - 1]);
                            negative_indices.push([id2 as isize, -id_pi2 - 1, -id_pi1 - 1]);
                        }
                    } else if id_pi2 != id_pi1 {
                        border_edges.push([id_pi2 as usize + 1, id_pi1 as usize + 1]);
                        positive_map[id0] = true;
                        negative_map[id2] = true;
                        negative_indices.push([id0 as isize, -id_pi1 - 1, -id_pi2 - 1]);
                        positive_indices.push([id2 as isize, -id_pi2 - 1, -id_pi1 - 1]);
                    }
                } else if side2 == PlaneSide::OnPlane
                    || (side0 != PlaneSide::OnPlane
                        && side1 != PlaneSide::OnPlane
                        && pi1.abs_diff_eq(pi2, COINCIDENT_EPSILON))
                {
                    add_point(&mut vertex_map, &mut border, p2, id2, &mut idx);
                    let id_pi2 = *vertex_map.get(&id2).unwrap() as isize;
                    edge_map.insert([id1, id2], id_pi2);
                    edge_map.insert([id2, id1], id_pi2);
                    edge_map.insert([id2, id0], id_pi2);
                    edge_map.insert([id0, id2], id_pi2);

                    add_edge_point(&mut edge_map, &mut border, pi0, id0, id1, &mut idx);
                    let id_pi0 = *edge_map.get(&[id0, id1]).unwrap();
                    if side0 == PlaneSide::Front {
                        if id_pi0 != id_pi2 {
                            border_edges.push([id_pi0 as usize + 1, id_pi2 as usize + 1]);
                            positive_map[id0] = true;
                            negative_map[id1] = true;
                            positive_indices.push([id0 as isize, -id_pi0 - 1, -id_pi2 - 1]);
                            negative_indices.push([id1 as isize, -id_pi2 - 1, -id_pi0 - 1]);
                        }
                    } else if id_pi2 != id_pi0 {
                        border_edges.push([id_pi2 as usize + 1, id_pi0 as usize + 1]);
                        negative_map[id0] = true;
                        positive_map[id1] = true;
                        negative_indices.push([id0 as isize, -id_pi0 - 1, -id_pi2 - 1]);
                        positive_indices.push([id1 as isize, -id_pi2 - 1, -id_pi0 - 1]);
                    }
                } else {
                    unreachable!("All three intersection points should not be unique.");
                }
            }
        }
    }

    let final_border: Vec<Vec3A>;
    let mut final_triangles: Vec<[usize; 3]> = Vec::new();
    let mut cut_area: f32;

    if border.len() > 2 {
        // TODO: Avoid this clone.
        match triangulation(plane, &border, border_edges.clone()) {
            Ok(mut triangulation) => {
                let result = remove_outlier_triangles(
                    &border,
                    &overlap,
                    &border_edges,
                    &mut triangulation,
                    &mut border_map,
                );
                final_border = result.border;
                final_triangles = result.border_triangles;
            }
            Err(TriangulationError::PlaneTransformFailed) => {
                final_border = border;
            }
            // Clipping failed.
            _ => return None,
        }
        cut_area = 0.0;
    } else {
        final_border = border;
        cut_area = -10.0;
    }

    // Original points in two parts
    let mut positive_aabb = Aabb::INVALID;
    let mut negative_aabb = Aabb::INVALID;

    let mut positive_index = 0;
    let mut negative_index = 0;
    let mut positive_projection = vec![0; mesh.vertices.len()];
    let mut negative_projection = vec![0; mesh.vertices.len()];

    for (i, vertex) in mesh.vertices.iter().enumerate() {
        if positive_map[i] {
            positive_vertices.push(*vertex);
            positive_index += 1;
            positive_projection[i] = positive_index;
            positive_aabb.min = positive_aabb.min.min(*vertex);
            positive_aabb.max = positive_aabb.max.max(*vertex);
        }
        if negative_map[i] {
            negative_vertices.push(*vertex);
            negative_index += 1;
            negative_projection[i] = negative_index;
            negative_aabb.min = negative_aabb.min.min(*vertex);
            negative_aabb.max = negative_aabb.max.max(*vertex);
        }
    }

    let pos_n = positive_vertices.len() as isize;
    let neg_n = negative_vertices.len() as isize;

    // Border points and triangles
    for vertex in final_border.iter().copied() {
        positive_vertices.push(vertex);
        positive_aabb.min = positive_aabb.min.min(vertex);
        positive_aabb.max = positive_aabb.max.max(vertex);

        negative_vertices.push(vertex);
        negative_aabb.min = negative_aabb.min.min(vertex);
        negative_aabb.max = negative_aabb.max.max(vertex);
    }

    // Triangles
    for [id0, id1, id2] in positive_indices.iter_mut() {
        *id0 = if *id0 >= 0 {
            positive_projection[*id0 as usize] - 1
        } else {
            -*id0 + pos_n - 1
        };
        *id1 = if *id1 >= 0 {
            positive_projection[*id1 as usize] - 1
        } else {
            -*id1 + pos_n - 1
        };
        *id2 = if *id2 >= 0 {
            positive_projection[*id2 as usize] - 1
        } else {
            -*id2 + pos_n - 1
        };
    }
    for [id0, id1, id2] in negative_indices.iter_mut() {
        *id0 = if *id0 >= 0 {
            negative_projection[*id0 as usize] - 1
        } else {
            -*id0 + neg_n - 1
        };
        *id1 = if *id1 >= 0 {
            negative_projection[*id1 as usize] - 1
        } else {
            -*id1 + neg_n - 1
        };
        *id2 = if *id2 >= 0 {
            negative_projection[*id2 as usize] - 1
        } else {
            -*id2 + neg_n - 1
        };
    }

    // Add the cut triangles.
    for [id0, id1, id2] in final_triangles.iter() {
        positive_indices.push([
            *border_map.get(id0).unwrap() as isize + pos_n - 1,
            *border_map.get(id1).unwrap() as isize + pos_n - 1,
            *border_map.get(id2).unwrap() as isize + pos_n - 1,
        ]);
        negative_indices.push([
            *border_map.get(id2).unwrap() as isize + neg_n - 1,
            *border_map.get(id1).unwrap() as isize + neg_n - 1,
            *border_map.get(id0).unwrap() as isize + neg_n - 1,
        ]);

        let p0 = final_border[*id0 - 1];
        let p1 = final_border[*id1 - 1];
        let p2 = final_border[*id2 - 1];
        cut_area += triangle_area(p0, p1, p2);
    }

    Some(ClipResult {
        positive_mesh: IndexedMesh {
            vertices: positive_vertices,
            indices: cast_indices_isize_to_usize(positive_indices),
        },
        positive_aabb,
        negative_mesh: IndexedMesh {
            vertices: negative_vertices,
            indices: cast_indices_isize_to_usize(negative_indices),
        },
        negative_aabb,
        cut_area,
    })
}

/// Casts a vector of `isize` arrays to a vector of `usize` arrays.
fn cast_indices_isize_to_usize(vec: Vec<[isize; 3]>) -> Vec<[usize; 3]> {
    assert_eq!(core::mem::size_of::<isize>(), core::mem::size_of::<usize>());
    assert_eq!(
        core::mem::align_of::<isize>(),
        core::mem::align_of::<usize>()
    );

    let len = vec.len();
    let capacity = vec.capacity();
    let ptr = vec.as_ptr();
    core::mem::forget(vec);

    unsafe { Vec::from_raw_parts(ptr as *mut [usize; 3], len, capacity) }
}

/// Add a point to the border if it is not already present.
pub fn add_point(
    vertex_map: &mut HashMap<usize, usize>,
    border: &mut Vec<Vec3A>,
    p: Vec3A,
    id: usize,
    idx: &mut usize,
) {
    if !vertex_map.contains_key(&id) {
        if let Some(i) = border
            .iter()
            .position(|&bp| bp.abs_diff_eq(p, COINCIDENT_EPSILON))
        {
            vertex_map.insert(id, i);
        } else {
            border.push(p);
            vertex_map.insert(id, *idx);
            *idx += 1;
        }
    }
}

/// Add an edge point to the border if it is not already present.
pub fn add_edge_point(
    edge_map: &mut HashMap<[usize; 2], isize>,
    border: &mut Vec<Vec3A>,
    p: Vec3A,
    id0: usize,
    id1: usize,
    idx: &mut usize,
) {
    let e01 = [id0, id1];
    let e10 = [id1, id0];
    if !edge_map.contains_key(&e01) && !edge_map.contains_key(&e10) {
        if let Some(i) = border
            .iter()
            .position(|&bp| bp.abs_diff_eq(p, COINCIDENT_EPSILON))
        {
            edge_map.insert(e01, i as isize);
            edge_map.insert(e10, i as isize);
        } else {
            border.push(p);
            edge_map.insert(e01, *idx as isize);
            edge_map.insert(e10, *idx as isize);
            *idx += 1;
        }
    }
}

// TODO: Tests

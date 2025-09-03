use std::collections::VecDeque;

use glam::{Affine3A, Mat3A, Vec2, Vec3A, Vec3Swizzles};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use crate::{
    collections::{HashMap, HashSet},
    math::{Plane, PlaneSide},
};

const DISTANCE_EPSILON_SQUARED: f32 = 1e-4;
const AREA_EPSILON_SQUARED: f32 = 1e-12;
const COINCIDENT_EPSILON: f32 = 1e-5;

// TODO: Errors
/// Computes the translation and rotation of a plane defined by three points on the given border.
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

// TODO: `TriangulationError`
pub fn triangulation(
    plane: &Plane,
    border: &[Vec3A],
    border_edges: Vec<[usize; 2]>,
) -> Option<Vec<[usize; 3]>> {
    let affine = compute_plane_transform(plane, border)?;

    let mut points: Vec<Point2<f32>> = Vec::with_capacity(border.len());
    let mut min = Vec2::MAX;
    let mut max = Vec2::MIN;

    // Project the border points onto the plane and operate in local space.
    for p in border.iter().copied() {
        // It's enough to compute the `x` and `y` coordinates.
        // Would a custom transformation be more efficient?
        let local_p = affine.transform_point3a(p).xy();
        points.push(Point2::new(local_p.x, local_p.y));

        min = min.min(local_p);
        max = max.max(local_p);
    }

    let triangulation =
        ConstrainedDelaunayTriangulation::<Point2<f32>>::bulk_load_cdt(points, border_edges)
            .ok()?;

    let mut border_triangles = Vec::with_capacity(triangulation.num_inner_faces());
    for face in triangulation.inner_faces() {
        border_triangles.push(face.vertices().map(|v| v.index()));
    }

    // TODO: Here, the C++ implementation has a loop to add more points to the border,
    //       it doesn't seem to actually iterate over anything? Double-check this.
    debug_assert_eq!(border.len(), triangulation.num_vertices());

    Some(border_triangles)
}

pub struct RemoveOutliersResult {
    pub border: Vec<Vec3A>,
    pub border_triangles: Vec<[usize; 3]>,
}

// TODO: Reuse allocations
pub fn remove_outlier_triangles(
    border: &[Vec3A],
    overlap: &[Vec3A],
    border_edges: Vec<[usize; 2]>,
    border_triangles: &mut Vec<[usize; 3]>,
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
        let mut idx = 0;
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
                    add_vertex[border_triangles[idx as usize][k] - 1 as usize] = true;
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
                    add_vertex[border_triangles[idx as usize][k] - 1 as usize] = true;
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
                        add_vertex[border_triangles[idx as usize][k] - 1 as usize] = true;
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

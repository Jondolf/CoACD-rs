use glam::{Vec3A, Vec4};

/// A plane in 3D space, defined by a normal vector and a distance from the origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// The normal vector and a signed distance from the origin.
    normal_d: Vec4,
}

/// Represents the side of a plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneSide {
    /// The front side of the plane.
    Front = 1,
    /// The back side of the plane.
    Back = -1,
    /// The plane itself.
    OnPlane = 0,
}

impl Plane {
    /// Creates a new [`Plane`] from a normal vector and a signed distance from the origin.
    #[inline]
    pub fn new(normal: impl Into<Vec3A>, d: f32) -> Self {
        let normal = normal.into();
        Self {
            normal_d: normal.extend(d),
        }
    }

    /// Creates a new [`Plane`] from a point on the plane and a normal vector.
    #[inline]
    pub fn from_point_and_normal(point: Vec3A, normal: Vec3A) -> Self {
        let d = -normal.dot(point);
        Self::new(normal, d)
    }

    /// Creates a new [`Plane`] from the coefficients of the plane equation `ax + by + cz + d = 0`.
    #[inline]
    pub fn from_coefficients(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self {
            normal_d: Vec4::new(a, b, c, d),
        }
    }

    /// Returns the normal vector of the plane.
    #[inline]
    pub fn normal(&self) -> Vec3A {
        Vec3A::from_vec4(self.normal_d)
    }

    /// Returns the signed distance from the origin to the plane.
    #[inline]
    pub fn d(&self) -> f32 {
        self.normal_d.w
    }

    /// Returns the normal vector and the signed distance from the origin as a [`Vec4`].
    #[inline]
    pub fn normal_d(&self) -> Vec4 {
        self.normal_d
    }

    /// Determines which side of the plane a triangle's normal points to.
    ///
    /// If any component of the normal is positive, the triangle is considered to be
    /// on the back side of the plane. Otherwise, it is considered to be on the front side.
    #[inline]
    pub fn cut_side(&self, p0: Vec3A, p1: Vec3A, p2: Vec3A) -> PlaneSide {
        let normal = triangle_normal(p0, p1, p2);
        // TODO: Is this faster than just manually checking products of individual components?
        if (normal * self.normal()).cmpgt(Vec3A::ZERO).any() {
            PlaneSide::Back
        } else {
            PlaneSide::Front
        }
    }

    /// Returns which [`PlaneSide`] a point lies on.
    #[inline]
    pub fn side(&self, p: Vec3A, epsilon: f32) -> PlaneSide {
        let res = self.normal().dot(p) + self.d();
        if res > epsilon {
            PlaneSide::Front
        } else if res < -epsilon {
            PlaneSide::Back
        } else {
            PlaneSide::OnPlane
        }
    }

    /// Computes the intersection point of a line segment with the plane.
    ///
    /// Returns `Some(point)` if the segment intersects the plane within the segment bounds,
    /// otherwise returns `None`.
    #[inline]
    pub fn intersect_segment(&self, p0: Vec3A, p1: Vec3A, tolerance: f32) -> Option<Vec3A> {
        let plane_normal = self.normal();
        let dir = p1 - p0;

        let denom = plane_normal.dot(dir);

        if denom.abs() < f32::EPSILON {
            // The segment is parallel to the plane.
            return None;
        }

        // Check if the intersection point is within the segment bounds.
        let t = -(plane_normal.dot(p0) + self.d()) / denom;
        if t >= -tolerance && t <= 1.0 + tolerance {
            Some(p0 + t * dir)
        } else {
            None
        }
    }
}

/// Computes the normal vector of a triangle defined by three points.
#[inline]
pub fn triangle_normal(p0: Vec3A, p1: Vec3A, p2: Vec3A) -> Vec3A {
    (p1 - p0).cross(p2 - p0).normalize()
}
/// Computes the area of a triangle defined by three points.
#[inline]
pub fn triangle_area(p0: Vec3A, p1: Vec3A, p2: Vec3A) -> f32 {
    0.5 * (p1 - p0).cross(p2 - p0).length()
}

/// Computes the signed volume of a tetrahedron with one point at the origin.
#[inline]
pub fn tetrahedron_signed_volume(p0: Vec3A, p1: Vec3A, p2: Vec3A) -> f32 {
    (p0.dot(p1.cross(p2))) / 6.0
}

/// Finds the point closest to the given `point` on a triangle ABC.
///
/// The Voronoi regions are A, B, C, AB, BC, AC, and ABC.
// TODO: Compare to CoACD's version.
#[inline]
pub fn project_on_triangle_3d(a: Vec3A, b: Vec3A, c: Vec3A, point: Vec3A) -> Vec3A {
    // Define A, B, and C as the vertices of the triangle ABC, Q as the input point,
    // and P as the projection of Q onto a line travelling through AB, BC, or AC,
    // or onto the triangle interior ABC.
    //
    // We have 3 vertex regions (A, B, C), 3 edge regions (AB, BC, AC), and 1 interior region (ABC).
    //
    //          |             |
    //          |  Region AB  |
    // Region B |             |  Region A
    //          B ----------- A
    //        /  \           /   \
    //      /     \ Region  /      \
    //   /         \  ABC  /         \
    //    Region BC \     / Region AC
    //               \   /
    //                 C
    //               /   \
    //            /         \
    //         /    Region C   \
    //
    // We'll find the Voronoi region of Q and compute P in one to three steps:
    //
    // 1. Check vertex regions using the barycentric coordinates of each segment.
    //    - If Q is in a vertex region, return P = Q.
    // 2. Check edge regions using the barycentric coordinates of both the segments and the triangle itself.
    //    - If Q is in an edge region, project Q onto the segment, and return P.
    // 3. Q must be in the interior region ABC.
    //    - In 2D, return P = Q.
    //    - In 3D, project Q onto the triangle face, and return P.
    //
    // First, we'll check each vertex region by determining the barycentric coordinates
    // of Q with respect to each line segment (see `project_on_segment`).
    //
    // - A: u_AC <= 0 and v_AB <= 0
    // - B: u_AB <= 0 and v_BC <= 0
    // - C: u_BC <= 0 and v_AC <= 0
    //
    // We don't need the actual uv values yet. Instead, we can compare dot products.

    let q = point;
    let ab = b - a;
    let ac = c - a;
    let aq = q - a;

    let ab_aq = ab.dot(aq);
    let ac_aq = ac.dot(aq);

    // u_AC <= 0 and  v_AB <= 0 (Voronoi region A)
    if ac_aq <= 0.0 && ab_aq <= 0.0 {
        return a;
    }

    let bq = q - b;
    let ab_bq = ab.dot(bq);
    let ac_bq = ac.dot(bq);

    // u_AB <= 0 and v_BC <= 0 (Voronoi region B)
    if ab_bq >= 0.0 && ac_bq <= ab_bq {
        return b;
    }

    let cq = q - c;
    let ab_cq = ab.dot(cq);
    let ac_cq = ac.dot(cq);

    // u_BC <= 0 and v_AC <= 0 (Voronoi region C)
    if ac_cq >= 0.0 && ab_cq <= ac_cq {
        return c;
    }

    // Next, we will check the edge regions AB, BC, and AC.
    //
    // We can use the barycentric coordinates (u, v, w) of Q to represent any point
    // in the triangle plane. For line segments, (u, v) represents fractional lengths,
    // but for triangles, (u, v, w) represents the signed fractional areas of three triangles inscribed in ABC.
    //
    // u_ABC = area(QBC) / area(ABC)
    // v_ABC = area(QAC) / area(ABC)
    // w_ABC = area(QAB) / area(ABC)
    //
    // We will use the signs of the barymetric coordinates to locate the correct Voronoi region.

    let bc = c - b;

    let normal = ab.cross(ac);

    // w_ABC * area < 0 and v_AB >= 0 and u_AB >= 0 (Voronoi region AB)
    let vc = normal.dot(ab.cross(aq));
    if vc <= 0.0 && ab_aq >= 0.0 && ab_bq <= 0.0 {
        let v = ab_aq / ab.length_squared();
        let p = a + ab * v;
        return p;
    }

    // v_ABC * area < 0 and u_AC >= 0 and v_AC >= 0 (Voronoi region AC)
    let vb = -normal.dot(ac.cross(cq));
    if vb <= 0.0 && ac_aq >= 0.0 && ac_cq <= 0.0 {
        let w = ac_aq / ac.length_squared();
        let p = a + ac * w;
        return p;
    }

    // u_ABC * area < 0 and v_BC >= 0 and u_BC >= 0 (Voronoi region BC)
    let va = normal.dot(bc.cross(bq));
    if va <= 0.0 && ac_bq - ab_bq >= 0.0 && ab_cq - ac_cq >= 0.0 {
        let w = bc.dot(bq) / bc.length_squared();
        let p = b + bc * w;
        return p;
    }

    // The projection is not in a vertex region or an edge region.
    // u, v, and w are all positive, and P is in the interior of the triangle ABC.

    if va + vb + vc != 0.0 {
        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;
        a + ab * v + ac * w
    } else {
        q
    }
}

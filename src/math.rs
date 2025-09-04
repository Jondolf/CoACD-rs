use glam::Vec3A;

// TODO: Use a `Vec4`?
/// A plane in 3D space, defined by the equation ax + by + cz + d = 0.
pub struct Plane {
    /// The `a` coefficient of the plane equation.
    pub a: f32,
    /// The `b` coefficient of the plane equation.
    pub b: f32,
    /// The `c` coefficient of the plane equation.
    pub c: f32,
    /// The `d` coefficient of the plane equation.
    pub d: f32,
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
    /// Creates a new plane from a normal vector and a distance from the origin.
    #[inline]
    pub fn new(normal: Vec3A, d: f32) -> Self {
        Self {
            a: normal.x,
            b: normal.y,
            c: normal.z,
            d,
        }
    }

    /// Determines which side of the plane a triangle's normal points to.
    ///
    /// If any component of the normal is positive, the triangle is considered to be
    /// on the back side of the plane. Otherwise, it is considered to be on the front side.
    #[inline]
    pub fn cut_side(&self, p0: Vec3A, p1: Vec3A, p2: Vec3A) -> PlaneSide {
        let normal = triangle_normal(p0, p1, p2);
        if normal.x * self.a > 0.0 || normal.y * self.b > 0.0 || normal.z * self.c > 0.0 {
            PlaneSide::Back
        } else {
            PlaneSide::Front
        }
    }

    /// Returns which [`PlaneSide`] a point lies on.
    #[inline]
    pub fn side(&self, p: Vec3A, epsilon: f32) -> PlaneSide {
        let res = self.a * p.x + self.b * p.y + self.c * p.z + self.d;
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
        let plane_normal = Vec3A::new(self.a, self.b, self.c);
        let dir = p1 - p0;

        let denom = plane_normal.dot(dir);

        if denom.abs() < f32::EPSILON {
            // The segment is parallel to the plane.
            return None;
        }

        // Check if the intersection point is within the segment bounds.
        let t = -(plane_normal.dot(p0) + self.d) / denom;
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

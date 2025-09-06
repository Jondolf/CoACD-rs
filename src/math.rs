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

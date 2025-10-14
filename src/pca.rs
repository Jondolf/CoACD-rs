//! [Principal Component Analysis (PCA)] on a set of 3D points.
//!
//! [Principal Component Analysis (PCA)]: https://en.wikipedia.org/wiki/Principal_component_analysis

use glam::{Mat3A, Vec3, Vec3A};
use glam_matrix_extras::{SymmetricEigen3, SymmetricMat3};
use obvhs::aabb::Aabb;
use thiserror::Error;

/// [Principal Component Analysis (PCA)] on a set of 3D points.
///
/// The result contains the eigenvectors and eigenvalues of the covariance matrix,
/// as well as the centroid of the input points.
///
/// The PCA can then be used to transform the points so that their centroid is at the origin,
/// and their principal axes are aligned with the X, Y, and Z axes. The transformation can
/// be applied using [`Pca::apply_transform`] and reverted using [`Pca::revert_transform`].
///
/// [Principal Component Analysis (PCA)]: https://en.wikipedia.org/wiki/Principal_component_analysis
///
/// # Example
///
/// ```
/// use approx::assert_relative_eq;
/// use coacd::pca::Pca;
/// use glam::Vec3A;
///
/// let points = vec![
///     Vec3A::new(1.0, 0.0, 0.0),
///     Vec3A::new(0.0, 1.0, 0.0),
///     Vec3A::new(0.0, 0.0, 1.0),
///     Vec3A::new(-1.0, -1.0, -1.0),
/// ];
///
/// // Perform PCA on the points.
/// let mut pca = Pca::from_points(&points).unwrap();
///
/// // Apply the PCA transformation.
/// let mut transformed_points = points.clone();
/// let transformed_aabb = pca.apply_transform(&mut transformed_points);
///
/// // Check that the centroid is at the origin after transformation.
/// let centroid = transformed_points.iter().sum::<Vec3A>() / transformed_points.len() as f32;
/// assert!(centroid.length() < 1e-6);
///
/// // Revert the transformation.
/// let reverted_aabb = pca.revert_transform(&mut transformed_points);
///
/// // Check that the points are the same after reverting the transformation.
/// for (original, reverted) in points.iter().zip(transformed_points.iter()) {
///     assert_relative_eq!(original, reverted, epsilon = 1e-6);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Pca {
    /// The eigenvectors of the covariance matrix, representing the principal axes.
    eigenvectors: Mat3A,
    /// The eigenvalues of the covariance matrix, representing the variance along each principal axis.
    eigenvalues: Vec3A,
    /// The centroid computed from the input points.
    centroid: Vec3A,
}

/// Errors that can occur during [`Pca`] computation.
#[derive(Error, Debug)]
pub enum PcaError {
    /// The input slice is empty.
    #[error("The input slice is empty.")]
    NoPoints,
    /// The number of significant eigenvalues is less than 3.
    #[error("The number of significant eigenvalues is less than 3. (rank == {rank})")]
    InsufficientRank {
        /// The number of significant eigenvalues.
        rank: usize,
        /// The PCA instance that was computed using the degenerate inputs.
        pca: Pca,
    },
}

impl Pca {
    /// Performs [Principal Component Analysis (PCA)] on a set of 3D points.
    ///
    /// Returns `None` if the input slice is empty. The PCA is always computed,
    /// even if the points are degenerate (for example, all points are collinear or coplanar).
    ///
    /// For a more robust version that checks for degenerate cases, see [`Pca::try_from_points`].
    ///
    /// [Principal Component Analysis (PCA)]: https://en.wikipedia.org/wiki/Principal_component_analysis
    pub fn from_points(points: &[Vec3A]) -> Option<Pca> {
        if points.is_empty() {
            return None;
        }

        // Compute the centroid.
        let centroid = points.iter().sum::<Vec3A>() / points.len() as f32;

        // Compute the covariance matrix of the points.
        let mut cov = SymmetricMat3::ZERO;
        for point in points {
            cov += SymmetricMat3::from_outer_product(Vec3::from(*point - centroid));
        }
        cov /= points.len() as f32;

        // Compute the Eigen decomposition of the covariance matrix.
        // The eigenvectors are the principal axes of the points.
        // Note: The C++ implementation of CoACD uses Jacobi iterations for diagonalization.
        //       `glam_matrix_extras` instead uses the robust non-iterative eigensolver
        //       described in "A Robust Eigensolver for 3 x 3 Symmetric Matrices" by David Eberly.
        //       https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf.
        let eigen = SymmetricEigen3::new(cov);

        Some(Pca {
            eigenvectors: Mat3A::from(eigen.eigenvectors),
            eigenvalues: Vec3A::from(eigen.eigenvalues),
            centroid,
        })
    }

    /// Performs [Principal Component Analysis (PCA)] on a set of 3D points.
    ///
    /// The given epsilon is used to determine the significance of eigenvalues.
    /// If the number of significant eigenvalues is less than 3, or if the input
    /// slice is empty, a [`PcaError`] is returned.
    ///
    /// [Principal Component Analysis (PCA)]: https://en.wikipedia.org/wiki/Principal_component_analysis
    ///
    /// # Errors
    ///
    /// - [`PcaError::NoPoints`]: The input slice is empty.
    /// - [`PcaError::InsufficientRank`]: The number of significant eigenvalues is less than 3.
    ///   - The `rank` field contains the number of significant eigenvalues.
    ///   - The `pca` field contains the PCA instance that was computed using the degenerate inputs.
    #[inline]
    pub fn try_from_points(points: &[Vec3A], epsilon: f32) -> Result<Pca, PcaError> {
        let pca = Pca::from_points(points).ok_or(PcaError::NoPoints)?;

        // Count the number of significant eigenvalues.
        let rank = pca
            .eigenvalues
            .to_array()
            .iter()
            .filter(|v| v.abs() > epsilon)
            .count();

        if rank < 3 {
            return Err(PcaError::InsufficientRank { rank, pca });
        }

        Ok(pca)
    }

    /// Creates a PCA instance from its eigenvectors, eigenvalues, and centroid.
    ///
    /// This can be used to create a PCA instance from precomputed values.
    /// No validation is performed on the input values.
    ///
    /// In most cases, you should use [`Pca::from_points`] instead.
    #[inline]
    pub fn from_raw(eigenvectors: Mat3A, eigenvalues: Vec3A, centroid: Vec3A) -> Pca {
        Pca {
            eigenvectors,
            eigenvalues,
            centroid,
        }
    }

    /// Returns the eigenvectors (principal axes) of the PCA.
    #[inline]
    pub fn eigenvectors(&self) -> Mat3A {
        self.eigenvectors
    }

    /// Returns the eigenvalues (variances along principal axes) of the PCA.
    #[inline]
    pub fn eigenvalues(&self) -> Vec3A {
        self.eigenvalues
    }

    /// Returns the centroid computed from the input points.
    #[inline]
    pub fn centroid(&self) -> Vec3A {
        self.centroid
    }

    /// Applies the PCA transformation to a set of points, modifying them in place.
    ///
    /// First, the points are translated so that their centroid is at the origin.
    /// Then, they are rotated so that their principal axes are aligned with the X, Y, and Z axes.
    ///
    /// Returns the axis-aligned bounding box (AABB) of the transformed set of points.
    #[inline]
    pub fn apply_transform(&mut self, points: &mut [Vec3A]) -> Aabb {
        let mut aabb = Aabb::INVALID;
        let rotation = self.eigenvectors.transpose();

        for point in &mut *points {
            let new_point = rotation * (*point - self.centroid);
            *point = new_point;
            aabb.extend(new_point);
        }

        aabb
    }

    /// Reverts the PCA transformation on a set of points, modifying them in place.
    ///
    /// Returns the axis-aligned bounding box (AABB) of the reverted set of points.
    #[inline]
    pub fn revert_transform(&mut self, points: &mut [Vec3A]) -> Aabb {
        let mut aabb = Aabb::INVALID;
        let inv_rotation = self.eigenvectors;

        for point in &mut *points {
            let new_point = inv_rotation * (*point) + self.centroid;
            *point = new_point;
            aabb.extend(new_point);
        }

        aabb
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_pca() {
        let points = vec![
            Vec3A::new(1.0, 0.0, 0.0),
            Vec3A::new(0.0, 1.0, 0.0),
            Vec3A::new(0.0, 0.0, 1.0),
            Vec3A::new(-1.0, -1.0, -1.0),
        ];

        // Perform PCA on the points.
        let mut pca = Pca::from_points(&points).unwrap();

        // Apply the PCA transformation.
        let mut transformed_points = points.clone();
        let transformed_aabb = pca.apply_transform(&mut transformed_points);
        let transformed_expected_aabb = Aabb::from_points(&transformed_points);

        // Check that the centroid is at the origin after transformation.
        let centroid = transformed_points.iter().sum::<Vec3A>() / transformed_points.len() as f32;
        assert!(centroid.length() < 1e-6);

        // Check that the points are aligned with the principal axes.
        // This is done by checking that the covariance matrix is diagonal.
        let cov = {
            let mut cov = SymmetricMat3::ZERO;
            for point in &transformed_points {
                cov += SymmetricMat3::from_outer_product(Vec3::from(*point - centroid));
            }
            cov / transformed_points.len() as f32
        };
        let off_diag_sum = cov.m01.abs() + cov.m02.abs() + cov.m12.abs();
        assert!(off_diag_sum < 1e-6);

        // Revert the transformation.
        let reverted_aabb = pca.revert_transform(&mut transformed_points);
        let reverted_expected_aabb = Aabb::from_points(&points);

        // Check that the points are the same after reverting the transformation.
        for (original, reverted) in points.iter().zip(transformed_points.iter()) {
            assert_relative_eq!(original, reverted, epsilon = 1e-6);
        }

        // Check that the AABBs are correct.
        assert_relative_eq!(
            transformed_aabb.min,
            transformed_expected_aabb.min,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            transformed_aabb.max,
            transformed_expected_aabb.max,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            reverted_aabb.min,
            reverted_expected_aabb.min,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            reverted_aabb.max,
            reverted_expected_aabb.max,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_pca_empty() {
        let points: Vec<Vec3A> = Vec::new();

        // Perform PCA on the points.
        let pca = Pca::from_points(&points);

        // Check that None is returned.
        assert!(pca.is_none());
    }

    #[test]
    fn test_pca_coincident() {
        let points = vec![
            Vec3A::new(1.0, 1.0, 1.0),
            Vec3A::new(1.0, 1.0, 1.0),
            Vec3A::new(1.0, 1.0, 1.0),
        ];

        // Perform PCA on the points.
        let pca = Pca::try_from_points(&points, 1e-6);

        // Check that an error is returned.
        assert!(matches!(
            pca,
            Err(PcaError::InsufficientRank { rank: 0, .. })
        ));
    }

    #[test]
    fn test_pca_collinear() {
        let points = vec![
            Vec3A::new(1.0, 0.0, 0.0),
            Vec3A::new(2.0, 0.0, 0.0),
            Vec3A::new(3.0, 0.0, 0.0),
        ];

        // Perform PCA on the points.
        let pca = Pca::try_from_points(&points, 1e-6);

        // Check that an error is returned.
        assert!(matches!(
            pca,
            Err(PcaError::InsufficientRank { rank: 1, .. })
        ));
    }

    #[test]
    fn test_pca_coplanar() {
        let points = vec![
            Vec3A::new(1.0, 0.0, 0.0),
            Vec3A::new(0.0, 1.0, 0.0),
            Vec3A::new(-1.0, 0.0, 0.0),
            Vec3A::new(0.0, -1.0, 0.0),
        ];

        // Perform PCA on the points.
        let pca = Pca::try_from_points(&points, 1e-6);

        // Check that an error is returned.
        assert!(matches!(
            pca,
            Err(PcaError::InsufficientRank { rank: 2, .. })
        ));
    }
}

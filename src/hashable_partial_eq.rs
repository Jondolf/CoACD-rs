// Adapted from Parry:
// <https://github.com/dimforge/parry/blob/86404116432cc580eae8a8a47b44ffd146950a2c/src/utils/hashable_partial_eq.rs>

use core::hash::{Hash, Hasher};

use glam::Vec3A;

/// A structure that implements [`Eq`] and is hashable even if the wrapped data
/// only implements [`PartialEq`].
#[derive(PartialEq, Clone, Debug)]
pub struct HashablePartialEq<T> {
    value: T,
}

impl<T> HashablePartialEq<T> {
    /// Creates a new `HashablePartialEq`. Please make sure that you really
    /// want to transform the wrapped object's partial equality to an equivalence relation.
    pub fn new(value: T) -> HashablePartialEq<T> {
        HashablePartialEq { value }
    }
}

impl<T: Default> Default for HashablePartialEq<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
        }
    }
}

impl<T: PartialEq> Eq for HashablePartialEq<T> {}

impl<T: AsBytes> Hash for HashablePartialEq<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(self.value.as_bytes())
    }
}

/// Trait that transforms thing to a slice of u8.
pub trait AsBytes {
    /// Converts `self` to a slice of bytes.
    fn as_bytes(&self) -> &[u8];
}

impl AsBytes for Vec3A {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                (self as *const Vec3A) as *const u8,
                core::mem::size_of::<Vec3A>(),
            )
        }
    }
}

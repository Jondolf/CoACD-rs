use crate::collections::FixedHasher;
use core::ops::{Deref, DerefMut};

/// A new-type for [`HashSet`](hashbrown::HashSet) with [`FixedHasher`] as the hashing provider.
/// Can be trivially converted to and from a [hashbrown] [`HashSet`](hashbrown::HashSet) using [`From`].
///
/// This hasher is not DoS-resistant and is not suitable for cryptographic purposes,
/// but it has high performance and determinism.
///
/// A new-type is used instead of a type alias due to critical methods like [`new`](hashbrown::HashSet::new)
/// being incompatible with a non-default hasher.
#[derive(Clone, Debug, Default)]
pub struct HashSet<K>(hashbrown::HashSet<K, FixedHasher>);

impl<K> Deref for HashSet<K> {
    type Target = hashbrown::HashSet<K, FixedHasher>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<K> DerefMut for HashSet<K> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<K, T> FromIterator<T> for HashSet<K>
where
    hashbrown::HashSet<K, FixedHasher>: FromIterator<T>,
{
    #[inline]
    fn from_iter<U: IntoIterator<Item = T>>(iter: U) -> Self {
        Self(FromIterator::from_iter(iter))
    }
}

impl<K> HashSet<K> {
    /// Creates an empty [`HashSet`] with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(hashbrown::HashSet::with_capacity_and_hasher(
            capacity,
            FixedHasher,
        ))
    }
}

//! Compiled kernels for Hafnium.
//!
//! This crate holds the numerically expensive, index-heavy code: Gaussian
//! integrals, DFT grids and basis-function evaluation on grids. It knows
//! nothing about Python; the `hafnium-py` crate wraps it. Matrix contractions
//! do not belong here, they run on the Python array backend.
//!
//! Integrals follow the McMurchie–Davidson scheme (ADR-0008): every shell
//! pair is expanded in Hermite Gaussians, and Coulomb-type integrals come
//! from Hermite integrals of the Boys function.

pub mod integrals;
pub mod special;

/// Version of this crate, as given in `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Errors raised by kernel functions. Never panics across the FFI boundary.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    /// Two arrays that must have equal length do not.
    #[error("length mismatch: {left} vs {right}")]
    LengthMismatch {
        /// Length of the first array.
        left: usize,
        /// Length of the second array.
        right: usize,
    },
    /// A shell table is inconsistent.
    #[error("invalid shell table: {0}")]
    InvalidShellTable(String),
    /// A kernel argument is out of its domain.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

/// Dot product of two slices of equal length.
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if the slices differ in length.
pub fn dot(a: &[f64], b: &[f64]) -> Result<f64, Error> {
    if a.len() != b.len() {
        return Err(Error::LengthMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

/// Convert a small count or index to `f64` exactly.
#[inline]
#[allow(
    clippy::cast_precision_loss,
    reason = "counts and indices stay far below 2^53"
)]
pub(crate) fn to_f64(n: usize) -> f64 {
    n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_of_matching_slices() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), Ok(32.0));
    }

    #[test]
    fn dot_of_empty_slices_is_zero() {
        assert_eq!(dot(&[], &[]), Ok(0.0));
    }

    #[test]
    fn dot_rejects_length_mismatch() {
        assert_eq!(
            dot(&[1.0], &[1.0, 2.0]),
            Err(Error::LengthMismatch { left: 1, right: 2 })
        );
    }
}

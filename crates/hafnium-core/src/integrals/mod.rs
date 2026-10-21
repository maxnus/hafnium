//! Gaussian integrals in the McMurchie–Davidson scheme (ADR-0008).
//!
//! Shells enter as a [`ShellTable`] in structure-of-arrays form with one
//! entry per contraction. Every kernel takes a bra table, a ket table and a
//! list of translation vectors applied to the ket, and returns the integral
//! matrix for each translation, so that lattice sums of periodic systems
//! (ADR-0007) and molecular integrals share one code path.

mod cartesian;
mod coulomb;
mod hermite;
mod one_electron;

pub use cartesian::{cartesian_components, n_cartesian, spherical_transformation};
pub use one_electron::{kinetic, nuclear_attraction, overlap};

use crate::Error;

/// Highest angular momentum the spherical transformation supports; the
/// exact integer factorials it uses fit in 128 bits up to this value.
pub const MAX_ANGULAR_MOMENTUM: usize = 16;

/// Contracted shells in structure-of-arrays form.
///
/// Each shell is one contraction: general contractions are split by the
/// caller. Coefficients multiply bare primitives `x^i y^j z^k exp(-α r²)`
/// with the normalization of ADR-0006 already applied, so a contracted
/// spherical function has unit norm.
#[derive(Debug, Clone, PartialEq)]
pub struct ShellTable {
    angular_momenta: Vec<usize>,
    centers: Vec<[f64; 3]>,
    primitive_offsets: Vec<usize>,
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
}

/// One shell of a [`ShellTable`], borrowed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ShellRef<'a> {
    pub angular_momentum: usize,
    pub center: [f64; 3],
    pub exponents: &'a [f64],
    pub coefficients: &'a [f64],
}

impl ShellTable {
    /// Build and validate a table.
    ///
    /// `primitive_offsets` has one more entry than there are shells; shell
    /// `i` owns the primitives `offsets[i]..offsets[i + 1]`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidShellTable`] if the lengths disagree, a shell
    /// has no primitives, an exponent is not positive, or an angular
    /// momentum exceeds [`MAX_ANGULAR_MOMENTUM`].
    pub fn new(
        angular_momenta: Vec<usize>,
        centers: Vec<[f64; 3]>,
        primitive_offsets: Vec<usize>,
        exponents: Vec<f64>,
        coefficients: Vec<f64>,
    ) -> Result<Self, Error> {
        let n_shells = angular_momenta.len();
        let invalid = |message: &str| Err(Error::InvalidShellTable(message.to_string()));
        if centers.len() != n_shells {
            return invalid("one centre per shell");
        }
        if primitive_offsets.len() != n_shells + 1 || primitive_offsets.first() != Some(&0) {
            return invalid(
                "primitive offsets start at zero and have one entry per shell plus one",
            );
        }
        if primitive_offsets.windows(2).any(|pair| pair[1] <= pair[0]) {
            return invalid("every shell needs at least one primitive");
        }
        if primitive_offsets.last() != Some(&exponents.len())
            || exponents.len() != coefficients.len()
        {
            return invalid(
                "exponents and coefficients cover exactly the primitives of the shells",
            );
        }
        if exponents.iter().any(|&a| a.is_nan() || a <= 0.0) {
            return invalid("exponents are positive");
        }
        if !coefficients.iter().all(|c| c.is_finite()) {
            return invalid("coefficients are finite");
        }
        if !centers
            .iter()
            .all(|center| center.iter().all(|x| x.is_finite()))
        {
            return invalid("centres are finite");
        }
        if angular_momenta.iter().any(|&l| l > MAX_ANGULAR_MOMENTUM) {
            return invalid("angular momentum exceeds the supported maximum");
        }
        Ok(Self {
            angular_momenta,
            centers,
            primitive_offsets,
            exponents,
            coefficients,
        })
    }

    /// Number of shells (contractions).
    #[must_use]
    pub fn n_shells(&self) -> usize {
        self.angular_momenta.len()
    }

    /// Number of spherical basis functions, `2l + 1` per shell.
    #[must_use]
    pub fn n_functions(&self) -> usize {
        self.angular_momenta.iter().map(|l| 2 * l + 1).sum()
    }

    /// Highest angular momentum in the table, or zero for an empty table.
    #[must_use]
    pub fn max_angular_momentum(&self) -> usize {
        self.angular_momenta.iter().copied().max().unwrap_or(0)
    }

    /// Index of the first basis function of each shell.
    #[must_use]
    pub fn function_offsets(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.n_shells());
        let mut total = 0;
        for l in &self.angular_momenta {
            offsets.push(total);
            total += 2 * l + 1;
        }
        offsets
    }

    pub(crate) fn shell(&self, index: usize) -> ShellRef<'_> {
        let range = self.primitive_offsets[index]..self.primitive_offsets[index + 1];
        ShellRef {
            angular_momentum: self.angular_momenta[index],
            center: self.centers[index],
            exponents: &self.exponents[range.clone()],
            coefficients: &self.coefficients[range],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> ShellTable {
        ShellTable::new(
            vec![0, 1],
            vec![[0.0; 3], [1.0, 0.0, 0.0]],
            vec![0, 2, 3],
            vec![1.0, 0.5, 0.3],
            vec![0.7, 0.4, 1.0],
        )
        .expect("valid table")
    }

    #[test]
    fn counts_functions_and_offsets() {
        let table = table();
        assert_eq!(table.n_shells(), 2);
        assert_eq!(table.n_functions(), 4);
        assert_eq!(table.max_angular_momentum(), 1);
        assert_eq!(table.function_offsets(), vec![0, 1]);
        assert_eq!(table.shell(1).exponents, &[0.3]);
    }

    #[test]
    fn rejects_inconsistent_tables() {
        let bad = ShellTable::new(vec![0], vec![[0.0; 3]], vec![0, 0], vec![], vec![]);
        assert!(matches!(bad, Err(Error::InvalidShellTable(_))));
        let bad = ShellTable::new(vec![0], vec![[0.0; 3]], vec![0, 1], vec![-1.0], vec![1.0]);
        assert!(matches!(bad, Err(Error::InvalidShellTable(_))));
        let bad = ShellTable::new(vec![17], vec![[0.0; 3]], vec![0, 1], vec![1.0], vec![1.0]);
        assert!(matches!(bad, Err(Error::InvalidShellTable(_))));
        let bad = ShellTable::new(
            vec![0],
            vec![[f64::NAN, 0.0, 0.0]],
            vec![0, 1],
            vec![1.0],
            vec![1.0],
        );
        assert!(matches!(bad, Err(Error::InvalidShellTable(_))));
        let bad = ShellTable::new(
            vec![0],
            vec![[0.0; 3]],
            vec![0, 1],
            vec![1.0],
            vec![f64::INFINITY],
        );
        assert!(matches!(bad, Err(Error::InvalidShellTable(_))));
    }
}

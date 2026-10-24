//! Gaussian integrals in the McMurchie–Davidson scheme (ADR-0008).
//!
//! Shells enter as a [`ShellTable`] in structure-of-arrays form with one
//! entry per shell, general contractions included. Every kernel takes a bra
//! table, a ket table and a list of translation vectors applied to the ket,
//! and returns the integral matrix for each translation, so that lattice
//! sums of periodic systems (ADR-0007) and molecular integrals share one
//! code path.

mod cartesian;
mod coulomb;
mod fourier;
mod hermite;
mod one_electron;
mod pairs;
#[cfg(test)]
pub(crate) mod test_tables;
mod three_center_bloch;
mod two_electron;

pub use cartesian::{cartesian_components, n_cartesian, spherical_transformation, transform_index};
pub use coulomb::CoulombKernel;
pub use fourier::{fourier_transform, pair_fourier_transform};
pub use one_electron::{kinetic, nuclear_attraction, overlap};
pub use three_center_bloch::{BlochSum, Transfers, coulomb_three_center_bloch};
pub use two_electron::{
    coulomb_four_center, coulomb_three_center, coulomb_two_center, schwarz_bounds,
};

use crate::Error;

/// Set the size of the global thread pool used by the kernels.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if the pool has already been built,
/// which happens on the first parallel kernel call or an earlier
/// configuration.
pub fn configure_threads(n_threads: usize) -> Result<(), Error> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .map_err(|error| Error::InvalidArgument(format!("thread pool: {error}")))
}

/// Highest angular momentum the spherical transformation supports; the
/// exact integer factorials it uses fit in 128 bits up to this value.
pub const MAX_ANGULAR_MOMENTUM: usize = 16;

/// Contracted shells in structure-of-arrays form.
///
/// A shell holds one or more contractions of the same primitives (a
/// general contraction, ADR-0011); its functions are ordered
/// contraction-major, then `m` (ADR-0006). Coefficients multiply bare
/// primitives `x^i y^j z^k exp(-α r²)` with the normalization of ADR-0006
/// already applied, so a contracted spherical function has unit norm, and
/// are stored shell by shell as `[contraction][primitive]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShellTable {
    angular_momenta: Vec<usize>,
    centers: Vec<[f64; 3]>,
    primitive_offsets: Vec<usize>,
    exponents: Vec<f64>,
    n_contractions: Vec<usize>,
    coefficient_offsets: Vec<usize>,
    coefficients: Vec<f64>,
}

/// One shell of a [`ShellTable`], borrowed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ShellRef<'a> {
    pub angular_momentum: usize,
    pub center: [f64; 3],
    pub exponents: &'a [f64],
    pub n_contractions: usize,
    /// `[contraction][primitive]`, row-major.
    pub coefficients: &'a [f64],
}

impl ShellRef<'_> {
    /// Coefficient of `primitive` in `contraction`.
    #[inline]
    pub(crate) fn coefficient(&self, contraction: usize, primitive: usize) -> f64 {
        self.coefficients[contraction * self.exponents.len() + primitive]
    }

    /// Number of spherical functions, `(2l + 1)` per contraction.
    #[inline]
    pub(crate) fn n_functions(&self) -> usize {
        self.n_contractions * (2 * self.angular_momentum + 1)
    }
}

impl ShellTable {
    /// Build and validate a table.
    ///
    /// `primitive_offsets` has one more entry than there are shells; shell
    /// `i` owns the primitives `offsets[i]..offsets[i + 1]` and the next
    /// `n_contractions[i]` rows of `coefficients`, each as long as its
    /// primitive list.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidShellTable`] if the lengths disagree, a shell
    /// has no primitives or no contractions, an exponent is not positive,
    /// or an angular momentum exceeds [`MAX_ANGULAR_MOMENTUM`].
    pub fn new(
        angular_momenta: Vec<usize>,
        centers: Vec<[f64; 3]>,
        primitive_offsets: Vec<usize>,
        exponents: Vec<f64>,
        coefficients: Vec<f64>,
        n_contractions: Vec<usize>,
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
        if primitive_offsets.last() != Some(&exponents.len()) {
            return invalid("exponents cover exactly the primitives of the shells");
        }
        if n_contractions.len() != n_shells || n_contractions.contains(&0) {
            return invalid("every shell needs at least one contraction");
        }
        let mut coefficient_offsets = Vec::with_capacity(n_shells + 1);
        let mut total = 0;
        for (shell, &count) in n_contractions.iter().enumerate() {
            coefficient_offsets.push(total);
            total += count * (primitive_offsets[shell + 1] - primitive_offsets[shell]);
        }
        coefficient_offsets.push(total);
        if coefficients.len() != total {
            return invalid("coefficients hold one row per contraction, one entry per primitive");
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
            n_contractions,
            coefficient_offsets,
            coefficients,
        })
    }

    /// Number of shells.
    #[must_use]
    pub fn n_shells(&self) -> usize {
        self.angular_momenta.len()
    }

    /// Number of spherical basis functions, `2l + 1` per contraction.
    #[must_use]
    pub fn n_functions(&self) -> usize {
        (0..self.n_shells())
            .map(|i| self.shell(i).n_functions())
            .sum()
    }

    /// Highest angular momentum in the table, or zero for an empty table.
    #[must_use]
    pub fn max_angular_momentum(&self) -> usize {
        self.angular_momenta.iter().copied().max().unwrap_or(0)
    }

    /// Largest number of contractions of any shell, or zero for an empty table.
    #[must_use]
    pub fn max_contractions(&self) -> usize {
        self.n_contractions.iter().copied().max().unwrap_or(0)
    }

    /// Index of the first basis function of each shell.
    #[must_use]
    pub fn function_offsets(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.n_shells());
        let mut total = 0;
        for index in 0..self.n_shells() {
            offsets.push(total);
            total += self.shell(index).n_functions();
        }
        offsets
    }

    pub(crate) fn shell(&self, index: usize) -> ShellRef<'_> {
        let primitives = self.primitive_offsets[index]..self.primitive_offsets[index + 1];
        let coefficients = self.coefficient_offsets[index]..self.coefficient_offsets[index + 1];
        ShellRef {
            angular_momentum: self.angular_momenta[index],
            center: self.centers[index],
            exponents: &self.exponents[primitives],
            n_contractions: self.n_contractions[index],
            coefficients: &self.coefficients[coefficients],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> ShellTable {
        // An s shell with two contractions of two primitives, and a p shell.
        ShellTable::new(
            vec![0, 1],
            vec![[0.0; 3], [1.0, 0.0, 0.0]],
            vec![0, 2, 3],
            vec![1.0, 0.5, 0.3],
            vec![0.7, 0.4, 0.2, 0.9, 1.0],
            vec![2, 1],
        )
        .expect("valid table")
    }

    #[test]
    fn counts_functions_and_offsets() {
        let table = table();
        assert_eq!(table.n_shells(), 2);
        assert_eq!(table.n_functions(), 5);
        assert_eq!(table.max_angular_momentum(), 1);
        assert_eq!(table.function_offsets(), vec![0, 2]);
        let s = table.shell(0);
        assert_eq!(s.n_contractions, 2);
        assert_eq!(s.coefficient(1, 0), 0.2);
        assert_eq!(table.shell(1).exponents, &[0.3]);
        assert_eq!(table.shell(1).coefficients, &[1.0]);
    }

    fn single(coefficients: Vec<f64>, contractions: Vec<usize>) -> Result<ShellTable, Error> {
        ShellTable::new(
            vec![0],
            vec![[0.0; 3]],
            vec![0, 1],
            vec![1.0],
            coefficients,
            contractions,
        )
    }

    #[test]
    fn rejects_inconsistent_tables() {
        let invalid = |bad: Result<ShellTable, Error>| {
            assert!(matches!(bad, Err(Error::InvalidShellTable(_))));
        };
        invalid(ShellTable::new(
            vec![0],
            vec![[0.0; 3]],
            vec![0, 0],
            vec![],
            vec![],
            vec![1],
        ));
        invalid(ShellTable::new(
            vec![0],
            vec![[0.0; 3]],
            vec![0, 1],
            vec![-1.0],
            vec![1.0],
            vec![1],
        ));
        invalid(ShellTable::new(
            vec![17],
            vec![[0.0; 3]],
            vec![0, 1],
            vec![1.0],
            vec![1.0],
            vec![1],
        ));
        invalid(ShellTable::new(
            vec![0],
            vec![[f64::NAN, 0.0, 0.0]],
            vec![0, 1],
            vec![1.0],
            vec![1.0],
            vec![1],
        ));
        invalid(single(vec![f64::INFINITY], vec![1]));
        invalid(single(vec![1.0], vec![0]));
        invalid(single(vec![1.0], vec![2]));
        invalid(single(vec![1.0, 2.0], vec![1]));
    }
}

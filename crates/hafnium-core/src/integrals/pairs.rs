//! Primitive pairs of two shells and the Hermite expansion of their
//! product, shared by every kernel.

use super::ShellRef;
use super::hermite::ExpansionCoefficients;
use crate::to_f64;

/// A pair of primitives with their Hermite expansion along each axis.
pub(crate) struct PrimitivePair {
    /// Total exponent `p = a + b`.
    pub p: f64,
    /// Exponent of the ket primitive, needed by derivative formulas.
    pub exponent_b: f64,
    /// Product centre `P = (aA + bB) / p`.
    pub center: [f64; 3],
    /// Product of the contraction coefficients.
    pub coefficient: f64,
    /// `l_a + l_b`, the highest Hermite order of the product.
    pub order: usize,
    /// Expansion coefficients along x, y and z.
    pub hermite: [ExpansionCoefficients; 3],
}

impl PrimitivePair {
    /// Product of the expansion coefficients along the three axes.
    #[inline]
    pub(crate) fn product(&self, a: &[usize; 3], b: &[usize; 3], t: [usize; 3]) -> f64 {
        self.hermite[0].get(a[0], b[0], t[0])
            * self.hermite[1].get(a[1], b[1], t[1])
            * self.hermite[2].get(a[2], b[2], t[2])
    }

    /// `⟨x^i | d²/dx² | x^j⟩ / sqrt(π/p)` along one axis (Helgaker 9.3.33).
    #[inline]
    pub(crate) fn second_derivative(&self, axis: usize, i: usize, j: usize) -> f64 {
        let e = &self.hermite[axis];
        let b = self.exponent_b;
        let mut value =
            4.0 * b * b * e.get(i, j + 2, 0) - 2.0 * b * to_f64(2 * j + 1) * e.get(i, j, 0);
        if j >= 2 {
            value += to_f64(j * (j - 1)) * e.get(i, j - 2, 0);
        }
        value
    }
}

/// All primitive pairs of two shells, with the ket centred at `b_center`.
/// `extra_b` raises the ket order of the expansion, as kinetic integrals
/// need.
pub(crate) fn primitive_pairs(
    a: &ShellRef<'_>,
    b: &ShellRef<'_>,
    b_center: [f64; 3],
    extra_b: usize,
) -> Vec<PrimitivePair> {
    let mut pairs = Vec::with_capacity(a.exponents.len() * b.exponents.len());
    for (&exponent_a, &coefficient_a) in a.exponents.iter().zip(a.coefficients) {
        for (&exponent_b, &coefficient_b) in b.exponents.iter().zip(b.coefficients) {
            let p = exponent_a + exponent_b;
            let mu = exponent_a * exponent_b / p;
            let center: [f64; 3] = std::array::from_fn(|axis| {
                (exponent_a * a.center[axis] + exponent_b * b_center[axis]) / p
            });
            let hermite = std::array::from_fn(|axis| {
                ExpansionCoefficients::new(
                    a.angular_momentum,
                    b.angular_momentum + extra_b,
                    p,
                    mu,
                    a.center[axis] - b_center[axis],
                    center[axis] - a.center[axis],
                    center[axis] - b_center[axis],
                )
            });
            pairs.push(PrimitivePair {
                p,
                exponent_b,
                center,
                coefficient: coefficient_a * coefficient_b,
                order: a.angular_momentum + b.angular_momentum,
                hermite,
            });
        }
    }
    pairs
}

/// The primitives of one shell at `center` as pairs with a unit partner,
/// so that a single Gaussian (an auxiliary function) can take the place of
/// a product distribution in the Coulomb kernels.
pub(crate) fn single_shell_pairs(a: &ShellRef<'_>, center: [f64; 3]) -> Vec<PrimitivePair> {
    let unit = ShellRef {
        angular_momentum: 0,
        center,
        exponents: &[0.0],
        coefficients: &[1.0],
    };
    let shifted = ShellRef { center, ..*a };
    primitive_pairs(&shifted, &unit, center, 0)
}

//! Primitive pairs of two shells and the Hermite expansion of their
//! product, shared by every kernel.
//!
//! A shell may hold several contractions of the same primitives (a general
//! contraction, ADR-0006). The expansion of a primitive pair does not
//! depend on the contraction, so it is built once and every kernel
//! accumulates the primitive values into all contraction combinations with
//! [`PrimitivePairs::accumulate`].

use std::ops::{AddAssign, Mul};

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

/// All primitive pairs of two shells with the contraction coefficients of
/// every pair, `coefficients[pair][k_a][k_b]`.
pub(crate) struct PrimitivePairs {
    pub pairs: Vec<PrimitivePair>,
    /// Number of contractions of the two shells.
    pub contractions: [usize; 2],
    coefficients: Vec<f64>,
}

impl PrimitivePairs {
    /// Product of the coefficients of contraction `k_a` of the bra and
    /// `k_b` of the ket in primitive pair `pair`.
    #[inline]
    pub(crate) fn coefficient(&self, pair: usize, k_a: usize, k_b: usize) -> f64 {
        self.coefficients[(pair * self.contractions[0] + k_a) * self.contractions[1] + k_b]
    }

    /// Add `values[i_a][i_b]`, the contribution of primitive pair `pair` to
    /// `n_a × n_b` Cartesian components, to `block` laid out
    /// `[k_a][i_a][k_b][i_b]`, scaled by every contraction coefficient.
    pub(crate) fn accumulate<T>(
        &self,
        pair: usize,
        values: &[T],
        n_a: usize,
        n_b: usize,
        block: &mut [T],
    ) where
        T: Copy + AddAssign + Mul<f64, Output = T>,
    {
        let [left_count, right_count] = self.contractions;
        for k_a in 0..left_count {
            for k_b in 0..right_count {
                let coefficient = self.coefficient(pair, k_a, k_b);
                for i_a in 0..n_a {
                    let target = ((k_a * n_a + i_a) * right_count + k_b) * n_b;
                    let source = i_a * n_b;
                    for i_b in 0..n_b {
                        block[target + i_b] += values[source + i_b] * coefficient;
                    }
                }
            }
        }
    }

    /// Number of contracted function pairs, `k_a k_b`.
    pub(crate) fn n_contracted(&self) -> usize {
        self.contractions[0] * self.contractions[1]
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
) -> PrimitivePairs {
    let (n_a, n_b) = (a.exponents.len(), b.exponents.len());
    let contractions = [a.n_contractions, b.n_contractions];
    let mut pairs = Vec::with_capacity(n_a * n_b);
    let mut coefficients = Vec::with_capacity(n_a * n_b * contractions[0] * contractions[1]);
    for (i_a, &exponent_a) in a.exponents.iter().enumerate() {
        for (i_b, &exponent_b) in b.exponents.iter().enumerate() {
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
                order: a.angular_momentum + b.angular_momentum,
                hermite,
            });
            for k_a in 0..contractions[0] {
                for k_b in 0..contractions[1] {
                    coefficients.push(a.coefficient(k_a, i_a) * b.coefficient(k_b, i_b));
                }
            }
        }
    }
    PrimitivePairs {
        pairs,
        contractions,
        coefficients,
    }
}

/// The primitives of one shell at `center` as pairs with a unit partner,
/// so that a single Gaussian (an auxiliary function) can take the place of
/// a product distribution in the Coulomb kernels.
pub(crate) fn single_shell_pairs(a: &ShellRef<'_>, center: [f64; 3]) -> PrimitivePairs {
    let unit = ShellRef {
        angular_momentum: 0,
        center,
        exponents: &[0.0],
        n_contractions: 1,
        coefficients: &[1.0],
    };
    let shifted = ShellRef { center, ..*a };
    primitive_pairs(&shifted, &unit, center, 0)
}

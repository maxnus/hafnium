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
    /// Exponent of the bra primitive, needed by the centre derivatives.
    pub exponent_a: f64,
    /// Exponent of the ket primitive, needed by derivative formulas.
    pub exponent_b: f64,
    /// Product centre `P = (aA + bB) / p`.
    pub center: [f64; 3],
    /// The highest Hermite order the expansion holds: `l_a + l_b` plus the
    /// extra orders the kernel asked for.
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

    /// The largest magnitude among the contraction coefficients of
    /// primitive pair `pair`, which bounds its weight in any contraction.
    pub(crate) fn coefficient_max(&self, pair: usize) -> f64 {
        let n = self.contractions[0] * self.contractions[1];
        self.coefficients[pair * n..(pair + 1) * n]
            .iter()
            .fold(0.0_f64, |acc, c| acc.max(c.abs()))
    }
}

/// Screening bounds on the integrals of a primitive pair.
///
/// Every kernel writes the product of two Cartesian primitives as
/// `Σ_tuv E_tuv Λ_tuv` and the integral of a Hermite Gaussian `Λ_tuv`
/// against a kernel is bounded by a weight `w_t w_u w_v` that depends on
/// the orders alone, so `max_ab Π_axis Σ_t |E^axis_t(a, b)| w_t` bounds
/// every Cartesian component of the pair's integrals. The weights of the
/// Fourier and the Coulomb kernels come from the Hermite polynomials:
/// `d^t/dx^t exp(−s² x²) = (−s)^t H_t(s x) exp(−s² x²)` with the physicists'
/// polynomials `H_t`.
impl PrimitivePair {
    /// `max_ab Π_axis Σ_{t ≤ i + j} |E^axis_t(i, j)| weights[t]` over the
    /// Cartesian components `a` of the bra and `b` of the ket.
    pub(crate) fn envelope(
        &self,
        components_a: &[[usize; 3]],
        components_b: &[[usize; 3]],
        weights: &[f64],
    ) -> f64 {
        let mut largest = 0.0_f64;
        for ca in components_a {
            for cb in components_b {
                let mut product = 1.0;
                for axis in 0..3 {
                    let (ia, ib) = (ca[axis], cb[axis]);
                    let expansion = &self.hermite[axis];
                    product *= (0..=ia + ib)
                        .map(|order| expansion.get(ia, ib, order).abs() * weights[order])
                        .sum::<f64>();
                }
                largest = largest.max(product);
            }
        }
        largest
    }
}

/// `ε` in the envelope `|H_t(y)| exp(−ε y²) ≤ h_t(ε)` of the Hermite
/// polynomials, which leaves `exp(−(1 − ε) y²)` of the Gaussian for the
/// distance screening of the Coulomb kernels. A smaller value keeps more of
/// the decay and makes the constants `h_t` larger; both enter the screening
/// radius logarithmically.
pub(crate) const ENVELOPE_EPSILON: f64 = 0.1;

/// `h_t(ε) ≥ max_y |H_t(y)| exp(−ε y²)` for `t ≤ max_order`, from the
/// coefficients of `H_t(y) = Σ_k (−1)^k t! / (k! (t − 2k)!) (2y)^{t − 2k}` and
/// `|y|^m exp(−ε y²) ≤ (m / 2eε)^{m/2}`.
pub(crate) fn hermite_envelopes(max_order: usize, epsilon: f64) -> Vec<f64> {
    (0..=max_order)
        .map(|t| {
            (0..=t / 2)
                .map(|k| {
                    let m = t - 2 * k;
                    let combinatorial = factorial(t) / (factorial(k) * factorial(m));
                    let moment = if m == 0 {
                        1.0
                    } else {
                        (to_f64(m) / (2.0 * std::f64::consts::E * epsilon)).powf(0.5 * to_f64(m))
                    };
                    combinatorial
                        * 2.0_f64.powi(i32::try_from(m).expect("order fits in i32"))
                        * moment
                })
                .sum()
        })
        .collect()
}

/// `∫ |d^t/dx^t exp(−p x²)| dx ≤ p^{(t − 1) / 2} sqrt(π 2^t t!)` for
/// `t ≤ max_order`, by the Cauchy–Schwarz inequality on
/// `∫ |H_t(y)| exp(−y²) dy` with `∫ H_t² exp(−y²) dy = sqrt(π) 2^t t!`. The
/// weights returned are `p^{t/2} sqrt(2^t t!)`; the caller supplies
/// `(π / p)^{3/2}`.
pub(crate) fn transform_weights(max_order: usize, p: f64) -> Vec<f64> {
    (0..=max_order)
        .map(|t| {
            p.powf(0.5 * to_f64(t))
                * (2.0_f64.powi(i32::try_from(t).expect("order fits in i32")) * factorial(t)).sqrt()
        })
        .collect()
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(to_f64).product()
}

/// All primitive pairs of two shells, with the ket centred at `b_center`.
/// `extra` raises the bra and the ket order of the expansion, as the
/// centre derivatives (one on the bra) and the kinetic integrals (two on
/// the ket) need.
pub(crate) fn primitive_pairs(
    a: &ShellRef<'_>,
    b: &ShellRef<'_>,
    b_center: [f64; 3],
    extra: [usize; 2],
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
                    a.angular_momentum + extra[0],
                    b.angular_momentum + extra[1],
                    p,
                    mu,
                    a.center[axis] - b_center[axis],
                    center[axis] - a.center[axis],
                    center[axis] - b_center[axis],
                )
            });
            pairs.push(PrimitivePair {
                p,
                exponent_a,
                exponent_b,
                center,
                order: a.angular_momentum + extra[0] + b.angular_momentum + extra[1],
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
    primitive_pairs(&shifted, &unit, center, [0, 0])
}

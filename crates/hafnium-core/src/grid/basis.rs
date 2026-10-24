//! Values and derivatives of contracted spherical Gaussians at grid points.
//!
//! A function `χ(r) = S_lm(r − C) Σ_k c_k exp(−α_k |r − C|²)` is evaluated
//! through its Cartesian components `x^i y^j z^k g(s)` with `s = |r − C|²`
//! and the contracted radial factor `g(s) = Σ_k c_k exp(−α_k s)`, whose
//! derivatives bring down powers of the exponents: with
//! `R_n = Σ_k c_k α_k^n exp(−α_k s)` and the monomial `m = x^i y^j z^k`,
//!
//! ```text
//! f      = m R_0
//! ∂_x f  = m_x R_0 − 2 x m R_1
//! ∂_xy f = m_xy R_0 − 2 (y m_x + x m_y) R_1 − 2 δ_xy m R_1 + 4 x y m R_2
//! ```
//!
//! The Cartesian components are transformed to the real solid harmonics of
//! ADR-0006 by the same matrix as the integrals.

use rayon::prelude::*;

use super::ShellTable;
use crate::Error;
use crate::integrals::{cartesian_components, n_cartesian, spherical_transformation};
use crate::to_f64;

/// Points evaluated per parallel task.
const POINTS_PER_TASK: usize = 32;

/// Number of derivative components up to `order`: the value, the three
/// first derivatives `x, y, z`, and the six second derivatives
/// `xx, xy, xz, yy, yz, zz`.
///
/// # Panics
///
/// Panics if `order` exceeds two; [`basis_values`] rejects such orders
/// before calling.
#[must_use]
pub fn n_components(order: usize) -> usize {
    match order {
        0 => 1,
        1 => 4,
        2 => 10,
        _ => panic!("derivatives are supported up to second order"),
    }
}

/// Values of every function of `table` at `points`, with derivatives up to
/// `order` (at most two), as a row-major `[point][component][ao]` array
/// with the components of [`n_components`].
///
/// `radii` holds one cutoff per shell in Bohr, infinity for none: at a
/// point farther than that from the shell's centre its functions are left
/// at zero. Python chooses
/// the radii from the exponents (`hafnium.integrals.periodic.function_radius`),
/// so the neglected values are below the precision it was given.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if `order` exceeds two, a point is
/// not finite, or a radius is negative or NaN, and
/// [`Error::LengthMismatch`] if there is not one radius per shell.
pub fn basis_values(
    table: &ShellTable,
    points: &[[f64; 3]],
    radii: &[f64],
    order: usize,
) -> Result<Vec<f64>, Error> {
    if order > 2 {
        return Err(Error::InvalidArgument(
            "derivatives are supported up to second order".to_string(),
        ));
    }
    if radii.len() != table.n_shells() {
        return Err(Error::LengthMismatch {
            left: radii.len(),
            right: table.n_shells(),
        });
    }
    if !radii.iter().all(|r| !r.is_nan() && *r >= 0.0) {
        return Err(Error::InvalidArgument(
            "cutoff radii are non-negative".to_string(),
        ));
    }
    if !points
        .iter()
        .all(|point| point.iter().all(|x| x.is_finite()))
    {
        return Err(Error::InvalidArgument("points are finite".to_string()));
    }
    let n_ao = table.n_functions();
    let n_comp = n_components(order);
    let max_l = table.max_angular_momentum();
    let transforms: Vec<Vec<f64>> = (0..=max_l).map(spherical_transformation).collect();
    let components: Vec<Vec<[usize; 3]>> = (0..=max_l).map(cartesian_components).collect();
    let offsets = table.function_offsets();
    let mut out = vec![0.0; points.len() * n_comp * n_ao];
    let stride = n_comp * n_ao;
    out.par_chunks_mut(POINTS_PER_TASK * stride)
        .zip(points.par_chunks(POINTS_PER_TASK))
        .for_each(|(block, chunk)| {
            let mut workspace = Workspace::new(max_l, n_comp);
            for (point, values) in chunk.iter().zip(block.chunks_mut(stride)) {
                for (shell_index, &offset) in offsets.iter().enumerate() {
                    let shell = table.shell(shell_index);
                    let l = shell.angular_momentum;
                    let d: [f64; 3] = std::array::from_fn(|axis| point[axis] - shell.center[axis]);
                    let s = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if s > radii[shell_index] * radii[shell_index] {
                        continue;
                    }
                    workspace.monomials(&components[l], l, d, order);
                    let n_sph = 2 * l + 1;
                    for contraction in 0..shell.n_contractions {
                        let radial = radial_sums(&shell, contraction, s, order);
                        workspace.cartesian(&components[l], d, &radial, order);
                        let target = &mut values[..];
                        workspace.scatter(
                            &transforms[l],
                            n_cartesian(l),
                            n_sph,
                            n_ao,
                            offset + contraction * n_sph,
                            target,
                        );
                    }
                }
            }
        });
    Ok(out)
}

/// `R_n = Σ_k c_k α_k^n exp(−α_k s)` for `n = 0, ..., order`.
pub(super) fn radial_sums(
    shell: &crate::integrals::ShellRef<'_>,
    contraction: usize,
    s: f64,
    order: usize,
) -> [f64; 3] {
    let mut sums = [0.0; 3];
    for (k, &alpha) in shell.exponents.iter().enumerate() {
        let term = shell.coefficient(contraction, k) * (-alpha * s).exp();
        sums[0] += term;
        if order >= 1 {
            sums[1] += alpha * term;
        }
        if order >= 2 {
            sums[2] += alpha * alpha * term;
        }
    }
    sums
}

/// Scratch buffers of one task: the monomials and their derivatives, the
/// Cartesian function components and the spherical result.
pub(super) struct Workspace {
    /// Powers of the displacement along each axis, `[axis][power]`.
    powers: [Vec<f64>; 3],
    /// `[cartesian][0..10]`: `m, m_x, m_y, m_z, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz`.
    monomials: Vec<[f64; 10]>,
    /// `[cartesian][component]` of the contracted function.
    cartesian: Vec<f64>,
    n_comp: usize,
}

impl Workspace {
    pub(super) fn new(max_l: usize, n_comp: usize) -> Self {
        let n_cart = n_cartesian(max_l);
        Self {
            powers: std::array::from_fn(|_| vec![0.0; max_l + 1]),
            monomials: vec![[0.0; 10]; n_cart],
            cartesian: vec![0.0; n_cart * n_comp],
            n_comp,
        }
    }

    /// Fill the monomial table for the components of degree `l` at the
    /// displacement `d`.
    pub(super) fn monomials(
        &mut self,
        components: &[[usize; 3]],
        l: usize,
        d: [f64; 3],
        order: usize,
    ) {
        for (powers, &x) in self.powers.iter_mut().zip(&d) {
            powers[0] = 1.0;
            for n in 1..=l {
                powers[n] = powers[n - 1] * x;
            }
        }
        let power = |axis: usize, n: i64| -> f64 {
            if n < 0 {
                0.0
            } else {
                self.powers[axis][usize::try_from(n).expect("non-negative")]
            }
        };
        for (slot, exponents) in self.monomials.iter_mut().zip(components) {
            let e: [i64; 3] =
                std::array::from_fn(|axis| i64::try_from(exponents[axis]).expect("small"));
            let f: [f64; 3] = std::array::from_fn(|axis| to_f64(exponents[axis]));
            // Factors of the monomial along each axis, with the derivative
            // orders 0, 1 and 2 along that axis.
            let along = |axis: usize, derivative: usize| -> f64 {
                match derivative {
                    0 => power(axis, e[axis]),
                    1 => f[axis] * power(axis, e[axis] - 1),
                    _ => f[axis] * (f[axis] - 1.0) * power(axis, e[axis] - 2),
                }
            };
            *slot = [0.0; 10];
            slot[0] = along(0, 0) * along(1, 0) * along(2, 0);
            if order >= 1 {
                slot[1] = along(0, 1) * along(1, 0) * along(2, 0);
                slot[2] = along(0, 0) * along(1, 1) * along(2, 0);
                slot[3] = along(0, 0) * along(1, 0) * along(2, 1);
            }
            if order >= 2 {
                slot[4] = along(0, 2) * along(1, 0) * along(2, 0);
                slot[5] = along(0, 1) * along(1, 1) * along(2, 0);
                slot[6] = along(0, 1) * along(1, 0) * along(2, 1);
                slot[7] = along(0, 0) * along(1, 2) * along(2, 0);
                slot[8] = along(0, 0) * along(1, 1) * along(2, 1);
                slot[9] = along(0, 0) * along(1, 0) * along(2, 2);
            }
        }
    }

    /// Combine the monomial table with the radial sums into the Cartesian
    /// function components, `[cartesian][component]`.
    pub(super) fn cartesian(
        &mut self,
        components: &[[usize; 3]],
        d: [f64; 3],
        radial: &[f64; 3],
        order: usize,
    ) {
        let [r0, r1, r2] = *radial;
        let n_comp = self.n_comp;
        for (c, m) in self.monomials[..components.len()].iter().enumerate() {
            let out = &mut self.cartesian[c * n_comp..(c + 1) * n_comp];
            out[0] = m[0] * r0;
            if order >= 1 {
                for axis in 0..3 {
                    out[1 + axis] = m[1 + axis] * r0 - 2.0 * d[axis] * m[0] * r1;
                }
            }
            if order >= 2 {
                // Second derivatives in the order xx, xy, xz, yy, yz, zz.
                let mut slot = 4;
                for a in 0..3 {
                    for b in a..3 {
                        let m_ab = m[slot];
                        let mut value = m_ab * r0 - 2.0 * (d[b] * m[1 + a] + d[a] * m[1 + b]) * r1
                            + 4.0 * d[a] * d[b] * m[0] * r2;
                        if a == b {
                            value -= 2.0 * m[0] * r1;
                        }
                        out[slot] = value;
                        slot += 1;
                    }
                }
            }
        }
    }

    /// Transform the Cartesian components to spherical functions and write
    /// them into `values[component][start + m]`.
    pub(super) fn scatter(
        &self,
        transform: &[f64],
        n_cart: usize,
        n_sph: usize,
        n_ao: usize,
        start: usize,
        values: &mut [f64],
    ) {
        let n_comp = self.n_comp;
        for (m, row) in transform.chunks(n_cart).enumerate() {
            for (c, &weight) in row.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                let source = &self.cartesian[c * n_comp..(c + 1) * n_comp];
                for (component, &value) in source.iter().enumerate() {
                    values[component * n_ao + start + m] += weight * value;
                }
            }
        }
        debug_assert_eq!(transform.len(), n_sph * n_cart);
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::integrals::test_tables::{general_table, s_shell, split_table};

    fn d_shell() -> ShellTable {
        ShellTable::new(
            vec![2],
            vec![[0.2, -0.1, 0.3]],
            vec![0, 2],
            vec![0.9, 0.3],
            vec![0.7, 0.5],
            vec![1],
        )
        .expect("valid")
    }

    fn values_at(table: &ShellTable, points: &[[f64; 3]], order: usize) -> Vec<f64> {
        let radii = vec![f64::INFINITY; table.n_shells()];
        basis_values(table, points, &radii, order).expect("valid")
    }

    #[test]
    fn s_function_matches_the_closed_form() {
        let a = 0.7;
        let center = [0.1, 0.2, 0.3];
        let shell = s_shell(a, center);
        let point = [0.6, -0.3, 0.9];
        let d: [f64; 3] = std::array::from_fn(|axis| point[axis] - center[axis]);
        let s = d.iter().map(|x| x * x).sum::<f64>();
        let norm = (2.0 * a / PI).powf(0.75);
        let f = norm * (-a * s).exp();
        let values = values_at(&shell, &[point], 2);
        assert_eq!(values.len(), 10);
        assert!((values[0] - f).abs() < 1e-14);
        for axis in 0..3 {
            assert!((values[1 + axis] + 2.0 * a * d[axis] * f).abs() < 1e-14);
        }
        let diagonal = |axis: usize| (4.0 * a * a * d[axis] * d[axis] - 2.0 * a) * f;
        let mixed = 4.0 * a * a * d[0] * d[1] * f;
        assert!((values[4] - diagonal(0)).abs() < 1e-14);
        assert!((values[5] - mixed).abs() < 1e-14);
        assert!((values[9] - diagonal(2)).abs() < 1e-14);
    }

    #[test]
    fn derivatives_match_finite_differences() {
        for table in [general_table(), d_shell()] {
            let n_ao = table.n_functions();
            let point = [0.45, -0.2, 0.35];
            let h = 1e-5;
            let values = values_at(&table, &[point], 2);
            for axis in 0..3 {
                let mut plus = point;
                plus[axis] += h;
                let mut minus = point;
                minus[axis] -= h;
                let up = values_at(&table, &[plus], 1);
                let down = values_at(&table, &[minus], 1);
                for ao in 0..n_ao {
                    let gradient = (up[ao] - down[ao]) / (2.0 * h);
                    assert!(
                        (gradient - values[(1 + axis) * n_ao + ao]).abs() < 1e-8,
                        "first derivative along {axis} of function {ao}"
                    );
                    // Second derivatives from the first derivatives.
                    let mut slot = 4;
                    for a in 0..3 {
                        for b in a..3 {
                            if a == axis {
                                let second = (up[(1 + b) * n_ao + ao] - down[(1 + b) * n_ao + ao])
                                    / (2.0 * h);
                                assert!(
                                    (second - values[slot * n_ao + ao]).abs() < 1e-7,
                                    "second derivative {a}{b} of function {ao}"
                                );
                            }
                            slot += 1;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn general_contractions_match_split_shells() {
        let points = [[0.3, 0.1, -0.2], [1.2, -0.7, 0.4]];
        let general = values_at(&general_table(), &points, 1);
        let split = values_at(&split_table(), &points, 1);
        assert_eq!(general.len(), split.len());
        for (g, s) in general.iter().zip(&split) {
            assert!((g - s).abs() < 1e-14);
        }
    }

    #[test]
    fn cutoff_radii_zero_the_far_functions() {
        let table = general_table();
        let points = [[0.1, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let n_ao = table.n_functions();
        let full = values_at(&table, &points, 0);
        let screened = basis_values(&table, &points, &[1.0, 0.5], 0).expect("valid");
        for ao in 0..n_ao {
            assert_eq!(screened[n_ao + ao], 0.0);
            assert!(full[n_ao + ao] != 0.0);
        }
        // The s shell at the origin is within reach of the first point, the p
        // shell at (0.9, -0.4, 0.3), 0.94 Bohr away, is not.
        assert_eq!(screened[0], full[0]);
        assert_eq!(screened[3], 0.0);
    }

    #[test]
    fn malformed_input_is_rejected() {
        let table = general_table();
        let point = [[0.0; 3]];
        assert!(matches!(
            basis_values(&table, &point, &[1.0, 1.0], 3),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            basis_values(&table, &point, &[1.0], 0),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            basis_values(&table, &point, &[-1.0, 1.0], 0),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            basis_values(&table, &[[f64::NAN, 0.0, 0.0]], &[1.0, 1.0], 0),
            Err(Error::InvalidArgument(_))
        ));
        assert_eq!(n_components(1), 4);
    }
}

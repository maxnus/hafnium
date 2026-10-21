//! One-electron integrals over shell pairs: overlap, kinetic energy and
//! nuclear attraction, with the ket shells translated by lattice vectors.

use std::f64::consts::PI;

use super::cartesian::{cartesian_components, n_cartesian, spherical_transformation};
use super::coulomb::HermiteIntegrals;
use super::hermite::ExpansionCoefficients;
use super::{ShellRef, ShellTable};
use crate::{Error, to_f64};

/// Overlap integrals `S[T][μ][ν] = ⟨μ | ν(r − T)⟩` for each translation
/// `T`, as a row-major `[n_translations, n_bra, n_ket]` array.
#[must_use]
pub fn overlap(bra: &ShellTable, ket: &ShellTable, translations: &[[f64; 3]]) -> Vec<f64> {
    assemble(bra, ket, translations, 0, |pairs, a, b, block| {
        for pair in pairs {
            let scale = pair.coefficient * (PI / pair.p).powf(1.5);
            for (ia, ca) in a.iter().enumerate() {
                for (ib, cb) in b.iter().enumerate() {
                    block[ia * b.len() + ib] += scale * pair.product(ca, cb, [0, 0, 0]);
                }
            }
        }
    })
}

/// Kinetic energy integrals `⟨μ | −½∇² | ν(r − T)⟩`, laid out as [`overlap`].
#[must_use]
pub fn kinetic(bra: &ShellTable, ket: &ShellTable, translations: &[[f64; 3]]) -> Vec<f64> {
    assemble(bra, ket, translations, 2, |pairs, a, b, block| {
        for pair in pairs {
            let scale = pair.coefficient * (PI / pair.p).powf(1.5);
            for (ia, ca) in a.iter().enumerate() {
                for (ib, cb) in b.iter().enumerate() {
                    let s: [f64; 3] =
                        std::array::from_fn(|axis| pair.hermite[axis].get(ca[axis], cb[axis], 0));
                    let d: [f64; 3] = std::array::from_fn(|axis| {
                        pair.second_derivative(axis, ca[axis], cb[axis])
                    });
                    let value = d[0] * s[1] * s[2] + s[0] * d[1] * s[2] + s[0] * s[1] * d[2];
                    block[ia * b.len() + ib] -= 0.5 * scale * value;
                }
            }
        }
    })
}

/// Nuclear attraction integrals `⟨μ | Σ_C −Z_C v(r_C) | ν(r − T)⟩` for point
/// charges `Z_C` at `positions`, laid out as [`overlap`].
///
/// With `omega = None` the kernel is the Coulomb kernel `1/r`; with
/// `Some(ω)` it is the long-range kernel `erf(ω r)/r` (ADR-0008).
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if there is not one position per
/// charge, and [`Error::InvalidArgument`] if `omega` is not positive.
pub fn nuclear_attraction(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
    charges: &[f64],
    positions: &[[f64; 3]],
    omega: Option<f64>,
) -> Result<Vec<f64>, Error> {
    if charges.len() != positions.len() {
        return Err(Error::LengthMismatch {
            left: charges.len(),
            right: positions.len(),
        });
    }
    if omega.is_some_and(|w| w.is_nan() || w <= 0.0) {
        return Err(Error::InvalidArgument(
            "the range-separation parameter omega is positive".to_string(),
        ));
    }
    Ok(assemble(bra, ket, translations, 0, |pairs, a, b, block| {
        for pair in pairs {
            // Attenuation replaces p by p ω² / (p + ω²) in every Hermite
            // integral and scales the prefactor by sqrt(alpha / p).
            let alpha = omega.map_or(pair.p, |w| pair.p * w * w / (pair.p + w * w));
            let prefactor = pair.coefficient * 2.0 * PI / pair.p * (alpha / pair.p).sqrt();
            let mut summed = HermiteIntegrals::zeros(pair.order);
            for (charge, position) in charges.iter().zip(positions) {
                let pc: [f64; 3] = std::array::from_fn(|axis| pair.center[axis] - position[axis]);
                summed.add_scaled(&HermiteIntegrals::new(pair.order, alpha, pc), -charge);
            }
            for (ia, ca) in a.iter().enumerate() {
                for (ib, cb) in b.iter().enumerate() {
                    let mut value = 0.0;
                    for t in 0..=ca[0] + cb[0] {
                        for u in 0..=ca[1] + cb[1] {
                            for v in 0..=ca[2] + cb[2] {
                                value += pair.product(ca, cb, [t, u, v]) * summed.get(t, u, v);
                            }
                        }
                    }
                    block[ia * b.len() + ib] += prefactor * value;
                }
            }
        }
    }))
}

/// A pair of primitives with their Hermite expansion along each axis.
struct PrimitivePair {
    p: f64,
    exponent_b: f64,
    center: [f64; 3],
    coefficient: f64,
    order: usize,
    hermite: [ExpansionCoefficients; 3],
}

impl PrimitivePair {
    /// Product of the expansion coefficients along the three axes.
    #[inline]
    fn product(&self, a: &[usize; 3], b: &[usize; 3], t: [usize; 3]) -> f64 {
        self.hermite[0].get(a[0], b[0], t[0])
            * self.hermite[1].get(a[1], b[1], t[1])
            * self.hermite[2].get(a[2], b[2], t[2])
    }

    /// `⟨x^i | d²/dx² | x^j⟩ / sqrt(π/p)` along one axis (Helgaker 9.3.33).
    #[inline]
    fn second_derivative(&self, axis: usize, i: usize, j: usize) -> f64 {
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
fn primitive_pairs(
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

/// Loop over translations and shell pairs, let `block_fn` fill the Cartesian
/// block of each pair, transform it to spherical functions and place it.
fn assemble(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
    extra_b: usize,
    block_fn: impl Fn(&[PrimitivePair], &[[usize; 3]], &[[usize; 3]], &mut [f64]),
) -> Vec<f64> {
    let n_bra = bra.n_functions();
    let n_ket = ket.n_functions();
    let max_l = bra.max_angular_momentum().max(ket.max_angular_momentum());
    let components: Vec<Vec<[usize; 3]>> = (0..=max_l).map(cartesian_components).collect();
    let transforms: Vec<Vec<f64>> = (0..=max_l).map(spherical_transformation).collect();
    let bra_offsets = bra.function_offsets();
    let ket_offsets = ket.function_offsets();
    let mut out = vec![0.0; translations.len() * n_bra * n_ket];
    for (it, translation) in translations.iter().enumerate() {
        let matrix = &mut out[it * n_bra * n_ket..(it + 1) * n_bra * n_ket];
        for (ia, &bra_offset) in bra_offsets.iter().enumerate() {
            let a = bra.shell(ia);
            for (ib, &ket_offset) in ket_offsets.iter().enumerate() {
                let b = ket.shell(ib);
                let b_center: [f64; 3] =
                    std::array::from_fn(|axis| b.center[axis] + translation[axis]);
                let pairs = primitive_pairs(&a, &b, b_center, extra_b);
                let (la, lb) = (a.angular_momentum, b.angular_momentum);
                let mut cartesian = vec![0.0; n_cartesian(la) * n_cartesian(lb)];
                block_fn(&pairs, &components[la], &components[lb], &mut cartesian);
                let spherical = to_spherical(&cartesian, &transforms[la], &transforms[lb], la, lb);
                for (row, values) in spherical.chunks(2 * lb + 1).enumerate() {
                    let start = (bra_offset + row) * n_ket + ket_offset;
                    matrix[start..start + values.len()].copy_from_slice(values);
                }
            }
        }
    }
    out
}

/// `C_a · block · C_bᵀ` for the spherical transformations of both shells.
fn to_spherical(block: &[f64], c_a: &[f64], c_b: &[f64], la: usize, lb: usize) -> Vec<f64> {
    let (nca, ncb) = (n_cartesian(la), n_cartesian(lb));
    let (nsa, nsb) = (2 * la + 1, 2 * lb + 1);
    // First contract the ket index: half[ca][mb] = Σ_cb block[ca][cb] C_b[mb][cb].
    let mut half = vec![0.0; nca * nsb];
    for ca in 0..nca {
        for mb in 0..nsb {
            half[ca * nsb + mb] = (0..ncb)
                .map(|cb| block[ca * ncb + cb] * c_b[mb * ncb + cb])
                .sum();
        }
    }
    let mut out = vec![0.0; nsa * nsb];
    for ma in 0..nsa {
        for mb in 0..nsb {
            out[ma * nsb + mb] = (0..nca)
                .map(|ca| c_a[ma * nca + ca] * half[ca * nsb + mb])
                .sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One normalized s primitive of exponent `a` at `center`.
    fn s_shell(a: f64, center: [f64; 3]) -> ShellTable {
        let norm = (2.0 * a / PI).powf(0.75) * (4.0 * PI).sqrt();
        ShellTable::new(vec![0], vec![center], vec![0, 1], vec![a], vec![norm]).expect("valid")
    }

    #[test]
    fn normalized_s_function_has_unit_overlap() {
        let shell = s_shell(0.7, [0.1, 0.2, 0.3]);
        let s = overlap(&shell, &shell, &[[0.0; 3]]);
        assert!((s[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn s_overlap_and_kinetic_match_closed_forms() {
        // Two s primitives: S = (π/p)^{3/2} exp(-μR²) (times norms), T = μ(3 − 2μR²) S.
        let (a, b) = (0.9, 0.4);
        let bra = s_shell(a, [0.0; 3]);
        let ket = s_shell(b, [0.0, 0.0, 1.2]);
        let s = overlap(&bra, &ket, &[[0.0; 3]])[0];
        let t = kinetic(&bra, &ket, &[[0.0; 3]])[0];
        let mu = a * b / (a + b);
        assert!((t - mu * (3.0 - 2.0 * mu * 1.44) * s).abs() < 1e-14);
    }

    #[test]
    fn translation_moves_the_ket() {
        let bra = s_shell(0.9, [0.0; 3]);
        let ket = s_shell(0.4, [0.0, 0.0, 0.5]);
        let moved = s_shell(0.4, [0.0, 0.0, 1.5]);
        let translated = overlap(&bra, &ket, &[[0.0; 3], [0.0, 0.0, 1.0]]);
        let direct = overlap(&bra, &moved, &[[0.0; 3]]);
        assert!((translated[1] - direct[0]).abs() < 1e-15);
        assert!(translated[0] > translated[1]);
    }

    #[test]
    fn nuclear_attraction_of_s_pair_matches_closed_form() {
        // ⟨a|−1/r_C|b⟩ = −(2π/p) exp(-μR²) F_0(p R_PC²) times the norms.
        let (exponent_a, exponent_b) = (1.1, 0.3);
        let bra = s_shell(exponent_a, [0.0; 3]);
        let ket = s_shell(exponent_b, [1.0, 0.0, 0.0]);
        let charge_position = [0.3, 0.4, 0.0];
        let attraction =
            nuclear_attraction(&bra, &ket, &[[0.0; 3]], &[1.0], &[charge_position], None)
                .expect("valid")[0];
        let overlap_value = overlap(&bra, &ket, &[[0.0; 3]])[0];
        let total = exponent_a + exponent_b;
        let center = [exponent_b / total, 0.0, 0.0];
        let distance_squared: f64 = (0..3)
            .map(|axis| (center[axis] - charge_position[axis]).powi(2))
            .sum();
        let boys = crate::special::boys_values(0, total * distance_squared)[0];
        let expected = -overlap_value * (2.0 * PI / total) * boys / (PI / total).powf(1.5);
        assert!((attraction - expected).abs() < 1e-14);
    }

    #[test]
    fn attenuated_attraction_approaches_the_full_kernel() {
        let bra = s_shell(1.1, [0.0; 3]);
        let ket = s_shell(0.3, [1.0, 0.0, 0.0]);
        let full = nuclear_attraction(&bra, &ket, &[[0.0; 3]], &[2.0], &[[0.2, 0.0, 0.0]], None)
            .expect("valid")[0];
        let long_range = nuclear_attraction(
            &bra,
            &ket,
            &[[0.0; 3]],
            &[2.0],
            &[[0.2, 0.0, 0.0]],
            Some(1e4),
        )
        .expect("valid")[0];
        assert!((full - long_range).abs() < 1e-7);
        assert!(long_range.abs() < full.abs());
    }

    #[test]
    fn rejects_bad_charges_and_omega() {
        let shell = s_shell(1.0, [0.0; 3]);
        assert!(nuclear_attraction(&shell, &shell, &[[0.0; 3]], &[1.0], &[], None).is_err());
        assert!(
            nuclear_attraction(&shell, &shell, &[[0.0; 3]], &[1.0], &[[0.0; 3]], Some(0.0))
                .is_err()
        );
    }
}

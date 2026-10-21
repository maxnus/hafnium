//! One-electron integrals over shell pairs: overlap, kinetic energy and
//! nuclear attraction, with the ket shells translated by lattice vectors.

use std::f64::consts::PI;

use super::ShellTable;
use super::cartesian::n_cartesian;
use super::coulomb::{HermiteTable, HermiteWorkspace};
use super::pairs::{PrimitivePair, primitive_pairs};
use super::two_electron::{Tables, check_finite, check_omega};
use crate::Error;

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
/// charge, and [`Error::InvalidArgument`] if `omega` is not positive or a
/// translation, charge or position is not finite (a non-finite distance
/// would otherwise reach the Boys function, which requires a finite
/// argument).
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
    check_omega(omega)?;
    check_finite(translations)?;
    check_finite(positions)?;
    if !charges.iter().all(|z| z.is_finite()) {
        return Err(Error::InvalidArgument("charges are finite".to_string()));
    }
    Ok(assemble(bra, ket, translations, 0, |pairs, a, b, block| {
        let mut workspace = HermiteWorkspace::default();
        for pair in pairs {
            // Attenuation replaces p by p ω² / (p + ω²) in every Hermite
            // integral and scales the prefactor by sqrt(alpha / p).
            let alpha = omega.map_or(pair.p, |w| pair.p * w * w / (pair.p + w * w));
            let prefactor = pair.coefficient * 2.0 * PI / pair.p * (alpha / pair.p).sqrt();
            let mut summed = HermiteTable::zeros(pair.order);
            for (charge, position) in charges.iter().zip(positions) {
                let pc: [f64; 3] = std::array::from_fn(|axis| pair.center[axis] - position[axis]);
                summed.add_scaled(workspace.compute(pair.order, alpha, pc), -charge);
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
    let tables = Tables::new(max_l);
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
                block_fn(
                    &pairs,
                    &tables.components[la],
                    &tables.components[lb],
                    &mut cartesian,
                );
                let spherical = tables.to_spherical(&cartesian, [Some(la), Some(lb), None, None]);
                for (row, values) in spherical.chunks(2 * lb + 1).enumerate() {
                    let start = (bra_offset + row) * n_ket + ket_offset;
                    matrix[start..start + values.len()].copy_from_slice(values);
                }
            }
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

    #[test]
    fn rejects_non_finite_positions_instead_of_panicking() {
        let shell = s_shell(1.0, [0.0; 3]);
        let nan = [f64::NAN, 0.0, 0.0];
        assert!(nuclear_attraction(&shell, &shell, &[[0.0; 3]], &[1.0], &[nan], None).is_err());
        assert!(nuclear_attraction(&shell, &shell, &[nan], &[1.0], &[[0.0; 3]], None).is_err());
        assert!(
            nuclear_attraction(
                &shell,
                &shell,
                &[[0.0; 3]],
                &[f64::INFINITY],
                &[[0.0; 3]],
                None
            )
            .is_err()
        );
    }
}

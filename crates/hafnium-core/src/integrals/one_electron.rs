//! One-electron integrals over shell pairs: overlap, kinetic energy and
//! nuclear attraction, with the ket shells translated by lattice vectors,
//! and their first derivatives with respect to the centres (ADR-0035).
//!
//! The nuclear attraction takes a [`CoulombKernel`] and a screening
//! threshold: every term of the sum over charges and primitive pairs whose
//! bound falls below the threshold is left out, so that the lattice sums
//! of periodic systems cost what their non-negligible terms cost. The
//! bounds are rigorous (ADR-0018): a Cartesian integral of a primitive
//! pair with a charge at distance `R` from the product centre is at most
//! `|Z| (2π/p) (1/√p) (s_hi − s_lo) exp(−(1 − ε) s_lo² R²) · M`, where
//! `[s_lo, s_hi]` is the exponent range of the kernel, `M` the pair's
//! envelope with the weights `s_hi^t h_t(ε)`, and `h_t` the Hermite
//! envelopes of [`super::pairs`].
//!
//! A derivative with respect to the centre of the bra function raises and
//! lowers its angular momentum, `∂_A G_a = 2α G_{a + e} − a_e G_{a − e}`,
//! so the derivative kernels evaluate the values of the undifferentiated
//! kernel for the raised and the lowered components from one expansion
//! with an extra order on the bra. A derivative with respect to the
//! position of a charge shifts the Hermite Coulomb integrals,
//! `∂_C R_{tuv} = −R_{t+1,uv}`. The derivatives with respect to the ket
//! centre follow from translational invariance and are not formed. The
//! derivative kernels evaluate every term.

use std::f64::consts::PI;

use rayon::prelude::*;

use super::ShellTable;
use super::cartesian::{center_derivative, n_cartesian};
use super::coulomb::{CoulombKernel, HermiteIntegrals, HermiteTable, HermiteWorkspace};
use super::pairs::{ENVELOPE_EPSILON, PrimitivePair, hermite_envelopes, primitive_pairs};
use super::two_electron::{Tables, check_finite, check_threshold};
use crate::Error;

/// Overlap integrals `S[T][μ][ν] = ⟨μ | ν(r − T)⟩` for each translation
/// `T`, as a row-major `[n_translations, n_bra, n_ket]` array.
#[must_use]
pub fn overlap(bra: &ShellTable, ket: &ShellTable, translations: &[[f64; 3]]) -> Vec<f64> {
    assemble(
        bra,
        ket,
        translations,
        [0, 0],
        1,
        |pair, [la, lb], tables, _screen, _workspace, values| {
            overlap_values(pair, &tables.components[la], &tables.components[lb], values);
            true
        },
    )
}

/// The derivative of the overlap with respect to the centre of the bra
/// function, `⟨∂_A μ | ν(r − T)⟩`, as `[n_translations, 3, n_bra, n_ket]`;
/// the derivative with respect to the ket centre is its negative.
#[must_use]
pub fn overlap_derivative(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
) -> Vec<f64> {
    assemble(
        bra,
        ket,
        translations,
        [1, 0],
        3,
        |pair, ls, tables, _screen, workspace, values| {
            bra_derivative_values(pair, ls, tables, workspace, values, |a, b, out| {
                overlap_values(pair, a, b, out);
            });
            true
        },
    )
}

/// Kinetic energy integrals `⟨μ | −½∇² | ν(r − T)⟩`, laid out as [`overlap`].
#[must_use]
pub fn kinetic(bra: &ShellTable, ket: &ShellTable, translations: &[[f64; 3]]) -> Vec<f64> {
    assemble(
        bra,
        ket,
        translations,
        [0, 2],
        1,
        |pair, [la, lb], tables, _screen, _workspace, values| {
            kinetic_values(pair, &tables.components[la], &tables.components[lb], values);
            true
        },
    )
}

/// The derivative of the kinetic energy integrals with respect to the
/// centre of the bra function, `⟨∂_A μ | −½∇² | ν(r − T)⟩`, laid out as
/// [`overlap_derivative`]; the derivative with respect to the ket centre
/// is its negative.
#[must_use]
pub fn kinetic_derivative(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
) -> Vec<f64> {
    assemble(
        bra,
        ket,
        translations,
        [1, 2],
        3,
        |pair, ls, tables, _screen, workspace, values| {
            bra_derivative_values(pair, ls, tables, workspace, values, |a, b, out| {
                kinetic_values(pair, a, b, out);
            });
            true
        },
    )
}

/// Nuclear attraction integrals `⟨μ | Σ_C −Z_C v(r_C) | ν(r − T)⟩` for point
/// charges `Z_C` at `positions`, laid out as [`overlap`].
///
/// `kernel` selects `v`: the Coulomb kernel `1/r`, its long-range part
/// `erf(ω r)/r` or its short-range part `erfc(ω r)/r` (ADR-0008,
/// ADR-0012). A primitive pair whose bound over all charges is below
/// `threshold` is skipped, and with the short-range kernel a charge whose
/// distance from the product centre puts the pair's bound below
/// `threshold` is skipped as well; every neglected term is smaller than
/// `threshold` in magnitude. A threshold of zero evaluates everything.
///
/// # Errors
///
/// Returns [`Error::LengthMismatch`] if there is not one position per
/// charge, and [`Error::InvalidArgument`] if the kernel's `omega` is not
/// positive, `threshold` is negative or not finite, or a translation, charge
/// or position is not finite (a non-finite distance would otherwise reach
/// the Boys function, which requires a finite argument).
pub fn nuclear_attraction(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
    charges: &[f64],
    positions: &[[f64; 3]],
    kernel: CoulombKernel,
    threshold: f64,
) -> Result<Vec<f64>, Error> {
    check_charges(translations, charges, positions, kernel)?;
    check_threshold(threshold)?;
    let charge_max = charges.iter().fold(0.0_f64, |acc, z| acc.max(z.abs()));
    let max_order = 2 * bra.max_angular_momentum().max(ket.max_angular_momentum());
    let envelopes = hermite_envelopes(max_order, ENVELOPE_EPSILON);
    Ok(assemble(
        bra,
        ket,
        translations,
        [0, 0],
        1,
        |pair, [la, lb], tables, screen, workspace, values| {
            let (a, b) = (&tables.components[la], &tables.components[lb]);
            let mut radius_squared = f64::INFINITY;
            if threshold > 0.0 {
                let (low, high) = kernel.exponent_range(pair.p);
                let bound =
                    charge_max * screen * attraction_bound(pair, a, b, low, high, &envelopes);
                if bound < threshold {
                    return false;
                }
                let decay = (1.0 - ENVELOPE_EPSILON) * low * low;
                if decay > 0.0 {
                    radius_squared = (bound / threshold).ln() / decay;
                }
            }
            let mut summed = HermiteTable::zeros(pair.order);
            for (charge, position) in charges.iter().zip(positions) {
                let pc = displacement(pair, position);
                if pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2] > radius_squared {
                    continue;
                }
                summed.add_scaled(
                    workspace.compute_kernel(pair.order, pair.p, kernel, pc),
                    -charge,
                );
            }
            contract_hermite(pair, a, b, &summed, 2.0 * PI / pair.p, values);
            true
        },
    ))
}

/// The derivatives of the nuclear attraction: with respect to the centre
/// of the bra function, `⟨∂_A μ | V | ν(r − T)⟩`, in the slots `0..3`, and
/// with respect to the position of each charge `C`,
/// `⟨μ | ∂_C (−Z_C v(r_C)) | ν(r − T)⟩`, in the slots `3 + 3C..6 + 3C`, as
/// `[n_translations, 3 (1 + n_charges), n_bra, n_ket]`. The derivative
/// with respect to the ket centre is minus the sum of all slots. Every
/// term is evaluated.
///
/// # Errors
///
/// As [`nuclear_attraction`], without the threshold.
pub fn nuclear_attraction_derivative(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
    charges: &[f64],
    positions: &[[f64; 3]],
    kernel: CoulombKernel,
) -> Result<Vec<f64>, Error> {
    check_charges(translations, charges, positions, kernel)?;
    Ok(assemble(
        bra,
        ket,
        translations,
        [1, 0],
        3 * (1 + charges.len()),
        |pair, ls, tables, _screen, workspace, values| {
            let [la, lb] = ls;
            let (a, b) = (&tables.components[la], &tables.components[lb]);
            let n_ab = a.len() * b.len();
            let prefactor = 2.0 * PI / pair.p;
            let mut summed = HermiteTable::zeros(pair.order);
            for (index, (charge, position)) in charges.iter().zip(positions).enumerate() {
                let table = workspace.compute_kernel(
                    pair.order,
                    pair.p,
                    kernel,
                    displacement(pair, position),
                );
                // ∂_C of −Z (2π/p) Σ E_tuv R_tuv is Z (2π/p) Σ E_tuv R_{tuv + e}.
                let slots = &mut values[(3 + 3 * index) * n_ab..][..3 * n_ab];
                contract_hermite_shifted(pair, a, b, &table, prefactor * charge, slots);
                summed.add_scaled(table, -charge);
            }
            bra_derivative_values(
                pair,
                ls,
                tables,
                workspace,
                &mut values[..3 * n_ab],
                |a, b, out| contract_hermite(pair, a, b, &summed, prefactor, out),
            );
            true
        },
    ))
}

/// Validate the arguments the attraction kernels share.
fn check_charges(
    translations: &[[f64; 3]],
    charges: &[f64],
    positions: &[[f64; 3]],
    kernel: CoulombKernel,
) -> Result<(), Error> {
    if charges.len() != positions.len() {
        return Err(Error::LengthMismatch {
            left: charges.len(),
            right: positions.len(),
        });
    }
    kernel.validate()?;
    check_finite(translations, "translations")?;
    check_finite(positions, "positions")?;
    if !charges.iter().all(|z| z.is_finite()) {
        return Err(Error::InvalidArgument("charges are finite".to_string()));
    }
    Ok(())
}

/// `P − C`, the displacement of a charge from the product centre.
#[inline]
fn displacement(pair: &PrimitivePair, position: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| pair.center[axis] - position[axis])
}

/// `values[a][b] = (π/p)^{3/2} E^{ab}_{000}` over the components.
fn overlap_values(pair: &PrimitivePair, a: &[[usize; 3]], b: &[[usize; 3]], values: &mut [f64]) {
    let scale = (PI / pair.p).powf(1.5);
    for (ia, ca) in a.iter().enumerate() {
        for (ib, cb) in b.iter().enumerate() {
            values[ia * b.len() + ib] = scale * pair.product(ca, cb, [0, 0, 0]);
        }
    }
}

/// `values[a][b] = ⟨a | −½∇² | b⟩` over the components, the second
/// derivative taken on the ket (Helgaker 9.3.33).
fn kinetic_values(pair: &PrimitivePair, a: &[[usize; 3]], b: &[[usize; 3]], values: &mut [f64]) {
    let scale = (PI / pair.p).powf(1.5);
    for (ia, ca) in a.iter().enumerate() {
        for (ib, cb) in b.iter().enumerate() {
            let s: [f64; 3] =
                std::array::from_fn(|axis| pair.hermite[axis].get(ca[axis], cb[axis], 0));
            let d: [f64; 3] =
                std::array::from_fn(|axis| pair.second_derivative(axis, ca[axis], cb[axis]));
            let value = d[0] * s[1] * s[2] + s[0] * d[1] * s[2] + s[0] * s[1] * d[2];
            values[ia * b.len() + ib] = -0.5 * scale * value;
        }
    }
}

/// `values[a][b] = scale Σ_{tuv} E^{ab}_{tuv} R_{tuv}` over the components,
/// for a table of Hermite Coulomb integrals.
fn contract_hermite(
    pair: &PrimitivePair,
    a: &[[usize; 3]],
    b: &[[usize; 3]],
    table: &impl HermiteIntegrals,
    scale: f64,
    values: &mut [f64],
) {
    for (ia, ca) in a.iter().enumerate() {
        for (ib, cb) in b.iter().enumerate() {
            let mut value = 0.0;
            for t in 0..=ca[0] + cb[0] {
                for u in 0..=ca[1] + cb[1] {
                    for v in 0..=ca[2] + cb[2] {
                        value += pair.product(ca, cb, [t, u, v]) * table.get(t, u, v);
                    }
                }
            }
            values[ia * b.len() + ib] = scale * value;
        }
    }
}

/// `values[axis][a][b] = scale Σ_{tuv} E^{ab}_{tuv} R_{tuv + e_axis}` over the
/// components, the three shifts of the table in one pass.
fn contract_hermite_shifted(
    pair: &PrimitivePair,
    a: &[[usize; 3]],
    b: &[[usize; 3]],
    table: &impl HermiteIntegrals,
    scale: f64,
    values: &mut [f64],
) {
    let n_ab = a.len() * b.len();
    for (ia, ca) in a.iter().enumerate() {
        for (ib, cb) in b.iter().enumerate() {
            let mut sums = [0.0; 3];
            for t in 0..=ca[0] + cb[0] {
                for u in 0..=ca[1] + cb[1] {
                    for v in 0..=ca[2] + cb[2] {
                        let weight = pair.product(ca, cb, [t, u, v]);
                        sums[0] += weight * table.get(t + 1, u, v);
                        sums[1] += weight * table.get(t, u + 1, v);
                        sums[2] += weight * table.get(t, u, v + 1);
                    }
                }
            }
            for (axis, sum) in sums.iter().enumerate() {
                values[axis * n_ab + ia * b.len() + ib] = scale * sum;
            }
        }
    }
}

/// Fill `values[axis][a][b]` with the derivative of a kernel's values with
/// respect to the bra centre, from the values `base` gives for the
/// components of the bra's angular momentum raised and lowered; the pair's
/// expansion carries the extra order on the bra.
fn bra_derivative_values(
    pair: &PrimitivePair,
    [la, lb]: [usize; 2],
    tables: &Tables,
    workspace: &mut HermiteWorkspace,
    values: &mut [f64],
    mut base: impl FnMut(&[[usize; 3]], &[[usize; 3]], &mut [f64]),
) {
    let b = &tables.components[lb];
    let mut raised = std::mem::take(&mut workspace.raised);
    raised.clear();
    raised.resize(n_cartesian(la + 1) * b.len(), 0.0);
    base(&tables.components[la + 1], b, &mut raised);
    let mut lowered = std::mem::take(&mut workspace.lowered);
    lowered.clear();
    if la > 0 {
        lowered.resize(n_cartesian(la - 1) * b.len(), 0.0);
        base(&tables.components[la - 1], b, &mut lowered);
    }
    center_derivative(
        pair.exponent_a,
        &tables.components[la],
        &raised,
        &lowered,
        b.len(),
        values,
    );
    workspace.raised = raised;
    workspace.lowered = lowered;
}

/// Bound on every Cartesian attraction integral of `pair` with a unit
/// charge at the product centre, for a kernel with the exponent range
/// `[low, high]`: `(2π/p) (1/√p) (high − low) max_ab Π_axis Σ_t |E_t| high^t h_t`.
/// At distance `R` the bound is this times `exp(−(1 − ε) low² R²)`, since
/// `R_tuv = (1/√p) ∫ ∂_x^t ∂_y^u ∂_z^v exp(−s² R²) ds` over the range and
/// `|∂_x^t exp(−s² x²)| ≤ s^t h_t exp(−(1 − ε) s² x²)`.
fn attraction_bound(
    pair: &PrimitivePair,
    a: &[[usize; 3]],
    b: &[[usize; 3]],
    low: f64,
    high: f64,
    envelopes: &[f64],
) -> f64 {
    let weights: Vec<f64> = envelopes
        .iter()
        .enumerate()
        .take(pair.order + 1)
        .map(|(t, h)| high.powi(i32::try_from(t).expect("order fits in i32")) * h)
        .collect();
    2.0 * PI / pair.p / pair.p.sqrt() * (high - low) * pair.envelope(a, b, &weights)
}

/// Loop over translations in parallel and over shell pairs, let `fill`
/// write the Cartesian values of each primitive pair into `n_slots` slots
/// `[slot][a][b]` (or return `false` to skip the pair), accumulate them into
/// every contraction, transform to spherical functions and place the
/// blocks as `[translation][slot][n_bra][n_ket]`. `extra` raises the bra
/// and the ket order of the expansion of every pair. `screen` passed to
/// `fill` is the factor by which a Cartesian primitive value can be
/// amplified on its way into the matrix: the largest contraction
/// coefficient of the pair times the row norms of the two spherical
/// transformations.
fn assemble(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
    extra: [usize; 2],
    n_slots: usize,
    fill: impl Fn(&PrimitivePair, [usize; 2], &Tables, f64, &mut HermiteWorkspace, &mut [f64]) -> bool
    + Sync,
) -> Vec<f64> {
    let n_bra = bra.n_functions();
    let n_ket = ket.n_functions();
    let max_l = bra.max_angular_momentum().max(ket.max_angular_momentum());
    let tables = Tables::new(max_l + extra[0]);
    let bra_offsets = bra.function_offsets();
    let ket_offsets = ket.function_offsets();
    let matrix_size = n_bra * n_ket;
    let mut out = vec![0.0; translations.len() * n_slots * matrix_size];
    out.par_chunks_mut((n_slots * matrix_size).max(1))
        .zip(translations.par_iter())
        .for_each(|(matrices, translation)| {
            let mut workspace = HermiteWorkspace::default();
            for (ia, &bra_offset) in bra_offsets.iter().enumerate() {
                let a = bra.shell(ia);
                for (ib, &ket_offset) in ket_offsets.iter().enumerate() {
                    let b = ket.shell(ib);
                    let b_center: [f64; 3] =
                        std::array::from_fn(|axis| b.center[axis] + translation[axis]);
                    let pairs = primitive_pairs(&a, &b, b_center, extra);
                    let (la, lb) = (a.angular_momentum, b.angular_momentum);
                    let (nca, ncb) = (n_cartesian(la), n_cartesian(lb));
                    let shell_screen = tables.row_norms[la] * tables.row_norms[lb];
                    let block = nca * ncb;
                    let contracted = pairs.n_contracted() * block;
                    let mut cartesian = vec![0.0; n_slots * contracted];
                    let mut values = vec![0.0; n_slots * block];
                    for (index, pair) in pairs.pairs.iter().enumerate() {
                        let screen = shell_screen * pairs.coefficient_max(index);
                        if fill(pair, [la, lb], &tables, screen, &mut workspace, &mut values) {
                            for slot in 0..n_slots {
                                pairs.accumulate(
                                    index,
                                    &values[slot * block..][..block],
                                    nca,
                                    ncb,
                                    &mut cartesian[slot * contracted..][..contracted],
                                );
                            }
                        }
                    }
                    for slot in 0..n_slots {
                        let spherical = tables.to_spherical(
                            &cartesian[slot * contracted..][..contracted],
                            [Some(la), Some(lb), None, None],
                            [a.n_contractions, b.n_contractions, 1, 1],
                        );
                        let matrix = &mut matrices[slot * matrix_size..][..matrix_size];
                        for (row, values) in spherical.chunks(b.n_functions()).enumerate() {
                            let start = (bra_offset + row) * n_ket + ket_offset;
                            matrix[start..start + values.len()].copy_from_slice(values);
                        }
                    }
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrals::test_tables::{general_table, s_shell, split_table};
    use crate::to_f64;

    fn assert_same(general: &[f64], split: &[f64]) {
        assert_eq!(general.len(), split.len());
        for (g, s) in general.iter().zip(split) {
            assert!((g - s).abs() < 1e-14, "{g} vs {s}");
        }
    }

    #[test]
    fn general_contractions_match_split_shells() {
        let (general, split) = (general_table(), split_table());
        let translations = [[0.0; 3], [0.3, -0.2, 0.5]];
        assert_eq!(general.n_functions(), split.n_functions());
        assert_same(
            &overlap(&general, &general, &translations),
            &overlap(&split, &split, &translations),
        );
        assert_same(
            &kinetic(&general, &split, &translations),
            &kinetic(&split, &split, &translations),
        );
        let charges = [1.0, 2.0];
        let positions = [[0.1, 0.2, 0.3], [-0.5, 0.4, 0.0]];
        let kernel = CoulombKernel::LongRange(0.7);
        let general_v = nuclear_attraction(
            &general,
            &general,
            &translations,
            &charges,
            &positions,
            kernel,
            0.0,
        )
        .expect("valid");
        let split_v = nuclear_attraction(
            &split,
            &split,
            &translations,
            &charges,
            &positions,
            kernel,
            0.0,
        )
        .expect("valid");
        assert_same(&general_v, &split_v);
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
        let attraction = nuclear_attraction(
            &bra,
            &ket,
            &[[0.0; 3]],
            &[1.0],
            &[charge_position],
            CoulombKernel::Full,
            0.0,
        )
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
        let full = nuclear_attraction(
            &bra,
            &ket,
            &[[0.0; 3]],
            &[2.0],
            &[[0.2, 0.0, 0.0]],
            CoulombKernel::Full,
            0.0,
        )
        .expect("valid")[0];
        let long_range = nuclear_attraction(
            &bra,
            &ket,
            &[[0.0; 3]],
            &[2.0],
            &[[0.2, 0.0, 0.0]],
            CoulombKernel::LongRange(1e4),
            0.0,
        )
        .expect("valid")[0];
        assert!((full - long_range).abs() < 1e-7);
        assert!(long_range.abs() < full.abs());
    }

    #[test]
    fn rejects_bad_charges_omega_and_threshold() {
        let shell = s_shell(1.0, [0.0; 3]);
        let full = CoulombKernel::Full;
        assert!(nuclear_attraction(&shell, &shell, &[[0.0; 3]], &[1.0], &[], full, 0.0).is_err());
        let unchecked = CoulombKernel::ShortRange(f64::NAN);
        assert!(
            nuclear_attraction(
                &shell,
                &shell,
                &[[0.0; 3]],
                &[1.0],
                &[[0.0; 3]],
                unchecked,
                0.0
            )
            .is_err()
        );
        for threshold in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                nuclear_attraction(
                    &shell,
                    &shell,
                    &[[0.0; 3]],
                    &[1.0],
                    &[[0.0; 3]],
                    full,
                    threshold
                )
                .is_err()
            );
        }
    }

    #[test]
    fn rejects_non_finite_positions_instead_of_panicking() {
        let shell = s_shell(1.0, [0.0; 3]);
        let nan = [f64::NAN, 0.0, 0.0];
        let full = CoulombKernel::Full;
        assert!(
            nuclear_attraction(&shell, &shell, &[[0.0; 3]], &[1.0], &[nan], full, 0.0).is_err()
        );
        assert!(
            nuclear_attraction(&shell, &shell, &[nan], &[1.0], &[[0.0; 3]], full, 0.0).is_err()
        );
        assert!(
            nuclear_attraction(
                &shell,
                &shell,
                &[[0.0; 3]],
                &[f64::INFINITY],
                &[[0.0; 3]],
                full,
                0.0
            )
            .is_err()
        );
    }

    /// Charges on a line, from the origin outwards, with the units mixed.
    fn line_of_charges(count: usize, spacing: f64) -> (Vec<f64>, Vec<[f64; 3]>) {
        let charges = (0..count)
            .map(|i| if i % 2 == 0 { 1.0 } else { 3.0 })
            .collect();
        let positions = (0..count)
            .map(|i| [0.3 * to_f64(i % 3), -0.2, spacing * to_f64(i)])
            .collect();
        (charges, positions)
    }

    #[test]
    fn short_range_kernel_is_the_full_minus_the_long_range_kernel() {
        let table = general_table();
        let translations = [[0.0; 3], [0.3, -0.2, 0.5]];
        let (charges, positions) = line_of_charges(4, 1.1);
        let evaluate = |kernel| {
            nuclear_attraction(
                &table,
                &table,
                &translations,
                &charges,
                &positions,
                kernel,
                0.0,
            )
            .expect("valid")
        };
        let full = evaluate(CoulombKernel::Full);
        let long_range = evaluate(CoulombKernel::LongRange(0.8));
        let short_range = evaluate(CoulombKernel::ShortRange(0.8));
        for ((f, l), s) in full.iter().zip(&long_range).zip(&short_range) {
            assert!((f - l - s).abs() < 1e-14, "{f} − {l} vs {s}");
        }
    }

    #[test]
    fn screening_neglects_only_terms_below_the_threshold() {
        let table = general_table();
        let translations = [[0.0; 3], [8.0, 0.0, 0.0], [40.0, 0.0, 0.0]];
        let (charges, positions) = line_of_charges(60, 0.7);
        let threshold = 1e-10;
        for kernel in [
            CoulombKernel::Full,
            CoulombKernel::LongRange(0.6),
            CoulombKernel::ShortRange(0.6),
        ] {
            let exact = nuclear_attraction(
                &table,
                &table,
                &translations,
                &charges,
                &positions,
                kernel,
                0.0,
            )
            .expect("valid");
            let screened = nuclear_attraction(
                &table,
                &table,
                &translations,
                &charges,
                &positions,
                kernel,
                threshold,
            )
            .expect("valid");
            let largest = exact
                .iter()
                .zip(&screened)
                .map(|(e, s)| (e - s).abs())
                .fold(0.0_f64, f64::max);
            // At most a few neglected terms per element, each below the threshold.
            assert!(largest < 10.0 * threshold, "{kernel:?}: {largest}");
            // The far image is screened away entirely by the short-range kernel.
            if kernel == CoulombKernel::ShortRange(0.6) {
                let n = table.n_functions();
                assert!(screened[2 * n * n..].iter().all(|v| *v == 0.0));
                assert!(exact[2 * n * n..].iter().any(|v| v.abs() > 0.0));
            }
        }
        // A threshold above every bound leaves nothing.
        let nothing = nuclear_attraction(
            &table,
            &table,
            &translations,
            &charges,
            &positions,
            CoulombKernel::Full,
            1e6,
        )
        .expect("valid");
        assert!(nothing.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn attraction_bound_holds_for_higher_angular_momenta() {
        // Single primitives with unit coefficients, so that the bound of a
        // pair applies to the Cartesian values up to the spherical row norms.
        let envelopes = hermite_envelopes(6, ENVELOPE_EPSILON);
        let tables = Tables::new(3);
        for (la, lb, exponent_a, exponent_b) in
            [(0, 2, 0.9, 0.3), (1, 3, 1.4, 0.5), (2, 2, 0.4, 0.6)]
        {
            let center_b = [0.8, -0.4, 0.5];
            let bra = ShellTable::new(
                vec![la],
                vec![[0.0; 3]],
                vec![0, 1],
                vec![exponent_a],
                vec![1.0],
                vec![1],
            )
            .expect("valid");
            let ket = ShellTable::new(
                vec![lb],
                vec![center_b],
                vec![0, 1],
                vec![exponent_b],
                vec![1.0],
                vec![1],
            )
            .expect("valid");
            let pairs = primitive_pairs(&bra.shell(0), &ket.shell(0), center_b, [0, 0]);
            let pair = &pairs.pairs[0];
            for omega in [0.4, 1.5] {
                let kernel = CoulombKernel::ShortRange(omega);
                let (low, high) = kernel.exponent_range(pair.p);
                let at_zero = attraction_bound(
                    pair,
                    &tables.components[la],
                    &tables.components[lb],
                    low,
                    high,
                    &envelopes,
                ) * tables.row_norms[la]
                    * tables.row_norms[lb];
                for distance in [0.0, 1.0, 2.5, 4.0, 7.0] {
                    let charge = [
                        pair.center[0] + 0.6 * distance,
                        pair.center[1] - 0.8 * distance,
                        pair.center[2],
                    ];
                    let values =
                        nuclear_attraction(&bra, &ket, &[[0.0; 3]], &[1.0], &[charge], kernel, 0.0)
                            .expect("valid");
                    let largest = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
                    let bound = at_zero
                        * (-(1.0 - ENVELOPE_EPSILON) * low * low * distance * distance).exp();
                    assert!(
                        largest <= bound,
                        "l = {la},{lb} ω = {omega} R = {distance}: {largest} > {bound}"
                    );
                    // The bound is loose by a factor that grows with the angular
                    // momenta (about 30 for s-d pairs, 2e4 for d-d pairs) and
                    // enters the screening radius logarithmically.
                    if distance == 0.0 {
                        assert!(bound < 1e6 * largest, "{bound} vs {largest}");
                    }
                }
            }
        }
    }

    use crate::integrals::test_tables::{
        assert_close, assert_rows_match, central_difference, d_shell, rows_of, sp_table_at,
        vector_difference,
    };

    #[test]
    fn overlap_and_kinetic_derivatives_match_finite_differences() {
        let ket = sp_table_at([0.4, 0.1, -0.2]);
        let translations = [[0.0; 3], [0.3, -0.2, 0.5]];
        let n_ket = ket.n_functions();
        for bra in [general_table(), d_shell()] {
            let n_bra = bra.n_functions();
            let s = overlap_derivative(&bra, &ket, &translations);
            let t = kinetic_derivative(&bra, &ket, &translations);
            for shell in 0..bra.n_shells() {
                for axis in 0..3 {
                    let rows = rows_of(&bra, shell);
                    let ds = central_difference(&bra, shell, axis, |moved| {
                        overlap(moved, &ket, &translations)
                    });
                    assert_rows_match(&s, (3, axis), (n_bra, n_ket), rows.clone(), &ds);
                    let dt = central_difference(&bra, shell, axis, |moved| {
                        kinetic(moved, &ket, &translations)
                    });
                    assert_rows_match(&t, (3, axis), (n_bra, n_ket), rows, &dt);
                }
            }
        }
    }

    #[test]
    fn attraction_derivatives_match_finite_differences() {
        let ket = sp_table_at([0.4, 0.1, -0.2]);
        let translations = [[0.0; 3], [0.3, -0.2, 0.5]];
        let charges = [1.0, 2.5];
        let positions = [[0.1, 0.2, 0.3], [-0.5, 0.4, 0.0]];
        let n_ket = ket.n_functions();
        let n_slots = 3 * (1 + charges.len());
        for (bra, kernel) in [
            (general_table(), CoulombKernel::Full),
            (d_shell(), CoulombKernel::Full),
            (general_table(), CoulombKernel::LongRange(0.7)),
        ] {
            let n_bra = bra.n_functions();
            let evaluate = |bra: &ShellTable, positions: &[[f64; 3]]| {
                nuclear_attraction(bra, &ket, &translations, &charges, positions, kernel, 0.0)
                    .expect("valid")
            };
            let derivative = nuclear_attraction_derivative(
                &bra,
                &ket,
                &translations,
                &charges,
                &positions,
                kernel,
            )
            .expect("valid");
            for axis in 0..3 {
                for shell in 0..bra.n_shells() {
                    let difference =
                        central_difference(&bra, shell, axis, |moved| evaluate(moved, &positions));
                    assert_rows_match(
                        &derivative,
                        (n_slots, axis),
                        (n_bra, n_ket),
                        rows_of(&bra, shell),
                        &difference,
                    );
                }
                for (index, position) in positions.iter().enumerate() {
                    let difference = vector_difference(*position, axis, |moved| {
                        let mut positions = positions;
                        positions[index] = moved;
                        evaluate(&bra, &positions)
                    });
                    assert_rows_match(
                        &derivative,
                        (n_slots, 3 + 3 * index + axis),
                        (n_bra, n_ket),
                        0..n_bra,
                        &difference,
                    );
                }
            }
        }
    }

    #[test]
    fn attraction_derivatives_sum_to_zero_under_translation() {
        // With one table on both sides and no translation the ket derivative
        // is the transpose of the bra derivative, and ∂_A + ∂_B + Σ_C ∂_C = 0.
        let table = general_table();
        let charges = [1.0, 2.5];
        let positions = [[0.1, 0.2, 0.3], [-0.5, 0.4, 0.0]];
        let n = table.n_functions();
        let derivative = nuclear_attraction_derivative(
            &table,
            &table,
            &[[0.0; 3]],
            &charges,
            &positions,
            CoulombKernel::Full,
        )
        .expect("valid");
        let slot = |s: usize, i: usize, j: usize| derivative[(s * n + i) * n + j];
        for axis in 0..3 {
            for i in 0..n {
                for j in 0..n {
                    let charges_total: f64 = (0..charges.len())
                        .map(|c| slot(3 + 3 * c + axis, i, j))
                        .sum();
                    let total = slot(axis, i, j) + slot(axis, j, i) + charges_total;
                    assert!(total.abs() < 1e-12, "{total}");
                }
            }
        }
    }

    #[test]
    fn derivative_kernels_treat_general_contractions_like_split_shells() {
        let (general, split) = (general_table(), split_table());
        let translations = [[0.0; 3], [0.3, -0.2, 0.5]];
        assert_close(
            &overlap_derivative(&general, &general, &translations),
            &overlap_derivative(&split, &split, &translations),
            1e-12,
        );
        assert_close(
            &kinetic_derivative(&general, &split, &translations),
            &kinetic_derivative(&split, &split, &translations),
            1e-12,
        );
        let charges = [1.0, 2.0];
        let positions = [[0.1, 0.2, 0.3], [-0.5, 0.4, 0.0]];
        let kernel = CoulombKernel::ShortRange(0.7);
        let evaluate = |table: &ShellTable| {
            nuclear_attraction_derivative(table, table, &translations, &charges, &positions, kernel)
                .expect("valid")
        };
        assert_close(&evaluate(&general), &evaluate(&split), 1e-12);
    }
}

//! One-electron integrals over shell pairs: overlap, kinetic energy and
//! nuclear attraction, with the ket shells translated by lattice vectors.
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

use std::f64::consts::PI;

use rayon::prelude::*;

use super::ShellTable;
use super::cartesian::n_cartesian;
use super::coulomb::{HermiteTable, HermiteWorkspace};
use super::pairs::{ENVELOPE_EPSILON, PrimitivePair, hermite_envelopes, primitive_pairs};
use super::two_electron::{Tables, check_finite, check_omega};
use crate::Error;

/// The Coulomb kernel of an attraction integral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoulombKernel {
    /// `1 / r`.
    Full,
    /// `erf(ω r) / r`, the long-range part of the range separation.
    LongRange(f64),
    /// `erfc(ω r) / r`, the short-range part.
    ShortRange(f64),
}

impl CoulombKernel {
    /// Build from the binding's arguments: `omega` selects the
    /// range-separated kernels and `short_range` picks `erfc` over `erf`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `omega` is not positive, or if
    /// `short_range` is set without `omega`.
    pub fn new(omega: Option<f64>, short_range: bool) -> Result<Self, Error> {
        check_omega(omega)?;
        match (omega, short_range) {
            (None, false) => Ok(Self::Full),
            (None, true) => Err(Error::InvalidArgument(
                "the short-range kernel needs the range-separation parameter omega".to_string(),
            )),
            (Some(w), false) => Ok(Self::LongRange(w)),
            (Some(w), true) => Ok(Self::ShortRange(w)),
        }
    }

    /// The range-separation parameter, if the kernel has one.
    fn omega(self) -> Option<f64> {
        match self {
            Self::Full => None,
            Self::LongRange(w) | Self::ShortRange(w) => Some(w),
        }
    }

    /// The exponent range `[s_lo, s_hi]` with
    /// `R_000 = (1/√p) ∫_{s_lo}^{s_hi} exp(−s² R²) ds` for a pair of total
    /// exponent `p`: `[0, √p]` for the full kernel, `[0, √α]` and `[√α, √p]`
    /// for the long- and short-range parts with `α = p ω² / (p + ω²)`.
    fn exponent_range(self, p: f64) -> (f64, f64) {
        let attenuated = |w: f64| (p * w * w / (p + w * w)).sqrt();
        match self {
            Self::Full => (0.0, p.sqrt()),
            Self::LongRange(w) => (0.0, attenuated(w)),
            Self::ShortRange(w) => (attenuated(w), p.sqrt()),
        }
    }
}

/// Overlap integrals `S[T][μ][ν] = ⟨μ | ν(r − T)⟩` for each translation
/// `T`, as a row-major `[n_translations, n_bra, n_ket]` array.
#[must_use]
pub fn overlap(bra: &ShellTable, ket: &ShellTable, translations: &[[f64; 3]]) -> Vec<f64> {
    assemble(
        bra,
        ket,
        translations,
        0,
        |pair, a, b, _screen, _workspace, values| {
            let scale = (PI / pair.p).powf(1.5);
            for (ia, ca) in a.iter().enumerate() {
                for (ib, cb) in b.iter().enumerate() {
                    values[ia * b.len() + ib] = scale * pair.product(ca, cb, [0, 0, 0]);
                }
            }
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
        2,
        |pair, a, b, _screen, _workspace, values| {
            let scale = (PI / pair.p).powf(1.5);
            for (ia, ca) in a.iter().enumerate() {
                for (ib, cb) in b.iter().enumerate() {
                    let s: [f64; 3] =
                        std::array::from_fn(|axis| pair.hermite[axis].get(ca[axis], cb[axis], 0));
                    let d: [f64; 3] = std::array::from_fn(|axis| {
                        pair.second_derivative(axis, ca[axis], cb[axis])
                    });
                    let value = d[0] * s[1] * s[2] + s[0] * d[1] * s[2] + s[0] * s[1] * d[2];
                    values[ia * b.len() + ib] = -0.5 * scale * value;
                }
            }
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
    if charges.len() != positions.len() {
        return Err(Error::LengthMismatch {
            left: charges.len(),
            right: positions.len(),
        });
    }
    check_omega(kernel.omega())?;
    check_threshold(threshold)?;
    check_finite(translations, "translations")?;
    check_finite(positions, "positions")?;
    if !charges.iter().all(|z| z.is_finite()) {
        return Err(Error::InvalidArgument("charges are finite".to_string()));
    }
    let charge_max = charges.iter().fold(0.0_f64, |acc, z| acc.max(z.abs()));
    let max_order = 2 * bra.max_angular_momentum().max(ket.max_angular_momentum());
    let envelopes = hermite_envelopes(max_order, ENVELOPE_EPSILON);
    Ok(assemble(
        bra,
        ket,
        translations,
        0,
        |pair, a, b, screen, workspace, values| {
            let (low, high) = kernel.exponent_range(pair.p);
            let mut radius_squared = f64::INFINITY;
            if threshold > 0.0 {
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
            let alpha = match kernel {
                CoulombKernel::Full => pair.p,
                CoulombKernel::LongRange(_) => high * high,
                CoulombKernel::ShortRange(_) => low * low,
            };
            let attenuation = (alpha / pair.p).sqrt();
            let mut summed = HermiteTable::zeros(pair.order);
            for (charge, position) in charges.iter().zip(positions) {
                let pc: [f64; 3] = std::array::from_fn(|axis| pair.center[axis] - position[axis]);
                if pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2] > radius_squared {
                    continue;
                }
                // Attenuation replaces p by α in every Hermite integral and
                // scales the result by sqrt(α / p); the short-range kernel is
                // the full kernel minus the attenuated one.
                match kernel {
                    CoulombKernel::Full => {
                        summed.add_scaled(workspace.compute(pair.order, pair.p, pc), -charge);
                    }
                    CoulombKernel::LongRange(_) => {
                        summed.add_scaled(
                            workspace.compute(pair.order, alpha, pc),
                            -charge * attenuation,
                        );
                    }
                    CoulombKernel::ShortRange(_) => {
                        summed.add_scaled(workspace.compute(pair.order, pair.p, pc), -charge);
                        summed.add_scaled(
                            workspace.compute(pair.order, alpha, pc),
                            charge * attenuation,
                        );
                    }
                }
            }
            let prefactor = 2.0 * PI / pair.p;
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
                    values[ia * b.len() + ib] = prefactor * value;
                }
            }
            true
        },
    ))
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

pub(crate) fn check_threshold(threshold: f64) -> Result<(), Error> {
    if threshold.is_finite() && threshold >= 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "the screening threshold is finite and non-negative".to_string(),
        ))
    }
}

/// Loop over translations in parallel and over shell pairs, let `values_fn`
/// fill the Cartesian values of each primitive pair (or return `false` to
/// skip it), accumulate them into every contraction, transform to spherical
/// functions and place the block. `screen` passed to `values_fn` is the
/// factor by which a Cartesian primitive value can be amplified on its way
/// into the matrix: the largest contraction coefficient of the pair times
/// the row norms of the two spherical transformations.
fn assemble(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
    extra_b: usize,
    values_fn: impl Fn(
        &PrimitivePair,
        &[[usize; 3]],
        &[[usize; 3]],
        f64,
        &mut HermiteWorkspace,
        &mut [f64],
    ) -> bool
    + Sync,
) -> Vec<f64> {
    let n_bra = bra.n_functions();
    let n_ket = ket.n_functions();
    let max_l = bra.max_angular_momentum().max(ket.max_angular_momentum());
    let tables = Tables::new(max_l);
    let bra_offsets = bra.function_offsets();
    let ket_offsets = ket.function_offsets();
    let mut out = vec![0.0; translations.len() * n_bra * n_ket];
    out.par_chunks_mut((n_bra * n_ket).max(1))
        .zip(translations.par_iter())
        .for_each(|(matrix, translation)| {
            let mut workspace = HermiteWorkspace::default();
            for (ia, &bra_offset) in bra_offsets.iter().enumerate() {
                let a = bra.shell(ia);
                for (ib, &ket_offset) in ket_offsets.iter().enumerate() {
                    let b = ket.shell(ib);
                    let b_center: [f64; 3] =
                        std::array::from_fn(|axis| b.center[axis] + translation[axis]);
                    let pairs = primitive_pairs(&a, &b, b_center, extra_b);
                    let (la, lb) = (a.angular_momentum, b.angular_momentum);
                    let (nca, ncb) = (n_cartesian(la), n_cartesian(lb));
                    let shell_screen = tables.row_norms[la] * tables.row_norms[lb];
                    let mut cartesian = vec![0.0; pairs.n_contracted() * nca * ncb];
                    let mut values = vec![0.0; nca * ncb];
                    for (index, pair) in pairs.pairs.iter().enumerate() {
                        let screen = shell_screen * pairs.coefficient_max(index);
                        if values_fn(
                            pair,
                            &tables.components[la],
                            &tables.components[lb],
                            screen,
                            &mut workspace,
                            &mut values,
                        ) {
                            pairs.accumulate(index, &values, nca, ncb, &mut cartesian);
                        }
                    }
                    let spherical = tables.to_spherical(
                        &cartesian,
                        [Some(la), Some(lb), None, None],
                        [a.n_contractions, b.n_contractions, 1, 1],
                    );
                    for (row, values) in spherical.chunks(b.n_functions()).enumerate() {
                        let start = (bra_offset + row) * n_ket + ket_offset;
                        matrix[start..start + values.len()].copy_from_slice(values);
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
        assert!(CoulombKernel::new(Some(0.0), false).is_err());
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
        assert!(CoulombKernel::new(Some(f64::NAN), true).is_err());
        assert!(CoulombKernel::new(None, true).is_err());
        assert_eq!(CoulombKernel::new(None, false).expect("valid"), full);
        assert_eq!(
            CoulombKernel::new(Some(0.5), true).expect("valid"),
            CoulombKernel::ShortRange(0.5)
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
            let pairs = primitive_pairs(&bra.shell(0), &ket.shell(0), center_b, 0);
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
}

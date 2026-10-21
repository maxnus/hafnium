//! Coulomb integrals between two charge distributions in the Hermite basis.
//!
//! `(ab|cd) = (2π^{5/2} / (pq √(p+q))) Σ_{tuv} E^{ab}_{tuv} Σ_{τνφ}
//! (−1)^{τ+ν+φ} E^{cd}_{τνφ} R_{t+τ, u+ν, v+φ}(α, R_PQ)` with the reduced
//! exponent `α = pq / (p + q)` (Helgaker 9.9.33). With a unit partner for
//! `b` or `d` (see [`super::pairs::single_shell_pairs`]) the same code
//! yields the two- and three-centre integrals of density fitting.
//!
//! Range separation replaces `α` by `α ω² / (α + ω²)` in the Hermite
//! integrals and scales the prefactor by `sqrt(α' / α)` (ADR-0008).

use std::f64::consts::PI;

use rayon::prelude::*;

use super::ShellTable;
use super::cartesian::{
    cartesian_components, n_cartesian, spherical_transformation, transform_index,
};
use super::coulomb::HermiteWorkspace;
use super::pairs::{PrimitivePair, primitive_pairs, single_shell_pairs};
use crate::Error;

/// Two-centre Coulomb integrals `(P | Q(r − T))` for each translation `T`
/// of the ket, as a row-major `[n_translations, n_bra, n_ket]` array.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if `omega` is not positive or a
/// translation is not finite.
pub fn coulomb_two_center(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
    omega: Option<f64>,
) -> Result<Vec<f64>, Error> {
    check_omega(omega)?;
    check_finite(translations)?;
    let n_bra = bra.n_functions();
    let n_ket = ket.n_functions();
    let max_l = bra.max_angular_momentum().max(ket.max_angular_momentum());
    let tables = Tables::new(max_l);
    let ket_offsets = ket.function_offsets();
    let mut out = vec![0.0; translations.len() * n_bra * n_ket];
    for (it, translation) in translations.iter().enumerate() {
        let slabs: Vec<Vec<f64>> = (0..bra.n_shells())
            .into_par_iter()
            .map(|ia| {
                let a = bra.shell(ia);
                let la = a.angular_momentum;
                let bra_pairs = single_shell_pairs(&a, a.center);
                let mut workspace = HermiteWorkspace::default();
                let mut slab = vec![0.0; (2 * la + 1) * n_ket];
                for (ic, &ket_offset) in ket_offsets.iter().enumerate() {
                    let c = ket.shell(ic);
                    let lc = c.angular_momentum;
                    let c_center: [f64; 3] =
                        std::array::from_fn(|axis| c.center[axis] + translation[axis]);
                    let ket_pairs = single_shell_pairs(&c, c_center);
                    let mut block = vec![0.0; n_cartesian(la) * n_cartesian(lc)];
                    coulomb_block(
                        &bra_pairs,
                        [la, 0],
                        &ket_pairs,
                        [lc, 0],
                        omega,
                        &tables,
                        &mut workspace,
                        &mut block,
                    );
                    let spherical = tables.to_spherical(&block, [Some(la), None, Some(lc), None]);
                    for (ma, row) in spherical.chunks(2 * lc + 1).enumerate() {
                        slab[ma * n_ket + ket_offset..ma * n_ket + ket_offset + row.len()]
                            .copy_from_slice(row);
                    }
                }
                slab
            })
            .collect();
        let image = &mut out[it * n_bra * n_ket..(it + 1) * n_bra * n_ket];
        for (offset, slab) in bra.function_offsets().into_iter().zip(slabs) {
            image[offset * n_ket..offset * n_ket + slab.len()].copy_from_slice(&slab);
        }
    }
    Ok(out)
}

/// Three-centre Coulomb integrals `(P(r − T_P) | μ ν(r − T_ν))` for each
/// pair of translations, as a row-major `[n_images, n_aux, n_bra, n_ket]`
/// array; `translations[image] = [T_P, T_ν]`.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if `omega` is not positive or a
/// translation is not finite.
pub fn coulomb_three_center(
    aux: &ShellTable,
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[[f64; 3]; 2]],
    omega: Option<f64>,
) -> Result<Vec<f64>, Error> {
    check_omega(omega)?;
    for pair in translations {
        check_finite(pair)?;
    }
    let (n_aux, n_bra, n_ket) = (aux.n_functions(), bra.n_functions(), ket.n_functions());
    let max_l = aux
        .max_angular_momentum()
        .max(bra.max_angular_momentum())
        .max(ket.max_angular_momentum());
    let tables = Tables::new(max_l);
    let bra_offsets = bra.function_offsets();
    let ket_offsets = ket.function_offsets();
    let mut out = vec![0.0; translations.len() * n_aux * n_bra * n_ket];
    for (it, [aux_translation, ket_translation]) in translations.iter().enumerate() {
        let slabs: Vec<Vec<f64>> = (0..aux.n_shells())
            .into_par_iter()
            .map(|ip| {
                let p_shell = aux.shell(ip);
                let lp = p_shell.angular_momentum;
                let p_center: [f64; 3] =
                    std::array::from_fn(|axis| p_shell.center[axis] + aux_translation[axis]);
                let aux_pairs = single_shell_pairs(&p_shell, p_center);
                let mut workspace = HermiteWorkspace::default();
                let mut slab = vec![0.0; (2 * lp + 1) * n_bra * n_ket];
                for (ia, &bra_offset) in bra_offsets.iter().enumerate() {
                    let a = bra.shell(ia);
                    for (ib, &ket_offset) in ket_offsets.iter().enumerate() {
                        let b = ket.shell(ib);
                        let b_center: [f64; 3] =
                            std::array::from_fn(|axis| b.center[axis] + ket_translation[axis]);
                        let (la, lb) = (a.angular_momentum, b.angular_momentum);
                        let pairs = primitive_pairs(&a, &b, b_center, 0);
                        let mut block =
                            vec![0.0; n_cartesian(la) * n_cartesian(lb) * n_cartesian(lp)];
                        coulomb_block(
                            &pairs,
                            [la, lb],
                            &aux_pairs,
                            [lp, 0],
                            omega,
                            &tables,
                            &mut workspace,
                            &mut block,
                        );
                        // spherical[ma][mb][mp] → slab[mp][μ][ν]
                        let spherical =
                            tables.to_spherical(&block, [Some(la), Some(lb), Some(lp), None]);
                        let (nsb, nsp) = (2 * lb + 1, 2 * lp + 1);
                        for (index, value) in spherical.iter().enumerate() {
                            let mp = index % nsp;
                            let mb = (index / nsp) % nsb;
                            let ma = index / (nsp * nsb);
                            slab[(mp * n_bra + bra_offset + ma) * n_ket + ket_offset + mb] = *value;
                        }
                    }
                }
                slab
            })
            .collect();
        let image = &mut out[it * n_aux * n_bra * n_ket..(it + 1) * n_aux * n_bra * n_ket];
        for (offset, slab) in aux.function_offsets().into_iter().zip(slabs) {
            let start = offset * n_bra * n_ket;
            image[start..start + slab.len()].copy_from_slice(&slab);
        }
    }
    Ok(out)
}

/// Per-angular-momentum tables shared by the kernels of one call.
pub(crate) struct Tables {
    pub components: Vec<Vec<[usize; 3]>>,
    pub transforms: Vec<Vec<f64>>,
}

impl Tables {
    pub(crate) fn new(max_l: usize) -> Self {
        Self {
            components: (0..=max_l).map(cartesian_components).collect(),
            transforms: (0..=max_l).map(spherical_transformation).collect(),
        }
    }

    /// Transform a row-major Cartesian block with index layout
    /// `[a, b, c, d]` to spherical functions. An index of `None` is a unit
    /// partner (see [`single_shell_pairs`]): one component, left untouched.
    pub(crate) fn to_spherical(&self, block: &[f64], ls: [Option<usize>; 4]) -> Vec<f64> {
        let mut data = block.to_vec();
        let mut shape: [usize; 4] = std::array::from_fn(|i| ls[i].map_or(1, n_cartesian));
        for (index, l) in ls.iter().enumerate() {
            let Some(l) = l else {
                continue;
            };
            let left: usize = shape[..index].iter().product();
            let right: usize = shape[index + 1..].iter().product();
            data = transform_index(&data, left, shape[index], right, &self.transforms[*l]);
            shape[index] = 2 * l + 1;
        }
        data
    }
}

/// Accumulate `(ab|cd)` over all primitive pairs into `out`, laid out as
/// `[cart_a, cart_b, cart_c, cart_d]`; `ls = [l_a, l_b]` and `[l_c, l_d]`
/// name the Cartesian component sets.
#[allow(
    clippy::too_many_arguments,
    reason = "inner kernel; every argument is a distinct table"
)]
pub(crate) fn coulomb_block(
    bra: &[PrimitivePair],
    ls_bra: [usize; 2],
    ket: &[PrimitivePair],
    ls_ket: [usize; 2],
    omega: Option<f64>,
    tables: &Tables,
    workspace: &mut HermiteWorkspace,
    out: &mut [f64],
) {
    let (comp_a, comp_b) = (&tables.components[ls_bra[0]], &tables.components[ls_bra[1]]);
    let (comp_c, comp_d) = (&tables.components[ls_ket[0]], &tables.components[ls_ket[1]]);
    let order_bra = ls_bra[0] + ls_bra[1];
    let order_ket = ls_ket[0] + ls_ket[1];
    let order = order_bra + order_ket;
    let n_t = order_bra + 1;
    let n_cd = comp_c.len() * comp_d.len();
    let mut intermediate = std::mem::take(&mut workspace.intermediate);
    intermediate.clear();
    intermediate.resize(n_cd * n_t * n_t * n_t, 0.0);
    for pair_bra in bra {
        for pair_ket in ket {
            let (p, q) = (pair_bra.p, pair_ket.p);
            let alpha = p * q / (p + q);
            let attenuated = omega.map_or(alpha, |w| alpha * w * w / (alpha + w * w));
            let prefactor = pair_bra.coefficient * pair_ket.coefficient * 2.0 * PI.powf(2.5)
                / (p * q * (p + q).sqrt())
                * (attenuated / alpha).sqrt();
            let pq: [f64; 3] =
                std::array::from_fn(|axis| pair_bra.center[axis] - pair_ket.center[axis]);
            let r = workspace.compute(order, attenuated, pq);
            // X[cd][tuv] = Σ_{τνφ} (−1)^{τ+ν+φ} E^{cd}_{τνφ} R_{t+τ, u+ν, v+φ}
            for (icd, (cc, cd)) in comp_c
                .iter()
                .flat_map(|cc| comp_d.iter().map(move |cd| (cc, cd)))
                .enumerate()
            {
                let slot = &mut intermediate[icd * n_t * n_t * n_t..(icd + 1) * n_t * n_t * n_t];
                for t in 0..n_t {
                    for u in 0..n_t {
                        for v in 0..n_t {
                            let mut sum = 0.0;
                            for tau in 0..=cc[0] + cd[0] {
                                for nu in 0..=cc[1] + cd[1] {
                                    for phi in 0..=cc[2] + cd[2] {
                                        let sign =
                                            if (tau + nu + phi) % 2 == 0 { 1.0 } else { -1.0 };
                                        sum += sign
                                            * pair_ket.product(cc, cd, [tau, nu, phi])
                                            * r.get(t + tau, u + nu, v + phi);
                                    }
                                }
                            }
                            slot[(t * n_t + u) * n_t + v] = sum;
                        }
                    }
                }
            }
            for (iab, (ca, cb)) in comp_a
                .iter()
                .flat_map(|ca| comp_b.iter().map(move |cb| (ca, cb)))
                .enumerate()
            {
                for icd in 0..n_cd {
                    let slot = &intermediate[icd * n_t * n_t * n_t..(icd + 1) * n_t * n_t * n_t];
                    let mut value = 0.0;
                    for t in 0..=ca[0] + cb[0] {
                        for u in 0..=ca[1] + cb[1] {
                            for v in 0..=ca[2] + cb[2] {
                                value += pair_bra.product(ca, cb, [t, u, v])
                                    * slot[(t * n_t + u) * n_t + v];
                            }
                        }
                    }
                    out[iab * n_cd + icd] += prefactor * value;
                }
            }
        }
    }
    workspace.intermediate = intermediate;
}

pub(crate) fn check_omega(omega: Option<f64>) -> Result<(), Error> {
    if omega.is_some_and(|w| w.is_nan() || w <= 0.0) {
        return Err(Error::InvalidArgument(
            "the range-separation parameter omega is positive".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn check_finite(vectors: &[[f64; 3]]) -> Result<(), Error> {
    if vectors
        .iter()
        .all(|vector| vector.iter().all(|x| x.is_finite()))
    {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "translations and positions are finite".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrals::nuclear_attraction;

    /// One normalized s primitive of exponent `a` at `center`.
    fn s_shell(a: f64, center: [f64; 3]) -> ShellTable {
        let norm = (2.0 * a / PI).powf(0.75) * (4.0 * PI).sqrt();
        ShellTable::new(vec![0], vec![center], vec![0, 1], vec![a], vec![norm]).expect("valid")
    }

    #[test]
    fn two_center_s_integral_matches_the_closed_form() {
        let (a, b) = (0.9, 0.4);
        let bra = s_shell(a, [0.0; 3]);
        let ket = s_shell(b, [0.0, 0.0, 1.3]);
        let value = coulomb_two_center(&bra, &ket, &[[0.0; 3]], None).expect("valid")[0];
        let alpha = a * b / (a + b);
        let boys = crate::special::boys_values(0, alpha * 1.69)[0];
        let norms = (2.0 * a / PI).powf(0.75) * (2.0 * b / PI).powf(0.75);
        let expected = norms * 2.0 * PI.powf(2.5) / (a * b * (a + b).sqrt()) * boys;
        assert!((value - expected).abs() < 1e-13);
    }

    #[test]
    fn two_center_is_symmetric_and_translates() {
        let bra = s_shell(0.9, [0.1, 0.2, 0.3]);
        let ket = s_shell(0.4, [0.5, -0.3, 1.0]);
        let forward = coulomb_two_center(&bra, &ket, &[[0.0; 3]], None).expect("valid")[0];
        let backward = coulomb_two_center(&ket, &bra, &[[0.0; 3]], None).expect("valid")[0];
        assert!((forward - backward).abs() < 1e-14);
        let moved = s_shell(0.4, [1.5, -0.3, 1.0]);
        let translated =
            coulomb_two_center(&bra, &ket, &[[1.0, 0.0, 0.0]], None).expect("valid")[0];
        let direct = coulomb_two_center(&bra, &moved, &[[0.0; 3]], None).expect("valid")[0];
        assert!((translated - direct).abs() < 1e-14);
    }

    #[test]
    fn three_center_with_an_s_auxiliary_is_an_attenuated_attraction() {
        // A unit-charge Gaussian of exponent q has the potential erf(√q r)/r;
        // PySCF's normalized s function of exponent q is that charge times
        // (2π/q)^{3/4}... in our convention the s function with unit norm is
        // the unit charge divided by (q / 2π)^{3/4}.
        let q = 0.7;
        let aux = s_shell(q, [0.3, -0.2, 0.5]);
        let bra = s_shell(1.1, [0.0; 3]);
        let ket = s_shell(0.3, [1.0, 0.0, 0.0]);
        let three = coulomb_three_center(&aux, &bra, &ket, &[[[0.0; 3], [0.0; 3]]], None)
            .expect("valid")[0];
        let attraction = nuclear_attraction(
            &bra,
            &ket,
            &[[0.0; 3]],
            &[1.0],
            &[[0.3, -0.2, 0.5]],
            Some(q.sqrt()),
        )
        .expect("valid")[0];
        let expected = -attraction / (q / (2.0 * PI)).powf(0.75);
        assert!((three - expected).abs() < 1e-13, "{three} vs {expected}");
    }

    #[test]
    fn attenuated_two_center_is_bounded_by_the_full_kernel() {
        let bra = s_shell(0.9, [0.0; 3]);
        let ket = s_shell(0.4, [0.0, 0.0, 1.3]);
        let full = coulomb_two_center(&bra, &ket, &[[0.0; 3]], None).expect("valid")[0];
        let long_range = coulomb_two_center(&bra, &ket, &[[0.0; 3]], Some(0.5)).expect("valid")[0];
        let nearly = coulomb_two_center(&bra, &ket, &[[0.0; 3]], Some(1e4)).expect("valid")[0];
        assert!(long_range < full && long_range > 0.0);
        assert!((nearly - full).abs() < 1e-7);
    }

    #[test]
    fn rejects_bad_arguments() {
        let shell = s_shell(1.0, [0.0; 3]);
        assert!(coulomb_two_center(&shell, &shell, &[[0.0; 3]], Some(-1.0)).is_err());
        assert!(coulomb_two_center(&shell, &shell, &[[f64::NAN; 3]], None).is_err());
        assert!(
            coulomb_three_center(
                &shell,
                &shell,
                &shell,
                &[[[0.0; 3], [f64::INFINITY; 3]]],
                None
            )
            .is_err()
        );
    }
}

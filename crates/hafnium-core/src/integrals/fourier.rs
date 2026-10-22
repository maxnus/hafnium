//! Fourier transforms of basis functions and of basis-function pairs with
//! the kernel `exp(−i G·r)` (CONVENTIONS.md §3), analytic in the Hermite
//! basis (ADR-0008).
//!
//! The transform of the Hermite Gaussian `Λ_{tuv}(r; p, P)` is
//! `(π/p)^{3/2} exp(−G²/4p) exp(−i G·P) (−i G_x)^t (−i G_y)^u (−i G_z)^v`,
//! so the transform of a pair factorizes along the axes:
//! `ρ̃_ab(G) = prefactor Π_axis Σ_t E^{ij}_t (−i G_axis)^t`.

use std::f64::consts::PI;

use num_complex::Complex64;
use rayon::prelude::*;

use super::ShellTable;
use super::cartesian::{n_cartesian, transform_index};
use super::pairs::{PrimitivePairs, primitive_pairs, single_shell_pairs, transform_weights};
use super::two_electron::{Tables, check_finite, check_threshold};
use crate::Error;

/// `ρ̃_P(G) = ∫ χ_P(r) exp(−i G·r) dr` for every function of `table` and
/// every vector of `g_vectors` (inverse Bohr), as a row-major
/// `[n_g, n_functions]` array.
///
/// A function translated by `T` has its transform multiplied by
/// `exp(−i G·T)`, so no translations are taken.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if a vector is not finite.
pub fn fourier_transform(
    table: &ShellTable,
    g_vectors: &[[f64; 3]],
) -> Result<Vec<Complex64>, Error> {
    check_finite(g_vectors, "G vectors")?;
    let tables = Tables::new(table.max_angular_momentum());
    let (n_g, n) = (g_vectors.len(), table.n_functions());
    let offsets = table.function_offsets();
    // One slab per shell, laid out `[g][f]` over the shell's functions.
    let slabs: Vec<Vec<Complex64>> = (0..table.n_shells())
        .into_par_iter()
        .map(|ia| {
            let a = table.shell(ia);
            let la = a.angular_momentum;
            let pairs = single_shell_pairs(&a, a.center);
            let mut block = vec![Complex64::default(); n_g * a.n_contractions * n_cartesian(la)];
            let unscreened = Screen {
                threshold: 0.0,
                row_norms: 1.0,
            };
            accumulate(&pairs, [la, 0], g_vectors, &tables, unscreened, &mut block);
            // The unit partner is one untouched component on the right.
            transform_index(
                &block,
                n_g * a.n_contractions,
                n_cartesian(la),
                1,
                &tables.transforms[la],
            )
        })
        .collect();
    let mut out = vec![Complex64::default(); n_g * n];
    for (ia, slab) in slabs.iter().enumerate() {
        let width = table.shell(ia).n_functions();
        for (ig, row) in slab.chunks(width).enumerate() {
            let start = ig * n + offsets[ia];
            out[start..start + width].copy_from_slice(row);
        }
    }
    Ok(out)
}

/// `ρ̃_μν(G) = ∫ φ_μ(r) φ_ν(r − T) exp(−i G·r) dr` for each translation
/// `T` of the ket and every vector of `g_vectors` (inverse Bohr), as a
/// row-major `[n_translations, n_g, n_bra, n_ket]` array.
///
/// At `G = 0` the transform is the overlap matrix of the same translation.
///
/// A pair whose transform is bounded below `threshold` at every `G` is
/// left at zero (ADR-0018); a threshold of zero evaluates everything.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if a translation or a vector is not
/// finite, or `threshold` is negative or not finite.
pub fn pair_fourier_transform(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
    g_vectors: &[[f64; 3]],
    threshold: f64,
) -> Result<Vec<Complex64>, Error> {
    check_finite(translations, "translations")?;
    check_finite(g_vectors, "G vectors")?;
    check_threshold(threshold)?;
    let max_l = bra.max_angular_momentum().max(ket.max_angular_momentum());
    let tables = Tables::new(max_l);
    let (n_g, n_bra, n_ket) = (g_vectors.len(), bra.n_functions(), ket.n_functions());
    let bra_offsets = bra.function_offsets();
    let ket_offsets = ket.function_offsets();
    let mut out = vec![Complex64::default(); translations.len() * n_g * n_bra * n_ket];
    for (it, translation) in translations.iter().enumerate() {
        // One slab per bra shell, laid out `[g][f_a][ν]`.
        let slabs: Vec<Vec<Complex64>> = (0..bra.n_shells())
            .into_par_iter()
            .map(|ia| {
                let a = bra.shell(ia);
                let la = a.angular_momentum;
                let (ka, nsa, nca) = (a.n_contractions, 2 * la + 1, n_cartesian(la));
                let nfa = a.n_functions();
                let mut slab = vec![Complex64::default(); n_g * nfa * n_ket];
                for (ib, &ket_offset) in ket_offsets.iter().enumerate() {
                    let b = ket.shell(ib);
                    let lb = b.angular_momentum;
                    let (kb, ncb) = (b.n_contractions, n_cartesian(lb));
                    let nfb = b.n_functions();
                    let b_center: [f64; 3] =
                        std::array::from_fn(|axis| b.center[axis] + translation[axis]);
                    let pairs = primitive_pairs(&a, &b, b_center, 0);
                    let mut block = vec![Complex64::default(); n_g * ka * nca * kb * ncb];
                    let screen = Screen {
                        threshold,
                        row_norms: tables.row_norms[la] * tables.row_norms[lb],
                    };
                    accumulate(&pairs, [la, lb], g_vectors, &tables, screen, &mut block);
                    // [g][k_a][cart_a][k_b][cart_b] → [g][f_a][f_b]
                    let spherical = transform_index(
                        &transform_index(&block, n_g * ka, nca, kb * ncb, &tables.transforms[la]),
                        n_g * ka * nsa * kb,
                        ncb,
                        1,
                        &tables.transforms[lb],
                    );
                    for (index, row) in spherical.chunks(nfb).enumerate() {
                        let (ig, fa) = (index / nfa, index % nfa);
                        let start = (ig * nfa + fa) * n_ket + ket_offset;
                        slab[start..start + nfb].copy_from_slice(row);
                    }
                }
                slab
            })
            .collect();
        let image = &mut out[it * n_g * n_bra * n_ket..(it + 1) * n_g * n_bra * n_ket];
        for (ia, slab) in slabs.iter().enumerate() {
            let nfa = bra.shell(ia).n_functions();
            for (index, row) in slab.chunks(n_ket).enumerate() {
                let (ig, fa) = (index / nfa, index % nfa);
                let start = (ig * n_bra + bra_offsets[ia] + fa) * n_ket;
                image[start..start + n_ket].copy_from_slice(row);
            }
        }
    }
    Ok(out)
}

/// The screening of a shell pair: the threshold and the amplification of
/// a Cartesian primitive value by the spherical transformations.
#[derive(Clone, Copy)]
struct Screen {
    threshold: f64,
    row_norms: f64,
}

/// Add the transform of every primitive pair to `block`, laid out
/// `[g][k_a][cart_a][k_b][cart_b]`; `ls = [l_a, l_b]`. A pair whose
/// transform is bounded below the threshold at every `G` is skipped: the
/// transform of a Cartesian pair is at most `∫ |φ_a φ_b| dr`, which the
/// weights of [`transform_weights`] bound (ADR-0018).
fn accumulate(
    pairs: &PrimitivePairs,
    ls: [usize; 2],
    g_vectors: &[[f64; 3]],
    tables: &Tables,
    screen: Screen,
    block: &mut [Complex64],
) {
    let (comp_a, comp_b) = (&tables.components[ls[0]], &tables.components[ls[1]]);
    let order = ls[0] + ls[1];
    let n_ab = comp_a.len() * comp_b.len();
    let per_g = pairs.n_contracted() * n_ab;
    let (n_i, n_j) = (ls[0] + 1, ls[1] + 1);
    // sums[axis][i][j] = Σ_t E^{ij}_t (−i G_axis)^t
    let mut sums = vec![Complex64::default(); 3 * n_i * n_j];
    let mut powers = vec![Complex64::default(); order + 1];
    let mut values = vec![Complex64::default(); n_ab];
    for (index, pair) in pairs.pairs.iter().enumerate() {
        let scale = (PI / pair.p).powf(1.5);
        if screen.threshold > 0.0 {
            let weights = transform_weights(order, pair.p);
            let bound = scale
                * screen.row_norms
                * pairs.coefficient_max(index)
                * pair.envelope(comp_a, comp_b, &weights);
            if bound < screen.threshold {
                continue;
            }
        }
        for (ig, g) in g_vectors.iter().enumerate() {
            let g_squared: f64 = g.iter().map(|x| x * x).sum();
            let g_dot_p: f64 = (0..3).map(|axis| g[axis] * pair.center[axis]).sum();
            let prefactor = scale * (-g_squared / (4.0 * pair.p)).exp() * Complex64::cis(-g_dot_p);
            for axis in 0..3 {
                powers[0] = Complex64::new(1.0, 0.0);
                for t in 1..=order {
                    powers[t] = powers[t - 1] * Complex64::new(0.0, -g[axis]);
                }
                let e = &pair.hermite[axis];
                for i in 0..n_i {
                    for j in 0..n_j {
                        let sum = (0..=i + j)
                            .map(|t| powers[t] * e.get(i, j, t))
                            .fold(Complex64::default(), |acc, x| acc + x);
                        sums[(axis * n_i + i) * n_j + j] = sum;
                    }
                }
            }
            let at = |axis: usize, i: usize, j: usize| sums[(axis * n_i + i) * n_j + j];
            for (iab, (ca, cb)) in comp_a
                .iter()
                .flat_map(|ca| comp_b.iter().map(move |cb| (ca, cb)))
                .enumerate()
            {
                values[iab] =
                    prefactor * at(0, ca[0], cb[0]) * at(1, ca[1], cb[1]) * at(2, ca[2], cb[2]);
            }
            pairs.accumulate(
                index,
                &values,
                comp_a.len(),
                comp_b.len(),
                &mut block[ig * per_g..(ig + 1) * per_g],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrals::overlap;
    use crate::integrals::test_tables::{
        general_table, s_shell, sp_table, sp_table_at, split_table,
    };

    #[test]
    fn general_contractions_match_split_shells() {
        let (general, split) = (general_table(), split_table());
        let g_vectors = [[0.0; 3], [0.4, -0.3, 0.2], [1.1, 0.5, -0.7]];
        let single_general = fourier_transform(&general, &g_vectors).expect("valid");
        let single_split = fourier_transform(&split, &g_vectors).expect("valid");
        assert_eq!(single_general.len(), single_split.len());
        for (g, s) in single_general.iter().zip(&single_split) {
            assert!(close(*g, *s));
        }
        let translations = [ZERO, [0.2, 0.1, -0.4]];
        let pair_general = pair_fourier_transform(&general, &split, &translations, &g_vectors, 0.0)
            .expect("valid");
        let pair_split =
            pair_fourier_transform(&split, &split, &translations, &g_vectors, 0.0).expect("valid");
        assert_eq!(pair_general.len(), pair_split.len());
        for (g, s) in pair_general.iter().zip(&pair_split) {
            assert!(close(*g, *s));
        }
    }

    const ZERO: [f64; 3] = [0.0; 3];

    fn close(x: Complex64, y: Complex64) -> bool {
        (x - y).norm() < 1e-13
    }

    #[test]
    fn s_transform_matches_the_closed_form() {
        let (a, center) = (0.8, [0.1, -0.4, 0.7]);
        let g = [0.3, -0.2, 0.5];
        let value = fourier_transform(&s_shell(a, center), &[g]).expect("valid")[0];
        let g_squared: f64 = g.iter().map(|x| x * x).sum();
        let phase = Complex64::new(0.0, -(0..3).map(|k| g[k] * center[k]).sum::<f64>()).exp();
        let expected =
            (2.0 * a / PI).powf(0.75) * (PI / a).powf(1.5) * (-g_squared / (4.0 * a)).exp() * phase;
        assert!(close(value, expected), "{value} vs {expected}");
    }

    #[test]
    fn p_transform_is_the_derivative_of_the_s_transform() {
        // ∫ x exp(−a r²) exp(−i G·r) dr = i ∂/∂G_x of the s transform
        // = (−i G_x / 2a) (π/a)^{3/2} exp(−G²/4a); the p_x function carries
        // sqrt(3/4π) and the phase of its centre.
        let (a, center) = (0.6, [0.2, 0.3, -0.1]);
        let g = [0.7, -0.4, 0.2];
        let table = ShellTable::new(
            vec![1],
            vec![center],
            vec![0, 1],
            vec![a],
            vec![1.0],
            vec![1],
        )
        .expect("valid");
        let values = fourier_transform(&table, &[g]).expect("valid");
        let g_squared: f64 = g.iter().map(|x| x * x).sum();
        let phase = Complex64::new(0.0, -(0..3).map(|k| g[k] * center[k]).sum::<f64>()).exp();
        let radial =
            (PI / a).powf(1.5) * (-g_squared / (4.0 * a)).exp() * (3.0 / (4.0 * PI)).sqrt();
        for axis in 0..3 {
            let expected = Complex64::new(0.0, -g[axis] / (2.0 * a)) * radial * phase;
            assert!(
                close(values[axis], expected),
                "{} vs {expected}",
                values[axis]
            );
        }
    }

    #[test]
    fn transforms_at_zero_are_overlaps_and_integrals() {
        let table = sp_table();
        let translation = [0.3, -0.5, 0.2];
        let pair = pair_fourier_transform(&table, &table, &[ZERO, translation], &[ZERO], 0.0)
            .expect("valid");
        let s = overlap(&table, &table, &[ZERO, translation]);
        for (p, s) in pair.iter().zip(&s) {
            assert!(close(*p, Complex64::new(*s, 0.0)));
        }
        // ∫ p = 0 and ∫ s = Σ c (π/a)^{3/2} / sqrt(4π), the l = 0 harmonic
        // being 1 / sqrt(4π).
        let single = fourier_transform(&table, &[ZERO]).expect("valid");
        assert_eq!(single.len(), table.n_functions());
        let s_integral =
            (0.7 * (PI / 1.3f64).powf(1.5) + 0.5 * (PI / 0.4f64).powf(1.5)) / (4.0 * PI).sqrt();
        assert!(close(single[0], Complex64::new(s_integral, 0.0)));
        for value in &single[1..4] {
            assert!(value.norm() < 1e-14);
        }
    }

    #[test]
    fn pair_transform_conjugates_and_translates() {
        let table = sp_table();
        let n = table.n_functions();
        let g = [0.4, 0.9, -0.3];
        let minus_g = [-0.4, -0.9, 0.3];
        let translation = [0.6, -0.2, 0.5];
        // Two images and two vectors: `[image][g][u][v]`.
        let both = pair_fourier_transform(&table, &table, &[ZERO, translation], &[g, minus_g], 0.0)
            .expect("valid");
        let at =
            |image: usize, ig: usize, u: usize, v: usize| both[((image * 2 + ig) * n + u) * n + v];
        for u in 0..n {
            for v in 0..n {
                for image in 0..2 {
                    assert!(close(at(image, 0, u, v).conj(), at(image, 1, u, v)));
                }
                // Without a translation the pair transform is symmetric.
                assert!(close(at(0, 0, u, v), at(0, 0, v, u)));
            }
        }
        // The ket translated by T is the ket table moved by T.
        let moved = sp_table_at(translation);
        let direct =
            pair_fourier_transform(&table, &moved, &[ZERO], &[g, minus_g], 0.0).expect("valid");
        for (b, d) in both[2 * n * n..].iter().zip(&direct) {
            assert!(close(*b, *d));
        }
        // Swapping the pair: ρ̃_{a b_T}(G) = exp(−i G·T) ρ̃_{b a_{−T}}(G).
        let minus_t: [f64; 3] = std::array::from_fn(|k| -translation[k]);
        let swapped = pair_fourier_transform(&table, &table, &[minus_t], &[g], 0.0).expect("valid");
        let phase = Complex64::cis(-(0..3).map(|k| g[k] * translation[k]).sum::<f64>());
        for u in 0..n {
            for v in 0..n {
                assert!(close(at(1, 0, u, v), phase * swapped[v * n + u]));
            }
        }
    }

    #[test]
    fn pair_transform_handles_different_tables() {
        // (s | s p s p) against the matching block of (s s p s p | s s p s p).
        let single = s_shell(0.9, [0.0; 3]);
        let table = sp_table();
        let n = table.n_functions();
        let g = [0.4, 0.9, -0.3];
        let mixed = pair_fourier_transform(&single, &table, &[ZERO], &[g], 0.0).expect("valid");
        assert_eq!(mixed.len(), n);
        // The same s function as the first entry of a combined table.
        let combined = ShellTable::new(
            vec![0, 0, 1, 0, 1],
            vec![
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
                [1.2, -0.3, 0.4],
                [1.2, -0.3, 0.4],
            ],
            vec![0, 1, 3, 5, 7, 9],
            vec![0.9, 1.3, 0.4, 0.9, 0.3, 1.1, 0.35, 0.8, 0.25],
            vec![
                (2.0 * 0.9 / PI).powf(0.75) * (4.0 * PI).sqrt(),
                0.7,
                0.5,
                0.6,
                0.4,
                0.8,
                0.3,
                0.5,
                0.5,
            ],
            vec![1; 5],
        )
        .expect("valid");
        let full = pair_fourier_transform(&combined, &combined, &[ZERO], &[g], 0.0).expect("valid");
        for v in 0..n {
            assert!(close(mixed[v], full[v + 1]));
        }
    }

    #[test]
    fn rejects_non_finite_vectors() {
        let table = sp_table();
        assert!(fourier_transform(&table, &[[f64::NAN; 3]]).is_err());
        assert!(
            pair_fourier_transform(&table, &table, &[ZERO], &[[f64::INFINITY; 3]], 0.0).is_err()
        );
        assert!(pair_fourier_transform(&table, &table, &[[f64::NAN; 3]], &[ZERO], 0.0).is_err());
        assert!(pair_fourier_transform(&table, &table, &[ZERO], &[ZERO], -1.0).is_err());
    }

    #[test]
    fn screening_neglects_only_pairs_below_the_threshold() {
        let table = general_table();
        let translations = [[0.0; 3], [6.0, -2.0, 1.0], [60.0, 0.0, 0.0]];
        let g_vectors = [[0.0; 3], [0.4, -0.3, 0.2], [2.0, 1.0, -1.5]];
        let threshold = 1e-10;
        let exact =
            pair_fourier_transform(&table, &table, &translations, &g_vectors, 0.0).expect("valid");
        let screened = pair_fourier_transform(&table, &table, &translations, &g_vectors, threshold)
            .expect("valid");
        let largest = exact
            .iter()
            .zip(&screened)
            .map(|(e, s)| (e - s).norm())
            .fold(0.0_f64, f64::max);
        assert!(largest < 10.0 * threshold, "{largest}");
        let per_image = g_vectors.len() * table.n_functions() * table.n_functions();
        assert!(screened[2 * per_image..].iter().all(|v| v.norm() == 0.0));
        assert!(exact[2 * per_image..].iter().any(|v| v.norm() > 0.0));
        let nothing =
            pair_fourier_transform(&table, &table, &translations, &g_vectors, 1e6).expect("valid");
        assert!(nothing.iter().all(|v| v.norm() == 0.0));
    }
}

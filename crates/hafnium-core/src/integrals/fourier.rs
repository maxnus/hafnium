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
use super::pairs::{PrimitivePair, primitive_pairs, single_shell_pairs};
use super::two_electron::{Tables, check_finite};
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
    check_finite(g_vectors)?;
    let tables = Tables::new(table.max_angular_momentum());
    let (n_g, n) = (g_vectors.len(), table.n_functions());
    let offsets = table.function_offsets();
    // One slab per shell, laid out `[g][m]`.
    let slabs: Vec<Vec<Complex64>> = (0..table.n_shells())
        .into_par_iter()
        .map(|ia| {
            let a = table.shell(ia);
            let la = a.angular_momentum;
            let pairs = single_shell_pairs(&a, a.center);
            let mut block = vec![Complex64::default(); n_g * n_cartesian(la)];
            accumulate(&pairs, [la, 0], g_vectors, &tables, &mut block);
            // The unit partner is one untouched component on the right.
            transform_index(&block, n_g, n_cartesian(la), 1, &tables.transforms[la])
        })
        .collect();
    let mut out = vec![Complex64::default(); n_g * n];
    for (ia, slab) in slabs.iter().enumerate() {
        let width = 2 * table.shell(ia).angular_momentum + 1;
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
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if a translation or a vector is not
/// finite.
pub fn pair_fourier_transform(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
    g_vectors: &[[f64; 3]],
) -> Result<Vec<Complex64>, Error> {
    check_finite(translations)?;
    check_finite(g_vectors)?;
    let max_l = bra.max_angular_momentum().max(ket.max_angular_momentum());
    let tables = Tables::new(max_l);
    let (n_g, n_bra, n_ket) = (g_vectors.len(), bra.n_functions(), ket.n_functions());
    let bra_offsets = bra.function_offsets();
    let ket_offsets = ket.function_offsets();
    let mut out = vec![Complex64::default(); translations.len() * n_g * n_bra * n_ket];
    for (it, translation) in translations.iter().enumerate() {
        // One slab per bra shell, laid out `[g][ma][ν]`.
        let slabs: Vec<Vec<Complex64>> = (0..bra.n_shells())
            .into_par_iter()
            .map(|ia| {
                let a = bra.shell(ia);
                let la = a.angular_momentum;
                let (nsa, nca) = (2 * la + 1, n_cartesian(la));
                let mut slab = vec![Complex64::default(); n_g * nsa * n_ket];
                for (ib, &ket_offset) in ket_offsets.iter().enumerate() {
                    let b = ket.shell(ib);
                    let lb = b.angular_momentum;
                    let (nsb, ncb) = (2 * lb + 1, n_cartesian(lb));
                    let b_center: [f64; 3] =
                        std::array::from_fn(|axis| b.center[axis] + translation[axis]);
                    let pairs = primitive_pairs(&a, &b, b_center, 0);
                    let mut block = vec![Complex64::default(); n_g * nca * ncb];
                    accumulate(&pairs, [la, lb], g_vectors, &tables, &mut block);
                    let spherical = transform_index(
                        &transform_index(&block, n_g, nca, ncb, &tables.transforms[la]),
                        n_g * nsa,
                        ncb,
                        1,
                        &tables.transforms[lb],
                    );
                    for (index, row) in spherical.chunks(nsb).enumerate() {
                        let (ig, ma) = (index / nsa, index % nsa);
                        let start = (ig * nsa + ma) * n_ket + ket_offset;
                        slab[start..start + nsb].copy_from_slice(row);
                    }
                }
                slab
            })
            .collect();
        let image = &mut out[it * n_g * n_bra * n_ket..(it + 1) * n_g * n_bra * n_ket];
        for (ia, slab) in slabs.iter().enumerate() {
            let nsa = 2 * bra.shell(ia).angular_momentum + 1;
            for (index, row) in slab.chunks(n_ket).enumerate() {
                let (ig, ma) = (index / nsa, index % nsa);
                let start = (ig * n_bra + bra_offsets[ia] + ma) * n_ket;
                image[start..start + n_ket].copy_from_slice(row);
            }
        }
    }
    Ok(out)
}

/// Add the transform of every primitive pair to `block`, laid out
/// `[g][cart_a][cart_b]`; `ls = [l_a, l_b]`.
fn accumulate(
    pairs: &[PrimitivePair],
    ls: [usize; 2],
    g_vectors: &[[f64; 3]],
    tables: &Tables,
    block: &mut [Complex64],
) {
    let (comp_a, comp_b) = (&tables.components[ls[0]], &tables.components[ls[1]]);
    let order = ls[0] + ls[1];
    let n_ab = comp_a.len() * comp_b.len();
    let (n_i, n_j) = (ls[0] + 1, ls[1] + 1);
    // sums[axis][i][j] = Σ_t E^{ij}_t (−i G_axis)^t
    let mut sums = vec![Complex64::default(); 3 * n_i * n_j];
    let mut powers = vec![Complex64::default(); order + 1];
    for pair in pairs {
        let scale = pair.coefficient * (PI / pair.p).powf(1.5);
        for (ig, g) in g_vectors.iter().enumerate() {
            let g_squared: f64 = g.iter().map(|x| x * x).sum();
            let g_dot_p: f64 = (0..3).map(|axis| g[axis] * pair.center[axis]).sum();
            let prefactor =
                scale * (-g_squared / (4.0 * pair.p)).exp() * Complex64::new(0.0, -g_dot_p).exp();
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
                block[ig * n_ab + iab] +=
                    prefactor * at(0, ca[0], cb[0]) * at(1, ca[1], cb[1]) * at(2, ca[2], cb[2]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrals::overlap;

    const ZERO: [f64; 3] = [0.0; 3];

    /// One normalized s primitive of exponent `a` at `center`.
    fn s_shell(a: f64, center: [f64; 3]) -> ShellTable {
        let norm = (2.0 * a / PI).powf(0.75) * (4.0 * PI).sqrt();
        ShellTable::new(vec![0], vec![center], vec![0, 1], vec![a], vec![norm]).expect("valid")
    }

    /// An s and a p shell of two primitives each on two centres.
    fn sp_table_at(shift: [f64; 3]) -> ShellTable {
        let centers = [[0.0; 3], [1.2, -0.3, 0.4]];
        let moved = |center: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|axis| center[axis] + shift[axis])
        };
        ShellTable::new(
            vec![0, 1, 0, 1],
            vec![
                moved(centers[0]),
                moved(centers[0]),
                moved(centers[1]),
                moved(centers[1]),
            ],
            vec![0, 2, 4, 6, 8],
            vec![1.3, 0.4, 0.9, 0.3, 1.1, 0.35, 0.8, 0.25],
            vec![0.7, 0.5, 0.6, 0.4, 0.8, 0.3, 0.5, 0.5],
        )
        .expect("valid")
    }

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
        let table =
            ShellTable::new(vec![1], vec![center], vec![0, 1], vec![a], vec![1.0]).expect("valid");
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
        let table = sp_table_at(ZERO);
        let translation = [0.3, -0.5, 0.2];
        let pair =
            pair_fourier_transform(&table, &table, &[ZERO, translation], &[ZERO]).expect("valid");
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
        let table = sp_table_at(ZERO);
        let n = table.n_functions();
        let g = [0.4, 0.9, -0.3];
        let minus_g = [-0.4, -0.9, 0.3];
        let translation = [0.6, -0.2, 0.5];
        let both =
            pair_fourier_transform(&table, &table, &[translation], &[g, minus_g]).expect("valid");
        for u in 0..n {
            for v in 0..n {
                let value = both[u * n + v];
                let negated = both[n * n + u * n + v];
                assert!(close(value.conj(), negated));
            }
        }
        // The ket translated by T is the ket table moved by T.
        let moved = sp_table_at(translation);
        let direct = pair_fourier_transform(&table, &moved, &[ZERO], &[g]).expect("valid");
        for (b, d) in both.iter().zip(&direct) {
            assert!(close(*b, *d));
        }
        // Swapping the pair: ρ̃_{a b_T}(G) = exp(−i G·T) ρ̃_{b a_{−T}}(G).
        let minus_t: [f64; 3] = std::array::from_fn(|k| -translation[k]);
        let swapped = pair_fourier_transform(&table, &table, &[minus_t], &[g]).expect("valid");
        let phase = Complex64::new(0.0, -(0..3).map(|k| g[k] * translation[k]).sum::<f64>()).exp();
        for u in 0..n {
            for v in 0..n {
                assert!(close(both[u * n + v], phase * swapped[v * n + u]));
            }
        }
    }

    #[test]
    fn rejects_non_finite_vectors() {
        let table = sp_table_at(ZERO);
        assert!(fourier_transform(&table, &[[f64::NAN; 3]]).is_err());
        assert!(pair_fourier_transform(&table, &table, &[ZERO], &[[f64::INFINITY; 3]]).is_err());
        assert!(pair_fourier_transform(&table, &table, &[[f64::NAN; 3]], &[ZERO]).is_err());
    }
}

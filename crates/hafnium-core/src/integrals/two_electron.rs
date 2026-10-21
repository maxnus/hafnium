//! Coulomb integrals between two charge distributions in the Hermite basis.
//!
//! `(ab|cd) = (2π^{5/2} / (pq √(p+q))) Σ_{tuv} E^{ab}_{tuv} Σ_{τνφ}
//! (−1)^{τ+ν+φ} E^{cd}_{τνφ} R_{t+τ, u+ν, v+φ}(α, R_PQ)` with the reduced
//! exponent `α = pq / (p + q)` (Helgaker 9.9.33). With a unit partner for
//! `b` or `d` (see [`super::pairs::single_shell_pairs`]) the same code
//! yields the two- and three-centre integrals of density fitting; with
//! real pairs on both sides it yields the four-centre integrals.
//!
//! Range separation replaces `α` by `α ω² / (α + ω²)` in the Hermite
//! integrals and scales the prefactor by `sqrt(α' / α)` (ADR-0008).

use std::f64::consts::PI;
use std::ops::{AddAssign, Mul};

use rayon::prelude::*;

use super::cartesian::{
    cartesian_components, n_cartesian, spherical_transformation, transform_index,
};
use super::coulomb::HermiteWorkspace;
use super::pairs::{PrimitivePair, primitive_pairs, single_shell_pairs};
use super::{ShellRef, ShellTable};
use crate::Error;

const ZERO: [f64; 3] = [0.0; 3];

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
    let n_ket = ket.n_functions();
    let max_l = bra.max_angular_momentum().max(ket.max_angular_momentum());
    let tables = Tables::new(max_l);
    let ket_offsets = ket.function_offsets();
    Ok(assemble_parallel(
        bra,
        n_ket,
        translations,
        |translation, a, slab| {
            let la = a.angular_momentum;
            let bra_pairs = single_shell_pairs(&a, a.center);
            let mut workspace = HermiteWorkspace::default();
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
        },
    ))
}

/// Three-centre Coulomb integrals `(P(r − T_P) | μ ν(r − T_ν))` for each
/// pair of translations, as a row-major `[n_images, n_aux, n_bra, n_ket]`
/// array; `translations[image] = [T_P, T_ν]`.
///
/// When `bra` and `ket` are the same table and `T_ν` is zero, only the
/// shell pairs `μ ≤ ν` are evaluated and the symmetry `(P|μν) = (P|νμ)`
/// fills the rest.
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
    let (n_bra, n_ket) = (bra.n_functions(), ket.n_functions());
    let max_l = aux
        .max_angular_momentum()
        .max(bra.max_angular_momentum())
        .max(ket.max_angular_momentum());
    let tables = Tables::new(max_l);
    let bra_offsets = bra.function_offsets();
    let ket_offsets = ket.function_offsets();
    Ok(assemble_parallel(
        aux,
        n_bra * n_ket,
        translations,
        |[aux_translation, ket_translation], p_shell, slab| {
            let symmetric = bra == ket && same_translation(*ket_translation, ZERO);
            let lp = p_shell.angular_momentum;
            let p_center: [f64; 3] =
                std::array::from_fn(|axis| p_shell.center[axis] + aux_translation[axis]);
            let aux_pairs = single_shell_pairs(&p_shell, p_center);
            let mut workspace = HermiteWorkspace::default();
            for (ia, &bra_offset) in bra_offsets.iter().enumerate() {
                let a = bra.shell(ia);
                let first = if symmetric { ia } else { 0 };
                for (ib, &ket_offset) in ket_offsets.iter().enumerate().skip(first) {
                    let b = ket.shell(ib);
                    let b_center: [f64; 3] =
                        std::array::from_fn(|axis| b.center[axis] + ket_translation[axis]);
                    let (la, lb) = (a.angular_momentum, b.angular_momentum);
                    let pairs = primitive_pairs(&a, &b, b_center, 0);
                    let mut block = vec![0.0; n_cartesian(la) * n_cartesian(lb) * n_cartesian(lp)];
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
                    // spherical[ma][mb][mp] → slab[mp][μ][ν] (and slab[mp][ν][μ])
                    let spherical =
                        tables.to_spherical(&block, [Some(la), Some(lb), Some(lp), None]);
                    let (nsb, nsp) = (2 * lb + 1, 2 * lp + 1);
                    for (index, value) in spherical.iter().enumerate() {
                        let mp = index % nsp;
                        let mb = (index / nsp) % nsb;
                        let ma = index / (nsp * nsb);
                        slab[(mp * n_bra + bra_offset + ma) * n_ket + ket_offset + mb] = *value;
                        if symmetric && ib != ia {
                            slab[(mp * n_bra + ket_offset + mb) * n_ket + bra_offset + ma] = *value;
                        }
                    }
                }
            }
        },
    ))
}

/// Four-centre Coulomb integrals `(a b(r − T_b) | c(r − T_c) d(r − T_d))`
/// for each triple of translations, as a row-major
/// `[n_images, n_a, n_b, n_c, n_d]` array in chemists' notation;
/// `translations[image] = [T_b, T_c, T_d]`.
///
/// Shell quartets whose Schwarz bound `Q_ab Q_cd` (see [`schwarz_bounds`])
/// is below `threshold` are not evaluated and stay zero. Where two tables
/// coincide and the translations permit, only the unique shell quartets are
/// evaluated and the permutational symmetry fills the rest: the bra pair
/// may be swapped when `bra[0] == bra[1]` and `T_b = 0`, the ket pair when
/// `ket[0] == ket[1]` and `T_c = T_d`, and the two pairs when `bra == ket`,
/// `T_c = 0` and `T_d = T_b`.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if `omega` is not positive, a
/// translation is not finite, or `threshold` is negative or not finite.
pub fn coulomb_four_center(
    bra: [&ShellTable; 2],
    ket: [&ShellTable; 2],
    translations: &[[[f64; 3]; 3]],
    omega: Option<f64>,
    threshold: f64,
) -> Result<Vec<f64>, Error> {
    check_omega(omega)?;
    check_threshold(threshold)?;
    for triple in translations {
        check_finite(triple)?;
    }
    let size: usize = output_shape(bra, ket).iter().product();
    let max_l = bra
        .iter()
        .chain(ket.iter())
        .map(|table| table.max_angular_momentum())
        .max()
        .unwrap_or(0);
    let tables = Tables::new(max_l);
    let mut out = vec![0.0; translations.len() * size];
    for (it, &translation) in translations.iter().enumerate() {
        let image = FourCenterImage {
            bra,
            ket,
            translations: translation,
            omega,
            threshold,
            swaps: admitted_swaps(bra, ket, translation),
        };
        image.fill(&tables, &mut out[it * size..(it + 1) * size]);
    }
    Ok(out)
}

/// Shape `[n_a, n_b, n_c, n_d]` of one image of the four-centre integrals.
fn output_shape(bra: [&ShellTable; 2], ket: [&ShellTable; 2]) -> [usize; 4] {
    [
        bra[0].n_functions(),
        bra[1].n_functions(),
        ket[0].n_functions(),
        ket[1].n_functions(),
    ]
}

/// The swaps an image admits: of the bra pair, of the ket pair, and of
/// the two pairs (see [`coulomb_four_center`]).
fn admitted_swaps(
    bra: [&ShellTable; 2],
    ket: [&ShellTable; 2],
    [t_b, t_c, t_d]: [[f64; 3]; 3],
) -> [bool; 3] {
    [
        bra[0] == bra[1] && same_translation(t_b, ZERO),
        ket[0] == ket[1] && same_translation(t_c, t_d),
        bra[0] == ket[0]
            && bra[1] == ket[1]
            && same_translation(t_c, ZERO)
            && same_translation(t_d, t_b),
    ]
}

/// Schwarz bounds `Q_ab = max_{a ∈ A, b ∈ B} sqrt((a b(r − T) | a b(r − T)))`
/// of every shell pair for each translation `T` of the ket, as a row-major
/// `[n_translations, n_bra_shells, n_ket_shells]` array.
///
/// The Cauchy–Schwarz inequality in the Coulomb metric gives
/// `|(ab|cd)| ≤ Q_ab Q_cd` for every integral with the same kernel; the
/// long-range kernel is positive definite as well, so the bound holds with
/// `omega` too.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if `omega` is not positive or a
/// translation is not finite.
pub fn schwarz_bounds(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
    omega: Option<f64>,
) -> Result<Vec<f64>, Error> {
    check_omega(omega)?;
    check_finite(translations)?;
    let max_l = bra.max_angular_momentum().max(ket.max_angular_momentum());
    let tables = Tables::new(max_l);
    let mut out = Vec::with_capacity(translations.len() * bra.n_shells() * ket.n_shells());
    for &translation in translations {
        let pairs = shell_pairs([bra, ket], [ZERO, translation], false, omega, &tables);
        out.extend(pairs.iter().map(|pair| pair.bound));
    }
    Ok(out)
}

/// One image of the four-centre integrals: the tables, their translations
/// and the symmetries the image admits (bra swap, ket swap, pair swap).
struct FourCenterImage<'a> {
    bra: [&'a ShellTable; 2],
    ket: [&'a ShellTable; 2],
    translations: [[f64; 3]; 3],
    omega: Option<f64>,
    threshold: f64,
    /// Derived by [`admitted_swaps`] in [`coulomb_four_center`]; tests set
    /// it directly to force the generic path.
    swaps: [bool; 3],
}

/// Values of evaluated blocks held between a parallel chunk of bra pairs
/// and their placement in the output; bounds the extra memory.
const CHUNK_BUDGET: usize = 1 << 23;

impl FourCenterImage<'_> {
    /// Evaluate the unique shell quartets above the threshold in parallel
    /// over bra pairs and place them, with their symmetric images, in `out`.
    fn fill(&self, tables: &Tables, out: &mut [f64]) {
        let [t_b, t_c, t_d] = self.translations;
        let bra_pairs = shell_pairs(self.bra, [ZERO, t_b], self.swaps[0], self.omega, tables);
        // With the pair swap the ket pairs are the bra pairs, in the same order.
        let ket_storage;
        let ket_pairs: &[ShellPair] = if self.swaps[2] {
            &bra_pairs
        } else {
            ket_storage = shell_pairs(self.ket, [t_c, t_d], self.swaps[1], self.omega, tables);
            &ket_storage
        };
        let shape = output_shape(self.bra, self.ket);
        let ket_size: usize = ket_pairs.iter().map(ShellPair::n_functions).sum();
        let mut start = 0;
        while start < bra_pairs.len() {
            let end = chunk_end(&bra_pairs, start, ket_size);
            let blocks: Vec<Vec<(usize, Vec<f64>)>> = (start..end)
                .into_par_iter()
                .map(|index| {
                    let bra_pair = &bra_pairs[index];
                    let partners = if self.swaps[2] {
                        &ket_pairs[..=index]
                    } else {
                        ket_pairs
                    };
                    let mut workspace = HermiteWorkspace::default();
                    partners
                        .iter()
                        .enumerate()
                        .filter(|(_, ket_pair)| bra_pair.bound * ket_pair.bound >= self.threshold)
                        .map(|(ket_index, ket_pair)| {
                            let block =
                                quartet(bra_pair, ket_pair, self.omega, tables, &mut workspace);
                            (ket_index, block)
                        })
                        .collect()
                })
                .collect();
            for (bra_pair, blocks) in bra_pairs[start..end].iter().zip(&blocks) {
                for (ket_index, block) in blocks {
                    scatter(
                        out,
                        shape,
                        bra_pair,
                        &ket_pairs[*ket_index],
                        block,
                        self.swaps,
                    );
                }
            }
            start = end;
        }
    }
}

/// End of the chunk of bra pairs starting at `start` whose blocks fit the
/// budget; always at least one pair.
fn chunk_end(pairs: &[ShellPair], start: usize, ket_size: usize) -> usize {
    let mut end = start + 1;
    let mut total = pairs[start].n_functions() * ket_size;
    while end < pairs.len() {
        total += pairs[end].n_functions() * ket_size;
        if total > CHUNK_BUDGET {
            break;
        }
        end += 1;
    }
    end
}

/// The spherical block `[ma][mb][mc][md]` of one shell quartet.
fn quartet(
    bra: &ShellPair,
    ket: &ShellPair,
    omega: Option<f64>,
    tables: &Tables,
    workspace: &mut HermiteWorkspace,
) -> Vec<f64> {
    let n_cart = |ls: [usize; 2]| n_cartesian(ls[0]) * n_cartesian(ls[1]);
    let mut block = vec![0.0; n_cart(bra.ls) * n_cart(ket.ls)];
    coulomb_block(
        &bra.primitives,
        bra.ls,
        &ket.primitives,
        ket.ls,
        omega,
        tables,
        workspace,
        &mut block,
    );
    tables.to_spherical(
        &block,
        [
            Some(bra.ls[0]),
            Some(bra.ls[1]),
            Some(ket.ls[0]),
            Some(ket.ls[1]),
        ],
    )
}

/// Write a quartet block into `out` (shape `[n_a, n_b, n_c, n_d]`) at its
/// own position and at every position the admitted swaps map it to. A swap
/// that maps the block onto itself is skipped, so each element is written
/// exactly once.
fn scatter(
    out: &mut [f64],
    shape: [usize; 4],
    bra: &ShellPair,
    ket: &ShellPair,
    block: &[f64],
    swaps: [bool; 3],
) {
    let dims = [
        2 * bra.ls[0] + 1,
        2 * bra.ls[1] + 1,
        2 * ket.ls[0] + 1,
        2 * ket.ls[1] + 1,
    ];
    let offsets = [
        bra.offsets[0],
        bra.offsets[1],
        ket.offsets[0],
        ket.offsets[1],
    ];
    let swaps = [
        swaps[0] && bra.shells[0] != bra.shells[1],
        swaps[1] && ket.shells[0] != ket.shells[1],
        swaps[2] && bra.shells != ket.shells,
    ];
    let index = |[u, v, w, x]: [usize; 4]| ((u * shape[1] + v) * shape[2] + w) * shape[3] + x;
    for (flat, &value) in block.iter().enumerate() {
        let md = flat % dims[3];
        let mc = (flat / dims[3]) % dims[2];
        let mb = (flat / (dims[3] * dims[2])) % dims[1];
        let ma = flat / (dims[3] * dims[2] * dims[1]);
        let base = [
            offsets[0] + ma,
            offsets[1] + mb,
            offsets[2] + mc,
            offsets[3] + md,
        ];
        for code in 0..8u8 {
            let (swap_bra, swap_ket, swap_pairs) = (code & 1 != 0, code & 2 != 0, code & 4 != 0);
            if (swap_bra && !swaps[0]) || (swap_ket && !swaps[1]) || (swap_pairs && !swaps[2]) {
                continue;
            }
            let [mut u, mut v, mut w, mut x] = base;
            if swap_bra {
                (u, v) = (v, u);
            }
            if swap_ket {
                (w, x) = (x, w);
            }
            if swap_pairs {
                (u, v, w, x) = (w, x, u, v);
            }
            out[index([u, v, w, x])] = value;
        }
    }
}

/// A pair of shells with its primitive pairs and Schwarz bound.
struct ShellPair {
    /// Shell indices in the two tables.
    shells: [usize; 2],
    ls: [usize; 2],
    /// First function of each shell in its table.
    offsets: [usize; 2],
    primitives: Vec<PrimitivePair>,
    /// `max sqrt((ab|ab))` over the functions of the pair.
    bound: f64,
}

impl ShellPair {
    fn n_functions(&self) -> usize {
        (2 * self.ls[0] + 1) * (2 * self.ls[1] + 1)
    }
}

/// All shell pairs of two tables, the first translated by `translations[0]`
/// and the second by `translations[1]`, in row-major shell order; with
/// `symmetric` only the pairs with `second ≥ first`. Runs in parallel over
/// the first shell.
fn shell_pairs(
    tables: [&ShellTable; 2],
    translations: [[f64; 3]; 2],
    symmetric: bool,
    omega: Option<f64>,
    cache: &Tables,
) -> Vec<ShellPair> {
    let [first, second] = tables;
    let offsets = [first.function_offsets(), second.function_offsets()];
    let rows: Vec<Vec<ShellPair>> = (0..first.n_shells())
        .into_par_iter()
        .map(|ia| {
            let mut workspace = HermiteWorkspace::default();
            let a = first.shell(ia);
            let a_center: [f64; 3] =
                std::array::from_fn(|axis| a.center[axis] + translations[0][axis]);
            let shifted = ShellRef {
                center: a_center,
                ..a
            };
            let start = if symmetric { ia } else { 0 };
            (start..second.n_shells())
                .map(|ib| {
                    let b = second.shell(ib);
                    let b_center: [f64; 3] =
                        std::array::from_fn(|axis| b.center[axis] + translations[1][axis]);
                    let ls = [a.angular_momentum, b.angular_momentum];
                    let primitives = primitive_pairs(&shifted, &b, b_center, 0);
                    let bound = schwarz_bound(&primitives, ls, omega, cache, &mut workspace);
                    ShellPair {
                        shells: [ia, ib],
                        ls,
                        offsets: [offsets[0][ia], offsets[1][ib]],
                        primitives,
                        bound,
                    }
                })
                .collect()
        })
        .collect();
    rows.into_iter().flatten().collect()
}

/// `max_{ab} sqrt((ab|ab))` over the spherical functions of a pair.
fn schwarz_bound(
    primitives: &[PrimitivePair],
    ls: [usize; 2],
    omega: Option<f64>,
    tables: &Tables,
    workspace: &mut HermiteWorkspace,
) -> f64 {
    let n_cart = n_cartesian(ls[0]) * n_cartesian(ls[1]);
    let mut block = vec![0.0; n_cart * n_cart];
    coulomb_block(
        primitives, ls, primitives, ls, omega, tables, workspace, &mut block,
    );
    let spherical =
        tables.to_spherical(&block, [Some(ls[0]), Some(ls[1]), Some(ls[0]), Some(ls[1])]);
    let n_sph = (2 * ls[0] + 1) * (2 * ls[1] + 1);
    (0..n_sph)
        .map(|i| spherical[i * n_sph + i])
        .fold(0.0_f64, f64::max)
        .sqrt()
}

/// Evaluate one slab per leading shell in parallel and place the slabs.
///
/// The output is `[n_images, n_leading, row_size]` row-major, where the
/// leading index runs over the functions of `leading` and `fill` writes the
/// `(2l + 1) * row_size` values of one shell for one image.
fn assemble_parallel<T: Sync>(
    leading: &ShellTable,
    row_size: usize,
    images: &[T],
    fill: impl Fn(&T, ShellRef<'_>, &mut [f64]) + Sync,
) -> Vec<f64> {
    let n_leading = leading.n_functions();
    let offsets = leading.function_offsets();
    let mut out = vec![0.0; images.len() * n_leading * row_size];
    for (it, image) in images.iter().enumerate() {
        let slabs: Vec<Vec<f64>> = (0..leading.n_shells())
            .into_par_iter()
            .map(|index| {
                let shell = leading.shell(index);
                let mut slab = vec![0.0; (2 * shell.angular_momentum + 1) * row_size];
                fill(image, shell, &mut slab);
                slab
            })
            .collect();
        let target = &mut out[it * n_leading * row_size..(it + 1) * n_leading * row_size];
        for (&offset, slab) in offsets.iter().zip(&slabs) {
            target[offset * row_size..offset * row_size + slab.len()].copy_from_slice(slab);
        }
    }
    out
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
    /// The values may be real or complex.
    pub(crate) fn to_spherical<T>(&self, block: &[T], ls: [Option<usize>; 4]) -> Vec<T>
    where
        T: Copy + Default + AddAssign + Mul<f64, Output = T>,
    {
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
                // Only t + u + v ≤ l_a + l_b is consumed by the bra contraction.
                for t in 0..n_t {
                    for u in 0..n_t - t {
                        for v in 0..n_t - t - u {
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

/// Exact equality of two translations: a symmetry between images needs
/// identical vectors, not nearly equal ones.
#[allow(
    clippy::float_cmp,
    reason = "translations are exact inputs, not computed values"
)]
fn same_translation(a: [f64; 3], b: [f64; 3]) -> bool {
    a == b
}

fn check_threshold(threshold: f64) -> Result<(), Error> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(Error::InvalidArgument(
            "the screening threshold is finite and non-negative".to_string(),
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

    /// An s and a p shell of two primitives each on two centres, with
    /// arbitrary coefficients, translated as a whole by `shift`.
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

    fn sp_table() -> ShellTable {
        sp_table_at(ZERO)
    }

    /// The four-centre integrals of one image without any symmetry shortcut.
    fn generic_image(table: &ShellTable, translations: [[f64; 3]; 3]) -> Vec<f64> {
        let n = table.n_functions();
        let mut out = vec![0.0; n * n * n * n];
        FourCenterImage {
            bra: [table, table],
            ket: [table, table],
            translations,
            omega: None,
            threshold: 0.0,
            swaps: [false; 3],
        }
        .fill(&Tables::new(table.max_angular_momentum()), &mut out);
        out
    }

    fn molecular(table: &ShellTable, threshold: f64) -> Vec<f64> {
        coulomb_four_center(
            [table, table],
            [table, table],
            &[[ZERO; 3]],
            None,
            threshold,
        )
        .expect("valid")
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
        // the s function with unit norm is that charge divided by (q / 2π)^{3/4}.
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
    fn three_center_symmetric_path_matches_the_generic_path() {
        let table = sp_table();
        let aux = s_shell(0.5, [0.2, 0.1, -0.1]);
        let symmetric =
            coulomb_three_center(&aux, &table, &table, &[[ZERO, ZERO]], None).expect("valid");
        // A ket with a different centre list is a different table, so the
        // generic path runs; moving it back by a translation gives the same
        // integrals.
        let shifted = sp_table_at([1.0; 3]);
        let generic = coulomb_three_center(&aux, &table, &shifted, &[[ZERO, [-1.0; 3]]], None)
            .expect("valid");
        for (s, g) in symmetric.iter().zip(&generic) {
            assert!((s - g).abs() < 1e-13);
        }
    }

    #[test]
    fn four_center_ssss_matches_the_closed_form() {
        // (ab|cd) = N 2π^{5/2}/(pq√(p+q)) exp(−μ_ab AB²) exp(−μ_cd CD²) F_0(α R_PQ²).
        let exponents = [0.9, 0.4, 1.3, 0.6];
        let centers = [[0.0; 3], [0.0, 0.0, 1.1], [0.5, 0.2, 0.0], [0.4, -0.3, 0.9]];
        let shells: Vec<ShellTable> = (0..4).map(|i| s_shell(exponents[i], centers[i])).collect();
        let value = coulomb_four_center(
            [&shells[0], &shells[1]],
            [&shells[2], &shells[3]],
            &[[ZERO; 3]],
            None,
            0.0,
        )
        .expect("valid")[0];
        let distance_squared = |x: [f64; 3], y: [f64; 3]| -> f64 {
            (0..3).map(|axis| (x[axis] - y[axis]).powi(2)).sum()
        };
        // Product exponent, centre and Gaussian prefactor of one side.
        let product = |first: usize, second: usize| -> (f64, [f64; 3], f64) {
            let (x, y) = (exponents[first], exponents[second]);
            let center = std::array::from_fn(|axis| {
                (x * centers[first][axis] + y * centers[second][axis]) / (x + y)
            });
            let gaussian =
                (-x * y / (x + y) * distance_squared(centers[first], centers[second])).exp();
            (x + y, center, gaussian)
        };
        let (p, big_p, gaussian_bra) = product(0, 1);
        let (q, big_q, gaussian_ket) = product(2, 3);
        let alpha = p * q / (p + q);
        let boys = crate::special::boys_values(0, alpha * distance_squared(big_p, big_q))[0];
        let norms: f64 = exponents
            .iter()
            .map(|x| (2.0 * x / PI).powf(0.75))
            .product();
        let expected = norms * 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt())
            * gaussian_bra
            * gaussian_ket
            * boys;
        assert!((value - expected).abs() < 1e-13, "{value} vs {expected}");
    }

    #[test]
    fn four_center_symmetric_paths_match_the_generic_path() {
        let table = sp_table();
        let n = table.n_functions();
        let symmetric = molecular(&table, 0.0);
        let generic = generic_image(&table, [ZERO; 3]);
        for (s, g) in symmetric.iter().zip(&generic) {
            assert!((s - g).abs() < 1e-13);
        }
        // Each partial symmetry on its own.
        for swaps in [
            [true, false, false],
            [false, true, false],
            [false, false, true],
        ] {
            let mut partial = vec![0.0; n * n * n * n];
            FourCenterImage {
                bra: [&table, &table],
                ket: [&table, &table],
                translations: [ZERO; 3],
                omega: None,
                threshold: 0.0,
                swaps,
            }
            .fill(&Tables::new(1), &mut partial);
            for (s, g) in symmetric.iter().zip(&partial) {
                assert!((s - g).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn translated_images_admit_the_documented_swaps() {
        let table = sp_table();
        let other = sp_table_at([0.3; 3]);
        let shift = [0.4, -0.2, 0.3];
        let same = [&table, &table];
        // Bra and ket swaps with a translated ket pair; the pair swap with
        // the second function of each pair translated alike.
        assert_eq!(
            admitted_swaps(same, same, [ZERO, shift, shift]),
            [true, true, false]
        );
        assert_eq!(
            admitted_swaps(same, same, [shift, ZERO, shift]),
            [false, false, true]
        );
        assert_eq!(admitted_swaps(same, same, [ZERO; 3]), [true; 3]);
        // Unequal ket translations, a translated first ket function, and
        // distinct tables each spoil the corresponding swap.
        assert_eq!(
            admitted_swaps(same, same, [ZERO, shift, ZERO]),
            [true, false, false]
        );
        assert_eq!(
            admitted_swaps(same, same, [ZERO, ZERO, shift]),
            [true, false, false]
        );
        assert_eq!(
            admitted_swaps(same, same, [shift, ZERO, ZERO]),
            [false, true, false]
        );
        assert_eq!(
            admitted_swaps([&table, &other], same, [ZERO; 3]),
            [false, true, false]
        );
        assert_eq!(
            admitted_swaps(same, [&table, &other], [ZERO; 3]),
            [true, false, false]
        );
        assert_eq!(
            admitted_swaps([&table, &other], [&table, &other], [ZERO; 3]),
            [false, false, true]
        );
        // The swaps of translated images reproduce the generic integrals.
        for translations in [[ZERO, shift, shift], [shift, ZERO, shift]] {
            let symmetric =
                coulomb_four_center(same, same, &[translations], None, 0.0).expect("valid");
            let generic = generic_image(&table, translations);
            for (s, g) in symmetric.iter().zip(&generic) {
                assert!((s - g).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn schwarz_bounds_translate_the_ket() {
        let table = sp_table();
        let shift = [0.5, 0.1, -0.3];
        let translated = schwarz_bounds(&table, &table, &[shift], None).expect("valid");
        let direct = schwarz_bounds(&table, &sp_table_at(shift), &[ZERO], None).expect("valid");
        let at_origin = schwarz_bounds(&table, &table, &[ZERO], None).expect("valid");
        for (t, d) in translated.iter().zip(&direct) {
            assert!((t - d).abs() < 1e-14);
        }
        assert!(
            translated
                .iter()
                .zip(&at_origin)
                .any(|(t, o)| (t - o).abs() > 1e-3)
        );
    }

    #[test]
    fn four_center_translations_move_the_functions() {
        let bra = s_shell(0.9, [0.1, 0.2, 0.3]);
        let ket = s_shell(0.4, [0.5, -0.3, 1.0]);
        let (t_b, t_c, t_d) = ([1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, -0.7]);
        let translated =
            coulomb_four_center([&bra, &ket], [&bra, &ket], &[[t_b, t_c, t_d]], None, 0.0)
                .expect("valid")[0];
        let moved_b = s_shell(0.4, [1.5, -0.3, 1.0]);
        let moved_c = s_shell(0.9, [0.1, 0.7, 0.3]);
        let moved_d = s_shell(0.4, [0.5, -0.3, 0.3]);
        let direct = coulomb_four_center(
            [&bra, &moved_b],
            [&moved_c, &moved_d],
            &[[ZERO; 3]],
            None,
            0.0,
        )
        .expect("valid")[0];
        assert!((translated - direct).abs() < 1e-14);
    }

    #[test]
    fn schwarz_bounds_bound_every_quartet() {
        let table = sp_table();
        let bounds = schwarz_bounds(&table, &table, &[ZERO], None).expect("valid");
        let n_shells = table.n_shells();
        assert_eq!(bounds.len(), n_shells * n_shells);
        let n = table.n_functions();
        let eri = molecular(&table, 0.0);
        let offsets = table.function_offsets();
        let size = |i: usize| 2 * table.shell(i).angular_momentum + 1;
        for ia in 0..n_shells {
            for ib in 0..n_shells {
                for ic in 0..n_shells {
                    for id in 0..n_shells {
                        let bound = bounds[ia * n_shells + ib] * bounds[ic * n_shells + id];
                        for u in offsets[ia]..offsets[ia] + size(ia) {
                            for v in offsets[ib]..offsets[ib] + size(ib) {
                                for w in offsets[ic]..offsets[ic] + size(ic) {
                                    for x in offsets[id]..offsets[id] + size(id) {
                                        let value = eri[((u * n + v) * n + w) * n + x];
                                        assert!(value.abs() <= bound + 1e-14);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // The bound of a pair is attained by its own diagonal integral.
        let value = eri[0];
        assert!((bounds[0] - value.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn screening_zeroes_only_quartets_below_the_threshold() {
        let far = ShellTable::new(
            vec![0, 0],
            vec![[0.0; 3], [8.0, 0.0, 0.0]],
            vec![0, 1, 2],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .expect("valid");
        let exact = molecular(&far, 0.0);
        let screened = molecular(&far, 1e-6);
        let bounds = schwarz_bounds(&far, &far, &[ZERO], None).expect("valid");
        assert!(bounds[1] < 1e-6 && bounds[0] > 1e-6);
        let zeros = screened.iter().filter(|&&x| x == 0.0).count();
        assert!(zeros > 0);
        for (e, s) in exact.iter().zip(&screened) {
            assert!(*s == 0.0 || s.to_bits() == e.to_bits());
            assert!((e - s).abs() < 1e-6);
        }
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
        let pair = [&shell, &shell];
        assert!(coulomb_four_center(pair, pair, &[[ZERO; 3]], None, -1.0).is_err());
        assert!(coulomb_four_center(pair, pair, &[[ZERO; 3]], None, f64::NAN).is_err());
        assert!(
            coulomb_four_center(pair, pair, &[[ZERO, ZERO, [f64::NAN; 3]]], None, 0.0).is_err()
        );
        assert!(schwarz_bounds(&shell, &shell, &[ZERO], Some(0.0)).is_err());
    }
}

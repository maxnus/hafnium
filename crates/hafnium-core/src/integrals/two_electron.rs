//! Coulomb integrals between two charge distributions in the Hermite basis.
//!
//! `(ab|cd) = (2π^{5/2} / (pq √(p+q))) Σ_{tuv} E^{ab}_{tuv} Σ_{τνφ}
//! (−1)^{τ+ν+φ} E^{cd}_{τνφ} R_{t+τ, u+ν, v+φ}(α, R_PQ)` with the reduced
//! exponent `α = pq / (p + q)` (Helgaker 9.9.33). With a unit partner for
//! `b` or `d` (see [`super::pairs::single_shell_pairs`]) the same code
//! yields the two- and three-centre integrals of density fitting; with
//! real pairs on both sides it yields the four-centre integrals.
//!
//! The [`CoulombKernel`] selects `1/r` or a part of its range separation
//! (ADR-0008, ADR-0012), and a threshold screens every primitive quartet
//! whose bound lies below it (ADR-0018): the integral of a quartet with
//! pair centres a distance `R` apart is at most
//! `(2π^{5/2}/(pq√(p+q))) (1/√α) (s_hi − s_lo) exp(−(1 − ε) s_lo² R²) M`
//! with `[s_lo, s_hi]` the exponent range of the kernel, `M` the quartet's
//! envelope with the weights `s_hi^n h_n(ε)` over the combined Hermite
//! orders `n = t + τ`, and `h_n` the Hermite envelopes of [`super::pairs`].

use std::collections::HashMap;
use std::f64::consts::PI;
use std::ops::{AddAssign, Mul};

use num_complex::Complex64;
use rayon::prelude::*;

use super::cartesian::{
    cartesian_components, n_cartesian, spherical_transformation, transform_index,
};
use super::coulomb::{CoulombKernel, HermiteWorkspace};
use super::pairs::{
    ENVELOPE_EPSILON, PrimitivePair, PrimitivePairs, hermite_envelopes, primitive_pairs,
    single_shell_pairs,
};
use super::{ShellRef, ShellTable};
use crate::Error;

const ZERO: [f64; 3] = [0.0; 3];

/// What one kernel call screens with: the kernel, the threshold below
/// which a primitive quartet is left out, and the Hermite envelopes of its
/// bounds (ADR-0018). A threshold of zero evaluates everything.
pub(crate) struct Context<'a> {
    kernel: CoulombKernel,
    threshold: f64,
    envelopes: &'a [f64],
}

/// The Hermite envelopes for every combined order the tables can reach.
fn envelopes_for(tables: &[&ShellTable]) -> Vec<f64> {
    let max_order: usize = tables
        .iter()
        .map(|table| table.max_angular_momentum())
        .sum();
    hermite_envelopes(max_order, ENVELOPE_EPSILON)
}

/// Two-centre Coulomb integrals `(P | Q(r − T))` for each translation `T`
/// of the ket, as a row-major `[n_translations, n_bra, n_ket]` array.
///
/// Primitive pairs bounded below `threshold` are left out; every neglected
/// term is smaller than `threshold`.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if the kernel's `omega` is not
/// positive, `threshold` is negative or not finite, or a translation is not
/// finite.
pub fn coulomb_two_center(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
    kernel: CoulombKernel,
    threshold: f64,
) -> Result<Vec<f64>, Error> {
    kernel.validate()?;
    check_threshold(threshold)?;
    check_finite(translations, "translations")?;
    let n_ket = ket.n_functions();
    let max_l = bra.max_angular_momentum().max(ket.max_angular_momentum());
    let tables = Tables::new(max_l);
    let envelopes = envelopes_for(&[bra, ket]);
    let context = Context {
        kernel,
        threshold,
        envelopes: &envelopes,
    };
    let ket_offsets = ket.function_offsets();
    Ok(assemble_parallel(
        bra,
        n_ket,
        translations,
        |translation, a, slab| {
            let (la, ka) = (a.angular_momentum, a.n_contractions);
            let bra_pairs = single_shell_pairs(&a, a.center);
            let mut workspace = HermiteWorkspace::default();
            for (ic, &ket_offset) in ket_offsets.iter().enumerate() {
                let c = ket.shell(ic);
                let (lc, kc) = (c.angular_momentum, c.n_contractions);
                let c_center: [f64; 3] =
                    std::array::from_fn(|axis| c.center[axis] + translation[axis]);
                let ket_pairs = single_shell_pairs(&c, c_center);
                let mut block = vec![0.0; ka * n_cartesian(la) * kc * n_cartesian(lc)];
                coulomb_block(
                    &bra_pairs,
                    [la, 0],
                    &ket_pairs,
                    [lc, 0],
                    &context,
                    tables.row_norms[la] * tables.row_norms[lc],
                    &tables,
                    &mut workspace,
                    &mut block,
                );
                let spherical =
                    tables.to_spherical(&block, [Some(la), None, Some(lc), None], [ka, 1, kc, 1]);
                for (fa, row) in spherical.chunks(c.n_functions()).enumerate() {
                    slab[fa * n_ket + ket_offset..fa * n_ket + ket_offset + row.len()]
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
/// fills the rest. Primitive quartets bounded below `threshold` are left
/// out; every neglected term is smaller than `threshold`.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if the kernel's `omega` is not
/// positive, `threshold` is negative or not finite, or a translation is not
/// finite.
pub fn coulomb_three_center(
    aux: &ShellTable,
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[[f64; 3]; 2]],
    kernel: CoulombKernel,
    threshold: f64,
) -> Result<Vec<f64>, Error> {
    kernel.validate()?;
    check_threshold(threshold)?;
    for pair in translations {
        check_finite(pair, "translations")?;
    }
    let (n_bra, n_ket) = (bra.n_functions(), ket.n_functions());
    let max_l = aux
        .max_angular_momentum()
        .max(bra.max_angular_momentum())
        .max(ket.max_angular_momentum());
    let tables = Tables::new(max_l);
    let envelopes = envelopes_for(&[aux, bra, ket]);
    let context = Context {
        kernel,
        threshold,
        envelopes: &envelopes,
    };
    let bra_offsets = bra.function_offsets();
    let ket_offsets = ket.function_offsets();
    Ok(assemble_parallel(
        aux,
        n_bra * n_ket,
        translations,
        |[aux_translation, ket_translation], p_shell, slab| {
            let symmetric = bra == ket && same_translation(*ket_translation, ZERO);
            let (lp, kp) = (p_shell.angular_momentum, p_shell.n_contractions);
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
                    let (ka, kb) = (a.n_contractions, b.n_contractions);
                    let pairs = primitive_pairs(&a, &b, b_center, 0);
                    let mut block =
                        vec![
                            0.0;
                            ka * n_cartesian(la) * kb * n_cartesian(lb) * kp * n_cartesian(lp)
                        ];
                    coulomb_block(
                        &pairs,
                        [la, lb],
                        &aux_pairs,
                        [lp, 0],
                        &context,
                        tables.row_norms[la] * tables.row_norms[lb] * tables.row_norms[lp],
                        &tables,
                        &mut workspace,
                        &mut block,
                    );
                    // spherical[fa][fb][fp] → slab[fp][μ][ν] (and slab[fp][ν][μ])
                    let spherical = tables.to_spherical(
                        &block,
                        [Some(la), Some(lb), Some(lp), None],
                        [ka, kb, kp, 1],
                    );
                    let (nsb, nsp) = (b.n_functions(), p_shell.n_functions());
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

/// The images and phases of a Bloch-summed three-centre integral.
///
/// The orbital functions are imaged over `images`, with the integer
/// lattice coordinates `image_coordinates` of every image; the ket image
/// of a pair is the bra image displaced by one of the `steps`, the
/// relative translations at which a pair of functions still overlaps,
/// found by their coordinates. `phases` are the row-major
/// `[n_k, n_images]` phase tables of the bra and the ket.
#[derive(Debug, Clone, Copy)]
pub struct BlochSum<'a> {
    /// The image list `[image, 3]` in Bohr, shared by the bra and the ket.
    pub images: &'a [[f64; 3]],
    /// The integer lattice coordinates of every image.
    pub image_coordinates: &'a [[i64; 3]],
    /// The relative translations `[step, 3]` of the pairs, in Bohr.
    pub steps: &'a [[f64; 3]],
    /// The integer lattice coordinates of every step.
    pub step_coordinates: &'a [[i64; 3]],
    /// The phase tables of the bra and the ket.
    pub phases: [&'a [Complex64]; 2],
}

impl BlochSum<'_> {
    /// Validate the lists and return the number of k-points.
    fn check(&self) -> Result<usize, Error> {
        let invalid = |message: &str| Err(Error::InvalidArgument(message.to_string()));
        check_finite(self.images, "translations")?;
        check_finite(self.steps, "translations")?;
        let n_images = self.images.len();
        if n_images == 0 {
            return invalid("a Bloch sum needs at least one image");
        }
        if self.image_coordinates.len() != n_images
            || self.step_coordinates.len() != self.steps.len()
        {
            return invalid("every image and every step has one set of lattice coordinates");
        }
        let n_k = self.phases[0].len() / n_images;
        if self.phases[0].len() != n_k * n_images || self.phases[1].len() != n_k * n_images {
            return invalid("the phase tables hold one row per k-point and one column per image");
        }
        if self
            .phases
            .iter()
            .any(|table| table.iter().any(|z| !z.is_finite()))
        {
            return invalid("phases are finite");
        }
        Ok(n_k)
    }

    /// The image index of every lattice coordinate, after checking that the
    /// coordinates agree with the vectors: for every step that reaches an
    /// image, the first pair found satisfies `T_j = T_i + step`.
    fn lookup(&self) -> Result<HashMap<[i64; 3], usize>, Error> {
        let mut lookup = HashMap::with_capacity(self.images.len());
        for (index, coordinates) in self.image_coordinates.iter().enumerate() {
            if lookup.insert(*coordinates, index).is_some() {
                return Err(Error::InvalidArgument(
                    "the image coordinates are distinct".to_string(),
                ));
            }
        }
        for (step, coordinates) in self.step_coordinates.iter().enumerate() {
            let found = self
                .image_coordinates
                .iter()
                .enumerate()
                .find_map(|(i, origin)| {
                    let target: [i64; 3] =
                        std::array::from_fn(|axis| origin[axis] + coordinates[axis]);
                    lookup.get(&target).map(|&j| (i, j))
                });
            if let Some((i, j)) = found {
                let mismatch = (0..3)
                    .map(|axis| {
                        (self.images[j][axis] - self.images[i][axis] - self.steps[step][axis]).abs()
                    })
                    .fold(0.0_f64, f64::max);
                if mismatch > COORDINATE_TOLERANCE {
                    return Err(Error::InvalidArgument(
                        "the lattice coordinates of the images and steps disagree with their vectors"
                            .to_string(),
                    ));
                }
            }
        }
        Ok(lookup)
    }
}

/// Largest difference in Bohr between an image vector and the one its
/// lattice coordinates predict.
const COORDINATE_TOLERANCE: f64 = 1e-8;

/// The tables shared by every image of a Bloch-summed three-centre call.
struct BlochSetup<'a> {
    aux: &'a ShellTable,
    bra: &'a ShellTable,
    ket: &'a ShellTable,
    tables: &'a Tables,
    context: &'a Context<'a>,
    aux_pairs: &'a [PrimitivePairs],
    /// Schwarz bounds `Q_ab` of every shell pair at every step, `[step, a, b]`.
    step_bounds: &'a [f64],
    /// Schwarz bounds `Q_P` of the auxiliary shells.
    aux_bounds: &'a [f64],
    /// The quartet bounds of every shell triple at every step summed over
    /// its primitive quartets at zero distance, `[step, a, b, P]`, and the
    /// decay rate `(1 − ε) s_lo²` of the slowest quartet of each triple,
    /// `[a, b, P]`: the triple's integral at a distance `R` between the
    /// auxiliary centre and the pair centres is at most
    /// `zero_bounds × exp(−decay R²)`.
    zero_bounds: &'a [f64],
    decays: &'a [f64],
    lookup: &'a HashMap<[i64; 3], usize>,
    n_k: usize,
}

/// The Schwarz bounds `[step, a, b]` and zero-distance bounds
/// `[step, a, b, P]` of [`BlochSetup`], from one pass over the shell pairs
/// of every step in parallel.
fn step_tables(
    tables: [&ShellTable; 3],
    aux_pairs: &[PrimitivePairs],
    steps: &[[f64; 3]],
    context: &Context<'_>,
    cache: &Tables,
) -> (Vec<f64>, Vec<f64>) {
    let [aux, bra, ket] = tables;
    let rows: Vec<(Vec<f64>, Vec<f64>)> = steps
        .par_iter()
        .map(|step| {
            let pairs = shell_pairs([bra, ket], [ZERO, *step], false, context.kernel, cache);
            let mut schwarz = Vec::with_capacity(pairs.len());
            let mut zero = Vec::with_capacity(pairs.len() * aux_pairs.len());
            for pair in &pairs {
                schwarz.push(pair.bound);
                let [la, lb] = pair.ls;
                for (ip, aux_pair) in aux_pairs.iter().enumerate() {
                    let lp = aux.shell(ip).angular_momentum;
                    let components: [&[[usize; 3]]; 4] = [
                        &cache.components[la],
                        &cache.components[lb],
                        &cache.components[lp],
                        &cache.components[0],
                    ];
                    let scale = cache.row_norms[la] * cache.row_norms[lb] * cache.row_norms[lp];
                    let mut total = 0.0;
                    for (index_bra, pair_bra) in pair.primitives.pairs.iter().enumerate() {
                        for (index_ket, pair_ket) in aux_pair.pairs.iter().enumerate() {
                            total += scale
                                * pair.primitives.coefficient_max(index_bra)
                                * aux_pair.coefficient_max(index_ket)
                                * quartet_bound_at_zero(pair_bra, pair_ket, components, context);
                        }
                    }
                    zero.push(total);
                }
            }
            (schwarz, zero)
        })
        .collect();
    let mut schwarz = Vec::new();
    let mut zero = Vec::new();
    for (row_schwarz, row_zero) in rows {
        schwarz.extend(row_schwarz);
        zero.extend(row_zero);
    }
    (schwarz, zero)
}

/// The decay rates `(1 − ε) s_lo²` `[a, b, P]` of the slowest primitive
/// quartet of every shell triple, from the smallest exponents.
fn triple_decays(tables: [&ShellTable; 3], kernel: CoulombKernel) -> Vec<f64> {
    let [aux, bra, ket] = tables;
    let smallest = |shell: &ShellRef<'_>| {
        shell
            .exponents
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    };
    let mut decays = Vec::with_capacity(bra.n_shells() * ket.n_shells() * aux.n_shells());
    for ia in 0..bra.n_shells() {
        for ib in 0..ket.n_shells() {
            let p_min = smallest(&bra.shell(ia)) + smallest(&ket.shell(ib));
            for ip in 0..aux.n_shells() {
                let q_min = smallest(&aux.shell(ip));
                let alpha = p_min * q_min / (p_min + q_min);
                let (low, _) = kernel.exponent_range(alpha);
                decays.push((1.0 - ENVELOPE_EPSILON) * low * low);
            }
        }
    }
    decays
}

/// The quartet bound of [`quartet_bound`] without its distance factor.
fn quartet_bound_at_zero(
    pair_bra: &PrimitivePair,
    pair_ket: &PrimitivePair,
    components: [&[[usize; 3]]; 4],
    context: &Context<'_>,
) -> f64 {
    let (p, q) = (pair_bra.p, pair_ket.p);
    let alpha = p * q / (p + q);
    let (low, high) = context.kernel.exponent_range(alpha);
    let prefactor = 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt()) / alpha.sqrt() * (high - low);
    prefactor * quartet_envelope(pair_bra, pair_ket, components, high, context.envelopes)
}

/// The squared distance from `point` to the segment between `start` and `end`.
fn segment_distance_squared(point: [f64; 3], start: [f64; 3], end: [f64; 3]) -> f64 {
    let direction: [f64; 3] = std::array::from_fn(|axis| end[axis] - start[axis]);
    let offset: [f64; 3] = std::array::from_fn(|axis| point[axis] - start[axis]);
    let length_squared: f64 = direction.iter().map(|x| x * x).sum();
    let along: f64 = (0..3).map(|axis| offset[axis] * direction[axis]).sum();
    let t = if length_squared > 0.0 {
        (along / length_squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0..3)
        .map(|axis| (offset[axis] - t * direction[axis]).powi(2))
        .sum()
}

/// Three-centre Coulomb integrals summed over the images of the bra and
/// the ket with phases,
/// `Σ_{(i, j)} b[k1, i] c[k2, j] (P | μ(r − T_i) ν(r − T_j))`, as a
/// row-major complex `[n_k, n_k, n_aux, n_bra, n_ket]` array over the
/// pairs `(i, j)` of images of `sum` whose difference is one of its steps,
/// with the phase tables `b` and `c` of the bra and the ket. With
/// `b[k1, i] = exp(−i k1·T_i)` and `c[k2, j] = exp(i k2·T_j)` the result is
/// the Bloch-summed integral `(P^q | μ k1 ν k2)` of the density fitting
/// with momentum transfer `q = k2 − k1`, the auxiliary function in the
/// home cell (ADR-0019).
///
/// A shell triple is skipped when the product of the Schwarz bounds of
/// the pair at its step and of the auxiliary shell is below `threshold`,
/// and primitive quartets bounded below `threshold` are left out as in
/// [`coulomb_three_center`]; every neglected term is smaller than
/// `threshold`.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if the kernel's `omega` is not
/// positive, `threshold` is negative or not finite, a translation is not
/// finite, the coordinate lists do not match the image and step lists or
/// repeat an image, or the phase tables do not hold the same number of
/// k-points for every image.
pub fn coulomb_three_center_bloch(
    aux: &ShellTable,
    bra: &ShellTable,
    ket: &ShellTable,
    sum: BlochSum<'_>,
    kernel: CoulombKernel,
    threshold: f64,
) -> Result<Vec<Complex64>, Error> {
    kernel.validate()?;
    check_threshold(threshold)?;
    let n_k = sum.check()?;
    let lookup = sum.lookup()?;
    let n_images = sum.images.len();
    let block_size = aux.n_functions() * bra.n_functions() * ket.n_functions();
    let size = n_k * n_k * block_size;
    let max_l = aux
        .max_angular_momentum()
        .max(bra.max_angular_momentum())
        .max(ket.max_angular_momentum());
    let tables = Tables::new(max_l);
    let envelopes = envelopes_for(&[aux, bra, ket]);
    let context = Context {
        kernel,
        threshold,
        envelopes: &envelopes,
    };
    let aux_pairs: Vec<PrimitivePairs> = (0..aux.n_shells())
        .map(|ip| {
            let p_shell = aux.shell(ip);
            single_shell_pairs(&p_shell, p_shell.center)
        })
        .collect();
    let aux_bounds: Vec<f64> = aux_pairs
        .iter()
        .enumerate()
        .map(|(ip, pairs)| {
            let lp = aux.shell(ip).angular_momentum;
            schwarz_bound(
                pairs,
                [lp, 0],
                kernel,
                &tables,
                &mut HermiteWorkspace::default(),
            )
        })
        .collect();
    let (step_bounds, zero_bounds) =
        step_tables([aux, bra, ket], &aux_pairs, sum.steps, &context, &tables);
    let decays = triple_decays([aux, bra, ket], kernel);
    let setup = BlochSetup {
        aux,
        bra,
        ket,
        tables: &tables,
        context: &context,
        aux_pairs: &aux_pairs,
        step_bounds: &step_bounds,
        aux_bounds: &aux_bounds,
        zero_bounds: &zero_bounds,
        decays: &decays,
        lookup: &lookup,
        n_k,
    };
    let bra_phases = sum.phases[0];
    Ok((0..n_images)
        .into_par_iter()
        .fold(
            || vec![Complex64::default(); size],
            |mut accumulated, i| {
                if let Some(buffer) = bloch_image(&setup, &sum, i) {
                    for k1 in 0..n_k {
                        let phase = bra_phases[k1 * n_images + i];
                        for k2 in 0..n_k {
                            let target =
                                &mut accumulated[(k1 * n_k + k2) * block_size..][..block_size];
                            let source = &buffer[k2 * block_size..][..block_size];
                            for (t, s) in target.iter_mut().zip(source) {
                                *t += phase * s;
                            }
                        }
                    }
                }
                accumulated
            },
        )
        .reduce(
            || vec![Complex64::default(); size],
            |mut left, right| {
                for (x, y) in left.iter_mut().zip(&right) {
                    *x += y;
                }
                left
            },
        ))
}

/// `buffer[k2][P][μ][ν] = Σ_j c[k2, j] (P | μ(T_i) ν(T_j))` over the ket
/// images `j = i + step` of bra image `i`, or `None` when every term was
/// screened away.
fn bloch_image(setup: &BlochSetup<'_>, sum: &BlochSum<'_>, i: usize) -> Option<Vec<Complex64>> {
    let (aux, bra, ket, tables) = (setup.aux, setup.bra, setup.ket, setup.tables);
    let (n_bra, n_ket) = (bra.n_functions(), ket.n_functions());
    let (n_a, n_b) = (bra.n_shells(), ket.n_shells());
    let block_size = aux.n_functions() * n_bra * n_ket;
    let n_images = sum.images.len();
    let ket_phases = sum.phases[1];
    let threshold = setup.context.threshold;
    let aux_max = setup.aux_bounds.iter().fold(0.0_f64, |acc, q| acc.max(*q));
    let (aux_offsets, bra_offsets) = (aux.function_offsets(), bra.function_offsets());
    let ket_offsets = ket.function_offsets();
    let mut workspace = HermiteWorkspace::default();
    let mut buffer = vec![Complex64::default(); setup.n_k * block_size];
    let mut block = Vec::new();
    let mut touched = false;
    let t_i = sum.images[i];
    let origin = sum.image_coordinates[i];
    for (step, coordinates) in sum.step_coordinates.iter().enumerate() {
        let target: [i64; 3] = std::array::from_fn(|axis| origin[axis] + coordinates[axis]);
        let Some(&j) = setup.lookup.get(&target) else {
            continue;
        };
        let t_j = sum.images[j];
        for (ia, &bra_offset) in bra_offsets.iter().enumerate() {
            let a = bra.shell(ia);
            let a_center: [f64; 3] = std::array::from_fn(|axis| a.center[axis] + t_i[axis]);
            let shifted = ShellRef {
                center: a_center,
                ..a
            };
            for (ib, &ket_offset) in ket_offsets.iter().enumerate() {
                let pair_index = (step * n_a + ia) * n_b + ib;
                let pair_bound = setup.step_bounds[pair_index];
                if pair_bound * aux_max < threshold {
                    continue;
                }
                let b = ket.shell(ib);
                let b_center: [f64; 3] = std::array::from_fn(|axis| b.center[axis] + t_j[axis]);
                let (la, lb) = (a.angular_momentum, b.angular_momentum);
                let (ka, kb) = (a.n_contractions, b.n_contractions);
                let mut shell_pair: Option<PrimitivePairs> = None;
                for (ip, &aux_offset) in aux_offsets.iter().enumerate() {
                    if pair_bound * setup.aux_bounds[ip] < threshold {
                        continue;
                    }
                    let p_shell = aux.shell(ip);
                    let triple = pair_index * aux.n_shells() + ip;
                    let distance_squared =
                        segment_distance_squared(p_shell.center, a_center, b_center);
                    let decay = setup.decays[(ia * n_b + ib) * aux.n_shells() + ip];
                    if threshold > 0.0
                        && setup.zero_bounds[triple] * (-decay * distance_squared).exp() < threshold
                    {
                        continue;
                    }
                    let shell_pair = shell_pair
                        .get_or_insert_with(|| primitive_pairs(&shifted, &b, b_center, 0));
                    let (lp, kp) = (p_shell.angular_momentum, p_shell.n_contractions);
                    let cartesian = ka * n_cartesian(la) * kb * n_cartesian(lb) * kp;
                    block.clear();
                    block.resize(cartesian * n_cartesian(lp), 0.0);
                    let scale = tables.row_norms[la] * tables.row_norms[lb] * tables.row_norms[lp];
                    if !coulomb_block(
                        shell_pair,
                        [la, lb],
                        &setup.aux_pairs[ip],
                        [lp, 0],
                        setup.context,
                        scale,
                        tables,
                        &mut workspace,
                        &mut block,
                    ) {
                        continue;
                    }
                    touched = true;
                    let spherical = tables.to_spherical(
                        &block,
                        [Some(la), Some(lb), Some(lp), None],
                        [ka, kb, kp, 1],
                    );
                    let (nsb, nsp) = (b.n_functions(), p_shell.n_functions());
                    for (index, value) in spherical.iter().enumerate() {
                        let mp = index % nsp;
                        let mb = (index / nsp) % nsb;
                        let ma = index / (nsp * nsb);
                        let target =
                            ((aux_offset + mp) * n_bra + bra_offset + ma) * n_ket + ket_offset + mb;
                        for k2 in 0..setup.n_k {
                            buffer[k2 * block_size + target] +=
                                ket_phases[k2 * n_images + j] * value;
                        }
                    }
                }
            }
        }
    }
    touched.then_some(buffer)
}

/// Four-centre Coulomb integrals `(a b(r − T_b) | c(r − T_c) d(r − T_d))`
/// for each triple of translations, as a row-major
/// `[n_images, n_a, n_b, n_c, n_d]` array in chemists' notation;
/// `translations[image] = [T_b, T_c, T_d]`.
///
/// Shell quartets whose Schwarz bound `Q_ab Q_cd` (see [`schwarz_bounds`])
/// is below `threshold` are not evaluated and stay zero; a kept quartet is
/// evaluated in full. Where two tables coincide and the translations
/// permit, only the unique shell quartets are evaluated and the
/// permutational symmetry fills the rest: the bra pair may be swapped when
/// `bra[0] == bra[1]` and `T_b = 0`, the ket pair when `ket[0] == ket[1]`
/// and `T_c = T_d`, and the two pairs when `bra == ket`, `T_c = 0` and
/// `T_d = T_b`.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if the kernel's `omega` is not
/// positive, a translation is not finite, or `threshold` is negative or not
/// finite.
pub fn coulomb_four_center(
    bra: [&ShellTable; 2],
    ket: [&ShellTable; 2],
    translations: &[[[f64; 3]; 3]],
    kernel: CoulombKernel,
    threshold: f64,
) -> Result<Vec<f64>, Error> {
    kernel.validate()?;
    check_threshold(threshold)?;
    for triple in translations {
        check_finite(triple, "translations")?;
    }
    let size: usize = output_shape(bra, ket).iter().product();
    let max_l = bra
        .iter()
        .chain(ket.iter())
        .map(|table| table.max_angular_momentum())
        .max()
        .unwrap_or(0);
    let tables = Tables::new(max_l);
    let envelopes = envelopes_for(&[bra[0], bra[1], ket[0], ket[1]]);
    let mut out = vec![0.0; translations.len() * size];
    for (it, &translation) in translations.iter().enumerate() {
        let image = FourCenterImage {
            bra,
            ket,
            translations: translation,
            context: Context {
                kernel,
                threshold,
                envelopes: &envelopes,
            },
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
/// Returns [`Error::InvalidArgument`] if the kernel's `omega` is not
/// positive or a translation is not finite.
pub fn schwarz_bounds(
    bra: &ShellTable,
    ket: &ShellTable,
    translations: &[[f64; 3]],
    kernel: CoulombKernel,
) -> Result<Vec<f64>, Error> {
    kernel.validate()?;
    check_finite(translations, "translations")?;
    let max_l = bra.max_angular_momentum().max(ket.max_angular_momentum());
    let tables = Tables::new(max_l);
    let mut out = Vec::with_capacity(translations.len() * bra.n_shells() * ket.n_shells());
    for &translation in translations {
        let pairs = shell_pairs([bra, ket], [ZERO, translation], false, kernel, &tables);
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
    context: Context<'a>,
    /// Derived by [`admitted_swaps`] in [`coulomb_four_center`]; tests set
    /// it directly to force the generic path.
    swaps: [bool; 3],
}

/// Values of evaluated blocks held between a parallel chunk of bra pairs
/// and their placement in the output; bounds the extra memory.
const CHUNK_BUDGET: usize = 1 << 23;

/// The evaluated blocks of one bra pair, concatenated: `entries` holds the
/// ket pair index and the start of each block in `values`.
#[derive(Default)]
struct PairBlocks {
    values: Vec<f64>,
    entries: Vec<(usize, usize)>,
}

impl PairBlocks {
    /// The ket pair index and values of each block.
    fn iter(&self) -> impl Iterator<Item = (usize, &[f64])> {
        self.entries
            .iter()
            .enumerate()
            .map(|(k, &(ket_index, start))| {
                let end = self
                    .entries
                    .get(k + 1)
                    .map_or(self.values.len(), |entry| entry.1);
                (ket_index, &self.values[start..end])
            })
    }
}

impl FourCenterImage<'_> {
    /// Evaluate the unique shell quartets above the threshold in parallel
    /// over bra pairs and place them, with their symmetric images, in `out`.
    fn fill(&self, tables: &Tables, out: &mut [f64]) {
        let [t_b, t_c, t_d] = self.translations;
        let kernel = self.context.kernel;
        // Shell quartets are screened by their Schwarz bounds; a kept
        // quartet is evaluated in full.
        let unscreened = Context {
            kernel,
            threshold: 0.0,
            envelopes: &[],
        };
        let bra_pairs = shell_pairs(self.bra, [ZERO, t_b], self.swaps[0], kernel, tables);
        // With the pair swap the ket pairs are the bra pairs, in the same order.
        let ket_storage;
        let ket_pairs: &[ShellPair] = if self.swaps[2] {
            &bra_pairs
        } else {
            ket_storage = shell_pairs(self.ket, [t_c, t_d], self.swaps[1], kernel, tables);
            &ket_storage
        };
        let shape = output_shape(self.bra, self.ket);
        let ket_size: usize = ket_pairs.iter().map(ShellPair::n_functions).sum();
        let mut start = 0;
        while start < bra_pairs.len() {
            let end = chunk_end(&bra_pairs, start, ket_size);
            let blocks: Vec<PairBlocks> = (start..end)
                .into_par_iter()
                .map(|index| {
                    let bra_pair = &bra_pairs[index];
                    let partners = if self.swaps[2] {
                        &ket_pairs[..=index]
                    } else {
                        ket_pairs
                    };
                    let mut workspace = HermiteWorkspace::default();
                    let mut blocks = PairBlocks::default();
                    for (ket_index, ket_pair) in partners.iter().enumerate() {
                        if bra_pair.bound * ket_pair.bound < self.context.threshold {
                            continue;
                        }
                        blocks.entries.push((ket_index, blocks.values.len()));
                        let block =
                            quartet(bra_pair, ket_pair, &unscreened, tables, &mut workspace);
                        blocks.values.extend_from_slice(&block);
                    }
                    blocks
                })
                .collect();
            scatter_parallel(
                out,
                shape,
                &bra_pairs[start..end],
                ket_pairs,
                &blocks,
                self.swaps,
            );
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
    context: &Context<'_>,
    tables: &Tables,
    workspace: &mut HermiteWorkspace,
) -> Vec<f64> {
    let mut block = vec![0.0; bra.n_cartesian() * ket.n_cartesian()];
    let scale = tables.row_norms[bra.ls[0]]
        * tables.row_norms[bra.ls[1]]
        * tables.row_norms[ket.ls[0]]
        * tables.row_norms[ket.ls[1]];
    coulomb_block(
        &bra.primitives,
        bra.ls,
        &ket.primitives,
        ket.ls,
        context,
        scale,
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
        [
            bra.contractions[0],
            bra.contractions[1],
            ket.contractions[0],
            ket.contractions[1],
        ],
    )
}

/// Place the blocks of a chunk of bra pairs in `out` (shape
/// `[n_a, n_b, n_c, n_d]`), in parallel over ranges of the first index.
///
/// Every element of `out` is written by exactly one block and swap, so the
/// tasks never collide; each task scans all blocks and writes the elements
/// whose first index falls in its range.
fn scatter_parallel(
    out: &mut [f64],
    shape: [usize; 4],
    bra_pairs: &[ShellPair],
    ket_pairs: &[ShellPair],
    blocks: &[PairBlocks],
    swaps: [bool; 3],
) {
    let row = shape[1] * shape[2] * shape[3];
    debug_assert_eq!(out.len(), shape[0] * row, "one image of the output");
    if row == 0 || shape[0] == 0 {
        return;
    }
    // Rows hold equal numbers of elements, so one task per thread balances.
    let rows_per_task = shape[0].div_ceil(rayon::current_num_threads()).max(1);
    out.par_chunks_mut(rows_per_task * row)
        .enumerate()
        .for_each(|(task, chunk)| {
            let first = task * rows_per_task;
            let range = first..first + chunk.len() / row;
            for (bra_pair, pair_blocks) in bra_pairs.iter().zip(blocks) {
                if !bra_pair.touches(&range) {
                    // Without the pair swap only the bra shells can supply the
                    // first index.
                    if !swaps[2] {
                        continue;
                    }
                }
                for (ket_index, block) in pair_blocks.iter() {
                    let ket_pair = &ket_pairs[ket_index];
                    if !bra_pair.touches(&range) && !ket_pair.touches(&range) {
                        continue;
                    }
                    scatter(chunk, &range, shape, bra_pair, ket_pair, block, swaps);
                }
            }
        });
}

/// Write the elements of a quartet block whose first index lies in
/// `range` into `chunk`, the rows `range` of the output, at their own
/// position and at every position the admitted swaps map them to. A swap
/// that maps the block onto itself is skipped, so each element is written
/// exactly once.
fn scatter(
    chunk: &mut [f64],
    range: &std::ops::Range<usize>,
    shape: [usize; 4],
    bra: &ShellPair,
    ket: &ShellPair,
    block: &[f64],
    swaps: [bool; 3],
) {
    let dims = [
        bra.n_functions_of(0),
        bra.n_functions_of(1),
        ket.n_functions_of(0),
        ket.n_functions_of(1),
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
    for code in 0..8u8 {
        let (swap_bra, swap_ket, swap_pairs) = (code & 1 != 0, code & 2 != 0, code & 4 != 0);
        if (swap_bra && !swaps[0]) || (swap_ket && !swaps[1]) || (swap_pairs && !swaps[2]) {
            continue;
        }
        // The shell that supplies the first index after the swaps.
        let source = match (swap_pairs, swap_bra, swap_ket) {
            (false, false, _) => 0,
            (false, true, _) => 1,
            (true, _, false) => 2,
            (true, _, true) => 3,
        };
        if offsets[source] >= range.end || offsets[source] + dims[source] <= range.start {
            continue;
        }
        // Only the functions of the source shell inside `range` are visited.
        let mut bounds: [std::ops::Range<usize>; 4] = std::array::from_fn(|i| 0..dims[i]);
        bounds[source] = range.start.saturating_sub(offsets[source])
            ..(range.end - offsets[source]).min(dims[source]);
        for ma in bounds[0].clone() {
            for mb in bounds[1].clone() {
                for mc in bounds[2].clone() {
                    for md in bounds[3].clone() {
                        let flat = ((ma * dims[1] + mb) * dims[2] + mc) * dims[3] + md;
                        let [mut u, mut v, mut w, mut x] = [
                            offsets[0] + ma,
                            offsets[1] + mb,
                            offsets[2] + mc,
                            offsets[3] + md,
                        ];
                        if swap_bra {
                            (u, v) = (v, u);
                        }
                        if swap_ket {
                            (w, x) = (x, w);
                        }
                        if swap_pairs {
                            (u, v, w, x) = (w, x, u, v);
                        }
                        chunk[index([u - range.start, v, w, x])] = block[flat];
                    }
                }
            }
        }
    }
}

/// A pair of shells with its primitive pairs and Schwarz bound.
struct ShellPair {
    /// Shell indices in the two tables.
    shells: [usize; 2],
    ls: [usize; 2],
    /// Number of contractions of each shell.
    contractions: [usize; 2],
    /// First function of each shell in its table.
    offsets: [usize; 2],
    primitives: PrimitivePairs,
    /// `max sqrt((ab|ab))` over the functions of the pair.
    bound: f64,
}

impl ShellPair {
    /// Spherical functions of shell `which` of the pair.
    fn n_functions_of(&self, which: usize) -> usize {
        self.contractions[which] * (2 * self.ls[which] + 1)
    }

    /// Whether a function of either shell lies in `range`.
    fn touches(&self, range: &std::ops::Range<usize>) -> bool {
        (0..2).any(|which| {
            let start = self.offsets[which];
            start < range.end && start + self.n_functions_of(which) > range.start
        })
    }

    /// Spherical function pairs.
    fn n_functions(&self) -> usize {
        self.n_functions_of(0) * self.n_functions_of(1)
    }

    /// Cartesian component pairs over all contractions.
    fn n_cartesian(&self) -> usize {
        self.contractions[0]
            * n_cartesian(self.ls[0])
            * self.contractions[1]
            * n_cartesian(self.ls[1])
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
    kernel: CoulombKernel,
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
                    let bound = schwarz_bound(&primitives, ls, kernel, cache, &mut workspace);
                    ShellPair {
                        shells: [ia, ib],
                        ls,
                        contractions: primitives.contractions,
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
    primitives: &PrimitivePairs,
    ls: [usize; 2],
    kernel: CoulombKernel,
    tables: &Tables,
    workspace: &mut HermiteWorkspace,
) -> f64 {
    let n_cart = primitives.n_contracted() * n_cartesian(ls[0]) * n_cartesian(ls[1]);
    let mut block = vec![0.0; n_cart * n_cart];
    let unscreened = Context {
        kernel,
        threshold: 0.0,
        envelopes: &[],
    };
    coulomb_block(
        primitives,
        ls,
        primitives,
        ls,
        &unscreened,
        1.0,
        tables,
        workspace,
        &mut block,
    );
    let [k_a, k_b] = primitives.contractions;
    let spherical = tables.to_spherical(
        &block,
        [Some(ls[0]), Some(ls[1]), Some(ls[0]), Some(ls[1])],
        [k_a, k_b, k_a, k_b],
    );
    let n_sph = primitives.n_contracted() * (2 * ls[0] + 1) * (2 * ls[1] + 1);
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
                let mut slab = vec![0.0; shell.n_functions() * row_size];
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
    /// The largest absolute row sum of each transform: a spherical function
    /// is at most this multiple of the largest Cartesian component.
    pub row_norms: Vec<f64>,
}

impl Tables {
    pub(crate) fn new(max_l: usize) -> Self {
        let transforms: Vec<Vec<f64>> = (0..=max_l).map(spherical_transformation).collect();
        let row_norms = transforms
            .iter()
            .enumerate()
            .map(|(l, transform)| {
                transform
                    .chunks(n_cartesian(l))
                    .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        Self {
            components: (0..=max_l).map(cartesian_components).collect(),
            transforms,
            row_norms,
        }
    }

    /// Transform a row-major Cartesian block with index layout
    /// `[k_a, a, k_b, b, k_c, c, k_d, d]` to spherical functions, where
    /// `ks` are the contraction counts and `ls` the angular momenta. An
    /// index of `None` is a unit partner (see [`single_shell_pairs`]): one
    /// component, left untouched. The values may be real or complex.
    pub(crate) fn to_spherical<T>(
        &self,
        block: &[T],
        ls: [Option<usize>; 4],
        ks: [usize; 4],
    ) -> Vec<T>
    where
        T: Copy + Default + AddAssign + Mul<f64, Output = T>,
    {
        let mut data = block.to_vec();
        let mut shape = [1; 8];
        for index in 0..4 {
            shape[2 * index] = ks[index];
            shape[2 * index + 1] = ls[index].map_or(1, n_cartesian);
        }
        for (index, l) in ls.iter().enumerate() {
            let Some(l) = l else {
                continue;
            };
            let axis = 2 * index + 1;
            let left: usize = shape[..axis].iter().product();
            let right: usize = shape[axis + 1..].iter().product();
            data = transform_index(&data, left, shape[axis], right, &self.transforms[*l]);
            shape[axis] = 2 * l + 1;
        }
        data
    }
}

/// Accumulate `(ab|cd)` over all primitive pairs into `out`, laid out
/// `[k_a, cart_a, k_b, cart_b, k_c, cart_c, k_d, cart_d]` over the
/// contractions and Cartesian components; `ls = [l_a, l_b]` and
/// `[l_c, l_d]` name the component sets. A primitive quartet whose bound,
/// times `scale` (the row norms of the spherical transformations) and the
/// largest contraction coefficients of its pairs, is below the context's
/// threshold is left out. Returns whether anything was added.
#[allow(
    clippy::too_many_arguments,
    reason = "inner kernel; every argument is a distinct table"
)]
pub(crate) fn coulomb_block(
    bra: &PrimitivePairs,
    ls_bra: [usize; 2],
    ket: &PrimitivePairs,
    ls_ket: [usize; 2],
    context: &Context<'_>,
    scale: f64,
    tables: &Tables,
    workspace: &mut HermiteWorkspace,
    out: &mut [f64],
) -> bool {
    let components: [&[[usize; 3]]; 4] = [
        &tables.components[ls_bra[0]],
        &tables.components[ls_bra[1]],
        &tables.components[ls_ket[0]],
        &tables.components[ls_ket[1]],
    ];
    let dims: [usize; 4] = std::array::from_fn(|i| components[i].len());
    let orders = [ls_bra[0] + ls_bra[1], ls_ket[0] + ls_ket[1]];
    let n_t = orders[0] + 1;
    let mut intermediate = std::mem::take(&mut workspace.intermediate);
    intermediate.clear();
    intermediate.resize(dims[2] * dims[3] * n_t * n_t * n_t, 0.0);
    let mut values = std::mem::take(&mut workspace.values);
    values.clear();
    values.resize(dims[0] * dims[1] * dims[2] * dims[3], 0.0);
    let mut added = false;
    for (index_bra, pair_bra) in bra.pairs.iter().enumerate() {
        for (index_ket, pair_ket) in ket.pairs.iter().enumerate() {
            if context.threshold > 0.0 {
                let bound = scale
                    * bra.coefficient_max(index_bra)
                    * ket.coefficient_max(index_ket)
                    * quartet_bound(pair_bra, pair_ket, components, context);
                if bound < context.threshold {
                    continue;
                }
            }
            added = true;
            let prefactor = primitive_quartet(
                (pair_bra, pair_ket),
                components,
                orders,
                context.kernel,
                workspace,
                &mut intermediate,
                &mut values,
            );
            accumulate_quartet(
                (bra, index_bra),
                (ket, index_ket),
                prefactor,
                &values,
                dims,
                out,
            );
        }
    }
    workspace.intermediate = intermediate;
    workspace.values = values;
    added
}

/// Bound on every Cartesian integral `(ab|cd)` of a primitive quartet for
/// the context's kernel:
/// `(2π^{5/2}/(pq√(p+q))) (1/√α) (s_hi − s_lo) exp(−(1 − ε) s_lo² R²) M`
/// with `R` the distance between the pair centres and `M` the quartet's
/// envelope, since `R_{tuv} = (1/√α) ∫_{s_lo}^{s_hi} ∂^{tuv} exp(−s² R²) ds`
/// over the exponent range of the kernel and
/// `|∂_x^n exp(−s² x²)| ≤ s^n h_n exp(−(1 − ε) s² x²)` (ADR-0018).
fn quartet_bound(
    pair_bra: &PrimitivePair,
    pair_ket: &PrimitivePair,
    components: [&[[usize; 3]]; 4],
    context: &Context<'_>,
) -> f64 {
    let (p, q) = (pair_bra.p, pair_ket.p);
    let alpha = p * q / (p + q);
    let (low, _) = context.kernel.exponent_range(alpha);
    let distance_squared: f64 = (0..3)
        .map(|axis| (pair_bra.center[axis] - pair_ket.center[axis]).powi(2))
        .sum();
    let decay = (-(1.0 - ENVELOPE_EPSILON) * low * low * distance_squared).exp();
    decay * quartet_bound_at_zero(pair_bra, pair_ket, components, context)
}

/// `max_abcd Π_axis Σ_{t ≤ i_a + i_b} Σ_{τ ≤ i_c + i_d} |E^{ab}_t E^{cd}_τ|
/// s^{t + τ} h_{t + τ}` over the Cartesian components of the quartet.
fn quartet_envelope(
    pair_bra: &PrimitivePair,
    pair_ket: &PrimitivePair,
    components: [&[[usize; 3]]; 4],
    s: f64,
    envelopes: &[f64],
) -> f64 {
    let ls: [usize; 4] = std::array::from_fn(|i| components[i][0].iter().sum());
    let extents: [usize; 4] = std::array::from_fn(|i| ls[i] + 1);
    let weights: Vec<f64> = envelopes
        .iter()
        .enumerate()
        .take(ls.iter().sum::<usize>() + 1)
        .map(|(n, h)| s.powi(i32::try_from(n).expect("order fits in i32")) * h)
        .collect();
    let per_axis = extents.iter().product::<usize>();
    let index = |ia: usize, ib: usize, ic: usize, id: usize| {
        ((ia * extents[1] + ib) * extents[2] + ic) * extents[3] + id
    };
    let mut table = vec![0.0; 3 * per_axis];
    for axis in 0..3 {
        let (e_bra, e_ket) = (&pair_bra.hermite[axis], &pair_ket.hermite[axis]);
        for ia in 0..extents[0] {
            for ib in 0..extents[1] {
                for ic in 0..extents[2] {
                    for id in 0..extents[3] {
                        let mut sum = 0.0;
                        for t in 0..=ia + ib {
                            let left = e_bra.get(ia, ib, t).abs();
                            for tau in 0..=ic + id {
                                sum += left * e_ket.get(ic, id, tau).abs() * weights[t + tau];
                            }
                        }
                        table[axis * per_axis + index(ia, ib, ic, id)] = sum;
                    }
                }
            }
        }
    }
    let [comp_a, comp_b, comp_c, comp_d] = components;
    let mut largest = 0.0_f64;
    for ca in comp_a {
        for cb in comp_b {
            for cc in comp_c {
                for cd in comp_d {
                    let product: f64 = (0..3)
                        .map(|axis| {
                            table[axis * per_axis + index(ca[axis], cb[axis], cc[axis], cd[axis])]
                        })
                        .product();
                    largest = largest.max(product);
                }
            }
        }
    }
    largest
}

/// `values[ab][cd] = Σ_{tuv} E^{ab}_{tuv} Σ_{τνφ} (−1)^{τ+ν+φ} E^{cd}_{τνφ}
/// R_{t+τ, u+ν, v+φ}` for one primitive quartet, returning the prefactor
/// `2π^{5/2} / (pq √(p+q))`; the Hermite table carries the kernel.
fn primitive_quartet(
    pairs: (&PrimitivePair, &PrimitivePair),
    components: [&[[usize; 3]]; 4],
    orders: [usize; 2],
    kernel: CoulombKernel,
    workspace: &mut HermiteWorkspace,
    intermediate: &mut [f64],
    values: &mut [f64],
) -> f64 {
    let (pair_bra, pair_ket) = pairs;
    let [comp_a, comp_b, comp_c, comp_d] = components;
    let n_t = orders[0] + 1;
    let n_cd = comp_c.len() * comp_d.len();
    let (p, q) = (pair_bra.p, pair_ket.p);
    let alpha = p * q / (p + q);
    let prefactor = 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt());
    let pq: [f64; 3] = std::array::from_fn(|axis| pair_bra.center[axis] - pair_ket.center[axis]);
    let r = workspace.compute_kernel(orders[0] + orders[1], alpha, kernel, pq);
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
                                let sign = if (tau + nu + phi) % 2 == 0 { 1.0 } else { -1.0 };
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
                        value +=
                            pair_bra.product(ca, cb, [t, u, v]) * slot[(t * n_t + u) * n_t + v];
                    }
                }
            }
            values[iab * n_cd + icd] = value;
        }
    }
    prefactor
}

/// Add `prefactor * values[ab][cd]` of one primitive quartet to `out`
/// (laid out `[k_a, a, k_b, b, k_c, c, k_d, d]`) for every combination of
/// contractions, scaled by their coefficients.
fn accumulate_quartet(
    bra: (&PrimitivePairs, usize),
    ket: (&PrimitivePairs, usize),
    prefactor: f64,
    values: &[f64],
    dims: [usize; 4],
    out: &mut [f64],
) {
    let [n_a, n_b, n_c, n_d] = dims;
    let [bra_first, bra_second] = bra.0.contractions;
    let [ket_first, ket_second] = ket.0.contractions;
    let stride_c = ket_second * n_d;
    let stride_b = ket_first * n_c * stride_c;
    let stride_a = bra_second * n_b * stride_b;
    for k_a in 0..bra_first {
        for k_b in 0..bra_second {
            let bra_coefficient = prefactor * bra.0.coefficient(bra.1, k_a, k_b);
            for k_c in 0..ket_first {
                for k_d in 0..ket_second {
                    let coefficient = bra_coefficient * ket.0.coefficient(ket.1, k_c, k_d);
                    let base = k_a * n_a * stride_a
                        + k_b * n_b * stride_b
                        + k_c * n_c * stride_c
                        + k_d * n_d;
                    for i_a in 0..n_a {
                        for i_b in 0..n_b {
                            let target = base + i_a * stride_a + i_b * stride_b;
                            let source = (i_a * n_b + i_b) * n_c * n_d;
                            for i_c in 0..n_c {
                                let row = &mut out[target + i_c * stride_c..][..n_d];
                                let column = &values[source + i_c * n_d..][..n_d];
                                for (o, v) in row.iter_mut().zip(column) {
                                    *o += coefficient * v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
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

pub(crate) fn check_threshold(threshold: f64) -> Result<(), Error> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(Error::InvalidArgument(
            "the screening threshold is finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

/// Reject non-finite vectors, naming the quantity in the error.
pub(crate) fn check_finite(vectors: &[[f64; 3]], name: &str) -> Result<(), Error> {
    if vectors
        .iter()
        .all(|vector| vector.iter().all(|x| x.is_finite()))
    {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!("{name} are finite")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrals::nuclear_attraction;
    use crate::integrals::test_tables::{
        general_table, s_shell, sp_table, sp_table_at, split_table,
    };

    fn assert_same(general: &[f64], split: &[f64]) {
        assert_eq!(general.len(), split.len());
        for (g, s) in general.iter().zip(split) {
            assert!((g - s).abs() < 1e-13, "{g} vs {s}");
        }
    }

    #[test]
    fn general_contractions_match_split_shells() {
        let (general, split) = (general_table(), split_table());
        let translation = [0.3, -0.1, 0.2];
        assert_same(
            &coulomb_two_center(
                &general,
                &general,
                &[ZERO, translation],
                CoulombKernel::LongRange(0.8),
                0.0,
            )
            .expect("valid"),
            &coulomb_two_center(
                &split,
                &split,
                &[ZERO, translation],
                CoulombKernel::LongRange(0.8),
                0.0,
            )
            .expect("valid"),
        );
        let images = [[ZERO, ZERO], [translation, ZERO], [ZERO, translation]];
        assert_same(
            &coulomb_three_center(
                &general,
                &general,
                &general,
                &images,
                CoulombKernel::Full,
                0.0,
            )
            .expect("valid"),
            &coulomb_three_center(&split, &split, &split, &images, CoulombKernel::Full, 0.0)
                .expect("valid"),
        );
        // Mixed tables take the generic path on the split side as well.
        assert_same(
            &coulomb_three_center(&split, &general, &split, &images, CoulombKernel::Full, 0.0)
                .expect("valid"),
            &coulomb_three_center(&split, &split, &split, &images, CoulombKernel::Full, 0.0)
                .expect("valid"),
        );
        let quartet_images = [[ZERO; 3], [translation, ZERO, translation]];
        for threshold in [0.0, 1e-12] {
            assert_same(
                &coulomb_four_center(
                    [&general, &general],
                    [&general, &general],
                    &quartet_images,
                    CoulombKernel::Full,
                    threshold,
                )
                .expect("valid"),
                &coulomb_four_center(
                    [&split, &split],
                    [&split, &split],
                    &quartet_images,
                    CoulombKernel::Full,
                    threshold,
                )
                .expect("valid"),
            );
        }
        assert_same(
            &coulomb_four_center(
                [&general, &split],
                [&split, &general],
                &quartet_images,
                CoulombKernel::LongRange(0.6),
                0.0,
            )
            .expect("valid"),
            &coulomb_four_center(
                [&split, &split],
                [&split, &split],
                &quartet_images,
                CoulombKernel::LongRange(0.6),
                0.0,
            )
            .expect("valid"),
        );
    }

    #[test]
    fn schwarz_bound_of_a_general_shell_is_the_largest_of_its_contractions() {
        let (general, split) = (general_table(), split_table());
        let bounds =
            schwarz_bounds(&general, &general, &[ZERO], CoulombKernel::Full).expect("valid");
        let fine = schwarz_bounds(&split, &split, &[ZERO], CoulombKernel::Full).expect("valid");
        // Split shells 0..3 are the s contractions, 3..5 the p contractions.
        let members = [0..3, 3..5];
        for (i, rows) in members.iter().enumerate() {
            for (j, columns) in members.iter().enumerate() {
                let mut expected = 0.0_f64;
                for r in rows.clone() {
                    for c in columns.clone() {
                        expected = expected.max(fine[r * 5 + c]);
                    }
                }
                assert!((bounds[i * 2 + j] - expected).abs() < 1e-13);
            }
        }
    }

    /// The four-centre integrals of one image without any symmetry shortcut.
    fn generic_image(table: &ShellTable, translations: [[f64; 3]; 3]) -> Vec<f64> {
        let n = table.n_functions();
        let mut out = vec![0.0; n * n * n * n];
        FourCenterImage {
            bra: [table, table],
            ket: [table, table],
            translations,
            context: Context {
                kernel: CoulombKernel::Full,
                threshold: 0.0,
                envelopes: &[],
            },
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
            CoulombKernel::Full,
            threshold,
        )
        .expect("valid")
    }

    #[test]
    fn two_center_s_integral_matches_the_closed_form() {
        let (a, b) = (0.9, 0.4);
        let bra = s_shell(a, [0.0; 3]);
        let ket = s_shell(b, [0.0, 0.0, 1.3]);
        let value = coulomb_two_center(&bra, &ket, &[[0.0; 3]], CoulombKernel::Full, 0.0)
            .expect("valid")[0];
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
        let forward = coulomb_two_center(&bra, &ket, &[[0.0; 3]], CoulombKernel::Full, 0.0)
            .expect("valid")[0];
        let backward = coulomb_two_center(&ket, &bra, &[[0.0; 3]], CoulombKernel::Full, 0.0)
            .expect("valid")[0];
        assert!((forward - backward).abs() < 1e-14);
        let moved = s_shell(0.4, [1.5, -0.3, 1.0]);
        let translated =
            coulomb_two_center(&bra, &ket, &[[1.0, 0.0, 0.0]], CoulombKernel::Full, 0.0)
                .expect("valid")[0];
        let direct = coulomb_two_center(&bra, &moved, &[[0.0; 3]], CoulombKernel::Full, 0.0)
            .expect("valid")[0];
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
        let three = coulomb_three_center(
            &aux,
            &bra,
            &ket,
            &[[[0.0; 3], [0.0; 3]]],
            CoulombKernel::Full,
            0.0,
        )
        .expect("valid")[0];
        let attraction = nuclear_attraction(
            &bra,
            &ket,
            &[[0.0; 3]],
            &[1.0],
            &[[0.3, -0.2, 0.5]],
            crate::integrals::CoulombKernel::LongRange(q.sqrt()),
            0.0,
        )
        .expect("valid")[0];
        let expected = -attraction / (q / (2.0 * PI)).powf(0.75);
        assert!((three - expected).abs() < 1e-13, "{three} vs {expected}");
    }

    #[test]
    fn three_center_symmetric_path_matches_the_generic_path() {
        let table = sp_table();
        let aux = s_shell(0.5, [0.2, 0.1, -0.1]);
        let symmetric = coulomb_three_center(
            &aux,
            &table,
            &table,
            &[[ZERO, ZERO]],
            CoulombKernel::Full,
            0.0,
        )
        .expect("valid");
        // A ket with a different centre list is a different table, so the
        // generic path runs; moving it back by a translation gives the same
        // integrals.
        let shifted = sp_table_at([1.0; 3]);
        let generic = coulomb_three_center(
            &aux,
            &table,
            &shifted,
            &[[ZERO, [-1.0; 3]]],
            CoulombKernel::Full,
            0.0,
        )
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
            CoulombKernel::Full,
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
                context: Context {
                    kernel: CoulombKernel::Full,
                    threshold: 0.0,
                    envelopes: &[],
                },
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
                coulomb_four_center(same, same, &[translations], CoulombKernel::Full, 0.0)
                    .expect("valid");
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
        let translated =
            schwarz_bounds(&table, &table, &[shift], CoulombKernel::Full).expect("valid");
        let direct = schwarz_bounds(&table, &sp_table_at(shift), &[ZERO], CoulombKernel::Full)
            .expect("valid");
        let at_origin =
            schwarz_bounds(&table, &table, &[ZERO], CoulombKernel::Full).expect("valid");
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
        let translated = coulomb_four_center(
            [&bra, &ket],
            [&bra, &ket],
            &[[t_b, t_c, t_d]],
            CoulombKernel::Full,
            0.0,
        )
        .expect("valid")[0];
        let moved_b = s_shell(0.4, [1.5, -0.3, 1.0]);
        let moved_c = s_shell(0.9, [0.1, 0.7, 0.3]);
        let moved_d = s_shell(0.4, [0.5, -0.3, 0.3]);
        let direct = coulomb_four_center(
            [&bra, &moved_b],
            [&moved_c, &moved_d],
            &[[ZERO; 3]],
            CoulombKernel::Full,
            0.0,
        )
        .expect("valid")[0];
        assert!((translated - direct).abs() < 1e-14);
    }

    #[test]
    fn schwarz_bounds_bound_every_quartet() {
        let table = sp_table();
        let bounds = schwarz_bounds(&table, &table, &[ZERO], CoulombKernel::Full).expect("valid");
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
            vec![1; 2],
        )
        .expect("valid");
        let exact = molecular(&far, 0.0);
        let screened = molecular(&far, 1e-6);
        let bounds = schwarz_bounds(&far, &far, &[ZERO], CoulombKernel::Full).expect("valid");
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
        let full = coulomb_two_center(&bra, &ket, &[[0.0; 3]], CoulombKernel::Full, 0.0)
            .expect("valid")[0];
        let long_range =
            coulomb_two_center(&bra, &ket, &[[0.0; 3]], CoulombKernel::LongRange(0.5), 0.0)
                .expect("valid")[0];
        let nearly =
            coulomb_two_center(&bra, &ket, &[[0.0; 3]], CoulombKernel::LongRange(1e4), 0.0)
                .expect("valid")[0];
        assert!(long_range < full && long_range > 0.0);
        assert!((nearly - full).abs() < 1e-7);
    }

    #[test]
    fn rejects_bad_arguments() {
        let shell = s_shell(1.0, [0.0; 3]);
        assert!(
            coulomb_two_center(
                &shell,
                &shell,
                &[[0.0; 3]],
                CoulombKernel::LongRange(-1.0),
                0.0
            )
            .is_err()
        );
        assert!(
            coulomb_two_center(&shell, &shell, &[[f64::NAN; 3]], CoulombKernel::Full, 0.0).is_err()
        );
        assert!(
            coulomb_three_center(
                &shell,
                &shell,
                &shell,
                &[[[0.0; 3], [f64::INFINITY; 3]]],
                CoulombKernel::Full,
                0.0
            )
            .is_err()
        );
        let pair = [&shell, &shell];
        assert!(coulomb_four_center(pair, pair, &[[ZERO; 3]], CoulombKernel::Full, -1.0).is_err());
        assert!(
            coulomb_four_center(pair, pair, &[[ZERO; 3]], CoulombKernel::Full, f64::NAN).is_err()
        );
        assert!(
            coulomb_four_center(
                pair,
                pair,
                &[[ZERO, ZERO, [f64::NAN; 3]]],
                CoulombKernel::Full,
                0.0
            )
            .is_err()
        );
        assert!(schwarz_bounds(&shell, &shell, &[ZERO], CoulombKernel::LongRange(0.0)).is_err());
    }

    #[test]
    fn short_range_kernels_are_the_full_minus_the_long_range_ones() {
        let table = general_table();
        let aux = sp_table();
        let translation = [0.4, -0.3, 0.6];
        let evaluate = |kernel| {
            (
                coulomb_two_center(&aux, &aux, &[ZERO, translation], kernel, 0.0).expect("valid"),
                coulomb_three_center(&aux, &table, &table, &[[ZERO, translation]], kernel, 0.0)
                    .expect("valid"),
                coulomb_four_center(
                    [&table, &table],
                    [&table, &table],
                    &[[ZERO, translation, ZERO]],
                    kernel,
                    0.0,
                )
                .expect("valid"),
            )
        };
        let full = evaluate(CoulombKernel::Full);
        let long = evaluate(CoulombKernel::LongRange(0.7));
        let short = evaluate(CoulombKernel::ShortRange(0.7));
        for (f, (l, s)) in full.0.iter().zip(long.0.iter().zip(&short.0)) {
            assert!((f - l - s).abs() < 1e-13, "{f} vs {l} + {s}");
        }
        for (f, (l, s)) in full.1.iter().zip(long.1.iter().zip(&short.1)) {
            assert!((f - l - s).abs() < 1e-13, "{f} vs {l} + {s}");
        }
        for (f, (l, s)) in full.2.iter().zip(long.2.iter().zip(&short.2)) {
            assert!((f - l - s).abs() < 1e-13, "{f} vs {l} + {s}");
        }
    }

    #[test]
    fn screening_neglects_only_quartets_below_the_threshold() {
        let table = general_table();
        let aux = sp_table();
        let threshold = 1e-10;
        let translations = [ZERO, [7.0, -2.0, 1.0], [60.0, 0.0, 0.0]];
        for kernel in [CoulombKernel::Full, CoulombKernel::ShortRange(0.5)] {
            let exact = coulomb_two_center(&aux, &aux, &translations, kernel, 0.0).expect("valid");
            let screened =
                coulomb_two_center(&aux, &aux, &translations, kernel, threshold).expect("valid");
            let largest = exact
                .iter()
                .zip(&screened)
                .map(|(e, s)| (e - s).abs())
                .fold(0.0_f64, f64::max);
            assert!(largest < 10.0 * threshold, "{kernel:?}: {largest}");
            let pairs: Vec<[[f64; 3]; 2]> = translations.iter().map(|t| [ZERO, *t]).collect();
            let exact =
                coulomb_three_center(&aux, &table, &table, &pairs, kernel, 0.0).expect("valid");
            let screened = coulomb_three_center(&aux, &table, &table, &pairs, kernel, threshold)
                .expect("valid");
            let largest = exact
                .iter()
                .zip(&screened)
                .map(|(e, s)| (e - s).abs())
                .fold(0.0_f64, f64::max);
            assert!(largest < 10.0 * threshold, "{kernel:?}: {largest}");
            let per_image = aux.n_functions() * table.n_functions() * table.n_functions();
            if kernel != CoulombKernel::Full {
                // The far image is screened away entirely by the short-range kernel.
                assert!(screened[2 * per_image..].iter().all(|v| *v == 0.0));
                assert!(exact[2 * per_image..].iter().any(|v| v.abs() > 0.0));
            }
        }
        let nothing =
            coulomb_two_center(&aux, &aux, &translations, CoulombKernel::Full, 1e6).expect("valid");
        assert!(nothing.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn quartet_bound_holds_for_higher_angular_momenta() {
        let tables = Tables::new(3);
        let envelopes = hermite_envelopes(6, ENVELOPE_EPSILON);
        for (la, lb, lc, exponents) in [
            (0, 2, 1, [0.9, 0.3, 0.5]),
            (1, 1, 3, [1.4, 0.5, 0.8]),
            (2, 2, 0, [0.4, 0.6, 1.1]),
        ] {
            let center_b = [0.8, -0.4, 0.5];
            let make = |l: usize, center: [f64; 3], exponent: f64| {
                ShellTable::new(
                    vec![l],
                    vec![center],
                    vec![0, 1],
                    vec![exponent],
                    vec![1.0],
                    vec![1],
                )
                .expect("valid")
            };
            let bra = make(la, [0.0; 3], exponents[0]);
            let ket = make(lb, center_b, exponents[1]);
            let pairs = primitive_pairs(&bra.shell(0), &ket.shell(0), center_b, 0);
            let components: [&[[usize; 3]]; 4] = [
                &tables.components[la],
                &tables.components[lb],
                &tables.components[lc],
                &tables.components[0],
            ];
            let scale = tables.row_norms[la] * tables.row_norms[lb] * tables.row_norms[lc];
            for kernel in [
                CoulombKernel::ShortRange(0.4),
                CoulombKernel::ShortRange(1.5),
            ] {
                let context = Context {
                    kernel,
                    threshold: 1e-300,
                    envelopes: &envelopes,
                };
                for distance in [0.0, 1.5, 3.0, 6.0] {
                    let aux_center = [
                        pairs.pairs[0].center[0] + 0.6 * distance,
                        pairs.pairs[0].center[1] - 0.8 * distance,
                        pairs.pairs[0].center[2],
                    ];
                    let aux = make(lc, aux_center, exponents[2]);
                    let aux_pairs = single_shell_pairs(&aux.shell(0), aux_center);
                    let bound = scale
                        * quartet_bound(&pairs.pairs[0], &aux_pairs.pairs[0], components, &context);
                    let values =
                        coulomb_three_center(&aux, &bra, &ket, &[[ZERO, ZERO]], kernel, 0.0)
                            .expect("valid");
                    let largest = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
                    assert!(
                        largest <= bound,
                        "l = {la},{lb},{lc} R = {distance}: {largest} > {bound}"
                    );
                    // The bound is loose by a factor that grows with the angular
                    // momenta and enters the screening radius logarithmically;
                    // at some distances the integrals vanish by symmetry.
                    if largest > 1e-6 {
                        assert!(bound < 1e7 * largest, "{bound} vs {largest}");
                    }
                }
            }
        }
    }

    /// The Bloch-summed three-centre integrals from the explicit image
    /// pairs: `Σ_{ij} b[k1, i] c[k2, j] (P | μ(T_i) ν(T_j))` with the aux at
    /// home expressed through the plain kernel as `(P(−T_i) | μ ν(T_j − T_i))`.
    fn explicit_bloch(
        aux: &ShellTable,
        table: &ShellTable,
        images: &[[f64; 3]],
        phases: [&[Complex64]; 2],
        n_k: usize,
        kernel: CoulombKernel,
    ) -> Vec<Complex64> {
        let n = table.n_functions();
        let block = aux.n_functions() * n * n;
        let mut out = vec![Complex64::default(); n_k * n_k * block];
        for (i, t_i) in images.iter().enumerate() {
            for (j, t_j) in images.iter().enumerate() {
                let minus: [f64; 3] = std::array::from_fn(|k| -t_i[k]);
                let shifted: [f64; 3] = std::array::from_fn(|k| t_j[k] - t_i[k]);
                let values =
                    coulomb_three_center(aux, table, table, &[[minus, shifted]], kernel, 0.0)
                        .expect("valid");
                for k1 in 0..n_k {
                    for k2 in 0..n_k {
                        let phase =
                            phases[0][k1 * images.len() + i] * phases[1][k2 * images.len() + j];
                        for (o, v) in out[(k1 * n_k + k2) * block..][..block]
                            .iter_mut()
                            .zip(&values)
                        {
                            *o += phase * v;
                        }
                    }
                }
            }
        }
        out
    }

    /// Images on a rectangular grid with their integer coordinates, every
    /// difference of two images as a step, and the phases of two k-points.
    struct BlochCase {
        images: Vec<[f64; 3]>,
        coordinates: Vec<[i64; 3]>,
        steps: Vec<[f64; 3]>,
        step_coordinates: Vec<[i64; 3]>,
        bra_phases: Vec<Complex64>,
        ket_phases: Vec<Complex64>,
    }

    impl BlochCase {
        fn new() -> Self {
            let coordinates: Vec<[i64; 3]> =
                vec![[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 1, 0]];
            let images: Vec<[f64; 3]> = vec![
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [-2.5, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [2.5, 3.0, 0.0],
            ];
            let mut step_coordinates: Vec<[i64; 3]> = Vec::new();
            let mut steps: Vec<[f64; 3]> = Vec::new();
            for (a, ta) in coordinates.iter().zip(&images) {
                for (b, tb) in coordinates.iter().zip(&images) {
                    let step = [b[0] - a[0], b[1] - a[1], 0];
                    if !step_coordinates.contains(&step) {
                        step_coordinates.push(step);
                        steps.push([tb[0] - ta[0], tb[1] - ta[1], 0.0]);
                    }
                }
            }
            let kpts = [[0.0, 0.0, 0.0], [0.3, -0.2, 0.1]];
            let phase = |k: [f64; 3], t: [f64; 3], sign: f64| {
                Complex64::cis(sign * (0..3).map(|a| k[a] * t[a]).sum::<f64>())
            };
            let bra_phases = kpts
                .iter()
                .flat_map(|k| images.iter().map(move |t| phase(*k, *t, -1.0)))
                .collect();
            let ket_phases = kpts
                .iter()
                .flat_map(|k| images.iter().map(move |t| phase(*k, *t, 1.0)))
                .collect();
            Self {
                images,
                coordinates,
                steps,
                step_coordinates,
                bra_phases,
                ket_phases,
            }
        }

        fn sum(&self) -> BlochSum<'_> {
            BlochSum {
                images: &self.images,
                image_coordinates: &self.coordinates,
                steps: &self.steps,
                step_coordinates: &self.step_coordinates,
                phases: [&self.bra_phases, &self.ket_phases],
            }
        }
    }

    #[test]
    fn bloch_three_center_matches_the_explicit_image_sum() {
        let table = general_table();
        let aux = sp_table();
        let case = BlochCase::new();
        let n_k = 2;
        for kernel in [CoulombKernel::Full, CoulombKernel::ShortRange(0.6)] {
            let summed = coulomb_three_center_bloch(&aux, &table, &table, case.sum(), kernel, 0.0)
                .expect("valid");
            let expected = explicit_bloch(
                &aux,
                &table,
                &case.images,
                [&case.bra_phases, &case.ket_phases],
                n_k,
                kernel,
            );
            assert_eq!(summed.len(), expected.len());
            for (s, e) in summed.iter().zip(&expected) {
                assert!((s - e).norm() < 1e-12, "{s} vs {e}");
            }
            // Hermiticity across the swap of the k-points and the functions.
            let n = table.n_functions();
            let n_aux = aux.n_functions();
            for k1 in 0..n_k {
                for k2 in 0..n_k {
                    for p in 0..n_aux {
                        for u in 0..n {
                            for v in 0..n {
                                let direct =
                                    summed[((k1 * n_k + k2) * n_aux + p) * n * n + u * n + v];
                                let swapped =
                                    summed[((k2 * n_k + k1) * n_aux + p) * n * n + v * n + u];
                                assert!((direct - swapped.conj()).norm() < 1e-12);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn bloch_three_center_screens_and_validates() {
        let table = general_table();
        let aux = sp_table();
        let case = BlochCase::new();
        let short = CoulombKernel::ShortRange(0.6);
        // A screened sum stays within the threshold of the full one.
        let threshold = 1e-10;
        let exact = coulomb_three_center_bloch(&aux, &table, &table, case.sum(), short, 0.0)
            .expect("valid");
        let screened =
            coulomb_three_center_bloch(&aux, &table, &table, case.sum(), short, threshold)
                .expect("valid");
        let largest = exact
            .iter()
            .zip(&screened)
            .map(|(e, s)| (e - s).norm())
            .fold(0.0_f64, f64::max);
        assert!(largest < 100.0 * threshold, "{largest}");
        // Steps that reach no image contribute nothing.
        let far_steps = [[50.0, 0.0, 0.0]];
        let far_coordinates = [[20, 0, 0]];
        let nothing = coulomb_three_center_bloch(
            &aux,
            &table,
            &table,
            BlochSum {
                steps: &far_steps,
                step_coordinates: &far_coordinates,
                ..case.sum()
            },
            short,
            0.0,
        )
        .expect("valid");
        assert!(nothing.iter().all(|v| v.norm() == 0.0));
        // Malformed lists are rejected.
        let repeated = [[0, 0, 0], [0, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 1, 0]];
        let bad = BlochSum {
            image_coordinates: &repeated,
            ..case.sum()
        };
        assert!(coulomb_three_center_bloch(&aux, &table, &table, bad, short, 0.0).is_err());
        let short_phases = BlochSum {
            phases: [&case.bra_phases[1..], &case.ket_phases],
            ..case.sum()
        };
        assert!(
            coulomb_three_center_bloch(&aux, &table, &table, short_phases, short, 0.0).is_err()
        );
        let fewer = BlochSum {
            step_coordinates: &case.step_coordinates[1..],
            ..case.sum()
        };
        assert!(coulomb_three_center_bloch(&aux, &table, &table, fewer, short, 0.0).is_err());
        let mut stretched = case.steps.clone();
        stretched[1][0] += 0.5;
        let inconsistent = BlochSum {
            steps: &stretched,
            ..case.sum()
        };
        assert!(
            coulomb_three_center_bloch(&aux, &table, &table, inconsistent, short, 0.0).is_err()
        );
    }
}

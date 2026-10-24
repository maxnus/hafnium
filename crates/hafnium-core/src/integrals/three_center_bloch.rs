//! Three-centre Coulomb integrals summed over the lattice images of the
//! orbital functions of a cell with Bloch phases (ADR-0019, ADR-0025).
//!
//! `(P^q | μ k1 ν k2) = Σ_i Σ_j exp(−i k1·T_i) exp(i k2·T_j)
//! (P | μ(r − T_i) ν(r − T_j))` with the auxiliary function `P` in the
//! home cell. The ket image of a pair is the bra image displaced by one of
//! the steps at which two functions still overlap, `T_j = T_i + T_s`, so
//! the phases factor as `exp(i q·T_i) exp(i k2·T_s)` with the momentum
//! transfer `q = k2 − k1`. On a uniform mesh `exp(i q·T)` is the same for
//! every image of one Born–von Kármán class, the lattice coordinates
//! modulo the mesh, and `exp(i k2·T_s)` the same for every step of one
//! class, so the real integrals are accumulated per class of the image
//! and class of the step and the two transforms over the mesh are applied
//! once at the end, auxiliary shell by auxiliary shell (ADR-0028). The
//! Coulomb matrix needs only the blocks of zero momentum transfer,
//! `(P^0 | μ k ν k)`, for which the class of the image does not matter:
//! [`Transfers::Zero`] accumulates per class of the step alone and returns
//! one block per k-point. The term with `μ` and `ν` exchanged is the same
//! integral at the mirrored step, so when the bra and the ket coincide each
//! pair of images is evaluated once and the sum is completed by its
//! conjugate transpose over k-points and functions.
//!
//! The screening tabulates, once per step, the reach of every shell
//! triple and primitive quartet: the distance beyond which the bound of
//! ADR-0018 falls below the threshold. The image loop then compares
//! squared distances with the tables and evaluates nothing it can prove
//! negligible.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use num_complex::Complex64;
use rayon::prelude::*;

use super::cartesian::n_cartesian;
use super::coulomb::{CoulombKernel, HermiteWorkspace};
use super::pairs::{ENVELOPE_EPSILON, PrimitivePairs, single_shell_pairs};
use super::two_electron::{
    Context, Tables, ZERO, accumulate_quartet, check_finite, check_threshold, envelopes_for,
    primitive_quartet, quartet_bound_at_zero, schwarz_bound, shell_pairs,
};
use super::{ShellRef, ShellTable};
use crate::Error;

/// Largest difference in Bohr between an image vector and the one its
/// lattice coordinates predict.
const COORDINATE_TOLERANCE: f64 = 1e-8;

/// Largest deviation of `exp(i q·(T − T_c))` from one within a class.
const CLASS_TOLERANCE: f64 = 1e-8;

/// The images, steps, classes and k-points of a Bloch-summed three-centre
/// integral.
///
/// The orbital functions are imaged over `images`, with the integer
/// lattice coordinates `image_coordinates` of every image; the ket image
/// of a pair is the bra image displaced by one of the `steps`, the
/// relative translations at which a pair of functions still overlaps,
/// found by their coordinates. Every image belongs to a class such that
/// `exp(i (k2 − k1)·T)` is the same for all images `T` of the class and
/// every pair of k-points, and every step to a class such that
/// `exp(i k·T_s)` is the same for all steps of the class and every
/// k-point; on a uniform mesh both are the lattice coordinates modulo the
/// mesh.
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
    /// The class of every image.
    pub image_classes: &'a [usize],
    /// The class of every step.
    pub step_classes: &'a [usize],
    /// Cartesian k-points `[k, 3]` in inverse Bohr.
    pub kpoints: &'a [[f64; 3]],
}

/// Which blocks of momentum transfer a Bloch sum returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transfers {
    /// Every pair of k-points, `[k1, k2, aux, bra, ket]`.
    All,
    /// Zero transfer only, `[k, aux, bra, ket]`: what the Coulomb matrix needs.
    Zero,
}

impl BlochSum<'_> {
    /// Validate the lists and return the image and step classes relabelled
    /// densely, in order of first appearance, so that their counts are the
    /// numbers of distinct classes whatever the indices given.
    fn check(&self) -> Result<(Vec<usize>, Vec<usize>), Error> {
        let invalid = |message: &str| Err(Error::InvalidArgument(message.to_string()));
        check_finite(self.images, "translations")?;
        check_finite(self.steps, "translations")?;
        check_finite(self.kpoints, "k-points")?;
        let n_images = self.images.len();
        if n_images == 0 {
            return invalid("a Bloch sum needs at least one image");
        }
        if self.image_coordinates.len() != n_images
            || self.step_coordinates.len() != self.steps.len()
        {
            return invalid("every image and every step has one set of lattice coordinates");
        }
        if self.image_classes.len() != n_images {
            return invalid("every image has one class");
        }
        if self.step_classes.len() != self.steps.len() {
            return invalid("every step has one class");
        }
        // The transfer phases `exp(i (k2 − k1)·T)` are constant on an image
        // class, the step phases `exp(i k·T_s)` on a step class; the latter
        // implies the former, so both lists are checked against every pair.
        let image_classes = dense_classes(
            self.image_classes,
            self.images,
            self.kpoints,
            true,
            "the phase of the momentum transfer varies within an image class",
        )?;
        let step_classes = dense_classes(
            self.step_classes,
            self.steps,
            self.kpoints,
            false,
            "the phase of a k-point varies within a step class",
        )?;
        Ok((image_classes, step_classes))
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

    /// Whether `−s` is a step for every step `s`.
    fn steps_closed_under_negation(&self) -> bool {
        let set: HashSet<[i64; 3]> = self.step_coordinates.iter().copied().collect();
        self.step_coordinates
            .iter()
            .all(|c| set.contains(&[-c[0], -c[1], -c[2]]))
    }

    /// `exp(i k·T_c)` as `[class][k]`, from the first step of every class
    /// of the dense labels of [`Self::check`].
    fn step_phases(&self, n_classes: usize) -> Vec<Complex64> {
        let mut representative: Vec<Option<[f64; 3]>> = vec![None; n_classes];
        for (s, &class) in self.step_classes.iter().enumerate() {
            representative[class].get_or_insert(self.steps[s]);
        }
        let n_k = self.kpoints.len();
        let mut phases = vec![Complex64::default(); n_classes * n_k];
        for (class, translation) in representative.iter().enumerate() {
            if let Some(translation) = translation {
                for (k, point) in self.kpoints.iter().enumerate() {
                    let angle: f64 = (0..3).map(|axis| point[axis] * translation[axis]).sum();
                    phases[class * n_k + k] = Complex64::cis(angle);
                }
            }
        }
        phases
    }

    /// `exp(i (k2 − k1)·T_c)` as `[k1][k2][class]`, from the first image of
    /// every class of the dense labels of [`Self::check`].
    fn class_phases(&self, n_classes: usize) -> Vec<Complex64> {
        let mut representative: Vec<Option<[f64; 3]>> = vec![None; n_classes];
        for (i, &class) in self.image_classes.iter().enumerate() {
            representative[class].get_or_insert(self.images[i]);
        }
        let n_k = self.kpoints.len();
        let mut phases = vec![Complex64::default(); n_k * n_k * n_classes];
        for (k1, point_1) in self.kpoints.iter().enumerate() {
            for (k2, point_2) in self.kpoints.iter().enumerate() {
                for (class, translation) in representative.iter().enumerate() {
                    if let Some(translation) = translation {
                        let angle: f64 = (0..3)
                            .map(|axis| (point_2[axis] - point_1[axis]) * translation[axis])
                            .sum();
                        phases[(k1 * n_k + k2) * n_classes + class] = Complex64::cis(angle);
                    }
                }
            }
        }
        phases
    }
}

/// Relabel `classes` densely in order of first appearance, checking that
/// the phase of every k-point (`transfer` false) or of every momentum
/// transfer (`transfer` true) is the same on all `vectors` of one class.
fn dense_classes(
    classes: &[usize],
    vectors: &[[f64; 3]],
    kpoints: &[[f64; 3]],
    transfer: bool,
    message: &str,
) -> Result<Vec<usize>, Error> {
    let mut dense: HashMap<usize, usize> = HashMap::new();
    let mut representatives: Vec<usize> = Vec::new();
    let mut relabelled = Vec::with_capacity(classes.len());
    let origin = [0.0; 3];
    for (i, &class) in classes.iter().enumerate() {
        let next = representatives.len();
        let index = *dense.entry(class).or_insert(next);
        if index == next {
            representatives.push(i);
        }
        relabelled.push(index);
        let representative = representatives[index];
        let difference: [f64; 3] =
            std::array::from_fn(|axis| vectors[i][axis] - vectors[representative][axis]);
        for k2 in kpoints {
            let firsts: &[[f64; 3]] = if transfer {
                kpoints
            } else {
                std::slice::from_ref(&origin)
            };
            for k1 in firsts {
                let angle: f64 = (0..3)
                    .map(|axis| (k2[axis] - k1[axis]) * difference[axis])
                    .sum();
                if (Complex64::cis(angle) - 1.0).norm() > CLASS_TOLERANCE {
                    return Err(Error::InvalidArgument(message.to_string()));
                }
            }
        }
    }
    Ok(relabelled)
}

/// The distinct shell types of the auxiliary table: shells with the same
/// angular momentum, exponents and coefficients share their bounds.
struct AuxTypes {
    of_shell: Vec<usize>,
    /// The primitives (with a unit partner) of one shell of each type.
    pairs: Vec<PrimitivePairs>,
    angular_momenta: Vec<usize>,
    smallest_exponents: Vec<f64>,
}

impl AuxTypes {
    fn new(aux: &ShellTable) -> Self {
        let mut keys: HashMap<(usize, usize, Vec<u64>, Vec<u64>), usize> = HashMap::new();
        let mut types = Self {
            of_shell: Vec::with_capacity(aux.n_shells()),
            pairs: Vec::new(),
            angular_momenta: Vec::new(),
            smallest_exponents: Vec::new(),
        };
        for ip in 0..aux.n_shells() {
            let shell = aux.shell(ip);
            let key = (
                shell.angular_momentum,
                shell.n_contractions,
                shell.exponents.iter().map(|x| x.to_bits()).collect(),
                shell.coefficients.iter().map(|x| x.to_bits()).collect(),
            );
            let next = types.pairs.len();
            let index = *keys.entry(key).or_insert(next);
            if index == next {
                types.pairs.push(single_shell_pairs(&shell, ZERO));
                types.angular_momenta.push(shell.angular_momentum);
                types.smallest_exponents.push(smallest_exponent(&shell));
            }
            types.of_shell.push(index);
        }
        types
    }

    fn len(&self) -> usize {
        self.pairs.len()
    }
}

fn smallest_exponent(shell: &ShellRef<'_>) -> f64 {
    shell
        .exponents
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
}

/// The squared distance beyond which `bound exp(−decay R²)` is below the
/// threshold: everything passes without a threshold or without decay (the
/// full kernel), nothing passes when the bound is already below it.
fn reach_squared(bound: f64, decay: f64, threshold: f64) -> f64 {
    if bound < threshold {
        f64::NEG_INFINITY
    } else if threshold <= 0.0 || decay <= 0.0 {
        f64::INFINITY
    } else {
        (bound / threshold).ln() / decay
    }
}

/// The decay rate `(1 − ε) s_lo²` of the slowest quartet with the pair
/// exponent `p` and the auxiliary exponent `q` (ADR-0018).
fn decay_rate(p: f64, q: f64, kernel: CoulombKernel) -> f64 {
    let alpha = p * q / (p + q);
    let (low, _) = kernel.exponent_range(alpha);
    (1.0 - ENVELOPE_EPSILON) * low * low
}

/// A shell pair at one step with everything the image loop needs.
struct StepPair {
    shells: [usize; 2],
    ls: [usize; 2],
    offsets: [usize; 2],
    /// The primitive pairs for the bra shell at its centre and the ket
    /// shell displaced by the step.
    primitives: PrimitivePairs,
    /// The two centres, the segment the product centres lie on.
    ends: [[f64; 3]; 2],
    /// Squared reach of the shell triple per auxiliary type: the distance
    /// from the auxiliary centre to the segment beyond which the triple is
    /// below the threshold.
    reach: Vec<f64>,
    /// Squared reach per primitive pair and auxiliary type, `[pair][type]`,
    /// measured from the product centre to the auxiliary centre.
    quartet_reach: Vec<f64>,
    /// The largest triple reach.
    max_reach: f64,
}

/// The pairs of every step whose Schwarz bound, with the largest auxiliary
/// bound, is above the threshold, with their reach tables; in parallel
/// over the steps.
fn step_tables(
    tables: [&ShellTable; 2],
    types: &AuxTypes,
    aux_max: f64,
    steps: &[[f64; 3]],
    context: &Context<'_>,
    cache: &Tables,
) -> Vec<Vec<StepPair>> {
    let [bra, ket] = tables;
    let threshold = context.threshold;
    let n_types = types.len();
    steps
        .par_iter()
        .map(|step| {
            shell_pairs([bra, ket], [ZERO, *step], false, context.kernel, cache)
                .into_iter()
                .filter(|pair| pair.bound * aux_max >= threshold)
                .map(|pair| {
                    let [la, lb] = pair.ls;
                    let [ia, ib] = pair.shells;
                    let a = bra.shell(ia);
                    let b = ket.shell(ib);
                    let b_center: [f64; 3] =
                        std::array::from_fn(|axis| b.center[axis] + step[axis]);
                    let p_min = smallest_exponent(&a) + smallest_exponent(&b);
                    let n_pairs = pair.primitives.pairs.len();
                    let mut reach = Vec::with_capacity(n_types);
                    let mut quartet_reach = vec![f64::NEG_INFINITY; n_pairs * n_types];
                    for (index, aux_pairs) in types.pairs.iter().enumerate() {
                        let lp = types.angular_momenta[index];
                        let components: [&[[usize; 3]]; 4] = [
                            &cache.components[la],
                            &cache.components[lb],
                            &cache.components[lp],
                            &cache.components[0],
                        ];
                        let scale = cache.row_norms[la] * cache.row_norms[lb] * cache.row_norms[lp];
                        let mut total = 0.0;
                        for (q, pair_bra) in pair.primitives.pairs.iter().enumerate() {
                            let mut farthest = f64::NEG_INFINITY;
                            for (k, pair_ket) in aux_pairs.pairs.iter().enumerate() {
                                let bound = scale
                                    * pair.primitives.coefficient_max(q)
                                    * aux_pairs.coefficient_max(k)
                                    * quartet_bound_at_zero(
                                        pair_bra, pair_ket, components, context,
                                    );
                                total += bound;
                                let decay = decay_rate(pair_bra.p, pair_ket.p, context.kernel);
                                farthest = farthest.max(reach_squared(bound, decay, threshold));
                            }
                            quartet_reach[q * n_types + index] = farthest;
                        }
                        let decay =
                            decay_rate(p_min, types.smallest_exponents[index], context.kernel);
                        reach.push(reach_squared(total, decay, threshold));
                    }
                    let max_reach = reach.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    StepPair {
                        shells: pair.shells,
                        ls: pair.ls,
                        offsets: pair.offsets,
                        primitives: pair.primitives,
                        ends: [a.center, b_center],
                        reach,
                        quartet_reach,
                        max_reach,
                    }
                })
                .collect()
        })
        .collect()
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

/// The tables shared by every image of one call.
struct Setup<'a> {
    aux: &'a ShellTable,
    bra: &'a ShellTable,
    ket: &'a ShellTable,
    tables: &'a Tables,
    kernel: CoulombKernel,
    /// The primitives of every auxiliary shell at its centre.
    aux_pairs: &'a [PrimitivePairs],
    types: &'a AuxTypes,
    /// The distinct auxiliary centres and the centre of every shell.
    centres: &'a [[f64; 3]],
    centre_of_shell: &'a [usize],
    steps: &'a [Vec<StepPair>],
    lookup: &'a HashMap<[i64; 3], usize>,
    /// Whether the bra and ket coincide and the steps are closed under
    /// negation, so that every pair of images is evaluated once and the
    /// sum is completed by its conjugate transpose.
    symmetric: bool,
}

/// The accumulators: one real slab per class of the image, class of the
/// step and auxiliary shell, laid out `[P][μ][ν]` over the functions of the
/// shell; with [`Transfers::Zero`] the image classes collapse to one.
struct Slabs {
    data: Vec<Mutex<Vec<f64>>>,
    n_image_classes: usize,
    n_step_classes: usize,
    n_shells: usize,
}

impl Slabs {
    fn new(
        n_image_classes: usize,
        n_step_classes: usize,
        aux: &ShellTable,
        n_bra: usize,
        n_ket: usize,
    ) -> Self {
        let slabs = (0..n_image_classes * n_step_classes)
            .flat_map(|_| {
                (0..aux.n_shells())
                    .map(|ip| Mutex::new(vec![0.0; aux.shell(ip).n_functions() * n_bra * n_ket]))
            })
            .collect();
        Self {
            data: slabs,
            n_image_classes,
            n_step_classes,
            n_shells: aux.n_shells(),
        }
    }

    fn index(&self, image_class: usize, step_class: usize, ip: usize) -> usize {
        (image_class * self.n_step_classes + step_class) * self.n_shells + ip
    }

    /// Add every value of a spherical block to its row of the slab
    /// `(image_class, step_class, ip)`.
    fn scatter(
        &self,
        image_class: usize,
        step_class: usize,
        ip: usize,
        rows: impl Iterator<Item = (usize, f64)>,
    ) {
        let mut slab = self.data[self.index(image_class, step_class, ip)]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (row, value) in rows {
            slab[row] += value;
        }
    }

    /// The slab `(image_class, step_class, ip)` after the parallel loop.
    fn take(&mut self, image_class: usize, step_class: usize, ip: usize) -> Vec<f64> {
        let index = self.index(image_class, step_class, ip);
        std::mem::take(
            self.data[index]
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }
}

/// Per-thread scratch of the image loop.
#[derive(Default)]
struct Workspace {
    hermite: HermiteWorkspace,
    block: Vec<f64>,
    scratch: Vec<f64>,
    spherical: Vec<f64>,
    distances: Vec<f64>,
}

/// Three-centre Coulomb integrals summed over the images of the bra and
/// the ket with Bloch phases,
/// `Σ_{(i, j)} exp(−i k1·T_i) exp(i k2·T_j) (P | μ(r − T_i) ν(r − T_j))`,
/// as a row-major complex `[n_k, n_k, n_aux, n_bra, n_ket]` array over the
/// pairs `(i, j)` of images of `sum` whose difference is one of its steps,
/// the auxiliary function in the home cell: the Bloch-summed integral
/// `(P^q | μ k1 ν k2)` of the density fitting with momentum transfer
/// `q = k2 − k1` (ADR-0019). With [`Transfers::Zero`] only the blocks of
/// zero transfer are formed, `[n_k, n_aux, n_bra, n_ket]` with
/// `(P^0 | μ k ν k)` at every k-point. The phases `exp(i q·T_i)` and
/// `exp(i k2·T_s)` are taken from the first image and step of their classes
/// (see [`BlochSum`]) and applied once after the sum (ADR-0028).
///
/// A shell pair is skipped at a step when the product of its Schwarz bound
/// and the largest auxiliary bound is below `threshold`; a shell triple
/// when the auxiliary centre lies beyond the reach of the triple, the
/// distance from the segment between the pair's centres at which the bound
/// of ADR-0018, summed over the primitive quartets at zero distance, falls
/// below the threshold; and a primitive quartet when the auxiliary centre
/// lies beyond its own reach from the product centre. Every neglected term
/// is smaller than `threshold`. When `bra` and `ket` are the same table and
/// the steps are closed under negation, every pair of images is evaluated
/// once and the sum is completed by its conjugate transpose over k-points
/// and functions, which supplies the integrals with `μ` and `ν` exchanged.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if the kernel's `omega` is not
/// positive, `threshold` is negative or not finite, a translation or
/// k-point is not finite, the coordinate and class lists do not match the
/// image and step lists or repeat an image, or the phase of a momentum
/// transfer or of a k-point varies within a class.
pub fn coulomb_three_center_bloch(
    aux: &ShellTable,
    bra: &ShellTable,
    ket: &ShellTable,
    sum: BlochSum<'_>,
    kernel: CoulombKernel,
    threshold: f64,
    transfers: Transfers,
) -> Result<Vec<Complex64>, Error> {
    kernel.validate()?;
    check_threshold(threshold)?;
    let (image_classes, step_classes) = sum.check()?;
    let sum = BlochSum {
        image_classes: &image_classes,
        step_classes: &step_classes,
        ..sum
    };
    let n_image_classes = match transfers {
        Transfers::All => image_classes.iter().max().map_or(0, |c| c + 1),
        Transfers::Zero => 1,
    };
    let n_step_classes = step_classes.iter().max().map_or(0, |c| c + 1);
    let lookup = sum.lookup()?;
    let n_k = sum.kpoints.len();
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
    let aux_pairs: Vec<PrimitivePairs> = (0..aux.n_shells())
        .map(|ip| {
            let p_shell = aux.shell(ip);
            single_shell_pairs(&p_shell, p_shell.center)
        })
        .collect();
    let aux_max = largest_schwarz_bound(aux, &aux_pairs, kernel, &tables);
    let types = AuxTypes::new(aux);
    let (centres, centre_of_shell) = aux_centres(aux);
    let steps = step_tables([bra, ket], &types, aux_max, sum.steps, &context, &tables);
    let setup = Setup {
        aux,
        bra,
        ket,
        tables: &tables,
        kernel,
        aux_pairs: &aux_pairs,
        types: &types,
        centres: &centres,
        centre_of_shell: &centre_of_shell,
        steps: &steps,
        lookup: &lookup,
        symmetric: bra == ket && sum.steps_closed_under_negation(),
    };
    let slabs = Slabs::new(n_image_classes, n_step_classes, aux, n_bra, n_ket);
    // One task per image: the near images carry most of the work.
    (0..sum.images.len())
        .into_par_iter()
        .with_max_len(1)
        .for_each_init(Workspace::default, |workspace, i| {
            image_terms(&setup, &sum, &slabs, workspace, i, transfers);
        });
    let phases = Phases {
        // The transfer phases serve the full set of blocks only.
        class: match transfers {
            Transfers::All => sum.class_phases(n_image_classes),
            Transfers::Zero => Vec::new(),
        },
        step: sum.step_phases(n_step_classes),
        n_k,
    };
    let mut out = assemble(slabs, &phases, aux, [n_bra, n_ket], transfers);
    if setup.symmetric {
        add_conjugate_transpose(&mut out, [aux.n_functions(), n_bra, n_k], transfers);
    }
    Ok(out)
}

/// The transforms over the mesh applied after the sum.
struct Phases {
    /// `exp(i (k2 − k1)·T_c)` as `[k1][k2][image class]`.
    class: Vec<Complex64>,
    /// `exp(i k·T_c)` as `[step class][k]`.
    step: Vec<Complex64>,
    n_k: usize,
}

/// `out[k1, k2, P, μ, ν] += conj(out[k2, k1, P, ν, μ])`: the terms with the
/// images and functions of a pair exchanged, from the half of the pairs
/// evaluated; with [`Transfers::Zero`] `out[k, P, μ, ν] += conj(out[k, P, ν, μ])`.
/// In place, one pair of blocks at a time.
fn add_conjugate_transpose(out: &mut [Complex64], sizes: [usize; 3], transfers: Transfers) {
    let [n_aux, n, n_k] = sizes;
    let block_size = n_aux * n * n;
    let mut copy = vec![Complex64::default(); block_size];
    // `target[p, u, v] += conj(source[p, v, u])` in parallel over `p`.
    let add_transposed = |target: &mut [Complex64], source: &[Complex64]| {
        target
            .par_chunks_mut(n * n)
            .zip(source.par_chunks(n * n))
            .for_each(|(block, other)| {
                for u in 0..n {
                    for v in 0..n {
                        block[u * n + v] += other[v * n + u].conj();
                    }
                }
            });
    };
    let diagonal = |block: &mut [Complex64], copy: &mut [Complex64]| {
        copy.copy_from_slice(block);
        add_transposed(block, copy);
    };
    match transfers {
        Transfers::Zero => {
            for block in out.chunks_mut(block_size) {
                diagonal(block, &mut copy);
            }
        }
        Transfers::All => {
            for k1 in 0..n_k {
                diagonal(
                    &mut out[(k1 * n_k + k1) * block_size..][..block_size],
                    &mut copy,
                );
                for k2 in k1 + 1..n_k {
                    let (left, right) = out.split_at_mut((k2 * n_k + k1) * block_size);
                    let first = &mut left[(k1 * n_k + k2) * block_size..][..block_size];
                    let second = &mut right[..block_size];
                    copy.copy_from_slice(first);
                    add_transposed(first, second);
                    add_transposed(second, &copy);
                }
            }
        }
    }
}

/// The largest Schwarz bound over the auxiliary shells.
fn largest_schwarz_bound(
    aux: &ShellTable,
    aux_pairs: &[PrimitivePairs],
    kernel: CoulombKernel,
    tables: &Tables,
) -> f64 {
    let mut workspace = HermiteWorkspace::default();
    aux_pairs
        .iter()
        .enumerate()
        .map(|(ip, pairs)| {
            schwarz_bound(
                pairs,
                [aux.shell(ip).angular_momentum, 0],
                kernel,
                tables,
                &mut workspace,
            )
        })
        .fold(0.0_f64, f64::max)
}

/// The distinct centres of the auxiliary shells and the centre of each.
fn aux_centres(aux: &ShellTable) -> (Vec<[f64; 3]>, Vec<usize>) {
    let mut centres: Vec<[f64; 3]> = Vec::new();
    let of_shell = (0..aux.n_shells())
        .map(|ip| {
            let centre = aux.shell(ip).center;
            let same = |c: &[f64; 3]| {
                c.iter()
                    .zip(&centre)
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            };
            centres.iter().position(same).unwrap_or_else(|| {
                centres.push(centre);
                centres.len() - 1
            })
        })
        .collect();
    (centres, of_shell)
}

/// Apply the phases to the slabs, auxiliary shell by auxiliary shell:
/// first `A[c_i][k2] = Σ_{c_s} step[c_s][k2] slab[c_i][c_s]` over the step
/// classes, then `out[k1, k2] = Σ_{c_i} class[k1, k2, c_i] A[c_i][k2]` over
/// the image classes, as the row-major `[n_k, n_k, n_aux, n_bra, n_ket]`
/// array; with [`Transfers::Zero`] only the first transform, into
/// `out[k] = Σ_{c_s} step[c_s][k] slab[c_s]`, `[n_k, n_aux, n_bra, n_ket]`.
fn assemble(
    mut slabs: Slabs,
    phases: &Phases,
    aux: &ShellTable,
    sizes: [usize; 2],
    transfers: Transfers,
) -> Vec<Complex64> {
    let [n_bra, n_ket] = sizes;
    let n_k = phases.n_k;
    let block_size = aux.n_functions() * n_bra * n_ket;
    let n_blocks = match transfers {
        Transfers::All => n_k * n_k,
        Transfers::Zero => n_k,
    };
    let (n_image_classes, n_step_classes) = (slabs.n_image_classes, slabs.n_step_classes);
    let aux_offsets = aux.function_offsets();
    let mut out = vec![Complex64::default(); n_blocks * block_size];
    for (ip, &offset) in aux_offsets.iter().enumerate() {
        let width = aux.shell(ip).n_functions() * n_bra * n_ket;
        let start = offset * n_bra * n_ket;
        // The step transform: `summed[c_i][k][·]`, in parallel over `(c_i, k)`.
        let real: Vec<Vec<Vec<f64>>> = (0..n_image_classes)
            .map(|c_i| {
                (0..n_step_classes)
                    .map(|c_s| slabs.take(c_i, c_s, ip))
                    .collect()
            })
            .collect();
        let mut summed = vec![Complex64::default(); n_image_classes * n_k * width];
        summed
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(index, target)| {
                let (c_i, k) = (index / n_k, index % n_k);
                for (c_s, slab) in real[c_i].iter().enumerate() {
                    let phase = phases.step[c_s * n_k + k];
                    for (entry, &value) in target.iter_mut().zip(slab) {
                        *entry += phase * value;
                    }
                }
            });
        // The transfer transform into the shell's rows of every block.
        out.par_chunks_mut(block_size)
            .enumerate()
            .for_each(|(block, target)| {
                let rows = &mut target[start..start + width];
                match transfers {
                    Transfers::Zero => {
                        rows.copy_from_slice(&summed[block * width..][..width]);
                    }
                    Transfers::All => {
                        let k2 = block % n_k;
                        for c_i in 0..n_image_classes {
                            let phase = phases.class[block * n_image_classes + c_i];
                            let source = &summed[(c_i * n_k + k2) * width..][..width];
                            for (entry, &value) in rows.iter_mut().zip(source) {
                                *entry += phase * value;
                            }
                        }
                    }
                }
            });
    }
    out
}

/// Evaluate every pair with bra image `i` and scatter it into the slabs.
fn image_terms(
    setup: &Setup<'_>,
    sum: &BlochSum<'_>,
    slabs: &Slabs,
    workspace: &mut Workspace,
    i: usize,
    transfers: Transfers,
) {
    let (aux, ket, tables) = (setup.aux, setup.ket, setup.tables);
    let t_i = sum.images[i];
    let origin = sum.image_coordinates[i];
    let class_i = match transfers {
        Transfers::All => sum.image_classes[i],
        Transfers::Zero => 0,
    };
    let n_types = setup.types.len();
    for (s, coordinates) in sum.step_coordinates.iter().enumerate() {
        let target: [i64; 3] = std::array::from_fn(|axis| origin[axis] + coordinates[axis]);
        let Some(&j) = setup.lookup.get(&target) else {
            continue;
        };
        if setup.symmetric && j < i {
            continue;
        }
        let same = setup.symmetric && j == i;
        let class_s = sum.step_classes[s];
        for pair in &setup.steps[s] {
            let [ia, ib] = pair.shells;
            if (same && ib < ia) || pair.max_reach == f64::NEG_INFINITY {
                continue;
            }
            workspace.distances.clear();
            workspace
                .distances
                .extend(setup.centres.iter().map(|centre| {
                    let shifted: [f64; 3] = std::array::from_fn(|axis| centre[axis] - t_i[axis]);
                    segment_distance_squared(shifted, pair.ends[0], pair.ends[1])
                }));
            let nearest = workspace
                .distances
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            if nearest > pair.max_reach {
                continue;
            }
            let [la, lb] = pair.ls;
            let [ka, kb] = pair.primitives.contractions;
            for ip in 0..aux.n_shells() {
                let kind = setup.types.of_shell[ip];
                if workspace.distances[setup.centre_of_shell[ip]] > pair.reach[kind] {
                    continue;
                }
                let p_shell = aux.shell(ip);
                let (lp, kp) = (p_shell.angular_momentum, p_shell.n_contractions);
                let cartesian = ka * n_cartesian(la) * kb * n_cartesian(lb) * kp * n_cartesian(lp);
                workspace.block.clear();
                workspace.block.resize(cartesian, 0.0);
                let reach = &pair.quartet_reach[..];
                let keep = |q: usize, d2: f64| d2 <= reach[q * n_types + kind];
                if !triple_block(
                    &pair.primitives,
                    [la, lb],
                    t_i,
                    &setup.aux_pairs[ip],
                    lp,
                    keep,
                    setup.kernel,
                    tables,
                    &mut workspace.hermite,
                    &mut workspace.block,
                ) {
                    continue;
                }
                tables.to_spherical_into(
                    &workspace.block,
                    [Some(la), Some(lb), Some(lp), None],
                    [ka, kb, kp, 1],
                    &mut workspace.scratch,
                    &mut workspace.spherical,
                );
                // A block of one shell with itself on one image is its own
                // transpose and enters the half sum with half its weight.
                let weight = if same && ia == ib { 0.5 } else { 1.0 };
                scatter_triple(
                    setup,
                    slabs,
                    &workspace.spherical,
                    (
                        pair,
                        ip,
                        [ket.shell(ib).n_functions(), p_shell.n_functions()],
                    ),
                    [class_i, class_s],
                    weight,
                );
            }
        }
    }
}

/// Scatter the spherical block `[μ][ν][P]` of one triple, times `weight`,
/// into the slab of the classes of the bra image and of the step.
fn scatter_triple(
    setup: &Setup<'_>,
    slabs: &Slabs,
    spherical: &[f64],
    triple: (&StepPair, usize, [usize; 2]),
    classes: [usize; 2],
    weight: f64,
) {
    let (pair, ip, [nsb, nsp]) = triple;
    let (n_bra, n_ket) = (setup.bra.n_functions(), setup.ket.n_functions());
    let [bra_offset, ket_offset] = pair.offsets;
    let rows = spherical.iter().enumerate().map(|(index, &value)| {
        let (ma, mb, mp) = (index / (nsp * nsb), (index / nsp) % nsb, index % nsp);
        (
            (mp * n_bra + bra_offset + ma) * n_ket + ket_offset + mb,
            weight * value,
        )
    });
    slabs.scatter(classes[0], classes[1], ip, rows);
}

/// Accumulate the Cartesian integrals of one shell triple into `out`, laid
/// out `[k_a, cart_a, k_b, cart_b, k_p, cart_p]`, over the primitive
/// quartets that `keep` admits by their index in the pair and the squared
/// distance from the product centre, displaced by `shift`, to the
/// auxiliary centre. Returns whether anything was added.
#[allow(
    clippy::too_many_arguments,
    reason = "inner kernel; every argument is a distinct table"
)]
fn triple_block(
    pairs: &PrimitivePairs,
    ls: [usize; 2],
    shift: [f64; 3],
    aux_pairs: &PrimitivePairs,
    lp: usize,
    keep: impl Fn(usize, f64) -> bool,
    kernel: CoulombKernel,
    tables: &Tables,
    workspace: &mut HermiteWorkspace,
    out: &mut [f64],
) -> bool {
    let components: [&[[usize; 3]]; 4] = [
        &tables.components[ls[0]],
        &tables.components[ls[1]],
        &tables.components[lp],
        &tables.components[0],
    ];
    let dims: [usize; 4] = std::array::from_fn(|i| components[i].len());
    let orders = [ls[0] + ls[1], lp];
    let n_t = orders[0] + 1;
    let mut intermediate = std::mem::take(&mut workspace.intermediate);
    intermediate.clear();
    intermediate.resize(dims[2] * dims[3] * n_t * n_t * n_t, 0.0);
    let mut values = std::mem::take(&mut workspace.values);
    values.clear();
    values.resize(dims[0] * dims[1] * dims[2] * dims[3], 0.0);
    let mut added = false;
    for (q, pair_bra) in pairs.pairs.iter().enumerate() {
        for (k, pair_ket) in aux_pairs.pairs.iter().enumerate() {
            let pq: [f64; 3] = std::array::from_fn(|axis| {
                pair_bra.center[axis] + shift[axis] - pair_ket.center[axis]
            });
            let distance_squared: f64 = pq.iter().map(|x| x * x).sum();
            if !keep(q, distance_squared) {
                continue;
            }
            added = true;
            let prefactor = primitive_quartet(
                (pair_bra, pair_ket),
                pq,
                components,
                orders,
                kernel,
                workspace,
                &mut intermediate,
                &mut values,
            );
            accumulate_quartet((pairs, q), (aux_pairs, k), prefactor, &values, dims, out);
        }
    }
    workspace.intermediate = intermediate;
    workspace.values = values;
    added
}

#[cfg(test)]
mod tests {
    use super::super::test_tables::{general_table, sp_table};
    use super::super::two_electron::coulomb_three_center;
    use super::*;

    /// The Bloch-summed three-centre integrals from the explicit image
    /// pairs: `Σ_{ij} exp(−i k1·T_i) exp(i k2·T_j) (P | μ(T_i) ν(T_j))`
    /// with the aux at home expressed through the plain kernel as
    /// `(P(−T_i) | μ ν(T_j − T_i))`, over the pairs whose difference is a step.
    fn explicit_bloch(
        aux: &ShellTable,
        bra: &ShellTable,
        ket: &ShellTable,
        case: &BlochCase,
        kernel: CoulombKernel,
    ) -> Vec<Complex64> {
        let block = aux.n_functions() * bra.n_functions() * ket.n_functions();
        let n_k = case.kpoints.len();
        let mut out = vec![Complex64::default(); n_k * n_k * block];
        for (i, t_i) in case.images.iter().enumerate() {
            for (j, t_j) in case.images.iter().enumerate() {
                let step: [i64; 3] =
                    std::array::from_fn(|a| case.coordinates[j][a] - case.coordinates[i][a]);
                if !case.step_coordinates.contains(&step) {
                    continue;
                }
                let minus: [f64; 3] = std::array::from_fn(|k| -t_i[k]);
                let shifted: [f64; 3] = std::array::from_fn(|k| t_j[k] - t_i[k]);
                let values = coulomb_three_center(aux, bra, ket, &[[minus, shifted]], kernel, 0.0)
                    .expect("valid");
                for (k1, point_1) in case.kpoints.iter().enumerate() {
                    for (k2, point_2) in case.kpoints.iter().enumerate() {
                        let angle: f64 = (0..3)
                            .map(|a| point_2[a] * t_j[a] - point_1[a] * t_i[a])
                            .sum();
                        let phase = Complex64::cis(angle);
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
    /// difference of two images as a step, two k-points of a 2×1×1 mesh
    /// of the grid and the classes of that mesh.
    struct BlochCase {
        images: Vec<[f64; 3]>,
        coordinates: Vec<[i64; 3]>,
        steps: Vec<[f64; 3]>,
        step_coordinates: Vec<[i64; 3]>,
        classes: Vec<usize>,
        step_classes: Vec<usize>,
        kpoints: Vec<[f64; 3]>,
    }

    impl BlochCase {
        fn new() -> Self {
            let coordinates: Vec<[i64; 3]> = vec![
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 0, 0],
            ];
            let images: Vec<[f64; 3]> = vec![
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [-2.5, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [2.5, 3.0, 0.0],
                [5.0, 0.0, 0.0],
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
            // A 2×1×1 mesh of the 2.5 × 3.0 grid: k = (0, 0, 0) and (π/2.5, 0, 0),
            // whose transfer has the phase (−1)^{n_x} on the image with coordinates n.
            let class_of =
                |c: &[i64; 3]| usize::try_from(c[0].rem_euclid(2)).expect("non-negative");
            let classes = coordinates.iter().map(class_of).collect();
            let step_classes = step_coordinates.iter().map(class_of).collect();
            let kpoints = vec![[0.0, 0.0, 0.0], [std::f64::consts::PI / 2.5, 0.0, 0.0]];
            Self {
                images,
                coordinates,
                steps,
                step_coordinates,
                classes,
                step_classes,
                kpoints,
            }
        }

        fn sum(&self) -> BlochSum<'_> {
            BlochSum {
                images: &self.images,
                image_coordinates: &self.coordinates,
                steps: &self.steps,
                step_coordinates: &self.step_coordinates,
                image_classes: &self.classes,
                step_classes: &self.step_classes,
                kpoints: &self.kpoints,
            }
        }
    }

    fn bloch(
        aux: &ShellTable,
        bra: &ShellTable,
        ket: &ShellTable,
        sum: BlochSum<'_>,
        kernel: CoulombKernel,
        threshold: f64,
    ) -> Result<Vec<Complex64>, Error> {
        coulomb_three_center_bloch(aux, bra, ket, sum, kernel, threshold, Transfers::All)
    }

    fn assert_close(summed: &[Complex64], expected: &[Complex64]) {
        assert_eq!(summed.len(), expected.len());
        for (s, e) in summed.iter().zip(expected) {
            assert!((s - e).norm() < 1e-12, "{s} vs {e}");
        }
    }

    #[test]
    fn bloch_three_center_matches_the_explicit_image_sum() {
        let table = general_table();
        let aux = sp_table();
        let case = BlochCase::new();
        let n_k = case.kpoints.len();
        for kernel in [CoulombKernel::Full, CoulombKernel::ShortRange(0.6)] {
            let summed = bloch(&aux, &table, &table, case.sum(), kernel, 0.0).expect("valid");
            assert_close(
                &summed,
                &explicit_bloch(&aux, &table, &table, &case, kernel),
            );
            // The zero-transfer blocks alone are the diagonal of the full result.
            let zero = coulomb_three_center_bloch(
                &aux,
                &table,
                &table,
                case.sum(),
                kernel,
                0.0,
                Transfers::Zero,
            )
            .expect("valid");
            let block = aux.n_functions() * table.n_functions() * table.n_functions();
            assert_eq!(zero.len(), n_k * block);
            for k in 0..n_k {
                assert_close(
                    &zero[k * block..][..block],
                    &summed[(k * n_k + k) * block..][..block],
                );
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
    fn the_general_path_matches_the_explicit_sum_too() {
        let table = general_table();
        let aux = sp_table();
        let mut case = BlochCase::new();
        let kernel = CoulombKernel::ShortRange(0.6);
        // Every image in its own class.
        let expected = explicit_bloch(&aux, &table, &table, &case, kernel);
        case.classes = (0..case.images.len()).collect();
        let summed = bloch(&aux, &table, &table, case.sum(), kernel, 0.0).expect("valid");
        assert_close(&summed, &expected);
        // A ket table that is not the bra table: no pair is mirrored.
        let ket = sp_table();
        let summed = bloch(&aux, &table, &ket, case.sum(), kernel, 0.0).expect("valid");
        assert_close(&summed, &explicit_bloch(&aux, &table, &ket, &case, kernel));
        // The zero-transfer blocks of the general path are its diagonal too.
        let zero = coulomb_three_center_bloch(
            &aux,
            &table,
            &ket,
            case.sum(),
            kernel,
            0.0,
            Transfers::Zero,
        )
        .expect("valid");
        let n_k = case.kpoints.len();
        let block = aux.n_functions() * table.n_functions() * ket.n_functions();
        for k in 0..n_k {
            assert_close(
                &zero[k * block..][..block],
                &summed[(k * n_k + k) * block..][..block],
            );
        }
        // Steps that are not closed under negation also take the general path.
        let open: Vec<usize> = (0..case.step_coordinates.len())
            .filter(|&s| case.step_coordinates[s][0] >= 0)
            .collect();
        let half = BlochCase {
            steps: open.iter().map(|&s| case.steps[s]).collect(),
            step_coordinates: open.iter().map(|&s| case.step_coordinates[s]).collect(),
            step_classes: open.iter().map(|&s| case.step_classes[s]).collect(),
            ..BlochCase::new()
        };
        let summed = bloch(&aux, &table, &table, half.sum(), kernel, 0.0).expect("valid");
        assert_close(
            &summed,
            &explicit_bloch(&aux, &table, &table, &half, kernel),
        );
    }

    #[test]
    fn bloch_three_center_screens_and_validates() {
        let table = general_table();
        let aux = sp_table();
        let case = BlochCase::new();
        let short = CoulombKernel::ShortRange(0.6);
        // A screened sum stays within the threshold of the full one.
        let threshold = 1e-10;
        let exact = bloch(&aux, &table, &table, case.sum(), short, 0.0).expect("valid");
        let screened = bloch(&aux, &table, &table, case.sum(), short, threshold).expect("valid");
        let largest = exact
            .iter()
            .zip(&screened)
            .map(|(e, s)| (e - s).norm())
            .fold(0.0_f64, f64::max);
        assert!(largest < 100.0 * threshold, "{largest}");
        // Steps that reach no image contribute nothing.
        let far_steps = [[50.0, 0.0, 0.0]];
        let far_coordinates = [[20, 0, 0]];
        let far_classes = [0];
        let nothing = bloch(
            &aux,
            &table,
            &table,
            BlochSum {
                steps: &far_steps,
                step_coordinates: &far_coordinates,
                step_classes: &far_classes,
                ..case.sum()
            },
            short,
            0.0,
        )
        .expect("valid");
        assert!(nothing.iter().all(|v| v.norm() == 0.0));
        // Malformed lists are rejected.
        let repeated = [
            [0, 0, 0],
            [0, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 0, 0],
        ];
        let bad = BlochSum {
            image_coordinates: &repeated,
            ..case.sum()
        };
        assert!(bloch(&aux, &table, &table, bad, short, 0.0).is_err());
        let fewer_classes = BlochSum {
            image_classes: &case.classes[1..],
            ..case.sum()
        };
        assert!(bloch(&aux, &table, &table, fewer_classes, short, 0.0).is_err());
        let wrong_classes = vec![0; case.images.len()];
        let mixed = BlochSum {
            image_classes: &wrong_classes,
            ..case.sum()
        };
        assert!(bloch(&aux, &table, &table, mixed, short, 0.0).is_err());
        let fewer = BlochSum {
            step_coordinates: &case.step_coordinates[1..],
            ..case.sum()
        };
        assert!(bloch(&aux, &table, &table, fewer, short, 0.0).is_err());
        let wrong_steps = vec![0; case.steps.len()];
        let mixed_steps = BlochSum {
            step_classes: &wrong_steps,
            ..case.sum()
        };
        assert!(bloch(&aux, &table, &table, mixed_steps, short, 0.0).is_err());
        let mut stretched = case.steps.clone();
        stretched[1][0] += 0.5;
        let inconsistent = BlochSum {
            steps: &stretched,
            ..case.sum()
        };
        assert!(bloch(&aux, &table, &table, inconsistent, short, 0.0).is_err());
    }
}

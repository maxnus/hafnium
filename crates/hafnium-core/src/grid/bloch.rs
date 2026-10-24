//! Bloch sums of the basis functions at grid points.
//!
//! The Bloch function of ADR-0007, `φ_μ^k(r) = Σ_T exp(i k·T) φ_μ(r − T)`,
//! and its derivatives are accumulated image by image from the real
//! values of [`super::basis_values`]: every translation `T` shifts the
//! point, the shells within their cutoff radii of the shifted point are
//! evaluated once, and their values enter the sum at every k-point with
//! the phase `exp(i k·T)` the caller supplies.

use num_complex::Complex64;
use rayon::prelude::*;

use super::basis::{Workspace, n_components, radial_sums};
use crate::Error;
use crate::integrals::{ShellTable, cartesian_components, n_cartesian, spherical_transformation};

/// Points evaluated per parallel task.
const POINTS_PER_TASK: usize = 16;

/// Bloch sums of every function of `table` at `points`, with derivatives
/// up to `order` (at most two), as a row-major
/// `[kpt][point][component][ao]` array of complex values.
///
/// `translations` are the lattice vectors `T` of the images summed over
/// and `phases` the row-major `[kpt][image]` table of `exp(i k·T)`; the
/// caller chooses both, so that one image list serves every batch of
/// points. `radii` holds one cutoff per shell in Bohr as for
/// [`super::basis_values`]: an image of a shell farther than that from a
/// point is left out.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if `order` exceeds two, a point,
/// translation or phase is not finite, a radius is negative or NaN, or
/// there are no translations, and [`Error::LengthMismatch`] if there is
/// not one radius per shell or one phase per k-point and image.
pub fn bloch_basis_values(
    table: &ShellTable,
    points: &[[f64; 3]],
    radii: &[f64],
    translations: &[[f64; 3]],
    phases: &[Complex64],
    order: usize,
) -> Result<Vec<Complex64>, Error> {
    check_inputs(table, points, radii, translations, phases, order)?;
    let n_kpt = phases.len() / translations.len();
    let evaluator = Evaluator::new(table, radii, translations, phases, order);
    let stride = evaluator.stride();
    let mut out = vec![Complex64::new(0.0, 0.0); n_kpt * points.len() * stride];
    if out.is_empty() {
        return Ok(out);
    }
    // Split every k-point's block of the output into the per-task point
    // chunks, then regroup the chunks by task, so that each task owns one
    // disjoint slice of every k-point's block.
    let task_len = POINTS_PER_TASK * stride;
    let mut per_kpt: Vec<_> = out
        .chunks_mut(points.len() * stride)
        .map(|block| block.chunks_mut(task_len))
        .collect();
    let n_tasks = points.len().div_ceil(POINTS_PER_TASK);
    let tasks: Vec<Vec<&mut [Complex64]>> = (0..n_tasks)
        .map(|_| per_kpt.iter_mut().filter_map(Iterator::next).collect())
        .collect();
    tasks
        .into_par_iter()
        .zip(points.par_chunks(POINTS_PER_TASK))
        .for_each(|(mut slices, chunk)| {
            let mut scratch = evaluator.scratch();
            for (local, point) in chunk.iter().enumerate() {
                evaluator.accumulate(point, local, &mut slices, &mut scratch);
            }
        });
    Ok(out)
}

fn check_inputs(
    table: &ShellTable,
    points: &[[f64; 3]],
    radii: &[f64],
    translations: &[[f64; 3]],
    phases: &[Complex64],
    order: usize,
) -> Result<(), Error> {
    if order > 2 {
        return Err(Error::InvalidArgument(
            "derivatives are supported up to second order".to_string(),
        ));
    }
    if radii.len() != table.n_shells() {
        return Err(Error::LengthMismatch {
            left: radii.len(),
            right: table.n_shells(),
        });
    }
    if !radii.iter().all(|r| !r.is_nan() && *r >= 0.0) {
        return Err(Error::InvalidArgument(
            "cutoff radii are non-negative".to_string(),
        ));
    }
    if !points
        .iter()
        .chain(translations)
        .all(|point| point.iter().all(|x| x.is_finite()))
    {
        return Err(Error::InvalidArgument(
            "points and translations are finite".to_string(),
        ));
    }
    if translations.is_empty() {
        return Err(Error::InvalidArgument(
            "a Bloch sum runs over at least one translation".to_string(),
        ));
    }
    if !phases.iter().all(|z| z.re.is_finite() && z.im.is_finite()) {
        return Err(Error::InvalidArgument("phases are finite".to_string()));
    }
    if phases.len() % translations.len() != 0 || phases.is_empty() {
        return Err(Error::LengthMismatch {
            left: phases.len(),
            right: translations.len(),
        });
    }
    Ok(())
}

/// The tables shared by every task of one evaluation.
struct Evaluator<'a> {
    table: &'a ShellTable,
    radii: &'a [f64],
    translations: &'a [[f64; 3]],
    phases: &'a [Complex64],
    order: usize,
    n_comp: usize,
    n_ao: usize,
    max_l: usize,
    transforms: Vec<Vec<f64>>,
    components: Vec<Vec<[usize; 3]>>,
    offsets: Vec<usize>,
}

/// The per-task buffers: the workspace of the Cartesian evaluation and the
/// spherical values of one shell, `[component][function]`.
struct Scratch {
    workspace: Workspace,
    shell_values: Vec<f64>,
}

impl<'a> Evaluator<'a> {
    fn new(
        table: &'a ShellTable,
        radii: &'a [f64],
        translations: &'a [[f64; 3]],
        phases: &'a [Complex64],
        order: usize,
    ) -> Self {
        let max_l = table.max_angular_momentum();
        Self {
            table,
            radii,
            translations,
            phases,
            order,
            n_comp: n_components(order),
            n_ao: table.n_functions(),
            max_l,
            transforms: (0..=max_l).map(spherical_transformation).collect(),
            components: (0..=max_l).map(cartesian_components).collect(),
            offsets: table.function_offsets(),
        }
    }

    /// Complex values per point and k-point.
    fn stride(&self) -> usize {
        self.n_comp * self.n_ao
    }

    fn scratch(&self) -> Scratch {
        let width = (2 * self.max_l + 1) * self.table.max_contractions();
        Scratch {
            workspace: Workspace::new(self.max_l, self.n_comp),
            shell_values: vec![0.0; self.n_comp * width],
        }
    }

    /// Add the Bloch sums at `point`, the `local`-th of its task, into the
    /// task's slice of every k-point.
    fn accumulate(
        &self,
        point: &[f64; 3],
        local: usize,
        slices: &mut [&mut [Complex64]],
        scratch: &mut Scratch,
    ) {
        let stride = self.stride();
        let n_image = self.translations.len();
        for (image, translation) in self.translations.iter().enumerate() {
            let shifted: [f64; 3] = std::array::from_fn(|axis| point[axis] - translation[axis]);
            for (shell_index, &offset) in self.offsets.iter().enumerate() {
                let Some(width) = self.shell_values(shell_index, shifted, scratch) else {
                    continue;
                };
                for (kpt, target) in slices.iter_mut().enumerate() {
                    let phase = self.phases[kpt * n_image + image];
                    let values = &mut target[local * stride..(local + 1) * stride];
                    for component in 0..self.n_comp {
                        let source =
                            &scratch.shell_values[component * width..(component + 1) * width];
                        let start = component * self.n_ao + offset;
                        for (out, &value) in values[start..start + width].iter_mut().zip(source) {
                            *out += phase.scale(value);
                        }
                    }
                }
            }
        }
    }

    /// Evaluate one shell at the shifted point into `scratch.shell_values`
    /// and return its number of functions, or `None` beyond its cutoff.
    fn shell_values(
        &self,
        shell_index: usize,
        shifted: [f64; 3],
        scratch: &mut Scratch,
    ) -> Option<usize> {
        let shell = self.table.shell(shell_index);
        let l = shell.angular_momentum;
        let d: [f64; 3] = std::array::from_fn(|axis| shifted[axis] - shell.center[axis]);
        let s = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if s > self.radii[shell_index] * self.radii[shell_index] {
            return None;
        }
        let n_sph = 2 * l + 1;
        let width = n_sph * shell.n_contractions;
        let values = &mut scratch.shell_values[..self.n_comp * width];
        values.fill(0.0);
        scratch
            .workspace
            .monomials(&self.components[l], l, d, self.order);
        for contraction in 0..shell.n_contractions {
            let radial = radial_sums(&shell, contraction, s, self.order);
            scratch
                .workspace
                .cartesian(&self.components[l], d, &radial, self.order);
            scratch.workspace.scatter(
                &self.transforms[l],
                n_cartesian(l),
                n_sph,
                width,
                contraction * n_sph,
                values,
            );
        }
        Some(width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::basis_values;
    use crate::integrals::test_tables::general_table;

    fn phases_for(kpoints: &[[f64; 3]], translations: &[[f64; 3]]) -> Vec<Complex64> {
        kpoints
            .iter()
            .flat_map(|k| {
                translations.iter().map(move |t| {
                    let angle = k[0] * t[0] + k[1] * t[1] + k[2] * t[2];
                    Complex64::new(angle.cos(), angle.sin())
                })
            })
            .collect()
    }

    #[test]
    fn bloch_sums_match_the_phased_sum_of_shifted_values() {
        let table = general_table();
        let n_ao = table.n_functions();
        let radii = vec![f64::INFINITY; table.n_shells()];
        let points = [[0.3, 0.1, -0.2], [1.2, -0.7, 0.4], [-0.4, 0.9, 0.6]];
        let translations = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 2.5, 0.5],
        ];
        let kpoints = [[0.0, 0.0, 0.0], [0.4, -0.3, 0.2]];
        let phases = phases_for(&kpoints, &translations);
        for order in 0..=2 {
            let n_comp = n_components(order);
            let bloch = bloch_basis_values(&table, &points, &radii, &translations, &phases, order)
                .expect("valid");
            assert_eq!(bloch.len(), kpoints.len() * points.len() * n_comp * n_ao);
            for (kpt, k) in kpoints.iter().enumerate() {
                for (g, point) in points.iter().enumerate() {
                    let mut expected = vec![Complex64::new(0.0, 0.0); n_comp * n_ao];
                    for t in &translations {
                        let shifted = [point[0] - t[0], point[1] - t[1], point[2] - t[2]];
                        let values =
                            basis_values(&table, &[shifted], &radii, order).expect("valid");
                        let angle = k[0] * t[0] + k[1] * t[1] + k[2] * t[2];
                        let phase = Complex64::new(angle.cos(), angle.sin());
                        for (e, &v) in expected.iter_mut().zip(&values) {
                            *e += phase.scale(v);
                        }
                    }
                    let start = (kpt * points.len() + g) * n_comp * n_ao;
                    for (i, e) in expected.iter().enumerate() {
                        assert!(
                            (bloch[start + i] - e).norm() < 1e-14,
                            "order {order} k {kpt} g {g} i {i}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn no_points_give_no_values() {
        let table = general_table();
        let phases = [Complex64::new(1.0, 0.0)];
        let values =
            bloch_basis_values(&table, &[], &[1.0, 1.0], &[[0.0; 3]], &phases, 1).expect("valid");
        assert_eq!(values, Vec::new());
    }

    #[test]
    fn cutoff_radii_leave_out_far_images() {
        let table = general_table();
        let points = [[0.1, 0.0, 0.0]];
        let translations = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let phases = phases_for(&[[0.2, 0.0, 0.0]], &translations);
        let near = bloch_basis_values(&table, &points, &[1.0, 1.0], &translations, &phases, 0)
            .expect("valid");
        let zero = bloch_basis_values(
            &table,
            &points,
            &[1.0, 1.0],
            &translations[..1],
            &phases[..1],
            0,
        )
        .expect("valid");
        assert_eq!(near, zero);
    }

    #[test]
    fn malformed_input_is_rejected() {
        let table = general_table();
        let points = [[0.0; 3]];
        let translations = [[0.0; 3]];
        let phases = [Complex64::new(1.0, 0.0)];
        assert!(matches!(
            bloch_basis_values(&table, &points, &[1.0, 1.0], &translations, &phases, 3),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            bloch_basis_values(&table, &points, &[1.0], &translations, &phases, 0),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            bloch_basis_values(&table, &points, &[1.0, 1.0], &[], &phases, 0),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            bloch_basis_values(&table, &points, &[1.0, 1.0], &translations, &phases[..0], 0),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            bloch_basis_values(
                &table,
                &points,
                &[1.0, 1.0],
                &translations,
                &[Complex64::new(f64::NAN, 0.0)],
                0
            ),
            Err(Error::InvalidArgument(_))
        ));
    }
}

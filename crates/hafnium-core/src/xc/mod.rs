//! Exchange–correlation functionals through the `xcx` crate (ADR-0026).
//!
//! `xcx` implements the functionals of libxc in Rust with their libxc ids,
//! names and numerical conventions, and differentiates them automatically,
//! so every functional comes with its first derivatives. This module maps a
//! linear combination of named functionals onto `xcx`, lays the arrays out
//! spin-major as the Python side holds them, and runs the evaluation in
//! parallel over the points.
//!
//! Conventions (libxc): `rho` holds the spin densities `[alpha, beta]`,
//! `sigma` the contracted gradients `[∇ρ_α·∇ρ_α, ∇ρ_α·∇ρ_β, ∇ρ_β·∇ρ_β]`,
//! `tau` the kinetic energy densities `½ Σ_i |∇ψ_iσ|²`; the result holds
//! the exchange–correlation energy per particle `ε_xc` and the derivatives
//! of the energy density `ρ ε_xc` with respect to each input.

use rayon::prelude::*;
use xcx::{Family, Functional, FunctionalId, Spin, XcInput};

use crate::Error;

pub mod noncollinear;

pub use noncollinear::{
    MAX_LEVEL, NoncollinearInput, NoncollinearValues, SPIN_NEGLECT, Sampling, SphereRule,
    evaluate_noncollinear,
};

/// Points evaluated per parallel task.
const POINTS_PER_TASK: usize = 256;

/// Rung of a functional: what it reads at every point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rung {
    /// The spin densities only.
    Lda,
    /// Densities and their gradients.
    Gga,
    /// Densities, gradients and kinetic energy densities.
    MetaGga,
}

impl Rung {
    /// The rung's name in lowercase, as the Python side spells it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Lda => "lda",
            Self::Gga => "gga",
            Self::MetaGga => "meta-gga",
        }
    }
}

/// What a named functional needs and the exact exchange it asks for.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionalInfo {
    /// The libxc id of the functional.
    pub id: u32,
    /// Its canonical libxc name.
    pub name: String,
    /// Its rung.
    pub rung: Rung,
    /// Whether it needs the Laplacian of the density, which no supported
    /// functional does.
    pub needs_laplacian: bool,
    /// Fraction of exact exchange with the full Coulomb kernel.
    pub exact_exchange: f64,
    /// Range separation `(omega, alpha, beta)`: the exact exchange is
    /// `alpha + beta erf(omega r)` of the Coulomb interaction, so `alpha`
    /// repeats `exact_exchange` and `beta` multiplies the long-range
    /// exchange; `None` for a functional without range separation.
    pub range_separation: Option<[f64; 3]>,
    /// Parameters `(b, C)` of the VV10 nonlocal correlation the functional
    /// is defined with, which the host must add; `None` without.
    pub vv10: Option<[f64; 2]>,
    /// Total density below which the functional returns zero.
    pub density_threshold: f64,
}

/// Energy per particle and first derivatives at every point, spin-major.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct XcValues {
    /// `ε_xc[g]`, the energy per particle.
    pub energy_per_particle: Vec<f64>,
    /// `∂(ρ ε_xc)/∂ρ_σ` as `[2][g]`.
    pub vrho: Vec<f64>,
    /// `∂(ρ ε_xc)/∂σ_στ` as `[3][g]`, empty for an LDA.
    pub vsigma: Vec<f64>,
    /// `∂(ρ ε_xc)/∂τ_σ` as `[2][g]`, empty below the meta-GGA rung.
    pub vtau: Vec<f64>,
}

/// Second derivatives of the energy density at every point, spin-major, in
/// libxc's packed layouts.
///
/// A pair of variables of one kind is stored once, upper triangle first:
/// `αα, αβ, ββ` over the spins and `αα·αα, αα·αβ, αα·ββ, αβ·αβ, αβ·ββ,
/// ββ·ββ` over the gradient contractions; a pair of different kinds is
/// row-major over the first kind. Rows a rung does not read are empty.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct XcSecondDerivatives {
    /// `∂²(ρ ε_xc)/∂ρ_σ ∂ρ_τ` as `[3][g]`.
    pub v2rho2: Vec<f64>,
    /// `∂²(ρ ε_xc)/∂ρ_σ ∂σ_k` as `[6][g]`, `3σ + k`; empty for an LDA.
    pub v2rhosigma: Vec<f64>,
    /// `∂²(ρ ε_xc)/∂σ_k ∂σ_l` as `[6][g]`; empty for an LDA.
    pub v2sigma2: Vec<f64>,
    /// `∂²(ρ ε_xc)/∂ρ_σ ∂τ_τ` as `[4][g]`, `2σ + τ`; empty below the meta-GGA rung.
    pub v2rhotau: Vec<f64>,
    /// `∂²(ρ ε_xc)/∂σ_k ∂τ_σ` as `[6][g]`, `2k + σ`; empty below the meta-GGA rung.
    pub v2sigmatau: Vec<f64>,
    /// `∂²(ρ ε_xc)/∂τ_σ ∂τ_τ` as `[3][g]`; empty below the meta-GGA rung.
    pub v2tau2: Vec<f64>,
}

/// Look up a functional by its libxc name.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if the name is unknown or the
/// functional is not implemented.
pub fn functional_info(name: &str) -> Result<FunctionalInfo, Error> {
    let id = lookup(name)?;
    let functional = Functional::new(id, Spin::Polarized)
        .map_err(|error| Error::InvalidArgument(format!("functional {name}: {error}")))?;
    let info = functional.info();
    let hybrid = info.hybrid.as_ref();
    Ok(FunctionalInfo {
        id: id.as_u32(),
        name: id.name().to_string(),
        rung: rung_of(info.family, info.needs_sigma, info.needs_tau),
        needs_laplacian: info.needs_lapl,
        exact_exchange: functional.exx_fraction(),
        range_separation: hybrid
            .and_then(|h| h.cam.as_ref())
            .map(|cam| [cam.omega, cam.alpha, cam.beta]),
        vv10: hybrid.and_then(|h| h.vv10.as_ref()).map(|v| [v.b, v.c]),
        density_threshold: info.dens_threshold,
    })
}

fn lookup(name: &str) -> Result<FunctionalId, Error> {
    FunctionalId::from_name(&name.to_ascii_lowercase())
        .ok_or_else(|| Error::InvalidArgument(format!("unknown functional {name:?}")))
}

fn rung_of(family: Family, needs_sigma: bool, needs_tau: bool) -> Rung {
    if needs_tau || matches!(family, Family::Mgga | Family::HybMgga) {
        Rung::MetaGga
    } else if needs_sigma || matches!(family, Family::Gga | Family::HybGga) {
        Rung::Gga
    } else {
        Rung::Lda
    }
}

/// Evaluate `Σ_i w_i f_i` over the named functionals at every point.
///
/// `rho` is `[2][g]`, `sigma` `[3][g]` and `tau` `[2][g]`; `sigma` and
/// `tau` may be empty when no component needs them. The combination is
/// evaluated spin-polarized; a restricted density passes equal spin
/// densities.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if a name is unknown, a component
/// needs an input that is empty, no component is given, or an input is
/// not finite, and [`Error::LengthMismatch`] if the arrays disagree on
/// the number of points.
pub fn evaluate_functional(
    components: &[(String, f64)],
    rho: &[f64],
    sigma: &[f64],
    tau: &[f64],
) -> Result<XcValues, Error> {
    let (ids, rung, n) = prepare(components, rho, sigma, tau)?;
    let chunks = evaluate_chunks(&ids, n, |functional, range| {
        let count = range.len();
        let packed = Packed::new(rung, n, range, rho, sigma, tau);
        let result = functional
            .eval(count, &packed.input(rung))
            .map_err(|error| Error::InvalidArgument(format!("functional evaluation: {error}")))?;
        Ok(XcValues {
            energy_per_particle: result.exc,
            vrho: deinterleave(&result.vrho, 2, count)?,
            vsigma: deinterleave_if(&result.vsigma, 3, count, rung != Rung::Lda)?,
            vtau: deinterleave_if(&result.vtau, 2, count, rung == Rung::MetaGga)?,
        })
    })?;
    Ok(XcValues {
        energy_per_particle: chunks
            .iter()
            .flat_map(|chunk| chunk.energy_per_particle.iter().copied())
            .collect(),
        vrho: join(&chunks, n, |chunk| &chunk.vrho, 2),
        vsigma: join(&chunks, n, |chunk| &chunk.vsigma, 3),
        vtau: join(&chunks, n, |chunk| &chunk.vtau, 2),
    })
}

/// Evaluate the second derivatives of `Σ_i w_i f_i` at every point.
///
/// The inputs are those of [`evaluate_functional`]; the result holds the
/// derivatives the rung of the combination reads, see
/// [`XcSecondDerivatives`].
///
/// # Errors
///
/// As for [`evaluate_functional`].
pub fn evaluate_functional_second(
    components: &[(String, f64)],
    rho: &[f64],
    sigma: &[f64],
    tau: &[f64],
) -> Result<XcSecondDerivatives, Error> {
    let (ids, rung, n) = prepare(components, rho, sigma, tau)?;
    let gga = rung != Rung::Lda;
    let meta = rung == Rung::MetaGga;
    let chunks = evaluate_chunks(&ids, n, |functional, range| {
        let count = range.len();
        let packed = Packed::new(rung, n, range, rho, sigma, tau);
        let result = functional
            .eval_fxc(count, &packed.input(rung))
            .map_err(|error| Error::InvalidArgument(format!("functional evaluation: {error}")))?;
        Ok(XcSecondDerivatives {
            v2rho2: deinterleave(&result.v2rho2, 3, count)?,
            v2rhosigma: deinterleave_if(&result.v2rhosigma, 6, count, gga)?,
            v2sigma2: deinterleave_if(&result.v2sigma2, 6, count, gga)?,
            v2rhotau: deinterleave_if(&result.v2rhotau, 4, count, meta)?,
            v2sigmatau: deinterleave_if(&result.v2sigmatau, 6, count, meta)?,
            v2tau2: deinterleave_if(&result.v2tau2, 3, count, meta)?,
        })
    })?;
    Ok(XcSecondDerivatives {
        v2rho2: join(&chunks, n, |chunk| &chunk.v2rho2, 3),
        v2rhosigma: join(&chunks, n, |chunk| &chunk.v2rhosigma, 6),
        v2sigma2: join(&chunks, n, |chunk| &chunk.v2sigma2, 6),
        v2rhotau: join(&chunks, n, |chunk| &chunk.v2rhotau, 4),
        v2sigmatau: join(&chunks, n, |chunk| &chunk.v2sigmatau, 6),
        v2tau2: join(&chunks, n, |chunk| &chunk.v2tau2, 3),
    })
}

/// The resolved components of a combination, its rung and the number of points.
type Prepared = (Vec<(FunctionalId, f64)>, Rung, usize);

/// Resolve the combination, check the inputs and return the ids, the rung
/// and the number of points.
fn prepare(
    components: &[(String, f64)],
    rho: &[f64],
    sigma: &[f64],
    tau: &[f64],
) -> Result<Prepared, Error> {
    let (ids, rung) = combination(components)?;
    if rho.len() % 2 != 0 {
        return Err(Error::InvalidArgument(
            "the densities are laid out [2][g]".to_string(),
        ));
    }
    let n = rho.len() / 2;
    check_input(rho, 2 * n, "densities")?;
    if rung != Rung::Lda {
        check_input(sigma, 3 * n, "contracted gradients")?;
    }
    if rung == Rung::MetaGga {
        check_input(tau, 2 * n, "kinetic energy densities")?;
    }
    // Built once here to validate the combination, and once per worker
    // in `evaluate_chunks`: the evaluator is not `Sync`, so it cannot be
    // shared.
    build(&ids)?;
    Ok((ids, rung, n))
}

/// Run `chunk` over the points in parallel tasks, each with its own evaluator.
fn evaluate_chunks<T, F>(ids: &[(FunctionalId, f64)], n: usize, chunk: F) -> Result<Vec<T>, Error>
where
    T: Send,
    F: Fn(&Functional, std::ops::Range<usize>) -> Result<T, Error> + Sync,
{
    (0..n.div_ceil(POINTS_PER_TASK).max(1))
        .into_par_iter()
        .map_init(
            || build(ids).ok(),
            |functional, task| {
                let functional = functional.as_ref().ok_or_else(|| {
                    Error::InvalidArgument("the functional could not be built".to_string())
                })?;
                let start = task * POINTS_PER_TASK;
                let end = (start + POINTS_PER_TASK).min(n);
                chunk(functional, start..end)
            },
        )
        .collect()
}

/// The points of one chunk of the spin-major inputs, interleaved per point
/// as `xcx` reads them.
struct Packed {
    rho: Vec<f64>,
    sigma: Vec<f64>,
    tau: Vec<f64>,
}

impl Packed {
    fn new(
        rung: Rung,
        n: usize,
        range: std::ops::Range<usize>,
        rho: &[f64],
        sigma: &[f64],
        tau: &[f64],
    ) -> Self {
        let interleave = |values: &[f64], parts: usize| -> Vec<f64> {
            let mut packed = Vec::with_capacity(range.len() * parts);
            for g in range.clone() {
                for part in 0..parts {
                    packed.push(values[part * n + g]);
                }
            }
            packed
        };
        Self {
            rho: interleave(rho, 2),
            sigma: if rung == Rung::Lda {
                Vec::new()
            } else {
                interleave(sigma, 3)
            },
            tau: if rung == Rung::MetaGga {
                interleave(tau, 2)
            } else {
                Vec::new()
            },
        }
    }

    fn input(&self, rung: Rung) -> XcInput<'_> {
        let mut input = XcInput::lda(&self.rho);
        if rung != Rung::Lda {
            input = XcInput::gga(&self.rho, &self.sigma);
        }
        if rung == Rung::MetaGga {
            input = input.with_tau(&self.tau);
        }
        input
    }
}

/// Spin-major `[parts][count]` from the per-point layout `[count][parts]`.
fn deinterleave(values: &[f64], parts: usize, count: usize) -> Result<Vec<f64>, Error> {
    if values.len() != count * parts {
        return Err(Error::LengthMismatch {
            left: values.len(),
            right: count * parts,
        });
    }
    let mut unpacked = vec![0.0; count * parts];
    for g in 0..count {
        for part in 0..parts {
            unpacked[part * count + g] = values[g * parts + part];
        }
    }
    Ok(unpacked)
}

/// [`deinterleave`] when `wanted`, else the empty vector of a row the rung does not read.
fn deinterleave_if(
    values: &[f64],
    parts: usize,
    count: usize,
    wanted: bool,
) -> Result<Vec<f64>, Error> {
    if wanted {
        deinterleave(values, parts, count)
    } else {
        Ok(Vec::new())
    }
}

/// Resolve the named components to `xcx` ids and the rung of the combination.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if no component is given, a name is
/// unknown, a weight is not finite, or a component needs the Laplacian of
/// the density.
fn combination(components: &[(String, f64)]) -> Result<(Vec<(FunctionalId, f64)>, Rung), Error> {
    if components.is_empty() {
        return Err(Error::InvalidArgument(
            "a functional needs at least one component".to_string(),
        ));
    }
    let ids = components
        .iter()
        .map(|(name, weight)| {
            if !weight.is_finite() {
                return Err(Error::InvalidArgument(
                    "functional weights are finite".to_string(),
                ));
            }
            Ok((lookup(name)?, *weight))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let infos = ids
        .iter()
        .map(|(id, _)| functional_info(id.name()))
        .collect::<Result<Vec<_>, _>>()?;
    let rung = infos
        .iter()
        .map(|info| info.rung)
        .fold(Rung::Lda, |acc, rung| match (acc, rung) {
            (Rung::MetaGga, _) | (_, Rung::MetaGga) => Rung::MetaGga,
            (Rung::Gga, _) | (_, Rung::Gga) => Rung::Gga,
            _ => Rung::Lda,
        });
    if infos.iter().any(|info| info.needs_laplacian) {
        return Err(Error::InvalidArgument(
            "functionals of the Laplacian of the density are not supported".to_string(),
        ));
    }
    Ok((ids, rung))
}

fn check_input(values: &[f64], expected: usize, name: &str) -> Result<(), Error> {
    if values.len() != expected {
        return Err(Error::LengthMismatch {
            left: values.len(),
            right: expected,
        });
    }
    if !values.iter().all(|x| x.is_finite()) {
        return Err(Error::InvalidArgument(format!("{name} are finite")));
    }
    Ok(())
}

fn build(ids: &[(FunctionalId, f64)]) -> Result<Functional, Error> {
    let parts = ids
        .iter()
        .map(|(id, weight)| {
            Functional::new(*id, Spin::Polarized)
                .map(|functional| (*weight, functional))
                .map_err(|error| {
                    Error::InvalidArgument(format!("functional {}: {error}", id.name()))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Functional::mix(parts)
        .map_err(|error| Error::InvalidArgument(format!("functional mix: {error}")))
}

/// Join one spin-major row set of the chunks into `[parts][n]` over all points.
///
/// Empty when the chunks carry no such rows, as for a row the rung does
/// not read.
fn join<T>(chunks: &[T], n: usize, rows: impl Fn(&T) -> &[f64], parts: usize) -> Vec<f64> {
    if chunks.iter().all(|chunk| rows(chunk).is_empty()) {
        return Vec::new();
    }
    let mut target = vec![0.0; parts * n];
    let mut start = 0;
    for chunk in chunks {
        let source = rows(chunk);
        let count = source.len() / parts;
        place(&mut target, source, parts, n, start, count);
        start += count;
    }
    target
}

fn place(target: &mut [f64], source: &[f64], parts: usize, n: usize, start: usize, count: usize) {
    for part in 0..parts {
        target[part * n + start..part * n + start + count]
            .copy_from_slice(&source[part * count..(part + 1) * count]);
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    fn component(name: &str, weight: f64) -> (String, f64) {
        (name.to_string(), weight)
    }

    #[test]
    fn metadata_follows_libxc() {
        let pbe = functional_info("gga_x_pbe").expect("known");
        assert_eq!(
            (pbe.id, pbe.rung, pbe.exact_exchange),
            (101, Rung::Gga, 0.0)
        );
        assert_eq!(pbe.name, "gga_x_pbe");
        let b3lyp = functional_info("HYB_GGA_XC_B3LYP").expect("case-insensitive");
        assert_eq!((b3lyp.id, b3lyp.rung), (402, Rung::Gga));
        assert!((b3lyp.exact_exchange - 0.2).abs() < 1e-15);
        assert_eq!(b3lyp.range_separation, None);
        let tpss = functional_info("mgga_c_tpss").expect("known");
        assert_eq!(tpss.rung, Rung::MetaGga);
        assert!(!tpss.needs_laplacian);
        let rsh = functional_info("hyb_gga_xc_wb97x_v").expect("known");
        let [omega, alpha, beta] = rsh.range_separation.expect("range separated");
        assert!((omega - 0.3).abs() < 1e-15 && (alpha - 0.167).abs() < 1e-12);
        assert!((alpha + beta - 1.0).abs() < 1e-12);
        assert_eq!(rsh.vv10, Some([6.0, 0.01]));
        assert!(matches!(
            functional_info("gga_x_nonsense"),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn second_derivatives_match_finite_differences_of_the_first() {
        // Two points of (ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ, τ_α, τ_β), spin-major.
        let points: [[f64; 7]; 2] = [
            [0.3, 0.2, 0.05, 0.02, 0.04, 0.4, 0.3],
            [0.7, 0.1, 0.3, 0.05, 0.02, 1.0, 0.2],
        ];
        let layout = |values: &[[f64; 7]; 2]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let column = |k: usize| values.iter().map(|point| point[k]).collect::<Vec<_>>();
            (
                [column(0), column(1)].concat(),
                [column(2), column(3), column(4)].concat(),
                [column(5), column(6)].concat(),
            )
        };
        for (name, parts) in [("gga_x_pbe", 5), ("mgga_x_tpss", 7)] {
            let components = [component(name, 1.0)];
            let (rho, sigma, tau) = layout(&points);
            let second =
                evaluate_functional_second(&components, &rho, &sigma, &tau).expect("valid");
            let first = |values: &[[f64; 7]; 2]| -> Vec<f64> {
                let (rho, sigma, tau) = layout(values);
                let v = evaluate_functional(&components, &rho, &sigma, &tau).expect("valid");
                // One row per variable, [variable][g].
                [v.vrho, v.vsigma, v.vtau].concat()
            };
            let n = points.len();
            for variable in 0..parts {
                let mut up = points;
                let mut down = points;
                let mut steps = [0.0; 2];
                for g in 0..n {
                    let step = 1e-4 * points[g][variable];
                    up[g][variable] += step;
                    down[g][variable] -= step;
                    steps[g] = step;
                }
                let (plus, minus) = (first(&up), first(&down));
                for other in 0..parts {
                    for g in 0..n {
                        let numeric =
                            (plus[other * n + g] - minus[other * n + g]) / (2.0 * steps[g]);
                        let analytic = second_derivative(&second, variable, other, n, g);
                        let scale = numeric.abs().max(1.0);
                        assert!(
                            (numeric - analytic).abs() < 1e-6 * scale,
                            "{name}: d v[{other}] / d x[{variable}] at point {g}: {numeric} vs {analytic}"
                        );
                    }
                }
            }
        }
    }

    /// `∂²(ρε)/∂x_a ∂x_b` at point `g` from the packed rows, with the
    /// variables numbered `ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ, τ_α, τ_β`.
    fn second_derivative(
        second: &XcSecondDerivatives,
        first: usize,
        other: usize,
        n_points: usize,
        point: usize,
    ) -> f64 {
        let (low, high) = if first <= other {
            (first, other)
        } else {
            (other, first)
        };
        let upper = |i: usize, j: usize, size: usize| i * size - i * (i + 1) / 2 + j;
        let kind = |x: usize| match x {
            0 | 1 => (0, x),
            2..=4 => (1, x - 2),
            _ => (2, x - 5),
        };
        let ((kind_low, i), (kind_high, j)) = (kind(low), kind(high));
        let row = match (kind_low, kind_high) {
            (0, 0) => &second.v2rho2[upper(i, j, 2) * n_points..],
            (0, 1) => &second.v2rhosigma[(3 * i + j) * n_points..],
            (1, 1) => &second.v2sigma2[upper(i, j, 3) * n_points..],
            (0, 2) => &second.v2rhotau[(2 * i + j) * n_points..],
            (1, 2) => &second.v2sigmatau[(2 * i + j) * n_points..],
            _ => &second.v2tau2[upper(i, j, 2) * n_points..],
        };
        row[point]
    }

    #[test]
    fn slater_exchange_matches_the_closed_form() {
        // Polarized Slater exchange with equal spin densities: the energy per
        // particle −(3/4)(3ρ/π)^{1/3} and the potential −(3ρ/π)^{1/3}.
        let densities = [0.1, 0.5, 2.0];
        let rho: Vec<f64> = densities
            .iter()
            .map(|r| r / 2.0)
            .chain(densities.iter().map(|r| r / 2.0))
            .collect();
        let values =
            evaluate_functional(&[component("lda_x", 1.0)], &rho, &[], &[]).expect("valid");
        for (g, &density) in densities.iter().enumerate() {
            let kf = (3.0 * density / PI).cbrt();
            assert!((values.energy_per_particle[g] + 0.75 * kf).abs() < 1e-14);
            assert!((values.vrho[g] + kf).abs() < 1e-14);
            assert!((values.vrho[3 + g] + kf).abs() < 1e-14);
        }
        assert!(values.vsigma.is_empty() && values.vtau.is_empty());
    }

    #[test]
    fn derivatives_match_finite_differences_of_the_energy_density() {
        let components = [component("gga_x_pbe", 0.75), component("gga_c_pbe", 1.0)];
        let point = ([0.3, 0.2], [0.05, 0.02, 0.04]);
        let energy = |rho: [f64; 2], sigma: [f64; 3]| -> f64 {
            let values = evaluate_functional(&components, &rho, &sigma, &[]).expect("valid");
            (rho[0] + rho[1]) * values.energy_per_particle[0]
        };
        let values = evaluate_functional(&components, &point.0, &point.1, &[]).expect("valid");
        let h = 1e-6;
        for spin in 0..2 {
            let mut up = point.0;
            up[spin] += h;
            let mut down = point.0;
            down[spin] -= h;
            let derivative = (energy(up, point.1) - energy(down, point.1)) / (2.0 * h);
            assert!((derivative - values.vrho[spin]).abs() < 1e-8);
        }
        for index in 0..3 {
            let mut up = point.1;
            up[index] += h;
            let mut down = point.1;
            down[index] -= h;
            let derivative = (energy(point.0, up) - energy(point.0, down)) / (2.0 * h);
            assert!((derivative - values.vsigma[index]).abs() < 1e-8);
        }
    }

    #[test]
    fn chunks_join_in_order_and_mixing_is_linear() {
        let n = 2 * POINTS_PER_TASK + 7;
        let rho: Vec<f64> = (0..2 * n).map(|i| 0.1 + 0.001 * to_f64_local(i)).collect();
        let sigma: Vec<f64> = (0..3 * n)
            .map(|i| 0.01 + 0.0001 * to_f64_local(i))
            .collect();
        let tau: Vec<f64> = (0..2 * n).map(|i| 0.2 + 0.001 * to_f64_local(i)).collect();
        let x = evaluate_functional(&[component("mgga_x_tpss", 1.0)], &rho, &sigma, &tau)
            .expect("valid");
        let c = evaluate_functional(&[component("mgga_c_tpss", 1.0)], &rho, &sigma, &tau)
            .expect("valid");
        let mixed = evaluate_functional(
            &[component("mgga_x_tpss", 0.5), component("mgga_c_tpss", 2.0)],
            &rho,
            &sigma,
            &tau,
        )
        .expect("valid");
        assert_eq!(mixed.vtau.len(), 2 * n);
        for g in 0..n {
            let expected = 0.5 * x.energy_per_particle[g] + 2.0 * c.energy_per_particle[g];
            assert!((mixed.energy_per_particle[g] - expected).abs() < 1e-14);
            let single = evaluate_functional(
                &[component("mgga_x_tpss", 1.0)],
                &[rho[g], rho[n + g]],
                &[sigma[g], sigma[n + g], sigma[2 * n + g]],
                &[tau[g], tau[n + g]],
            )
            .expect("valid");
            assert!((single.vtau[1] - x.vtau[n + g]).abs() < 1e-14);
            assert!((single.vsigma[2] - x.vsigma[2 * n + g]).abs() < 1e-14);
        }
    }

    fn to_f64_local(i: usize) -> f64 {
        crate::to_f64(i)
    }

    #[test]
    fn malformed_input_is_rejected() {
        let rho = [0.1, 0.2];
        assert!(matches!(
            evaluate_functional(&[], &rho, &[], &[]),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            evaluate_functional(&[component("gga_x_pbe", 1.0)], &rho, &[], &[]),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            evaluate_functional(&[component("lda_x", 1.0)], &[0.1, 0.2, 0.3], &[], &[]),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            evaluate_functional(&[component("lda_x", f64::NAN)], &rho, &[], &[]),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            evaluate_functional(&[component("lda_x", 1.0)], &[0.1, f64::INFINITY], &[], &[]),
            Err(Error::InvalidArgument(_))
        ));
    }
}

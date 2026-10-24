//! Noncollinear functionals by the multi-collinear approach (ADR-0029).
//!
//! A collinear functional `f(ρ_α, ρ_β, σ, τ)` of the spin densities becomes
//! a functional of the density `ρ` and the magnetization `m`, with their
//! gradients and the kinetic energy densities `τ` and `τ_m`, by projecting
//! the spin variables `S = (m, ∇m, τ_m)` onto a direction `n` of spin
//! space, `s_n = n·S`, evaluating `f` at the spin densities `(ρ ± s_n)/2`
//! and averaging over the sphere with the derivative term that restores
//! the collinear limit:
//!
//! ```text
//! f_mc(ρ, S) = ∫ dΩ/4π [ f(ρ, n·S) + (n·S)·∂f/∂s (ρ, n·S) ]
//!            = ∫ dΩ/4π d/dt [ t f(ρ, t n·S) ] at t = 1.
//! ```
//!
//! Where the spin variables are parallel (a collinear density) the
//! integrand depends on `u = n·m̂` alone, `½ ∫_{−1}^{1} d/du [u f(ρ, u s)]
//! du = ½ [f(ρ, s) + f(ρ, −s)] = f(ρ, s)`, so the collinear functional is
//! recovered exactly. For an LDA the integral has that closed form at
//! every point, `f_mc = f(ρ, |m|)`, the locally collinear functional. For
//! a GGA or meta-GGA the sphere is sampled with a quadrature rule, and
//! the derivatives of `f_mc` involve the second derivatives of `f`, which
//! `xcx` provides. The integrand loses its smoothness where a projected
//! spin density vanishes, `|m| → ρ`, and the rule converges slowly there;
//! points polarized beyond a threshold are evaluated with the locally
//! collinear functional, whose spin gradient is the gradient of `|m|`, and
//! its exact derivatives. Points whose spin variables vanish are evaluated
//! unpolarized, which spares the sphere.
//!
//! Layouts: every array is component-major and point-minor, `m[a][g]`,
//! `grad_rho[i][g]`, `grad_m[a][i][g]` with the spin direction `a` before
//! the spatial direction `i`, `tau_m[a][g]`.

use rayon::prelude::*;
use xcx::{Functional, XcInput, XcResult};

use super::{Rung, build, check_input, combination};
use crate::Error;

/// Points evaluated per parallel task.
const POINTS_PER_TASK: usize = 16;

/// Relative size below which the spin variables of a point count as zero.
///
/// Such a point is evaluated unpolarized: the multi-collinear energy
/// differs from `f(ρ, 0)` by the second order in the spin variables and
/// the spin potential by the first, both negligible at this size.
pub const SPIN_NEGLECT: f64 = 1e-14;

/// Densities of a batch of points in the total/magnetization picture.
///
/// `rho[g]`, `m[a][g]`, `grad_rho[i][g]`, `grad_m[a][i][g]`, `tau[g]` and
/// `tau_m[a][g]`; the gradients may be empty for an LDA, the kinetic
/// energy densities below the meta-GGA rung.
#[derive(Debug, Clone, Copy)]
pub struct NoncollinearInput<'a> {
    /// The density.
    pub rho: &'a [f64],
    /// The magnetization.
    pub m: &'a [f64],
    /// Gradient of the density.
    pub grad_rho: &'a [f64],
    /// Gradient of the magnetization.
    pub grad_m: &'a [f64],
    /// Kinetic energy density `½ Σ_i |∇ψ_i|²` summed over the spins.
    pub tau: &'a [f64],
    /// Spin kinetic energy density `½ Σ_i ∇ψ_i† σ ∇ψ_i`.
    pub tau_m: &'a [f64],
}

/// A quadrature rule on the unit sphere.
#[derive(Debug, Clone, Copy)]
pub struct SphereRule<'a> {
    /// Unit vectors `[k][3]`.
    pub directions: &'a [f64],
    /// Weights `[k]`, summing to one.
    pub weights: &'a [f64],
}

/// Energy density and derivatives of the multi-collinear functional.
///
/// The layout follows [`NoncollinearInput`]: `v_m[a][g]`, `v_grad_rho[i][g]`,
/// `v_grad_m[a][i][g]`, `v_tau_m[a][g]`; the gradient derivatives are
/// empty for an LDA, the kinetic ones below the meta-GGA rung.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NoncollinearValues {
    /// `f_mc[g]`, the energy density (not per particle).
    pub energy_density: Vec<f64>,
    /// `∂f_mc/∂ρ`.
    pub v_rho: Vec<f64>,
    /// `∂f_mc/∂m_a`.
    pub v_m: Vec<f64>,
    /// `∂f_mc/∂(∂_i ρ)`.
    pub v_grad_rho: Vec<f64>,
    /// `∂f_mc/∂(∂_i m_a)`.
    pub v_grad_m: Vec<f64>,
    /// `∂f_mc/∂τ`.
    pub v_tau: Vec<f64>,
    /// `∂f_mc/∂τ_m,a`.
    pub v_tau_m: Vec<f64>,
}

/// Evaluate the multi-collinear form of `Σ_i w_i f_i` at every point.
///
/// `rule` samples the sphere for the GGA and meta-GGA rungs; an LDA has
/// its closed form and ignores it. With `collinear_threshold = Some(θ)`
/// the points with `|m| ≥ θ ρ` take the locally collinear functional.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if the components are invalid (see
/// [`super::evaluate_functional`]), an input the rung needs is missing or
/// not finite, the rule is empty or its weights do not sum to one, or the
/// threshold is not positive, and [`Error::LengthMismatch`] if the arrays
/// disagree on the number of points.
pub fn evaluate_noncollinear(
    components: &[(String, f64)],
    input: &NoncollinearInput<'_>,
    rule: &SphereRule<'_>,
    collinear_threshold: Option<f64>,
) -> Result<NoncollinearValues, Error> {
    let (ids, rung) = combination(components)?;
    let n = input.rho.len();
    check_input(input.rho, n, "densities")?;
    check_input(input.m, 3 * n, "magnetizations")?;
    if rung != Rung::Lda {
        check_input(input.grad_rho, 3 * n, "density gradients")?;
        check_input(input.grad_m, 9 * n, "magnetization gradients")?;
        check_rule(rule)?;
    }
    if rung == Rung::MetaGga {
        check_input(input.tau, n, "kinetic energy densities")?;
        check_input(input.tau_m, 3 * n, "spin kinetic energy densities")?;
    }
    if let Some(threshold) = collinear_threshold {
        if !(threshold.is_finite() && threshold > 0.0) {
            return Err(Error::InvalidArgument(
                "the collinear threshold is a positive number".to_string(),
            ));
        }
    }
    // Built once here to validate the combination, and once per worker
    // below: the evaluator is not `Sync`, so it cannot be shared.
    build(&ids)?;
    let chunks: Vec<Vec<Potential>> = (0..n.div_ceil(POINTS_PER_TASK).max(1))
        .into_par_iter()
        .map_init(
            || build(&ids).ok(),
            |functional, task| {
                let functional = functional.as_ref().ok_or_else(|| {
                    Error::InvalidArgument("the functional could not be built".to_string())
                })?;
                let start = task * POINTS_PER_TASK;
                let end = (start + POINTS_PER_TASK).min(n);
                (start..end)
                    .map(|g| {
                        let point = Point::at(input, rung, n, g);
                        point.evaluate(functional, rung, rule, collinear_threshold)
                    })
                    .collect::<Result<Vec<_>, _>>()
            },
        )
        .collect::<Result<_, _>>()?;
    Ok(assemble(chunks, n, rung))
}

fn check_rule(rule: &SphereRule<'_>) -> Result<(), Error> {
    let count = rule.weights.len();
    if count == 0 {
        return Err(Error::InvalidArgument(
            "the sphere rule has at least one direction".to_string(),
        ));
    }
    check_input(rule.directions, 3 * count, "directions")?;
    check_input(rule.weights, count, "direction weights")?;
    let total: f64 = rule.weights.iter().sum();
    if (total - 1.0).abs() > 1e-10 {
        return Err(Error::InvalidArgument(
            "the weights of the sphere rule sum to one".to_string(),
        ));
    }
    Ok(())
}

/// The densities of one point.
#[derive(Debug, Clone, Copy, Default)]
struct Point {
    rho: f64,
    grad_rho: [f64; 3],
    tau: f64,
    m: [f64; 3],
    /// `grad_m[a][i] = ∂_i m_a`.
    grad_m: [[f64; 3]; 3],
    tau_m: [f64; 3],
}

/// The multi-collinear energy density and its derivatives at one point.
#[derive(Debug, Clone, Copy, Default)]
struct Potential {
    energy: f64,
    v_rho: f64,
    v_grad_rho: [f64; 3],
    v_tau: f64,
    v_m: [f64; 3],
    v_grad_m: [[f64; 3]; 3],
    v_tau_m: [f64; 3],
}

/// The spin variables projected onto one direction: `s = n·m`,
/// `gs_i = n·∂_i m`, `ts = n·τ_m`.
#[derive(Debug, Clone, Copy, Default)]
struct Projection {
    s: f64,
    gs: [f64; 3],
    ts: f64,
}

/// The spin densities of a projection in libxc's variables, with the
/// spin gradients the chain rule needs.
#[derive(Debug, Clone, Copy, Default)]
struct Collinear {
    rho: [f64; 2],
    grad: [[f64; 3]; 2],
    sigma: [f64; 3],
    tau: [f64; 2],
}

/// First derivatives of `f` in the total/spin variables at one projection.
#[derive(Debug, Clone, Copy, Default)]
struct Gradient {
    f: f64,
    t: f64,
    s: f64,
    gt: [f64; 3],
    gs: [f64; 3],
    tt: f64,
    ts: f64,
}

/// The second derivatives of `f` applied to the spin variables of a
/// projection, `Σ_y S_y ∂²f/∂S_y ∂X`, in the total/spin variables `X`.
#[derive(Debug, Clone, Copy, Default)]
struct SpinHessian {
    t: f64,
    s: f64,
    gt: [f64; 3],
    gs: [f64; 3],
    tt: f64,
    ts: f64,
}

/// How a point is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scheme {
    /// Spin variables negligible: `f(ρ, 0)`.
    Unpolarized,
    /// The locally collinear functional `f(ρ, |m|, ∇|m|, m̂·τ_m)`.
    LocallyCollinear,
    /// The quadrature over the sphere.
    Sphere,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

impl Point {
    fn at(input: &NoncollinearInput<'_>, rung: Rung, n: usize, g: usize) -> Self {
        let mut point = Self {
            rho: input.rho[g],
            ..Self::default()
        };
        for a in 0..3 {
            point.m[a] = input.m[a * n + g];
        }
        if rung != Rung::Lda {
            for i in 0..3 {
                point.grad_rho[i] = input.grad_rho[i * n + g];
                for a in 0..3 {
                    point.grad_m[a][i] = input.grad_m[(a * 3 + i) * n + g];
                }
            }
        }
        if rung == Rung::MetaGga {
            point.tau = input.tau[g];
            for a in 0..3 {
                point.tau_m[a] = input.tau_m[a * n + g];
            }
        }
        point
    }

    fn scheme(&self, rung: Rung, threshold: Option<f64>) -> Scheme {
        let negligible = |spin: f64, total: f64| spin <= SPIN_NEGLECT * total.abs();
        let magnetization = norm(self.m);
        let grad_m = self
            .grad_m
            .iter()
            .map(|row| dot(*row, *row))
            .sum::<f64>()
            .sqrt();
        if negligible(magnetization, self.rho)
            && negligible(grad_m, norm(self.grad_rho))
            && negligible(norm(self.tau_m), self.tau)
        {
            return Scheme::Unpolarized;
        }
        if rung == Rung::Lda || magnetization == 0.0 {
            // An LDA has the closed form; a vanishing magnetization with
            // other spin variables present (a node of m) stays on the sphere,
            // unless the rung does not read them.
            return if magnetization == 0.0 && rung != Rung::Lda {
                Scheme::Sphere
            } else {
                Scheme::LocallyCollinear
            };
        }
        match threshold {
            Some(theta) if magnetization >= theta * self.rho => Scheme::LocallyCollinear,
            _ => Scheme::Sphere,
        }
    }

    fn project(&self, n: [f64; 3]) -> Projection {
        let mut gs = [0.0; 3];
        for i in 0..3 {
            gs[i] = n[0] * self.grad_m[0][i] + n[1] * self.grad_m[1][i] + n[2] * self.grad_m[2][i];
        }
        Projection {
            s: dot(n, self.m),
            gs,
            ts: dot(n, self.tau_m),
        }
    }

    fn collinear(&self, projection: &Projection) -> Collinear {
        let mut grad = [[0.0; 3]; 2];
        for i in 0..3 {
            grad[0][i] = 0.5 * (self.grad_rho[i] + projection.gs[i]);
            grad[1][i] = 0.5 * (self.grad_rho[i] - projection.gs[i]);
        }
        Collinear {
            rho: [
                0.5 * (self.rho + projection.s),
                0.5 * (self.rho - projection.s),
            ],
            grad,
            sigma: [
                dot(grad[0], grad[0]),
                dot(grad[0], grad[1]),
                dot(grad[1], grad[1]),
            ],
            tau: [
                0.5 * (self.tau + projection.ts),
                0.5 * (self.tau - projection.ts),
            ],
        }
    }

    fn evaluate(
        &self,
        functional: &Functional,
        rung: Rung,
        rule: &SphereRule<'_>,
        threshold: Option<f64>,
    ) -> Result<Potential, Error> {
        match self.scheme(rung, threshold) {
            Scheme::Unpolarized => {
                let projection = Projection::default();
                let gradient = self.gradient_at(functional, rung, &projection)?;
                Ok(Potential {
                    energy: gradient.f,
                    v_rho: gradient.t,
                    v_grad_rho: gradient.gt,
                    v_tau: gradient.tt,
                    ..Potential::default()
                })
            }
            Scheme::LocallyCollinear => self.locally_collinear(functional, rung),
            Scheme::Sphere => self.sphere(functional, rung, rule),
        }
    }

    /// First derivatives at a single projection.
    fn gradient_at(
        &self,
        functional: &Functional,
        rung: Rung,
        projection: &Projection,
    ) -> Result<Gradient, Error> {
        let collinear = self.collinear(projection);
        let packed = Packed::new(rung, std::slice::from_ref(&collinear));
        let result = functional
            .eval(1, &packed.input(rung))
            .map_err(|error| Error::InvalidArgument(format!("functional evaluation: {error}")))?;
        Ok(gradient(&result, 0, rung, &collinear))
    }

    /// The locally collinear functional `f(ρ, |m|, m̂·∇m, m̂·τ_m)` and its
    /// exact derivatives; `m̂·∂_i m = ∂_i |m|`.
    fn locally_collinear(&self, functional: &Functional, rung: Rung) -> Result<Potential, Error> {
        let magnitude = norm(self.m);
        let unit = [
            self.m[0] / magnitude,
            self.m[1] / magnitude,
            self.m[2] / magnitude,
        ];
        let projection = self.project(unit);
        let gradient = self.gradient_at(functional, rung, &projection)?;
        let mut potential = Potential {
            energy: gradient.f,
            v_rho: gradient.t,
            v_grad_rho: gradient.gt,
            v_tau: gradient.tt,
            ..Potential::default()
        };
        for a in 0..3 {
            // ∂|m|/∂m_a = m̂_a; ∂(m̂·v)/∂m_a = (v_a − m̂_a m̂·v)/|m|.
            let mut v_m = gradient.s * unit[a];
            for i in 0..3 {
                v_m +=
                    gradient.gs[i] * (self.grad_m[a][i] - unit[a] * projection.gs[i]) / magnitude;
                potential.v_grad_m[a][i] = gradient.gs[i] * unit[a];
            }
            v_m += gradient.ts * (self.tau_m[a] - unit[a] * projection.ts) / magnitude;
            potential.v_m[a] = v_m;
            potential.v_tau_m[a] = gradient.ts * unit[a];
        }
        Ok(potential)
    }

    /// The quadrature over the sphere with the second derivatives of `f`.
    fn sphere(
        &self,
        functional: &Functional,
        rung: Rung,
        rule: &SphereRule<'_>,
    ) -> Result<Potential, Error> {
        let count = rule.weights.len();
        let projections: Vec<Projection> = (0..count)
            .map(|k| {
                self.project([
                    rule.directions[3 * k],
                    rule.directions[3 * k + 1],
                    rule.directions[3 * k + 2],
                ])
            })
            .collect();
        let collinear: Vec<Collinear> = projections.iter().map(|p| self.collinear(p)).collect();
        let packed = Packed::new(rung, &collinear);
        let result = functional
            .eval_fxc(count, &packed.input(rung))
            .map_err(|error| Error::InvalidArgument(format!("functional evaluation: {error}")))?;
        let mut potential = Potential::default();
        for (k, (projection, collinear)) in projections.iter().zip(&collinear).enumerate() {
            let w = rule.weights[k];
            let n = [
                rule.directions[3 * k],
                rule.directions[3 * k + 1],
                rule.directions[3 * k + 2],
            ];
            let g = gradient(&result, k, rung, collinear);
            let h = spin_hessian(&result, k, rung, collinear, projection);
            potential.energy +=
                w * (g.f + projection.s * g.s + dot(projection.gs, g.gs) + projection.ts * g.ts);
            potential.v_rho += w * (g.t + h.t);
            potential.v_tau += w * (g.tt + h.tt);
            for i in 0..3 {
                potential.v_grad_rho[i] += w * (g.gt[i] + h.gt[i]);
            }
            for a in 0..3 {
                potential.v_m[a] += w * n[a] * (2.0 * g.s + h.s);
                potential.v_tau_m[a] += w * n[a] * (2.0 * g.ts + h.ts);
                for i in 0..3 {
                    potential.v_grad_m[a][i] += w * n[a] * (2.0 * g.gs[i] + h.gs[i]);
                }
            }
        }
        Ok(potential)
    }
}

/// libxc-layout inputs of several projections, point-major as `xcx` reads them.
struct Packed {
    rho: Vec<f64>,
    sigma: Vec<f64>,
    tau: Vec<f64>,
}

impl Packed {
    fn new(rung: Rung, entries: &[Collinear]) -> Self {
        let mut packed = Self {
            rho: Vec::with_capacity(2 * entries.len()),
            sigma: Vec::new(),
            tau: Vec::new(),
        };
        for entry in entries {
            packed.rho.extend_from_slice(&entry.rho);
            if rung != Rung::Lda {
                packed.sigma.extend_from_slice(&entry.sigma);
            }
            if rung == Rung::MetaGga {
                packed.tau.extend_from_slice(&entry.tau);
            }
        }
        packed
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

/// First derivatives of `f` at entry `j` in the total/spin variables.
///
/// With `h_σ = ∂f/∂∇ρ_σ = 2 v_σσ ∇ρ_σ + v_αβ ∇ρ_σ'`, the derivatives with
/// respect to a total variable are `(∂_α + ∂_β)/2`, to a spin variable
/// `(∂_α − ∂_β)/2`.
fn gradient(result: &XcResult, j: usize, rung: Rung, c: &Collinear) -> Gradient {
    let va = result.vrho[2 * j];
    let vb = result.vrho[2 * j + 1];
    let mut g = Gradient {
        f: result.exc[j] * (c.rho[0] + c.rho[1]),
        t: 0.5 * (va + vb),
        s: 0.5 * (va - vb),
        ..Gradient::default()
    };
    if rung != Rung::Lda {
        let vs = &result.vsigma[3 * j..3 * j + 3];
        for i in 0..3 {
            let ha = 2.0 * vs[0] * c.grad[0][i] + vs[1] * c.grad[1][i];
            let hb = 2.0 * vs[2] * c.grad[1][i] + vs[1] * c.grad[0][i];
            g.gt[i] = 0.5 * (ha + hb);
            g.gs[i] = 0.5 * (ha - hb);
        }
    }
    if rung == Rung::MetaGga {
        let ta = result.vtau[2 * j];
        let tb = result.vtau[2 * j + 1];
        g.tt = 0.5 * (ta + tb);
        g.ts = 0.5 * (ta - tb);
    }
    g
}

/// Index of the symmetric pair `(a, b)` in libxc's upper-triangle layout
/// of three variables: `aa, ab, bb` for two, `aa, ab, ac, bb, bc, cc` for three.
fn upper(a: usize, b: usize, size: usize) -> usize {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    a * size - a * (a + 1) / 2 + b
}

/// The second derivatives of `f` at entry `j` applied to the spin variables
/// of `p`, as the change of the first derivatives along
/// `δ(ρ_α, ρ_β, ∇ρ_α, ∇ρ_β, τ_α, τ_β) = (s, −s, ∇s, −∇s, τ_s, −τ_s)/2`.
#[expect(
    clippy::too_many_lines,
    reason = "the chain rule through libxc's variables is written out once"
)]
fn spin_hessian(
    result: &XcResult,
    j: usize,
    rung: Rung,
    c: &Collinear,
    p: &Projection,
) -> SpinHessian {
    let d_rho = [0.5 * p.s, -0.5 * p.s];
    let mut d_grad = [[0.0; 3]; 2];
    for i in 0..3 {
        d_grad[0][i] = 0.5 * p.gs[i];
        d_grad[1][i] = -0.5 * p.gs[i];
    }
    let d_tau = [0.5 * p.ts, -0.5 * p.ts];
    let d_sigma = [
        2.0 * dot(c.grad[0], d_grad[0]),
        dot(c.grad[0], d_grad[1]) + dot(c.grad[1], d_grad[0]),
        2.0 * dot(c.grad[1], d_grad[1]),
    ];
    let rr = &result.v2rho2[3 * j..3 * j + 3];
    let mut dv_rho = [
        rr[0] * d_rho[0] + rr[1] * d_rho[1],
        rr[1] * d_rho[0] + rr[2] * d_rho[1],
    ];
    let mut dv_sigma = [0.0; 3];
    let mut dv_tau = [0.0; 2];
    if rung != Rung::Lda {
        let rs = &result.v2rhosigma[6 * j..6 * j + 6];
        let ss = &result.v2sigma2[6 * j..6 * j + 6];
        for k in 0..3 {
            for spin in 0..2 {
                dv_rho[spin] += rs[3 * spin + k] * d_sigma[k];
                dv_sigma[k] += rs[3 * spin + k] * d_rho[spin];
            }
            for l in 0..3 {
                dv_sigma[k] += ss[upper(k, l, 3)] * d_sigma[l];
            }
        }
    }
    if rung == Rung::MetaGga {
        let rt = &result.v2rhotau[4 * j..4 * j + 4];
        let st = &result.v2sigmatau[6 * j..6 * j + 6];
        let tt = &result.v2tau2[3 * j..3 * j + 3];
        for spin in 0..2 {
            for other in 0..2 {
                dv_rho[spin] += rt[2 * spin + other] * d_tau[other];
                dv_tau[other] += rt[2 * spin + other] * d_rho[spin];
                dv_tau[spin] += tt[upper(spin, other, 2)] * d_tau[other];
            }
            for k in 0..3 {
                dv_sigma[k] += st[2 * k + spin] * d_tau[spin];
                dv_tau[spin] += st[2 * k + spin] * d_sigma[k];
            }
        }
    }
    let mut h = SpinHessian {
        t: 0.5 * (dv_rho[0] + dv_rho[1]),
        s: 0.5 * (dv_rho[0] - dv_rho[1]),
        tt: 0.5 * (dv_tau[0] + dv_tau[1]),
        ts: 0.5 * (dv_tau[0] - dv_tau[1]),
        ..SpinHessian::default()
    };
    if rung != Rung::Lda {
        let vs = &result.vsigma[3 * j..3 * j + 3];
        for i in 0..3 {
            let dha = 2.0 * dv_sigma[0] * c.grad[0][i]
                + 2.0 * vs[0] * d_grad[0][i]
                + dv_sigma[1] * c.grad[1][i]
                + vs[1] * d_grad[1][i];
            let dhb = 2.0 * dv_sigma[2] * c.grad[1][i]
                + 2.0 * vs[2] * d_grad[1][i]
                + dv_sigma[1] * c.grad[0][i]
                + vs[1] * d_grad[0][i];
            h.gt[i] = 0.5 * (dha + dhb);
            h.gs[i] = 0.5 * (dha - dhb);
        }
    }
    h
}

/// Join the chunks into component-major arrays over all `n` points.
fn assemble(chunks: Vec<Vec<Potential>>, n: usize, rung: Rung) -> NoncollinearValues {
    let gradients = rung != Rung::Lda;
    let kinetic = rung == Rung::MetaGga;
    let mut values = NoncollinearValues {
        energy_density: vec![0.0; n],
        v_rho: vec![0.0; n],
        v_m: vec![0.0; 3 * n],
        v_grad_rho: vec![0.0; if gradients { 3 * n } else { 0 }],
        v_grad_m: vec![0.0; if gradients { 9 * n } else { 0 }],
        v_tau: vec![0.0; if kinetic { n } else { 0 }],
        v_tau_m: vec![0.0; if kinetic { 3 * n } else { 0 }],
    };
    for (g, potential) in chunks.into_iter().flatten().enumerate() {
        values.energy_density[g] = potential.energy;
        values.v_rho[g] = potential.v_rho;
        for a in 0..3 {
            values.v_m[a * n + g] = potential.v_m[a];
        }
        if gradients {
            for i in 0..3 {
                values.v_grad_rho[i * n + g] = potential.v_grad_rho[i];
                for a in 0..3 {
                    values.v_grad_m[(a * 3 + i) * n + g] = potential.v_grad_m[a][i];
                }
            }
        }
        if kinetic {
            values.v_tau[g] = potential.v_tau;
            for a in 0..3 {
                values.v_tau_m[a * n + g] = potential.v_tau_m[a];
            }
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::super::evaluate_functional;
    use super::*;

    fn components(names: &[&str]) -> Vec<(String, f64)> {
        names
            .iter()
            .map(|name| ((*name).to_string(), 1.0))
            .collect()
    }

    /// Nodes and weights of the Gauss–Legendre rule with `n` points on `[−1, 1]`.
    fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut nodes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for k in 0..n {
            let mut x = (PI * (k as f64 + 0.75) / (n as f64 + 0.5)).cos();
            let mut derivative = 0.0;
            for _ in 0..100 {
                let (mut p0, mut p1) = (1.0, x);
                for degree in 2..=n {
                    let d = degree as f64;
                    let p2 = ((2.0 * d - 1.0) * x * p1 - (d - 1.0) * p0) / d;
                    p0 = p1;
                    p1 = p2;
                }
                derivative = n as f64 * (x * p1 - p0) / (x * x - 1.0);
                let step = p1 / derivative;
                x -= step;
                if step.abs() < 1e-15 {
                    break;
                }
            }
            nodes.push(x);
            weights.push(2.0 / ((1.0 - x * x) * derivative * derivative));
        }
        (nodes, weights)
    }

    /// A product rule on the sphere: Gauss–Legendre in `cos θ`, uniform in `φ`.
    fn product_rule(n_theta: usize, n_phi: usize) -> (Vec<f64>, Vec<f64>) {
        let (nodes, node_weights) = gauss_legendre(n_theta);
        let mut directions = Vec::new();
        let mut weights = Vec::new();
        for (u, wu) in nodes.iter().zip(&node_weights) {
            let sine = (1.0 - u * u).sqrt();
            for k in 0..n_phi {
                let phi = 2.0 * PI * (k as f64 + 0.5) / n_phi as f64;
                directions.extend_from_slice(&[sine * phi.cos(), sine * phi.sin(), *u]);
                weights.push(wu / 2.0 / n_phi as f64);
            }
        }
        (directions, weights)
    }

    /// One noncollinear point of every variable, as flat component-major arrays.
    struct Sample {
        rho: Vec<f64>,
        m: Vec<f64>,
        grad_rho: Vec<f64>,
        grad_m: Vec<f64>,
        tau: Vec<f64>,
        tau_m: Vec<f64>,
    }

    impl Sample {
        /// A physical point: spin densities `(ρ ± s)/2` along the axis `u`
        /// with the beta quantities shrinking together as the polarization
        /// grows (`τ_β ∝ ρ_β^{5/3}`), plus transverse spin components about
        /// the axis so that `m`, `∇m` and `τ_m` are not parallel.
        fn noncollinear(polarization: f64) -> Self {
            let rho = 0.8;
            let u = {
                let axis = [0.3, -0.5, 0.4];
                let length = norm(axis);
                [axis[0] / length, axis[1] / length, axis[2] / length]
            };
            let v = [0.8, 0.6, 0.15];
            let v = {
                let along = dot(v, u);
                let t = [
                    v[0] - along * u[0],
                    v[1] - along * u[1],
                    v[2] - along * u[2],
                ];
                let length = norm(t);
                [t[0] / length, t[1] / length, t[2] / length]
            };
            let w = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let beta = 1.0 - polarization;
            let s = rho - 2.0 * (0.5 * rho * beta);
            let grad_rho = [0.2, -0.1, 0.3];
            let grad_beta = [0.05 * beta, -0.025 * beta, 0.075 * beta];
            let tau = 1.1;
            let tau_beta = 0.25 * (beta / 0.5).powf(5.0 / 3.0);
            let tau_s = tau - 2.0 * tau_beta;
            let transverse = 0.02 * (beta + 0.1);
            let mut grad_m = vec![0.0; 9];
            for a in 0..3 {
                for i in 0..3 {
                    let grad_s = grad_rho[i] - 2.0 * grad_beta[i];
                    grad_m[3 * a + i] = u[a] * grad_s
                        + transverse * (v[a] * [1.0, -2.0, 1.0][i] + w[a] * [0.5, 0.0, -1.0][i]);
                }
            }
            Self {
                rho: vec![rho],
                m: (0..3).map(|a| s * u[a] + 0.05 * beta * v[a]).collect(),
                grad_rho: grad_rho.to_vec(),
                grad_m,
                tau: vec![tau],
                tau_m: (0..3).map(|a| tau_s * u[a] + 0.1 * beta * w[a]).collect(),
            }
        }

        /// A collinear point: every spin variable along `z`.
        fn collinear(polarization: f64) -> Self {
            let rho = 0.8;
            Self {
                rho: vec![rho],
                m: vec![0.0, 0.0, polarization * rho],
                grad_rho: vec![0.2, -0.1, 0.3],
                grad_m: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, -0.08, 0.1],
                tau: vec![1.1],
                tau_m: vec![0.0, 0.0, 0.3],
            }
        }

        fn input(&self) -> NoncollinearInput<'_> {
            NoncollinearInput {
                rho: &self.rho,
                m: &self.m,
                grad_rho: &self.grad_rho,
                grad_m: &self.grad_m,
                tau: &self.tau,
                tau_m: &self.tau_m,
            }
        }

        /// The same point as libxc's spin densities for a collinear `m` along `z`.
        fn spin_densities(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let s = self.m[2];
            let rho = vec![0.5 * (self.rho[0] + s), 0.5 * (self.rho[0] - s)];
            let ga: Vec<f64> = (0..3)
                .map(|i| 0.5 * (self.grad_rho[i] + self.grad_m[6 + i]))
                .collect();
            let gb: Vec<f64> = (0..3)
                .map(|i| 0.5 * (self.grad_rho[i] - self.grad_m[6 + i]))
                .collect();
            let dot = |a: &[f64], b: &[f64]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            let sigma = vec![dot(&ga, &ga), dot(&ga, &gb), dot(&gb, &gb)];
            let tau = vec![
                0.5 * (self.tau[0] + self.tau_m[2]),
                0.5 * (self.tau[0] - self.tau_m[2]),
            ];
            (rho, sigma, tau)
        }

        /// All variables as one flat vector, for finite differences.
        fn flat(&self) -> Vec<f64> {
            [
                &self.rho[..],
                &self.m,
                &self.grad_rho,
                &self.grad_m,
                &self.tau,
                &self.tau_m,
            ]
            .concat()
        }

        fn from_flat(flat: &[f64]) -> Self {
            Self {
                rho: flat[0..1].to_vec(),
                m: flat[1..4].to_vec(),
                grad_rho: flat[4..7].to_vec(),
                grad_m: flat[7..16].to_vec(),
                tau: flat[16..17].to_vec(),
                tau_m: flat[17..20].to_vec(),
            }
        }
    }

    fn evaluate(
        names: &[&str],
        sample: &Sample,
        rule: &(Vec<f64>, Vec<f64>),
        threshold: Option<f64>,
    ) -> NoncollinearValues {
        evaluate_noncollinear(
            &components(names),
            &sample.input(),
            &SphereRule {
                directions: &rule.0,
                weights: &rule.1,
            },
            threshold,
        )
        .expect("valid")
    }

    /// The derivatives of `values` as one flat vector in the order of `Sample::flat`.
    fn flat_potential(values: &NoncollinearValues, rung: Rung) -> Vec<f64> {
        let zeros = |count: usize, present: bool, data: &[f64]| {
            if present {
                data.to_vec()
            } else {
                vec![0.0; count]
            }
        };
        [
            values.v_rho.clone(),
            values.v_m.clone(),
            zeros(3, rung != Rung::Lda, &values.v_grad_rho),
            zeros(9, rung != Rung::Lda, &values.v_grad_m),
            zeros(1, rung == Rung::MetaGga, &values.v_tau),
            zeros(3, rung == Rung::MetaGga, &values.v_tau_m),
        ]
        .concat()
    }

    #[test]
    fn collinear_points_recover_the_spin_polarized_functional() {
        // Along z the integrand depends on n_z alone and the rule of degree
        // 47 integrates it to rounding at moderate polarization; beyond the
        // threshold the locally collinear form is exact.
        let rule = product_rule(24, 16);
        for (names, rung) in [
            (vec!["lda_x", "lda_c_vwn"], Rung::Lda),
            (vec!["gga_x_pbe", "gga_c_pbe"], Rung::Gga),
            (vec!["mgga_x_tpss", "mgga_c_tpss"], Rung::MetaGga),
        ] {
            for (polarization, threshold) in [(0.4, None), (0.995, Some(0.99))] {
                let sample = Sample::collinear(polarization);
                let values = evaluate(&names, &sample, &rule, threshold);
                let (rho, sigma, tau) = sample.spin_densities();
                let reference = evaluate_functional(
                    &components(&names),
                    &rho,
                    if rung == Rung::Lda { &[] } else { &sigma },
                    if rung == Rung::MetaGga { &tau } else { &[] },
                )
                .expect("valid");
                let energy = reference.energy_per_particle[0] * (rho[0] + rho[1]);
                assert!((values.energy_density[0] - energy).abs() < 1e-12);
                let (va, vb) = (reference.vrho[0], reference.vrho[1]);
                assert!((values.v_rho[0] - 0.5 * (va + vb)).abs() < 1e-12);
                assert!(values.v_m[0].abs() < 1e-12 && values.v_m[1].abs() < 1e-12);
                assert!((values.v_m[2] - 0.5 * (va - vb)).abs() < 1e-12);
                if rung != Rung::Lda {
                    let ga: Vec<f64> = (0..3)
                        .map(|i| 0.5 * (sample.grad_rho[i] + sample.grad_m[6 + i]))
                        .collect();
                    let gb: Vec<f64> = (0..3)
                        .map(|i| 0.5 * (sample.grad_rho[i] - sample.grad_m[6 + i]))
                        .collect();
                    let vs = &reference.vsigma;
                    for i in 0..3 {
                        let ha = 2.0 * vs[0] * ga[i] + vs[1] * gb[i];
                        let hb = 2.0 * vs[2] * gb[i] + vs[1] * ga[i];
                        assert!((values.v_grad_rho[i] - 0.5 * (ha + hb)).abs() < 1e-12);
                        assert!((values.v_grad_m[6 + i] - 0.5 * (ha - hb)).abs() < 1e-12);
                        assert!(
                            values.v_grad_m[i].abs() < 1e-12
                                && values.v_grad_m[3 + i].abs() < 1e-12
                        );
                    }
                }
                if rung == Rung::MetaGga {
                    let (ta, tb) = (reference.vtau[0], reference.vtau[1]);
                    assert!((values.v_tau[0] - 0.5 * (ta + tb)).abs() < 1e-12);
                    assert!((values.v_tau_m[2] - 0.5 * (ta - tb)).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn lda_is_the_locally_collinear_functional() {
        let rule = product_rule(2, 2);
        let sample = Sample::noncollinear(0.6);
        let values = evaluate(&["lda_x", "lda_c_pw"], &sample, &rule, None);
        let magnitude = norm([sample.m[0], sample.m[1], sample.m[2]]);
        let rho = [
            0.5 * (sample.rho[0] + magnitude),
            0.5 * (sample.rho[0] - magnitude),
        ];
        let reference = evaluate_functional(&components(&["lda_x", "lda_c_pw"]), &rho, &[], &[])
            .expect("valid");
        let energy = reference.energy_per_particle[0] * sample.rho[0];
        assert!((values.energy_density[0] - energy).abs() < 1e-13);
        let spin = 0.5 * (reference.vrho[0] - reference.vrho[1]);
        for a in 0..3 {
            assert!((values.v_m[a] - spin * sample.m[a] / magnitude).abs() < 1e-13);
        }
        assert!(values.v_grad_rho.is_empty() && values.v_tau.is_empty());
    }

    #[test]
    fn derivatives_match_finite_differences_of_the_energy() {
        // Any rule with weights summing to one defines a functional whose
        // derivatives the kernel must reproduce; the threshold selects the
        // locally collinear branch at the strongly polarized point.
        let rule = product_rule(6, 8);
        // Small enough for the beta channel of the strongly polarized point,
        // whose kinetic energy density is 1e-4.
        let step = 1e-7;
        for (names, rung) in [
            (vec!["lda_x", "lda_c_vwn"], Rung::Lda),
            (vec!["gga_x_pbe", "gga_c_pbe"], Rung::Gga),
            (vec!["mgga_x_tpss", "mgga_c_tpss"], Rung::MetaGga),
        ] {
            for (polarization, threshold) in [(0.5, None), (0.5, Some(0.3)), (0.995, Some(0.99))] {
                let sample = Sample::noncollinear(polarization);
                let values = evaluate(&names, &sample, &rule, threshold);
                let potential = flat_potential(&values, rung);
                let flat = sample.flat();
                for (index, variable) in flat.iter().enumerate() {
                    let active = match index {
                        0..=3 => true,
                        4..=15 => rung != Rung::Lda,
                        _ => rung == Rung::MetaGga,
                    };
                    if !active {
                        continue;
                    }
                    let h = step * variable.abs().max(0.1);
                    let mut up = flat.clone();
                    up[index] += h;
                    let mut down = flat.clone();
                    down[index] -= h;
                    let plus = evaluate(&names, &Sample::from_flat(&up), &rule, threshold);
                    let minus = evaluate(&names, &Sample::from_flat(&down), &rule, threshold);
                    let difference = (plus.energy_density[0] - minus.energy_density[0]) / (2.0 * h);
                    assert!(
                        (difference - potential[index]).abs() < 1e-7,
                        "{names:?} at polarization {polarization} with threshold {threshold:?}, \
                         variable {index}: {difference} vs {}",
                        potential[index]
                    );
                }
            }
        }
    }

    #[test]
    fn vanishing_spin_variables_are_evaluated_unpolarized() {
        let rule = product_rule(2, 2);
        let mut sample = Sample::noncollinear(0.0);
        sample.m = vec![0.0; 3];
        sample.grad_m = vec![0.0; 9];
        sample.tau_m = vec![0.0; 3];
        let values = evaluate(&["gga_x_pbe", "gga_c_pbe"], &sample, &rule, None);
        let rho = [0.5 * sample.rho[0], 0.5 * sample.rho[0]];
        let sigma: Vec<f64> = [1.0, 1.0, 1.0]
            .iter()
            .map(|_| 0.25 * dot_slice(&sample.grad_rho, &sample.grad_rho))
            .collect();
        let reference =
            evaluate_functional(&components(&["gga_x_pbe", "gga_c_pbe"]), &rho, &sigma, &[])
                .expect("valid");
        assert!(
            (values.energy_density[0] - reference.energy_per_particle[0] * sample.rho[0]).abs()
                < 1e-14
        );
        assert!(values.v_m.iter().all(|v| *v == 0.0));
        assert!(values.v_grad_m.iter().all(|v| *v == 0.0));
    }

    fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn inputs_are_validated() {
        let rule = product_rule(2, 2);
        let sample = Sample::noncollinear(0.5);
        let names = ["gga_x_pbe"];
        assert!(matches!(
            evaluate_noncollinear(
                &components(&names),
                &sample.input(),
                &SphereRule {
                    directions: &rule.0,
                    weights: &rule.1
                },
                Some(0.0)
            ),
            Err(Error::InvalidArgument(_))
        ));
        let mut uneven = rule.1.clone();
        uneven[0] += 0.5;
        assert!(matches!(
            evaluate_noncollinear(
                &components(&names),
                &sample.input(),
                &SphereRule {
                    directions: &rule.0,
                    weights: &uneven
                },
                None
            ),
            Err(Error::InvalidArgument(_))
        ));
        let short = NoncollinearInput {
            grad_m: &sample.grad_m[..6],
            ..sample.input()
        };
        assert!(matches!(
            evaluate_noncollinear(
                &components(&names),
                &short,
                &SphereRule {
                    directions: &rule.0,
                    weights: &rule.1
                },
                None
            ),
            Err(Error::LengthMismatch { .. })
        ));
        assert_eq!(upper(2, 1, 3), 4);
        assert_eq!(upper(1, 1, 2), 2);
    }
}

//! Hermite Coulomb integrals `R_{tuv}(α, R_PC)`.
//!
//! `R^0_{tuv}` is the Coulomb interaction of the Hermite Gaussian
//! `Λ_{tuv}(r; α, P)` with a unit point charge at `C`, up to the factor
//! `2π/α` applied by the caller (Helgaker 9.9.9). The auxiliary integrals
//! `R^n_{000} = (-2α)^n F_n(α R_PC²)` feed the recursions 9.9.18 to 9.9.20.
//! A [`HermiteWorkspace`] owns the buffers so that the hot loops of the
//! kernels allocate nothing.

use crate::Error;
use crate::special::fill_boys;
use crate::to_f64;

/// The Coulomb kernel of an integral: `1/r`, or the long- or short-range
/// part of its range separation `1/r = erf(ωr)/r + erfc(ωr)/r` (ADR-0012).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoulombKernel {
    /// `1 / r`.
    Full,
    /// `erf(ω r) / r`, the long-range part.
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
    pub(crate) fn omega(self) -> Option<f64> {
        match self {
            Self::Full => None,
            Self::LongRange(w) | Self::ShortRange(w) => Some(w),
        }
    }

    /// Reject a kernel built with a non-positive `omega`.
    pub(crate) fn validate(self) -> Result<(), Error> {
        check_omega(self.omega())
    }

    /// The exponent range `[s_lo, s_hi]` with
    /// `R_000 = (1/√α) ∫_{s_lo}^{s_hi} exp(−s² R²) ds` for Hermite Coulomb
    /// integrals of reduced exponent `α`: `[0, √α]` for the full kernel,
    /// `[0, √α']` and `[√α', √α]` for the long- and short-range parts with
    /// the attenuated exponent `α' = α ω² / (α + ω²)`.
    pub(crate) fn exponent_range(self, alpha: f64) -> (f64, f64) {
        match self {
            Self::Full => (0.0, alpha.sqrt()),
            Self::LongRange(w) => (0.0, attenuated(alpha, w).sqrt()),
            Self::ShortRange(w) => (attenuated(alpha, w).sqrt(), alpha.sqrt()),
        }
    }
}

/// `α ω² / (α + ω²)`, the exponent of the attenuated Hermite integrals.
fn attenuated(alpha: f64, omega: f64) -> f64 {
    alpha * omega * omega / (alpha + omega * omega)
}

pub(crate) fn check_omega(omega: Option<f64>) -> Result<(), Error> {
    if omega.is_some_and(|w| w.is_nan() || w <= 0.0) {
        return Err(Error::InvalidArgument(
            "the range-separation parameter omega is positive".to_string(),
        ));
    }
    Ok(())
}

/// Buffers for the Hermite Coulomb recursion, reused across calls.
#[derive(Debug, Default)]
pub(crate) struct HermiteWorkspace {
    boys: Vec<f64>,
    previous: Vec<f64>,
    current: Vec<f64>,
    /// The short-range table, the full one minus the attenuated one.
    difference: Vec<f64>,
    /// Scratch for the kernels that contract the table with expansion
    /// coefficients before contracting with the other side.
    pub(crate) intermediate: Vec<f64>,
    /// Scratch for the Cartesian values of one primitive quartet.
    pub(crate) values: Vec<f64>,
    /// Scratch for the values with the bra's angular momentum raised and
    /// lowered, which the centre derivatives combine.
    pub(crate) raised: Vec<f64>,
    pub(crate) lowered: Vec<f64>,
}

/// A table of Hermite Coulomb integrals `R^0_{tuv}`, borrowed or owned.
pub(crate) trait HermiteIntegrals {
    /// `R^0_{tuv}`.
    fn get(&self, t: usize, u: usize, v: usize) -> f64;
}

impl HermiteIntegrals for HermiteView<'_> {
    #[inline]
    fn get(&self, t: usize, u: usize, v: usize) -> f64 {
        self.data[(t * self.stride + u) * self.stride + v]
    }
}

impl HermiteIntegrals for HermiteTable {
    #[inline]
    fn get(&self, t: usize, u: usize, v: usize) -> f64 {
        self.data[(t * self.stride + u) * self.stride + v]
    }
}

/// A read-only view of `R^0_{tuv}` for `t + u + v ≤ order`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HermiteView<'a> {
    data: &'a [f64],
    stride: usize,
}

/// An owned table of `R^0_{tuv}`, used to accumulate over point charges.
#[derive(Debug, Clone)]
pub(crate) struct HermiteTable {
    data: Vec<f64>,
    stride: usize,
}

impl HermiteTable {
    /// A zero table of the given order.
    pub(crate) fn zeros(order: usize) -> Self {
        let stride = order + 1;
        Self {
            data: vec![0.0; stride * stride * stride],
            stride,
        }
    }

    /// Add `scale` times a view of the same order.
    pub(crate) fn add_scaled(&mut self, other: HermiteView<'_>, scale: f64) {
        debug_assert_eq!(self.stride, other.stride);
        for (target, source) in self.data.iter_mut().zip(other.data) {
            *target += scale * source;
        }
    }
}

impl HermiteWorkspace {
    /// Compute `R^0_{tuv}` for all `t + u + v ≤ order` with exponent `alpha`
    /// and displacement `pc = P - C`; the view borrows the workspace.
    pub(crate) fn compute(&mut self, order: usize, alpha: f64, pc: [f64; 3]) -> HermiteView<'_> {
        let stride = order + 1;
        let size = stride * stride * stride;
        let distance_squared = pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2];
        self.boys.resize(order + 1, 0.0);
        fill_boys(alpha * distance_squared, &mut self.boys[..=order]);
        self.previous.resize(size, 0.0);
        self.current.resize(size, 0.0);
        let index = |t: usize, u: usize, v: usize| (t * stride + u) * stride + v;

        // `current` holds R^{n}_{tuv} for t + u + v ≤ order - n, built from
        // `previous` = R^{n+1}, going from n = order down to 0. Entries of
        // higher total order left over from earlier calls are never read.
        let mut power = (-2.0 * alpha).powi(i32::try_from(order).expect("order fits in i32"));
        for n in (0..=order).rev() {
            self.current[index(0, 0, 0)] = power * self.boys[n];
            for s in 1..=order - n {
                for t in 0..=s {
                    for u in 0..=s - t {
                        let v = s - t - u;
                        let previous = &self.previous;
                        self.current[index(t, u, v)] = if t > 0 {
                            pc[0] * previous[index(t - 1, u, v)]
                                + if t > 1 {
                                    to_f64(t - 1) * previous[index(t - 2, u, v)]
                                } else {
                                    0.0
                                }
                        } else if u > 0 {
                            pc[1] * previous[index(t, u - 1, v)]
                                + if u > 1 {
                                    to_f64(u - 1) * previous[index(t, u - 2, v)]
                                } else {
                                    0.0
                                }
                        } else {
                            pc[2] * previous[index(t, u, v - 1)]
                                + if v > 1 {
                                    to_f64(v - 1) * previous[index(t, u, v - 2)]
                                } else {
                                    0.0
                                }
                        };
                    }
                }
            }
            power /= -2.0 * alpha;
            std::mem::swap(&mut self.previous, &mut self.current);
        }
        HermiteView {
            data: &self.previous[..size],
            stride,
        }
    }

    /// `R^0_{tuv}` of `kernel` for the reduced exponent `alpha` and the
    /// displacement `pc`: the table at `alpha` for the full kernel,
    /// `sqrt(α'/α)` times the table at the attenuated exponent `α'` for the
    /// long-range part (ADR-0008), and the difference of the two for the
    /// short-range part.
    pub(crate) fn compute_kernel(
        &mut self,
        order: usize,
        alpha: f64,
        kernel: CoulombKernel,
        pc: [f64; 3],
    ) -> HermiteView<'_> {
        let stride = order + 1;
        let size = stride * stride * stride;
        match kernel {
            CoulombKernel::Full => self.compute(order, alpha, pc),
            CoulombKernel::LongRange(w) => {
                let weak = attenuated(alpha, w);
                let scale = (weak / alpha).sqrt();
                let _ = self.compute(order, weak, pc);
                for value in &mut self.previous[..size] {
                    *value *= scale;
                }
                HermiteView {
                    data: &self.previous[..size],
                    stride,
                }
            }
            CoulombKernel::ShortRange(w) => {
                let weak = attenuated(alpha, w);
                let scale = (weak / alpha).sqrt();
                let _ = self.compute(order, alpha, pc);
                self.difference.clear();
                self.difference.extend_from_slice(&self.previous[..size]);
                let _ = self.compute(order, weak, pc);
                for (target, source) in self.difference.iter_mut().zip(&self.previous[..size]) {
                    *target -= scale * source;
                }
                HermiteView {
                    data: &self.difference[..size],
                    stride,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_tables_split_the_full_one() {
        let mut workspace = HermiteWorkspace::default();
        let (order, alpha, pc) = (2, 0.8, [0.3, -0.5, 0.2]);
        let full: Vec<f64> = (0..27)
            .map(|i| {
                let view = workspace.compute_kernel(order, alpha, CoulombKernel::Full, pc);
                view.get(i / 9, (i / 3) % 3, i % 3)
            })
            .collect();
        for (i, total) in full.iter().enumerate() {
            let (t, u, v) = (i / 9, (i / 3) % 3, i % 3);
            if t + u + v > order {
                continue;
            }
            let long = workspace
                .compute_kernel(order, alpha, CoulombKernel::LongRange(0.6), pc)
                .get(t, u, v);
            let short = workspace
                .compute_kernel(order, alpha, CoulombKernel::ShortRange(0.6), pc)
                .get(t, u, v);
            assert!(
                (long + short - total).abs() < 1e-14,
                "{long} + {short} vs {total}"
            );
        }
        assert!(CoulombKernel::new(Some(0.0), false).is_err());
        assert!(CoulombKernel::new(Some(f64::NAN), true).is_err());
        assert!(CoulombKernel::new(None, true).is_err());
        assert_eq!(
            CoulombKernel::new(None, false).expect("valid"),
            CoulombKernel::Full
        );
        assert_eq!(
            CoulombKernel::new(Some(0.5), true).expect("valid"),
            CoulombKernel::ShortRange(0.5)
        );
        assert!(CoulombKernel::ShortRange(-1.0).validate().is_err());
    }

    #[test]
    fn zeroth_order_is_the_boys_function() {
        let mut workspace = HermiteWorkspace::default();
        let r = workspace.compute(0, 1.5, [0.3, -0.2, 0.1]);
        let t = 1.5 * (0.09 + 0.04 + 0.01);
        let expected = crate::special::boys_values(0, t)[0];
        assert!((r.get(0, 0, 0) - expected).abs() < 1e-15);
    }

    #[test]
    fn first_derivatives_match_finite_differences() {
        // R_{100} = ∂R_{000}/∂P_x, checked against a central difference.
        let alpha = 0.8;
        let pc = [0.4, 0.1, -0.3];
        let h = 1e-5;
        let mut workspace = HermiteWorkspace::default();
        let r100 = workspace.compute(1, alpha, pc).get(1, 0, 0);
        let plus = workspace
            .compute(0, alpha, [pc[0] + h, pc[1], pc[2]])
            .get(0, 0, 0);
        let minus = workspace
            .compute(0, alpha, [pc[0] - h, pc[1], pc[2]])
            .get(0, 0, 0);
        assert!((r100 - (plus - minus) / (2.0 * h)).abs() < 1e-8);
    }

    #[test]
    fn reuse_after_a_higher_order_gives_the_same_values() {
        let mut workspace = HermiteWorkspace::default();
        let fresh = workspace.compute(2, 2.0, [0.5, 0.0, 0.0]).get(2, 0, 0);
        let _ = workspace.compute(6, 0.3, [0.1, 0.2, 0.3]);
        let again = workspace.compute(2, 2.0, [0.5, 0.0, 0.0]).get(2, 0, 0);
        assert_eq!(fresh, again);
    }

    #[test]
    fn point_charge_at_the_centre_gives_the_sphere_average() {
        // At R_PC = 0 every F_n is 1/(2n+1); R_{200} = -2α F_1 = -2α/3.
        let alpha = 2.0;
        let mut workspace = HermiteWorkspace::default();
        let r = workspace.compute(2, alpha, [0.0; 3]);
        assert!((r.get(2, 0, 0) + 2.0 * alpha / 3.0).abs() < 1e-15);
        assert!((r.get(1, 1, 0)).abs() < 1e-15);
        let mut table = HermiteTable::zeros(2);
        table.add_scaled(r, -3.0);
        assert!((table.get(2, 0, 0) - 2.0 * alpha).abs() < 1e-15);
    }
}

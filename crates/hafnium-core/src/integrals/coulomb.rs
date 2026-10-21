//! Hermite Coulomb integrals `R_{tuv}(α, R_PC)`.
//!
//! `R^0_{tuv}` is the Coulomb interaction of the Hermite Gaussian
//! `Λ_{tuv}(r; α, P)` with a unit point charge at `C`, up to the factor
//! `2π/α` applied by the caller (Helgaker 9.9.9). The auxiliary integrals
//! `R^n_{000} = (-2α)^n F_n(α R_PC²)` feed the recursions 9.9.18 to 9.9.20.
//! A [`HermiteWorkspace`] owns the buffers so that the hot loops of the
//! kernels allocate nothing.

use crate::special::fill_boys;
use crate::to_f64;

/// Buffers for the Hermite Coulomb recursion, reused across calls.
#[derive(Debug, Default)]
pub(crate) struct HermiteWorkspace {
    boys: Vec<f64>,
    previous: Vec<f64>,
    current: Vec<f64>,
    /// Scratch for the kernels that contract the table with expansion
    /// coefficients before contracting with the other side.
    pub(crate) intermediate: Vec<f64>,
}

/// A read-only view of `R^0_{tuv}` for `t + u + v ≤ order`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HermiteView<'a> {
    data: &'a [f64],
    stride: usize,
}

impl HermiteView<'_> {
    /// `R^0_{tuv}`.
    #[inline]
    pub(crate) fn get(&self, t: usize, u: usize, v: usize) -> f64 {
        self.data[(t * self.stride + u) * self.stride + v]
    }
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

    /// `R^0_{tuv}`.
    #[inline]
    pub(crate) fn get(&self, t: usize, u: usize, v: usize) -> f64 {
        self.data[(t * self.stride + u) * self.stride + v]
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
}

#[cfg(test)]
mod tests {
    use super::*;

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

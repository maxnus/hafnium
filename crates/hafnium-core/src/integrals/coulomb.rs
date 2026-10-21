//! Hermite Coulomb integrals `R_{tuv}(α, R_PC)`.
//!
//! `R^0_{tuv}` is the Coulomb interaction of the Hermite Gaussian
//! `Λ_{tuv}(r; α, P)` with a unit point charge at `C`, up to the factor
//! `2π/α` applied by the caller (Helgaker 9.9.9). The auxiliary integrals
//! `R^n_{000} = (-2α)^n F_n(α R_PC²)` feed the recursions 9.9.18 to 9.9.20.

use crate::special::fill_boys;
use crate::to_f64;

/// `R^0_{tuv}` for all `t + u + v ≤ order`, indexed `[t][u][v]`.
#[derive(Debug, Clone)]
pub(crate) struct HermiteIntegrals {
    data: Vec<f64>,
    stride: usize,
}

impl HermiteIntegrals {
    /// Compute the table for exponent `alpha` and displacement `pc = P - C`.
    pub(crate) fn new(order: usize, alpha: f64, pc: [f64; 3]) -> Self {
        let stride = order + 1;
        let size = stride * stride * stride;
        let distance_squared = pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2];
        let mut boys = vec![0.0; order + 1];
        fill_boys(alpha * distance_squared, &mut boys);
        let index = |t: usize, u: usize, v: usize| (t * stride + u) * stride + v;

        // `current` holds R^{n}_{tuv} for t + u + v ≤ order - n, built from
        // `previous` = R^{n+1} going from n = order down to 0.
        let mut previous = vec![0.0; size];
        let mut current = vec![0.0; size];
        let mut power = (-2.0 * alpha).powi(i32::try_from(order).expect("order fits in i32"));
        for n in (0..=order).rev() {
            current[index(0, 0, 0)] = power * boys[n];
            for s in 1..=order - n {
                for t in 0..=s {
                    for u in 0..=s - t {
                        let v = s - t - u;
                        current[index(t, u, v)] = if t > 0 {
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
            std::mem::swap(&mut previous, &mut current);
        }
        Self {
            data: previous,
            stride,
        }
    }

    /// `R^0_{tuv}`.
    #[inline]
    pub(crate) fn get(&self, t: usize, u: usize, v: usize) -> f64 {
        self.data[(t * self.stride + u) * self.stride + v]
    }

    /// Add `scale` times another table of the same order into this one.
    pub(crate) fn add_scaled(&mut self, other: &Self, scale: f64) {
        for (target, source) in self.data.iter_mut().zip(&other.data) {
            *target += scale * source;
        }
    }

    /// A zero table of the given order.
    pub(crate) fn zeros(order: usize) -> Self {
        let stride = order + 1;
        Self {
            data: vec![0.0; stride * stride * stride],
            stride,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroth_order_is_the_boys_function() {
        let r = HermiteIntegrals::new(0, 1.5, [0.3, -0.2, 0.1]);
        let t = 1.5 * (0.09 + 0.04 + 0.01);
        let expected = crate::special::boys_values(0, t)[0];
        assert!((r.get(0, 0, 0) - expected).abs() < 1e-15);
    }

    #[test]
    fn first_derivatives_match_finite_differences() {
        // R_{100} = ∂R_{000}/∂P_x, checked against a central difference.
        let alpha = 0.8;
        let pc = [0.4, 0.1, -0.3];
        let r = HermiteIntegrals::new(1, alpha, pc);
        let h = 1e-5;
        let plus = HermiteIntegrals::new(0, alpha, [pc[0] + h, pc[1], pc[2]]).get(0, 0, 0);
        let minus = HermiteIntegrals::new(0, alpha, [pc[0] - h, pc[1], pc[2]]).get(0, 0, 0);
        assert!((r.get(1, 0, 0) - (plus - minus) / (2.0 * h)).abs() < 1e-8);
    }

    #[test]
    fn point_charge_at_the_centre_gives_the_sphere_average() {
        // At R_PC = 0 every F_n is 1/(2n+1); R_{200} = -2α F_1 = -2α/3.
        let alpha = 2.0;
        let r = HermiteIntegrals::new(2, alpha, [0.0; 3]);
        assert!((r.get(2, 0, 0) + 2.0 * alpha / 3.0).abs() < 1e-15);
        assert!((r.get(1, 1, 0)).abs() < 1e-15);
    }
}

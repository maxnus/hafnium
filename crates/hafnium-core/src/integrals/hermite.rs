//! Hermite expansion coefficients of a product of two Gaussians.
//!
//! Along one axis, the product of `x_A^i exp(-a x_A²)` and
//! `x_B^j exp(-b x_B²)` is `Σ_t E^{ij}_t Λ_t(x_P; p)` with Hermite
//! Gaussians `Λ_t` of exponent `p = a + b` at `P = (aA + bB)/p`
//! (Helgaker, Jørgensen and Olsen, section 9.5).

/// `E^{ij}_t` for `i ≤ i_max`, `j ≤ j_max` and `t ≤ i + j`, one axis.
#[derive(Debug, Clone)]
pub(crate) struct ExpansionCoefficients {
    data: Vec<f64>,
    n_j: usize,
    n_t: usize,
}

impl ExpansionCoefficients {
    /// Build the coefficients by the two-term recursions of Helgaker 9.5.6
    /// and 9.5.7, starting from `E^{00}_0 = exp(-μ X_AB²)` with the reduced
    /// exponent `μ = ab / p`.
    ///
    /// `pa` and `pb` are `P - A` and `P - B` along the axis, `ab` is
    /// `A - B`.
    pub(crate) fn new(
        i_max: usize,
        j_max: usize,
        p: f64,
        mu: f64,
        ab: f64,
        pa: f64,
        pb: f64,
    ) -> Self {
        let n_j = j_max + 1;
        let n_t = i_max + j_max + 1;
        let mut coefficients = Self {
            data: vec![0.0; (i_max + 1) * n_j * n_t],
            n_j,
            n_t,
        };
        let one_over_2p = 0.5 / p;
        coefficients.set(0, 0, 0, (-mu * ab * ab).exp());
        for i in 0..i_max {
            for t in 0..=i + 1 {
                let value = one_over_2p * coefficients.get_or_zero(i, 0, t.wrapping_sub(1))
                    + pa * coefficients.get(i, 0, t)
                    + crate::to_f64(t + 1) * coefficients.get(i, 0, t + 1);
                coefficients.set(i + 1, 0, t, value);
            }
        }
        for i in 0..=i_max {
            for j in 0..j_max {
                for t in 0..=i + j + 1 {
                    let value = one_over_2p * coefficients.get_or_zero(i, j, t.wrapping_sub(1))
                        + pb * coefficients.get(i, j, t)
                        + crate::to_f64(t + 1) * coefficients.get(i, j, t + 1);
                    coefficients.set(i, j + 1, t, value);
                }
            }
        }
        coefficients
    }

    /// `E^{ij}_t`, zero when `t > i + j`.
    #[inline]
    pub(crate) fn get(&self, i: usize, j: usize, t: usize) -> f64 {
        if t > i + j {
            0.0
        } else {
            self.data[self.index(i, j, t)]
        }
    }

    /// `E^{ij}_t` where `t` may have wrapped below zero, which counts as zero.
    #[inline]
    fn get_or_zero(&self, i: usize, j: usize, t: usize) -> f64 {
        if t > i + j { 0.0 } else { self.get(i, j, t) }
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, t: usize, value: f64) {
        let index = self.index(i, j, t);
        self.data[index] = value;
    }

    #[inline]
    fn index(&self, i: usize, j: usize, t: usize) -> usize {
        (i * self.n_j + j) * self.n_t + t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s_type_coefficient_is_the_prefactor() {
        let e = ExpansionCoefficients::new(0, 0, 2.0, 0.5, 1.0, 0.5, -0.5);
        assert!((e.get(0, 0, 0) - (-0.5f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn p_type_coefficients_follow_the_recursion() {
        // E^{10}_0 = X_PA E^{00}_0 and E^{10}_1 = E^{00}_0 / (2p).
        let (p, mu, ab, pa, pb) = (3.0, 0.4, 0.7, 0.2, -0.5);
        let e = ExpansionCoefficients::new(1, 1, p, mu, ab, pa, pb);
        let e00 = (-mu * ab * ab).exp();
        assert!((e.get(1, 0, 0) - pa * e00).abs() < 1e-15);
        assert!((e.get(1, 0, 1) - e00 / (2.0 * p)).abs() < 1e-15);
        assert!((e.get(0, 1, 0) - pb * e00).abs() < 1e-15);
        assert_eq!(e.get(0, 1, 2), 0.0);
    }
}

//! Cartesian Gaussian components and their transformation to real solid
//! harmonics.
//!
//! The transformation follows Helgaker, Jørgensen and Olsen equation 6.4.47
//! (Schlegel and Frisch 1995) for Racah-normalized real solid harmonics
//! `S_lm`, multiplied by `sqrt((2l + 1) / 4π)` so that the result is
//! normalized on the unit sphere (ADR-0006).

use std::f64::consts::PI;
use std::ops::{AddAssign, Mul};

use crate::to_f64;

/// Number of Cartesian components `x^i y^j z^k` with `i + j + k = l`.
#[must_use]
pub fn n_cartesian(l: usize) -> usize {
    (l + 1) * (l + 2) / 2
}

/// The exponents `(i, j, k)` of the Cartesian components of angular
/// momentum `l`, with `x^l` first and `z^l` last.
#[must_use]
pub fn cartesian_components(l: usize) -> Vec<[usize; 3]> {
    let mut components = Vec::with_capacity(n_cartesian(l));
    for i in (0..=l).rev() {
        for j in (0..=l - i).rev() {
            components.push([i, j, l - i - j]);
        }
    }
    components
}

/// Row-major `[2l + 1, n_cartesian(l)]` matrix taking Cartesian monomial
/// integrals to real solid harmonics normalized on the sphere.
///
/// Rows are ordered `m = -l, ..., +l`, except that a p shell is ordered
/// `x, y, z` (`m = +1, -1, 0`) as in libcint, `PySCF` and Molden.
///
/// # Panics
///
/// Panics if `l` exceeds [`super::MAX_ANGULAR_MOMENTUM`]; the shell table
/// rejects such shells before any kernel runs.
#[must_use]
pub fn spherical_transformation(l: usize) -> Vec<f64> {
    assert!(
        l <= super::MAX_ANGULAR_MOMENTUM,
        "angular momentum too high"
    );
    let components = cartesian_components(l);
    let n_cart = components.len();
    let mut matrix = vec![0.0; (2 * l + 1) * n_cart];
    let sphere = (to_f64(2 * l + 1) / (4.0 * PI)).sqrt();
    for (row, m) in rows(l).into_iter().enumerate() {
        for (exponents, coefficient) in solid_harmonic_terms(l, m) {
            let column = components
                .iter()
                .position(|c| *c == exponents)
                .expect("monomial degree equals l");
            matrix[row * n_cart + column] += sphere * coefficient;
        }
    }
    matrix
}

/// Transform one index of a row-major array with layout
/// `[left, n_cart, right]` to spherical functions:
/// `out[l][m][r] = Σ_c transform[m][c] data[l][c][r]`, where `transform`
/// is row-major `[n_sph, n_cart]`. The values may be real or complex.
#[must_use]
pub fn transform_index<T>(
    data: &[T],
    left: usize,
    n_cart: usize,
    right: usize,
    transform: &[f64],
) -> Vec<T>
where
    T: Copy + Default + AddAssign + Mul<f64, Output = T>,
{
    let mut out = Vec::new();
    transform_index_into(data, left, n_cart, right, transform, &mut out);
    out
}

/// [`transform_index`] into `out`, cleared and resized to the result.
pub(crate) fn transform_index_into<T>(
    data: &[T],
    left: usize,
    n_cart: usize,
    right: usize,
    transform: &[f64],
    out: &mut Vec<T>,
) where
    T: Copy + Default + AddAssign + Mul<f64, Output = T>,
{
    let n_sph = transform.len() / n_cart;
    out.clear();
    out.resize(left * n_sph * right, T::default());
    for l in 0..left {
        for (m, row) in transform.chunks(n_cart).enumerate() {
            let target = &mut out[(l * n_sph + m) * right..(l * n_sph + m + 1) * right];
            for (c, &weight) in row.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                let source = &data[(l * n_cart + c) * right..(l * n_cart + c + 1) * right];
                for (t, s) in target.iter_mut().zip(source) {
                    *t += *s * weight;
                }
            }
        }
    }
}

/// The `m` value of each row of the transformation.
fn rows(l: usize) -> Vec<i64> {
    if l == 1 {
        vec![1, -1, 0]
    } else {
        let l = i64::try_from(l).expect("angular momentum fits in i64");
        (-l..=l).collect()
    }
}

/// Terms `(exponents of x, y, z, coefficient)` of the Racah-normalized
/// real solid harmonic `S_lm` (Helgaker 6.4.47).
#[allow(clippy::many_single_char_names, reason = "textbook formula")]
fn solid_harmonic_terms(l: usize, m: i64) -> Vec<([usize; 3], f64)> {
    let m_abs = usize::try_from(m.abs()).expect("non-negative");
    // For m < 0 the inner index v is half-integer: v = v_half + 1/2.
    let negative = m < 0;
    let norm = {
        let delta = if m == 0 { 2.0 } else { 1.0 };
        let value = 2.0 * factorial(l + m_abs) * factorial(l - m_abs) / delta;
        value.sqrt() / (2f64.powi(i32::try_from(m_abs).expect("small")) * factorial(l))
    };
    let mut terms = Vec::new();
    for t in 0..=(l - m_abs) / 2 {
        for u in 0..=t {
            let v_upper = if negative { (m_abs - 1) / 2 } else { m_abs / 2 };
            for v_half in 0..=v_upper {
                let two_v = if negative { 2 * v_half + 1 } else { 2 * v_half };
                if two_v > m_abs {
                    continue;
                }
                let sign = if (t + v_half) % 2 == 0 { 1.0 } else { -1.0 };
                let coefficient = sign
                    * 0.25f64.powi(i32::try_from(t).expect("small"))
                    * binomial(l, t)
                    * binomial(l - t, m_abs + t)
                    * binomial(t, u)
                    * binomial(m_abs, two_v);
                let y_power = 2 * u + two_v;
                let x_power = 2 * t + m_abs - y_power;
                let z_power = l - 2 * t - m_abs;
                terms.push(([x_power, y_power, z_power], norm * coefficient));
            }
        }
    }
    terms
}

/// `n!` as an exact integer converted to `f64`; exact for `n ≤ 22` and
/// correctly rounded up to `2 * MAX_ANGULAR_MOMENTUM`.
#[allow(
    clippy::cast_precision_loss,
    reason = "the conversion from the exact integer is correctly rounded"
)]
fn factorial(n: usize) -> f64 {
    let mut value: u128 = 1;
    for k in 2..=n as u128 {
        value *= k;
    }
    value as f64
}

/// Binomial coefficient `n` over `k` as `f64`, zero when `k > n`.
#[allow(
    clippy::cast_precision_loss,
    reason = "the conversion from the exact integer is correctly rounded"
)]
fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut value: u128 = 1;
    for i in 0..k as u128 {
        value = value * (n as u128 - i) / (i + 1);
    }
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_order_is_x_major() {
        assert_eq!(
            cartesian_components(1),
            vec![[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        );
        assert_eq!(
            cartesian_components(2),
            vec![
                [2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2]
            ]
        );
        assert_eq!(n_cartesian(4), 15);
    }

    #[test]
    fn d_shell_matches_the_textbook_harmonics() {
        // S_{2,-2} = √3 xy, S_{2,0} = (3z² − r²)/2, S_{2,2} = (√3/2)(x² − y²),
        // each times sqrt(5 / 4π).
        let sphere = (5.0 / (4.0 * PI)).sqrt();
        let matrix = spherical_transformation(2);
        let at = |row: usize, column: usize| matrix[row * 6 + column];
        assert!((at(0, 1) - sphere * 3f64.sqrt()).abs() < 1e-14);
        assert!((at(2, 0) + sphere * 0.5).abs() < 1e-14);
        assert!((at(2, 5) - sphere).abs() < 1e-14);
        assert!((at(4, 0) - sphere * 0.5 * 3f64.sqrt()).abs() < 1e-14);
        assert!((at(4, 3) + sphere * 0.5 * 3f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn p_shell_is_ordered_x_y_z() {
        let sphere = (3.0 / (4.0 * PI)).sqrt();
        let matrix = spherical_transformation(1);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { sphere } else { 0.0 };
                assert!((matrix[i * 3 + j] - expected).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn transform_index_applies_to_the_middle_axis() {
        // data [2, 2, 1] with the transform [[1, 2], [0, 3]]: out[l][m] = Σ_c T[m][c] d[l][c].
        let data = [1.0, 10.0, 2.0, 20.0];
        let out = transform_index(&data, 2, 2, 1, &[1.0, 2.0, 0.0, 3.0]);
        assert_eq!(out, vec![21.0, 30.0, 42.0, 60.0]);
    }

    #[test]
    fn exact_small_factorials_and_binomials() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(binomial(6, 2), 15.0);
        assert_eq!(binomial(3, 5), 0.0);
    }
}

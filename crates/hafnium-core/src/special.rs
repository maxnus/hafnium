//! Special functions.
//!
//! The Boys function `F_n(T) = ∫_0^1 t^{2n} exp(-T t²) dt` is the only one
//! the McMurchie–Davidson integrals need (ADR-0008). The Ewald sums of
//! reduced dimension (ADR-0012) add the scaled complementary error
//! function, the exponential integral and an incomplete Bessel integral.

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::to_f64;

/// The Euler–Mascheroni constant γ.
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Below this argument the series expansion is used; above it, upward
/// recursion from the closed form of `F_0`. The series is also used for
/// larger arguments when the highest order is more than half the argument,
/// where the upward recursion would lose accuracy to cancellation.
const SERIES_LIMIT: f64 = 35.0;
/// Relative size of a series term at which the summation stops.
const SERIES_TOLERANCE: f64 = 1e-17;

/// Fill `out[n]` with `F_n(t)` for every `n` below `out.len()`.
///
/// Both branches are accurate to about `1e-14` relative; the test suite
/// checks this against an arbitrary-precision reference. An empty `out`
/// is left untouched.
///
/// # Panics
///
/// Panics if `t` is negative. Callers validate the argument before crossing
/// the FFI boundary.
pub fn fill_boys(t: f64, out: &mut [f64]) {
    let Some(n_max) = out.len().checked_sub(1) else {
        return;
    };
    assert!(t >= 0.0, "the Boys function argument is non-negative");
    let exp_t = (-t).exp();
    if t < SERIES_LIMIT.max(2.0 * to_f64(n_max)) {
        out[n_max] = series(n_max, t);
        for n in (1..=n_max).rev() {
            out[n - 1] = (2.0 * t * out[n] + exp_t) / to_f64(2 * n - 1);
        }
    } else {
        out[0] = 0.5 * (PI / t).sqrt() * libm::erf(t.sqrt());
        for n in 0..n_max {
            out[n + 1] = (to_f64(2 * n + 1) * out[n] - exp_t) / (2.0 * t);
        }
    }
}

/// `F_n(t)` for `n = 0, ..., n_max` as a new vector.
///
/// # Panics
///
/// Panics if `t` is negative, see [`fill_boys`].
#[must_use]
pub fn boys_values(n_max: usize, t: f64) -> Vec<f64> {
    let mut out = vec![0.0; n_max + 1];
    fill_boys(t, &mut out);
    out
}

/// `F_n(t)` by the series `exp(-t) Σ_i (2t)^i / ((2n+1)(2n+3)...(2n+2i+1))`,
/// whose terms are all positive, so the sum has no cancellation.
fn series(n: usize, t: f64) -> f64 {
    let two_t = 2.0 * t;
    let mut term = 1.0 / to_f64(2 * n + 1);
    let mut sum = term;
    let mut i = 1;
    loop {
        term *= two_t / to_f64(2 * n + 2 * i + 1);
        sum += term;
        if term < sum * SERIES_TOLERANCE {
            break;
        }
        i += 1;
    }
    (-t).exp() * sum
}

/// Above this argument `erfcx` uses its asymptotic series; below it the
/// direct product `exp(x²) erfc(x)` is exact to rounding, since `erfc`
/// keeps full relative accuracy down to its underflow threshold.
const ERFCX_SERIES_LIMIT: f64 = 25.0;
/// Size of a term of the asymptotic series of `erfcx`, relative to its
/// leading term, at which the summation stops.
const ERFCX_TOLERANCE: f64 = 1e-18;
/// Highest number of terms of that series; at the limit argument the terms
/// reach the tolerance after seven.
const ERFCX_MAX_TERMS: usize = 40;
/// Relative size of a term at which the series of `E_1` stops.
const E1_TOLERANCE: f64 = 1e-17;
/// Highest number of terms or iterations in the expansions of `E_1`.
const E1_MAX_TERMS: usize = 200;
/// Nodes per panel of the composite Gauss–Legendre rule.
const QUADRATURE_NODES: usize = 20;
/// Width of a quadrature panel; the integrands vary on the scale one.
const QUADRATURE_PANEL: f64 = 1.0;
/// Decades below the maximum at which a quadrature integrand is cut off.
const QUADRATURE_DECAY: f64 = 40.0;

/// Scaled complementary error function `exp(x²) erfc(x)` for `x ≥ 0`.
///
/// Stays finite where `exp(x²)` alone would overflow.
///
/// # Panics
///
/// Panics if `x` is negative; callers use the symmetry of their kernels to
/// keep the argument non-negative.
#[must_use]
pub fn erfcx(x: f64) -> f64 {
    assert!(x >= 0.0, "erfcx takes a non-negative argument");
    if x <= ERFCX_SERIES_LIMIT {
        return (x * x).exp() * libm::erfc(x);
    }
    // erfcx(x) ~ (1 / x√π) Σ_n (-1)^n (2n - 1)!! / (2x²)^n; beyond the limit
    // the terms fall below rounding long before the series turns.
    let inverse = 1.0 / (2.0 * x * x);
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 1..ERFCX_MAX_TERMS {
        term *= -to_f64(2 * n - 1) * inverse;
        sum += term;
        if term.abs() < ERFCX_TOLERANCE {
            break;
        }
    }
    sum / (x * PI.sqrt())
}

/// The exponential integral `E_1(x) = ∫_x^∞ exp(-t) / t dt` for `x > 0`.
///
/// # Panics
///
/// Panics if `x` is not positive.
#[must_use]
pub fn exponential_integral_1(x: f64) -> f64 {
    assert!(x > 0.0, "E_1 takes a positive argument");
    if x <= 1.0 {
        // E_1(x) = -γ - ln x - Σ_{n≥1} (-x)^n / (n n!)
        let mut term = 1.0;
        let mut sum = 0.0;
        for n in 1..=E1_MAX_TERMS {
            term *= -x / to_f64(n);
            let addend = term / to_f64(n);
            sum += addend;
            if addend.abs() < E1_TOLERANCE * sum.abs() {
                break;
            }
        }
        return -EULER_GAMMA - x.ln() - sum;
    }
    // Modified Lentz evaluation of the continued fraction
    // E_1(x) = exp(-x) / (x + 1 - 1 / (x + 3 - 4 / (x + 5 - ...))).
    let mut denominator = x + 1.0;
    let mut forward = 1.0 / f64::MIN_POSITIVE;
    let mut backward = 1.0 / denominator;
    let mut fraction = backward;
    for i in 1..=E1_MAX_TERMS {
        let numerator = -to_f64(i * i);
        denominator += 2.0;
        backward = 1.0 / (numerator * backward + denominator);
        forward = denominator + numerator / forward;
        let delta = forward * backward;
        fraction *= delta;
        if (delta - 1.0).abs() < E1_TOLERANCE {
            break;
        }
    }
    fraction * (-x).exp()
}

/// `∫_0^∞ exp(-a e^t - b e^{-t}) dt` for `a > 0` and `b ≥ 0`.
///
/// This is `∫_{1/4η²}^∞ exp(-G² u - ρ² / 4u) du / u` with `a = G² / 4η²`
/// and `b = η² ρ²`, the Fourier transform along a chain of the long-range
/// Ewald kernel at a perpendicular distance `ρ` (ADR-0012). At `b = 0` it
/// is `E_1(a)`.
///
/// # Panics
///
/// Panics unless `a` is positive and `b` non-negative.
#[must_use]
pub fn incomplete_bessel_integral(a: f64, b: f64) -> f64 {
    assert!(a > 0.0, "the incomplete Bessel integral takes a > 0");
    assert!(b >= 0.0, "the incomplete Bessel integral takes b >= 0");
    if b == 0.0 {
        return exponential_integral_1(a);
    }
    chain_quadrature(a, b)
}

/// Composite Gauss–Legendre quadrature of `exp(-a e^t - b e^{-t})` over
/// `t ≥ 0`, cut off where the integrand has fallen `QUADRATURE_DECAY`
/// decades below its maximum.
fn chain_quadrature(a: f64, b: f64) -> f64 {
    let log_maximum = if b > a {
        -2.0 * (a * b).sqrt()
    } else {
        -(a + b)
    };
    let upper = ((QUADRATURE_DECAY * std::f64::consts::LN_10 - log_maximum) / a).ln();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the panel count is a small positive number"
    )]
    let n_panels = (upper / QUADRATURE_PANEL).ceil().max(1.0) as usize;
    let width = upper / to_f64(n_panels);
    let half_width = 0.5 * width;
    let rule = gauss_legendre();
    let mut total = 0.0;
    for panel in 0..n_panels {
        let start = to_f64(panel) * width;
        for &(node, weight) in rule {
            let t = start + (node + 1.0) * half_width;
            total += weight * (-a * t.exp() - b * (-t).exp()).exp();
        }
    }
    half_width * total
}

/// Nodes and weights of the Gauss–Legendre rule on `[-1, 1]`, built once.
fn gauss_legendre() -> &'static [(f64, f64)] {
    static RULE: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    RULE.get_or_init(|| gauss_legendre_rule(QUADRATURE_NODES))
}

/// Gauss–Legendre nodes and weights by Newton iteration on `P_n`.
fn gauss_legendre_rule(n: usize) -> Vec<(f64, f64)> {
    let mut rule = vec![(0.0, 0.0); n];
    for i in 0..n.div_ceil(2) {
        let mut z = (PI * (to_f64(i) + 0.75) / (to_f64(n) + 0.5)).cos();
        loop {
            let (value, derivative) = legendre(n, z);
            let step = value / derivative;
            z -= step;
            if step.abs() < 1e-15 {
                break;
            }
        }
        let (_, derivative) = legendre(n, z);
        let weight = 2.0 / ((1.0 - z * z) * derivative * derivative);
        rule[i] = (-z, weight);
        rule[n - 1 - i] = (z, weight);
    }
    rule
}

/// `P_n(z)` and its derivative by the three-term recurrence.
fn legendre(n: usize, z: f64) -> (f64, f64) {
    let (mut previous, mut current) = (0.0, 1.0);
    for j in 1..=n {
        let next = (to_f64(2 * j - 1) * z * current - to_f64(j - 1) * previous) / to_f64(j);
        previous = current;
        current = next;
    }
    let derivative = to_f64(n) * (z * current - previous) / (z * z - 1.0);
    (current, derivative)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, relative: f64) {
        let scale = expected.abs().max(f64::MIN_POSITIVE);
        assert!(
            (actual - expected).abs() <= relative * scale,
            "{actual} differs from {expected} by more than {relative} relative"
        );
    }

    #[test]
    fn zero_argument_is_one_over_odd_numbers() {
        let values = boys_values(6, 0.0);
        for (n, value) in values.iter().enumerate() {
            assert_close(*value, 1.0 / to_f64(2 * n + 1), 1e-15);
        }
    }

    #[test]
    fn order_zero_matches_the_error_function() {
        for t in [1e-10, 0.3, 1.0, 7.5, 20.0, 34.9, 35.1, 120.0] {
            let expected = 0.5 * (PI / t).sqrt() * libm::erf(t.sqrt());
            assert_close(boys_values(0, t)[0], expected, 1e-14);
        }
    }

    #[test]
    fn both_branches_agree_at_the_switch() {
        // At t = 35 a request up to order 12 takes the upward recursion,
        // while a request up to order 18 takes the series; the shared
        // orders must coincide.
        let upward = boys_values(12, 35.0);
        let series = boys_values(18, 35.0);
        for (lower, upper) in upward.iter().zip(&series) {
            assert_close(*lower, *upper, 1e-13);
        }
    }

    #[test]
    fn high_order_uses_the_series_above_the_limit() {
        // n_max = 30 at t = 40 falls back to the series; the result must
        // still satisfy the upward recursion exactly.
        let values = boys_values(30, 40.0);
        let exp_t = (-40.0f64).exp();
        for n in 0..30 {
            let recurred = (to_f64(2 * n + 1) * values[n] - exp_t) / 80.0;
            assert_close(values[n + 1], recurred, 1e-10);
        }
    }

    #[test]
    fn empty_output_is_left_alone() {
        let mut out: [f64; 0] = [];
        fill_boys(1.0, &mut out);
    }

    #[test]
    #[should_panic(expected = "non-negative")]
    fn negative_argument_panics() {
        let _ = boys_values(0, -1.0);
    }

    #[test]
    fn erfcx_matches_the_direct_product_and_is_continuous() {
        for x in [0.0, 0.3, 1.0, 4.0, 10.0, 24.9] {
            assert_close(erfcx(x), (x * x).exp() * libm::erfc(x), 1e-14);
        }
        // The series branch starts just above the limit; both branches agree.
        let above = f64::from_bits(ERFCX_SERIES_LIMIT.to_bits() + 1);
        assert_close(erfcx(above), erfcx(ERFCX_SERIES_LIMIT), 1e-15);
        // erfcx(x) ~ 1 / (x sqrt(pi)) for large x
        assert_close(erfcx(1e4), 1.0 / (1e4 * PI.sqrt()), 1e-8);
    }

    #[test]
    fn exponential_integral_matches_tabulated_values() {
        // Abramowitz and Stegun, table 5.1.
        for (x, expected) in [
            (0.1, 1.822_923_958_419_390_6),
            (0.5, 0.559_773_594_776_16),
            (1.0, 0.219_383_934_395_52),
            (2.0, 0.048_900_510_708_061),
            (5.0, 0.001_148_295_591_275_3),
            (10.0, 4.156_968_929_685_3e-6),
        ] {
            assert_close(exponential_integral_1(x), expected, 1e-13);
        }
        assert_close(
            exponential_integral_1(1.0 + 1e-12),
            exponential_integral_1(1.0 - 1e-12),
            1e-11,
        );
    }

    /// Simpson's rule on the same integrand with many points, as an
    /// independent check of the composite Gauss–Legendre rule.
    fn simpson(a: f64, b: f64) -> f64 {
        let upper = 60.0_f64.min((100.0 / a).ln().max(1.0));
        let n_steps = 400_000;
        let step = upper / to_f64(n_steps);
        let integrand = |t: f64| (-a * t.exp() - b * (-t).exp()).exp();
        let mut sum = integrand(0.0) + integrand(upper);
        for i in 1..n_steps {
            sum += if i % 2 == 1 { 4.0 } else { 2.0 } * integrand(to_f64(i) * step);
        }
        sum * step / 3.0
    }

    #[test]
    fn chain_quadrature_reduces_to_the_exponential_integral() {
        for a in [0.05, 0.5, 3.0, 20.0] {
            assert_close(chain_quadrature(a, 0.0), exponential_integral_1(a), 1e-13);
        }
    }

    #[test]
    fn incomplete_bessel_integral_matches_simpson() {
        for (a, b) in [(0.3, 0.2), (1.0, 1.0), (0.2, 12.0), (4.0, 0.5), (2.0, 30.0)] {
            assert_close(incomplete_bessel_integral(a, b), simpson(a, b), 1e-11);
        }
        assert_close(
            incomplete_bessel_integral(1.5, 0.0),
            exponential_integral_1(1.5),
            1e-15,
        );
    }

    #[test]
    fn gauss_legendre_rule_integrates_polynomials_exactly() {
        let rule = gauss_legendre_rule(QUADRATURE_NODES);
        let total: f64 = rule.iter().map(|(_, w)| w).sum();
        assert_close(total, 2.0, 1e-14);
        // ∫_{-1}^{1} x^{2k} dx = 2 / (2k + 1) up to degree 2n - 1
        for k in 1..QUADRATURE_NODES {
            let moment: f64 = rule.iter().map(|(x, w)| w * x.powf(to_f64(2 * k))).sum();
            assert_close(moment, 2.0 / to_f64(2 * k + 1), 1e-13);
        }
    }
}

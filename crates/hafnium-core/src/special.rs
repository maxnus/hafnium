//! Special functions: the Boys function.
//!
//! `F_n(T) = ∫_0^1 t^{2n} exp(-T t²) dt` is the only special function the
//! McMurchie–Davidson integrals need (ADR-0008).

use std::f64::consts::PI;

use crate::to_f64;

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
}

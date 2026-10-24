//! Partition of space between atomic centres for the quadrature grids.
//!
//! Every point of an atom's grid receives the weight of that atom's
//! normalized cell function, `p_A(r) = P_A(r) / Σ_B P_B(r)` with
//! `P_A(r) = Π_{B ≠ A} s(ν_AB(r))`, from the confocal elliptical coordinate
//! `μ_AB = (r_A − r_B) / R_AB` of the point, adjusted for the atomic sizes
//! as `ν = μ + a_AB (1 − μ²)` when adjustments are given. Two step
//! functions `s` are provided: Becke's thrice-iterated polynomial
//! (J. Chem. Phys. 88, 2547 (1988)), and the compactly supported one of
//! Stratmann, Scuseria and Frisch (Chem. Phys. Lett. 257, 213 (1996)),
//! which is exactly one for `ν ≤ −a` and exactly zero for `ν ≥ a` with
//! `a = 0.64`, so that a far centre leaves the cell functions of the near
//! ones untouched. The Python layer builds the partition of a molecule in
//! array code; this kernel serves the many centres of a periodic partition,
//! an atom's neighbours and their lattice images, where the compact support
//! lets most cell functions be settled by their first factor.

use rayon::prelude::*;

use crate::Error;

/// Points evaluated per parallel task.
const POINTS_PER_TASK: usize = 64;

/// Half-width of the transition of the Stratmann–Scuseria–Frisch step.
const STRATMANN_A: f64 = 0.64;

/// The step function joining the cell functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition {
    /// Becke's `s(ν) = (1 − f(f(f(ν)))) / 2` with `f(x) = (3x − x³) / 2`.
    Becke,
    /// The piecewise polynomial of Stratmann, Scuseria and Frisch, one for
    /// `ν ≤ −0.64` and zero for `ν ≥ 0.64`.
    Stratmann,
}

impl Partition {
    /// The step `s(ν)`, one where the point belongs to the first atom of
    /// the pair and zero where it belongs to the second.
    #[must_use]
    pub fn step(self, nu: f64) -> f64 {
        match self {
            Self::Becke => {
                let mut x = nu;
                for _ in 0..3 {
                    x = 0.5 * x * (3.0 - x * x);
                }
                0.5 * (1.0 - x)
            }
            Self::Stratmann => {
                let x = nu / STRATMANN_A;
                if x <= -1.0 {
                    1.0
                } else if x >= 1.0 {
                    0.0
                } else {
                    let x2 = x * x;
                    let z = x * (35.0 + x2 * (-35.0 + x2 * (21.0 - 5.0 * x2))) / 16.0;
                    0.5 * (1.0 - z)
                }
            }
        }
    }
}

/// Weights `p_owner(r)` of the centre `owner` at `points`.
///
/// `centers` holds every centre that takes part in the partition, the
/// owner among them; `adjustments`, if given, is the row-major
/// `[centre][centre]` table of atomic size adjustments `a_AB`. The cell
/// functions are accumulated from the nearest centre outwards, so that
/// with the Stratmann step a centre far from the point is settled by its
/// first factor.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if `owner` is out of range, a point
/// or a centre is not finite, two centres coincide, or an adjustment is not
/// finite, and [`Error::LengthMismatch`] if the adjustments are not one per
/// pair of centres.
pub fn partition_weights(
    points: &[[f64; 3]],
    centers: &[[f64; 3]],
    owner: usize,
    adjustments: Option<&[f64]>,
    scheme: Partition,
) -> Result<Vec<f64>, Error> {
    let n = centers.len();
    if owner >= n {
        return Err(Error::InvalidArgument(
            "the owner is one of the centres".to_string(),
        ));
    }
    if !points
        .iter()
        .chain(centers)
        .all(|point| point.iter().all(|x| x.is_finite()))
    {
        return Err(Error::InvalidArgument(
            "points and centres are finite".to_string(),
        ));
    }
    if let Some(table) = adjustments {
        if table.len() != n * n {
            return Err(Error::LengthMismatch {
                left: table.len(),
                right: n * n,
            });
        }
        if !table.iter().all(|a| a.is_finite()) {
            return Err(Error::InvalidArgument("adjustments are finite".to_string()));
        }
    }
    let mut inverse_separation = vec![0.0; n * n];
    for a in 0..n {
        for b in 0..a {
            let separation = distance(&centers[a], &centers[b]);
            if separation == 0.0 {
                return Err(Error::InvalidArgument("two centres coincide".to_string()));
            }
            inverse_separation[a * n + b] = 1.0 / separation;
            inverse_separation[b * n + a] = 1.0 / separation;
        }
    }
    let pairs = Pairs {
        n,
        inverse_separation: &inverse_separation,
        adjustments,
        scheme,
    };
    let mut weights = vec![0.0; points.len()];
    weights
        .par_chunks_mut(POINTS_PER_TASK)
        .zip(points.par_chunks(POINTS_PER_TASK))
        .for_each(|(block, chunk)| {
            let mut distances = vec![0.0; n];
            let mut order: Vec<usize> = (0..n).collect();
            for (weight, point) in block.iter_mut().zip(chunk) {
                for (r, center) in distances.iter_mut().zip(centers) {
                    *r = distance(point, center);
                }
                order.sort_unstable_by(|&a, &b| distances[a].total_cmp(&distances[b]));
                *weight = pairs.weight_of(owner, &distances, &order);
            }
        });
    Ok(weights)
}

/// The pair table of a partition: inverse separations and adjustments.
struct Pairs<'a> {
    n: usize,
    inverse_separation: &'a [f64],
    adjustments: Option<&'a [f64]>,
    scheme: Partition,
}

impl Pairs<'_> {
    /// The step `s(ν_ab)` at a point with the distances `r_a` and `r_b`.
    fn step(&self, a: usize, b: usize, r_a: f64, r_b: f64) -> f64 {
        let mut mu = (r_a - r_b) * self.inverse_separation[a * self.n + b];
        if let Some(table) = self.adjustments {
            mu += table[a * self.n + b] * (1.0 - mu * mu);
        }
        self.scheme.step(mu)
    }

    /// The cell function of centre `a`, accumulated over the centres in
    /// `order`; zero as soon as one factor is zero.
    fn cell(&self, a: usize, distances: &[f64], order: &[usize]) -> f64 {
        let mut product = 1.0;
        for &b in order {
            if b == a {
                continue;
            }
            let step = self.step(a, b, distances[a], distances[b]);
            if step == 0.0 {
                return 0.0;
            }
            product *= step;
        }
        product
    }

    /// The normalized weight of `owner` at the point.
    fn weight_of(&self, owner: usize, distances: &[f64], order: &[usize]) -> f64 {
        let own = self.cell(owner, distances, order);
        if own == 0.0 {
            return 0.0;
        }
        let total: f64 = order.iter().map(|&b| self.cell(b, distances, order)).sum();
        own / total
    }
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d: [f64; 3] = std::array::from_fn(|axis| a[axis] - b[axis]);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CENTERS: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [2.0, 0.1, -0.3], [-0.5, 1.8, 0.7]];

    fn sample_points() -> Vec<[f64; 3]> {
        (0..40)
            .map(|i| {
                let t = f64::from(i) * 0.37;
                [t.sin() * 2.5, (1.3 * t).cos() * 1.5, (0.7 * t).sin() * 2.0]
            })
            .collect()
    }

    /// The straightforward `O(n²)` evaluation with no reordering.
    fn reference_weight(
        point: &[f64; 3],
        centers: &[[f64; 3]],
        owner: usize,
        adjustments: Option<&[f64]>,
        scheme: Partition,
    ) -> f64 {
        let n = centers.len();
        let r: Vec<f64> = centers.iter().map(|c| distance(point, c)).collect();
        let cell = |a: usize| -> f64 {
            (0..n)
                .filter(|&b| b != a)
                .map(|b| {
                    let mut mu = (r[a] - r[b]) / distance(&centers[a], &centers[b]);
                    if let Some(table) = adjustments {
                        mu += table[a * n + b] * (1.0 - mu * mu);
                    }
                    scheme.step(mu)
                })
                .product()
        };
        cell(owner) / (0..n).map(cell).sum::<f64>()
    }

    #[test]
    fn steps_have_the_limits_and_the_symmetry() {
        for scheme in [Partition::Becke, Partition::Stratmann] {
            assert!((scheme.step(-1.0) - 1.0).abs() < 1e-15);
            assert!(scheme.step(1.0).abs() < 1e-15);
            assert!((scheme.step(0.0) - 0.5).abs() < 1e-15);
            for &nu in &[0.1, 0.3, 0.55, 0.9] {
                assert!((scheme.step(nu) + scheme.step(-nu) - 1.0).abs() < 1e-15);
            }
        }
        // The Stratmann step is exactly one and zero beyond the transition.
        assert_eq!(Partition::Stratmann.step(-0.64), 1.0);
        assert_eq!(Partition::Stratmann.step(0.7), 0.0);
        assert!(Partition::Becke.step(-0.64) < 1.0);
    }

    #[test]
    fn weights_match_the_reference_and_sum_to_one() {
        let points = sample_points();
        let n = CENTERS.len();
        let mut table = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                table[a * n + b] = 0.3
                    * (f64::from(u8::try_from(a).unwrap()) - f64::from(u8::try_from(b).unwrap()))
                    / 2.0;
            }
        }
        for scheme in [Partition::Becke, Partition::Stratmann] {
            for adjustments in [None, Some(table.as_slice())] {
                let mut sums = vec![0.0; points.len()];
                for owner in 0..n {
                    let weights = partition_weights(&points, &CENTERS, owner, adjustments, scheme)
                        .expect("valid");
                    for (i, point) in points.iter().enumerate() {
                        let expected =
                            reference_weight(point, &CENTERS, owner, adjustments, scheme);
                        assert!((weights[i] - expected).abs() < 1e-14);
                        sums[i] += weights[i];
                    }
                }
                for sum in sums {
                    assert!((sum - 1.0).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn stratmann_weights_are_exact_near_the_nucleus_and_vanish_far_away() {
        // Within (1 − a) / 2 of the distance to the nearest neighbour the
        // owner's weight is exactly one; near a neighbour it is exactly zero.
        let near = [[0.1, 0.05, -0.02], [0.3, 0.0, 0.0]];
        let weights =
            partition_weights(&near, &CENTERS, 0, None, Partition::Stratmann).expect("valid");
        assert_eq!(weights, vec![1.0, 1.0]);
        let far = [[2.0, 0.1, -0.2]];
        let weights =
            partition_weights(&far, &CENTERS, 0, None, Partition::Stratmann).expect("valid");
        assert_eq!(weights, vec![0.0]);
        let weights = partition_weights(&far, &CENTERS, 0, None, Partition::Becke).expect("valid");
        assert!(weights[0] > 0.0 && weights[0] < 1e-6);
    }

    #[test]
    fn malformed_input_is_rejected() {
        let points = [[0.0; 3]];
        assert!(matches!(
            partition_weights(&points, &CENTERS, 3, None, Partition::Becke),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            partition_weights(&points, &CENTERS, 0, Some(&[0.0; 4]), Partition::Becke),
            Err(Error::LengthMismatch { .. })
        ));
        let coinciding = [[0.0; 3], [0.0; 3]];
        assert!(matches!(
            partition_weights(&points, &coinciding, 0, None, Partition::Becke),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            partition_weights(&[[f64::NAN; 3]], &CENTERS, 0, None, Partition::Becke),
            Err(Error::InvalidArgument(_))
        ));
    }
}

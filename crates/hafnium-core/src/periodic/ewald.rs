//! Ewald sums of point charges on a lattice of one to three dimensions.
//!
//! The Coulomb kernel is range-separated with the parameter `ω` of
//! CONVENTIONS.md §3, the Gaussian width being `1/ω`:
//! `1/r = erfc(ωr)/r + erf(ωr)/r`. The short-range part is summed over the
//! real-space images, the long-range part over the reciprocal lattice
//! after a Fourier transform along the periodic directions only, so a
//! slab or a chain needs no vacuum padding. The `G = 0` term of the
//! long-range part diverges for a charged cell; every dimension keeps its
//! finite part after removing the divergence that does not depend on `ω`,
//! which is the uniform-background convention in three dimensions
//! (ADR-0012). Energies are per unit cell.

use std::f64::consts::PI;

use crate::Error;
use crate::special::{EULER_GAMMA, erfcx, exponential_integral_1, incomplete_bessel_integral};

/// Distance below which two charges are taken to coincide.
const COINCIDENCE: f64 = 1e-10;
/// Perpendicular distance below which a chain pair is taken to lie on the
/// axis, where the `G = 0` kernel has its finite limit.
const ON_AXIS: f64 = 1e-12;

/// Ewald energy of `charges` at `positions` (Bohr) repeated on the lattice
/// spanned by `vectors`, in Hartree per cell.
///
/// `translations` are the real-space images (including the zero vector)
/// and `g_vectors` the reciprocal vectors (excluding zero) the sums run
/// over; both must be large enough for the splitting parameter `omega`, see
/// `hafnium.molecule.ewald`. The `G = 0` term and the self-interaction are
/// added analytically.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] for a lattice that does not span one
/// to three dimensions, a non-positive `omega`, a zero reciprocal vector or
/// two charges that coincide (modulo lattice translations), and
/// [`Error::LengthMismatch`] if `charges` and `positions` differ in length.
pub fn ewald_energy(
    vectors: &[[f64; 3]],
    charges: &[f64],
    positions: &[[f64; 3]],
    translations: &[[f64; 3]],
    g_vectors: &[[f64; 3]],
    omega: f64,
) -> Result<f64, Error> {
    if charges.len() != positions.len() {
        return Err(Error::LengthMismatch {
            left: charges.len(),
            right: positions.len(),
        });
    }
    if !(omega > 0.0 && omega.is_finite()) {
        return Err(Error::InvalidArgument("omega is a positive number".into()));
    }
    let geometry = Geometry::new(vectors)?;
    let real = real_space_sum(charges, positions, translations, omega)?;
    let reciprocal = reciprocal_space_sum(&geometry, charges, positions, g_vectors, omega)?;
    let self_energy = -omega / PI.sqrt() * charges.iter().map(|z| z * z).sum::<f64>();
    Ok(real + reciprocal + self_energy)
}

/// The periodic subspace: its dimension, its measure (length, area or
/// volume of the cell) and, for a slab or a chain, the unit vector that
/// gives the perpendicular coordinate of a pair.
struct Geometry {
    dimension: usize,
    measure: f64,
    axis: [f64; 3],
}

impl Geometry {
    fn new(vectors: &[[f64; 3]]) -> Result<Self, Error> {
        let dimension = vectors.len();
        let (measure, axis) = match vectors {
            [a] => (norm(a), scale(a, 1.0 / norm(a))),
            [a, b] => {
                let normal = cross(a, b);
                let area = norm(&normal);
                (area, scale(&normal, 1.0 / area))
            }
            [a, b, c] => (dot(a, &cross(b, c)).abs(), [0.0; 3]),
            _ => {
                return Err(Error::InvalidArgument(
                    "a lattice has one to three vectors".into(),
                ));
            }
        };
        let finite = measure.is_finite() && axis.iter().all(|x| x.is_finite());
        if !finite || measure <= 0.0 {
            return Err(Error::InvalidArgument(
                "lattice vectors are linearly dependent".into(),
            ));
        }
        Ok(Self {
            dimension,
            measure,
            axis,
        })
    }

    /// Distance of the pair vector `r` from the periodic subspace: zero in
    /// three dimensions, the signed height above a slab, the distance
    /// from the axis of a chain.
    fn perpendicular(&self, r: &[f64; 3]) -> f64 {
        match self.dimension {
            3 => 0.0,
            2 => dot(r, &self.axis),
            _ => {
                let along = dot(r, &self.axis);
                let residual: [f64; 3] = std::array::from_fn(|k| r[k] - along * self.axis[k]);
                norm(&residual)
            }
        }
    }

    /// Fourier transform of `erf(ωr)/r` along the periodic directions at a
    /// reciprocal vector of length `g > 0` and perpendicular coordinate
    /// `perpendicular` of the pair.
    fn kernel(&self, g: f64, perpendicular: f64, omega: f64) -> f64 {
        let half = g / (2.0 * omega);
        match self.dimension {
            3 => 4.0 * PI * (-half * half).exp() / (g * g),
            2 => {
                let z = perpendicular.abs();
                let rising = (-half * half - omega * omega * z * z).exp() * erfcx(half + omega * z);
                let falling = (-g * z).exp() * libm::erfc(half - omega * z);
                PI / g * (rising + falling)
            }
            _ => incomplete_bessel_integral(
                half * half,
                omega * omega * perpendicular * perpendicular,
            ),
        }
    }

    /// The finite part of the transform at `G = 0` (ADR-0012).
    fn kernel_at_zero(&self, perpendicular: f64, omega: f64) -> f64 {
        match self.dimension {
            3 => -PI / (omega * omega),
            2 => {
                let z = perpendicular.abs();
                -2.0 * PI
                    * (z * libm::erf(omega * z)
                        + (-omega * omega * z * z).exp() / (omega * PI.sqrt()))
            }
            _ => {
                let rho = perpendicular;
                if rho < ON_AXIS {
                    2.0 * omega.ln()
                } else {
                    -EULER_GAMMA
                        - (rho * rho).ln()
                        - exponential_integral_1(omega * omega * rho * rho)
                }
            }
        }
    }
}

/// `½ Σ_ij Σ_T Z_i Z_j erfc(ω|r_ij + T|) / |r_ij + T|` without the
/// self-term `i = j, T = 0`.
fn real_space_sum(
    charges: &[f64],
    positions: &[[f64; 3]],
    translations: &[[f64; 3]],
    omega: f64,
) -> Result<f64, Error> {
    let mut energy = 0.0;
    for translation in translations {
        let at_origin = norm(translation) < COINCIDENCE;
        for (i, (&z_i, r_i)) in charges.iter().zip(positions).enumerate() {
            for (j, (&z_j, r_j)) in charges.iter().zip(positions).enumerate() {
                let r: [f64; 3] = std::array::from_fn(|k| r_i[k] - r_j[k] + translation[k]);
                let distance = norm(&r);
                if distance < COINCIDENCE {
                    if i == j && at_origin {
                        continue;
                    }
                    return Err(Error::InvalidArgument(format!(
                        "charges {i} and {j} coincide (modulo lattice translations)"
                    )));
                }
                energy += 0.5 * z_i * z_j * libm::erfc(omega * distance) / distance;
            }
        }
    }
    Ok(energy)
}

/// `(1 / 2V) Σ_ij Z_i Z_j [Σ_{G≠0} cos(G·r_ij) K(G, r_ij⊥) + K_0(r_ij⊥)]`
/// with the kernels of [`Geometry`], over unordered pairs since every
/// factor is symmetric in `i` and `j`.
fn reciprocal_space_sum(
    geometry: &Geometry,
    charges: &[f64],
    positions: &[[f64; 3]],
    g_vectors: &[[f64; 3]],
    omega: f64,
) -> Result<f64, Error> {
    let lengths: Vec<f64> = g_vectors.iter().map(norm).collect();
    if lengths.iter().any(|&g| g < COINCIDENCE) {
        return Err(Error::InvalidArgument(
            "the reciprocal vectors exclude G = 0, whose term is analytic".into(),
        ));
    }
    let mut energy = 0.0;
    for (i, (&z_i, r_i)) in charges.iter().zip(positions).enumerate() {
        for (j, (&z_j, r_j)) in charges[..=i].iter().zip(positions).enumerate() {
            let r: [f64; 3] = std::array::from_fn(|k| r_i[k] - r_j[k]);
            let perpendicular = geometry.perpendicular(&r);
            let mut pair = geometry.kernel_at_zero(perpendicular, omega);
            for (g_vector, &g) in g_vectors.iter().zip(&lengths) {
                pair += dot(g_vector, &r).cos() * geometry.kernel(g, perpendicular, omega);
            }
            let multiplicity = if i == j { 1.0 } else { 2.0 };
            energy += multiplicity * z_i * z_j * pair;
        }
    }
    Ok(energy / (2.0 * geometry.measure))
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn scale(a: &[f64; 3], factor: f64) -> [f64; 3] {
    std::array::from_fn(|k| a[k] * factor)
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All lattice vectors `Σ n_i a_i` with `|n_i| ≤ extent`, the zero
    /// vector first.
    fn images(vectors: &[[f64; 3]], extent: i32) -> Vec<[f64; 3]> {
        let ranges: Vec<Vec<i32>> = vectors
            .iter()
            .map(|_| (-extent..=extent).collect())
            .collect();
        let mut result = vec![[0.0; 3]];
        let combos: Vec<Vec<i32>> = ranges.iter().fold(vec![vec![]], |acc, range| {
            acc.iter()
                .flat_map(|prefix| {
                    range.iter().map(move |&n| {
                        let mut next = prefix.clone();
                        next.push(n);
                        next
                    })
                })
                .collect()
        });
        for combo in combos {
            if combo.iter().all(|&n| n == 0) {
                continue;
            }
            let mut image = [0.0; 3];
            for (n, vector) in combo.iter().zip(vectors) {
                for k in 0..3 {
                    image[k] += f64::from(*n) * vector[k];
                }
            }
            result.push(image);
        }
        result
    }

    /// Reciprocal vectors `2π (A Aᵀ)⁻¹ A` for one to three lattice vectors,
    /// by explicit formulas.
    fn reciprocal(vectors: &[[f64; 3]]) -> Vec<[f64; 3]> {
        match vectors {
            [a] => vec![scale(a, 2.0 * PI / dot(a, a))],
            [a, b] => {
                let n = cross(a, b);
                let area2 = dot(&n, &n);
                vec![
                    scale(&cross(b, &n), 2.0 * PI / area2),
                    scale(&cross(&n, a), 2.0 * PI / area2),
                ]
            }
            [a, b, c] => {
                let volume = dot(a, &cross(b, c));
                vec![
                    scale(&cross(b, c), 2.0 * PI / volume),
                    scale(&cross(c, a), 2.0 * PI / volume),
                    scale(&cross(a, b), 2.0 * PI / volume),
                ]
            }
            _ => unreachable!(),
        }
    }

    fn energy(vectors: &[[f64; 3]], charges: &[f64], positions: &[[f64; 3]], omega: f64) -> f64 {
        let real = images(vectors, 12);
        let g_all = images(&reciprocal(vectors), 12);
        ewald_energy(vectors, charges, positions, &real, &g_all[1..], omega).expect("valid")
    }

    #[test]
    fn sodium_chloride_reproduces_the_madelung_constant() {
        // Primitive fcc cell of NaCl with nearest-neighbour distance 1.
        let vectors = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
        let positions = [[0.0; 3], [1.0, 0.0, 0.0]];
        let value = energy(&vectors, &[1.0, -1.0], &positions, 1.0);
        assert!((value + 1.747_564_594_633_18).abs() < 1e-10, "{value}");
    }

    #[test]
    fn energy_is_independent_of_the_splitting_parameter() {
        let vectors = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
        let positions = [[0.0; 3], [1.0, 0.0, 0.0]];
        let reference = energy(&vectors, &[1.0, -1.0], &positions, 1.0);
        for omega in [0.7, 1.4] {
            let value = energy(&vectors, &[1.0, -1.0], &positions, omega);
            assert!(
                (value - reference).abs() < 1e-10,
                "{omega}: {value} vs {reference}"
            );
        }
    }

    #[test]
    fn alternating_chain_reproduces_two_log_two() {
        let vectors = [[2.0, 0.0, 0.0]];
        let positions = [[0.0; 3], [1.0, 0.0, 0.0]];
        for omega in [0.8, 1.2] {
            let value = energy(&vectors, &[1.0, -1.0], &positions, omega);
            assert!(
                (value + 2.0 * 2.0_f64.ln()).abs() < 1e-10,
                "{omega}: {value}"
            );
        }
    }

    #[test]
    fn chain_with_off_axis_charges_is_independent_of_the_splitting_parameter() {
        let vectors = [[0.0, 0.0, 3.0]];
        let positions = [[0.3, 0.0, 0.0], [-0.4, 0.5, 1.5], [0.1, -0.2, 2.2]];
        let charges = [1.0, 2.0, -3.0];
        let reference = energy(&vectors, &charges, &positions, 0.6);
        let other = energy(&vectors, &charges, &positions, 1.1);
        assert!((other - reference).abs() < 1e-10, "{other} vs {reference}");
    }

    #[test]
    fn slab_is_independent_of_the_splitting_parameter() {
        let vectors = [[2.0, 0.0, 0.0], [0.5, 2.0, 0.0]];
        let positions = [[0.0; 3], [1.0, 1.0, 0.7], [0.3, 1.5, -0.4]];
        let charges = [3.0, 1.0, 1.0];
        let reference = energy(&vectors, &charges, &positions, 0.8);
        let other = energy(&vectors, &charges, &positions, 1.3);
        assert!((other - reference).abs() < 1e-10, "{other} vs {reference}");
    }

    #[test]
    fn coinciding_charges_are_rejected() {
        let vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let error = ewald_energy(
            &vectors,
            &[1.0, 1.0],
            &[[0.0; 3], [1.0, 0.0, 0.0]],
            &images(&vectors, 1),
            &images(&reciprocal(&vectors), 1)[1..],
            1.0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("coincide"));
    }

    #[test]
    fn malformed_input_is_rejected() {
        let cubic = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let none: [[f64; 3]; 0] = [];
        let zero = [[0.0; 3]];
        assert!(ewald_energy(&none, &[1.0], &[[0.0; 3]], &zero, &none, 1.0).is_err());
        assert!(ewald_energy(&cubic, &[1.0, 1.0], &[[0.0; 3]], &zero, &none, 1.0).is_err());
        assert!(ewald_energy(&cubic, &[1.0], &[[0.0; 3]], &zero, &none, 0.0).is_err());
        assert!(ewald_energy(&cubic, &[1.0], &[[0.0; 3]], &zero, &zero, 1.0).is_err());
        let dependent = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        assert!(ewald_energy(&dependent, &[1.0], &[[0.0; 3]], &zero, &none, 1.0).is_err());
    }
}

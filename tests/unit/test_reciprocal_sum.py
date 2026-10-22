"""Reciprocal-space sums by quadrature over the open directions (ADR-0021)."""

import math

import numpy as np
import pytest

from hafnium.basis import BasisLibrary, BasisSet
from hafnium.errors import InputError
from hafnium.integrals.periodic import (
    PerpendicularProfile,
    RangeSeparation,
    ReciprocalSum,
)
from hafnium.molecule import AtomicSystem, Lattice, ewald_energy, reciprocal_cutoff
from tests.cells import CUBIC, GENERAL_VECTORS, helium
from tests.tolerances import ENERGY_TOL, INTEGRAL_TOL, STRUCTURE_TOL

LIBRARY = BasisLibrary(offline=True)
AUXILIARY = "def2-universal-jkfit"
CHARGES = np.array([1.0, -2.0, 1.5])
POSITIONS = np.array([[0.1, 0.2, 0.3], [2.1, 1.9, -0.6], [1.0, 3.0, 0.5]])


def basis_for(system: AtomicSystem, name: str) -> BasisSet:
    return BasisSet.for_atoms(system, name, library=LIBRARY)


def erfc(values: np.ndarray) -> np.ndarray:
    return np.asarray([math.erfc(x) for x in values.ravel()]).reshape(values.shape)


def perpendicular_extent(lattice: Lattice, positions: np.ndarray) -> float:
    perpendicular = positions @ lattice.perpendicular_vectors_array().T
    separations = perpendicular[:, None, :] - perpendicular[None, :, :]
    return float(np.max(np.linalg.norm(separations, axis=-1)))


def ewald_by_reciprocal_sum(lattice: Lattice, separation: RangeSeparation) -> float:
    """The Ewald energy of CHARGES at POSITIONS with the quadrature kernels."""
    omega = separation.omega
    separations = POSITIONS[:, None] - POSITIONS[None]
    extent = float(np.max(np.linalg.norm(separations, axis=-1)))
    real = 0.0
    translations = np.asarray(
        lattice.translations(separation.short_range_radius + extent)
    )
    for translation in translations:
        r: np.ndarray = POSITIONS[:, None, :] - POSITIONS[None, :, :] + translation
        distance = np.sqrt(np.sum(r * r, axis=-1))
        apart = distance > STRUCTURE_TOL
        safe = np.where(apart, distance, 1.0)
        pairs = np.outer(CHARGES, CHARGES) * erfc(omega * safe) / safe
        real += 0.5 * float(np.sum(np.where(apart, pairs, 0.0)))
    profile = PerpendicularProfile(
        perpendicular_extent(lattice, POSITIONS), math.inf, 0
    )
    total = separation.reciprocal_sum(
        lattice, profile, charge_product=float(np.sum(np.abs(CHARGES))) ** 2
    )
    structure = np.exp(-1j * total.vectors @ POSITIONS.T) @ CHARGES
    reciprocal = float(np.sum(total.weights * np.abs(structure) ** 2)) / (
        2.0 * lattice.volume
    )
    self_energy = -omega / math.sqrt(math.pi) * float(np.sum(CHARGES**2))
    return real + reciprocal + self_energy


@pytest.mark.parametrize("dimension", [3, 2, 1])
@pytest.mark.parametrize("factor", [1.0, 0.6])
def test_point_charges_reproduce_the_analytic_ewald_energy(
    dimension: int, factor: float
) -> None:
    # The quadrature over the open directions with the finite part on the
    # zero line against the analytic kernels of ADR-0012 in the compiled
    # layer, for a charged cell, which exercises the finite part.
    lattice = Lattice(GENERAL_VECTORS[:dimension])
    separation = RangeSeparation.for_lattice(lattice)
    separation = RangeSeparation(factor * separation.omega, separation.precision)
    expected = ewald_energy(lattice, CHARGES, POSITIONS, omega=separation.omega)
    assert ewald_by_reciprocal_sum(lattice, separation) == pytest.approx(
        expected, abs=ENERGY_TOL
    )


def test_three_dimensional_sum_is_the_kernel_with_the_finite_part() -> None:
    separation = RangeSeparation.for_lattice(CUBIC, precision=1e-10)
    profile = PerpendicularProfile(0.0, 1.0, 0)
    total = separation.reciprocal_sum(CUBIC, profile, charge_product=4.0)
    np.testing.assert_array_equal(total.vectors[0], 0.0)
    lengths = np.linalg.norm(total.vectors, axis=1)
    assert np.all(np.diff(lengths) >= -INTEGRAL_TOL)
    assert total.weights[0] == pytest.approx(-math.pi / separation.omega**2)
    g_squared = lengths[1:] ** 2
    expected = (
        4.0 * math.pi * np.exp(-g_squared / (4.0 * separation.omega**2)) / g_squared
    )
    np.testing.assert_allclose(total.weights[1:], expected, atol=INTEGRAL_TOL)
    more = separation.reciprocal_sum(CUBIC, profile, charge_product=4e6)
    assert more.n_vectors > total.n_vectors
    assert separation.finite_part(Lattice(GENERAL_VECTORS[:2])) == pytest.approx(
        -2.0 * math.sqrt(math.pi) / separation.omega
    )
    assert separation.finite_part(Lattice(GENERAL_VECTORS[:1])) == pytest.approx(
        2.0 * math.log(separation.omega)
    )


@pytest.mark.parametrize("dimension", [2, 1])
def test_zero_line_weights_sum_to_the_finite_part(dimension: int) -> None:
    lattice = Lattice(GENERAL_VECTORS[:dimension])
    separation = RangeSeparation.for_lattice(lattice)
    profile = PerpendicularProfile(1.0, 0.5, 2)
    total = separation.reciprocal_sum(lattice, profile, charge_product=4.0)
    np.testing.assert_array_equal(total.vectors[0], 0.0)
    # Σ_j w_j ṽ(G_j) [F(G_j) − F(0)] + v_0 F(0): the weights of the line
    # G_∥ = 0, with the zero vector, add up to the finite part.
    fractional = lattice.fractional_coordinates(total.vectors)
    on_line = np.all(np.abs(fractional) < STRUCTURE_TOL, axis=1)
    assert float(np.sum(total.weights[on_line])) == pytest.approx(
        separation.finite_part(lattice), abs=STRUCTURE_TOL
    )
    assert np.all(total.weights[1:] > 0.0)
    perpendicular = total.vectors @ lattice.perpendicular_vectors_array().T
    assert np.all(np.linalg.norm(perpendicular[1:], axis=1) > 0.0)
    # With a momentum transfer no line passes through the origin.
    shifted = separation.reciprocal_sum(
        lattice,
        profile,
        charge_product=4.0,
        transfer=0.5 * lattice.reciprocal_vectors_array()[0],
    )
    assert np.all(shifted.weights > 0.0)
    assert np.min(np.linalg.norm(shifted.vectors, axis=1)) > 0.0


def test_profile_collects_extent_exponent_and_degree() -> None:
    cell = helium(2)
    basis, auxiliary = basis_for(cell, "cc-pvdz"), basis_for(cell, AUXILIARY)
    profile = PerpendicularProfile.from_functions(cell, basis, auxiliary)
    normal = cell.lattice.perpendicular_vectors_array()[0]
    positions = cell.positions_array()
    assert profile.extent == pytest.approx(
        abs(float(np.dot(positions[1] - positions[0], normal)))
    )
    assert profile.smallest_exponent == min(
        min(shell.exponents) for b in (basis, auxiliary) for shell in b.shells
    )
    assert profile.degree == 2 * (
        basis.max_angular_momentum + auxiliary.max_angular_momentum
    )
    crystal = helium(3)
    flat = PerpendicularProfile.from_functions(crystal, basis_for(crystal, "cc-pvdz"))
    assert flat.extent == 0.0


def test_inputs_are_validated() -> None:
    with pytest.raises(InputError, match="extent"):
        PerpendicularProfile(-1.0, 1.0, 0)
    with pytest.raises(InputError, match="exponent"):
        PerpendicularProfile(0.0, 0.0, 0)
    with pytest.raises(InputError, match="degree"):
        PerpendicularProfile(0.0, 1.0, -1)
    with pytest.raises(InputError, match="at least one basis"):
        PerpendicularProfile.from_functions(helium(2))
    separation = RangeSeparation.for_lattice(CUBIC)
    with pytest.raises(InputError, match="three components"):
        separation.reciprocal_sum(
            CUBIC,
            PerpendicularProfile(0.0, 1.0, 0),
            charge_product=1.0,
            transfer=np.zeros(2),
        )
    with pytest.raises(InputError, match=r"\[G, 3\]"):
        ReciprocalSum(np.zeros((2, 3)), np.zeros(3))
    with pytest.raises(InputError, match="one to three"):
        reciprocal_cutoff(CUBIC, 1.0, charge_product=1.0, precision=1e-12, dimension=4)

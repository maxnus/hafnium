"""Ewald sums against Madelung constants, direct sums and consistency checks."""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from hafnium.errors import InputError
from hafnium.molecule import (
    Lattice,
    ewald_energy,
    ewald_parameters,
)
from tests.tolerances import ENERGY_TOL, LITERATURE_TOL

# Madelung constants α with the energy per formula unit −α / R for the
# nearest-neighbour distance R (Wikipedia, "Madelung constant"; the two-
# dimensional square lattice from Borwein et al.; the chain is 2 ln 2).
MADELUNG_NACL = 1.747_564_594_633_18
MADELUNG_CSCL = 1.762_674_773_070_99
MADELUNG_SQUARE = 1.615_542_626_7
MADELUNG_CHAIN = 2.0 * math.log(2.0)
# Energy of a unit charge in a cubic cell of side L with a neutralizing
# background, −α_sc / (2L) with α_sc = 2.837 297 479 480 6 (Leslie and Gillan).
POINT_CHARGE_CUBIC = 2.837_297_479_480_6

PLUS_MINUS = np.array([1.0, -1.0])


def test_rocksalt_and_caesium_chloride_madelung_constants() -> None:
    fcc = Lattice(((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)))
    nacl = ewald_energy(fcc, PLUS_MINUS, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert nacl == pytest.approx(-MADELUNG_NACL, abs=ENERGY_TOL)
    cubic = Lattice(((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)))
    cscl = ewald_energy(cubic, PLUS_MINUS, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert cscl == pytest.approx(-MADELUNG_CSCL / math.sqrt(3.0), abs=ENERGY_TOL)


def test_square_lattice_madelung_constant() -> None:
    square = Lattice(((1.0, 1.0, 0.0), (1.0, -1.0, 0.0)))
    value = ewald_energy(
        square, PLUS_MINUS, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )
    assert value == pytest.approx(-MADELUNG_SQUARE, abs=LITERATURE_TOL)


def test_alternating_chain_madelung_constant() -> None:
    chain = Lattice(((0.0, 0.0, 2.0),))
    value = ewald_energy(
        chain, PLUS_MINUS, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert value == pytest.approx(-MADELUNG_CHAIN, abs=ENERGY_TOL)


def test_point_charge_in_a_cubic_box_with_background() -> None:
    length = 3.0
    cubic = Lattice(((length, 0.0, 0.0), (0.0, length, 0.0), (0.0, 0.0, length)))
    value = ewald_energy(cubic, np.array([1.0]), np.zeros((1, 3)))
    assert value == pytest.approx(-POINT_CHARGE_CUBIC / (2.0 * length), abs=ENERGY_TOL)


def random_system(dimension: int) -> tuple[Lattice, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(dimension)
    vectors = 3.0 * np.eye(3)[:dimension] + 0.4 * rng.normal(size=(dimension, 3))
    charges = np.array([3.0, 1.0, 1.0, 2.0])
    positions = rng.normal(size=(4, 3)) * 1.5
    return Lattice(tuple(map(tuple, vectors))), charges, positions


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_energy_is_independent_of_the_range_separation(dimension: int) -> None:
    lattice, charges, positions = random_system(dimension)
    reference = ewald_energy(lattice, charges, positions)
    default = ewald_parameters(lattice, charges).omega
    for factor in [0.5, 2.0, 3.0]:
        value = ewald_energy(lattice, charges, positions, omega=factor * default)
        assert value == pytest.approx(reference, abs=ENERGY_TOL)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_energy_is_invariant_under_lattice_translations_of_atoms(
    dimension: int,
) -> None:
    lattice, charges, positions = random_system(dimension)
    reference = ewald_energy(lattice, charges, positions)
    shifts = np.array([[1, 0, 0], [0, -2, 0], [3, 1, 0], [-1, 1, 0]], dtype=float)
    moved = positions + shifts[:, :dimension] @ lattice.vectors_array()
    assert ewald_energy(lattice, charges, moved) == pytest.approx(
        reference, abs=ENERGY_TOL
    )
    rigid = ewald_energy(lattice, charges, positions + np.array([0.3, -0.7, 1.1]))
    assert rigid == pytest.approx(reference, abs=ENERGY_TOL)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_energy_is_extensive_in_the_supercell(dimension: int) -> None:
    lattice, charges, positions = random_system(dimension)
    reference = ewald_energy(lattice, charges, positions)
    multiples = (2, 1, 3)[:dimension]
    images = [
        np.asarray(image, dtype=float) @ lattice.vectors_array()
        for image in np.ndindex(*multiples)
    ]
    big_positions = np.concatenate([positions + shift for shift in images])
    big_charges = np.tile(charges, len(images))
    value = ewald_energy(lattice.supercell(multiples), big_charges, big_positions)
    assert value == pytest.approx(len(images) * reference, abs=ENERGY_TOL)


def direct_sum(
    lattice: Lattice,
    charges: NDArray[np.float64],
    positions: NDArray[np.float64],
    extent: int,
) -> float:
    """Brute-force lattice sum of a neutral cell over ``±extent`` images."""
    translations = lattice.translations(
        extent * min(np.linalg.norm(v) for v in lattice.vectors) + 1e-9
    )
    energy = 0.0
    for translation in translations:
        shift = np.asarray(translation, dtype=float)
        separations = positions[:, None, :] - positions[None, :, :] + shift
        distances = np.linalg.norm(separations, axis=-1)
        with np.errstate(divide="ignore"):
            inverse = np.where(distances > 1e-12, 1.0 / distances, 0.0)
        energy += 0.5 * float(charges @ inverse @ charges)
    return energy


def test_neutral_chain_with_off_axis_charges_matches_direct_summation() -> None:
    # A centrosymmetric cell: the cell-cell interaction falls off as the
    # inverse fifth power, so the direct sum converges quickly.
    chain = Lattice(((0.0, 0.0, 4.0),))
    positions = np.array(
        [[0.5, 0.2, 0.0], [-0.5, -0.2, 1.0], [-0.5, -0.2, 3.0], [0.5, 0.2, 2.0]]
    )
    charges = np.array([1.0, -1.0, 1.0, -1.0])
    expected = direct_sum(chain, charges, positions, extent=400)
    assert ewald_energy(chain, charges, positions) == pytest.approx(expected, abs=1e-8)


def test_neutral_slab_with_buckling_matches_direct_summation() -> None:
    # Two dipole-free sublayers; the direct sum converges as the inverse cube
    # of the image count in two dimensions.
    slab = Lattice(((2.0, 0.0, 0.0), (0.0, 2.0, 0.0)))
    positions = np.array(
        [[0.0, 0.0, 0.3], [1.0, 1.0, -0.3], [1.0, 0.0, -0.3], [0.0, 1.0, 0.3]]
    )
    charges = np.array([1.0, 1.0, -1.0, -1.0])
    expected = direct_sum(slab, charges, positions, extent=120)
    assert ewald_energy(slab, charges, positions) == pytest.approx(expected, abs=1e-7)


def test_cutoffs_grow_with_the_precision() -> None:
    lattice, charges, _ = random_system(3)
    loose = ewald_parameters(lattice, charges, precision=1e-6)
    tight = ewald_parameters(lattice, charges, precision=1e-14)
    assert loose.omega == tight.omega
    assert tight.real_space_cutoff > loose.real_space_cutoff
    assert tight.reciprocal_cutoff > loose.reciprocal_cutoff
    assert loose.omega == pytest.approx(math.sqrt(math.pi) / lattice.volume ** (1 / 3))


def test_malformed_input_is_rejected() -> None:
    lattice, charges, positions = random_system(3)
    with pytest.raises(InputError, match="shape"):
        ewald_energy(lattice, charges[:3], positions)
    with pytest.raises(InputError, match="omega is positive"):
        ewald_energy(lattice, charges, positions, omega=-1.0)
    with pytest.raises(InputError, match="precision is a positive"):
        ewald_energy(lattice, charges, positions, precision=0.0)
    with pytest.raises(ValueError, match="coincide"):
        ewald_energy(lattice, charges[:2], np.zeros((2, 3)))

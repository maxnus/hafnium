"""Lattices: reciprocal vectors, coordinates, translations, supercells."""

import itertools
import math

import numpy as np
import pytest

from hafnium.errors import InputError
from hafnium.molecule import ANGSTROM_IN_BOHR, Lattice
from tests.tolerances import STRUCTURE_TOL

CHAIN = Lattice(((2.0, 0.5, -0.3),))
SLAB = Lattice(((3.0, 0.2, 0.1), (-0.4, 2.5, 0.3)))
CRYSTAL = Lattice(((3.0, 0.2, 0.1), (-0.4, 2.5, 0.3), (0.5, -0.6, 4.0)))
LATTICES = {"chain": CHAIN, "slab": SLAB, "crystal": CRYSTAL}


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_reciprocal_vectors_are_dual_to_the_lattice(name: str) -> None:
    lattice = LATTICES[name]
    a = lattice.vectors_array()
    b = lattice.reciprocal_vectors_array()
    assert b.shape == (lattice.dimension, 3)
    np.testing.assert_allclose(
        b @ a.T, 2 * np.pi * np.eye(lattice.dimension), atol=STRUCTURE_TOL
    )
    # the reciprocal vectors lie in the span of the lattice vectors
    projector = a.T @ np.linalg.inv(a @ a.T) @ a
    np.testing.assert_allclose(b @ projector, b, atol=STRUCTURE_TOL)
    again = lattice.reciprocal.reciprocal
    np.testing.assert_allclose(again.vectors_array(), a, atol=STRUCTURE_TOL)


def test_volume_is_length_area_or_volume() -> None:
    assert CHAIN.volume == pytest.approx(math.sqrt(4 + 0.25 + 0.09))
    a, b = SLAB.vectors_array()
    assert SLAB.volume == pytest.approx(np.linalg.norm(np.cross(a, b)))
    assert CRYSTAL.volume == pytest.approx(abs(np.linalg.det(CRYSTAL.vectors_array())))


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_fractional_coordinates_round_trip_and_wrap(name: str) -> None:
    lattice = LATTICES[name]
    rng = np.random.default_rng(7)
    positions = rng.normal(size=(5, 3)) * 10.0
    fractional = lattice.fractional_coordinates(positions)
    assert fractional.shape == (5, lattice.dimension)
    inside = lattice.cartesian_coordinates(fractional)
    # the periodic components are recovered exactly
    a = lattice.vectors_array()
    np.testing.assert_allclose(inside @ a.T, positions @ a.T, atol=STRUCTURE_TOL)
    wrapped = lattice.wrap(positions)
    wrapped_fractional = lattice.fractional_coordinates(wrapped)
    assert np.all(wrapped_fractional >= -STRUCTURE_TOL)
    assert np.all(wrapped_fractional < 1.0)
    # wrapping subtracts integer combinations of lattice vectors
    shifts = lattice.fractional_coordinates(positions - wrapped)
    np.testing.assert_allclose(shifts, np.round(shifts), atol=STRUCTURE_TOL)
    # components perpendicular to the lattice are unchanged
    projector = a.T @ np.linalg.inv(a @ a.T) @ a
    np.testing.assert_allclose(
        wrapped @ (np.eye(3) - projector),
        positions @ (np.eye(3) - projector),
        atol=STRUCTURE_TOL,
    )


@pytest.mark.parametrize("name", sorted(LATTICES))
@pytest.mark.parametrize("cutoff", [0.0, 2.5, 7.0])
def test_translations_within_cutoff_match_brute_force(name: str, cutoff: float) -> None:
    lattice = LATTICES[name]
    translations = lattice.translations(cutoff)
    lengths = np.linalg.norm(translations, axis=1)
    assert np.all(lengths <= cutoff + STRUCTURE_TOL)
    assert np.all(np.diff(lengths) >= -STRUCTURE_TOL)
    np.testing.assert_array_equal(translations[0], np.zeros(3))
    a = lattice.vectors_array()
    expected = {
        combo
        for combo in itertools.product(range(-10, 11), repeat=lattice.dimension)
        if np.linalg.norm(np.asarray(combo, dtype=float) @ a) <= cutoff
    }
    found = {
        tuple(round(x) for x in lattice.fractional_coordinates(t[None, :])[0])
        for t in translations
    }
    assert found == expected
    # closed under inversion
    assert all((-n[0], *(-x for x in n[1:])) in found for n in found)


def test_translations_reject_bad_cutoffs() -> None:
    with pytest.raises(InputError, match="non-negative"):
        CHAIN.translations(-1.0)
    with pytest.raises(InputError, match="non-negative"):
        CHAIN.translations(math.inf)


def test_from_vectors_converts_units() -> None:
    lattice = Lattice.from_vectors([(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)])
    assert lattice.vectors == (
        (ANGSTROM_IN_BOHR, 0.0, 0.0),
        (0.0, 2.0 * ANGSTROM_IN_BOHR, 0.0),
    )
    same = Lattice.from_vectors(np.eye(3) * 2.0, unit="bohr")
    assert same.vectors == ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))
    with pytest.raises(InputError, match="unknown length unit"):
        Lattice.from_vectors(np.eye(3), unit="pm")


def test_supercell_scales_the_vectors() -> None:
    doubled = SLAB.supercell((2, 3))
    np.testing.assert_allclose(
        doubled.vectors_array(), SLAB.vectors_array() * np.array([[2.0], [3.0]])
    )
    assert doubled.volume == pytest.approx(6 * SLAB.volume)
    with pytest.raises(InputError, match="positive integers"):
        SLAB.supercell((2,))
    with pytest.raises(InputError, match="positive integers"):
        SLAB.supercell((2, 0))


def test_invalid_lattices_are_rejected() -> None:
    with pytest.raises(InputError, match="one to three vectors"):
        Lattice(())
    with pytest.raises(InputError, match="one to three vectors"):
        Lattice(((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)))
    with pytest.raises(InputError, match="dependent"):
        Lattice(((1, 0, 0), (2, 0, 0)))
    with pytest.raises(InputError, match="dependent"):
        Lattice(((0, 0, 0),))
    with pytest.raises(InputError, match="three finite components"):
        Lattice.from_vectors([(1.0, 0.0)], unit="bohr")
    with pytest.raises(InputError, match="three finite components"):
        Lattice.from_vectors([(1.0, 0.0, math.nan)], unit="bohr")

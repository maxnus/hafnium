"""Wigner matrices of real solid harmonics and the rotation of a basis."""

import numpy as np
import pytest

from hafnium.basis import BasisSet, cartesian_components, solid_harmonic_coefficients
from hafnium.errors import InputError, SymmetryError
from hafnium.integrals import overlap
from hafnium.molecule import Atom, Molecule
from hafnium.symmetry import basis_rotation, solid_harmonic_rotation
from tests import molecules
from tests.hamiltonians import ENGINE
from tests.shell_tables import LIBRARY
from tests.tolerances import INTEGRAL_TOL, STRUCTURE_TOL


def random_rotation(rng: np.random.Generator, *, proper: bool) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return q if (np.linalg.det(q) > 0) == proper else -q


def harmonics(l: int, points: np.ndarray) -> np.ndarray:
    """``S_lm`` at ``points`` ``[N, 3]`` from the polynomial table, ``[N, 2l + 1]``."""
    monomials = np.stack(
        [
            np.prod(points ** np.asarray(exponents), axis=1)
            for exponents in cartesian_components(l)
        ],
        axis=1,
    )
    return monomials @ np.asarray(solid_harmonic_coefficients(l)).T


@pytest.mark.parametrize("proper", [True, False])
@pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
def test_rotation_matrices_transform_the_harmonics(l: int, proper: bool) -> None:
    rng = np.random.default_rng(l + (0 if proper else 10))
    rotation = random_rotation(rng, proper=proper)
    d = np.asarray(solid_harmonic_rotation(l, rotation))
    assert d.shape == (2 * l + 1, 2 * l + 1)
    np.testing.assert_allclose(d.T @ d, np.eye(2 * l + 1), atol=STRUCTURE_TOL)
    points = rng.standard_normal((12, 3))
    # S_lm(R⁻¹ r) = Σ_m' D[m', m] S_lm'(r); the rows of ``points`` map as r Rᵀ
    # under R, so R⁻¹ r is points @ R.
    np.testing.assert_allclose(
        harmonics(l, points @ rotation), harmonics(l, points) @ d, atol=STRUCTURE_TOL
    )


def test_rotation_matrices_form_a_representation() -> None:
    rng = np.random.default_rng(3)
    first = random_rotation(rng, proper=True)
    second = random_rotation(rng, proper=False)
    for l in range(4):
        np.testing.assert_allclose(
            solid_harmonic_rotation(l, first @ second),
            solid_harmonic_rotation(l, first) @ solid_harmonic_rotation(l, second),
            atol=STRUCTURE_TOL,
        )
    # p functions rotate like x, y, z, and the inversion gives (−1)^l.
    np.testing.assert_allclose(
        solid_harmonic_rotation(1, first), first, atol=STRUCTURE_TOL
    )
    for l in range(5):
        np.testing.assert_allclose(
            solid_harmonic_rotation(l, -np.eye(3)),
            (-1) ** l * np.eye(2 * l + 1),
            atol=STRUCTURE_TOL,
        )
    with pytest.raises(InputError, match="orthogonal"):
        solid_harmonic_rotation(2, 2.0 * np.eye(3))
    with pytest.raises(InputError, match="orthogonal"):
        solid_harmonic_rotation(2, np.eye(2))


def test_basis_rotation_relates_the_overlaps_of_rotated_geometries() -> None:
    # The overlap of the rotated molecule is Π S Πᵀ: rotating the functions
    # with the geometry leaves every integral unchanged.
    molecule = molecules.water()
    basis = BasisSet.for_atoms(molecule, "cc-pvdz", library=LIBRARY)
    rng = np.random.default_rng(5)
    rotation = random_rotation(rng, proper=False)
    rotated = Molecule(
        tuple(
            Atom(atom.symbol, tuple(rotation @ np.asarray(atom.position)))  # type: ignore[arg-type]
            for atom in molecule.atoms
        )
    )
    identity = tuple(range(molecule.n_atoms))
    pi = np.asarray(basis_rotation(basis, rotation, identity))
    n = basis.n_functions
    np.testing.assert_allclose(pi.T @ pi, np.eye(n), atol=STRUCTURE_TOL)
    s = np.asarray(overlap(ENGINE, basis, molecule))
    s_rotated = np.asarray(overlap(ENGINE, basis, rotated))
    np.testing.assert_allclose(pi @ s @ pi.T, s_rotated, atol=INTEGRAL_TOL)
    # Swapping the two hydrogens as well permutes their blocks.
    swapped = (0, 2, 1)
    pi_swapped = np.asarray(basis_rotation(basis, np.eye(3), swapped))
    offsets = basis.shell_offsets()
    first = offsets[basis.shell_atoms.index(1)]
    second = offsets[basis.shell_atoms.index(2)]
    width = n - second
    assert np.allclose(
        pi_swapped[second : second + width, first : first + width], np.eye(width)
    )
    with pytest.raises(SymmetryError, match="different basis functions"):
        basis_rotation(basis, np.eye(3), (1, 0, 2))

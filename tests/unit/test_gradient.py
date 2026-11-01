"""Analytic nuclear gradients against finite differences of the energy (ADR-0035)."""

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from hafnium.backend import Array
from hafnium.basis import BasisLibrary, BasisSet
from hafnium.errors import InputError
from hafnium.integrals import RustEngine
from hafnium.molecule import Molecule
from hafnium.properties import NuclearGradient, nuclear_gradient
from hafnium.scf import ScfOptions, ScfResult, hartree_fock
from hafnium.symmetry import SymmetrySpec
from tests.molecules import ammonia_tilted, displaced, water
from tests.tolerances import GRADIENT_FINITE_DIFFERENCE_TOL, GRADIENT_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
AUXILIARY = "def2-universal-jkfit"
CLASSES = [
    "RHF",
    "complex RHF",
    "UHF",
    "complex UHF",
    "paired UHF",
    "GHF",
    "complex GHF",
    "paired GHF",
]
OPEN_SHELL_CLASSES = ["UHF", "complex UHF", "GHF", "complex GHF"]
# Tight convergence, so that the finite differences of the energy are limited
# by the rounding of the energies and not by the residual orbital error.
OPTIONS = ScfOptions(
    gradient_tolerance=1e-10, energy_tolerance=1e-13, max_iterations=200
)
STEP = 2e-3


def bases(molecule: Molecule, basis_name: str) -> tuple[BasisSet, BasisSet]:
    return (
        BasisSet.for_atoms(molecule, basis_name, library=LIBRARY),
        BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY),
    )


def converge(
    molecule: Molecule, basis: BasisSet, auxiliary: BasisSet, name: str
) -> ScfResult:
    return hartree_fock(
        ENGINE,
        basis,
        molecule,
        symmetry=SymmetrySpec.from_name(name),
        auxiliary=auxiliary,
        options=OPTIONS,
    )


def analytic(
    molecule: Molecule, basis: BasisSet, auxiliary: BasisSet, name: str
) -> NuclearGradient:
    result = converge(molecule, basis, auxiliary, name)
    return nuclear_gradient(ENGINE, basis, auxiliary, molecule, result)


def finite_differences(
    molecule: Molecule, energy: Callable[[Molecule], float]
) -> Array:
    """Five-point central differences of ``energy`` along every coordinate."""
    gradient = np.zeros((molecule.n_atoms, 3))
    for atom in range(molecule.n_atoms):
        for axis in range(3):
            samples = [
                energy(displaced(molecule, atom, axis, k * STEP))
                for k in (-2, -1, 1, 2)
            ]
            gradient[atom, axis] = (
                samples[0] - 8 * samples[1] + 8 * samples[2] - samples[3]
            ) / (12 * STEP)
    return gradient


@pytest.fixture(scope="module")
def closed_shell() -> tuple[Molecule, BasisSet, BasisSet, Array]:
    molecule = ammonia_tilted()
    basis, auxiliary = bases(molecule, "sto-3g")
    numeric = finite_differences(
        molecule, lambda m: converge(m, basis, auxiliary, "RHF").energy
    )
    return molecule, basis, auxiliary, numeric


@pytest.fixture(scope="module")
def open_shell() -> tuple[Molecule, BasisSet, BasisSet, Array]:
    molecule = Molecule(ammonia_tilted().atoms, charge=1, spin=1)
    basis, auxiliary = bases(molecule, "sto-3g")
    numeric = finite_differences(
        molecule, lambda m: converge(m, basis, auxiliary, "UHF").energy
    )
    return molecule, basis, auxiliary, numeric


@pytest.mark.parametrize("name", CLASSES)
def test_every_class_matches_finite_differences(
    name: str, closed_shell: tuple[Molecule, BasisSet, BasisSet, Array]
) -> None:
    """Every class converges to the closed-shell state and differentiates it."""
    molecule, basis, auxiliary, numeric = closed_shell
    gradient = analytic(molecule, basis, auxiliary, name)
    np.testing.assert_allclose(
        gradient.total, numeric, atol=GRADIENT_FINITE_DIFFERENCE_TOL
    )


@pytest.mark.parametrize("name", OPEN_SHELL_CLASSES)
def test_open_shell_classes_match_finite_differences(
    name: str, open_shell: tuple[Molecule, BasisSet, BasisSet, Array]
) -> None:
    molecule, basis, auxiliary, numeric = open_shell
    gradient = analytic(molecule, basis, auxiliary, name)
    np.testing.assert_allclose(
        gradient.total, numeric, atol=GRADIENT_FINITE_DIFFERENCE_TOL
    )


def test_every_term_is_invariant_under_translation(
    closed_shell: tuple[Molecule, BasisSet, BasisSet, Array],
) -> None:
    molecule, basis, auxiliary, _ = closed_shell
    gradient = analytic(molecule, basis, auxiliary, "RHF")
    for term in dataclasses.fields(gradient):
        values: Array = getattr(gradient, term.name)
        assert values.shape == (molecule.n_atoms, 3)
        np.testing.assert_allclose(np.sum(values, axis=0), 0.0, atol=GRADIENT_TOL)
    # The terms are not individually small, so the invariance is a real check.
    assert np.abs(gradient.one_electron).max() > 1.0
    assert np.abs(gradient.coulomb).max() > 1.0


def test_point_group_sectors_give_the_same_gradient() -> None:
    molecule = water()
    basis, auxiliary = bases(molecule, "def2-svp")
    plain = analytic(molecule, basis, auxiliary, "RHF")
    adapted = converge_with_point_group(molecule, basis, auxiliary)
    np.testing.assert_allclose(adapted.total, plain.total, atol=GRADIENT_TOL)
    # The force along the symmetry axis and in the plane only.
    np.testing.assert_allclose(plain.total[:, 0], 0.0, atol=GRADIENT_TOL)


def converge_with_point_group(
    molecule: Molecule, basis: BasisSet, auxiliary: BasisSet
) -> NuclearGradient:
    symmetry = SymmetrySpec.from_name("RHF").with_point_group("C2v")
    result = hartree_fock(
        ENGINE, basis, molecule, symmetry=symmetry, auxiliary=auxiliary, options=OPTIONS
    )
    assert result.structure.irreps is not None
    return nuclear_gradient(ENGINE, basis, auxiliary, molecule, result)


def test_another_basis_is_rejected(
    closed_shell: tuple[Molecule, BasisSet, BasisSet, Array],
) -> None:
    molecule, _, auxiliary, _ = closed_shell
    basis, _ = bases(molecule, "def2-svp")
    result = converge(molecule, basis, auxiliary, "RHF")
    smaller = BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY)
    with pytest.raises(InputError, match="spatial functions"):
        nuclear_gradient(ENGINE, smaller, auxiliary, molecule, result)


def test_kohn_sham_results_are_rejected(
    closed_shell: tuple[Molecule, BasisSet, BasisSet, Array],
) -> None:
    molecule, basis, auxiliary, _ = closed_shell
    result = converge(molecule, basis, auxiliary, "RHF")
    kohn_sham = dataclasses.replace(
        result, energies=dataclasses.replace(result.energies, exchange_correlation=-1.0)
    )
    with pytest.raises(InputError, match="Kohn–Sham"):
        nuclear_gradient(ENGINE, basis, auxiliary, molecule, kohn_sham)

"""Derivative integrals and nuclear gradients against PySCF, and stored data."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.basis import BasisLibrary, BasisSet
from hafnium.integrals import (
    DensityFittingDerivatives,
    RustEngine,
    kinetic_derivative,
    nuclear_attraction_derivative,
    overlap_derivative,
)
from hafnium.molecule import Molecule
from hafnium.properties import nuclear_gradient
from hafnium.scf import ScfOptions, hartree_fock
from hafnium.symmetry import SymmetrySpec
from tests.molecules import ammonia_tilted, water, water_cation
from tests.tolerances import GRADIENT_TOL, INTEGRAL_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
AUXILIARY = "def2-universal-jkfit"
MOLECULES = {
    "water": water(),
    "water_cation": water_cation(),
    "ammonia_tilted": ammonia_tilted(),
}
INTEGRAL_CASES = [("water", "cc-pvdz"), ("ammonia_tilted", "def2-svp")]
GRADIENT_CASES = [
    ("water", "def2-svp", "rhf"),
    ("water_cation", "def2-svp", "uhf"),
    ("ammonia_tilted", "cc-pvdz", "rhf"),
]
STORED_CASES = GRADIENT_CASES[:2]
CLASSES = {"rhf": "RHF", "uhf": "UHF"}
OPTIONS = ScfOptions(
    gradient_tolerance=1e-10, energy_tolerance=1e-13, max_iterations=200
)


def bases(molecule: Molecule, basis_name: str) -> tuple[BasisSet, BasisSet]:
    return (
        BasisSet.for_atoms(molecule, basis_name, library=LIBRARY),
        BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY),
    )


def gradient_for(name: str, basis_name: str, kind: str) -> np.ndarray:
    molecule = MOLECULES[name]
    basis, auxiliary = bases(molecule, basis_name)
    result = hartree_fock(
        ENGINE,
        basis,
        molecule,
        symmetry=SymmetrySpec.from_name(CLASSES[kind]),
        auxiliary=auxiliary,
        options=OPTIONS,
    )
    return np.asarray(
        nuclear_gradient(ENGINE, basis, auxiliary, molecule, result).total
    )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), INTEGRAL_CASES)
def test_derivative_integrals_match_libcint(name: str, basis_name: str) -> None:
    from tests.reference.pyscf_oracle import derivative_integral_reference

    molecule = MOLECULES[name]
    basis, auxiliary = bases(molecule, basis_name)
    expected = derivative_integral_reference(molecule, basis, auxiliary)
    attraction, nuclei = nuclear_attraction_derivative(ENGINE, basis, molecule)
    fitting = DensityFittingDerivatives.build(ENGINE, basis, auxiliary, molecule)
    actual = {
        "overlap": overlap_derivative(ENGINE, basis, molecule),
        "kinetic": kinetic_derivative(ENGINE, basis, molecule),
        "attraction": attraction,
        "attraction_nuclei": nuclei,
        "metric": fitting.metric,
        "three_center_orbital": fitting.orbital,
        "three_center_auxiliary": fitting.auxiliary,
    }
    for key, matrix in actual.items():
        np.testing.assert_allclose(
            matrix, expected[key], atol=INTEGRAL_TOL, err_msg=key
        )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind"), GRADIENT_CASES)
def test_hartree_fock_gradient_matches_pyscf(
    name: str, basis_name: str, kind: str
) -> None:
    from tests.reference.pyscf_oracle import scf_gradient_reference

    molecule = MOLECULES[name]
    basis, auxiliary = bases(molecule, basis_name)
    expected = scf_gradient_reference(molecule, basis, kind, auxiliary)
    np.testing.assert_allclose(
        gradient_for(name, basis_name, kind), expected, atol=GRADIENT_TOL
    )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind"), STORED_CASES)
def test_hartree_fock_gradient_matches_stored_data(
    name: str, basis_name: str, kind: str
) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data") / f"gradient_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["method"] == kind
    assert stored["provenance"]["auxiliary"] == AUXILIARY
    expected: list[list[float]] = stored["gradient"]
    np.testing.assert_allclose(
        gradient_for(name, basis_name, kind), np.asarray(expected), atol=GRADIENT_TOL
    )

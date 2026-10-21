"""One-electron integrals against libcint through PySCF, and stored data."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.backend import Array
from hafnium.basis import BasisLibrary, BasisSet
from hafnium.integrals import RustEngine, kinetic, nuclear_attraction, overlap
from hafnium.molecule import Molecule
from tests.tolerances import INTEGRAL_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()

MOLECULES = {
    "water": Molecule.from_atoms(
        [
            ("O", (0.0, 0.0, 0.1173)),
            ("H", (0.0, 0.7572, -0.4692)),
            ("H", (0.0, -0.7572, -0.4692)),
        ]
    ),
    "ammonia_tilted": Molecule.from_atoms(
        [
            ("N", (0.1, -0.05, 0.0)),
            ("H", (0.9, 0.3, 0.4)),
            ("H", (-0.5, 0.8, -0.2)),
            ("H", (-0.4, -0.7, 0.6)),
        ]
    ),
}
CASES = [
    ("water", "sto-3g"),
    ("water", "def2-svp"),
    ("water", "cc-pvdz"),
    ("water", "def2-universal-jkfit"),
    ("ammonia_tilted", "cc-pvdz"),
    ("ammonia_tilted", "def2-universal-jkfit"),
]
# Recorded in tests/reference/data; the largest case is left out to keep the
# repository small and is covered by the live comparison only.
STORED_CASES = CASES[:-1]


def compute(name: str, basis_name: str) -> tuple[Molecule, BasisSet, dict[str, Array]]:
    molecule = MOLECULES[name]
    basis = BasisSet.for_molecule(molecule, basis_name, library=LIBRARY)
    matrices = {
        "overlap": overlap(ENGINE, basis, molecule),
        "kinetic": kinetic(ENGINE, basis, molecule),
        "nuclear_attraction": nuclear_attraction(ENGINE, basis, molecule),
    }
    return molecule, basis, matrices


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), CASES)
def test_one_electron_integrals_match_libcint(name: str, basis_name: str) -> None:
    from tests.reference.pyscf_oracle import one_electron_reference

    molecule, basis, matrices = compute(name, basis_name)
    expected = one_electron_reference(molecule, basis)
    for key, matrix in matrices.items():
        np.testing.assert_allclose(
            matrix, expected[key], atol=INTEGRAL_TOL, err_msg=key
        )


@pytest.mark.reference
@pytest.mark.parametrize("omega", [0.3, 1.1])
def test_attenuated_attraction_matches_gaussian_charges(omega: float) -> None:
    from tests.reference.pyscf_oracle import attenuated_attraction_reference

    molecule = MOLECULES["water"]
    basis = BasisSet.for_molecule(molecule, "cc-pvdz", library=LIBRARY)
    actual = nuclear_attraction(ENGINE, basis, molecule, omega=omega)
    expected = attenuated_attraction_reference(molecule, basis, omega)
    np.testing.assert_allclose(actual, expected, atol=INTEGRAL_TOL)


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), STORED_CASES)
def test_one_electron_integrals_match_stored_data(name: str, basis_name: str) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data")
        / f"one_electron_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    _, _, matrices = compute(name, basis_name)
    assert stored["provenance"]["program"] == "pyscf"
    for key, matrix in matrices.items():
        np.testing.assert_allclose(
            matrix, np.asarray(stored[key]), atol=INTEGRAL_TOL, err_msg=key
        )

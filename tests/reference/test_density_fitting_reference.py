"""Density-fitting integrals against libcint through PySCF, and stored data."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.backend import Array
from hafnium.basis import BasisLibrary, BasisSet
from hafnium.integrals import DensityFitting, RustEngine
from hafnium.molecule import Molecule
from tests.reference.test_one_electron_reference import MOLECULES
from tests.tolerances import DENSITY_FITTING_TOL, INTEGRAL_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
AUXILIARY = "def2-universal-jkfit"
CASES = [("water", "sto-3g"), ("water", "def2-svp"), ("ammonia_tilted", "cc-pvdz")]
STORED_CASES = CASES[:1]


def compute(name: str, basis_name: str, omega: float | None = None) -> DensityFitting:
    molecule = MOLECULES[name]
    basis = BasisSet.for_molecule(molecule, basis_name, library=LIBRARY)
    auxiliary = BasisSet.for_molecule(molecule, AUXILIARY, library=LIBRARY)
    return DensityFitting.build(ENGINE, basis, auxiliary, molecule, omega=omega)


def bases(name: str, basis_name: str) -> tuple[Molecule, BasisSet, BasisSet]:
    molecule = MOLECULES[name]
    return (
        molecule,
        BasisSet.for_molecule(molecule, basis_name, library=LIBRARY),
        BasisSet.for_molecule(molecule, AUXILIARY, library=LIBRARY),
    )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), CASES)
@pytest.mark.parametrize("omega", [None, 0.3])
def test_density_fitting_integrals_match_libcint(
    name: str, basis_name: str, omega: float | None
) -> None:
    from tests.reference.pyscf_oracle import density_fitting_reference

    fitting = compute(name, basis_name, omega)
    molecule, basis, auxiliary = bases(name, basis_name)
    expected = density_fitting_reference(molecule, basis, auxiliary, omega)
    np.testing.assert_allclose(fitting.metric, expected["metric"], atol=INTEGRAL_TOL)
    np.testing.assert_allclose(
        fitting.three_center, expected["three_center"], atol=INTEGRAL_TOL
    )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), CASES[:2])
def test_fitted_repulsion_matches_pyscf(name: str, basis_name: str) -> None:
    from tests.reference.pyscf_oracle import fitted_repulsion_reference

    fitting = compute(name, basis_name)
    molecule, basis, auxiliary = bases(name, basis_name)
    expected = fitted_repulsion_reference(molecule, basis, auxiliary)
    np.testing.assert_allclose(
        fitting.electron_repulsion(), expected, atol=DENSITY_FITTING_TOL
    )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), STORED_CASES)
def test_density_fitting_integrals_match_stored_data(
    name: str, basis_name: str
) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data")
        / f"density_fitting_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    fitting = compute(name, basis_name)
    assert stored["provenance"]["program"] == "pyscf"
    matrices: dict[str, Array] = {
        "metric": fitting.metric,
        "three_center": fitting.three_center,
    }
    for key, matrix in matrices.items():
        expected: list[Any] = stored[key]
        np.testing.assert_allclose(
            matrix, np.asarray(expected), atol=INTEGRAL_TOL, err_msg=key
        )

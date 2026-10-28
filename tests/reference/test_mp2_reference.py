"""DF-MP2 energies against PySCF's density-fitted MP2, and stored data."""

import json
from importlib import resources
from typing import Any

import pytest

from hafnium.basis import BasisSet
from hafnium.molecule import Molecule
from hafnium.mp import Mp2Result, df_mp2
from hafnium.scf import DensityFittedCoulombExchange, Hamiltonian, ScfOptions, solve_scf
from hafnium.symmetry import SymmetrySpec
from tests.hamiltonians import ENGINE
from tests.reference.test_scf_reference import MOLECULES
from tests.shell_tables import LIBRARY
from tests.tolerances import ENERGY_TOL, MP2_ENERGY_TOL

AUXILIARY = "def2-universal-jkfit"
# (molecule, basis, class): closed shells in the restricted and generalized
# classes, the doublet cation in the unrestricted and generalized ones.
CASES = [
    ("water", "sto-3g", "rhf"),
    ("water", "def2-svp", "rhf"),
    ("water", "def2-svp", "ghf"),
    ("ammonia_tilted", "def2-svp", "rhf"),
    ("water_cation", "sto-3g", "uhf"),
    ("water_cation", "def2-svp", "uhf"),
    ("water_cation", "def2-svp", "ghf"),
]
STORED_CASES = [CASES[0], CASES[4]]
# The second-order energy is first order in the orbital error: both codes
# converge the reference to a gradient of 1e-9 or better before comparing.
TIGHT = ScfOptions(energy_tolerance=1e-13, gradient_tolerance=1e-9)


def compute(
    name: str, basis_name: str, kind: str
) -> tuple[Molecule, BasisSet, BasisSet, Mp2Result]:
    molecule = MOLECULES[name]
    basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
    auxiliary = BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY)
    two_body = DensityFittedCoulombExchange.from_molecule(
        ENGINE, basis, auxiliary, molecule
    )
    hamiltonian = Hamiltonian.from_molecule(ENGINE, basis, molecule, two_body)
    reference = solve_scf(
        hamiltonian, SymmetrySpec.from_name(kind.upper()), options=TIGHT
    )
    return (
        molecule,
        basis,
        auxiliary,
        df_mp2(ENGINE, basis, molecule, auxiliary, reference),
    )


def check_against(result: Mp2Result, expected: dict[str, Any]) -> None:
    assert result.reference_energy == pytest.approx(
        expected["reference_energy"], abs=ENERGY_TOL
    )
    assert result.correlation_energy == pytest.approx(
        expected["correlation_energy"], abs=MP2_ENERGY_TOL
    )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind"), CASES)
def test_df_mp2_matches_pyscf(name: str, basis_name: str, kind: str) -> None:
    from tests.reference.pyscf_oracle import mp2_reference

    molecule, basis, auxiliary, result = compute(name, basis_name, kind)
    check_against(result, mp2_reference(molecule, basis, auxiliary, kind))


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind"), STORED_CASES)
def test_df_mp2_matches_stored_data(name: str, basis_name: str, kind: str) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data") / f"mp2_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["method"] == kind
    assert stored["provenance"]["auxiliary"] == AUXILIARY
    _, _, _, result = compute(name, basis_name, kind)
    check_against(result, stored)

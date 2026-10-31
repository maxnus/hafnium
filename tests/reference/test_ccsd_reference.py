"""DF-CCSD energies against PySCF's density-fitted CCSD, and stored data."""

import json
from importlib import resources
from typing import Any

import pytest

from hafnium.basis import BasisSet
from hafnium.cc import CcsdOptions, CcsdResult, df_ccsd
from hafnium.molecule import Molecule
from hafnium.scf import DensityFittedCoulombExchange, Hamiltonian, ScfOptions, solve_scf
from hafnium.symmetry import SymmetrySpec
from tests.hamiltonians import ENGINE
from tests.reference.test_scf_reference import MOLECULES
from tests.shell_tables import LIBRARY
from tests.tolerances import CCSD_ENERGY_TOL, ENERGY_TOL

AUXILIARY = "def2-universal-jkfit"
# (molecule, basis, class): closed shells in the restricted and generalized
# classes, the doublet cation in the unrestricted and generalized ones.
CASES = [
    ("water", "sto-3g", "rhf"),
    ("water", "sto-3g", "ghf"),
    ("water", "def2-svp", "rhf"),
    ("water_cation", "sto-3g", "uhf"),
    ("water_cation", "sto-3g", "ghf"),
    ("water_cation", "def2-svp", "uhf"),
]
STORED_CASES = [CASES[0], CASES[3]]
# Both codes converge the reference to a gradient of 1e-9 or better and the
# amplitudes to 1e-10 before comparing.
TIGHT = ScfOptions(energy_tolerance=1e-13, gradient_tolerance=1e-9)
OPTIONS = CcsdOptions(energy_tolerance=1e-12, amplitude_tolerance=1e-10)


def compute(
    name: str, basis_name: str, kind: str
) -> tuple[Molecule, BasisSet, BasisSet, CcsdResult]:
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
        df_ccsd(ENGINE, basis, molecule, auxiliary, reference, options=OPTIONS),
    )


def check_against(result: CcsdResult, expected: dict[str, Any]) -> None:
    assert result.reference_energy == pytest.approx(
        expected["reference_energy"], abs=ENERGY_TOL
    )
    assert result.correlation_energy == pytest.approx(
        expected["correlation_energy"], abs=CCSD_ENERGY_TOL
    )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind"), CASES)
def test_df_ccsd_matches_pyscf(name: str, basis_name: str, kind: str) -> None:
    from tests.reference.pyscf_oracle import ccsd_reference

    molecule, basis, auxiliary, result = compute(name, basis_name, kind)
    check_against(result, ccsd_reference(molecule, basis, auxiliary, kind))


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind"), STORED_CASES)
def test_df_ccsd_matches_stored_data(name: str, basis_name: str, kind: str) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data") / f"ccsd_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["method"] == kind
    assert stored["provenance"]["auxiliary"] == AUXILIARY
    _, _, _, result = compute(name, basis_name, kind)
    check_against(result, stored)

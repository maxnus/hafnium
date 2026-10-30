"""k-point DF-CCSD energies per cell against PySCF's KRCCSD, and stored data."""

import json
from importlib import resources
from typing import Any

import pytest

from hafnium.basis import BasisSet
from hafnium.cc import CcsdOptions, CcsdResult, ccsd_from_periodic_fitting
from hafnium.molecule import Cell
from hafnium.scf import Hamiltonian, ScfOptions, atomic_density_guess, solve_scf
from hafnium.symmetry import KPoints, SymmetrySpec
from tests.hamiltonians import ENGINE
from tests.integrals_k import fitting_and_two_body
from tests.reference.test_kscf_reference import AUXILIARY, CELLS, MESHES
from tests.shell_tables import LIBRARY
from tests.tolerances import CCSD_ENERGY_TOL, ENERGY_TOL

# (cell, basis, kind): a closed shell in the restricted class over three
# k-points. PySCF's KUCCSD and KGCCSD drop the Ewald exchange correction
# from the Fock matrix without an option to keep it (see
# ``kccsd_reference``), so the open-shell classes are checked against each
# other and by band folding in the unit tests.
CASES = [("helium_triclinic", "cc-pvdz", "krhf")]
STORED_CASES = CASES
TIGHT = ScfOptions(energy_tolerance=1e-13, gradient_tolerance=1e-9)
OPTIONS = CcsdOptions(energy_tolerance=1e-12, amplitude_tolerance=1e-10)


def compute(
    name: str, basis_name: str, kind: str
) -> tuple[Cell, BasisSet, BasisSet, KPoints, CcsdResult]:
    cell = CELLS[name]()
    basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
    auxiliary = BasisSet.for_atoms(cell, AUXILIARY, library=LIBRARY)
    mesh = MESHES[name]
    fitting, two_body = fitting_and_two_body(ENGINE, basis, auxiliary, cell, mesh)
    hamiltonian = Hamiltonian.from_cell(ENGINE, basis, cell, mesh, two_body)
    symmetry = SymmetrySpec.from_name(kind[1:].upper()).with_kpoints(mesh)
    guess = atomic_density_guess(ENGINE, basis, symmetry.structure())
    reference = solve_scf(hamiltonian, symmetry, guess=guess, options=TIGHT)
    return (
        cell,
        basis,
        auxiliary,
        mesh,
        ccsd_from_periodic_fitting(reference, fitting, options=OPTIONS),
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
def test_k_point_ccsd_matches_pyscf(name: str, basis_name: str, kind: str) -> None:
    from tests.reference.pyscf_oracle import kccsd_reference

    cell, basis, auxiliary, mesh, result = compute(name, basis_name, kind)
    check_against(result, kccsd_reference(cell, basis, auxiliary, mesh, kind))


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind"), STORED_CASES)
def test_k_point_ccsd_matches_stored_data(
    name: str, basis_name: str, kind: str
) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data") / f"kccsd_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["method"] == kind
    _, _, _, _, result = compute(name, basis_name, kind)
    check_against(result, stored)

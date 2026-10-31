"""k-point DF-CCSD(T) energies per cell against PySCF's, and stored data."""

import json
from importlib import resources
from typing import Any

import pytest

from hafnium.basis import BasisSet
from hafnium.cc import CcsdOptions, CcsdPtResult, ccsd_pt_from_periodic_fitting
from hafnium.integrals import density_fitting_k
from hafnium.integrals.periodic import overlap_k
from hafnium.integrals.periodic.exchange import madelung_constant
from hafnium.molecule import Cell
from hafnium.scf import (
    Hamiltonian,
    KPointDensityFittedCoulombExchange,
    ScfOptions,
    atomic_density_guess,
    solve_scf,
)
from hafnium.symmetry import KPoints, SymmetrySpec
from tests.hamiltonians import ENGINE
from tests.reference.test_kscf_reference import AUXILIARY, CELLS, MESHES
from tests.shell_tables import LIBRARY
from tests.tolerances import CCSD_ENERGY_TOL, ENERGY_TOL

# (cell, basis, kind, exchange correction): a closed shell over three
# k-points in the three classes. PySCF's KRCCSD can keep the Ewald exchange
# correction of the mean field in the Fock matrix; KUCCSD and KGCCSD drop
# it without an option to keep it (see ``kccsd_reference``), so those two
# are compared with the correction switched off on both sides.
CASES = [
    ("helium_triclinic", "cc-pvdz", "krhf", True),
    ("helium_triclinic", "cc-pvdz", "kuhf", False),
    ("helium_triclinic", "cc-pvdz", "kghf", False),
]
STORED_CASES = CASES
TIGHT = ScfOptions(energy_tolerance=1e-13, gradient_tolerance=1e-9)
OPTIONS = CcsdOptions(energy_tolerance=1e-12, amplitude_tolerance=1e-10)


def compute(
    name: str, basis_name: str, kind: str, exchange_correction: bool
) -> tuple[Cell, BasisSet, BasisSet, KPoints, CcsdPtResult]:
    cell = CELLS[name]()
    basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
    auxiliary = BasisSet.for_atoms(cell, AUXILIARY, library=LIBRARY)
    mesh = MESHES[name]
    fitting = density_fitting_k(ENGINE, basis, auxiliary, cell, mesh)
    two_body = KPointDensityFittedCoulombExchange.from_fitting(
        fitting,
        overlap_k(ENGINE, basis, cell, mesh),
        madelung_constant(cell.lattice, mesh) if exchange_correction else 0.0,
    )
    hamiltonian = Hamiltonian.from_cell(ENGINE, basis, cell, mesh, two_body)
    symmetry = SymmetrySpec.from_name(kind[1:].upper()).with_kpoints(mesh)
    guess = atomic_density_guess(ENGINE, basis, symmetry.structure())
    reference = solve_scf(hamiltonian, symmetry, guess=guess, options=TIGHT)
    return (
        cell,
        basis,
        auxiliary,
        mesh,
        ccsd_pt_from_periodic_fitting(reference, fitting, options=OPTIONS),
    )


def check_against(result: CcsdPtResult, expected: dict[str, Any]) -> None:
    assert result.reference_energy == pytest.approx(
        expected["reference_energy"], abs=ENERGY_TOL
    )
    assert result.correlation_energy == pytest.approx(
        expected["correlation_energy"], abs=CCSD_ENERGY_TOL
    )
    # PySCF has no (T) for KUCCSD; the unit tests check that class against
    # the others.
    if "triples_correction" in expected:
        assert result.triples_correction == pytest.approx(
            expected["triples_correction"], abs=CCSD_ENERGY_TOL
        )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind", "exchange_correction"), CASES)
def test_k_point_ccsd_matches_pyscf(
    name: str, basis_name: str, kind: str, exchange_correction: bool
) -> None:
    from tests.reference.pyscf_oracle import kccsd_reference

    cell, basis, auxiliary, mesh, result = compute(
        name, basis_name, kind, exchange_correction
    )
    check_against(
        result,
        kccsd_reference(
            cell, basis, auxiliary, mesh, kind, exchange_correction=exchange_correction
        ),
    )


@pytest.mark.reference
@pytest.mark.parametrize(
    ("name", "basis_name", "kind", "exchange_correction"), STORED_CASES
)
def test_k_point_ccsd_matches_stored_data(
    name: str, basis_name: str, kind: str, exchange_correction: bool
) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data")
        / f"kccsd_{name}_{basis_name}_{kind}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["method"] == kind
    assert stored["provenance"]["exchange_correction"] == exchange_correction
    _, _, _, _, result = compute(name, basis_name, kind, exchange_correction)
    check_against(result, stored)

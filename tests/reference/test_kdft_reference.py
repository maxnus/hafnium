"""Periodic Kohn–Sham against PySCF's KRKS and KUKS (ADR-0027)."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.grid import GridSpec, MolecularGrid
from hafnium.molecule import Cell
from hafnium.scf import (
    ExchangeCorrelation,
    Hamiltonian,
    KohnShamOptions,
    KPointDensityFittedCoulombExchange,
    ScfResult,
    atomic_density_guess,
    kohn_sham,
    solve_scf,
)
from hafnium.symmetry import SymmetrySpec
from tests.hamiltonians import ENGINE
from tests.reference.pyscf_oracle import kdft_reference
from tests.reference.test_dft_reference import functional_for
from tests.reference.test_kscf_reference import AUXILIARY, CELLS, MESHES
from tests.shell_tables import LIBRARY
from tests.tolerances import (
    KOHN_SHAM_ENERGY_TOL,
    MO_ENERGY_TOL,
    PERIODIC_GRID_ENERGY_TOL,
    REFERENCE_GRADIENT_TOL,
)

# (cell, basis, class, Hafnium name, PySCF code): closed shells in the
# restricted class, the doublet in the unrestricted one; the slab exercises
# the reduced-dimension grid against PySCF's grid in its padded box.
CASES = [
    ("helium_triclinic", "cc-pvdz", "krks", "svwn", "LDA,VWN"),
    ("helium_triclinic", "cc-pvdz", "krks", "pbe", "PBE"),
    ("helium_triclinic", "cc-pvdz", "krks", "b3lyp", "B3LYP"),
    ("helium_triclinic", "cc-pvdz", "krks", "tpss", "TPSS"),
    ("helium_hydride_box", "cc-pvdz", "kuks", "pbe", "PBE"),
    ("helium_slab", "cc-pvdz", "krks", "pbe", "PBE"),
]
# The stored case integrates on PySCF's level-0 grid, small enough to record.
STORED_CASES = [("helium_triclinic", "cc-pvdz", "krks", "pbe", "PBE")]
STORED_LEVEL = 0
SYMMETRY = {
    "krks": SymmetrySpec.from_name("RHF"),
    "kuks": SymmetrySpec.from_name("UHF"),
}


def systems(name: str, basis_name: str) -> tuple[Cell, BasisSet, BasisSet]:
    cell = CELLS[name]()
    return (
        cell,
        BasisSet.for_atoms(cell, basis_name, library=LIBRARY),
        BasisSet.for_atoms(cell, AUXILIARY, library=LIBRARY),
    )


def compute(
    name: str,
    basis_name: str,
    kind: str,
    functional: str,
    *,
    grid: MolecularGrid | GridSpec | None = None,
) -> ScfResult:
    """Converge the state; on a prebuilt ``grid`` (PySCF's) through the Hamiltonian."""
    cell, basis, auxiliary = systems(name, basis_name)
    mesh = MESHES[name]
    symmetry = SYMMETRY[kind].with_kpoints(mesh)
    options = KohnShamOptions(
        grid=grid if isinstance(grid, GridSpec) else GridSpec(),
        gradient_tolerance=REFERENCE_GRADIENT_TOL,
    )
    if not isinstance(grid, MolecularGrid):
        return kohn_sham(
            ENGINE,
            basis,
            cell,
            functional_for(functional),
            symmetry=symmetry,
            auxiliary=auxiliary,
            options=options,
        )
    exchange_correlation = ExchangeCorrelation.from_cell(
        ENGINE, basis, cell, mesh, functional_for(functional), grid=grid
    )
    two_body = KPointDensityFittedCoulombExchange.from_cell(
        ENGINE, basis, auxiliary, cell, mesh
    )
    hamiltonian = Hamiltonian.from_cell(
        ENGINE, basis, cell, mesh, two_body, exchange_correlation=exchange_correlation
    )
    guess = atomic_density_guess(ENGINE, basis, hamiltonian.structure(symmetry))
    return solve_scf(hamiltonian, symmetry, guess=guess, options=options)


def pyscf_grid(expected: dict[str, Any]) -> MolecularGrid:
    """PySCF's grid as a Hafnium grid; the owners are not recorded and not used."""
    weights = np.asarray(expected["grid_weights"], dtype=np.float64)
    return MolecularGrid(
        np.asarray(expected["grid_points"], dtype=np.float64),
        weights,
        np.zeros(weights.shape[0], dtype=np.int64),
    )


def check_against(result: ScfResult, expected: dict[str, Any]) -> None:
    assert result.energy == pytest.approx(expected["energy"], abs=KOHN_SHAM_ENERGY_TOL)
    # Orbital energies at every k-point of the full set, sorted per point;
    # PySCF stacks the unrestricted ones as [2, kpt, mo].
    ours = np.sort(np.asarray(result.mo_energies.materialized()[0]), axis=-1)
    if result.structure.is_collinear and result.structure.derived_partner is None:
        beta = np.sort(np.asarray(result.mo_energies.materialized()[1]), axis=-1)
        ours = np.concatenate([ours, beta])
    theirs = np.asarray(expected["mo_energies"])
    theirs = np.sort(theirs.reshape(-1, theirs.shape[-1]), axis=-1)
    np.testing.assert_allclose(ours, theirs, atol=MO_ENERGY_TOL)


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind", "functional", "code"), CASES)
def test_periodic_kohn_sham_matches_pyscf_on_its_grid(
    name: str, basis_name: str, kind: str, functional: str, code: str
) -> None:
    """On one and the same grid the two codes agree to the SCF tolerance."""
    cell, basis, auxiliary = systems(name, basis_name)
    expected = kdft_reference(cell, basis, auxiliary, MESHES[name], code, kind)
    check_against(
        compute(name, basis_name, kind, functional, grid=pyscf_grid(expected)), expected
    )


@pytest.mark.reference
@pytest.mark.parametrize(
    ("name", "basis_name", "kind", "functional", "code"), CASES[1:2] + CASES[4:5]
)
def test_periodic_grid_energies_are_close_to_pyscf(
    name: str, basis_name: str, kind: str, functional: str, code: str
) -> None:
    """Hafnium's grid and PySCF's truncated one agree to PySCF's grid error."""
    cell, basis, auxiliary = systems(name, basis_name)
    expected = kdft_reference(cell, basis, auxiliary, MESHES[name], code, kind)
    result = compute(name, basis_name, kind, functional)
    assert result.energy == pytest.approx(
        expected["energy"], abs=PERIODIC_GRID_ENERGY_TOL
    )


@pytest.mark.reference
@pytest.mark.parametrize(
    ("name", "basis_name", "kind", "functional", "code"), STORED_CASES
)
def test_periodic_kohn_sham_matches_stored_data(
    name: str, basis_name: str, kind: str, functional: str, code: str
) -> None:
    """Regression against values and a grid recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data") / f"kdft_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["method"] == kind
    assert stored["provenance"]["functional"] == code
    fractional: list[list[float]] = stored["provenance"]["kpoints_fractional"]
    np.testing.assert_allclose(
        np.asarray(fractional), np.asarray(MESHES[name].fractional)
    )
    check_against(
        compute(name, basis_name, kind, functional, grid=pyscf_grid(stored)), stored
    )

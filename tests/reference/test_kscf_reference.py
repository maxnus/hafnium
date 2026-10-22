"""k-point Hartree–Fock energies and orbital energies against PySCF, and stored data."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.basis import BasisLibrary, BasisSet
from hafnium.integrals import RustEngine
from hafnium.molecule import Cell, Molecule
from hafnium.scf import (
    Hamiltonian,
    KPointDensityFittedCoulombExchange,
    ScfOptions,
    ScfResult,
    atomic_density_guess,
    solve_scf,
)
from hafnium.symmetry import KPoints, SymmetrySpec
from tests.cells import box, helium, helium_slab
from tests.tolerances import ENERGY_TOL, MO_ENERGY_TOL, SPIN_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
AUXILIARY = "def2-universal-jkfit"


def helium_hydride_box() -> Cell:
    """HeH, a doublet, in a cubic box of 8 Bohr.

    A cell with beta electrons, since PySCF's ``KUHF`` fails in ``get_occ``
    on a cell without any; the box is wide enough for integer occupations
    across the mesh.
    """
    molecule = Molecule.from_atoms(
        [("He", (0.0, 0.0, 0.0)), ("H", (1.5, 0.0, 0.0))], unit="bohr", spin=1
    )
    return box(molecule, 8.0)


CELLS = {
    "helium_triclinic": lambda: helium(3),
    "helium_hydride_box": helium_hydride_box,
    "helium_slab": helium_slab,
}
MESHES = {
    "helium_triclinic": KPoints.mesh((3, 1, 1)),
    "helium_hydride_box": KPoints.mesh((2, 2, 2)),
    "helium_slab": KPoints.mesh((2, 2)),
}
# (cell, basis, kind): the closed shells in the restricted class, the doublet
# in the unrestricted and generalized classes; the slab exercises the
# reduced-dimension kernels (ADR-0021) against PySCF's two-dimensional ones.
CASES = [
    ("helium_triclinic", "cc-pvdz", "krhf"),
    ("helium_hydride_box", "cc-pvdz", "kuhf"),
    ("helium_hydride_box", "cc-pvdz", "kghf"),
    ("helium_slab", "cc-pvdz", "krhf"),
]
STORED_CASES = [case for case in CASES if case[2] != "kghf"]
# Converged to the gradient tolerance of the reference, so that the orbital
# energies, first order in the residual orbital error, compare at MO_ENERGY_TOL.
OPTIONS = ScfOptions(gradient_tolerance=1e-8)
SYMMETRY = {
    "krhf": SymmetrySpec.from_name("RHF"),
    "kuhf": SymmetrySpec.from_name("UHF"),
    "kghf": SymmetrySpec.from_name("GHF"),
}


def compute(
    name: str, basis_name: str, kind: str
) -> tuple[Cell, BasisSet, BasisSet, ScfResult]:
    cell = CELLS[name]()
    basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
    auxiliary = BasisSet.for_atoms(cell, AUXILIARY, library=LIBRARY)
    mesh = MESHES[name]
    two_body = KPointDensityFittedCoulombExchange.from_cell(
        ENGINE, basis, auxiliary, cell, mesh
    )
    hamiltonian = Hamiltonian.from_cell(ENGINE, basis, cell, mesh, two_body)
    symmetry = SYMMETRY[kind].with_kpoints(mesh)
    guess = atomic_density_guess(ENGINE, basis, symmetry.structure())
    if kind == "kghf":
        # From a spin-averaged guess the generalized class converges to a
        # higher non-magnetic stationary state, PySCF's KGHF too: both start
        # from the unrestricted solution.
        unrestricted = SYMMETRY["kuhf"].with_kpoints(mesh)
        guess = solve_scf(
            hamiltonian,
            unrestricted,
            guess=atomic_density_guess(ENGINE, basis, unrestricted.structure()),
            options=OPTIONS,
        ).density.embed(symmetry.structure())
    result = solve_scf(hamiltonian, symmetry, guess=guess, options=OPTIONS)
    return cell, basis, auxiliary, result


def check_against(result: ScfResult, expected: dict[str, Any], n_kpt: int) -> None:
    assert result.energy == pytest.approx(expected["energy"], abs=ENERGY_TOL)
    # Orbital energies at every k-point of the full set, sorted per point.
    ours = np.sort(np.asarray(result.mo_energies.materialized()[0]), axis=-1)
    theirs = np.asarray(expected["mo_energies"])
    theirs = np.sort(theirs.reshape(-1, theirs.shape[-1]), axis=-1)
    if result.structure.is_collinear and result.structure.derived_partner is None:
        # Unrestricted: alpha and beta energies, PySCF stacks them as [2, kpt, mo].
        beta = np.sort(np.asarray(result.mo_energies.materialized()[1]), axis=-1)
        ours = np.concatenate([ours, beta])
    np.testing.assert_allclose(ours, theirs, atol=MO_ENERGY_TOL)
    # PySCF reports ⟨S²⟩ of the supercell of the mesh, ours is per cell
    # (ADR-0020); the two-body part scales with the number of cells, the
    # square of the total spin with its square.
    magnitude = float(np.sum(np.asarray(result.spin.vector) ** 2))
    supercell = n_kpt * (result.spin.square - magnitude) + n_kpt**2 * magnitude
    assert supercell == pytest.approx(expected["spin_square_supercell"], abs=SPIN_TOL)


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind"), CASES)
def test_k_point_hartree_fock_matches_pyscf(
    name: str, basis_name: str, kind: str
) -> None:
    from tests.reference.pyscf_oracle import kscf_reference

    cell, basis, auxiliary, result = compute(name, basis_name, kind)
    mesh = MESHES[name]
    expected = kscf_reference(cell, basis, auxiliary, mesh, kind)
    check_against(result, expected, mesh.n_kpt)


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind"), STORED_CASES)
def test_k_point_hartree_fock_matches_stored_data(
    name: str, basis_name: str, kind: str
) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data") / f"kscf_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["method"] == kind
    fractional: list[list[float]] = stored["provenance"]["kpoints_fractional"]
    np.testing.assert_allclose(
        np.asarray(fractional), np.asarray(MESHES[name].fractional)
    )
    _, _, _, result = compute(name, basis_name, kind)
    check_against(result, stored, MESHES[name].n_kpt)

"""Periodic nuclear attraction and Madelung constant against PySCF, and stored data."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.backend import Array
from hafnium.basis import BasisLibrary, BasisSet
from hafnium.integrals import RustEngine, madelung_constant, nuclear_attraction_k
from hafnium.molecule import Cell
from hafnium.symmetry import KPoints
from tests.cells import helium, lithium_hydride
from tests.tolerances import ENERGY_TOL, PERIODIC_ATTRACTION_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()

CELLS = {
    "helium_triclinic": lambda: helium(3),
    "lithium_hydride": lithium_hydride,
}
MESHES = {
    "helium_triclinic": KPoints.mesh((2, 2, 2), shift=(0.5, 0.5, 0.5)),
    "lithium_hydride": KPoints.mesh((2, 2, 2)),
}
CASES = [("helium_triclinic", "cc-pvdz"), ("lithium_hydride", "sto-3g")]
# Recorded in tests/reference/data; the rest is covered by the live comparison.
STORED_CASES = CASES[:1]


def compute(name: str, basis_name: str) -> tuple[Cell, BasisSet, KPoints, Array]:
    cell = CELLS[name]()
    basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
    mesh = MESHES[name]
    return cell, basis, mesh, nuclear_attraction_k(ENGINE, basis, cell, mesh)


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), CASES)
def test_nuclear_attraction_matches_pyscf(name: str, basis_name: str) -> None:
    from tests.reference.pyscf_oracle import periodic_nuclear_attraction_reference

    cell, basis, mesh, matrix = compute(name, basis_name)
    expected = periodic_nuclear_attraction_reference(cell, basis, mesh)
    np.testing.assert_allclose(matrix, expected, atol=PERIODIC_ATTRACTION_TOL)


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), CASES)
def test_madelung_constant_matches_pyscf(name: str, basis_name: str) -> None:
    from tests.reference.pyscf_oracle import madelung_reference

    cell = CELLS[name]()
    basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
    for mesh in (MESHES[name], KPoints.mesh((3, 1, 2))):
        expected = madelung_reference(cell, basis, mesh)
        assert madelung_constant(cell.lattice, mesh) == pytest.approx(
            expected, abs=ENERGY_TOL
        )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), STORED_CASES)
def test_nuclear_attraction_matches_stored_data(name: str, basis_name: str) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data") / f"nuclear_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    cell, _, mesh, matrix = compute(name, basis_name)
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["dimension"] == cell.dimension
    fractional: list[list[float]] = stored["provenance"]["kpoints_fractional"]
    np.testing.assert_allclose(np.asarray(fractional), np.asarray(mesh.fractional))
    real: list[list[list[float]]] = stored["nuclear_attraction_real"]
    imag: list[list[list[float]]] = stored["nuclear_attraction_imag"]
    expected = np.asarray(real) + 1j * np.asarray(imag)
    np.testing.assert_allclose(matrix, expected, atol=PERIODIC_ATTRACTION_TOL)
    assert madelung_constant(cell.lattice, mesh) == pytest.approx(
        stored["madelung"], abs=ENERGY_TOL
    )

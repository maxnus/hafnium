"""Bloch-summed overlap and kinetic matrices against PySCF, and stored data."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.backend import Array
from hafnium.basis import BasisLibrary, BasisSet
from hafnium.integrals import RustEngine, kinetic_k, overlap_k
from hafnium.molecule import Cell
from hafnium.symmetry import KPoints
from tests.cells import helium, lithium_hydride
from tests.tolerances import INTEGRAL_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()

CELLS = {
    "helium_triclinic": lambda: helium(3),
    "helium_slab": lambda: helium(2),
    "helium_chain": lambda: helium(1),
    "lithium_hydride": lithium_hydride,
}
MESHES = {
    "helium_triclinic": KPoints.mesh((2, 2, 2), shift=(0.5, 0.5, 0.5)),
    "helium_slab": KPoints.mesh((2, 3)),
    "helium_chain": KPoints.mesh((3,)),
    "lithium_hydride": KPoints.mesh((2, 2, 2)),
}
CASES = [
    ("helium_triclinic", "cc-pvdz"),
    ("helium_chain", "cc-pvdz"),
    ("helium_slab", "cc-pvdz"),
    ("lithium_hydride", "sto-3g"),
]
# Recorded in tests/reference/data; the rest is covered by the live comparison.
STORED_CASES = CASES[:2]


def compute(
    name: str, basis_name: str
) -> tuple[Cell, BasisSet, KPoints, dict[str, Array]]:
    cell = CELLS[name]()
    basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
    mesh = MESHES[name]
    matrices = {
        "overlap": overlap_k(ENGINE, basis, cell, mesh),
        "kinetic": kinetic_k(ENGINE, basis, cell, mesh),
    }
    return cell, basis, mesh, matrices


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), CASES)
def test_bloch_sums_match_pyscf(name: str, basis_name: str) -> None:
    from tests.reference.pyscf_oracle import bloch_one_electron_reference

    cell, basis, mesh, matrices = compute(name, basis_name)
    expected = bloch_one_electron_reference(cell, basis, mesh)
    for key, matrix in matrices.items():
        np.testing.assert_allclose(
            matrix, expected[key], atol=INTEGRAL_TOL, err_msg=key
        )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), STORED_CASES)
def test_bloch_sums_match_stored_data(name: str, basis_name: str) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data") / f"bloch_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    cell, _, mesh, matrices = compute(name, basis_name)
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["dimension"] == cell.dimension
    fractional: list[list[float]] = stored["provenance"]["kpoints_fractional"]
    np.testing.assert_allclose(np.asarray(fractional), np.asarray(mesh.fractional))
    for key, matrix in matrices.items():
        real: list[list[list[float]]] = stored[f"{key}_real"]
        imag: list[list[list[float]]] = stored[f"{key}_imag"]
        expected = np.asarray(real) + 1j * np.asarray(imag)
        np.testing.assert_allclose(matrix, expected, atol=INTEGRAL_TOL, err_msg=key)

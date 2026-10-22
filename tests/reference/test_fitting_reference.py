"""Fitted periodic integrals against PySCF's density fitting, and stored data."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.basis import BasisLibrary, BasisSet
from hafnium.integrals import PeriodicDensityFitting, RustEngine, density_fitting_k
from hafnium.integrals.periodic import RangeSeparation
from hafnium.molecule import Cell
from hafnium.symmetry import KPoints
from tests.cells import helium, lithium_hydride
from tests.tolerances import (
    ILL_CONDITIONED_FITTING_TOL,
    METRIC_CONDITION_LIMIT,
    PERIODIC_FITTING_TOL,
)

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
AUXILIARY = "def2-universal-jkfit"

CELLS = {
    "helium_triclinic": lambda: helium(3),
    "helium_pair": lambda: helium(3),
    "lithium_hydride": lithium_hydride,
}
MESHES = {
    "helium_triclinic": KPoints.mesh((2, 2, 2), shift=(0.5, 0.5, 0.5)),
    "helium_pair": KPoints.mesh((2, 1, 1)),
    "lithium_hydride": KPoints.mesh((2, 2, 2)),
}
# Quadruples that conserve crystal momentum on the meshes above.
QUADRUPLES = {
    "helium_triclinic": [(0, 0, 0, 0), (0, 1, 1, 0), (1, 3, 5, 7), (2, 1, 0, 3)],
    "helium_pair": [(0, 1, 1, 0)],
    "lithium_hydride": [(0, 1, 1, 0), (2, 1, 0, 3)],
}
# The lattice-sum precision of each case; LiH's diffuse STO-3G functions
# make the default too slow for a test.
PRECISIONS = {"lithium_hydride": 1e-8}
CASES = [
    ("helium_triclinic", "cc-pvdz"),
    ("helium_pair", "cc-pvdz"),
    ("lithium_hydride", "sto-3g"),
]
# Recorded in tests/reference/data; the rest is covered by the live comparison.
STORED_CASES = CASES[1:2]


def compute(
    name: str, basis_name: str
) -> tuple[Cell, BasisSet, BasisSet, KPoints, PeriodicDensityFitting]:
    cell = CELLS[name]()
    basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
    auxiliary = BasisSet.for_atoms(cell, AUXILIARY, library=LIBRARY)
    mesh = MESHES[name]
    separation = RangeSeparation.for_lattice(
        cell.lattice, precision=PRECISIONS.get(name, 1e-12)
    )
    fit = density_fitting_k(ENGINE, basis, auxiliary, cell, mesh, separation=separation)
    return cell, basis, auxiliary, mesh, fit


def tolerance_for(fit: PeriodicDensityFitting) -> float:
    """The agreement a fit can reach: its metric's conditioning decides."""
    condition = min(
        float(
            np.linalg.eigvalsh(fit.metric[q]).min()
            / np.linalg.eigvalsh(fit.metric[q]).max()
        )
        for q in range(fit.transfers.n_kpt)
    )
    return (
        PERIODIC_FITTING_TOL
        if condition > METRIC_CONDITION_LIMIT
        else ILL_CONDITIONED_FITTING_TOL
    )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), CASES)
def test_fitted_integrals_match_pyscf(name: str, basis_name: str) -> None:
    from tests.reference.pyscf_oracle import periodic_fitted_eri_reference

    cell, basis, auxiliary, mesh, fit = compute(name, basis_name)
    expected = periodic_fitted_eri_reference(
        cell, basis, auxiliary, mesh, QUADRUPLES[name]
    )
    tolerance = tolerance_for(fit)
    assert (tolerance == ILL_CONDITIONED_FITTING_TOL) == (name == "lithium_hydride")
    for quadruple in QUADRUPLES[name]:
        key = "".join(str(k) for k in quadruple)
        np.testing.assert_allclose(
            fit.eri(quadruple), expected[key], atol=tolerance, err_msg=key
        )


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), STORED_CASES)
def test_fitted_integrals_match_stored_data(name: str, basis_name: str) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data") / f"fitting_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    cell, _, _, mesh, fit = compute(name, basis_name)
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["auxiliary_basis"] == AUXILIARY
    fractional: list[list[float]] = stored["provenance"]["kpoints_fractional"]
    np.testing.assert_allclose(np.asarray(fractional), np.asarray(mesh.fractional))
    assert cell.dimension == stored["provenance"]["dimension"]
    for quadruple in QUADRUPLES[name]:
        key = "".join(str(k) for k in quadruple)
        real: list[list[list[list[float]]]] = stored[f"eri_{key}_real"]
        imag: list[list[list[list[float]]]] = stored[f"eri_{key}_imag"]
        expected = np.asarray(real) + 1j * np.asarray(imag)
        np.testing.assert_allclose(
            fit.eri(quadruple), expected, atol=PERIODIC_FITTING_TOL, err_msg=key
        )

"""Kohn–Sham energies, grids and functional values against PySCF, and stored data."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.backend import Array
from hafnium.basis import BasisSet
from hafnium.grid import GridSpec, build_grid
from hafnium.molecule import Molecule
from hafnium.scf import KohnShamOptions, ScfResult, kohn_sham
from hafnium.symmetry import SymmetrySpec
from hafnium.xc import DensityOnGrid, Functional, evaluate_functional
from tests.hamiltonians import ENGINE
from tests.reference.test_scf_reference import MOLECULES
from tests.shell_tables import LIBRARY
from tests.tolerances import (
    GRID_POINT_TOL,
    GRID_WEIGHT_FLOOR,
    GRID_WEIGHT_REL_TOL,
    KOHN_SHAM_ENERGY_TOL,
    LIBXC_REL_TOL,
    MO_ENERGY_TOL,
    REFERENCE_GRADIENT_TOL,
    SPIN_TOL,
)
from tests.unit.test_xc import POINTS, density_at

AUXILIARY = "def2-universal-jkfit"
# (molecule, basis, class, Hafnium name, PySCF code): closed shells in the
# restricted class, the doublet cation in the unrestricted one; the ωB97
# functionals run without their VV10 term on both sides.
CASES = [
    ("water", "def2-svp", "rks", "svwn", "LDA,VWN"),
    ("water", "def2-svp", "rks", "pbe", "PBE"),
    ("water", "def2-svp", "rks", "blyp", "BLYP"),
    ("water", "def2-svp", "rks", "b3lyp", "B3LYP"),
    ("water", "def2-svp", "rks", "pbe0", "PBE0"),
    ("water", "def2-svp", "rks", "tpss", "TPSS"),
    ("water", "def2-svp", "rks", "r2scan", "R2SCAN"),
    ("water", "def2-svp", "rks", "m06-l", "M06L"),
    ("water", "def2-svp", "rks", "wb97x-v", "WB97X_V"),
    ("water_cation", "def2-svp", "uks", "pbe", "PBE"),
    ("water_cation", "def2-svp", "uks", "b3lyp", "B3LYP"),
    ("water_cation", "def2-svp", "uks", "tpss", "TPSS"),
    ("water_cation", "def2-svp", "uks", "m06-2x", "M06_2X"),
    ("water_cation", "def2-svp", "uks", "wb97m-v", "WB97M_V"),
]
STORED_CASES = [
    ("water", "sto-3g", "rks", "pbe", "PBE"),
    ("water_cation", "sto-3g", "uks", "b3lyp", "B3LYP"),
]
# Grid settings beside the default: (level, atom grid, prune) as PySCF takes
# them and the matching GridSpec.
GRIDS = [
    ((5, None, True), GridSpec(level=5)),
    (
        (3, (99, 590), False),
        GridSpec(radial_points=99, angular_points=590, prune=False),
    ),
]
LIBXC_NAMES = [
    "lda_x",
    "lda_c_vwn",
    "lda_c_vwn_rpa",
    "lda_c_pw",
    "gga_x_pbe",
    "gga_c_pbe",
    "gga_x_b88",
    "gga_c_lyp",
    "hyb_gga_xc_b3lyp",
    "hyb_gga_xc_pbeh",
    "mgga_x_tpss",
    "mgga_c_tpss",
    "mgga_x_r2scan",
    "mgga_c_r2scan",
    "mgga_x_m06_l",
    "mgga_c_m06_l",
    "hyb_gga_xc_wb97x_v",
]
SYMMETRY = {"rks": SymmetrySpec.from_name("RHF"), "uks": SymmetrySpec.from_name("UHF")}


def functional_for(name: str) -> Functional:
    functional = Functional.named(name)
    return functional.without_vv10() if functional.vv10 is not None else functional


def compute(
    name: str,
    basis_name: str,
    kind: str,
    functional: str,
    *,
    auxiliary: str | None = None,
    grid: GridSpec | None = None,
) -> tuple[Molecule, BasisSet, ScfResult]:
    molecule = MOLECULES[name]
    basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
    result = kohn_sham(
        ENGINE,
        basis,
        molecule,
        functional_for(functional),
        symmetry=SYMMETRY[kind],
        auxiliary=None
        if auxiliary is None
        else BasisSet.for_atoms(molecule, auxiliary, library=LIBRARY),
        options=KohnShamOptions(
            grid=GridSpec() if grid is None else grid,
            gradient_tolerance=REFERENCE_GRADIENT_TOL,
        ),
    )
    return molecule, basis, result


def check_against(result: ScfResult, expected: dict[str, Any]) -> None:
    assert result.energy == pytest.approx(expected["energy"], abs=KOHN_SHAM_ENERGY_TOL)
    mo_energies = result.mo_energies.to_dense()
    if result.structure.derived_partner is not None:
        mo_energies = mo_energies[: mo_energies.shape[0] // 2]
    np.testing.assert_allclose(
        np.sort(mo_energies),
        np.sort(np.asarray(expected["mo_energies"]).ravel()),
        atol=MO_ENERGY_TOL,
    )
    assert result.spin.square == pytest.approx(expected["spin_square"], abs=SPIN_TOL)


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind", "functional", "code"), CASES)
def test_kohn_sham_matches_pyscf(
    name: str, basis_name: str, kind: str, functional: str, code: str
) -> None:
    from tests.reference.pyscf_oracle import dft_reference

    molecule, basis, result = compute(name, basis_name, kind, functional)
    check_against(result, dft_reference(molecule, basis, code, kind))


@pytest.mark.reference
@pytest.mark.parametrize(
    ("functional", "code"), [("b3lyp", "B3LYP"), ("wb97x-v", "WB97X_V")]
)
def test_density_fitted_kohn_sham_matches_pyscf(functional: str, code: str) -> None:
    from tests.reference.pyscf_oracle import dft_reference

    molecule, basis, result = compute(
        "water", "def2-svp", "rks", functional, auxiliary=AUXILIARY
    )
    auxiliary = BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY)
    check_against(
        result, dft_reference(molecule, basis, code, "rks", auxiliary=auxiliary)
    )


@pytest.mark.reference
@pytest.mark.parametrize(("settings", "spec"), GRIDS)
def test_other_grid_sizes_match_pyscf(
    settings: tuple[int, tuple[int, int] | None, bool], spec: GridSpec
) -> None:
    from tests.reference.pyscf_oracle import dft_reference

    molecule, basis, result = compute("water", "def2-svp", "rks", "pbe", grid=spec)
    check_against(result, dft_reference(molecule, basis, "PBE", "rks", grid=settings))


def sorted_grid(points: Array, weights: Array) -> tuple[Array, Array]:
    """The grid in the lexicographic order of its rounded points."""
    order = np.lexsort(np.round(points.T[::-1], 8))
    return points[order], weights[order]


@pytest.mark.reference
@pytest.mark.parametrize("level", [1, 3])
def test_grid_points_and_weights_match_pyscf(level: int) -> None:
    """The grids are built from the same tables, so they hold the same points.

    PySCF orders the points of an atom by the size of their angular rule,
    Hafnium by radial shell, so the two are compared in a common order.
    """
    from tests.reference.pyscf_oracle import grid_reference

    molecule = MOLECULES["ammonia_tilted"]
    grid = build_grid(molecule, GridSpec(level=level))
    expected_points, expected_weights = grid_reference(molecule, level)
    assert grid.n_points == expected_points.shape[0]
    points, weights = sorted_grid(grid.points, grid.weights)
    reference_points, reference_weights = sorted_grid(expected_points, expected_weights)
    np.testing.assert_allclose(points, reference_points, atol=GRID_POINT_TOL)
    large = reference_weights > GRID_WEIGHT_FLOOR
    np.testing.assert_allclose(
        weights[large], reference_weights[large], rtol=GRID_WEIGHT_REL_TOL
    )
    assert float(np.sum(weights)) == pytest.approx(
        float(np.sum(reference_weights)), rel=GRID_WEIGHT_REL_TOL
    )


@pytest.mark.reference
@pytest.mark.parametrize("name", LIBXC_NAMES)
def test_functional_values_match_libxc(name: str) -> None:
    from tests.reference.pyscf_oracle import functional_reference

    density: DensityOnGrid = density_at(POINTS)
    assert density.sigma is not None
    assert density.tau is not None
    values = evaluate_functional(Functional.from_libxc(name), density)
    expected = functional_reference(name, density.rho, density.sigma, density.tau)
    np.testing.assert_allclose(
        values.energy_per_particle, expected.energy_per_particle, rtol=LIBXC_REL_TOL
    )
    np.testing.assert_allclose(values.vrho, expected.vrho, rtol=LIBXC_REL_TOL)
    for ours, theirs in (
        (values.vsigma, expected.vsigma),
        (values.vtau, expected.vtau),
    ):
        assert (ours is None) == (theirs is None)
        if ours is not None and theirs is not None:
            np.testing.assert_allclose(ours, theirs, rtol=LIBXC_REL_TOL)


@pytest.mark.reference
@pytest.mark.parametrize(
    ("name", "basis_name", "kind", "functional", "code"), STORED_CASES
)
def test_kohn_sham_matches_stored_data(
    name: str, basis_name: str, kind: str, functional: str, code: str
) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data") / f"dft_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["method"] == kind
    assert stored["provenance"]["functional"] == code
    _, _, result = compute(name, basis_name, kind, functional)
    check_against(result, stored)

"""Every method gives the same numbers under NumPy and JAX (ADR-0036).

The ``backend`` fixture runs each test under both backends; the NumPy run
is the reference the other is compared with, so a test also checks that
the arrays of a result belong to the backend it ran under.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import pytest

from hafnium.backend import Array, Backend, array_namespace, use_backend
from hafnium.basis import BasisSet
from hafnium.cc import df_ccsd, df_ccsd_lambda, df_ccsd_pt
from hafnium.mp import df_mp2
from hafnium.properties import nuclear_gradient
from hafnium.scf import ScfOptions, ScfResult, hartree_fock, kohn_sham
from hafnium.structured import BlockMatrix, BlockTensor
from hafnium.symmetry import KPoints, SymmetrySpec, detect_point_group
from tests.cells import helium
from tests.hamiltonians import ENGINE
from tests.molecules import water, water_cation
from tests.shell_tables import LIBRARY

T = TypeVar("T")

AUXILIARY = "def2-universal-jkfit"
OPTIONS = ScfOptions(energy_tolerance=1e-13, gradient_tolerance=1e-10)
PARITY_TOL = 1e-10
"""Agreement of the energies of the two backends, in Hartree; the libraries
sum in different orders and the SCF converges to ``gradient_tolerance``."""


_REFERENCES: dict[str, Any] = {}


def under_numpy(key: str, compute: Callable[[], T]) -> T:
    """The NumPy result of ``compute``, computed once per ``key`` for the module."""
    if key not in _REFERENCES:
        with use_backend("numpy"):
            _REFERENCES[key] = compute()
    reference: T = _REFERENCES[key]
    return reference


def molecular(name: str) -> tuple[BasisSet, BasisSet]:
    molecule = water_cation() if name in ("UHF", "complex GHF") else water()
    return (
        BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY),
        BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY),
    )


def assert_belongs(array: Array, backend: Backend) -> None:
    assert array_namespace(array) is backend.namespace


def assert_matrix_belongs(matrix: BlockMatrix | BlockTensor, backend: Backend) -> None:
    for block in matrix.blocks:
        assert_belongs(block, backend)


def scf(
    name: str, *, point_group: bool = False, functional: str | None = None
) -> ScfResult:
    molecule = water_cation() if name in ("UHF", "complex GHF") else water()
    symmetry = SymmetrySpec.from_name(name)
    if point_group:
        detection = detect_point_group(molecule)
        molecule = detection.orient(molecule)
        symmetry = symmetry.with_point_group(detection.group)
    basis = BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY)
    auxiliary = BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY)
    if functional is None:
        return hartree_fock(
            ENGINE,
            basis,
            molecule,
            symmetry=symmetry,
            auxiliary=auxiliary,
            options=OPTIONS,
        )
    return kohn_sham(
        ENGINE,
        basis,
        molecule,
        functional,
        symmetry=symmetry,
        auxiliary=auxiliary,
        options=OPTIONS,
    )


@pytest.mark.parametrize(
    ("name", "point_group", "functional"),
    [
        ("RHF", False, None),
        ("UHF", False, None),
        ("complex GHF", False, None),
        ("RHF", True, None),
        ("RHF", False, "pbe"),
        ("complex GHF", False, "pbe"),
    ],
    ids=["rhf", "uhf", "complex-ghf", "rhf-c2v", "pbe", "ghf-pbe"],
)
def test_mean_field_energies_agree(
    backend: Backend, name: str, point_group: bool, functional: str | None
) -> None:
    def compute() -> ScfResult:
        return scf(name, point_group=point_group, functional=functional)

    expected = under_numpy(f"scf-{name}-{point_group}-{functional}", compute)
    result = compute()
    assert result.converged
    assert result.energy == pytest.approx(expected.energy, abs=PARITY_TOL)
    assert_matrix_belongs(result.density, backend)
    assert_matrix_belongs(result.mo_coefficients, backend)
    for block in result.mo_energies.blocks:
        assert_belongs(block, backend)
    # The spin-summed density is invariant under the spin rotations that
    # relate equivalent generalized solutions.
    np.testing.assert_allclose(
        np.asarray(result.density.spin_trace().to_dense()),
        np.asarray(expected.density.spin_trace().to_dense()),
        atol=1e-8,
    )


@pytest.mark.parametrize("name", ["RHF", "complex GHF"])
def test_correlated_energies_agree(backend: Backend, name: str) -> None:
    def correlated() -> tuple[float, float, float, float, BlockTensor]:
        molecule = water_cation() if name == "complex GHF" else water()
        basis, auxiliary = molecular(name)
        state = scf(name)
        mp2 = df_mp2(ENGINE, basis, molecule, auxiliary, state)
        pt = df_ccsd_pt(ENGINE, basis, molecule, auxiliary, state)
        lam = df_ccsd_lambda(ENGINE, basis, molecule, auxiliary, state)
        return (
            mp2.correlation_energy,
            pt.correlation_energy,
            pt.triples_correction,
            lam.correlation_energy,
            lam.doubles,
        )

    expected = under_numpy(f"correlated-{name}", correlated)
    result = correlated()
    for value, reference_value in zip(result[:4], expected[:4], strict=True):
        assert value == pytest.approx(reference_value, abs=PARITY_TOL)
    assert_matrix_belongs(result[4], backend)


def test_the_gradient_agrees(backend: Backend) -> None:
    def gradient() -> Array:
        molecule = water()
        basis, auxiliary = molecular("RHF")
        state = scf("RHF")
        return nuclear_gradient(ENGINE, basis, auxiliary, molecule, state).total

    expected = under_numpy("gradient", gradient)
    result = gradient()
    assert_belongs(result, backend)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-9)


@pytest.mark.parametrize(
    ("name", "functional"),
    [("RHF", None), ("UHF", None), ("RHF", "pbe")],
    ids=["krhf", "kuhf", "kpbe"],
)
def test_k_point_energies_agree(
    backend: Backend, name: str, functional: str | None
) -> None:
    def periodic() -> tuple[tuple[float, ...], BlockMatrix | BlockTensor]:
        cell = helium(3)
        mesh = KPoints.mesh((3, 1, 1))
        basis = BasisSet.for_atoms(cell, "cc-pvdz", library=LIBRARY)
        auxiliary = BasisSet.for_atoms(cell, AUXILIARY, library=LIBRARY)
        symmetry = SymmetrySpec.from_name(name).with_kpoints(mesh)
        if functional is not None:
            state = kohn_sham(
                ENGINE,
                basis,
                cell,
                functional,
                symmetry=symmetry,
                auxiliary=auxiliary,
                options=OPTIONS,
            )
            return (state.energy,), state.density
        state = hartree_fock(
            ENGINE,
            basis,
            cell,
            symmetry=symmetry,
            auxiliary=auxiliary,
            options=OPTIONS,
        )
        mp2 = df_mp2(ENGINE, basis, cell, auxiliary, state, kpoints=mesh)
        ccsd = df_ccsd(ENGINE, basis, cell, auxiliary, state, kpoints=mesh)
        energies = (state.energy, mp2.correlation_energy, ccsd.correlation_energy)
        return energies, ccsd.doubles

    expected, _ = under_numpy(f"periodic-{name}-{functional}", periodic)
    result, arrays = periodic()
    for value, reference_value in zip(result, expected, strict=True):
        assert value == pytest.approx(reference_value, abs=PARITY_TOL)
    assert_matrix_belongs(arrays, backend)

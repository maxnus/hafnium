"""Fourier transforms against PySCF's analytical ``gto.ft_ao``, and stored data."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.integrals import RustEngine, ShellTable
from tests.reciprocal_vectors import G_VECTORS
from tests.reference.test_four_center_reference import high_angular_momentum_case
from tests.reference.test_one_electron_reference import MOLECULES
from tests.shell_tables import LIBRARY, table_for
from tests.tolerances import INTEGRAL_TOL

ENGINE = RustEngine()
CASES = [("water", "sto-3g"), ("water", "def2-svp"), ("ammonia_tilted", "cc-pvdz")]
STORED_CASES = CASES[:1]


def compute(table: ShellTable) -> dict[str, np.ndarray]:
    return {
        "single": ENGINE.fourier_transform(table, G_VECTORS),
        "pair": ENGINE.pair_fourier_transform(
            table, table, np.zeros((1, 3)), G_VECTORS
        )[0],
    }


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), CASES)
def test_transforms_match_pyscf(name: str, basis_name: str) -> None:
    from tests.reference.pyscf_oracle import fourier_transform_reference

    molecule = MOLECULES[name]
    basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
    expected = fourier_transform_reference(molecule, basis, G_VECTORS)
    actual = compute(table_for(molecule, basis_name))
    for key in ("single", "pair"):
        np.testing.assert_allclose(actual[key], expected[key], atol=INTEGRAL_TOL)


@pytest.mark.reference
def test_mixed_pair_transforms_match_pyscf() -> None:
    from tests.reference.pyscf_oracle import mixed_fourier_transform_reference

    molecule = MOLECULES["water"]
    first = BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY)
    second = BasisSet.for_atoms(molecule, "def2-svp", library=LIBRARY)
    expected = mixed_fourier_transform_reference(molecule, first, second, G_VECTORS)
    actual = ENGINE.pair_fourier_transform(
        table_for(molecule, "sto-3g"),
        table_for(molecule, "def2-svp"),
        np.zeros((1, 3)),
        G_VECTORS,
    )[0]
    assert actual.shape == (len(G_VECTORS), first.n_functions, second.n_functions)
    np.testing.assert_allclose(actual, expected, atol=INTEGRAL_TOL)


@pytest.mark.reference
def test_high_angular_momentum_transforms_match_pyscf() -> None:
    from tests.reference.pyscf_oracle import fourier_transform_reference

    molecule, basis = high_angular_momentum_case()
    expected = fourier_transform_reference(molecule, basis, G_VECTORS)
    actual = compute(ShellTable.from_basis(basis, molecule.positions_array()))
    for key in ("single", "pair"):
        np.testing.assert_allclose(actual[key], expected[key], atol=INTEGRAL_TOL)


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), STORED_CASES)
def test_transforms_match_stored_data(name: str, basis_name: str) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data") / f"fourier_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    g_vectors: list[Any] = stored["g_vectors"]
    assert np.array_equal(np.asarray(g_vectors), G_VECTORS)
    actual = compute(table_for(MOLECULES[name], basis_name))
    for key in ("single", "pair"):
        real: list[Any] = stored[f"{key}_real"]
        imag: list[Any] = stored[f"{key}_imag"]
        expected = np.asarray(real) + 1j * np.asarray(imag)
        np.testing.assert_allclose(
            actual[key], expected, atol=INTEGRAL_TOL, err_msg=key
        )

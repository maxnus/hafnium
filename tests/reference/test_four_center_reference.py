"""Four-centre integrals against libcint through PySCF, and stored data."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.basis import BasisLibrary, BasisSet, Shell
from hafnium.integrals import RustEngine, ShellTable, eri
from hafnium.molecule import Molecule
from tests.reference.test_one_electron_reference import MOLECULES
from tests.tolerances import INTEGRAL_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
CASES = [("water", "sto-3g"), ("water", "def2-svp"), ("ammonia_tilted", "cc-pvdz")]
STORED_CASES = CASES[:1]
# Index patterns over two basis sets: pair swap only, bra and ket swaps
# without the pair swap, and no symmetry at all.
MIXED_PATTERNS = ["abab", "aabb", "abba"]


def compute(name: str, basis_name: str, omega: float | None = None) -> np.ndarray:
    molecule = MOLECULES[name]
    basis = BasisSet.for_molecule(molecule, basis_name, library=LIBRARY)
    return eri(ENGINE, basis, molecule, omega=omega)


def high_angular_momentum_case() -> tuple[Molecule, BasisSet]:
    """d, f and g shells of two primitives each on two centres."""
    molecule = Molecule.from_atoms(
        [("He", (0.0, 0.0, 0.0)), ("He", (0.3, 0.9, 1.4))], unit="bohr"
    )
    shells = tuple(Shell(l, (0.9, 0.35), ((0.6, 0.5),)) for l in (2, 3, 4))
    return molecule, BasisSet("dfg", ("He", "He"), shells * 2, (0, 0, 0, 1, 1, 1))


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), CASES)
@pytest.mark.parametrize("omega", [None, 0.3])
def test_four_center_integrals_match_libcint(
    name: str, basis_name: str, omega: float | None
) -> None:
    from tests.reference.pyscf_oracle import four_center_reference

    molecule = MOLECULES[name]
    basis = BasisSet.for_molecule(molecule, basis_name, library=LIBRARY)
    expected = four_center_reference(molecule, basis, omega)
    actual = compute(name, basis_name, omega)
    np.testing.assert_allclose(actual, expected, atol=INTEGRAL_TOL)


@pytest.mark.reference
def test_high_angular_momentum_matches_libcint() -> None:
    from tests.reference.pyscf_oracle import four_center_reference

    molecule, basis = high_angular_momentum_case()
    expected = four_center_reference(molecule, basis)
    actual = eri(ENGINE, basis, molecule)
    np.testing.assert_allclose(actual, expected, atol=INTEGRAL_TOL)


@pytest.mark.reference
@pytest.mark.parametrize("pattern", MIXED_PATTERNS)
def test_mixed_tables_match_libcint(pattern: str) -> None:
    from tests.reference.pyscf_oracle import mixed_four_center_reference

    molecule = MOLECULES["water"]
    first = BasisSet.for_molecule(molecule, "sto-3g", library=LIBRARY)
    second = BasisSet.for_molecule(molecule, "def2-svp", library=LIBRARY)
    positions = molecule.positions_array()
    tables = {
        "a": ShellTable.from_basis(first, positions),
        "b": ShellTable.from_basis(second, positions),
    }
    bra = (tables[pattern[0]], tables[pattern[1]])
    ket = (tables[pattern[2]], tables[pattern[3]])
    actual = ENGINE.coulomb_four_center(bra, ket, np.zeros((1, 3, 3)))[0]
    expected = mixed_four_center_reference(molecule, first, second, pattern)
    np.testing.assert_allclose(actual, expected, atol=INTEGRAL_TOL)


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name"), STORED_CASES)
def test_four_center_integrals_match_stored_data(name: str, basis_name: str) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data")
        / f"four_center_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    expected: list[Any] = stored["eri"]
    np.testing.assert_allclose(
        compute(name, basis_name), np.asarray(expected), atol=INTEGRAL_TOL
    )

"""Ewald nuclear repulsion against PySCF's ``energy_nuc``, and stored data."""

import json
from importlib import resources
from typing import Any

import pytest

from hafnium.molecule import Cell, Lattice, Molecule
from tests.tolerances import ENERGY_TOL


def triclinic() -> Cell:
    """Three nuclei at general positions in a triclinic cell, in Bohr."""
    molecule = Molecule.from_atoms(
        [("Li", (0.1, 0.2, 0.3)), ("H", (2.1, 1.9, 2.4)), ("O", (1.0, 3.0, 0.5))],
        unit="bohr",
    )
    lattice = Lattice(((4.0, 0.2, 0.1), (-0.3, 3.8, 0.4), (0.5, -0.2, 4.5)))
    return Cell.from_molecule(molecule, lattice)


def slab() -> Cell:
    """Three nuclei above and below an oblique layer in the ``xy`` plane."""
    molecule = Molecule.from_atoms(
        [("Li", (0.1, 0.2, 0.3)), ("H", (2.1, 1.9, -0.6)), ("O", (1.0, 3.0, 0.5))],
        unit="bohr",
    )
    lattice = Lattice(((4.0, 0.2, 0.0), (-0.3, 3.8, 0.0)))
    return Cell.from_molecule(molecule, lattice)


CELLS = {"triclinic": triclinic, "slab": slab}
STORED_CASES = sorted(CELLS)


@pytest.mark.reference
@pytest.mark.parametrize("name", sorted(CELLS))
def test_nuclear_repulsion_matches_pyscf(name: str) -> None:
    from tests.reference.pyscf_oracle import ewald_reference

    cell = CELLS[name]()
    expected = ewald_reference(cell)
    assert cell.nuclear_repulsion_energy() == pytest.approx(expected, abs=ENERGY_TOL)
    omega = 1.7
    assert cell.nuclear_repulsion_energy(omega=omega) == pytest.approx(
        expected, abs=ENERGY_TOL
    )


@pytest.mark.reference
@pytest.mark.parametrize("name", STORED_CASES)
def test_nuclear_repulsion_matches_stored_data(name: str) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = resources.files("tests.reference.data") / f"ewald_{name}_nuclei.json"
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    cell = CELLS[name]()
    assert stored["provenance"]["dimension"] == cell.dimension
    assert cell.nuclear_repulsion_energy() == pytest.approx(
        stored["energy"], abs=ENERGY_TOL
    )

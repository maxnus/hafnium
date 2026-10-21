"""Molecule construction, parsing and derived quantities."""

import numpy as np
import pytest

from hafnium.errors import InputError
from hafnium.molecule import (
    ANGSTROM_IN_BOHR,
    Atom,
    Molecule,
    atomic_number,
    bohr_to_angstrom,
    element_symbol,
)
from tests.tolerances import ENERGY_TOL

WATER_XYZ = """3
water
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


def test_element_lookup_is_case_insensitive() -> None:
    assert atomic_number("cl") == 17
    assert atomic_number(" Hf ") == 72
    assert element_symbol(72) == "Hf"
    assert element_symbol(118) == "Og"
    with pytest.raises(InputError, match="unknown element"):
        atomic_number("Xx")
    with pytest.raises(InputError, match="no element"):
        element_symbol(119)


def test_atom_normalizes_symbol_and_position() -> None:
    atom = Atom("he", (1, 2, 3))
    assert atom.symbol == "He"
    assert atom.position == (1.0, 2.0, 3.0)
    assert atom.atomic_number == 2
    assert atom.nuclear_charge == 2.0


def test_from_xyz_converts_angstrom_to_bohr() -> None:
    water = Molecule.from_xyz(WATER_XYZ)
    assert water.symbols == ("O", "H", "H")
    assert water.positions[0][2] == pytest.approx(0.117790 * ANGSTROM_IN_BOHR)
    again = Molecule.from_xyz(WATER_XYZ, unit="bohr")
    assert again.positions[0][2] == pytest.approx(0.117790)
    headerless = Molecule.from_xyz("\n".join(WATER_XYZ.splitlines()[2:]))
    assert headerless.positions == water.positions


def test_from_xyz_rejects_inconsistent_input() -> None:
    with pytest.raises(InputError, match="announces 3 atoms"):
        Molecule.from_xyz("3\ncomment\nH 0 0 0\nH 0 0 1\n")
    with pytest.raises(InputError, match="cannot parse"):
        Molecule.from_xyz("H 0 0\n")
    with pytest.raises(InputError, match="unknown length unit"):
        Molecule.from_xyz("H 0 0 0\n", unit="pm")


def test_electron_counts_and_spin() -> None:
    water = Molecule.from_xyz(WATER_XYZ)
    assert (water.n_electrons, water.n_alpha, water.n_beta) == (10, 5, 5)
    cation = Molecule.from_xyz(WATER_XYZ, charge=1, spin=1)
    assert (cation.n_electrons, cation.n_alpha, cation.n_beta) == (9, 5, 4)
    oxygen = Molecule.from_atoms([("O", (0, 0, 0))], spin=2)
    assert (oxygen.n_alpha, oxygen.n_beta) == (5, 3)
    with pytest.raises(InputError, match="inconsistent"):
        Molecule.from_xyz(WATER_XYZ, spin=1)
    with pytest.raises(InputError, match="smallest consistent value is 1"):
        Molecule.from_atoms([("H", (0, 0, 0))])
    with pytest.raises(InputError, match="spin >= 0"):
        Molecule.from_xyz(WATER_XYZ, spin=-2)
    with pytest.raises(InputError, match="negative electron count"):
        Molecule.from_atoms([("H", (0, 0, 0))], charge=2)
    with pytest.raises(InputError, match="at least one atom"):
        Molecule(())


def test_nuclear_repulsion_energy() -> None:
    h2 = Molecule.from_atoms([("H", (0, 0, 0)), ("H", (0, 0, 1.4))], unit="bohr")
    assert h2.nuclear_repulsion_energy() == pytest.approx(1 / 1.4, abs=ENERGY_TOL)
    water = Molecule.from_xyz(WATER_XYZ)
    # value for this geometry, computed independently with CODATA 2018 units
    assert water.nuclear_repulsion_energy() == pytest.approx(9.189193229, abs=1e-8)
    with pytest.raises(InputError, match="coincide"):
        Molecule.from_atoms(
            [("H", (0, 0, 0)), ("H", (0, 0, 0))]
        ).nuclear_repulsion_energy()


def test_positions_array_and_xyz_round_trip() -> None:
    water = Molecule.from_xyz(WATER_XYZ)
    positions = water.positions_array()
    assert positions.shape == (3, 3)
    assert positions.dtype == np.float64
    again = Molecule.from_xyz(water.to_xyz(comment="water"))
    assert np.allclose(again.positions_array(), positions, atol=1e-10)
    assert bohr_to_angstrom(water.positions[1][1]) == pytest.approx(0.755453)
    assert water.to_xyz(unit="bohr").splitlines()[0] == "3"


@pytest.mark.parametrize(
    "comment",
    ["", "water", "Coordinates from ORCA-job input", "1 2 3 4", "   "],
)
def test_from_xyz_accepts_any_comment_line(comment: str) -> None:
    text = f"2\n{comment}\nH 0 0 0\nH 0 0 0.74\n"
    molecule = Molecule.from_xyz(text)
    assert molecule.symbols == ("H", "H")
    assert molecule.positions[1][2] == pytest.approx(0.74 * ANGSTROM_IN_BOHR)


def test_from_xyz_tolerates_line_endings_and_blank_lines() -> None:
    text = "2\r\ncomment\r\nH 0 0 0\r\n\r\nH 0 0 0.74\r\n\r\n"
    assert Molecule.from_xyz(text).n_atoms == 2
    assert Molecule.from_xyz("\nH 0 0 0\n\nH 0 0 1\n\n").n_atoms == 2


def test_units_are_case_insensitive_and_coordinates_validated() -> None:
    molecule = Molecule.from_atoms([("H", (0, 0, 1.0))], spin=1, unit="Bohr")
    assert molecule.positions[0][2] == 1.0
    with pytest.raises(InputError, match="three coordinates"):
        Molecule.from_atoms([("H", (0.0, 0.0))], spin=1)
    with pytest.raises(InputError, match="ghost atoms"):
        Molecule.from_atoms([("X", (0.0, 0.0, 0.0))])
    with pytest.raises(InputError, match="single line"):
        molecule.to_xyz(comment="two\nlines")

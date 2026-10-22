"""Cells: construction, extended XYZ, supercells, Ewald nuclear repulsion."""

import numpy as np
import pytest

from hafnium.errors import InputError
from hafnium.molecule import (
    ANGSTROM_IN_BOHR,
    AtomicSystem,
    Cell,
    Lattice,
    Molecule,
    ewald_energy,
)
from tests.cells import CUBIC, lithium_hydride
from tests.tolerances import ENERGY_TOL, STRUCTURE_TOL


def test_cell_and_molecule_are_siblings_under_the_atomic_system() -> None:
    cell = lithium_hydride()
    assert isinstance(cell, AtomicSystem)
    assert not isinstance(cell, Molecule)
    assert not isinstance(Molecule(cell.atoms), Cell)
    assert cell.dimension == 3
    assert cell.n_electrons == 4
    assert cell.symbols == ("Li", "H")
    assert cell.lattice == CUBIC
    charged = Cell(cell.atoms, charge=1, spin=1, lattice=CUBIC)
    assert charged.n_alpha == 2
    assert charged.n_beta == 1
    assert charged != Cell(
        cell.atoms, charge=1, spin=1, lattice=CUBIC.supercell((2, 1, 1))
    )
    with pytest.raises(InputError, match="at least one atom"):
        Cell((), lattice=CUBIC)


def test_from_atoms_takes_vectors_in_the_same_unit_or_a_lattice() -> None:
    cell = Cell.from_atoms(
        [("He", (0.0, 0.0, 0.0)), ("He", (1.0, 1.0, 1.0))],
        [(2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)],
    )
    assert cell.positions[1] == (ANGSTROM_IN_BOHR,) * 3
    assert cell.lattice.vectors[0] == (2.0 * ANGSTROM_IN_BOHR, 0.0, 0.0)
    in_bohr = Cell.from_atoms([("He", (0.0, 0.0, 0.0))], CUBIC, charge=1, spin=1)
    assert in_bohr.lattice == CUBIC
    assert in_bohr.n_electrons == 1
    with pytest.raises(InputError, match="dependent"):
        Cell.from_atoms([("He", (0.0, 0.0, 0.0))], [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])


def test_extended_xyz_round_trips_in_every_dimension() -> None:
    for vectors in [
        [(2.0, 0.1, 0.0)],
        [(2.0, 0.1, 0.0), (0.0, 3.0, 0.2)],
        [(2.0, 0.1, 0.0), (0.0, 3.0, 0.2), (0.1, 0.0, 4.0)],
    ]:
        lattice = Lattice.from_vectors(vectors)
        cell = Cell.from_molecule(
            Molecule.from_atoms([("C", (0.1, 0.2, 0.3)), ("O", (1.0, 0.5, 0.2))]),
            lattice,
        )
        text = cell.to_xyz(comment="carbon monoxide")
        lines = text.splitlines()
        assert lines[0] == "2"
        assert lines[1].startswith('Lattice="')
        assert lines[1].endswith(" carbon monoxide")
        again = Cell.from_xyz(text)
        assert again.lattice.dimension == lattice.dimension
        np.testing.assert_allclose(
            again.lattice.vectors_array(), lattice.vectors_array(), atol=STRUCTURE_TOL
        )
        np.testing.assert_allclose(
            again.positions_array(), cell.positions_array(), atol=STRUCTURE_TOL
        )
        assert again.symbols == cell.symbols
    bohr = Cell.from_xyz(cell.to_xyz(unit="bohr"), unit="bohr")
    np.testing.assert_allclose(bohr.positions_array(), cell.positions_array())
    # Molecule.from_xyz ignores the field and reads a plain molecule
    assert Molecule.from_xyz(text).positions == cell.positions


def test_extended_xyz_requires_a_well_formed_lattice_field() -> None:
    with pytest.raises(InputError, match=r'Lattice="\.\.\."'):
        Cell.from_xyz("1\nno lattice here\nH 0 0 0\n")
    with pytest.raises(InputError, match="three, six or nine"):
        Cell.from_xyz('1\nLattice="1 0 0 0"\nH 0 0 0\n')
    with pytest.raises(InputError, match="dependent"):
        Cell.from_xyz('1\nLattice="1 0 0 2 0 0"\nHe 0 0 0\n')


def test_supercell_replicates_atoms_and_scales_charge_and_spin() -> None:
    cell = Cell((lithium_hydride().atoms), charge=1, spin=1, lattice=CUBIC).supercell(
        (2, 1, 3)
    )
    assert cell.n_atoms == 12
    assert cell.charge == 6
    assert cell.spin == 6
    assert cell.lattice.volume == pytest.approx(6 * CUBIC.volume)
    positions = cell.positions_array()
    # image-major order: the first two atoms are the original cell
    np.testing.assert_allclose(positions[:2], lithium_hydride().positions_array())
    # every original atom appears once per image, shifted by a lattice vector
    fractional = CUBIC.fractional_coordinates(positions)
    np.testing.assert_allclose(
        fractional[2:] % 1.0, np.tile(fractional[:2] % 1.0, (5, 1)), atol=STRUCTURE_TOL
    )
    with pytest.raises(InputError, match="positive integers"):
        lithium_hydride().supercell((2, 2))


def test_nuclear_repulsion_is_the_ewald_energy_of_the_nuclei() -> None:
    cell = lithium_hydride()
    expected = ewald_energy(
        CUBIC, np.array([3.0, 1.0]), cell.positions_array(), omega=0.7
    )
    assert cell.nuclear_repulsion_energy(omega=0.7) == pytest.approx(
        expected, abs=ENERGY_TOL
    )
    assert cell.nuclear_repulsion_energy() == pytest.approx(expected, abs=ENERGY_TOL)
    # extensive in the number of cells
    doubled = cell.supercell((1, 2, 1)).nuclear_repulsion_energy()
    assert doubled == pytest.approx(2 * expected, abs=ENERGY_TOL)


def test_to_xyz_converts_the_lattice_with_the_positions() -> None:
    cell = lithium_hydride()
    line = cell.to_xyz().splitlines()[1]
    numbers = [float(x) for x in line.split('"')[1].split()]
    assert numbers[0] == pytest.approx(4.0 / ANGSTROM_IN_BOHR)

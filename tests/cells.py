"""Cells shared by the test suites, given in Bohr."""

import numpy as np

from hafnium.molecule import Cell, Lattice, Molecule

CUBIC = Lattice(((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)))

GENERAL_VECTORS = ((4.0, 0.3, 0.1), (-0.2, 4.2, 0.2), (0.1, 0.3, 3.8))


def lithium_hydride() -> Cell:
    """A rocksalt-like LiH cell in a cube of 4 Bohr."""
    molecule = Molecule.from_atoms(
        [("Li", (0.0, 0.0, 0.0)), ("H", (2.0, 2.0, 2.0))], unit="bohr"
    )
    return Cell.from_molecule(molecule, CUBIC)


def helium_atom() -> Cell:
    """One helium atom off the origin on the general three-dimensional lattice."""
    molecule = Molecule.from_atoms([("He", (0.3, 0.2, 0.1))], unit="bohr")
    return Cell.from_molecule(molecule, Lattice(GENERAL_VECTORS))


def helium(dimension: int) -> Cell:
    """Two helium atoms at general positions on a general lattice of ``dimension``."""
    molecule = Molecule.from_atoms(
        [("He", (0.0, 0.0, 0.0)), ("He", (1.9, 2.1, 1.8))], unit="bohr"
    )
    return Cell.from_molecule(molecule, Lattice(GENERAL_VECTORS[:dimension]))


def helium_slab() -> Cell:
    """Two helium atoms, one above the layer, on an oblique lattice in the ``xy`` plane.

    PySCF's two-dimensional Coulomb terms assume that orientation.
    """
    molecule = Molecule.from_atoms(
        [("He", (0.0, 0.0, 0.0)), ("He", (1.9, 2.1, 1.2))], unit="bohr"
    )
    return Cell.from_molecule(molecule, Lattice(((4.0, 0.3, 0.0), (-0.2, 4.2, 0.0))))


def box(molecule: Molecule, length: float, dimension: int = 3) -> Cell:
    """``molecule`` repeated on an orthogonal lattice of side ``length`` in Bohr."""
    vectors = tuple(tuple(row) for row in (length * np.eye(3))[:dimension].tolist())
    return Cell.from_molecule(molecule, Lattice(vectors))


def diamond_helium(a: float = 6.0) -> Cell:
    """Two helium atoms in the diamond structure: non-symmorphic Fd-3m."""
    lattice = Lattice(((0.0, a / 2, a / 2), (a / 2, 0.0, a / 2), (a / 2, a / 2, 0.0)))
    molecule = Molecule.from_atoms(
        [("He", (0.0, 0.0, 0.0)), ("He", (a / 4, a / 4, a / 4))], unit="bohr"
    )
    return Cell.from_molecule(molecule, lattice)


def hexagonal_helium(b: float = 5.0) -> Cell:
    """One helium atom per cell of a hexagonal slab."""
    lattice = Lattice(((b, 0.0, 0.0), (-b / 2, b * np.sqrt(3) / 2, 0.0)))
    return Cell.from_molecule(Molecule.from_atoms([("He", (0.0, 0.0, 0.0))]), lattice)


def helium_chain(a: float = 4.0) -> Cell:
    """One helium atom per cell of a chain along ``x``."""
    return Cell.from_molecule(
        Molecule.from_atoms([("He", (0.0, 0.0, 0.0))]), Lattice(((a, 0.0, 0.0),))
    )

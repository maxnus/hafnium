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


def helium(dimension: int) -> Cell:
    """Two helium atoms at general positions on a general lattice of ``dimension``."""
    molecule = Molecule.from_atoms(
        [("He", (0.0, 0.0, 0.0)), ("He", (1.9, 2.1, 1.8))], unit="bohr"
    )
    return Cell.from_molecule(molecule, Lattice(GENERAL_VECTORS[:dimension]))


def box(molecule: Molecule, length: float, dimension: int = 3) -> Cell:
    """``molecule`` repeated on an orthogonal lattice of side ``length`` in Bohr."""
    vectors = tuple(tuple(row) for row in (length * np.eye(3))[:dimension].tolist())
    return Cell.from_molecule(molecule, Lattice(vectors))

"""Finite differences of a quantity along the nuclear coordinates, for tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from hafnium.backend import Array
from hafnium.molecule import Molecule
from tests.molecules import displaced

STEP = 2e-3
"""Displacement in Bohr of the five-point central differences."""


def finite_differences(
    molecule: Molecule, value: Callable[[Molecule], float], step: float = STEP
) -> Array:
    """Five-point central differences of ``value`` along every coordinate."""
    gradient = np.zeros((molecule.n_atoms, 3))
    for atom in range(molecule.n_atoms):
        for axis in range(3):
            samples = [
                value(displaced(molecule, atom, axis, k * step)) for k in (-2, -1, 1, 2)
            ]
            gradient[atom, axis] = (
                samples[0] - 8 * samples[1] + 8 * samples[2] - samples[3]
            ) / (12 * step)
    return gradient

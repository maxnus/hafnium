"""What the tests of nuclear gradients share: the systems, the options, differences.

The analytic gradients (ADR-0035) and the gradients by automatic
differentiation (ADR-0038) are compared with each other and with finite
differences over the same classes, bases and convergence options, so the
two suites take them from here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from hafnium.backend import Array
from hafnium.basis import BasisSet
from hafnium.molecule import Molecule
from hafnium.scf import ScfOptions
from tests.molecules import displaced
from tests.shell_tables import LIBRARY

STEP = 2e-3
"""Displacement in Bohr of the five-point central differences."""

AUXILIARY = "def2-universal-jkfit"
"""The fitting basis of every gradient test."""

CLASSES = [
    "RHF",
    "complex RHF",
    "UHF",
    "complex UHF",
    "paired UHF",
    "GHF",
    "complex GHF",
    "paired GHF",
]
"""The Fukutome classes a gradient is tested in."""

# Tight convergence, so that the finite differences of the energy are limited
# by the rounding of the energies and not by the residual orbital error.
OPTIONS = ScfOptions(
    gradient_tolerance=1e-10, energy_tolerance=1e-13, max_iterations=200
)


def bases(molecule: Molecule, basis_name: str = "sto-3g") -> tuple[BasisSet, BasisSet]:
    """The orbital basis ``basis_name`` and the fitting basis on ``molecule``."""
    return (
        BasisSet.for_atoms(molecule, basis_name, library=LIBRARY),
        BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY),
    )


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

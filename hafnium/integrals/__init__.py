"""Integrals over Gaussian basis functions.

The :class:`IntegralEngine` protocol is the boundary between method code
and integral evaluation. Every kernel takes a bra and a ket
:class:`ShellTable` and a list of translation vectors applied to the ket,
so molecular integrals and the lattice sums of periodic systems share one
interface (ADR-0007). :mod:`hafnium.integrals.one_electron` and
:mod:`hafnium.integrals.two_electron` wrap the engine for molecules;
:mod:`hafnium.integrals.periodic` forms the Bloch sums of a cell at its
k-points over the images that its basis functions reach (ADR-0016).
"""

from hafnium.integrals.density_fitting import DensityFitting
from hafnium.integrals.engine import IntegralEngine
from hafnium.integrals.native import RustEngine
from hafnium.integrals.one_electron import (
    core_hamiltonian,
    kinetic,
    nuclear_attraction,
    overlap,
)
from hafnium.integrals.periodic import (
    DEFAULT_LATTICE_SUM_PRECISION,
    bloch_sum,
    kinetic_k,
    lattice_images,
    overlap_k,
)
from hafnium.integrals.shell_table import ShellTable
from hafnium.integrals.two_electron import eri

__all__ = [
    "DEFAULT_LATTICE_SUM_PRECISION",
    "DensityFitting",
    "IntegralEngine",
    "RustEngine",
    "ShellTable",
    "bloch_sum",
    "core_hamiltonian",
    "eri",
    "kinetic",
    "kinetic_k",
    "lattice_images",
    "nuclear_attraction",
    "overlap",
    "overlap_k",
]

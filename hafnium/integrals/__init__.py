"""Integrals over Gaussian basis functions.

The :class:`IntegralEngine` protocol is the boundary between method code
and integral evaluation. Every kernel takes a bra and a ket
:class:`ShellTable` and a list of translation vectors applied to the ket,
so molecular integrals and the lattice sums of periodic systems share one
interface (ADR-0007). :mod:`hafnium.integrals.one_electron` and
:mod:`hafnium.integrals.two_electron` wrap the engine for molecules.
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
from hafnium.integrals.shell_table import ShellTable
from hafnium.integrals.two_electron import eri

__all__ = [
    "DensityFitting",
    "IntegralEngine",
    "RustEngine",
    "ShellTable",
    "core_hamiltonian",
    "eri",
    "kinetic",
    "nuclear_attraction",
    "overlap",
]

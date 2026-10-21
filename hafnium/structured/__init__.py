"""Symmetry-structured matrices: the data model behind every method.

A symmetry constraint turns each one-body quantity (overlap, core
Hamiltonian, Fock matrix, density, orbital coefficients) into a direct sum
of independent dense blocks over the real, complex or quaternionic ring,
possibly with partner blocks derived by copying or complex conjugation
(PLAN.md §5). :class:`Structure` describes that decomposition,
:class:`BlockMatrix` and :class:`BlockVector` hold data in it, and
algorithms are written once against their operations.
"""

from hafnium.structured.kramers import (
    is_kramers_symmetric,
    kramers_adapt_eigenvectors,
    kramers_symmetrize,
    time_reverse,
    time_reverse_operator,
)
from hafnium.structured.matrix import BlockMatrix, BlockVector, density_matrix
from hafnium.structured.structure import Block, Partner, Ring, Spin, Structure

__all__ = [
    "Block",
    "BlockMatrix",
    "BlockVector",
    "Partner",
    "Ring",
    "Spin",
    "Structure",
    "density_matrix",
    "is_kramers_symmetric",
    "kramers_adapt_eigenvectors",
    "kramers_symmetrize",
    "time_reverse",
    "time_reverse_operator",
]

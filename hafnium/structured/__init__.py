"""Symmetry-structured matrices: the data model behind every method.

A symmetry constraint turns each one-body quantity (overlap, core
Hamiltonian, Fock matrix, density, orbital coefficients) into a direct sum
of independent dense blocks over the real, complex or quaternionic ring,
possibly with partner blocks derived by copying or complex conjugation
(PLAN.md §5). For a periodic system every block is a batch over the
k-points of a :class:`KAxis`, the real and paired classes storing a half
zone and deriving the block at ``−k`` (ADR-0015). :class:`BlockStructure`
describes that decomposition, :class:`BlockMatrix` and :class:`BlockVector`
hold data in it, and algorithms are written once against their operations.
"""

from hafnium.structured.density import density_matrix
from hafnium.structured.kaxis import KAxis
from hafnium.structured.kramers import (
    is_kramers_symmetric,
    kramers_adapt_eigenvectors,
    kramers_symmetrize,
    time_reverse,
    time_reverse_operator,
)
from hafnium.structured.matrix import BlockMatrix
from hafnium.structured.structure import Block, BlockStructure, Partner, Ring, Spin
from hafnium.structured.vector import BlockVector

__all__ = [
    "Block",
    "BlockMatrix",
    "BlockStructure",
    "BlockVector",
    "KAxis",
    "Partner",
    "Ring",
    "Spin",
    "density_matrix",
    "is_kramers_symmetric",
    "kramers_adapt_eigenvectors",
    "kramers_symmetrize",
    "time_reverse",
    "time_reverse_operator",
]

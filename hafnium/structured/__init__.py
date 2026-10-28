"""Symmetry-structured matrices: the data model behind every method.

A symmetry constraint turns each one-body quantity (overlap, core
Hamiltonian, Fock matrix, density, orbital coefficients) into a direct sum
of independent dense blocks over the real, complex or quaternionic ring,
possibly with partner blocks derived by copying or complex conjugation
(PLAN.md §5). For a periodic system every block is a batch over the
k-points of a :class:`KAxis`, the real and paired classes storing a half
zone and deriving the block at ``−k`` (ADR-0015); with the point
operations of a space group, :class:`KPointOperations`, every class
stores the irreducible wedge and derives the rotated points (ADR-0024).
The irrep sectors of an Abelian point group, :class:`IrrepSectors`,
multiply the spin blocks of a molecular structure, each copy acting on
the SALCs of its irrep (ADR-0023). :class:`BlockStructure` describes that decomposition,
:class:`BlockMatrix` and :class:`BlockVector` hold data in it, and
algorithms are written once against their operations. Quantities with
several orbital indices are :class:`BlockTensor` objects over the sectors
of a structure, whose blocks are selected by the conservation of spin and
irrep charges and contracted with :func:`contract` (ADR-0030).
"""

from hafnium.structured.density import density_matrix
from hafnium.structured.kaxis import KAxis, KDerivation, KPointOperations
from hafnium.structured.kramers import (
    is_kramers_symmetric,
    kramers_adapt_eigenvectors,
    kramers_symmetrize,
    time_reverse,
    time_reverse_operator,
)
from hafnium.structured.matrix import BlockMatrix
from hafnium.structured.sectors import IrrepSectors
from hafnium.structured.structure import Block, BlockStructure, Partner, Ring, Spin
from hafnium.structured.tensor import BlockTensor, TensorAxis, TensorStructure, contract
from hafnium.structured.vector import BlockVector

__all__ = [
    "Block",
    "BlockMatrix",
    "BlockStructure",
    "BlockTensor",
    "BlockVector",
    "IrrepSectors",
    "KAxis",
    "KDerivation",
    "KPointOperations",
    "Partner",
    "Ring",
    "Spin",
    "TensorAxis",
    "TensorStructure",
    "contract",
    "density_matrix",
    "is_kramers_symmetric",
    "kramers_adapt_eigenvectors",
    "kramers_symmetrize",
    "time_reverse",
    "time_reverse_operator",
]

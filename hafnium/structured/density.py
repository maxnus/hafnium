"""Density matrices from orbital coefficients and occupations, and back."""

from __future__ import annotations

from hafnium.backend import Array, adjoint, array_namespace
from hafnium.errors import StructureError
from hafnium.structured.kramers import is_kramers_symmetric
from hafnium.structured.matrix import BlockMatrix
from hafnium.structured.structure import Block, BlockStructure, Ring
from hafnium.structured.vector import BlockVector

REALITY_TOLERANCE = 1e-10
"""Relative imaginary part a real block may carry at an invariant k-point."""


def density_matrix(coefficients: BlockMatrix, occupations: BlockVector) -> BlockMatrix:
    """Return ``C diag(occupations) adjoint(C)`` block by block.

    Parameters
    ----------
    coefficients : BlockMatrix
        Orbital coefficients ``[ao, mo]`` per block.
    occupations : BlockVector
        Occupation numbers ``[mo]`` per block (``[kpt, mo]`` with a k
        axis), as floats. In a paired generalized structure they must be
        equal within each Kramers pair at the time-reversal invariant
        k-points, and in a real structure they must leave the density real
        there.
    """
    structure = coefficients.structure
    if structure != occupations.structure:
        raise StructureError("coefficients and occupations must share a structure")
    blocks: list[Array] = []
    for block, c, occupation in zip(
        structure.stored_blocks, coefficients.blocks, occupations.blocks, strict=True
    ):
        xp = array_namespace(c, occupation)
        density = (c * xp.expand_dims(occupation, axis=-2)) @ adjoint(c)
        _check_time_reversal(structure, block, density)
        blocks.append(density)
    return BlockMatrix(structure, tuple(blocks))


def natural_orbitals(
    density: BlockMatrix, overlap: BlockMatrix
) -> tuple[BlockVector, BlockMatrix]:
    """Return the occupations and orbitals with ``D = C diag(n) adjoint(C)``.

    The natural orbitals of a Hermitian density solve ``D S C = C n`` with
    ``C`` orthonormal in the metric ``S``, the eigenproblem of ``S D S``
    with the overlap (:meth:`BlockMatrix.eigh`); the occupations come
    block by block and point by point in ascending order.
    """
    return (overlap @ density @ overlap).eigh(overlap)


def _check_time_reversal(
    structure: BlockStructure, block: Block, density: Array
) -> None:
    """Raise unless ``density`` has the symmetry of ``block`` at invariant k-points."""
    if block.ring is Ring.COMPLEX:
        return
    batched = structure.k_axis is not None
    for position in structure.invariant_positions(block):
        matrix = density[position] if batched else density
        if block.ring is Ring.QUATERNION:
            if not is_kramers_symmetric(matrix):
                raise StructureError(
                    "occupations of a paired structure must be equal within each "
                    "Kramers pair"
                )
            continue
        xp = array_namespace(matrix)
        if not xp.isdtype(matrix.dtype, "complex floating"):
            continue
        scale = max(1.0, float(xp.max(xp.abs(matrix))))
        if float(xp.max(xp.abs(xp.imag(matrix)))) > REALITY_TOLERANCE * scale:
            raise StructureError(
                "the density of a real structure is real at a time-reversal "
                "invariant k-point; the occupations break time reversal within a "
                "degenerate set"
            )

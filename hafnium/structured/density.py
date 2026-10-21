"""Density matrices from orbital coefficients and occupations."""

from __future__ import annotations

from hafnium.backend import Array, adjoint, array_namespace
from hafnium.errors import StructureError
from hafnium.structured.kramers import is_kramers_symmetric
from hafnium.structured.matrix import BlockMatrix
from hafnium.structured.structure import Ring
from hafnium.structured.vector import BlockVector


def density_matrix(coefficients: BlockMatrix, occupations: BlockVector) -> BlockMatrix:
    """Return ``C diag(occupations) adjoint(C)`` block by block.

    Parameters
    ----------
    coefficients : BlockMatrix
        Orbital coefficients ``[ao, mo]`` per block.
    occupations : BlockVector
        Occupation numbers ``[mo]`` per block, as floats. In a paired
        generalized structure they must be equal within each Kramers pair.
    """
    if coefficients.structure != occupations.structure:
        raise StructureError("coefficients and occupations must share a structure")
    blocks: list[Array] = []
    for block, c, occupation in zip(
        coefficients.structure.stored_blocks,
        coefficients.blocks,
        occupations.blocks,
        strict=True,
    ):
        xp = array_namespace(c, occupation)
        density = (c * xp.expand_dims(occupation, axis=0)) @ adjoint(c)
        if block.ring is Ring.QUATERNION and not is_kramers_symmetric(density):
            raise StructureError(
                "occupations of a paired structure must be equal within each "
                "Kramers pair"
            )
        blocks.append(density)
    return BlockMatrix(coefficients.structure, tuple(blocks))

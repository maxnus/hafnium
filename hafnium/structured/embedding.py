"""Embedding along the Fukutome hierarchy without the dense representation.

Every member of a more symmetric structure is a member of a less symmetric
one: the spin structure may only be relaxed (spatial, restricted,
unrestricted, generalized), real data fits any ring, complex data only
complex rings, and paired (time-reversal symmetric) targets accept paired
sources or real sources whose alpha and beta blocks coincide. For such
pairs the stored blocks of the target follow from those of the source by
copying, casting and block-diagonal stacking, with nothing to verify.
"""

from __future__ import annotations

from collections.abc import Sequence

from hafnium.backend import Array, array_namespace
from hafnium.structured.dense import block_diag, spin_blocks
from hafnium.structured.structure import BlockStructure, Partner, Ring
from hafnium.structured.validation import cast_to_ring


def spin_level(structure: BlockStructure) -> int:
    """Position in the chain spatial < restricted < unrestricted < generalized."""
    if structure.is_spatial:
        return 0
    if structure.is_spinor:
        return 3
    return 1 if structure.derived_partner is Partner.COPY else 2


def is_paired(structure: BlockStructure) -> bool:
    """Whether the structure imposes time-reversal symmetry."""
    return structure.derived_partner is Partner.CONJUGATE or any(
        block.ring is Ring.QUATERNION for block in structure.blocks
    )


def is_structural_embedding(source: BlockStructure, target: BlockStructure) -> bool:
    """Whether every member of ``source`` is a member of ``target`` by construction."""
    if spin_level(source) > spin_level(target):
        return False
    source_is_real = not source.is_complex
    if is_paired(target):
        return is_paired(source) or (source_is_real and spin_level(source) <= 1)
    if not target.is_complex:
        return source_is_real
    return True


def embed_blocks(
    source: BlockStructure, stored: Sequence[Array], target: BlockStructure
) -> tuple[Array, ...]:
    """Return the stored blocks of ``target`` for a matrix given in ``source``.

    Only valid when :func:`is_structural_embedding` holds for the pair.
    """
    xp = array_namespace(*stored)
    if target.is_spinor:
        parts = spin_blocks(source, stored)
        spinor = parts[0] if len(parts) == 1 else block_diag(parts[0], parts[1])
        target_blocks: tuple[Array, ...] = (spinor,)
    elif target.is_spatial:
        target_blocks = (stored[0],)
    else:
        alpha, beta = spin_blocks(source, stored)
        target_blocks = (
            (alpha,) if target.derived_partner is not None else (alpha, beta)
        )
    return tuple(
        cast_to_ring(xp, part, block.ring)
        for part, block in zip(target_blocks, target.stored_blocks, strict=True)
    )

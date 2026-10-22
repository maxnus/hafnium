"""Per-orbital data per block, such as eigenvalues or occupations.

A block holds a vector ``[mo]``, or ``[kpt, mo]`` over the k-points the
block stores when the structure has a k axis. The data is real per orbital
and identical for time-reversal partners, so the derived k-points copy
their stored partner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from hafnium.backend import Array, array_namespace
from hafnium.errors import StructureError
from hafnium.structured.dense import materialize
from hafnium.structured.structure import BlockStructure
from hafnium.structured.validation import check_block_count, check_k_count


@dataclass(frozen=True, slots=True, eq=False)
class BlockVector:
    """Per-orbital data per stored block of a :class:`BlockStructure`."""

    structure: BlockStructure
    blocks: tuple[Array, ...]

    def __post_init__(self) -> None:
        check_block_count(self.structure, self.blocks)
        expected = 1 if self.structure.k_axis is None else 2
        for block, array in zip(self.structure.stored_blocks, self.blocks, strict=True):
            if array.ndim != expected:
                raise StructureError(
                    "vector blocks are one-dimensional"
                    if expected == 1
                    else "vector blocks with a k axis are two-dimensional, [kpt, mo]"
                )
            check_k_count(self.structure, block, array)

    def materialized(self) -> tuple[Array, ...]:
        """One array per block over the full k set, derived blocks included."""
        return materialize(self.structure, self.blocks, matrix=False)

    def to_dense(self) -> Array:
        """Concatenation over the spin-orbital space (alpha first), ``[(kpt,) 2m]``."""
        parts = self.materialized()
        if self.structure.is_spatial:
            parts = (parts[0], parts[0])
        xp = array_namespace(*parts)
        return xp.concat(parts, axis=-1)

    def map_blocks(self, function: Callable[[Array], Array]) -> BlockVector:
        """Apply ``function`` to every stored block."""
        return BlockVector(
            self.structure, tuple(function(block) for block in self.blocks)
        )

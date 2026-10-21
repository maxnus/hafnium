"""One-dimensional data per block, such as eigenvalues or occupations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from hafnium.backend import Array, array_namespace
from hafnium.errors import StructureError
from hafnium.structured.dense import materialize
from hafnium.structured.structure import BlockStructure
from hafnium.structured.validation import check_block_count


@dataclass(frozen=True, slots=True, eq=False)
class BlockVector:
    """One-dimensional data per stored block of a :class:`BlockStructure`."""

    structure: BlockStructure
    blocks: tuple[Array, ...]

    def __post_init__(self) -> None:
        check_block_count(self.structure, self.blocks)
        for array in self.blocks:
            if array.ndim != 1:
                raise StructureError("vector blocks are one-dimensional")

    def materialized(self) -> tuple[Array, ...]:
        """One array per block of the structure, derived blocks included."""
        return materialize(self.structure, self.blocks)

    def to_dense(self) -> Array:
        """Concatenation over the spin-orbital space (alpha first)."""
        parts = self.materialized()
        if self.structure.is_spatial:
            parts = (parts[0], parts[0])
        xp = array_namespace(*parts)
        return xp.concat(parts, axis=0)

    def map_blocks(self, function: Callable[[Array], Array]) -> BlockVector:
        """Apply ``function`` to every stored block."""
        return BlockVector(
            self.structure, tuple(function(block) for block in self.blocks)
        )

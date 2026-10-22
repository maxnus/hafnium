"""Validation shared by block matrices and vectors: block counts, rings, dtypes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hafnium.backend import Array, Namespace
from hafnium.errors import StructureError
from hafnium.structured.structure import Block, BlockStructure, Ring


def dtype_for(xp: Namespace, ring: Ring) -> Any:
    """Return the one dtype a block stored over ``ring`` may have (CONVENTIONS §5)."""
    return xp.float64 if ring is Ring.REAL else xp.complex128


def check_block_count(structure: BlockStructure, blocks: Sequence[Array]) -> None:
    """Raise unless ``blocks`` has one array per stored block of ``structure``."""
    if len(blocks) != structure.n_stored:
        raise StructureError(
            f"{structure.describe()} structure stores {structure.n_stored} blocks, "
            f"got {len(blocks)}"
        )


def check_k_count(structure: BlockStructure, block: Block, array: Array) -> None:
    """Raise unless the leading axis of ``array`` holds the k-points of ``block``."""
    stored = structure.stored_kpoints(block)
    if stored is not None and int(array.shape[0]) != len(stored):
        raise StructureError(
            f"the {block.spin.value} block of a {structure.describe()} structure "
            f"stores {len(stored)} k-points, got {int(array.shape[0])}"
        )


def check_ring(
    xp: Namespace, array: Array, structure: BlockStructure, block: Block
) -> None:
    """Raise unless ``array`` has exactly the dtype ``block`` is stored with."""
    if not xp.isdtype(array.dtype, ("real floating", "complex floating")):
        raise StructureError(f"blocks must be floating point, got {array.dtype}")
    is_complex = bool(xp.isdtype(array.dtype, "complex floating"))
    storage = structure.storage_ring(block)
    if storage is Ring.REAL and is_complex:
        raise StructureError(
            "a real block has complex entries; embed into a complex structure"
        )
    if storage is not Ring.REAL and not is_complex:
        reason = (
            "is complex away from the time-reversal invariant k-points"
            if block.ring is Ring.REAL
            else "has real entries; cast explicitly"
        )
        raise StructureError(f"a {block.ring.value} block {reason}")
    expected = dtype_for(xp, storage)
    if array.dtype != expected:
        raise StructureError(f"blocks must be {expected}, got {array.dtype}")


def cast_to_ring(xp: Namespace, array: Array, ring: Ring) -> Array:
    """Cast ``array`` to the dtype of ``ring``, dropping imaginary parts for REAL."""
    if ring is Ring.REAL:
        real = xp.real(array) if xp.isdtype(array.dtype, "complex floating") else array
        return xp.astype(real, xp.float64)
    return xp.astype(array, xp.complex128)

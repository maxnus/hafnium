"""Validation shared by block matrices and vectors: block counts, rings, dtypes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hafnium.backend import Array, Namespace
from hafnium.errors import StructureError
from hafnium.structured.structure import BlockStructure, Ring


def dtype_for(xp: Namespace, ring: Ring) -> Any:
    """Return the one dtype a block of ``ring`` may have (CONVENTIONS.md §5)."""
    return xp.float64 if ring is Ring.REAL else xp.complex128


def check_block_count(structure: BlockStructure, blocks: Sequence[Array]) -> None:
    """Raise unless ``blocks`` has one array per stored block of ``structure``."""
    if len(blocks) != structure.n_stored:
        raise StructureError(
            f"{structure.describe()} structure stores {structure.n_stored} blocks, "
            f"got {len(blocks)}"
        )


def check_ring(xp: Namespace, array: Array, ring: Ring) -> None:
    """Raise unless ``array`` has exactly the dtype of ``ring``."""
    if not xp.isdtype(array.dtype, ("real floating", "complex floating")):
        raise StructureError(f"blocks must be floating point, got {array.dtype}")
    is_complex = bool(xp.isdtype(array.dtype, "complex floating"))
    if ring is Ring.REAL and is_complex:
        raise StructureError(
            "a real block has complex entries; embed into a complex structure"
        )
    if ring is not Ring.REAL and not is_complex:
        raise StructureError(f"a {ring.value} block has real entries; cast explicitly")
    expected = dtype_for(xp, ring)
    if array.dtype != expected:
        raise StructureError(f"blocks must be {expected}, got {array.dtype}")


def cast_to_ring(xp: Namespace, array: Array, ring: Ring) -> Array:
    """Cast ``array`` to the dtype of ``ring``, dropping imaginary parts for REAL."""
    if ring is Ring.REAL:
        real = xp.real(array) if xp.isdtype(array.dtype, "complex floating") else array
        return xp.astype(real, xp.float64)
    return xp.astype(array, xp.complex128)

"""The dense spin-orbital representation of block matrices.

The dense form has ``2n`` rows with the alpha components first; a spatial
block ``X`` expands to ``block_diag(X, X)``, collinear blocks to
``block_diag(alpha, beta)`` and a spinor block is already dense. It is the
canonical form that relates all structures. The functions here act on a
structure and its stored blocks rather than on :class:`BlockMatrix`, which
keeps that class thin.
"""

from __future__ import annotations

from collections.abc import Sequence

from hafnium.backend import Array, Namespace, adjoint, array_namespace
from hafnium.errors import StructureError
from hafnium.structured.kramers import (
    half_dimension,
    is_kramers_symmetric,
    kramers_symmetrize,
)
from hafnium.structured.structure import Block, BlockStructure, Partner, Ring
from hafnium.structured.validation import cast_to_ring


def materialize(
    structure: BlockStructure, stored: Sequence[Array]
) -> tuple[Array, ...]:
    """Expand stored blocks to one array per block of the structure."""
    arrays: list[Array] = []
    position = 0
    for block in structure.blocks:
        if block.derived_from is None:
            arrays.append(stored[position])
            position += 1
            continue
        source = arrays[block.derived_from]
        if block.partner is Partner.CONJUGATE:
            xp = array_namespace(source)
            source = xp.conj(source)
        arrays.append(source)
    return tuple(arrays)


def spin_blocks(
    structure: BlockStructure, stored: Sequence[Array]
) -> tuple[Array, ...]:
    """``(alpha, beta)`` blocks, or the single spinor block."""
    arrays = materialize(structure, stored)
    if structure.is_spatial:
        return (arrays[0], arrays[0])
    return arrays


def block_diag(a: Array, b: Array) -> Array:
    """Block-diagonal matrix with ``a`` and ``b`` on the diagonal."""
    xp = array_namespace(a, b)
    dtype = xp.result_type(a, b)
    top = xp.concat(
        [xp.astype(a, dtype), xp.zeros((a.shape[0], b.shape[1]), dtype=dtype)], axis=1
    )
    bottom = xp.concat(
        [xp.zeros((b.shape[0], a.shape[1]), dtype=dtype), xp.astype(b, dtype)], axis=1
    )
    return xp.concat([top, bottom], axis=0)


def to_dense(structure: BlockStructure, stored: Sequence[Array]) -> Array:
    """Expand into the spin-orbital representation ``[2n, m_alpha + m_beta]``."""
    parts = spin_blocks(structure, stored)
    if len(parts) == 1:
        return parts[0]
    return block_diag(parts[0], parts[1])


def column_split(
    structure: BlockStructure, stored: Sequence[Array]
) -> tuple[int, int] | None:
    """Column counts of the alpha and beta blocks; ``None`` for spinor matrices."""
    if structure.is_spinor:
        return None
    columns = [int(block.shape[1]) for block in stored]
    if len(columns) == 1:
        return (columns[0], columns[0])
    return (columns[0], columns[1])


def blocks_from_dense(
    structure: BlockStructure,
    dense: Array,
    *,
    column_split: tuple[int, int] | None,
    check: bool,
    tolerance: float,
) -> tuple[Array, ...]:
    """Extract the stored blocks of ``structure`` from a spin-orbital matrix.

    With ``check`` the result must reproduce ``dense`` to within
    ``tolerance`` (a lossless embedding); without it the matrix is projected
    (alpha and beta averaged, real part taken, Kramers-symmetrized).
    """
    xp = array_namespace(dense)
    if dense.ndim != 2:
        raise StructureError("the dense representation is two-dimensional")
    rows, cols = int(dense.shape[0]), int(dense.shape[1])
    n = half_dimension(rows, "the number of rows of a spin-orbital matrix")
    if structure.is_spinor:
        (block,) = structure.stored_blocks
        parts: tuple[Array, ...] = (_spinor_from_dense(block, dense, check, tolerance),)
    else:
        parts = _collinear_from_dense(
            structure, dense, n, _split_columns(cols, column_split)
        )
    cast = tuple(
        cast_to_ring(xp, part, block.ring)
        for part, block in zip(parts, structure.stored_blocks, strict=True)
    )
    if check:
        _require_close(
            xp,
            to_dense(structure, cast),
            dense,
            tolerance,
            f"the matrix does not have the {structure.describe()} structure",
        )
    return cast


def _split_columns(cols: int, column_split: tuple[int, int] | None) -> tuple[int, int]:
    if column_split is None:
        if cols % 2:
            raise StructureError(
                f"the dense matrix has {cols} columns; pass "
                "column_split=(m_alpha, m_beta) to split them between the spin blocks"
            )
        return (cols // 2, cols // 2)
    if sum(column_split) != cols:
        raise StructureError("column_split must sum to the number of columns")
    return column_split


def _spinor_from_dense(
    block: Block, dense: Array, check: bool, tolerance: float
) -> Array:
    if block.ring is not Ring.QUATERNION:
        return dense
    xp = array_namespace(dense)
    if dense.shape[0] == dense.shape[1]:
        return kramers_symmetrize(xp.astype(dense, xp.complex128))
    if not check:
        raise StructureError(
            "projecting a rectangular matrix onto a paired structure is not defined"
        )
    if not _has_kramers_invariant_range(dense, tolerance):
        raise StructureError("the column space is not invariant under time reversal")
    return dense


def _collinear_from_dense(
    structure: BlockStructure, dense: Array, n: int, split: tuple[int, int]
) -> tuple[Array, ...]:
    m_alpha, m_beta = split
    alpha, beta = dense[:n, :m_alpha], dense[n:, m_alpha:]
    partner = structure.derived_partner
    if partner is None and not structure.is_spatial:
        return (alpha, beta)
    if m_alpha != m_beta:
        raise StructureError("alpha and beta blocks must have equal shapes")
    if partner is Partner.CONJUGATE:
        xp = array_namespace(dense)
        return ((alpha + xp.conj(beta)) / 2,)
    return ((alpha + beta) / 2,)


def _require_close(
    xp: Namespace, actual: Array, expected: Array, tolerance: float, what: str
) -> None:
    scale = max(1.0, float(xp.max(xp.abs(expected))))
    deviation = float(xp.max(xp.abs(actual - expected)))
    if deviation > tolerance * scale:
        raise StructureError(f"{what} (deviation {deviation:.2e})")


def _has_kramers_invariant_range(dense: Array, tolerance: float) -> bool:
    xp = array_namespace(dense)
    orthonormal, _ = xp.linalg.qr(xp.astype(dense, xp.complex128))
    projector = orthonormal @ adjoint(orthonormal)
    return is_kramers_symmetric(projector, tolerance=tolerance)

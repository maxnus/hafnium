"""The dense spin-orbital representation of block matrices.

The dense form has ``2n`` rows with the alpha components first; a spatial
block ``X`` expands to ``block_diag(X, X)``, collinear blocks to
``block_diag(alpha, beta)`` and a spinor block is already dense. With a k
axis the dense form carries every k-point of the set as a leading axis,
``[kpt, 2n, m]``, the blocks that are stored for a half zone being
expanded by the antiunitary map of their ring (ADR-0015). With irrep
sectors the dense form is in the atomic basis (ADR-0023): the sector
blocks of an operator are assembled as ``Σ_Γ U_Γ X_Γ U_Γᵀ`` and those of
orbital coefficients placed side by side as ``U_Γ C_Γ``, so the dense form
of a matrix over sectors is the dense form of the same matrix without
them. It is the canonical form that relates all structures. The functions
here act on a structure and its stored blocks rather than on
:class:`BlockMatrix`, which keeps that class thin.
"""

from __future__ import annotations

from collections.abc import Sequence

from hafnium.backend import Array, Namespace, adjoint, array_namespace, block_diag
from hafnium.errors import StructureError
from hafnium.structured.kramers import (
    half_dimension,
    is_kramers_symmetric,
    time_reverse,
    time_reverse_operator,
)
from hafnium.structured.structure import Block, BlockStructure, Partner, Ring, Spin
from hafnium.structured.validation import cast_to_ring

# ----------------------------------------------------------------- k partners


def antiunitary(ring: Ring, array: Array, *, matrix: bool) -> Array:
    """Map a block at ``k`` to its time-reversal partner at ``−k``.

    Real blocks are conjugated. Square quaternionic blocks are operators and
    receive the Kramers map ``Θ X Θ⁻¹``; rectangular ones are coefficient
    matrices whose columns are time-reversed, ``Θ C``. Vectors
    (``matrix=False``) hold per-orbital real data and are conjugated, which
    leaves them unchanged.
    """
    xp = array_namespace(array)
    if ring is Ring.COMPLEX:
        raise StructureError("a complex block has no time-reversal partner")
    if ring is Ring.REAL or not matrix:
        return xp.conj(array)
    if array.shape[-2] == array.shape[-1]:
        return time_reverse_operator(array)
    return time_reverse(array)


def expand_k(
    structure: BlockStructure,
    block: Block,
    array: Array,
    wanted: Sequence[int] | None,
    *,
    matrix: bool = True,
) -> Array:
    """Return the stored ``array`` of ``block`` at the full-set indices ``wanted``.

    Points the block stores are taken as they are, the others are derived
    from their stored partner by :func:`antiunitary`. Without a k axis the
    array is returned unchanged.
    """
    axis = structure.k_axis
    if axis is None or wanted is None:
        return array
    stored = structure.stored_kpoints(block)
    assert stored is not None
    if tuple(wanted) == stored:
        return array
    position = {k: index for index, k in enumerate(stored)}
    source = [
        position[k] if k in position else position[axis.partners[k]] for k in wanted
    ]
    flip = [k not in position for k in wanted]
    xp = array_namespace(array)
    taken = xp.take(array, xp.asarray(source, dtype=xp.int64), axis=0)
    if not any(flip):
        return taken
    mask = xp.reshape(
        xp.asarray(flip, dtype=xp.bool), (len(flip),) + (1,) * (array.ndim - 1)
    )
    return xp.where(mask, antiunitary(block.ring, taken, matrix=matrix), taken)


def materialize_at(
    structure: BlockStructure,
    stored: Sequence[Array],
    wanted: Sequence[int] | None,
    *,
    matrix: bool = True,
    orbitals: bool = False,
) -> tuple[Array, ...]:
    """One array per block of the spin pattern at the k-points ``wanted``.

    Derived blocks are built from their source: a copy at the same k, or
    the conjugate of the source at ``−k``. With irrep sectors the blocks of
    each sector are brought to the atomic basis and combined: operators as
    ``Σ_Γ U_Γ X_Γ U_Γᵀ``, orbital coefficients (``orbitals``) and vectors
    (``matrix`` false) side by side over the sectors, so the result has one
    array per block of :attr:`BlockStructure.sector_structure`.
    """
    sectors = structure.irreps
    if sectors is None:
        return _materialize_sector(structure, stored, wanted, matrix=matrix)
    base = structure.sector_structure
    per_sector = [
        _materialize_sector(
            base, stored[structure.sector_slice(sector)], wanted, matrix=matrix
        )
        for sector in range(sectors.n_irreps)
    ]
    arrays: list[Array] = []
    for index, block in enumerate(base.blocks):
        parts = [arrays_of_sector[index] for arrays_of_sector in per_sector]
        if not matrix:
            xp = array_namespace(*parts)
            arrays.append(xp.concat(parts, axis=-1))
            continue
        arrays.append(
            sectors.to_atomic(
                parts, spinor=block.spin is Spin.SPINOR, orbitals=orbitals
            )
        )
    return tuple(arrays)


def _materialize_sector(
    structure: BlockStructure,
    stored: Sequence[Array],
    wanted: Sequence[int] | None,
    *,
    matrix: bool,
) -> tuple[Array, ...]:
    """One array per block of a structure without sectors at ``wanted``."""
    axis = structure.k_axis
    positions: dict[int, int] = {}
    arrays: list[Array] = []
    for index, block in enumerate(structure.blocks):
        if block.derived_from is None:
            positions[index] = len(positions)
            source = stored[positions[index]]
            arrays.append(expand_k(structure, block, source, wanted, matrix=matrix))
            continue
        if block.partner is Partner.COPY:
            arrays.append(arrays[block.derived_from])
            continue
        source_block = structure.blocks[block.derived_from]
        source = stored[positions[block.derived_from]]
        flipped = wanted
        if axis is not None and wanted is not None:
            flipped = [axis.partners[k] for k in wanted]
        xp = array_namespace(source)
        arrays.append(
            xp.conj(expand_k(structure, source_block, source, flipped, matrix=matrix))
        )
    return tuple(arrays)


def materialize(
    structure: BlockStructure,
    stored: Sequence[Array],
    *,
    matrix: bool = True,
    orbitals: bool = False,
) -> tuple[Array, ...]:
    """Expand stored blocks to one array per spin block over the full k set."""
    return materialize_at(
        structure, stored, structure.all_kpoints, matrix=matrix, orbitals=orbitals
    )


def spin_blocks_at(
    structure: BlockStructure,
    stored: Sequence[Array],
    wanted: Sequence[int] | None,
    *,
    orbitals: bool = False,
) -> tuple[Array, ...]:
    """``(alpha, beta)`` blocks, or the single spinor block, at ``wanted``."""
    arrays = materialize_at(structure, stored, wanted, orbitals=orbitals)
    if structure.is_spatial:
        return (arrays[0], arrays[0])
    return arrays


def spin_blocks(
    structure: BlockStructure, stored: Sequence[Array], *, orbitals: bool = False
) -> tuple[Array, ...]:
    """``(alpha, beta)`` blocks, or the single spinor block, over the full k set."""
    return spin_blocks_at(structure, stored, structure.all_kpoints, orbitals=orbitals)


# ---------------------------------------------------------------- weighted sums


def weighted_total(structure: BlockStructure, values: Sequence[Array]) -> Array:
    """Sum per-block values over spins and k-points.

    ``values`` holds one array per stored block: a scalar without a k axis,
    ``[kpt]`` over the stored k-points otherwise. A stored half-zone point
    contributes its value with its own weight and the conjugate of its value
    with the weight of the derived partner (``tr(conj(X)) = conj(tr(X))``
    and ``tr(Θ X Θ⁻¹) = conj(tr(X))``). A derived spin block adds the total
    of its source, conjugated for a conjugate partner; a spatial block acts
    on both spins and counts twice.
    """
    xp = array_namespace(*values)
    totals: list[Array] = []
    for block, value in zip(structure.stored_blocks, values, strict=True):
        own, partner = structure.stored_weights(block)
        total = xp.sum(xp.asarray(own, dtype=xp.float64) * value)
        if any(partner):
            total = total + xp.sum(
                xp.asarray(partner, dtype=xp.float64) * xp.conj(value)
            )
        totals.append(total * block.weight)
    contributions: dict[int, Array] = {}
    result: Array | None = None
    for index, block in enumerate(structure.blocks):
        if block.derived_from is None:
            contribution = totals[len(contributions)]
            contributions[index] = contribution
        else:
            source = contributions[block.derived_from]
            contribution = (
                xp.conj(source) if block.partner is Partner.CONJUGATE else source
            )
        result = contribution if result is None else result + contribution
    assert result is not None
    if structure.is_spatial:
        result = 2 * result
    return xp.asarray(result)


# ------------------------------------------------------------------ dense form


def to_dense(
    structure: BlockStructure, stored: Sequence[Array], *, orbitals: bool = False
) -> Array:
    """Expand into the spin-orbital representation ``[(kpt,) 2n, m_alpha + m_beta]``.

    ``orbitals`` marks orbital coefficients, whose columns stay as they are
    when the rows of a matrix over irrep sectors are brought to the atomic
    basis; an operator is transformed on both sides.
    """
    parts = spin_blocks(structure, stored, orbitals=orbitals)
    if len(parts) == 1:
        return parts[0]
    return block_diag(parts[0], parts[1])


def column_split(
    structure: BlockStructure, stored: Sequence[Array]
) -> tuple[int, int] | None:
    """Column counts of the alpha and beta blocks; ``None`` for spinor matrices.

    With irrep sectors the columns of a spin block are summed over them.
    """
    if structure.is_spinor:
        return None
    size = structure.stored_per_sector
    columns = [
        sum(
            int(stored[sector * size + index].shape[-1])
            for sector in range(structure.n_sectors)
        )
        for index in range(size)
    ]
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
    (alpha and beta averaged, real part taken, Kramers-symmetrized, the
    block at ``k`` averaged with the partner map of the block at ``−k``,
    and the couplings between irrep sectors dropped). Only square
    operators enter a structure with irrep sectors: the columns of orbital
    coefficients carry no irrep to assign them by.
    """
    xp = array_namespace(dense)
    _check_dense_shape(structure, dense)
    sectors = structure.irreps
    if sectors is not None:
        if int(dense.shape[-2]) != int(dense.shape[-1]):
            raise StructureError(
                "only square operators enter a structure with irrep sectors from "
                "the dense form; orbital coefficients are built per sector"
            )
        base = structure.sector_structure
        sector_blocks: list[Array] = []
        for part in sectors.from_atomic(dense, spinor=True):
            sector_blocks.extend(
                blocks_from_dense(
                    base, part, column_split=None, check=check, tolerance=tolerance
                )
            )
        if check:
            require_close(
                xp,
                to_dense(structure, sector_blocks),
                dense,
                tolerance,
                f"the matrix couples the irreps of {sectors.group}, so it does "
                f"not have the {structure.describe()} structure",
            )
        return tuple(sector_blocks)
    n = half_dimension(
        int(dense.shape[-2]), "the number of rows of a spin-orbital matrix"
    )
    if structure.is_spinor:
        (block,) = structure.stored_blocks
        parts: tuple[Array, ...] = (
            _reduce_k(structure, block, dense, check=check, tolerance=tolerance),
        )
    else:
        parts = _collinear_from_dense(
            structure,
            dense,
            n,
            _split_columns(int(dense.shape[-1]), column_split),
            check=check,
            tolerance=tolerance,
        )
    cast = tuple(
        cast_to_ring(xp, part, structure.storage_ring(block))
        for part, block in zip(parts, structure.stored_blocks, strict=True)
    )
    if check:
        require_close(
            xp,
            to_dense(structure, cast),
            dense,
            tolerance,
            f"the matrix does not have the {structure.describe()} structure",
        )
    return cast


def _check_dense_shape(structure: BlockStructure, dense: Array) -> None:
    axis = structure.k_axis
    if axis is None:
        if dense.ndim != 2:
            raise StructureError("the dense representation is two-dimensional")
        return
    if dense.ndim != 3:
        raise StructureError(
            "the dense representation of a structure with a k axis is "
            "three-dimensional, [kpt, 2n, m]"
        )
    if int(dense.shape[0]) != axis.n_kpt:
        raise StructureError(
            f"the dense representation holds {int(dense.shape[0])} k-points, the "
            f"structure {axis.n_kpt}"
        )


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


def _collinear_from_dense(
    structure: BlockStructure,
    dense: Array,
    n: int,
    split: tuple[int, int],
    *,
    check: bool,
    tolerance: float,
) -> tuple[Array, ...]:
    m_alpha, m_beta = split
    alpha, beta = dense[..., :n, :m_alpha], dense[..., n:, m_alpha:]
    partner = structure.derived_partner
    blocks = structure.stored_blocks
    if partner is None and not structure.is_spatial:
        return tuple(
            _reduce_k(structure, block, part, check=check, tolerance=tolerance)
            for block, part in zip(blocks, (alpha, beta), strict=True)
        )
    if m_alpha != m_beta:
        raise StructureError("alpha and beta blocks must have equal shapes")
    if partner is Partner.CONJUGATE:
        xp = array_namespace(dense)
        axis = structure.k_axis
        if axis is not None:
            beta = xp.take(beta, xp.asarray(axis.partners, dtype=xp.int64), axis=0)
        return ((alpha + xp.conj(beta)) / 2,)
    return (
        _reduce_k(
            structure, blocks[0], (alpha + beta) / 2, check=check, tolerance=tolerance
        ),
    )


def _reduce_k(
    structure: BlockStructure,
    block: Block,
    full: Array,
    *,
    check: bool,
    tolerance: float,
) -> Array:
    """Reduce a block given at every k-point to what ``block`` stores.

    Blocks over the complex ring store everything. Real and quaternionic
    operators keep the half zone, each stored point averaged with the
    antiunitary image of its partner, which at a time-reversal invariant
    point is the real part or the Kramers symmetrization. Rectangular
    quaternionic blocks (coefficients) are only checked, at the invariant
    points, for a Θ-invariant column space.
    """
    if block.ring is Ring.COMPLEX:
        return full
    axis = structure.k_axis
    xp = array_namespace(full)
    stored = structure.stored_kpoints(block)
    if block.ring is Ring.QUATERNION and full.shape[-2] != full.shape[-1]:
        if not check:
            raise StructureError(
                "projecting a rectangular matrix onto a paired structure is not defined"
            )
        kept = (
            full
            if stored is None
            else xp.take(full, xp.asarray(stored, dtype=xp.int64), axis=0)
        )
        for position in structure.invariant_positions(block):
            matrix = kept if stored is None else kept[position]
            if not _has_kramers_invariant_range(matrix, tolerance):
                raise StructureError(
                    "the column space is not invariant under time reversal"
                )
        return kept
    if axis is None or stored is None:
        return (full + antiunitary(block.ring, full, matrix=True)) / 2
    own = xp.take(full, xp.asarray(stored, dtype=xp.int64), axis=0)
    partners = xp.asarray([axis.partners[k] for k in stored], dtype=xp.int64)
    mirrored = antiunitary(block.ring, xp.take(full, partners, axis=0), matrix=True)
    return (own + mirrored) / 2


def require_close(
    xp: Namespace, actual: Array, expected: Array, tolerance: float, what: str
) -> None:
    """Raise :class:`StructureError` unless ``actual`` matches ``expected``.

    The deviation is measured relative to the largest entry of ``expected``,
    at least one.
    """
    scale = max(1.0, float(xp.max(xp.abs(expected))))
    deviation = float(xp.max(xp.abs(actual - expected)))
    if deviation > tolerance * scale:
        raise StructureError(f"{what} (deviation {deviation:.2e})")


def _has_kramers_invariant_range(dense: Array, tolerance: float) -> bool:
    xp = array_namespace(dense)
    orthonormal, _ = xp.linalg.qr(xp.astype(dense, xp.complex128))
    projector = orthonormal @ adjoint(orthonormal)
    return is_kramers_symmetric(projector, tolerance=tolerance)

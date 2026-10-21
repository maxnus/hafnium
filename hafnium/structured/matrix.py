"""Block matrices and vectors over a :class:`Structure`.

The dense spin-orbital representation (``2n`` rows, alpha components first)
is the canonical form that relates all structures: :meth:`BlockMatrix.to_dense`
expands a matrix into it and :meth:`BlockMatrix.from_dense` extracts the
blocks of any structure from it. Embedding into a less symmetric structure
is lossless; projecting onto a more symmetric one averages.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from hafnium.backend import (
    Array,
    Namespace,
    adjoint,
    array_namespace,
    default_namespace,
    eigh,
    generalized_eigh,
)
from hafnium.errors import StructureError
from hafnium.structured.kramers import (
    is_kramers_symmetric,
    kramers_adapt_eigenvectors,
    kramers_symmetrize,
)
from hafnium.structured.structure import Partner, Ring, Spin, Structure

Scalar: TypeAlias = int | float | complex


def _check_block_count(structure: Structure, blocks: Sequence[Array]) -> None:
    if len(blocks) != structure.n_stored:
        raise StructureError(
            f"{structure.describe()} structure stores {structure.n_stored} blocks, "
            f"got {len(blocks)}"
        )


def _check_ring(xp: Namespace, array: Array, ring: Ring) -> None:
    if not xp.isdtype(array.dtype, ("real floating", "complex floating")):
        raise StructureError(f"blocks must be floating point, got {array.dtype}")
    is_complex = bool(xp.isdtype(array.dtype, "complex floating"))
    if ring is Ring.REAL and is_complex:
        raise StructureError(
            "a real block has complex entries; embed into a complex structure"
        )
    if ring is not Ring.REAL and not is_complex:
        raise StructureError(f"a {ring.value} block has real entries; cast explicitly")


def _half(dimension: int) -> int:
    if dimension % 2:
        raise StructureError("the dense representation has an even number of rows")
    return dimension // 2


def _materialize(structure: Structure, stored: Sequence[Array]) -> tuple[Array, ...]:
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
            source = array_namespace(source).conj(source)
        arrays.append(source)
    return tuple(arrays)


def _block_diag(a: Array, b: Array) -> Array:
    xp = array_namespace(a, b)
    dtype = xp.result_type(a, b)
    top = xp.concat(
        [xp.astype(a, dtype), xp.zeros((a.shape[0], b.shape[1]), dtype=dtype)], axis=1
    )
    bottom = xp.concat(
        [xp.zeros((b.shape[0], a.shape[1]), dtype=dtype), xp.astype(b, dtype)], axis=1
    )
    return xp.concat([top, bottom], axis=0)


def _cast_to_ring(xp: Namespace, array: Array, ring: Ring) -> Array:
    if ring is Ring.REAL:
        return xp.real(array) if xp.isdtype(array.dtype, "complex floating") else array
    return xp.astype(array, xp.complex128)


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


@dataclass(frozen=True, slots=True, eq=False)
class BlockVector:
    """One-dimensional data per stored block, such as eigenvalues or occupations."""

    structure: Structure
    blocks: tuple[Array, ...]

    def __post_init__(self) -> None:
        _check_block_count(self.structure, self.blocks)
        for array in self.blocks:
            if array.ndim != 1:
                raise StructureError("vector blocks are one-dimensional")

    def materialized(self) -> tuple[Array, ...]:
        """One array per block of the structure, derived blocks included."""
        return _materialize(self.structure, self.blocks)

    def to_dense(self) -> Array:
        """Concatenation over the spin-orbital space (alpha first)."""
        parts = self.materialized()
        if self.structure.is_spatial:
            parts = (parts[0], parts[0])
        return array_namespace(*parts).concat(parts, axis=0)

    def map_blocks(self, function: Callable[[Array], Array]) -> BlockVector:
        """Apply ``function`` to every stored block."""
        return BlockVector(
            self.structure, tuple(function(block) for block in self.blocks)
        )


@dataclass(frozen=True, slots=True, eq=False)
class BlockMatrix:
    """A one-body operator, density or coefficient matrix in a :class:`Structure`.

    ``blocks`` holds one array per stored block of ``structure``, in order.
    Spatial and collinear blocks have shape ``[n, m]``; spinor blocks have
    shape ``[2n, 2m]`` with alpha components first. Real-ring blocks are
    float64, the others complex128; the constructor rejects mismatches, so
    changing ring or structure is always an explicit :meth:`embed` or
    :meth:`project`.
    """

    structure: Structure
    blocks: tuple[Array, ...]

    def __post_init__(self) -> None:
        _check_block_count(self.structure, self.blocks)
        xp = array_namespace(*self.blocks)
        for block, array in zip(self.structure.stored_blocks, self.blocks, strict=True):
            if array.ndim != 2:
                raise StructureError("matrix blocks are two-dimensional")
            if block.spin is Spin.SPINOR and (array.shape[0] % 2 or array.shape[1] % 2):
                raise StructureError("spinor blocks have even dimensions")
            _check_ring(xp, array, block.ring)

    # ------------------------------------------------------------------ building

    @classmethod
    def zeros(
        cls,
        structure: Structure,
        shape: tuple[int, int],
        *,
        xp: Namespace | None = None,
    ) -> BlockMatrix:
        """All-zero matrix with spatial shape ``shape`` (doubled for spinor blocks)."""
        xp = default_namespace() if xp is None else xp
        blocks: list[Array] = []
        for block in structure.stored_blocks:
            factor = 2 if block.spin is Spin.SPINOR else 1
            dtype = xp.float64 if block.ring is Ring.REAL else xp.complex128
            blocks.append(xp.zeros((factor * shape[0], factor * shape[1]), dtype=dtype))
        return cls(structure, tuple(blocks))

    @classmethod
    def identity(
        cls, structure: Structure, n: int, *, xp: Namespace | None = None
    ) -> BlockMatrix:
        """Identity matrix with ``n`` spatial basis functions."""
        xp = default_namespace() if xp is None else xp
        blocks: list[Array] = []
        for block in structure.stored_blocks:
            size = 2 * n if block.spin is Spin.SPINOR else n
            dtype = xp.float64 if block.ring is Ring.REAL else xp.complex128
            blocks.append(xp.eye(size, dtype=dtype))
        return cls(structure, tuple(blocks))

    def map_blocks(self, function: Callable[[Array], Array]) -> BlockMatrix:
        """Apply ``function`` to every stored block."""
        return BlockMatrix(
            self.structure, tuple(function(block) for block in self.blocks)
        )

    @property
    def shapes(self) -> tuple[tuple[int, int], ...]:
        """Shape of each stored block."""
        return tuple(
            (int(block.shape[0]), int(block.shape[1])) for block in self.blocks
        )

    @property
    def n_spatial(self) -> int:
        """Number of spatial basis functions in the row space."""
        rows = int(self.blocks[0].shape[0])
        return rows // 2 if self.structure.is_spinor else rows

    # ------------------------------------------------------------------ algebra

    def _require_same_structure(self, other: BlockMatrix, operation: str) -> None:
        if self.structure != other.structure:
            raise StructureError(
                f"cannot {operation} a {self.structure.describe()} matrix and a "
                f"{other.structure.describe()} matrix; embed one into the other first"
            )

    def __add__(self, other: BlockMatrix) -> BlockMatrix:
        self._require_same_structure(other, "add")
        return BlockMatrix(
            self.structure,
            tuple(a + b for a, b in zip(self.blocks, other.blocks, strict=True)),
        )

    def __sub__(self, other: BlockMatrix) -> BlockMatrix:
        self._require_same_structure(other, "subtract")
        return BlockMatrix(
            self.structure,
            tuple(a - b for a, b in zip(self.blocks, other.blocks, strict=True)),
        )

    def __neg__(self) -> BlockMatrix:
        return self.map_blocks(lambda block: -block)

    def __mul__(self, other: Scalar) -> BlockMatrix:
        return self.map_blocks(lambda block: block * other)

    def __rmul__(self, other: Scalar) -> BlockMatrix:
        return self.map_blocks(lambda block: other * block)

    def __truediv__(self, other: Scalar) -> BlockMatrix:
        return self.map_blocks(lambda block: block / other)

    def __matmul__(self, other: BlockMatrix) -> BlockMatrix:
        self._require_same_structure(other, "multiply")
        return BlockMatrix(
            self.structure,
            tuple(a @ b for a, b in zip(self.blocks, other.blocks, strict=True)),
        )

    def adjoint(self) -> BlockMatrix:
        """Conjugate transpose of every block."""
        return self.map_blocks(adjoint)

    def conj(self) -> BlockMatrix:
        """Complex conjugate of every block."""
        return self.map_blocks(lambda block: array_namespace(block).conj(block))

    def transpose(self) -> BlockMatrix:
        """Transpose of every block."""
        return self.map_blocks(
            lambda block: array_namespace(block).linalg.matrix_transpose(block)
        )

    # ------------------------------------------------------------------ traces

    def trace(self) -> Array:
        """Trace over the full spin-orbital space, as a 0-d array."""
        xp = array_namespace(*self.blocks)
        parts = self.spin_blocks()
        total = sum(xp.linalg.trace(part) for part in parts)
        return xp.asarray(total)

    def trace_product(self, other: BlockMatrix) -> Array:
        """``trace(self @ other)`` over the spin-orbital space.

        The product itself is not formed.
        """
        self._require_same_structure(other, "trace-multiply")
        xp = array_namespace(*self.blocks)
        total = sum(
            xp.sum(a * xp.linalg.matrix_transpose(b))
            for a, b in zip(self.spin_blocks(), other.spin_blocks(), strict=True)
        )
        return xp.asarray(total)

    # ------------------------------------------------------------------ dense form

    def materialized(self) -> tuple[Array, ...]:
        """One array per block of the structure, derived blocks included."""
        return _materialize(self.structure, self.blocks)

    def spin_blocks(self) -> tuple[Array, ...]:
        """``(alpha, beta)`` blocks, or the single spinor block."""
        arrays = self.materialized()
        if self.structure.is_spatial:
            return (arrays[0], arrays[0])
        if self.structure.is_spinor:
            return (arrays[0],)
        by_spin = dict(
            zip((block.spin for block in self.structure.blocks), arrays, strict=True)
        )
        return (by_spin[Spin.ALPHA], by_spin[Spin.BETA])

    @property
    def column_split(self) -> tuple[int, int] | None:
        """Column counts of the alpha and beta blocks; ``None`` for spinor matrices."""
        parts = self.spin_blocks()
        if len(parts) == 1:
            return None
        return int(parts[0].shape[1]), int(parts[1].shape[1])

    def to_dense(self) -> Array:
        """Expand into the spin-orbital representation ``[2n, m_alpha + m_beta]``."""
        parts = self.spin_blocks()
        if len(parts) == 1:
            return parts[0]
        return _block_diag(parts[0], parts[1])

    @classmethod
    def from_dense(
        cls,
        structure: Structure,
        dense: Array,
        *,
        column_split: tuple[int, int] | None = None,
        check: bool = True,
        tolerance: float = 1e-10,
    ) -> BlockMatrix:
        """Extract the blocks of ``structure`` from a spin-orbital matrix.

        Parameters
        ----------
        dense : Array[2n, m]
            Spin-orbital matrix, alpha components first.
        column_split
            Column counts of the alpha and beta parts for collinear targets
            with a rectangular ``dense``; defaults to equal halves.
        check
            If true, raise :class:`StructureError` unless ``dense`` has the
            requested structure to within ``tolerance`` (a lossless
            embedding). If false, project: average alpha and beta, take the
            real part, or Kramers-symmetrize as the structure requires.
        """
        xp = array_namespace(dense)
        if dense.ndim != 2:
            raise StructureError("the dense representation is two-dimensional")
        rows, cols = int(dense.shape[0]), int(dense.shape[1])
        n = _half(rows)
        if structure.is_spinor:
            (block,) = structure.stored_blocks
            stored = dense
            if block.ring is Ring.QUATERNION:
                if rows == cols:
                    stored = kramers_symmetrize(xp.astype(dense, xp.complex128))
                elif not check:
                    raise StructureError(
                        "projecting a rectangular matrix onto a paired structure "
                        "is not defined"
                    )
                elif not _has_kramers_invariant_range(dense, tolerance):
                    raise StructureError(
                        "the column space is not invariant under time reversal"
                    )
            parts: tuple[Array, ...] = (stored,)
        else:
            m_alpha, m_beta = (
                (_half(cols), _half(cols)) if column_split is None else column_split
            )
            if m_alpha + m_beta != cols:
                raise StructureError("column_split must sum to the number of columns")
            alpha, beta = dense[:n, :m_alpha], dense[n:, m_alpha:]
            if structure.is_spatial or structure.derived_partner is Partner.COPY:
                if m_alpha != m_beta:
                    raise StructureError("alpha and beta blocks must have equal shapes")
                parts = ((alpha + beta) / 2,)
            elif structure.derived_partner is Partner.CONJUGATE:
                if m_alpha != m_beta:
                    raise StructureError("alpha and beta blocks must have equal shapes")
                parts = ((alpha + xp.conj(beta)) / 2,)
            else:
                parts = (alpha, beta)
        cast = tuple(
            _cast_to_ring(xp, part, block.ring)
            for part, block in zip(parts, structure.stored_blocks, strict=True)
        )
        result = cls(structure, cast)
        if check:
            _require_close(
                xp,
                result.to_dense(),
                dense,
                tolerance,
                f"the matrix does not have the {structure.describe()} structure",
            )
        return result

    def embed(self, target: Structure) -> BlockMatrix:
        """Re-express the matrix in a less symmetric structure, losslessly.

        Raises
        ------
        StructureError
            If the matrix cannot be represented exactly in ``target``.
        """
        return BlockMatrix.from_dense(
            target, self.to_dense(), column_split=self.column_split, check=True
        )

    def project(self, target: Structure) -> BlockMatrix:
        """Project the matrix onto a more symmetric structure by averaging."""
        return BlockMatrix.from_dense(
            target, self.to_dense(), column_split=self.column_split, check=False
        )

    def spin_trace(self) -> BlockMatrix:
        """Sum over the spin diagonal, as a spatial matrix.

        For a density matrix this is the total (spin-summed) density.
        """
        parts = self.spin_blocks()
        if len(parts) == 2:
            total = parts[0] + parts[1]
        else:
            n, m = _half(int(parts[0].shape[0])), _half(int(parts[0].shape[1]))
            total = parts[0][:n, :m] + parts[0][n:, m:]
        xp = array_namespace(total)
        ring = (
            Ring.COMPLEX if xp.isdtype(total.dtype, "complex floating") else Ring.REAL
        )
        return BlockMatrix(Structure.spatial(ring), (total,))

    # ------------------------------------------------------------------ eigenproblem

    def eigh(
        self, overlap: BlockMatrix | None = None
    ) -> tuple[BlockVector, BlockMatrix]:
        """Diagonalize each block of a Hermitian matrix.

        With ``overlap`` (same structure) the generalized problem
        ``F C = S C ε`` is solved and the eigenvectors are orthonormal in the
        overlap metric. Quaternionic blocks return Kramers-paired
        eigenvectors in adjacent columns.
        """
        if overlap is not None:
            self._require_same_structure(overlap, "diagonalize with the overlap of")
        values: list[Array] = []
        vectors: list[Array] = []
        for index, (block, array) in enumerate(
            zip(self.structure.stored_blocks, self.blocks, strict=True)
        ):
            adapt = (
                kramers_adapt_eigenvectors if block.ring is Ring.QUATERNION else None
            )
            if overlap is None:
                w, v = eigh(array)
                if adapt is not None:
                    v = adapt(w, v)
            else:
                w, v = generalized_eigh(array, overlap.blocks[index], adapt=adapt)
            values.append(w)
            vectors.append(v)
        return BlockVector(self.structure, tuple(values)), BlockMatrix(
            self.structure, tuple(vectors)
        )


def density_matrix(coefficients: BlockMatrix, occupations: BlockVector) -> BlockMatrix:
    """Return ``C diag(occupations) adjoint(C)`` block by block.

    Parameters
    ----------
    coefficients : BlockMatrix
        Orbital coefficients ``[ao, mo]`` per block.
    occupations : BlockVector
        Occupation numbers ``[mo]`` per block, as floats.
    """
    if coefficients.structure != occupations.structure:
        raise StructureError("coefficients and occupations must share a structure")
    blocks: list[Array] = []
    for c, occupation in zip(coefficients.blocks, occupations.blocks, strict=True):
        xp = array_namespace(c, occupation)
        blocks.append((c * xp.expand_dims(occupation, axis=0)) @ adjoint(c))
    return BlockMatrix(coefficients.structure, tuple(blocks))

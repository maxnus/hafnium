"""Block matrices: one-body operators, densities and coefficients in a structure."""

from __future__ import annotations

from collections.abc import Callable
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
from hafnium.structured import dense, embedding
from hafnium.structured.kramers import (
    half_dimension,
    is_kramers_symmetric,
    kramers_adapt_eigenvectors,
)
from hafnium.structured.structure import BlockStructure, Ring, Spin
from hafnium.structured.validation import check_block_count, check_ring, dtype_for
from hafnium.structured.vector import BlockVector

Scalar: TypeAlias = int | float | complex


def _require_scalar(other: object) -> None:
    if isinstance(other, BlockMatrix | BlockVector):
        raise StructureError(
            "elementwise products of block matrices are not defined; use @ for "
            "matrix products or trace_product for tr(A B)"
        )


@dataclass(frozen=True, slots=True, eq=False)
class BlockMatrix:
    """A one-body operator, density or coefficient matrix in a :class:`BlockStructure`.

    ``blocks`` holds one array per stored block of ``structure``, in order.
    Spatial and collinear blocks have shape ``[n, m]``; spinor blocks have
    shape ``[2n, 2m]`` with alpha components first. Real-ring blocks are
    float64, the others complex128; the constructor rejects mismatches, so
    changing ring or structure is always an explicit :meth:`embed` or
    :meth:`project`.

    A quaternionic block holds either a Θ-symmetric operator or a coefficient
    matrix whose columns come in Kramers pairs. The constructor cannot tell
    the two apart by shape, so Θ-symmetry is verified where it matters: in
    :meth:`eigh` and in :func:`hafnium.structured.density_matrix`.
    """

    structure: BlockStructure
    blocks: tuple[Array, ...]

    def __post_init__(self) -> None:
        check_block_count(self.structure, self.blocks)
        xp = array_namespace(*self.blocks)
        for block, array in zip(self.structure.stored_blocks, self.blocks, strict=True):
            if array.ndim != 2:
                raise StructureError("matrix blocks are two-dimensional")
            if block.spin is Spin.SPINOR and (array.shape[0] % 2 or array.shape[1] % 2):
                raise StructureError("spinor blocks have even dimensions")
            check_ring(xp, array, block.ring)
        rows = {int(array.shape[0]) for array in self.blocks}
        if len(rows) > 1:
            raise StructureError(
                "all blocks act on the same basis, but row counts differ: "
                f"{sorted(rows)}"
            )

    # ------------------------------------------------------------------ building

    @classmethod
    def zeros(
        cls,
        structure: BlockStructure,
        shape: tuple[int, int],
        *,
        xp: Namespace | None = None,
    ) -> BlockMatrix:
        """All-zero matrix with spatial shape ``shape`` (doubled for spinor blocks)."""
        xp = default_namespace() if xp is None else xp
        blocks: list[Array] = []
        for block in structure.stored_blocks:
            factor = 2 if block.spin is Spin.SPINOR else 1
            block_shape = (factor * shape[0], factor * shape[1])
            blocks.append(xp.zeros(block_shape, dtype=dtype_for(xp, block.ring)))
        return cls(structure, tuple(blocks))

    @classmethod
    def identity(
        cls, structure: BlockStructure, n: int, *, xp: Namespace | None = None
    ) -> BlockMatrix:
        """Identity matrix with ``n`` spatial basis functions."""
        xp = default_namespace() if xp is None else xp
        blocks: list[Array] = []
        for block in structure.stored_blocks:
            size = 2 * n if block.spin is Spin.SPINOR else n
            blocks.append(xp.eye(size, dtype=dtype_for(xp, block.ring)))
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
        _require_scalar(other)
        return self.map_blocks(lambda block: block * other)

    def __rmul__(self, other: Scalar) -> BlockMatrix:
        _require_scalar(other)
        return self.map_blocks(lambda block: other * block)

    def __truediv__(self, other: Scalar) -> BlockMatrix:
        _require_scalar(other)
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
        xp = array_namespace(*self.blocks)
        return self.map_blocks(xp.conj)

    def transpose(self) -> BlockMatrix:
        """Transpose of every block."""
        xp = array_namespace(*self.blocks)
        return self.map_blocks(xp.linalg.matrix_transpose)

    # ------------------------------------------------------------------ traces

    def trace(self) -> Array:
        """Trace over the full spin-orbital space, as a 0-d array."""
        xp = array_namespace(*self.blocks)
        total = sum(xp.linalg.trace(part) for part in self.spin_blocks())
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
        return dense.materialize(self.structure, self.blocks)

    def spin_blocks(self) -> tuple[Array, ...]:
        """``(alpha, beta)`` blocks, or the single spinor block."""
        return dense.spin_blocks(self.structure, self.blocks)

    @property
    def column_split(self) -> tuple[int, int] | None:
        """Column counts of the alpha and beta blocks; ``None`` for spinor matrices."""
        return dense.column_split(self.structure, self.blocks)

    def to_dense(self) -> Array:
        """Expand into the spin-orbital representation ``[2n, m_alpha + m_beta]``."""
        return dense.to_dense(self.structure, self.blocks)

    @classmethod
    def from_dense(
        cls,
        structure: BlockStructure,
        matrix: Array,
        *,
        column_split: tuple[int, int] | None = None,
        check: bool = True,
        tolerance: float = 1e-10,
    ) -> BlockMatrix:
        """Extract the blocks of ``structure`` from a spin-orbital matrix.

        Parameters
        ----------
        matrix : Array[2n, m]
            Spin-orbital matrix, alpha components first.
        column_split
            Column counts of the alpha and beta parts for collinear targets
            with a rectangular ``matrix``; defaults to equal halves.
        check
            If true, raise :class:`StructureError` unless ``matrix`` has the
            requested structure to within ``tolerance`` (a lossless
            embedding). If false, project: average alpha and beta, take the
            real part, or Kramers-symmetrize as the structure requires.
        """
        blocks = dense.blocks_from_dense(
            structure,
            matrix,
            column_split=column_split,
            check=check,
            tolerance=tolerance,
        )
        return cls(structure, blocks)

    def embed(self, target: BlockStructure) -> BlockMatrix:
        """Re-express the matrix in a less symmetric structure, losslessly.

        Pairs within the Fukutome hierarchy are converted block by block.
        Other pairs go through the dense representation and are verified.

        Raises
        ------
        StructureError
            If the matrix cannot be represented exactly in ``target``.
        """
        if target == self.structure:
            return self
        if embedding.is_structural_embedding(self.structure, target):
            return BlockMatrix(
                target, embedding.embed_blocks(self.structure, self.blocks, target)
            )
        return BlockMatrix.from_dense(
            target, self.to_dense(), column_split=self.column_split, check=True
        )

    def project(self, target: BlockStructure) -> BlockMatrix:
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
            n = half_dimension(int(parts[0].shape[0]), "the number of spinor rows")
            m = half_dimension(int(parts[0].shape[1]), "the number of spinor columns")
            total = parts[0][:n, :m] + parts[0][n:, m:]
        xp = array_namespace(total)
        is_complex = xp.isdtype(total.dtype, "complex floating")
        ring = Ring.COMPLEX if is_complex else Ring.REAL
        return BlockMatrix(BlockStructure.spatial(ring), (total,))

    # ------------------------------------------------------------------ eigenproblem

    def eigh(
        self, overlap: BlockMatrix | None = None
    ) -> tuple[BlockVector, BlockMatrix]:
        """Diagonalize each block of a Hermitian matrix.

        With ``overlap`` (same structure) the generalized problem
        ``F C = S C ε`` is solved and the eigenvectors are orthonormal in the
        overlap metric. Quaternionic blocks must be Θ-symmetric and return
        Kramers-paired eigenvectors in adjacent columns.
        """
        if overlap is not None:
            self._require_same_structure(overlap, "diagonalize with the overlap of")
        values: list[Array] = []
        vectors: list[Array] = []
        for index, (block, array) in enumerate(
            zip(self.structure.stored_blocks, self.blocks, strict=True)
        ):
            adapt = _kramers_adaptation(block.ring, array)
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


def _kramers_adaptation(
    ring: Ring, array: Array
) -> Callable[[Array, Array], Array] | None:
    """Return the eigenvector post-processing a block needs, after validating it."""
    if ring is not Ring.QUATERNION:
        return None
    if not is_kramers_symmetric(array):
        raise StructureError(
            "a paired (quaternionic) block must be Θ-symmetric to be diagonalized; "
            "build it with from_dense or kramers_symmetrize"
        )
    return kramers_adapt_eigenvectors

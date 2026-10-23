"""Block matrices: one-body operators, densities and coefficients in a structure."""

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
from hafnium.structured import dense, embedding
from hafnium.structured.kaxis import KAxis
from hafnium.structured.kramers import (
    half_dimension,
    is_kramers_symmetric,
    kramers_adapt_eigenvectors,
)
from hafnium.structured.sectors import IrrepSectors
from hafnium.structured.structure import Block, BlockStructure, Ring, Spin
from hafnium.structured.validation import (
    cast_to_ring,
    check_block_count,
    check_k_count,
    check_ring,
    dtype_for,
)
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
    shape ``[2n, 2m]`` with alpha components first. With a k axis every
    block carries a leading axis over the k-points it stores,
    ``[kpt, n, m]`` (ADR-0015). With irrep sectors the blocks of a sector
    act on its SALCs, ``[salc, m]`` with as many rows as the sector has
    functions (ADR-0023). Real-ring blocks are float64, the others
    complex128, except that a real block over k-points that are not all
    time-reversal invariant is complex128; the constructor rejects
    mismatches, so changing ring or structure is always an explicit
    :meth:`embed` or :meth:`project`.

    A quaternionic block holds either a Θ-symmetric operator or a coefficient
    matrix whose columns come in Kramers pairs. The constructor cannot tell
    the two apart by shape, so Θ-symmetry is verified where it matters: in
    :meth:`eigh` and in :func:`hafnium.structured.density_matrix`, at the
    time-reversal invariant k-points. The derived block at ``−k`` of a
    half-zone block follows the operator rule for square blocks and the
    column rule for rectangular ones; a square coefficient matrix must not
    be materialized through the dense form.
    """

    structure: BlockStructure
    blocks: tuple[Array, ...]

    def __post_init__(self) -> None:
        check_block_count(self.structure, self.blocks)
        xp = array_namespace(*self.blocks)
        expected = 2 if self.structure.k_axis is None else 3
        for block, array in zip(self.structure.stored_blocks, self.blocks, strict=True):
            if array.ndim != expected:
                raise StructureError(
                    "matrix blocks are two-dimensional"
                    if expected == 2
                    else "matrix blocks with a k axis are [kpt, n, m]"
                )
            check_k_count(self.structure, block, array)
            if block.spin is Spin.SPINOR and (
                array.shape[-2] % 2 or array.shape[-1] % 2
            ):
                raise StructureError("spinor blocks have even dimensions")
            check_ring(xp, array, self.structure, block)
        sectors = self.structure.irreps
        for sector in range(self.structure.n_sectors):
            arrays = self.blocks[self.structure.sector_slice(sector)]
            rows = {int(array.shape[-2]) for array in arrays}
            if len(rows) > 1:
                where = (
                    ""
                    if sectors is None
                    else f" of the {sectors.labels[sector]} sector"
                )
                raise StructureError(
                    f"all blocks{where} act on the same basis, but row counts differ: "
                    f"{sorted(rows)}"
                )
            if sectors is None:
                continue
            factor = 2 if self.structure.is_spinor else 1
            expected = factor * sectors.dimensions[sector]
            if rows != {expected}:
                raise StructureError(
                    f"the blocks of the {sectors.labels[sector]} sector act on its "
                    f"{expected} functions, got {rows.pop()} rows"
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
        """All-zero matrix with spatial shape ``shape`` (doubled for spinor blocks).

        With irrep sectors the shape is that of a square operator over the
        atomic basis, and each sector's blocks are square over its SALCs.
        """
        xp = default_namespace() if xp is None else xp
        blocks: list[Array] = []
        for block, (rows, columns) in zip(
            structure.stored_blocks, _sector_shapes(structure, shape), strict=True
        ):
            factor = 2 if block.spin is Spin.SPINOR else 1
            block_shape = _batched(structure, block, (factor * rows, factor * columns))
            blocks.append(
                xp.zeros(
                    block_shape, dtype=dtype_for(xp, structure.storage_ring(block))
                )
            )
        return cls(structure, tuple(blocks))

    @classmethod
    def identity(
        cls, structure: BlockStructure, n: int, *, xp: Namespace | None = None
    ) -> BlockMatrix:
        """Identity matrix with ``n`` spatial basis functions, at every stored k."""
        xp = default_namespace() if xp is None else xp
        blocks: list[Array] = []
        for block, (rows, _) in zip(
            structure.stored_blocks, _sector_shapes(structure, (n, n)), strict=True
        ):
            size = 2 * rows if block.spin is Spin.SPINOR else rows
            eye = xp.eye(size, dtype=dtype_for(xp, structure.storage_ring(block)))
            stored = structure.stored_kpoints(block)
            if stored is not None:
                eye = xp.stack([eye] * len(stored), axis=0)
            blocks.append(eye)
        return cls(structure, tuple(blocks))

    @classmethod
    def from_spatial(
        cls,
        matrix: Array,
        k_axis: KAxis | None = None,
        *,
        ring: Ring = Ring.REAL,
        tolerance: float = 1e-10,
        irreps: IrrepSectors | None = None,
        check: bool = True,
    ) -> BlockMatrix:
        """Build a spatial operator from its matrix, ``[n, n]`` or ``[kpt, n, n]``.

        With ``k_axis`` the matrix runs over the full k set. The structure
        is :meth:`BlockStructure.spatial` over ``ring`` with the axis, and
        the matrix is reduced to the points that structure stores
        (ADR-0015, ADR-0024): each stored point is averaged over the points
        derived from it, mapped back by their point operations and by time
        reversal. With ``check`` the derived points must reproduce the
        matrix to within ``tolerance``, relative to the largest entry, so a
        real ring requires the matrix at ``−k`` to be the complex conjugate
        of the matrix at ``k``, which makes it real at the time-reversal
        invariant points, and an axis with point operations requires the
        matrix at ``Rk`` to be the rotation of the matrix at ``k``; the
        overlap, the core Hamiltonian and the Coulomb matrix of a symmetric
        density have these properties. Without ``check`` the matrix is
        projected. With ``irreps`` the matrix is projected onto the sectors
        (ADR-0023): the blocks ``U_Γᵀ X U_Γ`` are kept and the couplings
        between irreps dropped, which vanish for an operator that commutes
        with the group and are bounded by the tolerance of the symmetry
        adaptation otherwise.

        Raises
        ------
        StructureError
            If the shape does not fit the axis, or ``check`` finds the
            matrix without the symmetry of the ring and the axis.
        """
        structure = BlockStructure.spatial(ring).with_k_axis(k_axis)
        (block,) = structure.stored_blocks
        xp = array_namespace(matrix)
        if k_axis is None:
            if matrix.ndim != 2:
                raise StructureError(
                    "a spatial matrix without a k axis is two-dimensional, [n, m]"
                )
        elif matrix.ndim != 3 or int(matrix.shape[0]) != k_axis.n_kpt:
            raise StructureError(
                "a spatial matrix over a k axis is [kpt, n, m] with "
                f"{k_axis.n_kpt} k-points"
            )
        kept = dense.reduce_k(
            structure, block, matrix, check=False, tolerance=tolerance
        )
        if check:
            symmetry = (
                "time-reversal symmetric"
                if k_axis is None or k_axis.operations is None
                else "symmetric under time reversal and the point operations"
            )
            dense.require_close(
                xp,
                dense.expand_k(structure, block, kept, structure.all_kpoints),
                matrix,
                tolerance,
                f"the matrix is not {symmetry}, so it is not a {ring.value} spatial "
                "operator over this k axis; use the complex ring, or project with "
                "check=False",
            )
        storage = structure.storage_ring(block)
        if irreps is None:
            return cls(structure, (cast_to_ring(xp, kept, storage),))
        return cls(
            structure.with_irreps(irreps),
            tuple(
                cast_to_ring(xp, part, storage)
                for part in irreps.from_atomic(kept, spinor=False)
            ),
        )

    @classmethod
    def from_sectors(
        cls, structure: BlockStructure, parts: Sequence[BlockMatrix]
    ) -> BlockMatrix:
        """Assemble a matrix over the irrep sectors of ``structure`` from its sectors.

        ``parts`` holds one matrix per sector in the structure's
        :attr:`~BlockStructure.sector_structure`; see :meth:`sectors`.
        """
        if len(parts) != structure.n_sectors:
            raise StructureError(
                f"the {structure.describe()} structure has {structure.n_sectors} "
                f"sectors, got {len(parts)} matrices"
            )
        base = structure.sector_structure
        blocks: list[Array] = []
        for part in parts:
            if part.structure != base:
                raise StructureError(
                    f"a sector of the {structure.describe()} structure has the "
                    f"{base.describe()} structure, got {part.structure.describe()}"
                )
            blocks.extend(part.blocks)
        return cls(structure, tuple(blocks))

    def sectors(self) -> tuple[BlockMatrix, ...]:
        """Return the matrix of every irrep sector, in the sector structure.

        A matrix without sectors is its own single sector.
        """
        base = self.structure.sector_structure
        return tuple(
            BlockMatrix(base, self.blocks[self.structure.sector_slice(sector)])
            for sector in range(self.structure.n_sectors)
        )

    def map_blocks(self, function: Callable[[Array], Array]) -> BlockMatrix:
        """Apply ``function`` to every stored block."""
        return BlockMatrix(
            self.structure, tuple(function(block) for block in self.blocks)
        )

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        """Shape of each stored block."""
        return tuple(tuple(int(size) for size in block.shape) for block in self.blocks)

    @property
    def n_spatial(self) -> int:
        """Number of spatial basis functions in the row space, over all sectors."""
        size = self.structure.stored_per_sector
        rows = sum(
            int(self.blocks[sector * size].shape[-2])
            for sector in range(self.structure.n_sectors)
        )
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
        """Complex conjugate of every block, the conjugate of the dense form.

        Raises
        ------
        StructureError
            Over a k axis with point operations: the conjugate of a matrix
            at ``Rk`` is the rotation of the conjugate at ``k`` by the
            conjugate Bloch phases, those of ``−k``, so the elementwise
            conjugate is not a member of the structure (``adjoint`` is).
        """
        self._require_no_operations("conjugate")
        xp = array_namespace(*self.blocks)
        return self.map_blocks(xp.conj)

    def transpose(self) -> BlockMatrix:
        """Transpose of every block, the transpose of the dense form.

        Raises
        ------
        StructureError
            Over a k axis with point operations, as for :meth:`conj`.
        """
        self._require_no_operations("transpose")
        xp = array_namespace(*self.blocks)
        return self.map_blocks(xp.linalg.matrix_transpose)

    def _require_no_operations(self, operation: str) -> None:
        axis = self.structure.k_axis
        if axis is not None and axis.operations is not None:
            raise StructureError(
                f"the elementwise {operation} of a matrix over the wedge of "
                f"{axis.operations.group} is not a member of its structure, since "
                "the Bloch phases of the rotated points are complex; take the "
                "adjoint, or work on the dense form"
            )

    # ------------------------------------------------------------------ traces

    def trace(self) -> Array:
        """Trace over the spin-orbital space and the k-point weights, as a 0-d array."""
        xp = array_namespace(*self.blocks)
        return dense.weighted_total(
            self.structure, [xp.linalg.trace(block) for block in self.blocks]
        )

    def trace_product(self, other: BlockMatrix) -> Array:
        """``trace(self @ other)`` over the spin-orbital space and the k weights.

        The product itself is not formed.
        """
        self._require_same_structure(other, "trace-multiply")
        xp = array_namespace(*self.blocks)
        return dense.weighted_total(
            self.structure,
            [
                xp.sum(a * xp.linalg.matrix_transpose(b), axis=(-2, -1))
                for a, b in zip(self.blocks, other.blocks, strict=True)
            ],
        )

    # ------------------------------------------------------------------ dense form

    def materialized(self, *, orbitals: bool = False) -> tuple[Array, ...]:
        """One array per spin block over the full k set, derived blocks included.

        With irrep sectors the arrays are in the atomic basis (ADR-0023):
        ``orbitals`` marks orbital coefficients, whose rows alone are
        transformed; square blocks are otherwise taken as operators and
        transformed on both sides.
        """
        return dense.materialize(self.structure, self.blocks, orbitals=orbitals)

    def spin_blocks(self, *, orbitals: bool = False) -> tuple[Array, ...]:
        """``(alpha, beta)`` blocks, or the single spinor block, over the full k set.

        See :meth:`materialized` for ``orbitals``.
        """
        return dense.spin_blocks(self.structure, self.blocks, orbitals=orbitals)

    @property
    def column_split(self) -> tuple[int, int] | None:
        """Column counts of the alpha and beta blocks; ``None`` for spinor matrices."""
        return dense.column_split(self.structure, self.blocks)

    def to_dense(self, *, orbitals: bool = False) -> Array:
        """Expand into the spin-orbital form ``[(kpt,) 2n, m_alpha + m_beta]``.

        See :meth:`materialized` for ``orbitals``.
        """
        return dense.to_dense(self.structure, self.blocks, orbitals=orbitals)

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
            Spin-orbital matrix, alpha components first; ``[kpt, 2n, m]``
            over the full k set for a structure with a k axis.
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

        Pairs within the Fukutome hierarchy with the same k axis and irrep
        sectors are converted block by block. Other pairs, among them axes
        over the same k-points with different point operations, go through
        the dense representation and are verified; across different
        sectors this serves square operators only, since the dense form
        cannot assign the columns of orbital coefficients to sectors.

        Raises
        ------
        StructureError
            If the matrix cannot be represented exactly in ``target``.
        """
        if target == self.structure:
            return self
        self._require_same_k_axis(target)
        if embedding.is_structural_embedding(self.structure, target):
            return BlockMatrix(
                target, embedding.embed_blocks(self.structure, self.blocks, target)
            )
        self._require_operator_across_sectors(target)
        return BlockMatrix.from_dense(
            target, self.to_dense(), column_split=self.column_split, check=True
        )

    def project(self, target: BlockStructure) -> BlockMatrix:
        """Project the matrix onto a more symmetric structure by averaging.

        Across different irrep sectors this serves square operators only.
        """
        self._require_same_k_axis(target)
        self._require_operator_across_sectors(target)
        return BlockMatrix.from_dense(
            target, self.to_dense(), column_split=self.column_split, check=False
        )

    def _require_operator_across_sectors(self, target: BlockStructure) -> None:
        if target.irreps == self.structure.irreps:
            return
        if any(block.shape[-2] != block.shape[-1] for block in self.blocks):
            raise StructureError(
                "orbital coefficients do not change irrep sectors by embedding or "
                "projection; to_dense(orbitals=True) gives them in the atomic basis"
            )

    def _require_same_k_axis(self, target: BlockStructure) -> None:
        """Both structures hold the same k-point set; their operations may differ."""
        mine = self.structure.k_axis
        if (mine is None and target.k_axis is not None) or (
            mine is not None and not mine.same_points(target.k_axis)
        ):
            raise StructureError(
                "cannot change the k-point set of a matrix by embedding or "
                "projection; the structures carry different k axes"
            )

    def spin_trace(self) -> BlockMatrix:
        """Sum over the spin diagonal, as a spatial matrix.

        For a density matrix this is the total (spin-summed) density. The
        result is real unless the structure has complex-ring blocks: the
        spin trace of a paired structure is real by its symmetry.
        """
        structure = self.structure
        if structure.irreps is not None:
            target = BlockStructure.spatial(
                Ring.COMPLEX if structure.is_complex else Ring.REAL
            ).with_irreps(structure.irreps)
            return BlockMatrix.from_sectors(
                target, [sector.spin_trace() for sector in self.sectors()]
            )
        ring = Ring.COMPLEX if structure.is_complex else Ring.REAL
        target = BlockStructure.spatial(ring).with_k_axis(structure.k_axis)
        (block,) = target.stored_blocks
        parts = dense.spin_blocks_at(
            structure, self.blocks, target.stored_kpoints(block)
        )
        if len(parts) == 2:
            total = parts[0] + parts[1]
        else:
            n = half_dimension(int(parts[0].shape[-2]), "the number of spinor rows")
            m = half_dimension(int(parts[0].shape[-1]), "the number of spinor columns")
            total = parts[0][..., :n, :m] + parts[0][..., n:, m:]
        xp = array_namespace(total)
        return BlockMatrix(
            target, (cast_to_ring(xp, total, target.storage_ring(block)),)
        )

    # ------------------------------------------------------------------ eigenproblem

    def eigh(
        self, overlap: BlockMatrix | None = None
    ) -> tuple[BlockVector, BlockMatrix]:
        """Diagonalize each block of a Hermitian matrix.

        With ``overlap`` (same structure) the generalized problem
        ``F C = S C ε`` is solved and the eigenvectors are orthonormal in the
        overlap metric. Each stored k-point is diagonalized on its own.
        Quaternionic blocks must be Θ-symmetric at the time-reversal
        invariant k-points and return Kramers-paired eigenvectors in
        adjacent columns there.
        """
        if overlap is not None:
            self._require_same_structure(overlap, "diagonalize with the overlap of")
        values: list[Array] = []
        vectors: list[Array] = []
        for index, (block, array) in enumerate(
            zip(self.structure.stored_blocks, self.blocks, strict=True)
        ):
            adapt = _kramers_adaptation(self.structure, block, array)
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


def _batched(
    structure: BlockStructure, block: Block, shape: tuple[int, int]
) -> tuple[int, ...]:
    """Prepend the stored k count of ``block`` to ``shape`` when there is a k axis."""
    stored = structure.stored_kpoints(block)
    return shape if stored is None else (len(stored), *shape)


def _sector_shapes(
    structure: BlockStructure, shape: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    """Spatial shape of every stored block for a matrix of spatial ``shape``."""
    sectors = structure.irreps
    size = structure.stored_per_sector
    if sectors is None:
        return (shape,) * size
    if shape != (sectors.n_functions, sectors.n_functions):
        raise StructureError(
            "a matrix over irrep sectors is built as a square operator over the "
            f"{sectors.n_functions} functions of the basis, got shape {list(shape)}"
        )
    return tuple(
        (dimension, dimension) for dimension in sectors.dimensions for _ in range(size)
    )


def _kramers_adaptation(
    structure: BlockStructure, block: Block, array: Array
) -> Callable[[Array, Array], Array] | None:
    """Return the eigenvector post-processing a block needs, after validating it.

    Only the time-reversal invariant k-points of a quaternionic block are
    Θ-symmetric on their own; elsewhere the partner at ``−k`` carries the
    symmetry and nothing is adapted.
    """
    if block.ring is not Ring.QUATERNION:
        return None
    positions = structure.invariant_positions(block)
    batched = structure.k_axis is not None
    for position in positions:
        matrix = array[position] if batched else array
        if not is_kramers_symmetric(matrix):
            where = f" at k-point {position}" if batched else ""
            raise StructureError(
                "a paired (quaternionic) block must be Θ-symmetric to be "
                f"diagonalized{where}; build it with from_dense or kramers_symmetrize"
            )
    if not batched:
        return kramers_adapt_eigenvectors
    if not positions:
        return None

    def adapt(eigenvalues: Array, eigenvectors: Array) -> Array:
        xp = array_namespace(eigenvalues, eigenvectors)
        return xp.stack(
            [
                kramers_adapt_eigenvectors(eigenvalues[k], eigenvectors[k])
                if k in positions
                else eigenvectors[k]
                for k in range(int(eigenvectors.shape[0]))
            ],
            axis=0,
        )

    return adapt

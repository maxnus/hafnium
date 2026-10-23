"""Space groups of crystals and their action on the Bloch basis (ADR-0024).

The space group of a cell is found by spglib, an optional dependency
(``hafnium[symmetry]``). Its operations ``{R | t}`` map the crystal onto
itself; the point operations ``R`` map a k-point to ``Rk`` and the Bloch
functions at ``k`` to those at ``Rk``,

``O_g φ_μ^k = exp(−i (Rk)·T_A) Σ_m' D^l(R)[m', m] φ_{g(A) m'}^{Rk}``,

where the function ``μ`` of a shell on atom ``A`` is carried to the same
shell on the image atom ``g(A)``, rotated by the Wigner matrix of its
harmonics (:mod:`hafnium.symmetry.rotations`), and ``T_A`` is the lattice
vector by which ``R r_A + t`` differs from ``r_{g(A)}``. An operator that
commutes with the group therefore satisfies ``X(Rk) = U_g(k) X(k) U_g(k)†``
with the unitary ``U_g(k) = Π_g diag(exp(−i (Rk)·T_A))``, which is the
rule by which the structured matrices derive the blocks at rotated
k-points from the irreducible wedge (:class:`hafnium.structured.KPointOperations`).

A chain or a slab is padded with vectors along its open directions for
spglib, and only the operations that keep the open directions apart from
the periodic ones are kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from hafnium.backend import Array, Namespace, array_namespace, default_namespace
from hafnium.basis import BasisSet
from hafnium.errors import InputError, SymmetryError
from hafnium.molecule import Cell, atomic_number
from hafnium.structured import KPointOperations
from hafnium.symmetry.detection import DEFAULT_SYMMETRY_TOLERANCE
from hafnium.symmetry.kpoints import PARTNER_TOLERANCE, KPoints
from hafnium.symmetry.rotations import basis_rotation

Vector = tuple[float, float, float]
Frame = tuple[Vector, Vector, Vector]
IntegerMatrix = tuple[tuple[int, int, int], ...]

PADDING_FACTOR = 1.37
"""The open directions of a chain or a slab are padded with vectors this
many times the longest periodic vector, a length no periodic vector shares."""

_TRANSLATION_TOLERANCE = 1e-8
"""Fractional translation below which an operation counts as symmorphic."""


@dataclass(frozen=True, slots=True)
class SpaceGroupOperation:
    """One operation ``{R | t}`` of a space group.

    Parameters
    ----------
    rotation
        The integer matrix ``W`` of the operation in the lattice basis
        (padded to three vectors for a chain or a slab): fractional
        coordinates map as ``x' = x Wᵀ + w`` for row vectors.
    translation
        The fractional translation ``w``.
    matrix
        The orthogonal Cartesian matrix ``R``, rows, so that positions map
        as ``r' = R r + t``.
    shift
        The Cartesian translation ``t`` in Bohr.
    """

    rotation: IntegerMatrix
    translation: Vector
    matrix: Frame
    shift: Vector

    @property
    def is_identity(self) -> bool:
        """Whether the operation is the identity with no translation."""
        return self.rotation == ((1, 0, 0), (0, 1, 0), (0, 0, 1)) and all(
            abs(w) < _TRANSLATION_TOLERANCE for w in self.translation
        )

    @property
    def is_proper(self) -> bool:
        """Whether ``R`` is a rotation (determinant ``+1``) rather than improper."""
        return _determinant(self.matrix) > 0.0

    @property
    def is_inversion(self) -> bool:
        """Whether ``R`` is the inversion ``−1``."""
        return self.rotation == ((-1, 0, 0), (0, -1, 0), (0, 0, -1))

    @property
    def has_translation(self) -> bool:
        """Whether the operation carries a fractional translation."""
        return any(abs(w) >= _TRANSLATION_TOLERANCE for w in self.translation)


@dataclass(frozen=True, slots=True)
class SpaceGroup:
    """The space group of a cell: its operations and standard names.

    Build it with :func:`detect_space_group`; :meth:`restricted_to`
    returns the subgroup that maps a k-point set onto itself.

    Parameters
    ----------
    international
        Hermann–Mauguin symbol of the group; for a chain or a slab that of
        the padded three-dimensional cell.
    number
        Number of the group in the International Tables.
    point_group
        Hermann–Mauguin symbol of the crystal class.
    operations
        The operations, the identity first.
    dimension
        Number of periodic directions of the cell.
    tolerance
        Distance in Bohr within which the operations map every atom onto
        an atom.
    """

    international: str
    number: int
    point_group: str
    operations: tuple[SpaceGroupOperation, ...]
    dimension: int
    tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        if not self.operations or not self.operations[0].is_identity:
            raise SymmetryError("the identity is the first operation of a space group")
        if not 1 <= self.dimension <= 3:
            raise SymmetryError("a cell has one to three periodic directions")

    @property
    def name(self) -> str:
        """``"Pm-3m (221)"``: the symbol and the number, for messages."""
        return f"{self.international} ({self.number})"

    @property
    def order(self) -> int:
        """Number of operations."""
        return len(self.operations)

    @property
    def has_fractional_translations(self) -> bool:
        """Whether an operation carries a fractional translation.

        This depends on the origin of the cell: the operations of a
        symmorphic group carry none only about one of its fixed points.
        """
        return any(operation.has_translation for operation in self.operations)

    @property
    def has_inversion(self) -> bool:
        """Whether the inversion is among the point operations."""
        return any(operation.is_inversion for operation in self.operations)

    def kpoint_images(self, kpoints: KPoints) -> tuple[tuple[int, ...], ...]:
        """Index ``[operation][kpt]`` of the image ``Rk`` of every point.

        In the reciprocal basis a point maps as ``k' = k W⁻¹``.

        Raises
        ------
        InputError
            If an operation maps a point of the set outside it; see
            :meth:`restricted_to`.
        """
        images: list[tuple[int, ...]] = []
        for index, operation in enumerate(self.operations):
            row = self._images_under(kpoints, operation)
            if row is None:
                raise InputError(
                    f"the k-point set is not invariant under operation {index} of "
                    f"{self.name}; choose a mesh with the symmetry of the lattice, "
                    "or SpaceGroup.restricted_to(kpoints) for the subgroup that "
                    "maps the set onto itself"
                )
            images.append(row)
        return tuple(images)

    def restricted_to(self, kpoints: KPoints) -> SpaceGroup:
        """Return the subgroup of operations that map ``kpoints`` onto itself."""
        kept = tuple(
            operation
            for operation in self.operations
            if self._images_under(kpoints, operation) is not None
        )
        return SpaceGroup(
            self.international,
            self.number,
            self.point_group,
            kept,
            self.dimension,
            self.tolerance,
        )

    def _images_under(
        self, kpoints: KPoints, operation: SpaceGroupOperation
    ) -> tuple[int, ...] | None:
        if kpoints.dimension != self.dimension:
            raise InputError(
                f"{kpoints.dimension}-dimensional k-points do not fit a "
                f"{self.dimension}-dimensional cell"
            )
        xp = default_namespace()
        d = self.dimension
        inverse = xp.asarray(
            _integer_inverse(operation.rotation)[:d], dtype=xp.float64
        )[:, :d]
        points = kpoints.fractional_array(xp)
        return _match_points(xp, points, points @ inverse)


def detect_space_group(
    cell: Cell, *, tolerance: float = DEFAULT_SYMMETRY_TOLERANCE
) -> SpaceGroup:
    """Detect the space group of ``cell`` with spglib.

    Atoms are equivalent when they are the same element; the basis set is
    checked when the adaptation is built. Every operation must map each
    atom onto an atom within ``tolerance`` Bohr. The operations of a chain
    or a slab are those of the cell padded along its open directions that
    keep the open directions apart from the periodic ones.

    Raises
    ------
    InputError
        If spglib is not installed, or the tolerance is not positive.
    """
    if not tolerance > 0.0:
        raise InputError("the symmetry tolerance is positive")
    try:
        import spglib
    except ImportError:
        raise InputError(
            "space-group detection needs spglib; install the 'symmetry' extra "
            "(hafnium[symmetry])"
        ) from None
    # spglib 2.x reports failures through exceptions once its old error
    # handling is switched off; the switch disappears in a later version.
    error_module: Any = getattr(spglib, "error", None)
    if hasattr(error_module, "OLD_ERROR_HANDLING"):
        error_module.OLD_ERROR_HANDLING = False
    xp = default_namespace()
    padded = _padded_vectors(cell, xp)
    fractional = cell.positions_array(xp) @ xp.linalg.inv(padded)
    numbers = [atomic_number(symbol) for symbol in cell.symbols]
    try:
        dataset: Any = spglib.get_symmetry_dataset(
            (_rows(padded), _rows(fractional), numbers), symprec=tolerance
        )
    except spglib.SpglibError as error:
        raise SymmetryError(
            f"spglib could not determine the space group of the cell: {error}"
        ) from error
    if dataset is None:
        raise SymmetryError("spglib could not determine the space group of the cell")
    operations: list[SpaceGroupOperation] = []
    for rotation, translation in zip(
        dataset.rotations, dataset.translations, strict=True
    ):
        operation = _operation(xp, padded, rotation, translation, cell.dimension)
        if operation is not None:
            operations.append(operation)
    operations.sort(key=lambda operation: not operation.is_identity)
    return SpaceGroup(
        str(dataset.international),
        int(dataset.number),
        str(dataset.pointgroup),
        tuple(operations),
        cell.dimension,
        tolerance,
    )


def _padded_vectors(cell: Cell, xp: Namespace) -> Array:
    """Three lattice vectors ``[3, 3]``, the open directions padded."""
    lattice = cell.lattice
    vectors = lattice.vectors_array(xp)
    if lattice.dimension == 3:
        return vectors
    longest = float(xp.max(xp.linalg.vector_norm(vectors, axis=-1)))
    padding = PADDING_FACTOR * longest * lattice.perpendicular_vectors_array(xp)
    return xp.concat([vectors, padding], axis=0)


def _rows(array: Array) -> list[list[float]]:
    return [[float(x) for x in row] for row in array]


def _operation(
    xp: Namespace, padded: Array, rotation: Any, translation: Any, dimension: int
) -> SpaceGroupOperation | None:
    """Build one operation from spglib's integer matrix and translation.

    Returns ``None`` for an operation of the padded cell that mixes the
    open directions of a chain or a slab with the periodic ones, or
    translates along them.
    """
    w = [[int(x) for x in row] for row in rotation]
    t = [float(x) for x in translation]
    if dimension < 3:
        for i in range(3):
            for j in range(3):
                if (i < dimension) != (j < dimension) and w[i][j] != 0:
                    return None
        if any(
            abs(t[i] - round(t[i])) > _TRANSLATION_TOLERANCE
            for i in range(dimension, 3)
        ):
            return None
        t = [t[i] if i < dimension else 0.0 for i in range(3)]
    t = [x - math.floor(x + _TRANSLATION_TOLERANCE) for x in t]
    integer = xp.asarray(w, dtype=xp.float64)
    # r' = R r + t with r = x A: R = Aᵀ W (Aᵀ)⁻¹, orthogonalized by its
    # polar decomposition so that a lattice symmetric within the tolerance
    # yields an exactly orthogonal matrix.
    transposed = xp.linalg.matrix_transpose(padded)
    raw = transposed @ integer @ xp.linalg.inv(transposed)
    left, _, right = xp.linalg.svd(raw)
    matrix = left @ right
    shift = xp.asarray(t, dtype=xp.float64) @ padded
    return SpaceGroupOperation(
        tuple(tuple(row) for row in w),  # type: ignore[arg-type]
        (t[0], t[1], t[2]),
        tuple(tuple(float(x) for x in row) for row in matrix),  # type: ignore[arg-type]
        (float(shift[0]), float(shift[1]), float(shift[2])),
    )


def _determinant(frame: Frame) -> float:
    a, b, c = frame
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _integer_inverse(rotation: IntegerMatrix) -> IntegerMatrix:
    """Inverse of a unimodular integer matrix, exact."""
    xp = default_namespace()
    inverse = xp.linalg.inv(xp.asarray(rotation, dtype=xp.float64))
    rounded = [[round(float(x)) for x in row] for row in inverse]
    if float(xp.max(xp.abs(inverse - xp.asarray(rounded, dtype=xp.float64)))) > 1e-8:
        raise SymmetryError(
            "a space-group rotation is an integer matrix of determinant ±1"
        )
    return tuple(tuple(row) for row in rounded)  # type: ignore[return-value]


def _match_points(
    xp: Namespace, points: Array, targets: Array
) -> tuple[int, ...] | None:
    """Index in ``points`` of each target modulo one, or ``None`` if one is missing."""
    difference = xp.expand_dims(points, 0) - xp.expand_dims(targets, 1)
    distance = xp.abs((difference + 0.5) % 1.0 - 0.5)
    matches = xp.all(distance <= PARTNER_TOLERANCE, axis=-1)
    counts = xp.sum(xp.astype(matches, xp.int64), axis=1)
    if bool(xp.any(counts != 1)):
        return None
    return tuple(int(index) for index in xp.argmax(matches, axis=1))


@dataclass(frozen=True, slots=True, eq=False)
class SpaceGroupAdaptation:
    """The action of a space group on the Bloch basis over a k-point set.

    Build it with :meth:`from_basis`. The arrays are host arrays of the
    default namespace.

    Parameters
    ----------
    group
        The space group.
    kpoints
        The k-point set, closed under the point operations.
    kpoints_cartesian
        The k-points ``[kpt, 3]`` in inverse Bohr.
    atom_images
        ``[operation][atom]``: the atom each operation maps an atom onto.
    shifts
        ``[operation, atom, 3]``: the lattice vector ``T_A`` in Bohr by
        which the image of an atom differs from the atom it lands on.
    matrices
        One ``[ao, ao]`` orthogonal matrix ``Π_g`` per operation, carrying
        every function to the rotated functions of its image atom.
    function_atoms
        The atom of every basis function.
    """

    group: SpaceGroup
    kpoints: KPoints
    kpoints_cartesian: Array
    atom_images: tuple[tuple[int, ...], ...]
    shifts: Array
    matrices: tuple[Array, ...]
    function_atoms: tuple[int, ...]
    _operations: KPointOperations = field(init=False, repr=False)

    def __post_init__(self) -> None:
        xp = array_namespace(self.shifts, *self.matrices)
        images = self.group.kpoint_images(self.kpoints)
        atoms = xp.asarray(self.function_atoms, dtype=xp.int64)
        phases: list[Array] = []
        for operation, row in enumerate(images):
            # The phase of a function at the source point is exp(−i k'·T_A)
            # with k' the image point and A the function's atom.
            moved = xp.take(
                self.kpoints_cartesian, xp.asarray(row, dtype=xp.int64), axis=0
            )
            angle = -(moved @ xp.linalg.matrix_transpose(self.shifts[operation]))
            phases.append(xp.exp(1j * xp.take(angle, atoms, axis=1)))
        object.__setattr__(
            self,
            "_operations",
            KPointOperations(
                self.group.name, images, self.matrices, xp.stack(phases, axis=0)
            ),
        )

    @classmethod
    def from_basis(
        cls,
        basis: BasisSet,
        cell: Cell,
        group: SpaceGroup,
        kpoints: KPoints,
    ) -> SpaceGroupAdaptation:
        """Build the action of ``group`` on ``basis`` over ``cell`` at ``kpoints``.

        Raises
        ------
        SymmetryError
            If an operation does not map every atom onto an atom of the same
            element within the group's tolerance, or atoms the group relates
            carry different shells.
        InputError
            If the k-point set is not closed under the point operations.
        """
        if basis.n_atoms != cell.n_atoms:
            raise InputError("the basis set and the cell have different atom counts")
        if kpoints.dimension != cell.dimension:
            raise InputError(
                f"{kpoints.dimension}-dimensional k-points do not fit a "
                f"{cell.dimension}-dimensional cell"
            )
        group.kpoint_images(kpoints)
        xp = default_namespace()
        atom_images: list[tuple[int, ...]] = []
        shifts: list[Array] = []
        matrices: list[Array] = []
        for index, operation in enumerate(group.operations):
            images, shift = _atom_images(cell, operation, index, group.tolerance)
            atom_images.append(images)
            shifts.append(shift)
            matrices.append(
                basis_rotation(
                    basis, xp.asarray(operation.matrix, dtype=xp.float64), images
                )
            )
        return cls(
            group,
            kpoints,
            kpoints.cartesian(cell.lattice),
            tuple(atom_images),
            xp.stack(shifts, axis=0),
            tuple(matrices),
            tuple(label.atom for label in basis.function_labels()),
        )

    @property
    def n_functions(self) -> int:
        """Number of basis functions."""
        return int(self.matrices[0].shape[0])

    def operations(self) -> KPointOperations:
        """Return the point operations for the structured matrices (ADR-0024).

        The same object is returned every time, so structures built from
        one adaptation share it.
        """
        return self._operations


def _atom_images(
    cell: Cell, operation: SpaceGroupOperation, index: int, tolerance: float
) -> tuple[tuple[int, ...], Array]:
    """Return the image atom of every atom and the lattice vector it picks up."""
    xp = default_namespace()
    positions = cell.positions_array(xp)
    matrix = xp.asarray(operation.matrix, dtype=xp.float64)
    moved = positions @ xp.linalg.matrix_transpose(matrix) + xp.asarray(
        operation.shift, dtype=xp.float64
    )
    lattice = cell.lattice
    images: list[int] = []
    shifts: list[Array] = []
    for atom in range(cell.n_atoms):
        difference = xp.expand_dims(moved[atom], 0) - positions
        whole = xp.floor(lattice.fractional_coordinates(difference) + 0.5)
        vectors = lattice.cartesian_coordinates(whole)
        residual = xp.linalg.vector_norm(difference - vectors, axis=-1)
        image = int(xp.argmin(residual))
        if (
            float(residual[image]) > tolerance
            or cell.symbols[image] != cell.symbols[atom]
        ):
            raise SymmetryError(
                f"operation {index} of the space group does not map atom {atom} onto "
                "an equivalent atom within the tolerance"
            )
        images.append(image)
        shifts.append(vectors[image])
    return tuple(images), xp.stack(shifts, axis=0)

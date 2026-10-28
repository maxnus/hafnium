"""The k axis of a block structure: partners, weights, point operations, storage.

A k-point set enters the structured types as sector metadata shared by every
block (ADR-0007, ADR-0015): for each point, the index of its time-reversal
partner ``−k`` and its weight, and optionally the point operations of the
crystal's space group acting on the set (ADR-0024). Which points a block
stores follows from its ring. Blocks over the real and quaternionic rings
store one representative of every orbit of the point operations combined
with time reversal, the point with the lowest index, in ascending order;
blocks over the complex ring store one representative of every orbit of
the point operations alone. Without operations these are the half zone
and the full set. The block at a derived point follows from its stored
source by the unitary rotation of the Bloch basis for the operation,
``X(Rk) = U X(k) U†``, and then by the antiunitary map of the ring for
time reversal (complex conjugation, or the Kramers map ``Θ X Θ⁻¹``).

:class:`KAxis` knows nothing about reciprocal lattices. It is built from a
:class:`hafnium.symmetry.KPoints` set, which keeps the geometry, and the
:class:`KPointOperations` from a :class:`hafnium.symmetry.SpaceGroupAdaptation`,
so that the ``symmetry`` package depends on ``structured`` and not the other
way round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from hafnium.backend import Array, array_namespace
from hafnium.errors import StructureError

WEIGHT_TOLERANCE = 1e-12
"""Allowed deviation of the weight sum from one, and between partner weights."""

UNITARITY_TOLERANCE = 1e-10
"""Largest deviation of an operation's matrix from orthogonality, and of a
Bloch phase from unit modulus."""


@dataclass(frozen=True, slots=True, eq=False)
class KPointOperations:
    """The point operations of a crystal on a k-point set and its Bloch basis.

    Operation ``g`` maps the point ``k`` to ``images[g][k]`` and the Bloch
    functions at ``k`` to those at the image by the unitary matrix
    ``U_g(k) = Π_g diag(phases[g, k])`` (ADR-0024): ``Π_g`` permutes the
    atoms and rotates the harmonics of every shell, and the phases carry
    the lattice translations the atoms pick up. Operation ``0`` is the
    identity, and the images are closed under composition, which is
    checked. The matrices and phases must represent the group as well,
    ``U_g(hk) U_h(k) = U_{gh}(k)``, which the projection onto the stored
    points relies on and which the construction from a space group
    guarantees. Two operation sets are equal when their group, images and
    arrays are.

    Parameters
    ----------
    group
        Name of the space group, for messages.
    images
        ``[operation][kpt]``: the index of the image of every point.
    matrices
        One real orthogonal ``[ao, ao]`` matrix per operation, the
        identity first.
    phases
        ``[operation, kpt, ao]`` complex factors of unit modulus, one per
        function at the source point; ones for the identity.
    """

    group: str
    images: tuple[tuple[int, ...], ...]
    matrices: tuple[Array, ...]
    phases: Array

    def __post_init__(self) -> None:
        images = tuple(tuple(int(k) for k in row) for row in self.images)
        matrices = tuple(self.matrices)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "matrices", matrices)
        if not images or len(matrices) != len(images):
            raise StructureError(
                "k-point operations need one image row and one matrix per operation"
            )
        _check_images(images)
        _check_matrices(matrices)
        xp = array_namespace(self.phases)
        shape = (len(images), len(images[0]), int(matrices[0].shape[0]))
        if self.phases.shape != shape:
            raise StructureError("the phases are [operation, kpt, ao]")
        if float(xp.max(xp.abs(xp.abs(self.phases) - 1.0))) > UNITARITY_TOLERANCE:
            raise StructureError("the Bloch phases have unit modulus")
        if float(xp.max(xp.abs(self.phases[0] - 1.0))) > UNITARITY_TOLERANCE:
            raise StructureError("the first operation has unit phases")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KPointOperations):
            return NotImplemented
        if self is other:
            return True
        if self.group != other.group or self.images != other.images:
            return False
        if self.phases.shape != other.phases.shape:
            return False
        xp = array_namespace(self.phases)
        return all(
            a.shape == b.shape and bool(xp.all(a == b))
            for a, b in zip(self.matrices, other.matrices, strict=True)
        ) and bool(xp.all(self.phases == other.phases))

    def __hash__(self) -> int:
        return hash((self.group, self.images))

    @property
    def n_operations(self) -> int:
        """Number of operations, the identity included."""
        return len(self.images)

    @property
    def n_kpt(self) -> int:
        """Number of k-points in the full set."""
        return len(self.images[0])

    @property
    def n_functions(self) -> int:
        """Number of basis functions the matrices act on."""
        return int(self.matrices[0].shape[0])

    def unitary(self, operation: int, k: int) -> Array:
        """``U_g(k) [ao, ao]``, mapping the Bloch basis at ``k`` to its image."""
        xp = array_namespace(self.phases)
        matrix = xp.astype(self.matrices[operation], xp.complex128)
        return matrix * xp.expand_dims(self.phases[operation, k], axis=0)

    def rotate(self, operation: int, k: int, array: Array, *, operator: bool) -> Array:
        """Map a block at ``k`` to its image: ``U X U†`` or, for columns, ``U X``.

        A block with twice the number of functions is a spinor block, on
        which the operation acts within each spin.
        """
        unitary = self._for_rows(operation, k, array)
        xp = array_namespace(array)
        moved = unitary @ array
        if operator:
            return moved @ xp.conj(xp.linalg.matrix_transpose(unitary))
        return moved

    def unrotate(
        self, operation: int, k: int, array: Array, *, operator: bool
    ) -> Array:
        """Map a block at the image back to ``k``: ``U† X U`` or ``U† X``."""
        unitary = self._for_rows(operation, k, array)
        xp = array_namespace(array)
        moved = xp.conj(xp.linalg.matrix_transpose(unitary)) @ array
        if operator:
            return moved @ unitary
        return moved

    def _for_rows(self, operation: int, k: int, array: Array) -> Array:
        xp = array_namespace(array)
        unitary = self.unitary(operation, k)
        rows, n = int(array.shape[-2]), self.n_functions
        if rows == n:
            return unitary
        if rows == 2 * n:
            zero = xp.zeros((n, n), dtype=unitary.dtype)
            return xp.concat(
                [
                    xp.concat([unitary, zero], axis=1),
                    xp.concat([zero, unitary], axis=1),
                ],
                axis=0,
            )
        raise StructureError(
            f"a block with {rows} rows does not act on the {n} functions the "
            "k-point operations are built for"
        )


def _check_images(images: tuple[tuple[int, ...], ...]) -> None:
    n_kpt = len(images[0])
    if n_kpt == 0 or images[0] != tuple(range(n_kpt)):
        raise StructureError("the first operation is the identity on the k-points")
    for row in images:
        if len(row) != n_kpt or sorted(row) != list(range(n_kpt)):
            raise StructureError("every operation permutes the k-point set")
    rows = set(images)
    for outer in images:
        for inner in images:
            if tuple(outer[k] for k in inner) not in rows:
                raise StructureError(
                    "the operations on the k-points are closed under composition"
                )


def _check_matrices(matrices: tuple[Array, ...]) -> None:
    xp = array_namespace(*matrices)
    first = matrices[0]
    n = int(first.shape[0]) if first.ndim == 2 else -1
    if n < 1 or any(matrix.shape != (n, n) for matrix in matrices):
        raise StructureError("the matrices of the operations are [ao, ao]")
    eye = xp.eye(n)
    for matrix in matrices:
        deviation = xp.linalg.matrix_transpose(matrix) @ matrix - eye
        if float(xp.max(xp.abs(deviation))) > UNITARITY_TOLERANCE:
            raise StructureError("the matrix of an operation is orthogonal")
    if float(xp.max(xp.abs(first - eye))) > UNITARITY_TOLERANCE:
        raise StructureError("the first operation has the identity matrix")


@dataclass(frozen=True, slots=True)
class KDerivation:
    """How a k-point of the full set is derived from a stored point.

    The block at the point is ``operation`` applied to the block at
    ``source``, conjugated (time-reversed) afterwards when ``conjugate``;
    a stored point is its own source with the identity and no conjugation.
    """

    source: int
    operation: int
    conjugate: bool

    @property
    def is_stored(self) -> bool:
        """Whether the point is stored itself."""
        return self.operation == 0 and not self.conjugate


@dataclass(frozen=True, slots=True)
class KAxis:
    """The k-points shared by every block of a structure.

    Parameters
    ----------
    partners
        Index of the time-reversal partner of every point; a point equal
        to its own partner is a time-reversal invariant momentum.
    weights
        Weight of every point, positive and summing to one over the set,
        equal for the members of an orbit.
    operations
        The point operations of the space group on the set (ADR-0024);
        ``None`` reduces the set by time reversal alone.
    momenta
        The exact fractional coordinates of every point as integers over
        the common denominators ``modulus``, ``k_i = momenta[k][i] /
        modulus[i]``, in ``[0, modulus[i])``, for a uniform mesh: a block
        tensor conserves the crystal momentum of its indices modulo the
        modulus (ADR-0031). ``None`` when the set is not a uniform mesh;
        tensors then cannot be formed over the axis.
    modulus
        The denominators of ``momenta``, twice the mesh sizes so that a
        shift of one half is an integer; given with ``momenta``.
    """

    partners: tuple[int, ...]
    weights: tuple[float, ...]
    operations: KPointOperations | None = field(default=None, kw_only=True)
    momenta: tuple[tuple[int, ...], ...] | None = field(default=None, kw_only=True)
    modulus: tuple[int, ...] | None = field(default=None, kw_only=True)
    _orbit: tuple[KDerivation, ...] = field(init=False, repr=False, compare=False)
    _full: tuple[KDerivation, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        partners = tuple(int(partner) for partner in self.partners)
        weights = tuple(float(weight) for weight in self.weights)
        n = len(partners)
        if n == 0:
            raise StructureError("a k axis has at least one point")
        if len(weights) != n:
            raise StructureError(f"{n} k-points need {n} weights, got {len(weights)}")
        if any(not 0 <= partner < n for partner in partners) or any(
            partners[partner] != k for k, partner in enumerate(partners)
        ):
            raise StructureError(
                "the partner map pairs every k-point with one point of the set "
                "and is its own inverse"
            )
        if any(not weight > 0.0 for weight in weights) or not math.isclose(
            sum(weights), 1.0, abs_tol=WEIGHT_TOLERANCE
        ):
            raise StructureError("k-point weights are positive and sum to one")
        if any(
            not math.isclose(weights[k], weights[partner], abs_tol=WEIGHT_TOLERANCE)
            for k, partner in enumerate(partners)
        ):
            raise StructureError("time-reversal partners carry equal weights")
        images = (
            (tuple(range(n)),) if self.operations is None else self.operations.images
        )
        if len(images[0]) != n:
            raise StructureError(
                f"the k-point operations act on {len(images[0])} points, the axis "
                f"holds {n}"
            )
        object.__setattr__(self, "partners", partners)
        object.__setattr__(self, "weights", weights)
        momenta, modulus = _check_momenta(self.momenta, self.modulus, partners)
        object.__setattr__(self, "momenta", momenta)
        object.__setattr__(self, "modulus", modulus)
        object.__setattr__(self, "_orbit", _derivations(images, None))
        object.__setattr__(self, "_full", _derivations(images, partners))
        for table in (self._orbit, self._full):
            if any(
                not math.isclose(
                    weights[k], weights[derivation.source], abs_tol=WEIGHT_TOLERANCE
                )
                for k, derivation in enumerate(table)
            ):
                raise StructureError(
                    "k-points related by a point operation carry equal weights"
                )

    @classmethod
    def gamma(cls, dimension: int = 3) -> KAxis:
        """Build the axis of the Γ point alone, with its momentum in ``dimension``."""
        return cls((0,), (1.0,), momenta=((0,) * dimension,), modulus=(2,) * dimension)

    @property
    def n_kpt(self) -> int:
        """Number of k-points in the full set."""
        return len(self.partners)

    @property
    def invariant(self) -> tuple[bool, ...]:
        """Whether each point is its own time-reversal partner."""
        return tuple(partner == k for k, partner in enumerate(self.partners))

    @property
    def all_invariant(self) -> bool:
        """Whether every point is time-reversal invariant (for example Γ alone)."""
        return all(self.invariant)

    @property
    def representatives(self) -> tuple[int, ...]:
        """The points a half-zone block stores: see :meth:`stored`."""
        return self.stored(half_zone=True)

    @property
    def images(self) -> tuple[tuple[int, ...], ...]:
        """Image ``[operation][kpt]`` of each point, the identity alone without any."""
        if self.operations is None:
            return (tuple(range(self.n_kpt)),)
        return self.operations.images

    def derivations(self, half_zone: bool) -> tuple[KDerivation, ...]:
        """How every point of the full set is derived, for either storage rule.

        With ``half_zone`` the orbits of the point operations combined with
        time reversal are reduced, otherwise those of the point operations
        alone; a point reached by an operation alone is never conjugated.
        """
        return self._full if half_zone else self._orbit

    def derivation(self, k: int, half_zone: bool) -> KDerivation:
        """How the point ``k`` is derived under either storage rule."""
        return self.derivations(half_zone)[k]

    def stored(self, half_zone: bool) -> tuple[int, ...]:
        """Return the indices a block stores, in ascending order.

        The orbit representatives, the lowest index of every orbit: of the
        point operations with time reversal for a half-zone block, of the
        point operations alone otherwise. Without operations these are the
        half zone and the full set.
        """
        return tuple(
            k
            for k, derivation in enumerate(self.derivations(half_zone))
            if derivation.is_stored
        )

    def stored_weights(
        self, half_zone: bool
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Weights ``(own, partner)`` of the stored points.

        ``own`` is the total weight of the stored point and the points
        derived from it without conjugation, whose traces equal its own;
        ``partner`` the total weight of the points derived with
        conjugation, whose traces are the conjugate, zero when there are
        none (every point stored, or a time-reversal invariant point).
        """
        table = self.derivations(half_zone)
        own: list[float] = []
        partner: list[float] = []
        for s in self.stored(half_zone):
            own.append(
                sum(
                    self.weights[k]
                    for k, d in enumerate(table)
                    if d.source == s and not d.conjugate
                )
            )
            partner.append(
                sum(
                    self.weights[k]
                    for k, d in enumerate(table)
                    if d.source == s and d.conjugate
                )
            )
        return tuple(own), tuple(partner)

    def same_points(self, other: KAxis | None) -> bool:
        """Whether ``other`` holds the same k-point set, operations aside."""
        return (
            other is not None
            and self.partners == other.partners
            and self.weights == other.weights
        )


def _check_momenta(
    momenta: tuple[tuple[int, ...], ...] | None,
    modulus: tuple[int, ...] | None,
    partners: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...] | None, tuple[int, ...] | None]:
    """Normalize the momenta, which label every point and negate across partners."""
    if (momenta is None) != (modulus is None):
        raise StructureError("the momenta of a k axis come with their modulus")
    if momenta is None or modulus is None:
        return None, None
    rows = tuple(tuple(int(m) for m in row) for row in momenta)
    sizes = tuple(int(n) for n in modulus)
    if len(rows) != len(partners):
        raise StructureError("a k axis has one momentum per point")
    if not sizes or any(n < 1 for n in sizes):
        raise StructureError("the modulus of the momenta is positive")
    for row in rows:
        if len(row) != len(sizes) or any(
            not 0 <= m < n for m, n in zip(row, sizes, strict=True)
        ):
            raise StructureError(
                "every momentum has one coordinate per direction below the modulus"
            )
    for k, partner in enumerate(partners):
        if any(
            (a + b) % n for a, b, n in zip(rows[k], rows[partner], sizes, strict=True)
        ):
            raise StructureError(
                "the momentum of a time-reversal partner is the negative modulo "
                "the modulus"
            )
    return rows, sizes


def _derivations(
    images: tuple[tuple[int, ...], ...], partners: tuple[int, ...] | None
) -> tuple[KDerivation, ...]:
    """Assign every point to the lowest-index representative of its orbit.

    A point is reached from its representative by one operation, and with
    ``partners`` also by one operation followed by time reversal; the
    operations are tried first, so a point is conjugated only when no
    operation alone reaches it.
    """
    n = len(images[0])
    table: list[KDerivation | None] = [None] * n
    for source in range(n):
        if table[source] is not None:
            continue
        for operation, row in enumerate(images):
            if table[row[source]] is None:
                table[row[source]] = KDerivation(source, operation, False)
        if partners is None:
            continue
        for operation, row in enumerate(images):
            k = partners[row[source]]
            if table[k] is None:
                table[k] = KDerivation(source, operation, True)
    if any(derivation is None for derivation in table):
        raise StructureError("the identity operation reaches every k-point")
    return tuple(derivation for derivation in table if derivation is not None)

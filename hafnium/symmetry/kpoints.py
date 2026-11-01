"""k-point sets: uniform meshes, explicit lists, time-reversal pairing, weights.

k-points are fractional coordinates in the reciprocal basis wrapped into
``[0, 1)``; the time-reversal partner of ``k`` is ``(1 − k) mod 1``, and a
point equal to its partner is a time-reversal invariant momentum
(CONVENTIONS.md §3, ADR-0007). Weights sum to one, so traces and energies
over the set are per unit cell.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass

from hafnium.backend import Array, Namespace, array_namespace, host_namespace
from hafnium.errors import InputError
from hafnium.molecule import Lattice
from hafnium.structured import KAxis, KPointOperations

WEIGHT_TOLERANCE = 1e-12
"""Allowed deviation of the weight sum from one."""

PARTNER_TOLERANCE = 1e-10
"""Distance in fractional coordinates within which two k-points coincide."""

_PARTNER_CHUNK = 256
"""Points matched at a time against the whole set, bounding the work array."""

_ALLOWED_SHIFTS = (0.0, 0.5)


def _wrapped(value: float) -> float:
    if not math.isfinite(value):
        raise InputError("k-point coordinates are finite")
    wrapped = value % 1.0
    return 0.0 if wrapped >= 1.0 else wrapped


@dataclass(frozen=True, slots=True)
class KPoints:
    """A set of k-points with weights.

    Parameters
    ----------
    fractional
        Coordinates ``[kpt, d]`` in the reciprocal basis; they are wrapped
        into ``[0, 1)`` on construction.
    weights
        Weights ``[kpt]``, positive and summing to one.
    """

    fractional: tuple[tuple[float, ...], ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        points = tuple(
            tuple(_wrapped(float(x)) for x in point) for point in self.fractional
        )
        weights = tuple(float(weight) for weight in self.weights)
        if not points:
            raise InputError("a k-point set has at least one point")
        dimensions = {len(point) for point in points}
        if len(dimensions) != 1 or not dimensions <= {1, 2, 3}:
            raise InputError(
                "every k-point has the same number of fractional coordinates, "
                "one to three"
            )
        if len(weights) != len(points):
            raise InputError(
                f"{len(points)} k-points need {len(points)} weights, got {len(weights)}"
            )
        if any(not (weight > 0.0) for weight in weights) or not math.isclose(
            sum(weights), 1.0, abs_tol=WEIGHT_TOLERANCE
        ):
            raise InputError("k-point weights are positive and sum to one")
        object.__setattr__(self, "fractional", points)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def mesh(
        cls, size: tuple[int, ...], *, shift: tuple[float, ...] | None = None
    ) -> KPoints:
        """Uniform mesh ``k_i = (m_i + s_i) / n_i`` with equal weights.

        ``size`` gives ``n_i`` per periodic direction and ``shift`` the
        ``s_i``, each 0 (the default, Γ-centred) or 1/2. Points are ordered
        with the first direction varying slowest.
        """
        if not 1 <= len(size) <= 3 or any(n < 1 for n in size):
            raise InputError(f"a mesh has one to three positive sizes, got {size}")
        shifts = (0.0,) * len(size) if shift is None else tuple(shift)
        if len(shifts) != len(size) or any(s not in _ALLOWED_SHIFTS for s in shifts):
            raise InputError(
                f"a mesh shift has one entry of 0 or 1/2 per direction, got {shift}"
            )
        points = tuple(
            tuple((m + s) / n for m, s, n in zip(indices, shifts, size, strict=True))
            for indices in itertools.product(*(range(n) for n in size))
        )
        return cls(points, (1.0 / len(points),) * len(points))

    @classmethod
    def monkhorst_pack(cls, size: tuple[int, ...]) -> KPoints:
        """Build the original Monkhorst–Pack mesh, shifted by 1/2 along even sizes."""
        return cls.mesh(size, shift=tuple(0.5 if n % 2 == 0 else 0.0 for n in size))

    @classmethod
    def gamma(cls, dimension: int = 3) -> KPoints:
        """Build the set holding the Γ point alone."""
        return cls.mesh((1,) * dimension)

    @classmethod
    def from_fractional(
        cls, points: Iterable[Iterable[float]], weights: Iterable[float] | None = None
    ) -> KPoints:
        """Build from an explicit list of points, with equal weights unless given."""
        fractional = tuple(tuple(float(x) for x in point) for point in points)
        if weights is None:
            if not fractional:
                raise InputError("a k-point set has at least one point")
            weights = (1.0 / len(fractional),) * len(fractional)
        return cls(fractional, tuple(weights))

    @property
    def n_kpt(self) -> int:
        """Number of k-points."""
        return len(self.fractional)

    @property
    def dimension(self) -> int:
        """Number of periodic directions."""
        return len(self.fractional[0])

    def fractional_array(self, xp: Namespace | None = None) -> Array:
        """Fractional coordinates as an array ``[kpt, d]``."""
        xp = host_namespace() if xp is None else xp
        return xp.asarray(self.fractional, dtype=xp.float64)

    def weights_array(self, xp: Namespace | None = None) -> Array:
        """Weights as an array ``[kpt]``."""
        xp = host_namespace() if xp is None else xp
        return xp.asarray(self.weights, dtype=xp.float64)

    def cartesian(self, lattice: Lattice) -> Array:
        """Cartesian k-points ``[kpt, 3]`` in inverse Bohr on ``lattice``."""
        if lattice.dimension != self.dimension:
            raise InputError(
                f"{self.dimension}-dimensional k-points do not fit a "
                f"{lattice.dimension}-dimensional lattice"
            )
        xp = host_namespace()
        return self.fractional_array(xp) @ lattice.reciprocal_vectors_array(xp)

    def mesh_size(self) -> tuple[int, ...]:
        """Return the sizes ``n_i`` of the uniform mesh the set forms.

        A uniform mesh holds the points ``(m_i + s_i) / n_i`` of :meth:`mesh`
        with equal weights; ``n_i`` is the number of distinct coordinates
        along direction ``i``.

        Raises
        ------
        InputError
            If the set is not a uniform mesh with equal weights.
        """
        return self._mesh()[0]

    def momenta(self) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
        """Return the exact coordinates of a uniform mesh over a common denominator.

        ``(momenta, modulus)`` with ``k_i = momenta[k][i] / modulus[i]``:
        the denominators are twice the mesh sizes, so that the half shift
        of a Monkhorst–Pack mesh is an integer, and the differences of two
        points, the momentum transfers, are even. The block tensors of
        ADR-0031 conserve these modulo the modulus.

        Raises
        ------
        InputError
            If the set is not a uniform mesh with equal weights.
        """
        sizes, shifts, indices = self._mesh()
        modulus = tuple(2 * n for n in sizes)
        momenta = tuple(
            tuple(
                (2 * m + s) % n for m, s, n in zip(index, shifts, modulus, strict=True)
            )
            for index in indices
        )
        return momenta, modulus

    def _mesh(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """Return the sizes, the doubled shifts and the integer index of every point."""
        sizes: list[int] = []
        shifts: list[float] = []
        for axis in range(self.dimension):
            values = sorted(point[axis] for point in self.fractional)
            distinct = 1 + sum(
                1
                for previous, value in itertools.pairwise(values)
                if value - previous > PARTNER_TOLERANCE
            )
            sizes.append(distinct)
            shifts.append(values[0] * distinct)
        if any(min(abs(s), abs(s - 0.5)) > PARTNER_TOLERANCE for s in shifts):
            raise InputError("the k-point set is not a uniform mesh")
        indices: list[tuple[int, ...]] = []
        for point in self.fractional:
            index = tuple(
                round(k * n - s) for k, n, s in zip(point, sizes, shifts, strict=True)
            )
            if any(
                not 0 <= m < n or abs(k - (m + s) / n) > PARTNER_TOLERANCE
                for k, m, n, s in zip(point, index, sizes, shifts, strict=True)
            ):
                raise InputError("the k-point set is not a uniform mesh")
            indices.append(index)
        if len(set(indices)) != self.n_kpt or math.prod(sizes) != self.n_kpt:
            raise InputError("the k-point set is not a uniform mesh")
        if any(abs(w - 1.0 / self.n_kpt) > WEIGHT_TOLERANCE for w in self.weights):
            raise InputError("a uniform mesh has equal weights")
        doubled = tuple(1 if abs(s - 0.5) <= PARTNER_TOLERANCE else 0 for s in shifts)
        return tuple(sizes), doubled, tuple(indices)

    def momentum_transfers(self) -> tuple[KPoints, Array]:
        """Return the differences ``k2 − k1`` of a uniform mesh and their table.

        The transfers ``q = (k2 − k1) mod 1`` of a mesh form the Γ-centred
        mesh of the same sizes, whatever the shift; the table ``[kpt, kpt]``
        holds the index of the transfer of every pair ``(k1, k2)`` in that
        set (CONVENTIONS.md §2.3).

        Raises
        ------
        InputError
            If the set is not a uniform mesh with equal weights.
        """
        sizes = self.mesh_size()
        transfers = KPoints.mesh(sizes)
        xp = host_namespace()
        points = self.fractional_array(xp)
        shifts = xp.asarray(
            [
                min(point[axis] for point in self.fractional)
                for axis in range(len(sizes))
            ],
            dtype=xp.float64,
        )
        n = xp.asarray(sizes, dtype=xp.float64)
        indices = xp.astype(xp.floor(points * n - shifts * n + 0.5), xp.int64)
        difference = (
            xp.expand_dims(indices, 0) - xp.expand_dims(indices, 1)
        ) % xp.astype(n, xp.int64)
        strides = xp.asarray(
            [math.prod(sizes[axis + 1 :]) for axis in range(len(sizes))], dtype=xp.int64
        )
        return transfers, xp.sum(difference * strides, axis=-1)

    @property
    def is_time_reversal_invariant(self) -> tuple[bool, ...]:
        """Whether each point equals its own partner, ``k ≡ −k``."""
        return tuple(
            all(min(x, abs(x - 0.5), 1.0 - x) <= PARTNER_TOLERANCE for x in point)
            for point in self.fractional
        )

    def axis(self, operations: KPointOperations | None = None) -> KAxis:
        """Build the k axis of structured matrices over this set (ADR-0015).

        With ``operations``, the point operations of a space group on the
        set from :meth:`hafnium.symmetry.SpaceGroupAdaptation.operations`,
        the axis stores the irreducible wedge (ADR-0024).

        A uniform mesh carries its :meth:`momenta` on the axis, which the
        block tensors of ADR-0031 conserve; another set carries none.

        Raises
        ------
        InputError
            If the set is not closed under time reversal.
        """
        try:
            momenta, modulus = self.momenta()
        except InputError:
            momenta, modulus = None, None
        return KAxis(
            self.time_reversal_partners(),
            self.weights,
            operations=operations,
            momenta=momenta,
            modulus=modulus,
        )

    def time_reversal_partners(self) -> tuple[int, ...]:
        """Index of the partner ``(1 − k) mod 1`` of every point.

        Raises
        ------
        InputError
            If a partner is missing from the set, or a point is duplicated.
        """
        xp = host_namespace()
        points = self.fractional_array(xp)
        indices, counts = match_points(points, (-points) % 1.0)
        for k, count in enumerate(counts):
            if count == 0:
                raise InputError(
                    f"k-point {k} at {self.fractional[k]} has no time-reversal "
                    "partner in the set"
                )
            if count > 1:
                raise InputError("the k-point set holds duplicate points")
        return indices


def match_points(
    points: Array, targets: Array
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Locate ``targets`` ``[m, d]`` among ``points`` ``[n, d]`` modulo one.

    Returns the index in ``points`` of every target (the first match, or
    the nearest point when there is none) and the number of points within
    :data:`PARTNER_TOLERANCE` of it, zero for a missing target and more
    than one for duplicates. The targets are matched in chunks, bounding
    the work array.
    """
    xp = array_namespace(points, targets)
    indices: list[int] = []
    counts: list[int] = []
    for start in range(0, int(targets.shape[0]), _PARTNER_CHUNK):
        chunk = targets[start : start + _PARTNER_CHUNK, :]
        difference = xp.expand_dims(points, 0) - xp.expand_dims(chunk, 1)
        distance = xp.abs((difference + 0.5) % 1.0 - 0.5)
        matches = xp.all(distance <= PARTNER_TOLERANCE, axis=-1)
        counts.extend(int(c) for c in xp.sum(xp.astype(matches, xp.int64), axis=1))
        indices.extend(int(index) for index in xp.argmax(matches, axis=1))
    return tuple(indices), tuple(counts)

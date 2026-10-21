"""Lattices of one to three translation vectors (ADR-0007)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass

from hafnium.backend import Array, Namespace, array_namespace, default_namespace
from hafnium.errors import InputError
from hafnium.molecule.molecule import bohr_per_unit

Vector = tuple[float, float, float]

DEPENDENCE_TOLERANCE = 1e-12
"""Gram determinant, relative to the squared lengths, below which lattice
vectors count as linearly dependent."""


def _vector(values: Iterable[float], scale: float = 1.0) -> Vector:
    entries = tuple(float(value) * scale for value in values)
    if len(entries) != 3 or not all(math.isfinite(x) for x in entries):
        raise InputError(f"a lattice vector has three finite components, got {entries}")
    return (entries[0], entries[1], entries[2])


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _scaled(a: Vector, factor: float) -> Vector:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


@dataclass(frozen=True, slots=True)
class Lattice:
    """Translation vectors of a periodic system, in Bohr.

    ``vectors[i]`` is the i-th translation vector, shape ``[d, 3]`` with
    ``d`` in {1, 2, 3}; a chain or a slab stores only its periodic
    directions and carries no vacuum padding (ADR-0007). Reciprocal vectors
    satisfy ``b_i · a_j = 2π δ_ij``.
    """

    vectors: tuple[Vector, ...]

    def __post_init__(self) -> None:
        vectors = tuple(_vector(vector) for vector in self.vectors)
        object.__setattr__(self, "vectors", vectors)
        if not 1 <= len(vectors) <= 3:
            raise InputError(f"a lattice has one to three vectors, got {len(vectors)}")
        squared_lengths = math.prod(_dot(vector, vector) for vector in vectors)
        if self._gram_determinant() <= DEPENDENCE_TOLERANCE * squared_lengths:
            raise InputError("lattice vectors are linearly dependent or zero")

    @classmethod
    def from_vectors(
        cls, vectors: Iterable[Iterable[float]], *, unit: str = "angstrom"
    ) -> Lattice:
        """Build from one to three vectors, in Ångström by default."""
        scale = bohr_per_unit(unit)
        return cls(tuple(_vector(vector, scale) for vector in vectors))

    def _gram_determinant(self) -> float:
        """``det(A Aᵀ)``, the squared d-dimensional volume of the cell."""
        if len(self.vectors) == 1:
            return _dot(self.vectors[0], self.vectors[0])
        if len(self.vectors) == 2:
            normal = _cross(self.vectors[0], self.vectors[1])
            return _dot(normal, normal)
        a, b, c = self.vectors
        return _dot(a, _cross(b, c)) ** 2

    @property
    def dimension(self) -> int:
        """Number of periodic directions."""
        return len(self.vectors)

    @property
    def volume(self) -> float:
        """Measure of the unit cell: length, area or volume by dimension."""
        return math.sqrt(self._gram_determinant())

    @property
    def reciprocal_vectors(self) -> tuple[Vector, ...]:
        """Vectors ``b_i`` with ``b_i · a_j = 2π δ_ij``, in inverse Bohr."""
        if len(self.vectors) == 1:
            (a,) = self.vectors
            return (_scaled(a, 2.0 * math.pi / _dot(a, a)),)
        if len(self.vectors) == 2:
            a, b = self.vectors
            normal = _cross(a, b)
            factor = 2.0 * math.pi / _dot(normal, normal)
            return (
                _scaled(_cross(b, normal), factor),
                _scaled(_cross(normal, a), factor),
            )
        a, b, c = self.vectors
        factor = 2.0 * math.pi / _dot(a, _cross(b, c))
        return (
            _scaled(_cross(b, c), factor),
            _scaled(_cross(c, a), factor),
            _scaled(_cross(a, b), factor),
        )

    @property
    def reciprocal(self) -> Lattice:
        """The reciprocal lattice, so that its translations are the ``G`` vectors."""
        return Lattice(self.reciprocal_vectors)

    def vectors_array(self, xp: Namespace | None = None) -> Array:
        """Return the translation vectors as an array ``[d, 3]`` in Bohr."""
        xp = default_namespace() if xp is None else xp
        return xp.asarray(self.vectors, dtype=xp.float64)

    def reciprocal_vectors_array(self, xp: Namespace | None = None) -> Array:
        """Return the reciprocal vectors as an array ``[d, 3]`` in inverse Bohr."""
        xp = default_namespace() if xp is None else xp
        return xp.asarray(self.reciprocal_vectors, dtype=xp.float64)

    def fractional_coordinates(self, positions: Array) -> Array:
        """Coordinates ``[n, d]`` of ``positions`` ``[n, 3]`` in the lattice basis.

        Only the components along the periodic directions are returned.
        """
        xp = array_namespace(positions)
        reciprocal = self.reciprocal_vectors_array(xp)
        return positions @ xp.linalg.matrix_transpose(reciprocal) / (2.0 * math.pi)

    def cartesian_coordinates(self, fractional: Array) -> Array:
        """Positions ``[n, 3]`` in Bohr of fractional coordinates ``[n, d]``."""
        xp = array_namespace(fractional)
        return fractional @ self.vectors_array(xp)

    def wrap(self, positions: Array) -> Array:
        """Translate ``positions`` ``[n, 3]`` into the home cell.

        The fractional coordinates of the result lie in ``[0, 1)``; the
        components along open directions are unchanged.
        """
        xp = array_namespace(positions)
        shifts = xp.floor(self.fractional_coordinates(positions))
        return positions - self.cartesian_coordinates(shifts)

    def translations(self, cutoff: float) -> Array:
        """All lattice vectors ``T`` with ``|T| ≤ cutoff``, as ``[T, 3]`` in Bohr.

        Sorted by length, the zero vector first; vectors of equal length
        follow the lexicographic order of their integer coefficients, so
        ``T`` and ``−T`` are both present.
        """
        if not (math.isfinite(cutoff) and cutoff >= 0.0):
            raise InputError("the translation cutoff is a non-negative length")
        xp = default_namespace()
        bounds = [
            math.floor(cutoff * math.sqrt(_dot(b, b)) / (2.0 * math.pi) + 1e-9)
            for b in self.reciprocal_vectors
        ]
        combinations = list(itertools.product(*(range(-n, n + 1) for n in bounds)))
        coefficients = xp.asarray(combinations, dtype=xp.float64)
        vectors = coefficients @ self.vectors_array(xp)
        lengths = xp.linalg.vector_norm(vectors, axis=-1)
        within = lengths <= cutoff
        vectors, lengths = vectors[within], lengths[within]
        return vectors[xp.argsort(lengths, stable=True)]

    def supercell(self, multiples: tuple[int, ...]) -> Lattice:
        """Lattice with every vector ``i`` multiplied by ``multiples[i]``."""
        if len(multiples) != self.dimension or any(m < 1 for m in multiples):
            raise InputError(
                f"supercell multiples are {self.dimension} positive integers, "
                f"got {multiples}"
            )
        return Lattice(
            tuple(
                _scaled(vector, float(m))
                for vector, m in zip(self.vectors, multiples, strict=True)
            )
        )

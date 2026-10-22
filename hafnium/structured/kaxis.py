"""The k axis of a block structure: partner map, weights and half-zone storage.

A k-point set enters the structured types as sector metadata shared by every
block (ADR-0007, ADR-0015): for each point, the index of its time-reversal
partner ``−k`` and its weight. Which points a block stores follows from its
ring. Blocks over the real and quaternionic rings store one representative
of every pair ``{k, −k}``, the point with the lower index, in ascending
order, and derive the block at the partner by the antiunitary map of their
ring (complex conjugation, or the Kramers map ``Θ X Θ⁻¹``); blocks over the
complex ring store every point.

:class:`KAxis` knows nothing about reciprocal lattices. It is built from a
:class:`hafnium.symmetry.KPoints` set, which keeps the geometry, so that the
``symmetry`` package depends on ``structured`` and not the other way round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hafnium.errors import StructureError

WEIGHT_TOLERANCE = 1e-12
"""Allowed deviation of the weight sum from one, and between partner weights."""


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
        equal for the two members of a pair.
    """

    partners: tuple[int, ...]
    weights: tuple[float, ...]

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
        object.__setattr__(self, "partners", partners)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def gamma(cls) -> KAxis:
        """Build the axis holding the Γ point alone."""
        return cls((0,), (1.0,))

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
        """The half zone: the lower index of every pair, in ascending order."""
        return tuple(k for k, partner in enumerate(self.partners) if k <= partner)

    def stored(self, half_zone: bool) -> tuple[int, ...]:
        """Return the indices a block stores: the representatives, or every point."""
        return self.representatives if half_zone else tuple(range(self.n_kpt))

    def stored_weights(
        self, half_zone: bool
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Weights ``(own, partner)`` of the stored points.

        ``own`` is the weight of the stored point itself and ``partner`` the
        weight of the point derived from it, zero when nothing is derived
        (every point stored, or a time-reversal invariant point).
        """
        stored = self.stored(half_zone)
        own = tuple(self.weights[k] for k in stored)
        partner = tuple(
            self.weights[self.partners[k]]
            if half_zone and self.partners[k] != k
            else 0.0
            for k in stored
        )
        return own, partner

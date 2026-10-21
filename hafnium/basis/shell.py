"""Contracted shells of spherical Gaussian functions and their normalization.

Conventions (ADR-0006, ADR-0009): a shell of angular momentum ``l`` spans
``2l + 1`` real solid harmonics per contraction, ordered ``m = -l, ..., +l``
except for p shells, which are ordered ``x, y, z``. The
``coefficients`` of a shell are those of *normalized* primitives, the
convention of the Basis Set Exchange and of the Gaussian94 format;
:meth:`Shell.primitive_coefficients` returns the coefficients of bare
primitives ``r^l S_lm(θ, φ) exp(-α r²)`` such that each contracted
function has unit norm.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from hafnium.errors import BasisSetError


def primitive_norm(exponent: float, angular_momentum: int) -> float:
    """Norm of ``r^l exp(-α r²)`` times a unit-norm real solid harmonic."""
    power = angular_momentum + 1.5
    return math.sqrt(2.0 * (2.0 * exponent) ** power / math.gamma(power))


def primitive_overlap(first: float, second: float, angular_momentum: int) -> float:
    """Overlap of two normalized primitives with the same harmonic on one centre."""
    power = angular_momentum + 1.5
    return (2.0 * math.sqrt(first * second) / (first + second)) ** power


@dataclass(frozen=True, slots=True)
class Shell:
    """A contracted shell of spherical Gaussians on one centre.

    Parameters
    ----------
    angular_momentum
        ``l``; the shell spans ``2l + 1`` functions per contraction.
    exponents
        Primitive exponents in inverse square Bohr.
    coefficients
        One row per contraction, giving coefficients of normalized
        primitives. General contractions share the exponents and have
        several rows.
    """

    angular_momentum: int
    exponents: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "exponents", tuple(float(a) for a in self.exponents))
        object.__setattr__(
            self,
            "coefficients",
            tuple(tuple(float(c) for c in row) for row in self.coefficients),
        )
        if self.angular_momentum < 0:
            raise BasisSetError("angular momentum is non-negative")
        if not self.exponents:
            raise BasisSetError("a shell needs at least one primitive")
        if any(a <= 0.0 for a in self.exponents):
            raise BasisSetError("exponents are positive")
        if not self.coefficients:
            raise BasisSetError("a shell needs at least one contraction")
        if any(len(row) != len(self.exponents) for row in self.coefficients):
            raise BasisSetError("each contraction has one coefficient per primitive")

    @property
    def n_primitives(self) -> int:
        """Number of primitive Gaussians."""
        return len(self.exponents)

    @property
    def n_contractions(self) -> int:
        """Number of contracted radial functions sharing the exponents."""
        return len(self.coefficients)

    @property
    def n_functions(self) -> int:
        """Number of basis functions, ``(2l + 1)`` per contraction."""
        return self.n_contractions * (2 * self.angular_momentum + 1)

    def primitive_coefficients(self) -> tuple[tuple[float, ...], ...]:
        """Coefficients of bare primitives that make each contraction unit-norm."""
        l = self.angular_momentum
        norms = [primitive_norm(a, l) for a in self.exponents]
        rows: list[tuple[float, ...]] = []
        for row in self.coefficients:
            square = sum(
                ci * cj * primitive_overlap(ai, aj, l)
                for ci, ai in zip(row, self.exponents, strict=True)
                for cj, aj in zip(row, self.exponents, strict=True)
            )
            if square <= 0.0:
                raise BasisSetError("a contraction has zero norm")
            scale = 1.0 / math.sqrt(square)
            rows.append(
                tuple(c * norm * scale for c, norm in zip(row, norms, strict=True))
            )
        return tuple(rows)

    @classmethod
    def from_rows(
        cls,
        angular_momentum: int,
        exponents: Iterable[float | str],
        coefficients: Iterable[Iterable[float | str]],
    ) -> Shell:
        """Build from any iterables of numbers or numeric strings."""
        return cls(
            angular_momentum,
            tuple(float(a) for a in exponents),
            tuple(tuple(float(c) for c in row) for row in coefficients),
        )

"""The range-separated Coulomb kernel of a cell (ADR-0012, ADR-0018).

``1/r = erfc(ωr)/r + erf(ωr)/r``: the short-range part is summed over
real-space images out to the distance where ``erfc`` falls below a
precision, the long-range part over the reciprocal vectors within a cutoff
from the same precision, with the Fourier transform of ``erf(ωr)/r`` as the
kernel and its finite part at ``G = 0`` (ADR-0012). The kernels of three
dimensions are implemented; those of chains and slabs depend on the
profile of the pair densities along the open directions and follow later
in M5 (PLAN.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hafnium.backend import Array, array_namespace
from hafnium.errors import InputError
from hafnium.integrals.periodic.bloch import DEFAULT_LATTICE_SUM_PRECISION
from hafnium.molecule import Lattice, default_omega, reciprocal_cutoff


def require_three_dimensions(lattice: Lattice) -> None:
    """Raise unless ``lattice`` has the reciprocal-space kernels implemented."""
    if lattice.dimension != 3:
        raise InputError(
            "the reciprocal-space Coulomb kernels of chains and slabs are not "
            "implemented yet; only three-dimensional cells are supported"
        )


@dataclass(frozen=True, slots=True)
class RangeSeparation:
    """The split of the Coulomb kernel and the precision of its lattice sums.

    Parameters
    ----------
    omega
        Range-separation parameter of CONVENTIONS.md §3; results do not
        depend on it beyond the precision.
    precision
        Every neglected term of a lattice sum is below this value, and
        every cutoff is chosen so that the neglected tails are as well.
    """

    omega: float
    precision: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.omega) and self.omega > 0.0):
            raise InputError("the range-separation parameter omega is positive")
        if not (math.isfinite(self.precision) and self.precision > 0.0):
            raise InputError("the lattice-sum precision is a positive number")

    @classmethod
    def for_lattice(
        cls,
        lattice: Lattice,
        *,
        omega: float | None = None,
        precision: float = DEFAULT_LATTICE_SUM_PRECISION,
    ) -> RangeSeparation:
        """Build with ``omega`` defaulting to :func:`hafnium.molecule.default_omega`."""
        return cls(default_omega(lattice) if omega is None else omega, precision)

    @property
    def short_range_radius(self) -> float:
        """Distance beyond which ``erfc(ω r)`` is below the precision.

        From ``erfc(x) ≤ exp(−x²)`` the radius is ``sqrt(ln(1/precision)) / ω``.
        """
        return math.sqrt(max(math.log(1.0 / self.precision), 0.0)) / self.omega

    def reciprocal_vectors(self, lattice: Lattice, *, charge_product: float) -> Array:
        """Return the reciprocal vectors ``[G, 3]`` of the long-range sum.

        The zero vector comes first, the rest follow by length within the
        cutoff of :func:`hafnium.molecule.reciprocal_cutoff` for the
        ``charge_product`` of the two interacting distributions.
        """
        require_three_dimensions(lattice)
        cutoff = reciprocal_cutoff(
            lattice, self.omega, charge_product=charge_product, precision=self.precision
        )
        return lattice.reciprocal.translations(cutoff)

    def long_range_kernel(self, lattice: Lattice, g_vectors: Array) -> Array:
        """Fourier transform of ``erf(ω r) / r`` at every row of ``g_vectors``.

        ``4π exp(−G²/4ω²) / G²`` for ``G ≠ 0`` and the finite part
        ``−π/ω²`` at ``G = 0`` (ADR-0012), as a real array ``[G]`` in three
        dimensions.
        """
        require_three_dimensions(lattice)
        xp = array_namespace(g_vectors)
        g_squared = xp.sum(g_vectors * g_vectors, axis=-1)
        at_zero = g_squared == 0.0
        safe = xp.where(at_zero, xp.ones_like(g_squared), g_squared)
        kernel = 4.0 * math.pi * xp.exp(-safe / (4.0 * self.omega**2)) / safe
        finite_part = xp.full(
            g_squared.shape, -math.pi / self.omega**2, dtype=xp.float64
        )
        return xp.where(at_zero, finite_part, kernel)

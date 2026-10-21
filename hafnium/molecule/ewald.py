"""Ewald sums of point charges on a lattice (ADR-0012).

The Coulomb kernel is range-separated with the parameter ``omega`` of
CONVENTIONS.md §3, ``1/r = erfc(ωr)/r + erf(ωr)/r``. The short-range part
is summed over real-space images, the long-range part over reciprocal
vectors of the periodic directions, and both sums are cut off where their
neglected terms fall below a precision. The ``G = 0`` term of a charged
cell keeps its finite part, the uniform-background convention. The loops
run in the compiled layer; the cutoffs are chosen here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from hafnium import _kernels
from hafnium.backend import Array, array_namespace, to_host
from hafnium.errors import InputError
from hafnium.molecule.lattice import Lattice

DEFAULT_EWALD_PRECISION = 1e-12
"""Bound on the neglected tail of either lattice sum, in Hartree."""

_FIXED_POINT_STEPS = 12
"""Iterations of the cutoff equations, which converge geometrically."""

_TAIL_MARGIN = 2.0
"""Factor on the continuum estimate of a tail, which a lattice sum can exceed."""


@dataclass(frozen=True, slots=True)
class EwaldParameters:
    """Range separation and cutoffs of an Ewald sum.

    Parameters
    ----------
    omega
        Range-separation parameter: ``erfc(ωr)/r`` is summed in real space
        and ``erf(ωr)/r`` in reciprocal space.
    real_space_cutoff
        Images farther than this from a charge are neglected, in Bohr.
    reciprocal_cutoff
        Reciprocal vectors longer than this are neglected, in inverse Bohr.
    """

    omega: float
    real_space_cutoff: float
    reciprocal_cutoff: float


def ewald_parameters(
    lattice: Lattice,
    charges: Array,
    *,
    omega: float | None = None,
    precision: float = DEFAULT_EWALD_PRECISION,
) -> EwaldParameters:
    """Choose the cutoffs so that the neglected tails are below ``precision``.

    Each tail is estimated by the integral over the neglected images or
    reciprocal vectors with the charges concentrated, ``Q = Σ|Z_i|``, and
    a factor of two on top. ``omega`` defaults to ``sqrt(π) / V^(1/d)`` for
    the d-dimensional cell measure ``V``, which balances the two sums.
    """
    if not (math.isfinite(precision) and precision > 0.0):
        raise InputError("the Ewald precision is a positive number")
    if omega is None:
        omega = math.sqrt(math.pi) / math.pow(lattice.volume, 1.0 / lattice.dimension)
    if not (math.isfinite(omega) and omega > 0.0):
        raise InputError("the range-separation parameter omega is positive")
    xp = array_namespace(charges)
    scale = _TAIL_MARGIN * float(xp.sum(xp.abs(charges))) ** 2
    return EwaldParameters(
        omega,
        _real_space_cutoff(lattice, omega, scale, precision),
        _reciprocal_cutoff(lattice, omega, scale, precision),
    )


def _solve_tail(prefactor: Callable[[float], float], precision: float) -> float:
    """Return ``x ≥ 1`` with ``prefactor(x) exp(−x²) ≤ precision``.

    The prefactors fall off at most algebraically, so the fixed-point
    iteration ``x ← sqrt(ln(prefactor(x) / precision))`` converges.
    """
    x = 1.0
    for _ in range(_FIXED_POINT_STEPS):
        x = math.sqrt(max(math.log(max(prefactor(x) / precision, 1.0)), 1.0))
    return x


def _real_space_cutoff(
    lattice: Lattice, omega: float, scale: float, precision: float
) -> float:
    """Distance beyond which the real-space tail is below ``precision``.

    With ``erfc(ωt) ≤ exp(−ω²t²)`` the tail over images farther than
    ``r = x/ω`` is at most ``πQ² exp(−x²) / (Vω²)`` in three dimensions,
    ``πQ² exp(−x²) / (2Aωx)`` for a slab and ``Q² exp(−x²) / (2Lx²)`` for a
    chain.
    """
    volume = lattice.volume

    def prefactor(x: float) -> float:
        if lattice.dimension == 3:
            return math.pi * scale / (volume * omega * omega)
        if lattice.dimension == 2:
            return math.pi * scale / (2.0 * volume * omega * x)
        return scale / (2.0 * volume * x * x)

    return _solve_tail(prefactor, precision) / omega


def _reciprocal_cutoff(
    lattice: Lattice, omega: float, scale: float, precision: float
) -> float:
    """Length beyond which the reciprocal-space tail is below ``precision``.

    The kernels are bounded by ``4π exp(−G²/4ω²)/G²``, ``3π exp(−G²/4ω²)/G``
    and ``4ω² exp(−G²/4ω²)/G²`` by dimension; integrated over the vectors
    longer than ``G = 2ωy`` the tails are at most ``Q²ω exp(−y²) / (πy)``,
    ``3Q²ω exp(−y²) / (4y)`` and ``Q²ω exp(−y²) / (2πy³)``.
    """

    def prefactor(y: float) -> float:
        if lattice.dimension == 3:
            return scale * omega / (math.pi * y)
        if lattice.dimension == 2:
            return 3.0 * scale * omega / (4.0 * y)
        return scale * omega / (2.0 * math.pi * y**3)

    return 2.0 * omega * _solve_tail(prefactor, precision)


def ewald_energy(
    lattice: Lattice,
    charges: Array,
    positions: Array,
    *,
    omega: float | None = None,
    precision: float = DEFAULT_EWALD_PRECISION,
) -> float:
    """Electrostatic energy per cell of point ``charges`` on ``lattice``, in Hartree.

    Parameters
    ----------
    charges : Array[n]
        Charges in units of the elementary charge; the cell need not be
        neutral, a net charge is compensated by a uniform background.
    positions : Array[n, 3]
        Positions in Bohr, anywhere in space.
    omega
        Range-separation parameter; the result is independent of it to
        within ``precision``.
    """
    xp = array_namespace(charges, positions)
    n_charges = int(charges.shape[0]) if charges.ndim == 1 else -1
    if charges.ndim != 1 or positions.shape != (n_charges, 3):
        raise InputError("charges have shape [n] and positions shape [n, 3]")
    parameters = ewald_parameters(lattice, charges, omega=omega, precision=precision)
    wrapped = lattice.wrap(positions)
    separations = xp.expand_dims(wrapped, 0) - xp.expand_dims(wrapped, 1)
    diameter = float(xp.max(xp.linalg.vector_norm(separations, axis=-1)))
    translations = lattice.translations(parameters.real_space_cutoff + diameter)
    g_vectors = lattice.reciprocal.translations(parameters.reciprocal_cutoff)[1:]
    return float(
        _kernels.ewald_energy(
            to_host(lattice.vectors_array()),
            to_host(charges),
            to_host(wrapped),
            to_host(translations),
            to_host(g_vectors),
            parameters.omega,
        )
    )

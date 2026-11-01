"""The range-separated Coulomb kernel of a cell (ADR-0012, ADR-0018, ADR-0021).

``1/r = erfc(ωr)/r + erf(ωr)/r``: the short-range part is summed over
real-space images out to the distance where ``erfc`` falls below a
precision, the long-range part over reciprocal space within a cutoff from
the same precision, with the Fourier transform of ``erf(ωr)/r`` as the
kernel and its finite part at ``G = 0`` (ADR-0012). A slab or a chain has
reciprocal vectors along its periodic directions only; along the open
directions the long-range term is an integral over the perpendicular
components of the three-dimensional transform, evaluated by quadrature,
with the finite part of ADR-0012 on the line ``q + G_∥ = 0`` (ADR-0021).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from hafnium.backend import Array, Namespace, array_namespace, host_namespace
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals.periodic.bloch import DEFAULT_LATTICE_SUM_PRECISION
from hafnium.molecule import Cell, Lattice, default_omega, reciprocal_cutoff

_PANEL_NODES = 16
"""Gauss–Legendre nodes per panel of the perpendicular quadrature.

Sixteen nodes integrate a function analytic inside the Bernstein ellipse of
parameter four around a panel to about 1e-19 of its size on that ellipse;
the panel width keeps the kernel's poles at ``±i|q + G_∥|``, the Gaussian
and the oscillation of the transforms within such an ellipse."""

_PANEL_GAUSSIAN_WIDTH = 2.0
"""A panel spans at most this multiple of the Gaussian width ``1/sqrt(β)``
of the integrand, ``exp(−β G²)`` with ``β`` from the kernel and the most
diffuse functions."""

_PANEL_OSCILLATION = 4.0
"""A panel spans at most this many radians of the oscillation
``exp(i G·Δ)`` of the largest perpendicular separation ``Δ``."""

_ANGULAR_PRECISION = 1e-14
"""Relative aliasing error allowed of the trapezoid rule in the polar angle
of a chain's perpendicular plane."""

_NEWTON_STEPS = 100
"""Iterations allowed for a Gauss–Legendre node, which converge in a few."""


@dataclass(frozen=True, slots=True)
class PerpendicularProfile:
    """How the transforms vary along the open directions of a slab or a chain.

    The quadrature over the perpendicular components resolves three
    features of the integrand (ADR-0021): the oscillation ``exp(i G·Δ)`` of
    the largest perpendicular separation ``extent`` of two centres, the
    Gaussian ``exp(−G²/4α)`` of the most diffuse function, whose exponent
    is ``smallest_exponent`` (two such functions give the narrowest
    product), and the polynomial of ``degree`` in the components that the
    transforms of Hermite Gaussians carry. A three-dimensional cell has no
    open direction and ignores the profile.
    """

    extent: float
    smallest_exponent: float
    degree: int

    def __post_init__(self) -> None:
        # Point charges carry no Gaussian factor: an infinite exponent is allowed.
        if not (math.isfinite(self.extent) and self.extent >= 0.0):
            raise InputError("the perpendicular extent is a non-negative length")
        if not self.smallest_exponent > 0.0:
            raise InputError("the smallest exponent is a positive number")
        if self.degree < 0:
            raise InputError("the polynomial degree is a non-negative integer")

    @classmethod
    def from_functions(cls, cell: Cell, *bases: BasisSet) -> PerpendicularProfile:
        """Build for products of one transform per basis of ``bases`` on ``cell``.

        The extent is the largest perpendicular separation of two atoms of
        the cell, the exponent the smallest over every shell, and the degree
        twice the largest angular momentum of every basis added up, a bound
        on the degree of a pair transform of each basis.
        """
        if not bases:
            raise InputError("a perpendicular profile takes at least one basis")
        xp = host_namespace()
        perpendicular = cell.positions_array(xp) @ xp.linalg.matrix_transpose(
            cell.lattice.perpendicular_vectors_array(xp)
        )
        separations = xp.expand_dims(perpendicular, 1) - xp.expand_dims(
            perpendicular, 0
        )
        extent = float(xp.max(xp.linalg.vector_norm(separations, axis=-1)))
        smallest = min(
            min(shell.exponents) for basis in bases for shell in basis.shells
        )
        degree = sum(2 * basis.max_angular_momentum for basis in bases)
        return cls(extent, smallest, degree)


@dataclass(frozen=True, slots=True)
class ReciprocalSum:
    """The vectors and weights of a long-range reciprocal-space sum.

    A long-range term is ``(1/V) Σ_G weights[G] conj(ρ̂_A(G)) ρ̂_B(G)`` over
    the ``vectors`` ``[G, 3]`` in inverse Bohr, for the Fourier transforms
    of two distributions and the measure ``V`` of the cell. In three
    dimensions the vectors are the reciprocal lattice within the cutoff and
    the weights the kernel, with the finite part at the zero vector. For a
    slab or a chain the vectors are the quadrature nodes along the open
    directions on every line ``q + G_∥``, the weights the kernel times the
    quadrature weights, and the zero vector carries the finite part minus
    the quadrature's own sum of the kernel over the line ``q + G_∥ = 0``, so
    that the sum evaluates the regularized integral of ADR-0021.
    """

    vectors: Array
    weights: Array

    def __post_init__(self) -> None:
        n = self.vectors.shape[0] if self.vectors.ndim == 2 else -1
        if self.vectors.shape != (n, 3) or self.weights.shape != (n,):
            raise InputError("vectors have shape [G, 3] and weights shape [G]")

    @property
    def n_vectors(self) -> int:
        """Number of vectors."""
        return int(self.vectors.shape[0])


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

    def finite_part(self, lattice: Lattice) -> float:
        """Return the finite part of the kernel at ``G = 0`` for coinciding centres.

        ``−π/ω²`` in three dimensions, ``−2 sqrt(π)/ω`` for a slab and
        ``2 ln ω`` for a chain (ADR-0012), which the regularized integral of
        ADR-0021 adds to the quadrature over the line ``G_∥ = 0``.
        """
        if lattice.dimension == 3:
            return -math.pi / self.omega**2
        if lattice.dimension == 2:
            return -2.0 * math.sqrt(math.pi) / self.omega
        return 2.0 * math.log(self.omega)

    def reciprocal_sum(
        self,
        lattice: Lattice,
        profile: PerpendicularProfile,
        *,
        charge_product: float,
        transfer: Array | None = None,
    ) -> ReciprocalSum:
        """Return the vectors and weights of the long-range sum at ``transfer``.

        The lines ``q + G_∥`` run over the reciprocal lattice within the
        cutoff of :func:`hafnium.molecule.reciprocal_cutoff` for the
        ``charge_product`` of the two interacting distributions, extended
        by the length of the momentum transfer ``q`` (zero by default), so
        that ``q + G_∥`` still reaches the cutoff; the three-dimensional
        bound applies in every dimension, since the perpendicular integral
        of a slab or a chain approximates the same continuum sum. Along the
        open directions the quadrature is composite Gauss–Legendre out to
        the cutoff, in panels narrow enough for the poles of the kernel at
        ``±i|q + G_∥|``, for the Gaussian and for the oscillation of
        ``profile``, and on a chain a trapezoid rule in the polar angle
        besides. The zero vector comes first when the transfer is zero.
        """
        xp = host_namespace()
        q = (
            xp.zeros(3, dtype=xp.float64)
            if transfer is None
            else xp.asarray(transfer, dtype=xp.float64)
        )
        if q.shape != (3,):
            raise InputError("a momentum transfer is a vector of three components")
        cutoff = reciprocal_cutoff(
            lattice,
            self.omega,
            charge_product=charge_product,
            precision=self.precision,
            dimension=3,
        )
        margin = float(xp.linalg.vector_norm(q))
        lines = lattice.reciprocal.translations(cutoff + margin) + q
        lengths = xp.linalg.vector_norm(lines, axis=-1)
        on_zero = lengths == 0.0
        finite = self.finite_part(lattice)
        if lattice.dimension == 3:
            at_zero = xp.full(lengths.shape, finite, dtype=xp.float64)
            return ReciprocalSum(lines, xp.where(on_zero, at_zero, self._kernel(lines)))
        nodes, node_weights = self._perpendicular_rule(
            lattice, profile, cutoff, lengths
        )
        vectors = xp.reshape(
            xp.expand_dims(lines, 1) + xp.expand_dims(nodes, 0), (-1, 3)
        )
        weights = self._kernel(vectors) * xp.tile(node_weights, (int(lines.shape[0]),))
        if bool(xp.any(on_zero)):
            # Σ_j w_j ṽ(G_j) [F(G_j) − F(0)] + v_0 F(0): the zero vector takes
            # the finite part minus the quadrature's own sum over the line.
            subtracted = finite - float(xp.sum(node_weights * self._kernel(nodes)))
            vectors = xp.concat([xp.zeros((1, 3), dtype=xp.float64), vectors])
            weights = xp.concat([xp.asarray([subtracted], dtype=xp.float64), weights])
        return ReciprocalSum(vectors, weights)

    def _kernel(self, vectors: Array) -> Array:
        """``4π exp(−G²/4ω²) / G²`` at every row, infinite at the zero vector."""
        xp = array_namespace(vectors)
        g_squared = xp.sum(vectors * vectors, axis=-1)
        safe = xp.where(g_squared == 0.0, xp.ones_like(g_squared), g_squared)
        kernel = 4.0 * math.pi * xp.exp(-safe / (4.0 * self.omega**2)) / safe
        return xp.where(g_squared == 0.0, xp.asarray(math.inf), kernel)

    def _perpendicular_rule(
        self,
        lattice: Lattice,
        profile: PerpendicularProfile,
        cutoff: float,
        line_lengths: Array,
    ) -> tuple[Array, Array]:
        """Nodes ``[node, 3]`` and weights ``[node]`` over the open directions.

        The weights include the measure ``d^n G_⊥ / (2π)^n`` of the
        ``n = 3 − d`` open directions. No node lies at the origin.
        """
        xp = array_namespace(line_lengths)
        nonzero = line_lengths[line_lengths > 0.0]
        pole = float(xp.min(nonzero)) if int(nonzero.shape[0]) else cutoff
        # The kernel's Gaussian and that of two functions of the smallest
        # exponent, the narrowest product of transforms in G.
        gaussian = 1.0 / (4.0 * self.omega**2) + 1.0 / (2.0 * profile.smallest_exponent)
        width = min(pole, _PANEL_GAUSSIAN_WIDTH / math.sqrt(gaussian), cutoff)
        if profile.extent > 0.0:
            width = min(width, _PANEL_OSCILLATION / profile.extent)
        n_panels = max(1, math.ceil(cutoff / width))
        radial, radial_weights = _composite_gauss_legendre(xp, n_panels, cutoff)
        perpendicular = lattice.perpendicular_vectors_array(xp)
        if lattice.dimension == 2:
            nodes = xp.concat([radial, -radial])
            weights = xp.concat([radial_weights, radial_weights]) / (2.0 * math.pi)
            return xp.expand_dims(nodes, 1) * perpendicular[0], weights
        n_angles = profile.degree + 1 + _aliasing_order(cutoff * profile.extent)
        angles = 2.0 * math.pi * xp.arange(n_angles, dtype=xp.float64) / n_angles
        directions = (
            xp.expand_dims(xp.cos(angles), 1) * perpendicular[0]
            + xp.expand_dims(xp.sin(angles), 1) * perpendicular[1]
        )
        nodes = xp.reshape(
            xp.expand_dims(radial, (1, 2)) * xp.expand_dims(directions, 0), (-1, 3)
        )
        weights = xp.reshape(
            xp.broadcast_to(
                xp.expand_dims(radial_weights * radial, 1),
                (int(radial.shape[0]), n_angles),
            )
            / (2.0 * math.pi * n_angles),
            (-1,),
        )
        return nodes, weights


def _aliasing_order(x: float) -> int:
    """Smallest ``m`` with ``(x/2)^m / m! ≤ _ANGULAR_PRECISION``.

    ``exp(i x cos θ)`` has the Fourier coefficients ``J_m(x)``, bounded by
    ``(x/2)^m / m!``, so a trapezoid rule with this many nodes beyond the
    polynomial degree aliases less than the precision of it.
    """
    m, term = 0, 1.0
    while term > _ANGULAR_PRECISION:
        m += 1
        term *= 0.5 * x / m
    return m


def _composite_gauss_legendre(
    xp: Namespace, n_panels: int, upper: float
) -> tuple[Array, Array]:
    """Nodes and weights of ``n_panels`` Gauss–Legendre panels on ``(0, upper)``."""
    nodes, weights = _gauss_legendre(_PANEL_NODES)
    width = upper / n_panels
    starts = width * xp.arange(n_panels, dtype=xp.float64)
    half = 0.5 * width
    all_nodes = xp.expand_dims(starts, 1) + half * (
        xp.asarray(nodes, dtype=xp.float64) + 1.0
    )
    all_weights = xp.broadcast_to(
        half * xp.asarray(weights, dtype=xp.float64), (n_panels, _PANEL_NODES)
    )
    return xp.reshape(all_nodes, (-1,)), xp.reshape(all_weights, (-1,))


@functools.cache
def _gauss_legendre(n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Nodes and weights of the ``n``-point Gauss–Legendre rule on ``[−1, 1]``.

    By Newton iteration on ``P_n`` from the asymptotic estimate of each root.
    """
    nodes, weights = [0.0] * n, [0.0] * n
    for i in range((n + 1) // 2):
        z = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        for _ in range(_NEWTON_STEPS):
            value, derivative = _legendre(n, z)
            step = value / derivative
            z -= step
            if abs(step) < 1e-15:
                break
        _, derivative = _legendre(n, z)
        weight = 2.0 / ((1.0 - z * z) * derivative * derivative)
        nodes[i], nodes[n - 1 - i] = -z, z
        weights[i] = weights[n - 1 - i] = weight
    return tuple(nodes), tuple(weights)


def _legendre(n: int, z: float) -> tuple[float, float]:
    """``P_n(z)`` and its derivative by the three-term recurrence."""
    previous, current = 0.0, 1.0
    for j in range(1, n + 1):
        previous, current = (
            current,
            ((2 * j - 1) * z * current - (j - 1) * previous) / j,
        )
    return current, n * (z * current - previous) / (z * z - 1.0)

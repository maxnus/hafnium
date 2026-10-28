"""Noncollinear functionals by the multi-collinear approach (M7, ADR-0029).

A collinear functional of the spin densities ``f(ρ_α, ρ_β, σ, τ)`` becomes
a functional of the density ``ρ`` and the magnetization ``m`` by projecting
the spin variables ``S = (m, ∇m, τ_m)`` onto every direction ``n`` of spin
space and averaging the collinear functional of ``(ρ ± n·S)/2`` over the
sphere with the derivative term that restores the collinear limit,

``f_mc(ρ, S) = ∫ dΩ/4π [ f(ρ, n·S) + (n·S)·∂f/∂s (ρ, n·S) ]``,

so that a collinear density (all spin variables parallel) recovers the
spin-polarized functional exactly. For an LDA the integral has the closed
form ``f(ρ, |m|)``; a GGA or meta-GGA samples the sphere and needs the
second derivatives of ``f``, which the compiled layer takes from ``xcx``.
The sphere is sampled with product rules aligned with the magnetization of
every point and refined until the energy density and the energy response
converge to a precision, or with one Lebedev rule for every point as PySCF
does. Where ``|m|`` approaches ``ρ`` the integrand loses its smoothness and
no rule converges, so points polarized beyond a threshold take the locally
collinear functional ``f(ρ, |m|, ∇|m|, m̂·τ_m)`` with its exact derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hafnium import _kernels
from hafnium.backend import Array, array_namespace, host_array, to_host
from hafnium.errors import InputError
from hafnium.grid.angular import lebedev_degree, lebedev_grid
from hafnium.xc.functional import Functional

DEFAULT_PRECISION = 1e-14
"""Tolerated change of the energy density and energy response between two
levels of the aligned rules, in Hartree per point once the quadrature
weight is applied."""

DEFAULT_MAX_LEVEL = 6
"""Finest level of the aligned rules, 128 × 128 directions."""

MAX_LEVEL = 7
"""Level the compiled layer refines to at most, 256 × 256 directions."""

DEFAULT_COLLINEAR_THRESHOLD = 0.99
"""Polarization ``|m|/ρ`` from which a point is evaluated locally collinear,
PySCF's default."""


@dataclass(frozen=True, slots=True)
class NoncollinearOptions:
    """Numerical controls of the multi-collinear evaluation.

    Parameters
    ----------
    precision
        Tolerated change of the energy density and of the first-order
        energy response between two levels of the aligned rules, in
        Hartree per point once the point's quadrature weight is applied;
        the energy integrates to within the number of points times this.
    max_level
        Finest level of the aligned rules, ``2·2^level`` nodes along the
        magnetization times as many azimuths; points that do not converge
        at this level are counted and reported.
    collinear_threshold
        Points with ``|m| ≥ collinear_threshold · ρ`` take the locally
        collinear functional; ``None`` keeps every point on the sphere,
        which loses accuracy where a spin density vanishes.
    lebedev_points
        Size of a Lebedev rule to use for every point instead of the
        aligned rules, as PySCF does with 770 points; ``None`` for the
        aligned rules.
    """

    precision: float = DEFAULT_PRECISION
    max_level: int = DEFAULT_MAX_LEVEL
    collinear_threshold: float | None = DEFAULT_COLLINEAR_THRESHOLD
    lebedev_points: int | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.precision) and self.precision > 0.0):
            raise InputError("the precision is a positive number")
        if not 0 <= self.max_level <= MAX_LEVEL:
            raise InputError(f"the level of the aligned rules is at most {MAX_LEVEL}")
        if self.lebedev_points is not None:
            lebedev_degree(self.lebedev_points)
        threshold = self.collinear_threshold
        if threshold is not None and not (math.isfinite(threshold) and threshold > 0.0):
            raise InputError("the collinear threshold is a positive number or None")


@dataclass(frozen=True, slots=True)
class NoncollinearDensity:
    """Density and magnetization with their derivatives at grid points.

    Parameters
    ----------
    rho : Array[g]
        The density ``ρ_αα + ρ_ββ``.
    m : Array[3, g]
        The magnetization ``(2 Re ρ_αβ, −2 Im ρ_αβ, ρ_αα − ρ_ββ)`` of the
        spin-density matrix ``ρ_στ``.
    grad_rho : Array[3, g]
        ``∇ρ``; ``None`` for an LDA.
    grad_m : Array[3, 3, g]
        ``∂_i m_a`` with the spin direction ``a`` before the spatial
        direction ``i``; ``None`` for an LDA.
    tau : Array[g]
        The kinetic energy density ``½ Σ_i |∇ψ_i|²`` summed over the spins;
        ``None`` below the meta-GGA rung.
    tau_m : Array[3, g]
        The spin kinetic energy density ``½ Σ_i ∇ψ_i† σ ∇ψ_i``; ``None``
        below the meta-GGA rung.
    """

    rho: Array
    m: Array
    grad_rho: Array | None = None
    grad_m: Array | None = None
    tau: Array | None = None
    tau_m: Array | None = None

    def __post_init__(self) -> None:
        if self.rho.ndim != 1:
            raise InputError("the density has shape [g]")
        n = self.n_points
        for name, array, shape in (
            ("magnetization", self.m, (3, n)),
            ("density gradient", self.grad_rho, (3, n)),
            ("magnetization gradient", self.grad_m, (3, 3, n)),
            ("kinetic energy density", self.tau, (n,)),
            ("spin kinetic energy density", self.tau_m, (3, n)),
        ):
            if array is not None and tuple(array.shape) != shape:
                raise InputError(
                    f"the {name} has shape {list(shape)} over the points of the density"
                )
        if (self.grad_rho is None) != (self.grad_m is None):
            raise InputError("the gradients of density and magnetization come together")
        if (self.tau is None) != (self.tau_m is None):
            raise InputError("the kinetic energy densities come together")

    @property
    def n_points(self) -> int:
        """Number of points."""
        return int(self.rho.shape[0])


@dataclass(frozen=True, slots=True)
class NoncollinearValues:
    """Energy density and derivatives of the multi-collinear functional.

    The layouts follow :class:`NoncollinearDensity`; the derivatives of
    variables the rung does not read are ``None``.

    Parameters
    ----------
    energy_density : Array[g]
        ``f_mc``, the energy per volume (not per particle).
    v_rho, v_m
        ``∂f_mc/∂ρ`` and ``∂f_mc/∂m_a``.
    v_grad_rho, v_grad_m
        ``∂f_mc/∂(∂_i ρ)`` and ``∂f_mc/∂(∂_i m_a)``.
    v_tau, v_tau_m
        ``∂f_mc/∂τ`` and ``∂f_mc/∂τ_m,a``.
    unconverged
        Points whose aligned rules reached the finest level without
        converging to the precision; zero for a Lebedev rule. The caller
        decides whether to report them.
    """

    energy_density: Array
    v_rho: Array
    v_m: Array
    v_grad_rho: Array | None
    v_grad_m: Array | None
    v_tau: Array | None
    v_tau_m: Array | None
    unconverged: int = 0


def evaluate_noncollinear(
    functional: Functional,
    density: NoncollinearDensity,
    options: NoncollinearOptions | None = None,
    *,
    weights: Array | None = None,
) -> NoncollinearValues:
    """Evaluate the multi-collinear form of ``functional`` at every point.

    ``weights`` are the quadrature weights of the points, which scale the
    precision of the aligned rules; without them every point is refined to
    the precision itself. The exact exchange of a hybrid is not part of the
    result. A functional without components returns zeros.

    Raises
    ------
    InputError
        If the rung needs gradients or kinetic energy densities the
        density does not carry, or the weights do not match the points.
    """
    options = NoncollinearOptions() if options is None else options
    xp = array_namespace(density.rho)
    n = density.n_points
    gradients = functional.needs_gradient
    kinetic = functional.needs_tau
    if weights is not None and tuple(weights.shape) != (n,):
        raise InputError("the point weights have shape [g] over the points")
    if not functional.components:
        zeros = xp.zeros((n,), dtype=xp.float64)
        return NoncollinearValues(
            zeros,
            zeros,
            xp.zeros((3, n), dtype=xp.float64),
            xp.zeros((3, n), dtype=xp.float64) if gradients else None,
            xp.zeros((3, 3, n), dtype=xp.float64) if gradients else None,
            zeros if kinetic else None,
            xp.zeros((3, n), dtype=xp.float64) if kinetic else None,
        )
    if gradients and density.grad_rho is None:
        raise InputError(f"{functional.name} needs the density gradients")
    if kinetic and density.tau is None:
        raise InputError(f"{functional.name} needs the kinetic energy densities")
    names = [component for component, _ in functional.components]
    component_weights = host_array([weight for _, weight in functional.components])
    directions = direction_weights = None
    if gradients and options.lebedev_points is not None:
        points, rule_weights = lebedev_grid(options.lebedev_points)
        directions, direction_weights = to_host(points), to_host(rule_weights)
    try:
        results = _kernels.evaluate_noncollinear(
            names,
            component_weights,
            to_host(density.rho),
            to_host(density.m),
            _host_if(density.grad_rho, gradients),
            _host_if(density.grad_m, gradients),
            _host_if(density.tau, kinetic),
            _host_if(density.tau_m, kinetic),
            directions,
            direction_weights,
            options.precision,
            options.max_level,
            options.collinear_threshold,
            None if weights is None else to_host(weights),
        )
    except ValueError as error:
        raise InputError(str(error)) from None
    energy, v_rho, v_m, v_grad_rho, v_grad_m, v_tau, v_tau_m, unconverged = results
    return NoncollinearValues(
        xp.asarray(energy),
        xp.asarray(v_rho),
        xp.asarray(v_m),
        None if v_grad_rho is None else xp.asarray(v_grad_rho),
        None if v_grad_m is None else xp.asarray(v_grad_m),
        None if v_tau is None else xp.asarray(v_tau),
        None if v_tau_m is None else xp.asarray(v_tau_m),
        unconverged,
    )


def _host_if(array: Array | None, wanted: bool) -> Array | None:
    """``array`` on the host when ``wanted`` and present, else ``None``."""
    if array is None or not wanted:
        return None
    return to_host(array)

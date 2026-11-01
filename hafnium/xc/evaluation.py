"""Evaluation of a functional on the densities of a grid.

Under JAX the evaluation is a host callback with a custom Jacobian-vector
product from the second derivatives the ``xcx`` crate supplies, so the
energy density and the potential differentiate under ``jvp``, ``grad`` and
``jit`` (ADR-0037); under any other backend the kernel is called directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hafnium import _kernels
from hafnium.backend import Array, array_namespace, host_array, is_jax_array, to_host
from hafnium.errors import InputError
from hafnium.xc.functional import Functional

Values = tuple[Array, Array, Array | None, Array | None]
"""``(energy_density, vrho, vsigma, vtau)``."""


@dataclass(frozen=True, slots=True)
class DensityOnGrid:
    """Collinear spin densities and their derivatives at grid points.

    Parameters
    ----------
    rho : Array[2, g]
        The alpha and beta densities.
    sigma : Array[3, g]
        The contracted gradients ``∇ρ_α·∇ρ_α, ∇ρ_α·∇ρ_β, ∇ρ_β·∇ρ_β``;
        ``None`` for an LDA.
    tau : Array[2, g]
        The kinetic energy densities ``½ Σ_i |∇ψ_iσ|²``; ``None`` below the
        meta-GGA rung.
    """

    rho: Array
    sigma: Array | None = None
    tau: Array | None = None

    def __post_init__(self) -> None:
        n = int(self.rho.shape[1]) if self.rho.ndim == 2 else -1
        if self.rho.shape != (2, n):
            raise InputError("densities have shape [2, g]")
        if self.sigma is not None and self.sigma.shape != (3, n):
            raise InputError("contracted gradients have shape [3, g]")
        if self.tau is not None and self.tau.shape != (2, n):
            raise InputError("kinetic energy densities have shape [2, g]")

    @property
    def n_points(self) -> int:
        """Number of points."""
        return int(self.rho.shape[1])


@dataclass(frozen=True, slots=True)
class XcValues:
    """Energy per particle and first derivatives of the energy density.

    Parameters
    ----------
    energy_per_particle : Array[g]
        ``ε_xc``; the energy density is ``(ρ_α + ρ_β) ε_xc``.
    vrho : Array[2, g]
        ``∂(ρ ε_xc)/∂ρ_σ``.
    vsigma : Array[3, g]
        ``∂(ρ ε_xc)/∂σ``; ``None`` for an LDA.
    vtau : Array[2, g]
        ``∂(ρ ε_xc)/∂τ_σ``; ``None`` below the meta-GGA rung.
    """

    energy_per_particle: Array
    vrho: Array
    vsigma: Array | None
    vtau: Array | None


def evaluate_functional(functional: Functional, density: DensityOnGrid) -> XcValues:
    """Evaluate the semilocal components of ``functional`` at every point.

    The exact exchange of a hybrid is not part of the result. A functional
    without components returns zeros.
    """
    xp = array_namespace(density.rho)
    n = density.n_points
    if not functional.components:
        zeros = xp.zeros((n,), dtype=xp.float64)
        return XcValues(zeros, xp.zeros((2, n), dtype=xp.float64), None, None)
    if functional.needs_gradient and density.sigma is None:
        raise InputError(f"{functional.name} needs the density gradients")
    if functional.needs_tau and density.tau is None:
        raise InputError(f"{functional.name} needs the kinetic energy densities")
    sigma = density.sigma if functional.needs_gradient else None
    tau = density.tau if functional.needs_tau else None
    if is_jax_array(density.rho):
        values: Values = _rules().functional_values(functional, density.rho, sigma, tau)
        energy_density, vrho, vsigma, vtau = values
        total = density.rho[0] + density.rho[1]
        positive = total > 0.0
        safe = xp.where(positive, total, xp.ones_like(total))
        per_particle = xp.where(positive, energy_density / safe, xp.zeros_like(total))
        return XcValues(per_particle, vrho, vsigma, vtau)
    energy, vrho, vsigma, vtau = host_values(
        functional,
        to_host(density.rho),
        None if sigma is None else to_host(sigma),
        None if tau is None else to_host(tau),
        per_particle=True,
    )
    return XcValues(
        xp.asarray(energy),
        xp.asarray(vrho),
        None if vsigma is None else xp.asarray(vsigma),
        None if vtau is None else xp.asarray(vtau),
    )


def host_values(
    functional: Functional,
    rho: Array,
    sigma: Array | None,
    tau: Array | None,
    *,
    per_particle: bool = False,
) -> Values:
    """Evaluate ``functional`` on host arrays: the energy density and first derivatives.

    With ``per_particle`` the first array is the energy per particle, as
    the kernel returns it, instead of the energy density ``ρ ε_xc``.
    """
    names = [component for component, _ in functional.components]
    weights = host_array([weight for _, weight in functional.components])
    try:
        energy, vrho, vsigma, vtau = _kernels.evaluate_functional(
            names, weights, rho, sigma, tau
        )
    except ValueError as error:
        raise InputError(str(error)) from None
    if not per_particle:
        energy = (rho[0] + rho[1]) * energy
    return energy, vrho, vsigma, vtau


def host_second_derivatives(
    functional: Functional, rho: Array, sigma: Array | None, tau: Array | None
) -> tuple[Array, Array | None, Array | None, Array | None, Array | None, Array | None]:
    """Return the second derivatives of the energy density on host arrays.

    ``(v2rho2 [3, g], v2rhosigma [6, g], v2sigma2 [6, g], v2rhotau [4, g],
    v2sigmatau [6, g], v2tau2 [3, g])`` in libxc's packed layouts, ``None``
    for the rows the rung does not read.
    """
    names = [component for component, _ in functional.components]
    weights = host_array([weight for _, weight in functional.components])
    try:
        return _kernels.evaluate_functional_second(names, weights, rho, sigma, tau)
    except ValueError as error:
        raise InputError(str(error)) from None


def _rules() -> Any:
    """Import the JAX rule when a JAX array arrives, so JAX stays optional."""
    from hafnium.xc import _jax_rules

    return _jax_rules

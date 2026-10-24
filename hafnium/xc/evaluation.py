"""Evaluation of a functional on the densities of a grid."""

from __future__ import annotations

from dataclasses import dataclass

from hafnium import _kernels
from hafnium.backend import Array, array_namespace, host_array, to_host
from hafnium.errors import InputError
from hafnium.xc.functional import Functional


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
    names = [component for component, _ in functional.components]
    weights = host_array([weight for _, weight in functional.components])
    try:
        energy, vrho, vsigma, vtau = _kernels.evaluate_functional(
            names,
            weights,
            to_host(density.rho),
            None
            if not functional.needs_gradient or density.sigma is None
            else to_host(density.sigma),
            None
            if not functional.needs_tau or density.tau is None
            else to_host(density.tau),
        )
    except ValueError as error:
        raise InputError(str(error)) from None
    return XcValues(
        xp.asarray(energy),
        xp.asarray(vrho),
        None if vsigma is None else xp.asarray(vsigma),
        None if vtau is None else xp.asarray(vtau),
    )

"""Density fitting: metric, three-centre integrals and the fitted tensor.

The fitted tensor is ``B[aux, ao, ao]`` of CONVENTIONS.md §3.
The Coulomb metric is used: ``(uv|wx) ≈ Σ_P B[P, u, v] B[P, w, x]`` with
``B = M^{-1/2} (P|uv)`` and ``M = (P|Q)``. The symmetric inverse square
root of the metric comes from its eigendecomposition, which handles
near-linear dependence of the auxiliary basis by dropping small
eigenvalues; a Cholesky route may replace it later if measured to matter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hafnium.backend import Array, adjoint, array_namespace, eigh, host_array
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals.engine import IntegralEngine
from hafnium.integrals.shell_table import ShellTable
from hafnium.molecule import Molecule

logger = logging.getLogger(__name__)

DEFAULT_METRIC_THRESHOLD = 1e-10
"""Eigenvalues of the metric below this fraction of the largest are dropped."""


@dataclass(frozen=True, slots=True)
class DensityFitting:
    """Metric and three-centre integrals of an auxiliary basis.

    Parameters
    ----------
    metric : Array[aux, aux]
        ``(P|Q)``, with the full or the long-range Coulomb kernel.
    three_center : Array[aux, ao, ao]
        ``(P|uv)`` with the same kernel.
    """

    metric: Array
    three_center: Array

    def __post_init__(self) -> None:
        n_aux = self.metric.shape[0]
        if self.metric.shape != (n_aux, n_aux):
            raise InputError("the metric is square")
        if self.three_center.ndim != 3 or self.three_center.shape[0] != n_aux:
            raise InputError("three-centre integrals have shape [aux, ao, ao]")

    @classmethod
    def build(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        auxiliary: BasisSet,
        molecule: Molecule,
        *,
        omega: float | None = None,
    ) -> DensityFitting:
        """Evaluate the metric and three-centre integrals of a molecule.

        With ``omega`` both use the long-range kernel ``erf(ω r) / r``.
        """
        positions = molecule.positions_array()
        orbital = ShellTable.from_basis(basis, positions)
        fitting = ShellTable.from_basis(auxiliary, positions)
        zero = host_array([[0.0, 0.0, 0.0]])
        metric = engine.coulomb_two_center(fitting, fitting, zero, omega=omega)[0]
        three_center = engine.coulomb_three_center(
            fitting, orbital, orbital, host_array([[[0.0] * 3, [0.0] * 3]]), omega=omega
        )[0]
        return cls(metric, three_center)

    @property
    def n_aux(self) -> int:
        """Number of auxiliary functions."""
        return int(self.metric.shape[0])

    @property
    def n_ao(self) -> int:
        """Number of orbital basis functions."""
        return int(self.three_center.shape[1])

    def df_tensor(self, *, threshold: float = DEFAULT_METRIC_THRESHOLD) -> Array:
        """``df_tensor[aux, ao, ao]`` with the metric absorbed: ``M^{-1/2} (P|uv)``.

        Metric eigenvalues below ``threshold`` times the largest are dropped,
        which removes near-linear dependence of the auxiliary basis; the
        number dropped is logged, which reads the count on the host (the
        JAX milestone revisits this).
        """
        xp = array_namespace(self.metric)
        eigenvalues, eigenvectors = eigh(self.metric)
        cutoff = threshold * xp.max(eigenvalues)
        keep = eigenvalues > cutoff
        dropped = self.n_aux - int(xp.sum(keep))
        if dropped:
            logger.info(
                "density fitting dropped %d near-null metric eigenvalues", dropped
            )
        safe = xp.where(keep, eigenvalues, xp.ones_like(eigenvalues))
        inverse_root = xp.where(keep, 1.0 / xp.sqrt(safe), xp.zeros_like(eigenvalues))
        half = (eigenvectors * inverse_root) @ adjoint(eigenvectors)
        flat = xp.reshape(self.three_center, (self.n_aux, -1))
        return xp.reshape(half @ flat, self.three_center.shape)

    def eri(self, *, threshold: float = DEFAULT_METRIC_THRESHOLD) -> Array:
        """Fitted four-centre integrals ``eri[ao, ao, ao, ao]``, chemists' notation.

        Stored in full; for tests and small systems.
        """
        xp = array_namespace(self.metric)
        fitted = self.df_tensor(threshold=threshold)
        return xp.einsum("Puv,Pwx->uvwx", fitted, fitted)

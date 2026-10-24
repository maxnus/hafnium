"""The exchange–correlation term of a Kohn–Sham Hamiltonian (M7, ADR-0026).

The densities of a structured density matrix are evaluated on a molecular
grid from the basis functions and their derivatives, the functional is
evaluated point by point, and the potential is assembled back into the
structure of the density. The term is collinear: it reads the alpha and
beta densities of a restricted or unrestricted class, real or complex, so
that one implementation serves RKS and UKS; the spinor classes need the
noncollinear evaluation of PLAN.md §5.4, a later addition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hafnium.backend import Array, array_namespace, host_array
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.grid import GridSpec, MolecularGrid, build_grid
from hafnium.integrals import IntegralEngine, ShellTable
from hafnium.integrals.periodic import function_radius
from hafnium.molecule import Molecule
from hafnium.structured import BlockMatrix, BlockStructure, Spin
from hafnium.structured.validation import cast_to_ring
from hafnium.xc import DensityOnGrid, Functional, XcValues, evaluate_functional

logger = logging.getLogger(__name__)

DEFAULT_BASIS_CUTOFF = 1e-12
"""Basis function values below this are left out of the densities on the grid."""

DEFAULT_BATCH_SIZE = 8192
"""Grid points evaluated at a time, which bounds the memory of the basis values."""


@dataclass(frozen=True, slots=True, eq=False)
class ExchangeCorrelation:
    """A functional on a grid, for the densities of one basis.

    Parameters
    ----------
    functional
        The functional; its semilocal components are evaluated here, its
        exact exchange by the Hamiltonian.
    grid
        The quadrature grid.
    engine
        Evaluates the basis functions on the grid.
    table
        The shells of the basis on the atoms.
    radii : Array[shell]
        Cutoff radius of every shell for the basis values.
    batch_size
        Grid points evaluated at a time.
    """

    functional: Functional
    grid: MolecularGrid
    engine: IntegralEngine
    table: ShellTable
    radii: Array
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        if self.functional.vv10 is not None:
            raise InputError(
                f"{self.functional.name} is defined with VV10 nonlocal correlation, "
                "which is not implemented; functional.without_vv10() evaluates the "
                "rest of the functional"
            )
        if self.radii.shape != (self.table.n_shells,):
            raise InputError("one cutoff radius per shell")
        if self.batch_size < 1:
            raise InputError("a batch holds at least one point")

    @classmethod
    def from_molecule(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        molecule: Molecule,
        functional: Functional,
        *,
        grid: GridSpec | None = None,
        basis_cutoff: float = DEFAULT_BASIS_CUTOFF,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> ExchangeCorrelation:
        """Build the grid of ``molecule`` and lay ``basis`` out on its atoms.

        ``basis_cutoff`` bounds the basis function values neglected far
        from their centres.
        """
        table = ShellTable.from_basis(basis, molecule)
        radii = host_array(
            [function_radius(shell, basis_cutoff) for shell in basis.shells]
        )
        return cls(
            functional, build_grid(molecule, grid), engine, table, radii, batch_size
        )

    @property
    def n_ao(self) -> int:
        """Number of atomic orbitals."""
        return self.table.n_functions

    def evaluate(self, density: BlockMatrix) -> tuple[float, BlockMatrix]:
        """Exchange–correlation energy and potential of a Hermitian ``density``.

        The potential is in the structure of the density. The number of
        electrons found on the grid is logged at DEBUG level.

        Raises
        ------
        InputError
            For a spinor (generalized) structure or a k axis, which this
            version does not serve.
        """
        alpha, beta = self._spin_densities(density)
        xp = array_namespace(alpha, beta)
        order = 1 if self.functional.needs_gradient or self.functional.needs_tau else 0
        energy = 0.0
        electrons = 0.0
        potentials = [xp.zeros_like(alpha), xp.zeros_like(beta)]
        for batch in self.grid.batches(self.batch_size):
            values = self.engine.basis_values(
                self.table, self.grid.points[batch], self.radii, order=order
            )
            values = xp.asarray(values)
            weights = self.grid.weights[batch]
            batch_density = _densities(values, (alpha, beta), self.functional)
            xc = evaluate_functional(self.functional, batch_density.on_grid)
            total = batch_density.on_grid.rho[0] + batch_density.on_grid.rho[1]
            energy += float(xp.sum(weights * total * xc.energy_per_particle))
            electrons += float(xp.sum(weights * total))
            for spin in range(2):
                potentials[spin] = potentials[spin] + _potential(
                    values, weights, batch_density.gradients, xc, spin
                )
        logger.debug(
            "exchange-correlation: %.10f Hartree, %.8f electrons on %d points",
            energy,
            electrons,
            self.grid.n_points,
        )
        return energy, _structured(density.structure, potentials[0], potentials[1])

    def _spin_densities(self, density: BlockMatrix) -> tuple[Array, Array]:
        structure = density.structure
        if structure.k_axis is not None:
            raise InputError(
                "periodic Kohn–Sham is not available yet; the exchange–correlation "
                "term serves molecules"
            )
        if structure.is_spinor:
            raise InputError(
                "the exchange–correlation term is collinear: it serves the "
                "restricted and unrestricted classes, not the generalized ones "
                "(noncollinear functionals are a later addition, PLAN.md §5.4)"
            )
        if density.n_spatial != self.n_ao:
            raise InputError(
                f"the density has {density.n_spatial} atomic orbitals, the grid's "
                f"basis {self.n_ao}"
            )
        alpha, beta = density.spin_blocks()
        xp = array_namespace(alpha, beta)
        # The basis functions are real, so only the real part of a Hermitian
        # density enters the densities on the grid.
        return (
            xp.astype(xp.real(alpha), xp.float64),
            xp.astype(xp.real(beta), xp.float64),
        )


@dataclass(frozen=True, slots=True)
class _BatchDensities:
    """The densities of a batch with the spin gradients the potential needs."""

    on_grid: DensityOnGrid
    gradients: tuple[Array, ...]


def _densities(
    values: Array, densities: tuple[Array, Array], functional: Functional
) -> _BatchDensities:
    """Return the spin densities, gradients and kinetic energy densities of a batch.

    ``values`` is ``[g, component, ao]``; with real symmetric density matrices
    ``ρ_σ = Σ_uv χ_u D_σ,uv χ_v``, ``∇ρ_σ = 2 Σ_uv ∇χ_u D_σ,uv χ_v`` and
    ``τ_σ = ½ Σ_uv ∇χ_u · D_σ,uv ∇χ_v``.
    """
    xp = array_namespace(values)
    functions = values[:, 0, :]
    derivatives = values[:, 1:4, :] if values.shape[1] > 1 else None
    rho: list[Array] = []
    gradients: list[Array] = []
    taus: list[Array] = []
    for matrix in densities:
        half = functions @ matrix
        rho.append(xp.sum(half * functions, axis=1))
        if derivatives is None:
            continue
        gradients.append(2.0 * xp.einsum("gxu,gu->xg", derivatives, half))
        if functional.needs_tau:
            taus.append(
                0.5
                * xp.sum(
                    xp.einsum("gxu,uv->gxv", derivatives, matrix) * derivatives,
                    axis=(1, 2),
                )
            )
    sigma = None
    if gradients:
        sigma = xp.stack(
            [
                xp.sum(gradients[0] * gradients[0], axis=0),
                xp.sum(gradients[0] * gradients[1], axis=0),
                xp.sum(gradients[1] * gradients[1], axis=0),
            ]
        )
    return _BatchDensities(
        DensityOnGrid(xp.stack(rho), sigma, xp.stack(taus) if taus else None),
        tuple(gradients),
    )


def _potential(
    values: Array,
    weights: Array,
    gradients: tuple[Array, ...],
    xc: XcValues,
    spin: int,
) -> Array:
    """Return the potential matrix of one spin from a batch of points.

    With ``A_σ = ½ w v_ρσ Φ + Σ_x w (2 v_σσ ∂_xρ_σ + v_αβ ∂_xρ_σ') ∂_xΦ`` the
    matrix is ``Φᵀ A_σ + A_σᵀ Φ``, plus ``½ Σ_x w v_τσ ∂_xΦᵀ ∂_xΦ`` for a
    meta-GGA, with ``Φ[g, ao]`` the function values and ``∂_xΦ`` their
    derivatives.
    """
    xp = array_namespace(values)
    functions = values[:, 0, :]
    scaled = 0.5 * (weights * xc.vrho[spin])[:, None] * functions
    if xc.vsigma is not None:
        other = 1 - spin
        same = 2 * xc.vsigma[2 * spin]
        cross = xc.vsigma[1]
        combination = weights * (same * gradients[spin] + cross * gradients[other])
        scaled = scaled + xp.einsum("xg,gxu->gu", combination, values[:, 1:4, :])
    potential = xp.linalg.matrix_transpose(functions) @ scaled
    potential = potential + xp.linalg.matrix_transpose(potential)
    if xc.vtau is not None:
        derivatives = values[:, 1:4, :]
        weighted = (0.5 * weights * xc.vtau[spin])[:, None, None] * derivatives
        potential = potential + xp.einsum("gxu,gxv->uv", weighted, derivatives)
    return potential


def _structured(structure: BlockStructure, alpha: Array, beta: Array) -> BlockMatrix:
    """Place the alpha and beta potentials into the stored blocks of ``structure``."""
    xp = array_namespace(alpha, beta)
    sectors = structure.irreps
    by_spin = {Spin.ALPHA: alpha, Spin.BETA: beta}
    projected = (
        None
        if sectors is None
        else {
            spin: sectors.from_atomic(matrix, spinor=False)
            for spin, matrix in by_spin.items()
        }
    )
    blocks: list[Array] = []
    for block in structure.stored_blocks:
        if projected is None:
            matrix = by_spin[block.spin]
        else:
            assert block.irrep is not None
            matrix = projected[block.spin][block.irrep]
        blocks.append(cast_to_ring(xp, matrix, structure.storage_ring(block)))
    return BlockMatrix(structure, tuple(blocks))

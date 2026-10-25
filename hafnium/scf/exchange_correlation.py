"""The exchange–correlation term of a Kohn–Sham Hamiltonian (M7, ADR-0026, ADR-0027).

The densities of a structured density matrix are evaluated on a grid from
the basis functions and their derivatives, the functional is evaluated
point by point, and the potential is assembled back into the structure of
the density. A restricted or unrestricted class, real or complex, gives
the alpha and beta densities and the collinear functional, so that one
implementation serves RKS and UKS; a generalized (spinor) class gives the
density and the magnetization ``m`` of the spin-density matrix
``ρ_στ(r)``, ``m = (2 Re ρ_αβ, −2 Im ρ_αβ, ρ_αα − ρ_ββ)``, and the
multi-collinear form of the functional (ADR-0029), whose potential enters
the spinor Fock matrix as ``v_ρ + v_m·σ`` with the Pauli matrices.

For a cell the densities come from the k-point density matrices through
the Bloch sums of the basis functions over the lattice images,
``ρ(r) = Σ_k w_k Σ_uv D[v, u](k) conj(φ_u^k(r)) φ_v^k(r)`` with the index
convention of the Coulomb matrix (:mod:`hafnium.scf.coulomb_exchange`),
integrated over one cell on the grid of ADR-0027, and the potential
``V[u, v](k) = ∫ conj(φ_u^k) v_xc φ_v^k`` at every k-point of the full set
is projected onto the stored points of the structure as the Coulomb matrix
is. A molecule is the case of one k-point with the real basis functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hafnium.backend import Array, adjoint, array_namespace, host_array
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.grid import GridSpec, MolecularGrid, build_grid
from hafnium.integrals import IntegralEngine, ShellTable
from hafnium.integrals.periodic import (
    function_radius,
    reaching_images,
    require_kpoints_for_lattice,
)
from hafnium.molecule import Cell, Lattice, Molecule
from hafnium.structured import BlockMatrix, BlockStructure, Spin
from hafnium.structured.dense import reduce_k
from hafnium.structured.validation import cast_to_ring
from hafnium.symmetry import KPoints
from hafnium.xc import (
    DensityOnGrid,
    Functional,
    NoncollinearDensity,
    NoncollinearOptions,
    NoncollinearValues,
    XcValues,
    evaluate_functional,
    evaluate_noncollinear,
)

logger = logging.getLogger(__name__)

DEFAULT_BASIS_CUTOFF = 1e-12
"""Basis function values below this are left out of the densities on the grid."""

DEFAULT_BATCH_SIZE = 8192
"""Grid points evaluated at a time, which bounds the memory of the basis
values; for a cell the points times the k-points."""


@dataclass(frozen=True, slots=True, eq=False)
class ExchangeCorrelation:
    """A functional on a grid, for the densities of one basis.

    Parameters
    ----------
    functional
        The functional; its semilocal components are evaluated here, its
        exact exchange by the Hamiltonian.
    grid
        The quadrature grid, of the molecule or of one cell.
    engine
        Evaluates the basis functions on the grid.
    table
        The shells of the basis on the atoms.
    radii : Array[shell]
        Cutoff radius of every shell for the basis values.
    batch_size
        Grid points evaluated at a time; for a cell the points times the
        k-points.
    noncollinear
        Controls of the multi-collinear evaluation for the generalized
        classes (ADR-0029).
    kpoints, lattice, translations
        For a cell: the k-point set of the densities, the lattice, and the
        lattice vectors ``[image, 3]`` over which the Bloch sums of the
        basis functions run on the grid; ``None`` for a molecule.
    """

    functional: Functional
    grid: MolecularGrid
    engine: IntegralEngine
    table: ShellTable
    radii: Array
    batch_size: int = DEFAULT_BATCH_SIZE
    noncollinear: NoncollinearOptions = field(
        default_factory=NoncollinearOptions, kw_only=True
    )
    kpoints: KPoints | None = field(default=None, kw_only=True)
    lattice: Lattice | None = field(default=None, kw_only=True)
    translations: Array | None = field(default=None, kw_only=True)

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
        periodic = (self.kpoints, self.lattice, self.translations)
        if any(item is None for item in periodic) and any(
            item is not None for item in periodic
        ):
            raise InputError(
                "a cell's term carries its k-points, lattice and translations "
                "together; a molecule's none of them"
            )
        if self.kpoints is not None and self.lattice is not None:
            require_kpoints_for_lattice(self.kpoints, self.lattice)
        if self.translations is not None and (
            self.translations.ndim != 2
            or self.translations.shape[1] != 3
            or self.translations.shape[0] < 1
        ):
            raise InputError("translations have shape [image, 3] with an image")

    @classmethod
    def from_molecule(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        molecule: Molecule,
        functional: Functional,
        *,
        grid: GridSpec | MolecularGrid | None = None,
        basis_cutoff: float = DEFAULT_BASIS_CUTOFF,
    ) -> ExchangeCorrelation:
        """Build the grid of ``molecule`` and lay ``basis`` out on its atoms.

        ``grid`` is a :class:`GridSpec`, or a grid built beforehand;
        ``basis_cutoff`` bounds the basis function values neglected far
        from their centres. The batch size is the default,
        :func:`dataclasses.replace` changes it.
        """
        table = ShellTable.from_basis(basis, molecule)
        radii = _cutoff_radii(basis, basis_cutoff)
        if not isinstance(grid, MolecularGrid):
            grid = build_grid(molecule, grid)
        return cls(functional, grid, engine, table, radii)

    @classmethod
    def from_cell(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        cell: Cell,
        kpoints: KPoints,
        functional: Functional,
        *,
        grid: GridSpec | MolecularGrid | None = None,
        basis_cutoff: float = DEFAULT_BASIS_CUTOFF,
    ) -> ExchangeCorrelation:
        """Build the grid of ``cell`` and the image list of its Bloch sums.

        The densities are those of the k-point density matrices over
        ``kpoints``; ``grid`` is a :class:`GridSpec`, or a grid built
        beforehand. The Bloch sums on the grid run over every lattice image
        of a shell within its cutoff radius of a grid point, the radii from
        ``basis_cutoff`` as for a molecule. The batch size is the default,
        :func:`dataclasses.replace` changes it.
        """
        require_kpoints_for_lattice(kpoints, cell.lattice)
        table = ShellTable.from_basis(basis, cell)
        radii = _cutoff_radii(basis, basis_cutoff)
        if not isinstance(grid, MolecularGrid):
            grid = build_grid(cell, grid)
        translations = _bloch_translations(cell.lattice, grid, table, radii)
        return cls(
            functional,
            grid,
            engine,
            table,
            radii,
            kpoints=kpoints,
            lattice=cell.lattice,
            translations=translations,
        )

    @property
    def n_ao(self) -> int:
        """Number of atomic orbitals."""
        return self.table.n_functions

    @property
    def is_periodic(self) -> bool:
        """Whether the term serves a cell."""
        return self.kpoints is not None

    def evaluate(self, density: BlockMatrix) -> tuple[float, BlockMatrix]:
        """Exchange–correlation energy and potential of a Hermitian ``density``.

        The potential is in the structure of the density; for a cell both
        are per cell. A spinor structure takes the multi-collinear path.
        The number of electrons found on the grid is logged at DEBUG level.

        Raises
        ------
        InputError
            For a density of another basis, or over other k-points than
            the term's.
        """
        structure = density.structure
        self._check_density(density)
        if structure.is_spinor:
            return self._evaluate_spinor(density)
        spins, k_weights = self._spin_densities(density)
        xp = array_namespace(*spins)
        potentials = [xp.zeros_like(matrix) for matrix in spins]
        if not self.functional.components:
            # Exact exchange alone: nothing to integrate.
            return 0.0, self._structured(structure, potentials)
        order = 1 if self.functional.needs_gradient or self.functional.needs_tau else 0
        phases = self._phases()
        energy = 0.0
        electrons = 0.0
        for batch in self.grid.batches(self._points_per_batch()):
            values = xp.asarray(self._values(batch, order, phases))
            weights = self.grid.weights[batch]
            batch_density = _densities(values, spins, k_weights, self.functional)
            xc = evaluate_functional(self.functional, batch_density.on_grid)
            total = batch_density.on_grid.rho[0] + batch_density.on_grid.rho[1]
            energy += float(xp.sum(weights * total * xc.energy_per_particle))
            electrons += float(xp.sum(weights * total))
            for spin in range(len(spins)):
                potentials[spin] = potentials[spin] + _potential(
                    values, weights, batch_density.gradients, xc, spin
                )
        logger.debug(
            "exchange-correlation: %.10f Hartree, %.8f electrons on %d points",
            energy,
            electrons,
            self.grid.n_points,
        )
        return energy, self._structured(structure, potentials)

    def _evaluate_spinor(self, density: BlockMatrix) -> tuple[float, BlockMatrix]:
        """Evaluate a generalized class through the multi-collinear form (ADR-0029)."""
        structure = density.structure
        (matrix,) = density.spin_blocks()
        xp = array_namespace(matrix)
        matrix = xp.astype(matrix, xp.complex128)
        if self.kpoints is None:
            matrix = xp.expand_dims(matrix, 0)
            k_weights = xp.ones((1,), dtype=xp.float64)
        else:
            k_weights = self.kpoints.weights_array(xp)
        potential = xp.zeros_like(matrix)
        if not self.functional.components:
            return 0.0, self._structured(structure, [potential])
        order = 1 if self.functional.needs_gradient or self.functional.needs_tau else 0
        phases = self._phases()
        energy = 0.0
        electrons = 0.0
        for batch in self.grid.batches(self._points_per_batch()):
            values = xp.asarray(self._values(batch, order, phases))
            weights = self.grid.weights[batch]
            on_grid = _spinor_densities(values, matrix, k_weights, self.functional)
            xc = evaluate_noncollinear(
                self.functional, on_grid, self.noncollinear, weights=weights
            )
            energy += float(xp.sum(weights * xc.energy_density))
            electrons += float(xp.sum(weights * on_grid.rho))
            potential = potential + _spinor_potential(values, weights, xc)
        logger.debug(
            "exchange-correlation (noncollinear): %.10f Hartree, %.8f electrons on "
            "%d points",
            energy,
            electrons,
            self.grid.n_points,
        )
        return energy, self._structured(structure, [potential])

    def _points_per_batch(self) -> int:
        n_kpt = 1 if self.kpoints is None else self.kpoints.n_kpt
        return max(1, self.batch_size // n_kpt)

    def _phases(self) -> Array | None:
        """``exp(i k·T)`` as ``[kpt, image]`` for a cell, ``None`` for a molecule."""
        if self.kpoints is None or self.lattice is None or self.translations is None:
            return None
        xp = array_namespace(self.translations)
        kpts = xp.asarray(self.kpoints.cartesian(self.lattice))
        angles = kpts @ xp.linalg.matrix_transpose(self.translations)
        return xp.exp(1j * xp.astype(angles, xp.complex128))

    def _values(self, batch: slice, order: int, phases: Array | None) -> Array:
        """Basis values of a batch as ``[kpt, g, component, ao]``.

        A molecule's values carry one k-point.
        """
        points = self.grid.points[batch]
        if phases is None or self.translations is None:
            values = self.engine.basis_values(
                self.table, points, self.radii, order=order
            )
            return array_namespace(values).expand_dims(values, 0)
        return self.engine.bloch_basis_values(
            self.table, points, self.radii, self.translations, phases, order=order
        )

    def _check_density(self, density: BlockMatrix) -> None:
        """Require a density of this term's basis, system and k-points."""
        structure = density.structure
        axis = structure.k_axis
        if (axis is None) != (self.kpoints is None):
            raise InputError(
                "the exchange–correlation term serves "
                + ("a molecule" if self.kpoints is None else "a cell")
                + ", the density is "
                + ("periodic" if axis is not None else "molecular")
            )
        if (
            axis is not None
            and self.kpoints is not None
            and axis.n_kpt != self.kpoints.n_kpt
        ):
            raise InputError(
                f"the density runs over {axis.n_kpt} k-points, the "
                f"exchange–correlation term over {self.kpoints.n_kpt}"
            )
        if structure.is_spatial:
            raise InputError(
                "the exchange–correlation term reads a density with spin: a "
                "restricted, unrestricted or generalized structure"
            )
        if density.n_spatial != self.n_ao:
            raise InputError(
                f"the density has {density.n_spatial} atomic orbitals, the grid's "
                f"basis {self.n_ao}"
            )

    def _spin_densities(self, density: BlockMatrix) -> tuple[tuple[Array, ...], Array]:
        """Return the spin density matrices ``[kpt, ao, ao]`` and the k-point weights.

        One matrix for a restricted structure, whose beta block repeats the
        alpha block, two for an unrestricted one; real over the one k-point
        of a molecule, complex over the full k set of a cell.
        """
        structure = density.structure
        blocks = density.spin_blocks()
        xp = array_namespace(*blocks)
        if _is_restricted(structure):
            blocks = blocks[:1]
        if self.kpoints is None:
            # The basis functions are real, so only the real part of a
            # Hermitian density enters the densities on the grid.
            spins = tuple(
                xp.expand_dims(xp.astype(xp.real(block), xp.float64), 0)
                for block in blocks
            )
            return spins, xp.ones((1,), dtype=xp.float64)
        spins = tuple(xp.astype(block, xp.complex128) for block in blocks)
        return spins, self.kpoints.weights_array(xp)

    def _structured(
        self, structure: BlockStructure, potentials: list[Array]
    ) -> BlockMatrix:
        """Place the potentials ``[kpt, ao, ao]`` into the stored blocks.

        The alpha and beta potentials of a collinear structure, a single
        one serving a restricted structure, whose stored blocks are all
        alpha; the spinor potential ``[kpt, 2n, 2n]`` of a generalized one.
        Over a k axis each stored block is the reduction of the full set
        onto its stored points, as for the Coulomb matrix; with irrep
        sectors the matrices are projected onto the sectors.
        """
        xp = array_namespace(*potentials)
        if structure.is_spinor:
            by_spin = {Spin.SPINOR: potentials[0]}
        else:
            by_spin = {Spin.ALPHA: potentials[0], Spin.BETA: potentials[-1]}
        sectors = structure.irreps
        projected = None
        if structure.k_axis is None:
            by_spin = {spin: matrix[0] for spin, matrix in by_spin.items()}
            if sectors is not None:
                projected = {
                    spin: sectors.from_atomic(matrix, spinor=structure.is_spinor)
                    for spin, matrix in by_spin.items()
                }
        blocks: list[Array] = []
        for block in structure.stored_blocks:
            if structure.k_axis is not None:
                matrix = reduce_k(
                    structure, block, by_spin[block.spin], check=False, tolerance=0.0
                )
            elif projected is None:
                matrix = by_spin[block.spin]
            else:
                assert block.irrep is not None
                matrix = projected[block.spin][block.irrep]
            blocks.append(cast_to_ring(xp, matrix, structure.storage_ring(block)))
        return BlockMatrix(structure, tuple(blocks))


def _cutoff_radii(basis: BasisSet, basis_cutoff: float) -> Array:
    return host_array([function_radius(shell, basis_cutoff) for shell in basis.shells])


def _bloch_translations(
    lattice: Lattice, grid: MolecularGrid, table: ShellTable, radii: Array
) -> Array:
    """Lattice vectors ``[image, 3]`` at which some shell's image may reach some point.

    A shell centred at ``C`` with cutoff radius ``R`` contributes to the
    Bloch sum at a point ``r`` through the image ``T`` when
    ``|r − C − T| ≤ R``. Every point lies within the extent of its atom's
    grid of that atom, so the images with ``|A + T − C| ≤ R + extent_A`` for
    some atom ``A`` cover every such pair, a few more than the points
    themselves demand; the kernel screens each point and image by the
    radii. The zero vector comes first.
    """
    xp = array_namespace(grid.points)
    centers = xp.asarray(table.centers)
    owners = xp.asarray(grid.atoms)
    n_atoms = int(xp.max(owners)) + 1 if int(owners.shape[0]) > 0 else 0
    anchors: list[Array] = []
    extents: list[float] = []
    for atom in range(n_atoms):
        points = grid.points[owners == atom]
        if int(points.shape[0]) == 0:
            continue
        anchor = xp.mean(points, axis=0)
        anchors.append(anchor)
        extents.append(float(xp.max(xp.linalg.vector_norm(points - anchor, axis=-1))))
    reach = xp.expand_dims(xp.asarray(radii), 1) + xp.asarray(extents)[None, :]
    # reaching_images keeps T with |C − a − T| ≤ reach, the image −T of the shell.
    return -reaching_images(lattice, centers, xp.stack(anchors), reach)


def _is_restricted(structure: BlockStructure) -> bool:
    """Whether the beta blocks of ``structure`` are derived from the alpha blocks."""
    return all(block.spin is not Spin.BETA for block in structure.stored_blocks)


@dataclass(frozen=True, slots=True)
class _BatchDensities:
    """The densities of a batch with the spin gradients the potential needs."""

    on_grid: DensityOnGrid
    gradients: tuple[Array, ...]


def _densities(
    values: Array,
    densities: tuple[Array, ...],
    k_weights: Array,
    functional: Functional,
) -> _BatchDensities:
    """Return the spin densities, gradients and kinetic energy densities of a batch.

    ``values`` is ``[kpt, g, component, ao]`` and each density matrix
    ``[kpt, ao, ao]``, Hermitian. With ``M = conj(Φ D)`` at every k-point,
    ``ρ_σ = Σ_k w_k Re Σ_u Φ_u M_u``, ``∇ρ_σ = 2 Σ_k w_k Re Σ_u ∇Φ_u M_u`` and
    ``τ_σ = ½ Σ_k w_k Re Σ_u ∇Φ_u · conj(∇Φ D)_u``, which for the real
    functions of a molecule are ``Σ_uv χ_u D_uv χ_v``, ``2 Σ_uv ∇χ_u D_uv χ_v``
    and ``½ Σ_uv ∇χ_u · D_uv ∇χ_v``. A single density matrix stands for both
    spins, and its quantities are repeated for the beta spin.
    """
    xp = array_namespace(values)
    functions = values[:, :, 0, :]
    derivatives = values[:, :, 1:4, :] if values.shape[2] > 1 else None
    rho: list[Array] = []
    gradients: list[Array] = []
    taus: list[Array] = []
    for matrix in densities:
        half = xp.conj(functions @ matrix)
        rho.append(xp.real(xp.einsum("k,kgu,kgu->g", k_weights, functions, half)))
        if derivatives is None:
            continue
        gradients.append(
            2.0 * xp.real(xp.einsum("k,kgxu,kgu->xg", k_weights, derivatives, half))
        )
        if functional.needs_tau:
            moved = xp.conj(derivatives @ xp.expand_dims(matrix, 1))
            taus.append(
                0.5
                * xp.real(xp.einsum("k,kgxu,kgxu->g", k_weights, derivatives, moved))
            )
    if len(densities) == 1:
        rho, gradients, taus = 2 * rho, 2 * gradients, 2 * taus
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
    """Return the potential matrices ``[kpt, ao, ao]`` of one spin from a batch.

    With ``A_σ = ½ w v_ρσ Φ + Σ_x w (2 v_σσ ∂_xρ_σ + v_αβ ∂_xρ_σ') ∂_xΦ`` the
    matrix at every k-point is ``Φ† A_σ + A_σ† Φ``, plus
    ``½ Σ_x w v_τσ ∂_xΦ† ∂_xΦ`` for a meta-GGA, with ``Φ[g, ao]`` the
    function values and ``∂_xΦ`` their derivatives.
    """
    xp = array_namespace(values)
    functions = values[:, :, 0, :]
    scaled = (0.5 * (weights * xc.vrho[spin]))[None, :, None] * functions
    if xc.vsigma is not None:
        other = 1 - spin
        same = 2 * xc.vsigma[2 * spin]
        cross = xc.vsigma[1]
        combination = weights * (same * gradients[spin] + cross * gradients[other])
        scaled = scaled + xp.einsum("xg,kgxu->kgu", combination, values[:, :, 1:4, :])
    potential = adjoint(functions) @ scaled
    potential = potential + adjoint(potential)
    if xc.vtau is not None:
        derivatives = values[:, :, 1:4, :]
        weighted = (0.5 * weights * xc.vtau[spin])[None, :, None, None] * derivatives
        potential = potential + xp.einsum(
            "kgxu,kgxv->kuv", xp.conj(weighted), derivatives
        )
    return potential


def _spinor_densities(
    values: Array, matrix: Array, k_weights: Array, functional: Functional
) -> NoncollinearDensity:
    """Density and magnetization of a batch from a spinor density matrix.

    ``values`` is ``[kpt, g, component, ao]`` and ``matrix`` ``[kpt, 2n, 2n]``
    with the alpha components first. The spin-density matrix on the grid is
    ``ρ_στ = Σ_k w_k Σ_uv φ_v^k D[vσ, uτ](k) conj(φ_u^k)``: with
    ``T_στ = Φ D_στ`` for every component, ``ρ_στ = Σ_u T_στ[0, u] conj(Φ_u)``,
    ``∂_i ρ_στ = C_στ,i + conj(C_τσ,i)`` for ``C_στ,i = Σ_u T_στ[i, u] conj(Φ_u)``
    and ``τ_στ = ½ Σ_i Σ_u T_στ[i, u] conj(∂_iΦ_u)``. The density is
    ``ρ_αα + ρ_ββ`` and the magnetization ``(2 Re ρ_αβ, −2 Im ρ_αβ, ρ_αα − ρ_ββ)``,
    likewise for the gradients and the kinetic energy densities.
    """
    xp = array_namespace(values)
    n = int(matrix.shape[-1]) // 2
    conj_functions = xp.conj(values[:, :, 0, :])
    derivatives = values[:, :, 1:4, :] if values.shape[2] > 1 else None
    values_of: dict[tuple[int, int], Array] = {}
    fluxes: dict[tuple[int, int], Array] = {}
    kinetic: dict[tuple[int, int], Array] = {}
    for pair in ((0, 0), (1, 1), (0, 1), (1, 0)):
        rows, columns = (slice(s * n, (s + 1) * n) for s in pair)
        moved = values @ xp.expand_dims(matrix[:, rows, columns], 1)
        values_of[pair] = xp.einsum(
            "k,kgu,kgu->g", k_weights, moved[:, :, 0, :], conj_functions
        )
        if derivatives is None:
            continue
        fluxes[pair] = xp.einsum(
            "k,kgxu,kgu->xg", k_weights, moved[:, :, 1:4, :], conj_functions
        )
        if functional.needs_tau:
            kinetic[pair] = 0.5 * xp.einsum(
                "k,kgxu,kgxu->g", k_weights, moved[:, :, 1:4, :], xp.conj(derivatives)
            )
    rho, m = _total_and_magnetization(values_of)
    grad_rho = grad_m = tau = tau_m = None
    if fluxes:
        gradients = {(s, t): fluxes[s, t] + xp.conj(fluxes[t, s]) for s, t in fluxes}
        grad_rho, grad_m = _total_and_magnetization(gradients)
    if kinetic:
        tau, tau_m = _total_and_magnetization(kinetic)
    return NoncollinearDensity(rho, m, grad_rho, grad_m, tau, tau_m)


def _total_and_magnetization(
    pairs: dict[tuple[int, int], Array],
) -> tuple[Array, Array]:
    """Return ``ρ_αα + ρ_ββ`` and ``(2 Re ρ_αβ, −2 Im ρ_αβ, ρ_αα − ρ_ββ)``.

    ``pairs`` holds a value per spin pair ``(σ, τ)``.
    """
    xp = array_namespace(*pairs.values())
    total = xp.real(pairs[0, 0] + pairs[1, 1])
    return total, xp.stack(
        [
            2.0 * xp.real(pairs[0, 1]),
            -2.0 * xp.imag(pairs[0, 1]),
            xp.real(pairs[0, 0] - pairs[1, 1]),
        ]
    )


def _spinor_potential(values: Array, weights: Array, xc: NoncollinearValues) -> Array:
    """Return the potential ``[kpt, 2n, 2n]`` of a batch from the derivatives.

    The derivative of the energy with respect to ``D[vτ, uσ]`` is
    ``∫ conj(φ_u) (v_ρ δ_στ + v_m·σ_στ) φ_v`` with the Pauli matrices, plus
    the same combinations of the gradient derivatives on
    ``∂_i(conj(φ_u) φ_v)`` and of the kinetic ones on
    ``½ conj(∂_iφ_u) ∂_iφ_v``: the alpha–alpha block carries ``v + v_z``,
    the beta–beta block ``v − v_z``, the alpha–beta block ``v_x − i v_y``
    and the beta–alpha block its adjoint.
    """
    xp = array_namespace(values)

    def pauli(total: Array | None, spin: Array | None) -> tuple[Array, Array, Array]:
        assert total is not None
        assert spin is not None
        complex_spin = xp.astype(spin, xp.complex128)
        return (
            total + spin[2],
            total - spin[2],
            complex_spin[0] - 1j * complex_spin[1],
        )

    scalar = pauli(xc.v_rho, xc.v_m)
    gradient = None if xc.v_grad_rho is None else pauli(xc.v_grad_rho, xc.v_grad_m)
    kinetic = None if xc.v_tau is None else pauli(xc.v_tau, xc.v_tau_m)
    blocks = [
        _block_potential(
            values,
            weights,
            scalar[index],
            None if gradient is None else gradient[index],
            None if kinetic is None else kinetic[index],
        )
        for index in range(3)
    ]
    alpha_alpha, beta_beta, alpha_beta = blocks
    top = xp.concat([alpha_alpha, alpha_beta], axis=-1)
    bottom = xp.concat([adjoint(alpha_beta), beta_beta], axis=-1)
    return xp.concat([top, bottom], axis=-2)


def _block_potential(
    values: Array,
    weights: Array,
    scalar: Array,
    gradient: Array | None,
    kinetic: Array | None,
) -> Array:
    """Return ``Σ_g w [p conj(φ_u) φ_v + q·∂(conj(φ_u) φ_v) + ½ r ∂conj(φ_u)·∂φ_v]``.

    One matrix per k-point.

    ``scalar`` is ``p[g]``, ``gradient`` ``q[3, g]`` and ``kinetic`` ``r[g]``,
    real or complex.
    """
    xp = array_namespace(values)
    functions = values[:, :, 0, :]
    scaled = (weights * scalar)[None, :, None] * functions
    if gradient is not None:
        derivatives = values[:, :, 1:4, :]
        flux = weights * gradient
        scaled = scaled + xp.einsum("xg,kgxu->kgu", flux, derivatives)
        back = xp.einsum("xg,kgxu->kgu", flux, xp.conj(derivatives))
        potential = adjoint(functions) @ scaled
        potential = potential + xp.linalg.matrix_transpose(back) @ functions
    else:
        potential = adjoint(functions) @ scaled
    if kinetic is not None:
        derivatives = values[:, :, 1:4, :]
        potential = potential + 0.5 * xp.einsum(
            "g,kgxu,kgxv->kuv", weights * kinetic, xp.conj(derivatives), derivatives
        )
    return potential

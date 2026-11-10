"""The Hartree–Fock state as a differentiable function of the positions (ADR-0038).

The self-consistent field converges on the host as it always has; what makes
its solution differentiable is the stationarity condition it satisfies.
Under JAX :meth:`DifferentiableScf.solve` is a host callback with a custom
vector-Jacobian product in which the converged orbitals follow the
positions in two ways: through a re-orthonormalization in the moving
overlap metric, the connection that carries the Pulay term of the gradient,
and through the occupied–virtual rotations that keep the orbital gradient
zero, the response of the coupled-perturbed equations, which the backward
pass solves by conjugate gradients with the Hessian–vector products of the
Fock build. A function of the state that is stationary, the energy, needs no
response and gets none beyond the convergence of the field; any other
function of the state, an orbital energy or an expectation value, gets the
full response. For positions of any other array library the same methods run
the host solver and return its state as arrays of that library: the state
lives in the namespace of the positions it is solved at, as every array
derived from an input does (CONVENTIONS.md §5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

from hafnium.backend import (
    Array,
    Namespace,
    adjoint,
    array_namespace,
    host_array,
    host_namespace,
    is_jax_array,
    is_traced,
    to_host,
    use_backend,
)
from hafnium.basis import BasisSet
from hafnium.errors import BackendError, InputError
from hafnium.integrals import DifferentiableIntegrals, IntegralEngine, nuclear_repulsion
from hafnium.molecule import Molecule
from hafnium.scf.atoms import atomic_density_guess
from hafnium.scf.coulomb_exchange import DensityFittedCoulombExchange
from hafnium.scf.driver import ScfOptions, ScfResult, solve_scf
from hafnium.scf.hamiltonian import Hamiltonian
from hafnium.structured import BlockMatrix, BlockStructure, BlockVector, density_matrix
from hafnium.symmetry import SymmetryAdaptation, SymmetrySpec


@dataclass(frozen=True, slots=True)
class ResponseOptions:
    """Controls of the conjugate-gradient solve of the response equations.

    Parameters
    ----------
    tolerance
        The residual norm at which the equations count as solved, relative
        to the right-hand side and absolute.
    iterations
        Iterations allowed before :class:`~hafnium.errors.ConvergenceError`.
    """

    tolerance: float = 1e-10
    iterations: int = 100

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0:
            raise InputError("the response tolerance is positive")
        if self.iterations < 1:
            raise InputError("at least one response iteration is needed")


class FittedIntegrals(NamedTuple):
    """What a molecular Hamiltonian is built from at one geometry.

    ``overlap`` and ``core`` are ``[ao, ao]`` in the atomic basis; ``fitted``
    ``[aux, ao, ao]`` is the three-centre integrals with the Cholesky factor
    of the metric applied, ``L⁻¹ (P|uv)`` for ``(P|Q) = L L†``, so that
    their products reproduce the fitted four-centre integrals;
    ``nuclear_repulsion`` is the repulsion of the nuclei as a 0-d array,
    which depends on the positions as the integrals do.
    """

    overlap: Array
    core: Array
    fitted: Array
    nuclear_repulsion: Array


@dataclass(frozen=True, slots=True, eq=False)
class ScfSolution:
    """A mean-field state as arrays of the library of the positions it was solved at.

    The state :meth:`DifferentiableScf.solve` returns: for JAX positions
    every field is differentiable with respect to them. The
    orbital coefficients are taken in the gauge of the solver, so a function
    of them that is not invariant under rotations within the occupied and
    within the virtual orbitals differentiates in that gauge; the density,
    the energy and the orbital energies are gauge-invariant.

    Parameters
    ----------
    mo_coefficients : BlockMatrix
        Orbital coefficients ``[ao, mo]`` per block.
    mo_occupations : BlockVector
        Occupation numbers, constants of the state.
    mo_energies : BlockVector
        The diagonal of the Fock matrix in the orbitals, the orbital
        energies of a converged state.
    density : BlockMatrix
        The density of the occupied orbitals.
    fock : BlockMatrix
        The Fock matrix of that density.
    energy : Array
        The total energy in Hartree as a 0-d array.
    """

    mo_coefficients: BlockMatrix
    mo_occupations: BlockVector
    mo_energies: BlockVector
    density: BlockMatrix
    fock: BlockMatrix
    energy: Array


@dataclass(frozen=True, slots=True, eq=False)
class DifferentiableScf:
    """The density-fitted Hartree–Fock state of a molecule as a function of positions.

    Built once for a molecule, a basis and a Fukutome class with
    :meth:`from_molecule`, and solved at any positions ``[atom, 3]`` with
    :meth:`solve`. The occupations are those of the reference solve at the
    molecule's own positions and stay fixed, as the aufbau filling of a
    state away from a level crossing does.

    Parameters
    ----------
    integrals
        The integrals as functions of the shell centres.
    charges : Array[atom]
        The nuclear charges, a host array.
    symmetry
        The Fukutome class, with the point group it may impose.
    structure
        The block structure of the state.
    n_alpha, n_beta
        Electron counts.
    adaptation
        The symmetry adaptation of the basis when ``symmetry`` names a
        point group; ``None`` otherwise.
    occupations
        The occupations of the reference solve, a host block vector in
        which every block fills its first orbitals with one occupation.
    guess
        The atomic density guess every solve starts from, a host block
        matrix.
    options
        The convergence controls of the field.
    response
        Controls of the solve of the response equations.

    Attributes
    ----------
    occupied : tuple[int, ...]
        The number of occupied orbitals in every block, counted from
        ``occupations``.

    Raises
    ------
    InputError
        If the charges are not one per atom, if the occupations belong to
        another structure, or if a block has fractional occupations.
    """

    integrals: DifferentiableIntegrals
    charges: Array
    symmetry: SymmetrySpec
    structure: BlockStructure
    n_alpha: int
    n_beta: int
    adaptation: SymmetryAdaptation | None
    occupations: BlockVector
    guess: BlockMatrix
    options: ScfOptions
    response: ResponseOptions = field(default_factory=ResponseOptions)
    occupied: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "charges", host_array(self.charges))
        if self.charges.ndim != 1:
            raise InputError("the charges are one per atom")
        if self.occupations.structure != self.structure:
            raise InputError("the occupations belong to another structure")
        object.__setattr__(self, "occupied", _occupied_counts(self.occupations))

    @classmethod
    def from_molecule(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        molecule: Molecule,
        auxiliary: BasisSet,
        *,
        symmetry: SymmetrySpec | None = None,
        options: ScfOptions | None = None,
        response: ResponseOptions | None = None,
    ) -> DifferentiableScf:
        """Lay out ``basis`` and ``auxiliary`` on ``molecule`` and solve the reference.

        The reference solve at the molecule's positions fixes the
        occupations and checks that the basis is retained in full.

        Raises
        ------
        InputError
            If ``symmetry`` carries k-points or a space group, if the
            reference state has fractional occupations, or if the overlap
            threshold removed functions from the basis.
        SymmetryError
            If ``symmetry`` names a point group the molecule does not have
            in its frame.
        """
        symmetry = SymmetrySpec() if symmetry is None else symmetry
        if symmetry.kpoints is not None or symmetry.space_group is not None:
            raise InputError(
                "a molecule has no k-points or space group; drop them from the symmetry"
            )
        options = ScfOptions() if options is None else options
        integrals = DifferentiableIntegrals.from_basis(
            engine, basis, molecule, auxiliary
        )
        adaptation = (
            None
            if symmetry.point_group is None
            else SymmetryAdaptation.from_basis(basis, molecule, symmetry.point_group)
        )
        charges = host_array(molecule.nuclear_charges)
        with use_backend("numpy"):
            structure = symmetry.structure(adaptation)
            guess = atomic_density_guess(engine, basis, structure, options=options)
            hamiltonian = _hamiltonian(
                _fitted_integrals(integrals, charges, molecule.positions_array()),
                molecule.n_alpha,
                molecule.n_beta,
                adaptation,
            )
            reference = solve_scf(hamiltonian, symmetry, guess=guess, options=options)
        scf = cls(
            integrals,
            charges,
            symmetry,
            structure,
            molecule.n_alpha,
            molecule.n_beta,
            adaptation,
            reference.mo_occupations,
            guess,
            options,
            ResponseOptions() if response is None else response,
        )
        scf._check_state(reference)
        return scf

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return int(self.charges.shape[0])

    # ------------------------------------------------------------------ solving

    def solve(self, positions: Array) -> ScfSolution:
        """Converge the state at ``positions`` ``[atom, 3]``, in their array library.

        For JAX positions the orbitals come from the host solver with a
        vector-Jacobian product that differentiates the solution
        implicitly (ADR-0038); for any other library from the host solver
        alone. The library of ``positions`` decides, not the active
        backend: host positions give a host state under any backend.

        Raises
        ------
        InputError
            If ``positions`` are not ``[atom, 3]``, or the state's
            occupations differ from the reference's.
        BackendError
            If ``positions`` are JAX arrays that are not 64-bit.
        SymmetryError
            If ``positions`` break the point group of the state.
        ConvergenceError
            If the field does not converge, or under differentiation the
            response equations do not. Under ``jit`` and ``vmap`` JAX runs
            the host solver as a callback and reports whatever it raises
            as its own runtime error, which quotes the original.
        """
        positions = self._checked(positions)
        if is_jax_array(positions):
            coefficients: BlockMatrix = _rules().solve(self, positions)
        else:
            xp = array_namespace(positions)
            coefficients = self._solve_on_host(positions).mo_coefficients.map_blocks(
                lambda block: xp.asarray(block)
            )
        return self.complete(coefficients, positions)

    def energy(self, positions: Array) -> Array:
        """Return the total energy at ``positions`` as a 0-d array."""
        return self.solve(positions).energy

    def host_solve(self, positions: Array) -> ScfResult:
        """Converge the state at ``positions`` on the host, from the atomic guess.

        Raises
        ------
        InputError
            If the state's occupations differ from the reference's.
        SymmetryError
            If ``positions`` break the point group of the state.
        ConvergenceError
            If the field does not converge.
        """
        return self._solve_on_host(self._checked(positions))

    def _solve_on_host(self, positions: Array) -> ScfResult:
        with use_backend("numpy"):
            result = solve_scf(
                self.hamiltonian(to_host(positions)),
                self.symmetry,
                guess=self.guess,
                options=self.options,
            )
        self._check_state(result)
        return result

    def complete(self, coefficients: BlockMatrix, positions: Array) -> ScfSolution:
        """Return the state of ``coefficients`` with the integrals at ``positions``.

        The density, the Fock matrix of the density, the orbital energies as
        its diagonal in the orbitals, and the total energy; each traces
        through the integrals, while the dependence of the coefficients
        themselves on the positions is the solver's.
        """
        xp = array_namespace(positions)
        hamiltonian = self.hamiltonian(positions)
        density = self.density(coefficients)
        fock, terms = hamiltonian.fock(density)
        mo_energies = BlockVector(
            self.structure,
            tuple(
                xp.real(xp.linalg.diagonal(adjoint(c) @ f @ c))
                for c, f in zip(coefficients.blocks, fock.blocks, strict=True)
            ),
        )
        return ScfSolution(
            coefficients,
            self.occupations_in(xp),
            mo_energies,
            density,
            fock,
            xp.asarray(terms.total),
        )

    # -------------------------------------------------------------- integrals

    def fitted_integrals(self, positions: Array) -> FittedIntegrals:
        """Evaluate the integrals at ``positions``, in their namespace."""
        return _fitted_integrals(self.integrals, self.charges, positions)

    def hamiltonian_from(self, integrals: FittedIntegrals) -> Hamiltonian:
        """Build the Hamiltonian of ``integrals``, with their nuclear repulsion."""
        return _hamiltonian(integrals, self.n_alpha, self.n_beta, self.adaptation)

    def hamiltonian(self, positions: Array) -> Hamiltonian:
        """Build the Hamiltonian at ``positions``, in their array library.

        With a point group the symmetry adaptation is that of the reference
        geometry, so the positions must keep the group. Concrete positions
        are checked here; abstract ones under a JAX transformation cannot
        be read, and are checked by the host solve of :meth:`solve` that
        they pass through.

        Raises
        ------
        SymmetryError
            If concrete ``positions`` break the point group of the state.
        """
        if self.adaptation is not None and not is_traced(positions):
            self.adaptation.require_symmetric(positions)
        return self.hamiltonian_from(self.fitted_integrals(positions))

    def overlap_from(self, integrals: FittedIntegrals) -> BlockMatrix:
        """Embed the overlap of ``integrals`` into the structure of the state."""
        return self.hamiltonian_from(integrals).metric(self.structure)

    def occupations_in(self, xp: Namespace) -> BlockVector:
        """Return the occupations as arrays of the namespace ``xp``."""
        return self.occupations.map_blocks(lambda block: xp.asarray(block))

    def density(self, coefficients: BlockMatrix) -> BlockMatrix:
        """Return the density of ``coefficients`` with the occupations of the state.

        The time-reversal check of :func:`~hafnium.structured.density_matrix`
        reads entries, so it runs on concrete coefficients. Abstract ones
        under a JAX transformation are the host solver's, which checked
        its own density, moved along the connection and the rotations.
        """
        xp = array_namespace(*coefficients.blocks)
        traced = any(is_traced(block) for block in coefficients.blocks)
        return density_matrix(coefficients, self.occupations_in(xp), check=not traced)

    # ------------------------------------------------ the stationarity condition

    def connect(self, coefficients: BlockMatrix, overlap: BlockMatrix) -> BlockMatrix:
        """Re-orthonormalize ``coefficients`` in the metric ``overlap``, block by block.

        ``C L^{-†}`` with the Cholesky factor ``C† S C = L L†`` is
        orthonormal in ``S`` and is ``C`` itself where ``C`` already is.
        This is the connection along which the converged orbitals follow
        the positions (ADR-0038): any smooth re-orthonormalization gives the
        same derivative of a stationary energy, and this one has no
        eigenvalue degeneracies to differentiate through.
        """
        blocks: list[Array] = []
        for c, s in zip(coefficients.blocks, overlap.blocks, strict=True):
            xp = array_namespace(c, s)
            factor = xp.linalg.cholesky(adjoint(c) @ s @ c)
            blocks.append(adjoint(xp.linalg.solve(factor, adjoint(c))))
        return BlockMatrix(coefficients.structure, tuple(blocks))

    def rotate(
        self, coefficients: BlockMatrix, rotations: tuple[Array, ...]
    ) -> BlockMatrix:
        """Mix the occupied and virtual orbitals of every block to first order.

        ``rotations[b]`` is ``[virt, occ]``: the occupied orbitals gain
        ``C_v κ`` and the virtual ones lose ``C_o κ†``, the first order of
        the unitary ``exp(K)`` with ``K = [[0, −κ†], [κ, 0]]``, which is all
        a derivative at zero rotation needs.
        """
        blocks: list[Array] = []
        for c, kappa, n in zip(
            coefficients.blocks, rotations, self.occupied, strict=True
        ):
            xp = array_namespace(c, kappa)
            occupied, virtual = c[..., :n], c[..., n:]
            blocks.append(
                xp.concat(
                    [occupied + virtual @ kappa, virtual - occupied @ adjoint(kappa)],
                    axis=-1,
                )
            )
        return BlockMatrix(coefficients.structure, tuple(blocks))

    def zero_rotations(self, coefficients: BlockMatrix) -> tuple[Array, ...]:
        """Return zero rotations ``[virt, occ]`` per block, in the block's dtype."""
        xp = array_namespace(*coefficients.blocks)
        return tuple(
            xp.zeros((int(c.shape[-1]) - n, n), dtype=c.dtype)
            for c, n in zip(coefficients.blocks, self.occupied, strict=True)
        )

    def residual(
        self, coefficients: BlockMatrix, hamiltonian: Hamiltonian
    ) -> tuple[Array, ...]:
        """``C_v† F C_o`` per block, the orbital gradient that vanishes at the solution.

        ``F`` is the Fock matrix of the density of ``coefficients`` with the
        occupations of the state. Up to the occupation numbers the residual
        is the derivative of the energy with respect to the rotations of
        :meth:`rotate`, so its Jacobian in them is the electronic Hessian,
        self-adjoint and positive definite at a stable state.
        """
        fock, _ = hamiltonian.fock(self.density(coefficients))
        return tuple(
            adjoint(c[..., n:]) @ f @ c[..., :n]
            for c, f, n in zip(
                coefficients.blocks, fock.blocks, self.occupied, strict=True
            )
        )

    def gaps(self, mo_energies: BlockVector) -> tuple[Array, ...]:
        """``ε_a − ε_i`` as ``[virt, occ]`` per block, the diagonal of the Hessian."""
        return tuple(
            e[n:, None] - e[None, :n]
            for e, n in zip(mo_energies.blocks, self.occupied, strict=True)
        )

    # ------------------------------------------------------------------ checks

    def _checked(self, positions: Array) -> Array:
        if positions.shape != (self.n_atoms, 3):
            raise InputError(f"the positions are [atom, 3] with {self.n_atoms} atoms")
        if is_jax_array(positions) and positions.dtype != host_namespace().float64:
            raise BackendError(
                f"the positions are JAX arrays of {positions.dtype}, and the solver "
                "works in float64: select the JAX backend, which switches JAX to "
                "64-bit arrays, with use_backend('jax') before creating them"
            )
        return positions

    def _check_state(self, result: ScfResult) -> None:
        xp = host_namespace()
        for block, template in zip(
            result.mo_coefficients.blocks, self.guess.blocks, strict=True
        ):
            if block.shape != template.shape:
                raise InputError(
                    "the overlap threshold removed functions from the basis; the "
                    "differentiable solver needs the basis retained in full"
                )
        for block, reference in zip(
            result.mo_occupations.blocks, self.occupations.blocks, strict=True
        ):
            if not bool(xp.all(to_host(block) == to_host(reference))):
                raise InputError(
                    "the occupations of the state differ from the reference's; "
                    "the differentiable solver holds them fixed"
                )


def _fitted_integrals(
    integrals: DifferentiableIntegrals, charges: Array, positions: Array
) -> FittedIntegrals:
    xp = array_namespace(positions)
    centers = integrals.orbital_centers(positions)
    auxiliary_centers = integrals.auxiliary_centers(positions)
    core = integrals.kinetic(centers) + integrals.nuclear_attraction(
        centers, positions, charges
    )
    metric = integrals.metric(auxiliary_centers)
    three_center = integrals.three_center(auxiliary_centers, centers)
    n_aux = int(metric.shape[0])
    factor = xp.linalg.cholesky(metric)
    fitted = xp.reshape(
        xp.linalg.solve(factor, xp.reshape(three_center, (n_aux, -1))),
        three_center.shape,
    )
    return FittedIntegrals(
        integrals.overlap(centers),
        core,
        fitted,
        nuclear_repulsion(positions, charges),
    )


def _hamiltonian(
    integrals: FittedIntegrals,
    n_alpha: int,
    n_beta: int,
    adaptation: SymmetryAdaptation | None,
) -> Hamiltonian:
    return Hamiltonian(
        integrals.overlap,
        integrals.core,
        DensityFittedCoulombExchange(integrals.fitted),
        integrals.nuclear_repulsion,
        n_alpha,
        n_beta,
        adaptation=adaptation,
    )


def _occupied_counts(occupations: BlockVector) -> tuple[int, ...]:
    """Count the occupied orbitals per block, which fill the first columns equally.

    Raises
    ------
    InputError
        If a block has fractional occupations, which the stationarity
        condition of :meth:`DifferentiableScf.residual` does not describe.
    """
    xp = host_namespace()
    counts: list[int] = []
    for block in occupations.blocks:
        values = to_host(block)
        n = int(xp.sum(values > 0.0))
        equal = n == 0 or bool(xp.all(values[:n] == values[0]))
        if not (equal and bool(xp.all(values[n:] == 0.0))):
            raise InputError(
                "the state has fractional occupations; the differentiable solver "
                "needs every block to fill its lowest orbitals equally"
            )
        counts.append(n)
    return tuple(counts)


def _rules() -> Any:
    """Import the JAX rules when a JAX array arrives, so JAX stays optional."""
    from hafnium.scf import _jax_rules

    return _jax_rules

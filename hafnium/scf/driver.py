"""The self-consistent field driver, one implementation for every class.

Everything here acts on :class:`~hafnium.structured.BlockMatrix` in the
structure of the requested Fukutome class (PLAN.md §5), so the same loop
serves real and complex, restricted, unrestricted and generalized states.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from hafnium.backend import Array, array_namespace, eigh
from hafnium.errors import ConvergenceError, InputError, StructureError
from hafnium.scf.diis import Diis
from hafnium.scf.hamiltonian import EnergyTerms, Hamiltonian
from hafnium.scf.occupation import (
    AufbauOccupation,
    AveragedOccupation,
    OccupationPolicy,
)
from hafnium.structured import BlockMatrix, BlockStructure, BlockVector, density_matrix
from hafnium.symmetry import SymmetrySpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScfOptions:
    """Tolerances and limits of the iteration.

    Parameters
    ----------
    max_iterations
        Fock builds before :class:`~hafnium.errors.ConvergenceError` is raised.
    energy_tolerance
        Largest energy change between iterations at convergence, in Hartree.
    gradient_tolerance
        Largest Frobenius norm of the orbital gradient at convergence (the
        orthonormalized commutator ``[F, D]`` of :func:`orbital_gradient`).
    diis_size
        Fock matrices kept for extrapolation; ``1`` iterates without DIIS.
    """

    max_iterations: int = 100
    energy_tolerance: float = 1e-10
    gradient_tolerance: float = 1e-6
    diis_size: int = 8

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise InputError("at least one iteration is needed")
        if self.energy_tolerance <= 0.0 or self.gradient_tolerance <= 0.0:
            raise InputError("tolerances must be positive")
        if self.diis_size < 1:
            raise InputError("the DIIS subspace holds at least one matrix")


@dataclass(frozen=True, slots=True)
class ScfIteration:
    """One row of the convergence history."""

    index: int
    energy: float
    energy_change: float
    gradient_norm: float


@dataclass(frozen=True, slots=True, eq=False)
class ScfResult:
    """A mean-field state, consistent with its canonical orbitals.

    The density is built from the occupied canonical orbitals of the final
    Fock matrix, and the Fock matrix and energies are those of that density,
    so the last entry of ``history`` differs from ``energies.total`` by at
    most the energy tolerance.
    """

    energies: EnergyTerms
    mo_coefficients: BlockMatrix
    mo_energies: BlockVector
    mo_occupations: BlockVector
    density: BlockMatrix
    fock: BlockMatrix
    converged: bool
    history: tuple[ScfIteration, ...]

    @property
    def energy(self) -> float:
        """Total energy in Hartree."""
        return self.energies.total

    @property
    def structure(self) -> BlockStructure:
        """Block structure of the state."""
        return self.density.structure

    @property
    def n_iterations(self) -> int:
        """Number of Fock builds in the iteration."""
        return len(self.history)


def solve_scf(
    hamiltonian: Hamiltonian,
    symmetry: SymmetrySpec,
    *,
    guess: BlockMatrix | None = None,
    occupation: OccupationPolicy | None = None,
    options: ScfOptions | None = None,
) -> ScfResult:
    """Converge the mean-field state of ``hamiltonian`` in a Fukutome class.

    Parameters
    ----------
    guess
        Starting density in the structure of ``symmetry``; a converged
        density of another class is brought there with
        :meth:`~hafnium.structured.BlockMatrix.embed`, and a perturbation
        of it starts a symmetry-broken search. Without a guess the iteration
        starts from :func:`core_hamiltonian_guess`.
    occupation
        Defaults to the aufbau filling of the Hamiltonian's electron counts,
        which must fit the class; a policy given here is trusted instead.

    Raises
    ------
    ConvergenceError
        After ``max_iterations`` Fock builds; ``last`` holds the result.
    """
    options = ScfOptions() if options is None else options
    structure = symmetry.structure()
    if occupation is None:
        symmetry.validate_electrons(hamiltonian.n_alpha, hamiltonian.n_beta)
        occupation = AufbauOccupation(hamiltonian.n_alpha, hamiltonian.n_beta)
    setup = _Setup.build(hamiltonian, structure, occupation)
    density = (
        core_hamiltonian_guess(hamiltonian, structure)
        if guess is None
        else setup.checked(guess)
    )
    diis = Diis(options.diis_size)
    history: list[ScfIteration] = []
    for index in range(options.max_iterations):
        fock, energies = hamiltonian.fock(density)
        gradient = setup.gradient(fock, density)
        row = _record(history, index, energies, _norm(gradient))
        if _is_converged(row, options):
            return setup.finish(fock, tuple(history), converged=True)
        diis = diis.push(fock, gradient)
        density = setup.next_density(diis.extrapolate())
    # The history always keeps the most recent Fock matrix.
    result = setup.finish(diis.focks[-1], tuple(history), converged=False)
    raise ConvergenceError(
        f"the SCF did not converge in {options.max_iterations} iterations "
        f"(last energy change {history[-1].energy_change:.2e}, gradient "
        f"{history[-1].gradient_norm:.2e})",
        result,
    )


def core_hamiltonian_guess(
    hamiltonian: Hamiltonian, structure: BlockStructure
) -> BlockMatrix:
    """Density of the lowest core-Hamiltonian orbitals, spin-averaged.

    Half the electrons are placed in each spin with
    :class:`~hafnium.scf.AveragedOccupation`, so the guess is spin-symmetric
    and spherical for atoms; the first Fock build then separates the spins
    of an open-shell state. Occupying the core orbitals with ``n_alpha``
    and ``n_beta`` directly would commit the iteration to whichever
    orbitals the core Hamiltonian happens to order lowest.
    """
    half = hamiltonian.n_electrons / 2
    occupation = AveragedOccupation(half, half)
    mo_energies, mo_coefficients = hamiltonian.one_body(structure).eigh(
        hamiltonian.metric(structure)
    )
    return density_matrix(mo_coefficients, occupation.occupy(mo_energies))


def orbital_gradient(
    fock: BlockMatrix,
    density: BlockMatrix,
    overlap: BlockMatrix,
    orthogonalizer: BlockMatrix,
) -> BlockMatrix:
    """``X† (F D S − S D F) X`` with ``X = S^{-1/2}`` the ``orthogonalizer``.

    The commutator vanishes at a stationary point; in the orthonormal basis
    its norm does not depend on the overlap of the atomic orbitals.
    """
    commutator = fock @ density @ overlap - overlap @ density @ fock
    return orthogonalizer.adjoint() @ commutator @ orthogonalizer


def symmetric_orthogonalizer(overlap: Array) -> Array:
    """``S^{-1/2}`` of a positive-definite overlap ``[ao, ao]``."""
    xp = array_namespace(overlap)
    eigenvalues, eigenvectors = eigh(overlap)
    return (eigenvectors / xp.sqrt(eigenvalues)) @ xp.linalg.matrix_transpose(
        eigenvectors
    )


@dataclass(frozen=True, slots=True, eq=False)
class _Setup:
    """Per-structure matrices and policies the loop reuses."""

    hamiltonian: Hamiltonian
    structure: BlockStructure
    occupation: OccupationPolicy
    overlap: BlockMatrix
    orthogonalizer: BlockMatrix

    @classmethod
    def build(
        cls,
        hamiltonian: Hamiltonian,
        structure: BlockStructure,
        occupation: OccupationPolicy,
    ) -> _Setup:
        inverse_root = symmetric_orthogonalizer(hamiltonian.overlap)
        spatial = BlockMatrix(BlockStructure.spatial(), (inverse_root,))
        return cls(
            hamiltonian,
            structure,
            occupation,
            hamiltonian.metric(structure),
            spatial.embed(structure),
        )

    def checked(self, guess: BlockMatrix) -> BlockMatrix:
        if guess.structure != self.structure:
            raise StructureError(
                f"the guess has the {guess.structure.describe()} structure; embed "
                f"it into the {self.structure.describe()} structure first"
            )
        if guess.n_spatial != self.hamiltonian.n_ao:
            raise StructureError(
                f"the guess acts on {guess.n_spatial} atomic orbitals, the "
                f"Hamiltonian on {self.hamiltonian.n_ao}"
            )
        return guess

    def gradient(self, fock: BlockMatrix, density: BlockMatrix) -> BlockMatrix:
        return orbital_gradient(fock, density, self.overlap, self.orthogonalizer)

    def next_density(self, fock: BlockMatrix) -> BlockMatrix:
        mo_energies, mo_coefficients = fock.eigh(self.overlap)
        return density_matrix(mo_coefficients, self.occupation.occupy(mo_energies))

    def finish(
        self, fock: BlockMatrix, history: tuple[ScfIteration, ...], *, converged: bool
    ) -> ScfResult:
        mo_energies, mo_coefficients = fock.eigh(self.overlap)
        mo_occupations = self.occupation.occupy(mo_energies)
        density = density_matrix(mo_coefficients, mo_occupations)
        final_fock, energies = self.hamiltonian.fock(density)
        return ScfResult(
            energies,
            mo_coefficients,
            mo_energies,
            mo_occupations,
            density,
            final_fock,
            converged,
            history,
        )


def _record(
    history: list[ScfIteration], index: int, energies: EnergyTerms, gradient_norm: float
) -> ScfIteration:
    change = energies.total - history[-1].energy if history else math.inf
    row = ScfIteration(index, energies.total, change, gradient_norm)
    history.append(row)
    logger.info(
        "scf iteration %3d  energy %20.12f  change %10.2e  gradient %10.2e",
        index,
        row.energy,
        row.energy_change,
        row.gradient_norm,
    )
    return row


def _is_converged(row: ScfIteration, options: ScfOptions) -> bool:
    return (
        abs(row.energy_change) < options.energy_tolerance
        and row.gradient_norm < options.gradient_tolerance
    )


def _norm(matrix: BlockMatrix) -> float:
    """Frobenius norm over the spin-orbital space, as a host float."""
    xp = array_namespace(*matrix.blocks)
    return math.sqrt(float(xp.real(matrix.adjoint().trace_product(matrix))))

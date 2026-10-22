"""The self-consistent field driver, one implementation for every class.

Everything here acts on :class:`~hafnium.structured.BlockMatrix` in the
structure of the requested Fukutome class (PLAN.md §5), so the same loop
serves real and complex, restricted, unrestricted and generalized states.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from hafnium.backend import array_namespace
from hafnium.errors import ConvergenceError, InputError, StructureError
from hafnium.scf.diis import Diis
from hafnium.scf.hamiltonian import EnergyTerms, Hamiltonian
from hafnium.scf.occupation import (
    AufbauOccupation,
    AveragedOccupation,
    OccupationPolicy,
)
from hafnium.scf.orthogonalization import DEFAULT_OVERLAP_THRESHOLD
from hafnium.scf.spin import SpinExpectation, spin_expectation
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
    level_shift
        Energy in Hartree added to the virtual orbitals before each
        diagonalization, which damps oscillations between nearly degenerate
        occupied and virtual orbitals at the cost of slower convergence.
    damping
        Fraction of the previous density kept when forming the next one.
    overlap_threshold
        Eigenvalues of the overlap matrix below this are removed, so a
        linearly dependent basis loses the dependent combinations.
    """

    max_iterations: int = 100
    energy_tolerance: float = 1e-10
    gradient_tolerance: float = 1e-6
    diis_size: int = 8
    level_shift: float = 0.0
    damping: float = 0.0
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise InputError("at least one iteration is needed")
        if self.energy_tolerance <= 0.0 or self.gradient_tolerance <= 0.0:
            raise InputError("tolerances must be positive")
        if self.diis_size < 1:
            raise InputError("the DIIS subspace holds at least one matrix")
        if self.level_shift < 0.0:
            raise InputError("the level shift is non-negative")
        if not 0.0 <= self.damping < 1.0:
            raise InputError("damping is a fraction in [0, 1)")
        if self.overlap_threshold <= 0.0:
            raise InputError("the overlap threshold is positive")


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
    most the energy tolerance. ``mo_coefficients`` are ``[ao, mo]`` with
    fewer orbitals than functions when the overlap threshold removed
    linearly dependent combinations.
    """

    energies: EnergyTerms
    mo_coefficients: BlockMatrix
    mo_energies: BlockVector
    mo_occupations: BlockVector
    density: BlockMatrix
    fock: BlockMatrix
    spin: SpinExpectation
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
    if symmetry.kpoints != hamiltonian.kpoints:
        raise InputError(
            "the symmetry specification and the Hamiltonian carry different "
            "k-point sets; use SymmetrySpec.with_kpoints with the Hamiltonian's"
        )
    structure = symmetry.structure()
    if occupation is None:
        symmetry.validate_electrons(hamiltonian.n_alpha, hamiltonian.n_beta)
        occupation = AufbauOccupation(hamiltonian.n_alpha, hamiltonian.n_beta)
    setup = _Setup.build(hamiltonian, structure, occupation, options)
    density = _averaged_core_density(setup) if guess is None else setup.checked(guess)
    diis = Diis(options.diis_size)
    history: list[ScfIteration] = []
    for index in range(options.max_iterations):
        fock, energies = hamiltonian.fock(density)
        gradient = setup.gradient(fock, density)
        row = _record(history, index, energies, _norm(gradient))
        if _is_converged(row, options):
            return setup.finish(fock, tuple(history), converged=True)
        diis = diis.push(fock, gradient)
        density = setup.next_density(diis.extrapolate(), density)
    # The history always keeps the most recent Fock matrix.
    result = setup.finish(diis.focks[-1], tuple(history), converged=False)
    raise ConvergenceError(
        f"the SCF did not converge in {options.max_iterations} iterations "
        f"(last energy change {history[-1].energy_change:.2e}, gradient "
        f"{history[-1].gradient_norm:.2e})",
        result,
    )


def core_hamiltonian_guess(
    hamiltonian: Hamiltonian,
    structure: BlockStructure,
    *,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
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
    options = ScfOptions(overlap_threshold=overlap_threshold)
    setup = _Setup.build(
        hamiltonian, structure, AveragedOccupation(half, half), options
    )
    return _averaged_core_density(setup)


def _averaged_core_density(setup: _Setup) -> BlockMatrix:
    """Build the core-Hamiltonian guess in the retained basis of ``setup``."""
    hamiltonian = setup.hamiltonian
    half = hamiltonian.n_electrons / 2
    occupation = AveragedOccupation(half, half)
    mo_energies, mo_coefficients = setup.solve(hamiltonian.one_body(setup.structure))
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


@dataclass(frozen=True, slots=True, eq=False)
class _Setup:
    """Per-structure matrices and policies the loop reuses."""

    hamiltonian: Hamiltonian
    structure: BlockStructure
    occupation: OccupationPolicy
    options: ScfOptions
    overlap: BlockMatrix
    orthogonalizer: BlockMatrix

    @classmethod
    def build(
        cls,
        hamiltonian: Hamiltonian,
        structure: BlockStructure,
        occupation: OccupationPolicy,
        options: ScfOptions,
    ) -> _Setup:
        return cls(
            hamiltonian,
            structure,
            occupation,
            options,
            hamiltonian.metric(structure),
            hamiltonian.orthogonalizer(structure, options.overlap_threshold),
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

    def solve(self, fock: BlockMatrix) -> tuple[BlockVector, BlockMatrix]:
        """Canonical orbitals of ``fock`` in the retained basis, ``[ao, mo]``."""
        reduced = self.orthogonalizer.adjoint() @ fock @ self.orthogonalizer
        mo_energies, reduced_coefficients = reduced.eigh()
        return mo_energies, self.orthogonalizer @ reduced_coefficients

    def next_density(self, fock: BlockMatrix, previous: BlockMatrix) -> BlockMatrix:
        """Diagonalize ``fock``, level-shifted and damped as the options say.

        The shift is applied through ``S − S D S`` of the previous density,
        which is the virtual projector once the iteration has settled; while
        damping keeps the density non-idempotent the shift is approximate,
        which does not move the fixed point.
        """
        shift, damping = self.options.level_shift, self.options.damping
        if shift:
            virtual = self.overlap - self.overlap @ previous @ self.overlap
            fock = fock + shift * virtual
        mo_energies, mo_coefficients = self.solve(fock)
        density = density_matrix(mo_coefficients, self.occupation.occupy(mo_energies))
        if damping:
            density = (1.0 - damping) * density + damping * previous
        return density

    def finish(
        self, fock: BlockMatrix, history: tuple[ScfIteration, ...], *, converged: bool
    ) -> ScfResult:
        mo_energies, mo_coefficients = self.solve(fock)
        mo_occupations = self.occupation.occupy(mo_energies)
        density = density_matrix(mo_coefficients, mo_occupations)
        final_fock, energies = self.hamiltonian.fock(density)
        spin = spin_expectation(density, self.hamiltonian.overlap)
        logger.info(
            "scf %s after %d iterations: energy %.12f, <S^2> %.6f, 2S+1 %.4f",
            "converged" if converged else "stopped",
            len(history),
            energies.total,
            spin.square,
            spin.multiplicity,
        )
        return ScfResult(
            energies,
            mo_coefficients,
            mo_energies,
            mo_occupations,
            density,
            final_fock,
            spin,
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

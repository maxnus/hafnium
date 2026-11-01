"""The Λ equations of CCSD and its one-particle density (ADR-0034).

The de-excitation operators ``Λ1 = Σ l1[i, a] a†_i a_a`` and
``Λ2 = ¼ Σ l2[i, j, a, b] a†_i a†_j a_b a_a`` make the energy functional
``<Φ| (1 + Λ) e^{-T} H_N e^T |Φ>`` stationary in the amplitudes when the
Λ equations hold, ``0 = <Φ| (1 + Λ) (H_N e^T)_C |Φ_μ>`` over the linked
terms, generated like the amplitude equations
(:mod:`hafnium.cc._ccsd_lambda_equations`). They are linear in ``Λ`` and
are iterated like the amplitude equations, each iteration adding
``R / Δ`` with the DIIS extrapolation, from the conjugate amplitudes,
which solve them to first order. The one-particle reduced density matrix
``D[p, q] = <Φ| (1 + Λ) e^{-T} a†_q a_p e^T |Φ>`` is the occupations of
the reference plus the blocks the same generator derives from ``t`` and
``l``; coupled-cluster theory being non-variational it is not Hermitian,
and its Hermitian part is taken and transformed to the atomic basis over
the structure of the reference, like the density of the mean field. The
entry points mirror those of CCSD and return a :class:`CcsdLambdaResult`,
a :class:`~hafnium.cc.CcsdResult` with the de-excitation amplitudes and
the density.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hafnium.backend import Array, array_namespace
from hafnium.basis import BasisSet
from hafnium.cc._ccsd_lambda_equations import (
    FOCK_BLOCKS,
    INTEGRAL_BLOCKS,
    lambda_doubles_residual,
    lambda_singles_residual,
    one_particle_density,
)
from hafnium.cc.ccsd import CcsdOptions, CcsdResult, solve_ccsd
from hafnium.cc.diis import AmplitudeDiis
from hafnium.cc.hamiltonian import NormalOrderedHamiltonian
from hafnium.errors import ConvergenceError, InputError
from hafnium.integrals import IntegralEngine
from hafnium.integrals.density_fitting import DEFAULT_METRIC_THRESHOLD
from hafnium.integrals.periodic.fitting import PeriodicDensityFitting
from hafnium.molecule import Cell, Molecule
from hafnium.orbitals import FittedOrbitalTensors, OrbitalSpaces, Space
from hafnium.scf import ScfResult
from hafnium.structured import BlockMatrix, BlockTensor, IrrepSectors, TensorStructure
from hafnium.structured.validation import dtype_for
from hafnium.symmetry import KPoints

OCCUPIED, VIRTUAL = Space.OCCUPIED, Space.VIRTUAL


@dataclass(frozen=True, slots=True)
class LambdaIteration:
    """One evaluation of the residuals of the Λ equations."""

    index: int
    update_norm: float


@dataclass(frozen=True, slots=True, eq=False)
class CcsdLambdaResult(CcsdResult):
    """A :class:`~hafnium.cc.CcsdResult` with its de-excitation amplitudes and density.

    Parameters
    ----------
    lambda_singles, lambda_doubles
        The de-excitation amplitudes ``l1[i, a]`` and ``l2[i, j, a, b]``
        solving the Λ equations, block tensors over the orbital spaces of
        the reference with the occupied indices conjugated (CONVENTIONS.md
        §3), the structure of the conjugate amplitudes.
    density
        The Hermitian part of the one-particle reduced density matrix of
        CCSD in the atomic basis, over the structure of the reference like
        :attr:`~hafnium.scf.ScfResult.density`; its trace with the overlap
        is the number of electrons.
    lambda_converged
        Whether the Λ iteration reached its tolerance.
    lambda_history
        One entry per Λ iteration.
    """

    lambda_singles: BlockTensor
    lambda_doubles: BlockTensor
    density: BlockMatrix
    lambda_converged: bool
    lambda_history: tuple[LambdaIteration, ...]

    @property
    def n_lambda_iterations(self) -> int:
        """Number of evaluations of the Λ residuals."""
        return len(self.lambda_history)

    @classmethod
    def from_ccsd(
        cls,
        result: CcsdResult,
        lambda_singles: BlockTensor,
        lambda_doubles: BlockTensor,
        density: BlockMatrix,
        converged: bool,
        history: tuple[LambdaIteration, ...],
    ) -> CcsdLambdaResult:
        """Attach the solution of the Λ equations to the fields of ``result``."""
        return cls(
            result.reference_energy,
            result.correlation_energy,
            result.singles,
            result.doubles,
            result.converged,
            result.history,
            lambda_singles,
            lambda_doubles,
            density,
            converged,
            history,
        )


def df_ccsd_lambda(  # noqa: PLR0913
    engine: IntegralEngine,
    basis: BasisSet,
    system: Molecule | Cell,
    auxiliary: BasisSet,
    reference: ScfResult,
    *,
    kpoints: KPoints | None = None,
    threshold: float = DEFAULT_METRIC_THRESHOLD,
    options: CcsdOptions | None = None,
) -> CcsdLambdaResult:
    """Compute the DF-CCSD energy and density of a converged Hartree–Fock ``reference``.

    The fitted tensors are built as :func:`~hafnium.cc.df_ccsd` builds
    them, then :func:`solve_ccsd_lambda` iterates CCSD and its Λ equations.
    """
    fitted = FittedOrbitalTensors.build(
        engine,
        basis,
        system,
        auxiliary,
        reference,
        kpoints=kpoints,
        threshold=threshold,
    )
    return solve_ccsd_lambda(reference, fitted, options)


def ccsd_lambda_from_df_tensor(
    reference: ScfResult,
    df_tensor: Array,
    *,
    auxiliary: IrrepSectors | None = None,
    options: CcsdOptions | None = None,
) -> CcsdLambdaResult:
    """Compute the DF-CCSD energy and density of a molecular ``reference``.

    ``df_tensor`` and ``auxiliary`` are as for
    :func:`~hafnium.cc.ccsd_from_df_tensor`.
    """
    fitted = FittedOrbitalTensors.from_df_tensor(
        df_tensor, OrbitalSpaces.from_scf(reference), auxiliary=auxiliary
    )
    return solve_ccsd_lambda(reference, fitted, options)


def ccsd_lambda_from_periodic_fitting(
    reference: ScfResult,
    fitting: PeriodicDensityFitting,
    *,
    threshold: float = DEFAULT_METRIC_THRESHOLD,
    options: CcsdOptions | None = None,
) -> CcsdLambdaResult:
    """Compute the DF-CCSD energy per cell and density of a periodic ``reference``.

    The fitted tensors are those of
    :func:`~hafnium.cc.ccsd_from_periodic_fitting`; the density holds a
    block per stored k-point of the reference.
    """
    fitted = FittedOrbitalTensors.from_periodic_fitting(
        fitting, OrbitalSpaces.from_scf(reference), threshold=threshold
    )
    return solve_ccsd_lambda(reference, fitted, options)


def solve_ccsd_lambda(
    reference: ScfResult,
    fitted: FittedOrbitalTensors,
    options: CcsdOptions | None = None,
    *,
    guess: tuple[BlockTensor, BlockTensor] | None = None,
) -> CcsdLambdaResult:
    """Iterate CCSD (:func:`~hafnium.cc.solve_ccsd`) and then its Λ equations.

    Raises
    ------
    ConvergenceError
        If CCSD does not converge, with its last
        :class:`~hafnium.cc.CcsdResult`, or the Λ iteration does not, with
        the last :class:`CcsdLambdaResult`.
    """
    result = solve_ccsd(reference, fitted, options, guess=guess)
    return solve_lambda(fitted, result, options)


def solve_lambda(
    fitted: FittedOrbitalTensors,
    result: CcsdResult,
    options: CcsdOptions | None = None,
    *,
    guess: tuple[BlockTensor, BlockTensor] | None = None,
) -> CcsdLambdaResult:
    """Iterate the Λ equations for the amplitudes of ``result`` and form the density.

    The de-excitation amplitudes start from the conjugates of the
    amplitudes, which solve the equations to first order, or from
    ``guess``, the de-excitation amplitudes of an earlier result; each
    iteration adds ``R / Δ`` with the DIIS extrapolation and the iteration
    converges when the norm of the update is below the amplitude
    tolerance of ``options``, the equations being linear in ``Λ`` with no
    energy to watch.

    Raises
    ------
    ConvergenceError
        If the tolerance is not met within the iteration limit of
        ``options``; the exception carries the last :class:`CcsdLambdaResult`.
    """
    if options is None:
        options = CcsdOptions()
    spaces = fitted.spaces
    hamiltonian = NormalOrderedHamiltonian.from_fitted(
        fitted, fock_blocks=FOCK_BLOCKS, integral_blocks=INTEGRAL_BLOCKS
    )
    singles, doubles = result.singles, result.doubles
    evaluated = _starting_point(singles, doubles, guess)
    denominators = (
        spaces.denominators(evaluated[0].structure, (OCCUPIED, VIRTUAL)),
        spaces.denominators(
            evaluated[1].structure, (OCCUPIED, OCCUPIED, VIRTUAL, VIRTUAL)
        ),
    )
    diis = AmplitudeDiis(options.diis_size)
    history: list[LambdaIteration] = []
    for index in range(options.max_iterations):
        update = (
            lambda_singles_residual(hamiltonian, singles, doubles, *evaluated)
            / denominators[0],
            lambda_doubles_residual(hamiltonian, singles, doubles, *evaluated)
            / denominators[1],
        )
        norm = math.sqrt(AmplitudeDiis.overlap(update, update))
        history.append(LambdaIteration(index, norm))
        if norm < options.amplitude_tolerance:
            density = _density(spaces, singles, doubles, *evaluated)
            return CcsdLambdaResult.from_ccsd(
                result, *evaluated, density, True, tuple(history)
            )
        diis = diis.push((evaluated[0] + update[0], evaluated[1] + update[1]), update)
        extrapolated = diis.extrapolate()
        evaluated = (extrapolated[0], extrapolated[1])
    density = _density(spaces, singles, doubles, *evaluated)
    raise ConvergenceError(
        f"the Λ iteration did not converge in {options.max_iterations} iterations "
        f"(last update norm {history[-1].update_norm:.2e})",
        CcsdLambdaResult.from_ccsd(result, *evaluated, density, False, tuple(history)),
    )


def _starting_point(
    singles: BlockTensor,
    doubles: BlockTensor,
    guess: tuple[BlockTensor, BlockTensor] | None,
) -> tuple[BlockTensor, BlockTensor]:
    """Return the starting de-excitation amplitudes, the conjugate amplitudes."""
    if guess is None:
        return singles.conj(), doubles.conj()
    lambda_singles, lambda_doubles = guess
    if (
        lambda_singles.structure != singles.structure.conjugated()
        or lambda_doubles.structure != doubles.structure.conjugated()
    ):
        raise InputError(
            "the starting de-excitation amplitudes have another structure than the "
            "conjugate amplitudes"
        )
    return lambda_singles, lambda_doubles


def _density(
    spaces: OrbitalSpaces,
    singles: BlockTensor,
    doubles: BlockTensor,
    lambda_singles: BlockTensor,
    lambda_doubles: BlockTensor,
) -> BlockMatrix:
    """Form the Hermitian part of the CCSD density in the atomic basis."""
    gamma = one_particle_density(singles, doubles, lambda_singles, lambda_doubles)
    # ``D[p, q] = <a†_q a_p>`` is the generated ``[q, p]`` block plus the
    # occupations of the reference; the Hermitian part is taken block by block.
    transposed = {(q, p): block.permute("pq->qp") for (p, q), block in gamma.items()}
    density = {
        (p, q): (block + transposed[q, p].permute("pq->qp").conj()) / 2
        for (p, q), block in transposed.items()
    }
    occupied = density[OCCUPIED, OCCUPIED]
    density[OCCUPIED, OCCUPIED] = occupied + _occupations(occupied)
    return spaces.to_atomic_basis(density)


def _occupations(block: BlockTensor) -> BlockTensor:
    """``δ_ij`` over the structure of ``block``, the density of the reference."""
    structure: TensorStructure = block.structure
    xp = array_namespace(*block.blocks)
    dtype = dtype_for(xp, structure.ring)

    def build(key: tuple[int, ...]) -> Array:
        shape = structure.shape(key)
        if key[0] != key[1]:
            return xp.zeros(shape, dtype=dtype)
        return xp.astype(xp.eye(shape[0]), dtype)

    return BlockTensor.from_function(structure, build)

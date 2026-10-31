"""Coupled-cluster singles and doubles with density fitting (ADR-0032).

The spin-orbital CCSD equations of :mod:`hafnium.cc._ccsd_equations`,
generated once, are iterated for any Fukutome class of the reference:
the residuals ``R1 = <Φ_i^a| H_N e^T |Φ>_C`` and ``R2 = <Φ_ij^ab| H_N e^T
|Φ>_C`` vanish at the solution, and each iteration adds ``R / Δ`` to the
amplitudes, ``Δ`` the orbital energy denominators, with the DIIS
extrapolation of :class:`~hafnium.cc.diis.AmplitudeDiis`. The sums run
over the blocks the symmetry of the reference allows, as for MP2, and
over the k-points of a cell the energy of the supercell of the mesh
divides by the number of points per cell (ADR-0031).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hafnium.backend import Array, array_namespace
from hafnium.basis import BasisSet
from hafnium.cc._ccsd_equations import (
    FOCK_BLOCKS,
    INTEGRAL_BLOCKS,
    doubles_residual,
    energy,
    singles_residual,
)
from hafnium.cc.diis import AmplitudeDiis
from hafnium.cc.hamiltonian import NormalOrderedHamiltonian
from hafnium.correlated import FittedOrbitalTensors, OrbitalSpaces, Space
from hafnium.errors import ConvergenceError, InputError
from hafnium.integrals import IntegralEngine
from hafnium.integrals.density_fitting import DEFAULT_METRIC_THRESHOLD
from hafnium.integrals.periodic.fitting import PeriodicDensityFitting
from hafnium.molecule import Cell, Molecule
from hafnium.scf import ScfResult
from hafnium.structured import BlockTensor, IrrepSectors, TensorStructure
from hafnium.symmetry import KPoints

OCCUPIED, VIRTUAL = Space.OCCUPIED, Space.VIRTUAL


@dataclass(frozen=True, slots=True)
class CcsdOptions:
    """Tolerances and limits of the amplitude iteration.

    Parameters
    ----------
    max_iterations
        Residual evaluations before :class:`~hafnium.errors.ConvergenceError`
        is raised.
    energy_tolerance
        Largest change of the correlation energy between iterations at
        convergence, in Hartree.
    amplitude_tolerance
        Largest norm of the amplitude update ``R / Δ`` at convergence,
        over the stored blocks of the singles and doubles together.
    diis_size
        Sets of amplitudes kept for extrapolation; ``1`` iterates
        without DIIS.
    """

    max_iterations: int = 100
    energy_tolerance: float = 1e-10
    amplitude_tolerance: float = 1e-8
    diis_size: int = 8

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise InputError("at least one iteration is needed")
        if self.energy_tolerance <= 0.0 or self.amplitude_tolerance <= 0.0:
            raise InputError("tolerances must be positive")
        if self.diis_size < 1:
            raise InputError("the DIIS subspace holds at least one set of amplitudes")


@dataclass(frozen=True, slots=True)
class CcsdIteration:
    """One evaluation of the energy and the residuals."""

    index: int
    correlation_energy: float
    energy_change: float
    update_norm: float


@dataclass(frozen=True, slots=True, eq=False)
class CcsdResult:
    """The CCSD energy and amplitudes of a mean-field reference.

    Parameters
    ----------
    reference_energy
        Total energy of the reference state in Hartree, per cell for a
        periodic system.
    correlation_energy
        The CCSD correlation energy in Hartree, per cell for a periodic
        system.
    singles, doubles
        The amplitudes ``t1[i, a]`` and ``t2[i, j, a, b]`` the energy was
        evaluated with, block tensors over the occupied and virtual
        orbitals of the reference with the virtual indices conjugated
        (CONVENTIONS.md §3); over k-points those of the supercell of the
        mesh.
    converged
        Whether the iteration reached its tolerances.
    history
        One entry per iteration.
    """

    reference_energy: float
    correlation_energy: float
    singles: BlockTensor
    doubles: BlockTensor
    converged: bool
    history: tuple[CcsdIteration, ...]

    @property
    def total_energy(self) -> float:
        """Reference plus correlation energy in Hartree."""
        return self.reference_energy + self.correlation_energy

    @property
    def n_iterations(self) -> int:
        """Number of residual evaluations."""
        return len(self.history)


def df_ccsd(  # noqa: PLR0913
    engine: IntegralEngine,
    basis: BasisSet,
    system: Molecule | Cell,
    auxiliary: BasisSet,
    reference: ScfResult,
    *,
    kpoints: KPoints | None = None,
    threshold: float = DEFAULT_METRIC_THRESHOLD,
    options: CcsdOptions | None = None,
) -> CcsdResult:
    """Compute the DF-CCSD energy of a converged Hartree–Fock ``reference``.

    The two-electron integrals are fitted as
    :meth:`~hafnium.correlated.FittedOrbitalTensors.build` says: in the
    ``auxiliary`` basis with the Coulomb metric, adapted to the point
    group of a molecular reference over irrep sectors, with momentum
    transfer over the ``kpoints`` of a cell. Any Fukutome class serves as
    the reference; its orbitals must be occupied by whole electrons.
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
    return solve_ccsd(reference, fitted, options)


def ccsd_from_df_tensor(
    reference: ScfResult,
    df_tensor: Array,
    *,
    auxiliary: IrrepSectors | None = None,
    options: CcsdOptions | None = None,
) -> CcsdResult:
    """Compute the DF-CCSD energy of a molecular ``reference`` from its fitted tensor.

    ``auxiliary`` holds the irrep sectors of the auxiliary basis when the
    reference is blocked over a point group; see
    :func:`~hafnium.correlated.transform_df_tensor`.
    """
    fitted = FittedOrbitalTensors.from_df_tensor(
        df_tensor, OrbitalSpaces.from_scf(reference), auxiliary=auxiliary
    )
    return solve_ccsd(reference, fitted, options)


def ccsd_from_periodic_fitting(
    reference: ScfResult,
    fitting: PeriodicDensityFitting,
    *,
    threshold: float = DEFAULT_METRIC_THRESHOLD,
    options: CcsdOptions | None = None,
) -> CcsdResult:
    """Compute the DF-CCSD energy per cell of a periodic ``reference``.

    The fitted tensor of ``fitting`` enters the Bloch orbital basis in the
    supercell normalization (:func:`~hafnium.correlated.transform_periodic_fitting`),
    the equations of the supercell of the mesh are iterated over the
    blocks that conserve crystal momentum, and the energy per cell is
    that of the supercell divided by the number of k-points.
    """
    fitted = FittedOrbitalTensors.from_periodic_fitting(
        fitting, OrbitalSpaces.from_scf(reference), threshold=threshold
    )
    return solve_ccsd(reference, fitted, options)


def solve_ccsd(
    reference: ScfResult,
    fitted: FittedOrbitalTensors,
    options: CcsdOptions | None = None,
    *,
    guess: tuple[BlockTensor, BlockTensor] | None = None,
) -> CcsdResult:
    """Iterate the CCSD equations from the fitted tensors of ``reference``.

    Without a ``guess`` the singles start from zero and the doubles from
    the first-order amplitudes ``<ab||ij> / Δ``, so the energy of the
    first iteration is the MP2 energy; ``guess`` holds starting singles
    and doubles instead, the amplitudes of an earlier result.

    Raises
    ------
    ConvergenceError
        If the tolerances of ``options`` are not met within its iteration
        limit; the exception carries the last :class:`CcsdResult`.
    """
    if options is None:
        options = CcsdOptions()
    spaces = fitted.spaces
    hamiltonian = NormalOrderedHamiltonian.from_fitted(
        fitted, fock_blocks=FOCK_BLOCKS, integral_blocks=INTEGRAL_BLOCKS
    )
    singles, doubles, denominators_singles, denominators_doubles = _starting_point(
        hamiltonian.integrals[OCCUPIED, OCCUPIED, VIRTUAL, VIRTUAL], spaces, guess
    )
    per_cell = 1.0 / fitted.n_cells
    diis = AmplitudeDiis(options.diis_size)
    history: list[CcsdIteration] = []
    evaluated = (singles, doubles)
    for index in range(options.max_iterations):
        evaluated = (singles, doubles)
        correlation = per_cell * _real(energy(hamiltonian, singles, doubles))
        update_singles = (
            singles_residual(hamiltonian, singles, doubles) / denominators_singles
        )
        update_doubles = (
            doubles_residual(hamiltonian, singles, doubles) / denominators_doubles
        )
        change = correlation - history[-1].correlation_energy if history else math.inf
        norm = math.sqrt(_squared_norm(update_singles) + _squared_norm(update_doubles))
        history.append(CcsdIteration(index, correlation, change, norm))
        # The first iteration has no energy change to judge; the update does.
        converged = norm < options.amplitude_tolerance and (
            not history[:-1] or abs(change) < options.energy_tolerance
        )
        if converged:
            return CcsdResult(
                reference.energy, correlation, singles, doubles, True, tuple(history)
            )
        diis = diis.push(
            (singles + update_singles, doubles + update_doubles),
            (update_singles, update_doubles),
        )
        singles, doubles = diis.extrapolate()
    result = CcsdResult(
        reference.energy,
        history[-1].correlation_energy,
        *evaluated,
        False,
        tuple(history),
    )
    raise ConvergenceError(
        f"the CCSD iteration did not converge in {options.max_iterations} "
        f"iterations (last energy change {history[-1].energy_change:.2e}, update "
        f"norm {history[-1].update_norm:.2e})",
        result,
    )


def _starting_point(
    integrals: BlockTensor,
    spaces: OrbitalSpaces,
    guess: tuple[BlockTensor, BlockTensor] | None,
) -> tuple[BlockTensor, BlockTensor, BlockTensor, BlockTensor]:
    """Return the starting singles and doubles with their energy denominators."""
    excitations = integrals.conj()
    denominators_doubles = spaces.denominators(excitations.structure)
    singles_structure = TensorStructure(
        (
            spaces.axis(Space.OCCUPIED, conjugated=False),
            spaces.axis(Space.VIRTUAL, conjugated=True),
        ),
        ring=excitations.structure.ring,
    )
    denominators_singles = spaces.denominators(singles_structure)
    if guess is None:
        doubles = excitations / denominators_doubles
        singles = BlockTensor.zeros(
            singles_structure,
            xp=array_namespace(*doubles.blocks) if doubles.blocks else None,
        )
    else:
        singles, doubles = guess
        if (
            singles.structure != singles_structure
            or doubles.structure != excitations.structure
        ):
            raise InputError(
                "the starting amplitudes have another structure than the orbital "
                "spaces of the reference give"
            )
    return singles, doubles, denominators_singles, denominators_doubles


def _real(value: BlockTensor) -> float:
    scalar = value.scalar
    xp = array_namespace(scalar)
    return float(xp.real(scalar))


def _squared_norm(tensor: BlockTensor) -> float:
    total = 0.0
    for block in tensor.blocks:
        xp = array_namespace(block)
        total += float(xp.sum(xp.abs(block) ** 2))
    return total

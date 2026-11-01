"""CCSD(T): the (T) correction over windows of the occupied axis (ADR-0033).

With the converged CCSD amplitudes, the correction is

``E(T) = 1/36 Σ_ijkabc conj(W + V) W / D``,

``W = <Φ_ijk^abc| H_N T2 |Φ>_C`` the connected triples the doubles
generate, ``V = <Φ_ijk^abc| H_N (T1 + T2) |Φ>`` the disconnected ones and
``D = ε_i + ε_j + ε_k − ε_a − ε_b − ε_c``: the fourth-order energy of the
triples ``<Φ| T2† H_N T3 |Φ>`` with ``T3 = W / D`` and the fifth-order
``<Φ| T1† H_N T3 |Φ>``, the bra giving the conjugates. The programs of
:mod:`hafnium.cc._ccsd_pt_equations` evaluate ``W`` and ``V`` over three
windows of the occupied axis at a time, so that the triples are never
held in full: the occupied axis is cut into windows
(:meth:`~hafnium.structured.TensorAxis.windows`), and since the summand
is symmetric under the permutations of ``i, j, k`` the sum runs over the
triples ``I ≤ J ≤ K`` of windows, each counted as often as its orderings.
Over the k-points of a cell the energy of the supercell of the mesh is
divided by the number of points per cell (ADR-0031). The entry points
mirror those of CCSD and return a :class:`CcsdPtResult`, a
:class:`~hafnium.cc.CcsdResult` with the correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from hafnium.backend import Array, array_namespace
from hafnium.basis import BasisSet
from hafnium.cc._ccsd_pt_equations import (
    FOCK_BLOCKS,
    INTEGRAL_BLOCKS,
    connected_triples,
    disconnected_triples,
)
from hafnium.cc.ccsd import CcsdOptions, CcsdResult, solve_ccsd
from hafnium.cc.hamiltonian import NormalOrderedHamiltonian
from hafnium.errors import InputError
from hafnium.integrals import IntegralEngine
from hafnium.integrals.density_fitting import DEFAULT_METRIC_THRESHOLD
from hafnium.integrals.periodic.fitting import PeriodicDensityFitting
from hafnium.molecule import Cell, Molecule
from hafnium.orbitals import FittedOrbitalTensors, OrbitalSpaces, Space
from hafnium.scf import ScfResult
from hafnium.structured import (
    BlockTensor,
    IrrepSectors,
    Ring,
    TensorAxis,
    TensorStructure,
    contract,
)
from hafnium.symmetry import KPoints

SLOTS = (Space.OCCUPIED,) * 3 + (Space.VIRTUAL,) * 3
"""The spaces of the axes of a triples tensor ``[i, j, k, a, b, c]``."""

TENSORS_PER_BATCH = 8
"""Tensors of the extent of a batch of triples that are alive at once."""


@dataclass(frozen=True, slots=True)
class TriplesOptions:
    """How the triples are batched.

    Parameters
    ----------
    window
        Occupied orbitals per window of a sector
        (:meth:`~hafnium.structured.TensorAxis.windows`); ``None`` takes
        the largest window whose batch of triples is estimated to fit
        ``memory_limit``.
    memory_limit
        Bytes the tensors of one batch may take when the window is chosen
        by it: the stored blocks of a triples tensor over three windows,
        counted :data:`TENSORS_PER_BATCH` times.
    """

    window: int | None = None
    memory_limit: int = 2**30

    def __post_init__(self) -> None:
        if self.window is not None and self.window < 1:
            raise InputError("a window holds at least one occupied orbital")
        if self.memory_limit < 1:
            raise InputError("the memory limit of a batch is positive")


@dataclass(frozen=True, slots=True, eq=False)
class CcsdPtResult(CcsdResult):
    """A :class:`~hafnium.cc.CcsdResult` with its (T) correction.

    Parameters
    ----------
    triples_correction
        The (T) correction in Hartree, per cell for a periodic system;
        ``correlation_energy`` stays that of CCSD.
    """

    triples_correction: float

    @property
    def total_energy(self) -> float:
        """Reference plus CCSD correlation energy plus the (T) correction."""
        return self.reference_energy + self.correlation_energy + self.triples_correction

    @classmethod
    def from_ccsd(cls, result: CcsdResult, triples_correction: float) -> CcsdPtResult:
        """Attach ``triples_correction`` to the fields of ``result``."""
        return cls(
            result.reference_energy,
            result.correlation_energy,
            result.singles,
            result.doubles,
            result.converged,
            result.history,
            triples_correction,
        )


def df_ccsd_pt(  # noqa: PLR0913
    engine: IntegralEngine,
    basis: BasisSet,
    system: Molecule | Cell,
    auxiliary: BasisSet,
    reference: ScfResult,
    *,
    kpoints: KPoints | None = None,
    threshold: float = DEFAULT_METRIC_THRESHOLD,
    options: CcsdOptions | None = None,
    triples: TriplesOptions | None = None,
) -> CcsdPtResult:
    """Compute the DF-CCSD(T) energy of a converged Hartree–Fock ``reference``.

    The fitted tensors are built as :func:`~hafnium.cc.df_ccsd` builds
    them, then :func:`solve_ccsd_pt` iterates CCSD and adds the correction.
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
    return solve_ccsd_pt(reference, fitted, options, triples)


def ccsd_pt_from_df_tensor(
    reference: ScfResult,
    df_tensor: Array,
    *,
    auxiliary: IrrepSectors | None = None,
    options: CcsdOptions | None = None,
    triples: TriplesOptions | None = None,
) -> CcsdPtResult:
    """Compute the DF-CCSD(T) energy of a molecular ``reference`` from a fitted tensor.

    ``auxiliary`` is as for :func:`~hafnium.cc.ccsd_from_df_tensor`.
    """
    fitted = FittedOrbitalTensors.from_df_tensor(
        df_tensor, OrbitalSpaces.from_scf(reference), auxiliary=auxiliary
    )
    return solve_ccsd_pt(reference, fitted, options, triples)


def ccsd_pt_from_periodic_fitting(
    reference: ScfResult,
    fitting: PeriodicDensityFitting,
    *,
    threshold: float = DEFAULT_METRIC_THRESHOLD,
    options: CcsdOptions | None = None,
    triples: TriplesOptions | None = None,
) -> CcsdPtResult:
    """Compute the DF-CCSD(T) energy per cell of a periodic ``reference``.

    The fitted tensors are those of
    :func:`~hafnium.cc.ccsd_from_periodic_fitting`; the correction is per
    cell like the correlation energy.
    """
    fitted = FittedOrbitalTensors.from_periodic_fitting(
        fitting, OrbitalSpaces.from_scf(reference), threshold=threshold
    )
    return solve_ccsd_pt(reference, fitted, options, triples)


def solve_ccsd_pt(
    reference: ScfResult,
    fitted: FittedOrbitalTensors,
    options: CcsdOptions | None = None,
    triples: TriplesOptions | None = None,
    *,
    guess: tuple[BlockTensor, BlockTensor] | None = None,
) -> CcsdPtResult:
    """Iterate CCSD (:func:`~hafnium.cc.solve_ccsd`) and add the (T) correction.

    Raises
    ------
    ConvergenceError
        If CCSD does not converge; the exception carries its last
        :class:`~hafnium.cc.CcsdResult`, without a correction.
    """
    result = solve_ccsd(reference, fitted, options, guess=guess)
    correction = triples_correction(fitted, result.singles, result.doubles, triples)
    return CcsdPtResult.from_ccsd(result, correction)


def triples_correction(
    fitted: FittedOrbitalTensors,
    singles: BlockTensor,
    doubles: BlockTensor,
    options: TriplesOptions | None = None,
) -> float:
    """Compute the (T) correction of converged CCSD amplitudes, in Hartree.

    ``singles`` and ``doubles`` are the amplitudes of a
    :class:`~hafnium.cc.CcsdResult` over the orbital spaces of ``fitted``;
    over k-points the correction is per cell. The entry points above
    call it on the result of CCSD.
    """
    if options is None:
        options = TriplesOptions()
    hamiltonian = NormalOrderedHamiltonian.from_fitted(
        fitted, fock_blocks=FOCK_BLOCKS, integral_blocks=INTEGRAL_BLOCKS
    )
    occupied = doubles.structure.axes[0]
    size = options.window
    if size is None:
        size = window_size(doubles.structure, options.memory_limit)
    windows = occupied.windows(size)
    total = 0.0
    for p, i in enumerate(windows):
        for q in range(p, len(windows)):
            for r in range(q, len(windows)):
                orderings = 6 if p < q < r else 1 if p == q == r else 3
                total += orderings * _batch(
                    hamiltonian,
                    singles,
                    doubles,
                    fitted.spaces,
                    i,
                    windows[q],
                    windows[r],
                )
    return total / (36 * fitted.n_cells)


def window_size(doubles: TensorStructure, memory_limit: int) -> int:
    """Return the largest window whose batch of triples fits ``memory_limit`` bytes.

    The estimate takes the stored blocks of the triples over the first
    window of one occupied orbital on every index, which grow with the
    cube of the window, times :data:`TENSORS_PER_BATCH`; the window is at
    least one orbital and at most the largest sector.
    """
    occupied, virtual = doubles.axes[0], doubles.axes[2]
    windows = occupied.windows(1)
    if not windows:
        return 1
    unit = windows[0]
    structure = TensorStructure(
        (unit, unit, unit, virtual, virtual, virtual), ring=doubles.ring
    )
    volume = sum(prod(structure.shape(key)) for key in structure.stored_keys)
    bytes_per_unit = volume * (16 if doubles.ring is Ring.COMPLEX else 8)
    affordable = int(
        (memory_limit / max(1, TENSORS_PER_BATCH * bytes_per_unit)) ** (1 / 3)
    )
    return max(1, min(affordable, max(occupied.sizes)))


def _batch(
    hamiltonian: NormalOrderedHamiltonian,
    singles: BlockTensor,
    doubles: BlockTensor,
    spaces: OrbitalSpaces,
    i: TensorAxis,
    j: TensorAxis,
    k: TensorAxis,
) -> float:
    """``Σ conj(W + V) W / D`` over one triple of windows."""
    connected = connected_triples(hamiltonian, singles, doubles, i, j, k)
    disconnected = disconnected_triples(hamiltonian, singles, doubles, i, j, k)
    denominators = spaces.denominators(connected.structure, SLOTS)
    value = contract(
        "ijkabc,ijkabc->",
        (connected + disconnected).conj(),
        connected / denominators,
    ).scalar
    xp = array_namespace(value)
    return float(xp.real(value))

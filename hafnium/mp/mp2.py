"""Second-order Møller–Plesset perturbation theory with density fitting (ADR-0030).

In spin-orbital form the second-order energy of a single determinant is

``E₂ = ¼ Σ_ijab |<ij||ab>|² / (ε_i + ε_j − ε_a − ε_b)``

with the antisymmetrized integrals ``<ij||ab> = (ia|jb) − (ib|ja)`` of
CONVENTIONS.md §3 and ``(ia|jb) = Σ_P B[P, i, a] B[P, j, b]`` from the
fitted tensor in the orbital basis. The sums run over the blocks the
symmetry of the reference allows: for a restricted reference the spin
blocks of one spin follow from the other, for an unrestricted one the
same-spin and opposite-spin blocks are the familiar UMP2 terms, and a
generalized reference has one spinor block. Over the k-points of a cell
the orbital indices carry a crystal momentum, the blocks that conserve it
are summed, and the energy of the supercell of the mesh divides by the
number of points per cell (ADR-0031). The amplitudes
``t2[i, j, a, b] = <ab||ij> / (ε_i + ε_j − ε_a − ε_b)`` of the first-order
wavefunction are kept.
"""

from __future__ import annotations

from dataclasses import dataclass

from hafnium.backend import Array, array_namespace
from hafnium.basis import BasisSet
from hafnium.correlated import FittedOrbitalTensors, OrbitalSpaces, Space
from hafnium.integrals import IntegralEngine
from hafnium.integrals.density_fitting import DEFAULT_METRIC_THRESHOLD
from hafnium.integrals.periodic.fitting import PeriodicDensityFitting
from hafnium.molecule import Cell, Molecule
from hafnium.scf import ScfResult
from hafnium.structured import BlockTensor, IrrepSectors, contract
from hafnium.symmetry import KPoints


@dataclass(frozen=True, slots=True, eq=False)
class Mp2Result:
    """The second-order energy of a mean-field reference.

    Parameters
    ----------
    reference_energy
        Total energy of the reference state in Hartree, per cell for a
        periodic system.
    correlation_energy
        The second-order correction in Hartree, per cell for a periodic
        system.
    amplitudes
        ``t2[i, j, a, b] = <ab||ij> / (ε_i + ε_j − ε_a − ε_b)`` as a block
        tensor over the occupied and virtual orbitals of the reference,
        with the index character of the excitation ``a†_a a†_b a_j a_i``:
        the virtual indices conjugated, the occupied ones not. Over
        k-points the orbitals carry their crystal momentum and the
        amplitudes are those of the supercell of the mesh.
    """

    reference_energy: float
    correlation_energy: float
    amplitudes: BlockTensor

    @property
    def total_energy(self) -> float:
        """Reference plus correlation energy in Hartree."""
        return self.reference_energy + self.correlation_energy


def df_mp2(
    engine: IntegralEngine,
    basis: BasisSet,
    system: Molecule | Cell,
    auxiliary: BasisSet,
    reference: ScfResult,
    *,
    kpoints: KPoints | None = None,
    threshold: float = DEFAULT_METRIC_THRESHOLD,
) -> Mp2Result:
    """Compute the DF-MP2 energy of a converged Hartree–Fock ``reference``.

    The two-electron integrals are fitted as
    :meth:`~hafnium.correlated.FittedOrbitalTensors.build` says: in the
    ``auxiliary`` basis with the Coulomb metric, whose eigenvalues below
    ``threshold`` of the largest are dropped, adapted to the point group
    of a molecular reference over irrep sectors, with momentum transfer
    over the ``kpoints`` of a cell (ADR-0019). Any Fukutome class serves
    as the reference; its orbitals must be occupied by whole electrons.
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
    return second_order(reference, fitted)


def mp2_from_df_tensor(
    reference: ScfResult, df_tensor: Array, *, auxiliary: IrrepSectors | None = None
) -> Mp2Result:
    """Compute the DF-MP2 energy of a molecular ``reference`` from its fitted tensor.

    ``auxiliary`` holds the irrep sectors of the auxiliary basis when the
    reference is blocked over a point group; see
    :func:`~hafnium.correlated.transform_df_tensor`.
    """
    fitted = FittedOrbitalTensors.from_df_tensor(
        df_tensor, OrbitalSpaces.from_scf(reference), auxiliary=auxiliary
    )
    return second_order(reference, fitted)


def mp2_from_periodic_fitting(
    reference: ScfResult,
    fitting: PeriodicDensityFitting,
    *,
    threshold: float = DEFAULT_METRIC_THRESHOLD,
) -> Mp2Result:
    """Compute the DF-MP2 energy per cell of a periodic ``reference``.

    The fitted tensor of ``fitting`` enters the Bloch orbital basis in the
    supercell normalization (:func:`~hafnium.correlated.transform_periodic_fitting`),
    the second-order energy of the supercell of the mesh is summed over
    the blocks that conserve crystal momentum, and the energy per cell is
    that divided by the number of k-points.
    """
    fitted = FittedOrbitalTensors.from_periodic_fitting(
        fitting, OrbitalSpaces.from_scf(reference), threshold=threshold
    )
    return second_order(reference, fitted)


def second_order(reference: ScfResult, fitted: FittedOrbitalTensors) -> Mp2Result:
    """Form the second-order energy and amplitudes from the fitted tensors.

    ``<ij||ab>`` is :meth:`~hafnium.correlated.FittedOrbitalTensors.antisymmetrized`
    of the occupied–occupied–virtual–virtual block; ``t2 = <ab||ij> / Δ``
    is its conjugate over the denominators, and ``E₂ = ¼ Σ <ij||ab> t2``
    divided by the number of cells in the supercell of the mesh.
    """
    integrals = fitted.antisymmetrized(
        (Space.OCCUPIED, Space.OCCUPIED, Space.VIRTUAL, Space.VIRTUAL)
    )
    excitations = integrals.conj()
    amplitudes = excitations / fitted.spaces.denominators(excitations.structure)
    pairing = contract("ijab,ijab->", integrals, amplitudes).scalar
    xp = array_namespace(pairing)
    return Mp2Result(
        reference.energy,
        float(xp.real(pairing)) / (4 * fitted.n_cells),
        amplitudes,
    )

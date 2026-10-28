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
from hafnium.correlated import (
    OrbitalSpaces,
    Space,
    energy_denominators,
    fitted_periodic_tensors,
    transform_df_tensor,
    transform_fitted_tensors,
)
from hafnium.errors import InputError
from hafnium.integrals import DensityFitting, IntegralEngine, density_fitting_k
from hafnium.integrals.density_fitting import DEFAULT_METRIC_THRESHOLD
from hafnium.integrals.periodic.fitting import PeriodicDensityFitting
from hafnium.molecule import Cell, Molecule
from hafnium.scf import ScfResult
from hafnium.structured import BlockTensor, IrrepSectors, contract
from hafnium.symmetry import KPoints, PointGroup, SymmetryAdaptation


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

    The two-electron integrals are fitted in the ``auxiliary`` basis with
    the Coulomb metric, whose eigenvalues below ``threshold`` of the
    largest are dropped (:meth:`~hafnium.integrals.DensityFitting.df_tensor`).
    A molecular reference over irrep sectors adapts the auxiliary basis to
    its group, in whose standard frame the molecule is. A cell takes the
    ``kpoints`` of the reference and fits with momentum transfer
    (ADR-0019). Any Fukutome class serves as the reference; its orbitals
    must be occupied by whole electrons.
    """
    if isinstance(system, Cell):
        if kpoints is None:
            raise InputError(
                "the MP2 energy of a cell needs the k-points of the reference"
            )
        fitting = density_fitting_k(engine, basis, auxiliary, system, kpoints)
        return mp2_from_periodic_fitting(reference, fitting, threshold=threshold)
    if kpoints is not None:
        raise InputError("a molecule has no k-points; drop them")
    sectors = reference.structure.irreps
    adapted = (
        None
        if sectors is None
        else SymmetryAdaptation.from_basis(
            auxiliary, system, PointGroup.named(sectors.group)
        ).irreps()
    )
    molecular = DensityFitting.build(engine, basis, auxiliary, system)
    return mp2_from_df_tensor(
        reference, molecular.df_tensor(threshold=threshold), auxiliary=adapted
    )


def mp2_from_df_tensor(
    reference: ScfResult, df_tensor: Array, *, auxiliary: IrrepSectors | None = None
) -> Mp2Result:
    """Compute the DF-MP2 energy of a molecular ``reference`` from its fitted tensor.

    ``auxiliary`` holds the irrep sectors of the auxiliary basis when the
    reference is blocked over a point group; see :func:`transform_df_tensor`.
    """
    spaces = OrbitalSpaces.from_scf(reference)
    occupied_virtual = transform_df_tensor(
        df_tensor, spaces, Space.OCCUPIED, Space.VIRTUAL, auxiliary=auxiliary
    )
    virtual_occupied = transform_df_tensor(
        df_tensor, spaces, Space.VIRTUAL, Space.OCCUPIED, auxiliary=auxiliary
    )
    return _second_order(reference, spaces, occupied_virtual, virtual_occupied, 1.0)


def mp2_from_periodic_fitting(
    reference: ScfResult,
    fitting: PeriodicDensityFitting,
    *,
    threshold: float = DEFAULT_METRIC_THRESHOLD,
) -> Mp2Result:
    """Compute the DF-MP2 energy per cell of a periodic ``reference``.

    The fitted tensor of ``fitting`` enters the Bloch orbital basis in the
    supercell normalization (:func:`transform_periodic_fitting`), the
    second-order energy of the supercell of the mesh is summed over the
    blocks that conserve crystal momentum, and the energy per cell is
    that divided by the number of k-points.
    """
    spaces = OrbitalSpaces.from_scf(reference)
    tensors = fitted_periodic_tensors(fitting, threshold)
    occupied_virtual = transform_fitted_tensors(
        tensors, spaces, Space.OCCUPIED, Space.VIRTUAL
    )
    virtual_occupied = transform_fitted_tensors(
        tensors, spaces, Space.VIRTUAL, Space.OCCUPIED, signed=True
    )
    return _second_order(
        reference,
        spaces,
        occupied_virtual,
        virtual_occupied,
        1.0 / fitting.kpoints.n_kpt,
    )


def _second_order(
    reference: ScfResult,
    spaces: OrbitalSpaces,
    occupied_virtual: BlockTensor,
    virtual_occupied: BlockTensor,
    per_cell: float,
) -> Mp2Result:
    """Form the second-order energy and amplitudes from the two fitted tensors."""
    integrals = antisymmetrized_integrals(occupied_virtual, virtual_occupied)
    # t2 = <ab||ij> / Δ, the conjugate of <ij||ab> over Δ; E₂ = ¼ Σ <ij||ab> t2.
    excitations = integrals.conj()
    amplitudes = excitations / energy_denominators(spaces, excitations.structure)
    pairing = contract("ijab,ijab->", integrals, amplitudes).scalar
    xp = array_namespace(pairing)
    return Mp2Result(
        reference.energy, per_cell * float(xp.real(pairing)) / 4, amplitudes
    )


def antisymmetrized_integrals(
    occupied_virtual: BlockTensor, virtual_occupied: BlockTensor
) -> BlockTensor:
    """``<ij||ab> = (ia|jb) − (ib|ja)`` as the block tensor ``[i, j, a, b]``.

    ``(ia|jb) = Σ_P B[P, i, a] conj(B[P, b, j])`` pairs the fitted tensor
    between occupied and virtual orbitals with the conjugate of the one
    between virtual and occupied orbitals, ``B[P, j, b] = conj(B[P, b, j])``
    for the real auxiliary functions of a molecule and, with the signature
    of the metric carried by the second tensor, for the Bloch functions of
    a cell; the sum over the auxiliary index then pairs an unconjugated
    with a conjugated axis.
    """
    coulomb = contract("Pia,Pbj->iajb", occupied_virtual, virtual_occupied.conj())
    return coulomb.permute("iajb->ijab") - coulomb.permute("iajb->ijba")

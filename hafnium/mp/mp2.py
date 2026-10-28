"""Second-order Møller–Plesset perturbation theory with density fitting (ADR-0030).

In spin-orbital form the second-order energy of a single determinant is

``E₂ = ¼ Σ_ijab |<ij||ab>|² / (ε_i + ε_j − ε_a − ε_b)``

with the antisymmetrized integrals ``<ij||ab> = (ia|jb) − (ib|ja)`` of
CONVENTIONS.md §3 and ``(ia|jb) = Σ_P B[P, i, a] B[P, j, b]`` from the
fitted tensor in the orbital basis. The sums run over the blocks the
symmetry of the reference allows: for a restricted reference the spin
blocks of one spin follow from the other, for an unrestricted one the
same-spin and opposite-spin blocks are the familiar UMP2 terms, and a
generalized reference has one spinor block. The amplitudes
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
    transform_df_tensor,
)
from hafnium.errors import InputError
from hafnium.integrals import DensityFitting, IntegralEngine
from hafnium.integrals.density_fitting import DEFAULT_METRIC_THRESHOLD
from hafnium.molecule import Molecule
from hafnium.scf import ScfResult
from hafnium.structured import BlockTensor, IrrepSectors, contract
from hafnium.symmetry import PointGroup, SymmetryAdaptation


@dataclass(frozen=True, slots=True, eq=False)
class Mp2Result:
    """The second-order energy of a mean-field reference.

    Parameters
    ----------
    reference_energy
        Total energy of the reference state in Hartree.
    correlation_energy
        The second-order correction in Hartree.
    amplitudes
        ``t2[i, j, a, b] = <ab||ij> / (ε_i + ε_j − ε_a − ε_b)`` as a block
        tensor over the occupied and virtual orbitals of the reference,
        with the index character of the excitation ``a†_a a†_b a_j a_i``:
        the virtual indices conjugated, the occupied ones not.
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
    molecule: Molecule,
    auxiliary: BasisSet,
    reference: ScfResult,
    *,
    threshold: float = DEFAULT_METRIC_THRESHOLD,
) -> Mp2Result:
    """Compute the DF-MP2 energy of a converged Hartree–Fock ``reference``.

    The two-electron integrals are fitted in the ``auxiliary`` basis with
    the Coulomb metric, whose eigenvalues below ``threshold`` of the
    largest are dropped (:meth:`~hafnium.integrals.DensityFitting.df_tensor`).
    A reference over irrep sectors adapts the auxiliary basis to its group,
    in whose standard frame the molecule is. Any Fukutome class serves as
    the reference; its orbitals must be occupied by whole electrons.
    """
    fitting = DensityFitting.build(engine, basis, auxiliary, molecule)
    sectors = reference.structure.irreps
    adapted = (
        None
        if sectors is None
        else SymmetryAdaptation.from_basis(
            auxiliary, molecule, PointGroup.named(sectors.group)
        ).irreps()
    )
    return mp2_from_df_tensor(
        reference, fitting.df_tensor(threshold=threshold), auxiliary=adapted
    )


def mp2_from_df_tensor(
    reference: ScfResult, df_tensor: Array, *, auxiliary: IrrepSectors | None = None
) -> Mp2Result:
    """Compute the DF-MP2 energy of ``reference`` from the fitted tensor.

    ``auxiliary`` holds the irrep sectors of the auxiliary basis when the
    reference is blocked over a point group; see :func:`transform_df_tensor`.
    """
    if reference.structure.k_axis is not None:
        raise InputError("MP2 of a cell comes with M8, part two")
    spaces = OrbitalSpaces.from_scf(reference)
    occupied_virtual = transform_df_tensor(
        df_tensor, spaces, Space.OCCUPIED, Space.VIRTUAL, auxiliary=auxiliary
    )
    virtual_occupied = transform_df_tensor(
        df_tensor, spaces, Space.VIRTUAL, Space.OCCUPIED, auxiliary=auxiliary
    )
    integrals = antisymmetrized_integrals(occupied_virtual, virtual_occupied)
    # t2 = <ab||ij> / Δ, the conjugate of <ij||ab> over Δ; E₂ = ¼ Σ <ij||ab> t2.
    excitations = integrals.conj()
    amplitudes = excitations / energy_denominators(spaces, excitations.structure)
    pairing = contract("ijab,ijab->", integrals, amplitudes).scalar
    xp = array_namespace(pairing)
    return Mp2Result(reference.energy, float(xp.real(pairing)) / 4, amplitudes)


def antisymmetrized_integrals(
    occupied_virtual: BlockTensor, virtual_occupied: BlockTensor
) -> BlockTensor:
    """``<ij||ab> = (ia|jb) − (ib|ja)`` as the block tensor ``[i, j, a, b]``.

    ``(ia|jb) = Σ_P B[P, i, a] conj(B[P, b, j])`` pairs the fitted tensor
    between occupied and virtual orbitals with the conjugate of the one
    between virtual and occupied orbitals, ``B[P, j, b] = conj(B[P, b, j])``
    for the real auxiliary functions of a molecule, so that the sum over
    the auxiliary index pairs an unconjugated with a conjugated axis.
    """
    coulomb = contract("Pia,Pbj->iajb", occupied_virtual, virtual_occupied.conj())
    return coulomb.permute("iajb->ijab") - coulomb.permute("iajb->ijba")

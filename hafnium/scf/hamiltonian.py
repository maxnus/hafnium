"""The Hamiltonian as the self-consistent field consumes it, molecular or periodic."""

from __future__ import annotations

from dataclasses import dataclass, field

from hafnium.backend import Array, array_namespace
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals import (
    IntegralEngine,
    core_hamiltonian,
    core_hamiltonian_k,
    overlap,
    overlap_k,
)
from hafnium.molecule import Cell, Molecule
from hafnium.scf.coulomb_exchange import CoulombExchange, coulomb_exchange_matrices
from hafnium.scf.orthogonalization import (
    DEFAULT_OVERLAP_THRESHOLD,
    canonical_orthogonalizer,
)
from hafnium.structured import BlockMatrix, BlockStructure
from hafnium.symmetry import KPoints, PointGroup, SymmetryAdaptation, SymmetrySpec


@dataclass(frozen=True, slots=True)
class EnergyTerms:
    """Components of a mean-field energy in Hartree, per cell for a periodic system.

    ``E = E_nuc + tr(D h) + tr(D J) / 2 − tr(D K) / 2`` with the structured
    trace (CONVENTIONS.md §3).
    """

    nuclear_repulsion: float
    one_electron: float
    coulomb: float
    exchange: float

    @property
    def electronic(self) -> float:
        """Energy of the electrons in the field of the nuclei."""
        return self.one_electron + self.coulomb + self.exchange

    @property
    def total(self) -> float:
        """Total energy."""
        return self.nuclear_repulsion + self.electronic


@dataclass(frozen=True, slots=True, eq=False)
class Hamiltonian:
    """One-electron integrals, a two-electron builder and the electron counts.

    Parameters
    ----------
    overlap : Array[ao, ao]
        ``[kpt, ao, ao]`` over the full k set of ``kpoints`` for a cell.
    core_hamiltonian : Array[ao, ao]
        Kinetic energy plus nuclear attraction, ``[kpt, ao, ao]`` for a cell.
    two_body
        Representation of the two-electron integrals, over the k-points
        for a cell.
    nuclear_repulsion
        Coulomb repulsion of the nuclei in Hartree, the Ewald sum of a cell.
    n_alpha, n_beta
        Electron counts, per cell for a periodic system.
    kpoints
        The k-point set of a cell; ``None`` for a molecule.
    adaptation
        The symmetry adaptation of the basis to an Abelian point group of
        the molecule, which a specification imposing that group needs
        (ADR-0023); ``None`` when no group is imposed.
    """

    overlap: Array
    core_hamiltonian: Array
    two_body: CoulombExchange
    nuclear_repulsion: float
    n_alpha: int
    n_beta: int
    kpoints: KPoints | None = field(default=None, kw_only=True)
    adaptation: SymmetryAdaptation | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        n = self.two_body.n_ao
        shape = (n, n) if self.kpoints is None else (self.kpoints.n_kpt, n, n)
        if self.overlap.shape != shape or self.core_hamiltonian.shape != shape:
            raise InputError(
                f"the one-electron matrices must have shape {list(shape)} to match "
                "the two-electron integrals and the k-points"
            )
        if self.n_alpha < 0 or self.n_beta < 0:
            raise InputError("electron counts must be non-negative")
        if self.adaptation is not None:
            if self.kpoints is not None:
                raise InputError("a point group is imposed on a molecule, not a cell")
            if self.adaptation.n_functions != n:
                raise InputError(
                    f"the symmetry adaptation covers {self.adaptation.n_functions} "
                    f"functions, the integrals {n}"
                )

    @classmethod
    def from_molecule(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        molecule: Molecule,
        two_body: CoulombExchange,
        *,
        point_group: PointGroup | None = None,
    ) -> Hamiltonian:
        """Evaluate the one-electron integrals of a molecule.

        With ``point_group`` the basis is adapted to the group, which
        requires the molecule in the group's standard frame
        (:meth:`~hafnium.symmetry.PointGroupDetection.orient`).
        """
        adaptation = (
            None
            if point_group is None
            else SymmetryAdaptation.from_basis(basis, molecule, point_group)
        )
        return cls(
            overlap(engine, basis, molecule),
            core_hamiltonian(engine, basis, molecule),
            two_body,
            molecule.nuclear_repulsion_energy(),
            molecule.n_alpha,
            molecule.n_beta,
            adaptation=adaptation,
        )

    @classmethod
    def from_cell(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        cell: Cell,
        kpoints: KPoints,
        two_body: CoulombExchange,
    ) -> Hamiltonian:
        """Evaluate the one-electron integrals of a cell at its k-points.

        The nuclear attraction is the regularized periodic potential of
        ADR-0018 and the nuclear repulsion the Ewald sum of ADR-0012, so the
        energy of a neutral cell is well defined; ``two_body`` carries the
        exchange correction of ADR-0017.
        """
        return cls(
            overlap_k(engine, basis, cell, kpoints),
            core_hamiltonian_k(engine, basis, cell, kpoints),
            two_body,
            cell.nuclear_repulsion_energy(),
            cell.n_alpha,
            cell.n_beta,
            kpoints=kpoints,
        )

    @property
    def n_ao(self) -> int:
        """Number of atomic orbitals."""
        return self.two_body.n_ao

    @property
    def n_electrons(self) -> int:
        """Total number of electrons, per cell for a periodic system."""
        return self.n_alpha + self.n_beta

    def structure(self, symmetry: SymmetrySpec) -> BlockStructure:
        """Block structure of the state ``symmetry`` specifies for this Hamiltonian.

        Raises
        ------
        InputError
            If the specification carries a k-point set other than the
            Hamiltonian's, or imposes a point group the Hamiltonian is not
            adapted to.
        """
        if symmetry.kpoints != self.kpoints:
            raise InputError(
                "the symmetry specification and the Hamiltonian carry different "
                "k-point sets; use SymmetrySpec.with_kpoints with the Hamiltonian's"
            )
        return symmetry.structure(self.adaptation)

    def metric(self, structure: BlockStructure) -> BlockMatrix:
        """Embed the overlap matrix into ``structure``.

        With irrep sectors the matrix is projected onto them, as every
        one-electron matrix is (ADR-0023): the couplings between irreps,
        which vanish for a molecule exactly in the standard frame, are
        bounded by the tolerance of the symmetry adaptation.
        """
        return self._embed_spatial(self.overlap, structure)

    def one_body(self, structure: BlockStructure) -> BlockMatrix:
        """Embed the core Hamiltonian into ``structure``; see :meth:`metric`."""
        return self._embed_spatial(self.core_hamiltonian, structure)

    def orthogonalizer(
        self, structure: BlockStructure, threshold: float = DEFAULT_OVERLAP_THRESHOLD
    ) -> BlockMatrix:
        """Embed the canonical orthogonalizer ``X = S^{-1/2}`` into ``structure``.

        Over k-points the orthogonalizer is built at the stored points of a
        real spatial block, over that block's storage ring: at a mesh whose
        points are all time-reversal invariant the overlap is real and the
        eigendecomposition stays real, otherwise it is complex. The
        eigenvectors of ``S(k)`` and ``S(−k)`` carry independent phases,
        but the conjugate of the orthogonalizer at ``k``, which the
        structure derives at ``−k``, orthogonalizes ``S(−k) = conj(S(k))``
        just as well.
        """
        self._require_matching_sectors(structure)
        overlap = BlockMatrix.from_spatial(
            self.overlap, structure.k_axis, irreps=structure.irreps
        )
        return overlap.map_blocks(
            lambda block: canonical_orthogonalizer(block, threshold)
        ).embed(structure)

    def fock(self, density: BlockMatrix) -> tuple[BlockMatrix, EnergyTerms]:
        """Fock matrix ``h + J − K`` and energy terms of a Hermitian density."""
        one_body = self.one_body(density.structure)
        coulomb, exchange = coulomb_exchange_matrices(self.two_body, density)
        energies = EnergyTerms(
            self.nuclear_repulsion,
            _real_trace(density, one_body),
            _real_trace(density, coulomb) / 2,
            -_real_trace(density, exchange) / 2,
        )
        return one_body + coulomb - exchange, energies

    def _embed_spatial(self, matrix: Array, structure: BlockStructure) -> BlockMatrix:
        self._require_matching_sectors(structure)
        return BlockMatrix.from_spatial(
            matrix, structure.k_axis, irreps=structure.irreps
        ).embed(structure)

    def _require_matching_sectors(self, structure: BlockStructure) -> None:
        expected = None if self.kpoints is None else self.kpoints.axis()
        if structure.k_axis != expected:
            raise InputError(
                "the structure's k-points do not match the Hamiltonian's: build the "
                "structure from a SymmetrySpec with the Hamiltonian's k-point set"
            )
        if structure.irreps is not None and (
            self.adaptation is None or structure.irreps != self.adaptation.irreps()
        ):
            raise InputError(
                "the structure's irrep sectors do not match the Hamiltonian's "
                "symmetry adaptation: build the structure with Hamiltonian.structure"
            )


def _real_trace(density: BlockMatrix, operator: BlockMatrix) -> float:
    """``Re tr(D X)`` as a host float (the JAX milestone revisits the transfer)."""
    xp = array_namespace(*density.blocks)
    return float(xp.real(density.trace_product(operator)))

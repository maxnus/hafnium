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
from hafnium.scf.coulomb_exchange import (
    CoulombExchange,
    coulomb_matrix,
    exchange_matrix,
)
from hafnium.scf.exchange_correlation import ExchangeCorrelation
from hafnium.scf.orthogonalization import (
    DEFAULT_OVERLAP_THRESHOLD,
    canonical_orthogonalizer,
)
from hafnium.structured import BlockMatrix, BlockStructure
from hafnium.symmetry import (
    KPoints,
    PointGroup,
    SpaceGroup,
    SpaceGroupAdaptation,
    SymmetryAdaptation,
    SymmetrySpec,
)


@dataclass(frozen=True, slots=True)
class EnergyTerms:
    """Components of a mean-field energy in Hartree, per cell for a periodic system.

    ``E = E_nuc + tr(D h) + tr(D J) / 2 − tr(D K) / 2 + E_xc`` with the
    structured trace (CONVENTIONS.md §3); for a Kohn–Sham state ``K`` is the
    exact exchange scaled by the functional's fractions, zero for a pure
    functional, and ``E_xc`` the exchange–correlation energy of the
    functional, zero for Hartree–Fock.
    """

    nuclear_repulsion: float
    one_electron: float
    coulomb: float
    exchange: float
    exchange_correlation: float = 0.0

    @property
    def electronic(self) -> float:
        """Energy of the electrons in the field of the nuclei."""
        return (
            self.one_electron + self.coulomb + self.exchange + self.exchange_correlation
        )

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
        the molecule (ADR-0023), or the space-group adaptation of the basis
        of a cell over its k-points (ADR-0024), which a specification
        naming that group needs; ``None`` when no group is imposed.
    exchange_correlation
        The exchange–correlation term of a Kohn–Sham Hamiltonian
        (ADR-0026); ``None`` for Hartree–Fock. Its functional scales the
        exact exchange built from ``two_body``.
    long_range_two_body
        The two-electron integrals with the long-range kernel
        ``erf(ω r) / r`` at the functional's ``omega``, which the long-range
        exact exchange of a range-separated hybrid needs; ``None`` otherwise.
    """

    overlap: Array
    core_hamiltonian: Array
    two_body: CoulombExchange
    nuclear_repulsion: float
    n_alpha: int
    n_beta: int
    kpoints: KPoints | None = field(default=None, kw_only=True)
    adaptation: SymmetryAdaptation | SpaceGroupAdaptation | None = field(
        default=None, kw_only=True
    )
    exchange_correlation: ExchangeCorrelation | None = field(default=None, kw_only=True)
    long_range_two_body: CoulombExchange | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        n = self.two_body.n_ao
        self._check_exchange_correlation(n)
        shape = (n, n) if self.kpoints is None else (self.kpoints.n_kpt, n, n)
        if self.overlap.shape != shape or self.core_hamiltonian.shape != shape:
            raise InputError(
                f"the one-electron matrices must have shape {list(shape)} to match "
                "the two-electron integrals and the k-points"
            )
        if self.n_alpha < 0 or self.n_beta < 0:
            raise InputError("electron counts must be non-negative")
        adaptation = self.adaptation
        if isinstance(adaptation, SymmetryAdaptation) and self.kpoints is not None:
            raise InputError("a point group is imposed on a molecule, not a cell")
        if isinstance(adaptation, SpaceGroupAdaptation) and (
            adaptation.kpoints != self.kpoints
        ):
            raise InputError(
                "the space-group adaptation is built over other k-points than the "
                "Hamiltonian's"
            )
        if adaptation is not None and adaptation.n_functions != n:
            raise InputError(
                f"the symmetry adaptation covers {adaptation.n_functions} "
                f"functions, the integrals {n}"
            )

    def _check_exchange_correlation(self, n: int) -> None:
        xc = self.exchange_correlation
        long_range = self.long_range_two_body
        if xc is None:
            if long_range is not None:
                raise InputError(
                    "long-range two-electron integrals serve the range-separated "
                    "exchange of a functional; pass an exchange–correlation term"
                )
            return
        if xc.kpoints != self.kpoints:
            raise InputError(
                "the exchange–correlation term was built for "
                + ("a molecule" if xc.kpoints is None else "other k-points")
                + "; build it for this Hamiltonian's system and k-points"
            )
        if self.kpoints is not None and xc.functional.is_range_separated:
            raise InputError(
                f"{xc.functional.name} is range-separated, and the long-range "
                "fitted integrals of a cell are not available yet: range-separated "
                "hybrids serve molecules only (ADR-0027)"
            )
        if xc.n_ao != n:
            raise InputError(
                f"the exchange–correlation term covers {xc.n_ao} functions, the "
                f"integrals {n}"
            )
        if xc.functional.is_range_separated:
            if long_range is None:
                raise InputError(
                    f"{xc.functional.name} needs long-range exact exchange: pass the "
                    "two-electron integrals with the long-range kernel at omega "
                    f"{xc.functional.omega:g} as long_range_two_body"
                )
            if long_range.n_ao != n:
                raise InputError(
                    "the long-range two-electron integrals cover another basis than "
                    "the full-range ones"
                )
        elif long_range is not None:
            raise InputError(
                f"{xc.functional.name} has no long-range exact exchange; drop the "
                "long-range two-electron integrals"
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
        exchange_correlation: ExchangeCorrelation | None = None,
        long_range_two_body: CoulombExchange | None = None,
    ) -> Hamiltonian:
        """Evaluate the one-electron integrals of a molecule.

        With ``point_group`` the basis is adapted to the group, which
        requires the molecule in the group's standard frame
        (:meth:`~hafnium.symmetry.PointGroupDetection.orient`). With
        ``exchange_correlation`` the Hamiltonian is a Kohn–Sham one; a
        range-separated functional also needs ``long_range_two_body``.
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
            exchange_correlation=exchange_correlation,
            long_range_two_body=long_range_two_body,
        )

    @classmethod
    def from_cell(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        cell: Cell,
        kpoints: KPoints,
        two_body: CoulombExchange,
        *,
        space_group: SpaceGroup | None = None,
        exchange_correlation: ExchangeCorrelation | None = None,
    ) -> Hamiltonian:
        """Evaluate the one-electron integrals of a cell at its k-points.

        The nuclear attraction is the regularized periodic potential of
        ADR-0018 and the nuclear repulsion the Ewald sum of ADR-0012, so the
        energy of a neutral cell is well defined; ``two_body`` carries the
        exchange correction of ADR-0017. With ``space_group`` the basis is
        adapted to the group over ``kpoints``, which must be closed under
        its point operations (ADR-0024). With ``exchange_correlation``,
        built for the cell over the same ``kpoints``, the Hamiltonian is a
        Kohn–Sham one (ADR-0027).
        """
        adaptation = (
            None
            if space_group is None
            else SpaceGroupAdaptation.from_basis(basis, cell, space_group, kpoints)
        )
        return cls(
            overlap_k(engine, basis, cell, kpoints),
            core_hamiltonian_k(engine, basis, cell, kpoints),
            two_body,
            cell.nuclear_repulsion_energy(),
            cell.n_alpha,
            cell.n_beta,
            kpoints=kpoints,
            adaptation=adaptation,
            exchange_correlation=exchange_correlation,
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
            Hamiltonian's, or names a group the Hamiltonian is not adapted
            to.
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
        bounded by the tolerance of the symmetry adaptation. Over a k axis
        the matrix is likewise projected onto the stored points, each the
        average over its orbit under time reversal and the point operations
        (ADR-0024), which changes nothing for a symmetric cell.
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
            self.overlap, structure.k_axis, irreps=structure.irreps, check=False
        )
        return overlap.map_blocks(
            lambda block: canonical_orthogonalizer(block, threshold)
        ).embed(structure)

    def fock(self, density: BlockMatrix) -> tuple[BlockMatrix, EnergyTerms]:
        """Fock matrix and energy terms of a Hermitian density.

        ``h + J − K`` for Hartree–Fock; for a Kohn–Sham Hamiltonian
        ``h + J − a K − b K_lr + V_xc`` with the fractions ``a`` and ``b`` of
        full-range and long-range exact exchange of the functional, the
        exchange terms left out when their fraction is zero.
        """
        one_body = self.one_body(density.structure)
        coulomb = coulomb_matrix(self.two_body, density)
        fock = one_body + coulomb
        exchange_energy = 0.0
        xc = self.exchange_correlation
        if xc is None:
            fractions: tuple[tuple[float, CoulombExchange | None], ...] = (
                (1.0, self.two_body),
            )
        else:
            fractions = (
                (xc.functional.exact_exchange, self.two_body),
                (xc.functional.long_range_exchange, self.long_range_two_body),
            )
        for fraction, two_body in fractions:
            if fraction == 0.0 or two_body is None:
                continue
            exchange = fraction * exchange_matrix(two_body, density)
            fock = fock - exchange
            exchange_energy -= _real_trace(density, exchange) / 2
        xc_energy = 0.0
        if xc is not None:
            xc_energy, potential = xc.evaluate(density)
            fock = fock + potential
        energies = EnergyTerms(
            self.nuclear_repulsion,
            _real_trace(density, one_body),
            _real_trace(density, coulomb) / 2,
            exchange_energy,
            xc_energy,
        )
        return fock, energies

    def _embed_spatial(self, matrix: Array, structure: BlockStructure) -> BlockMatrix:
        self._require_matching_sectors(structure)
        return BlockMatrix.from_spatial(
            matrix, structure.k_axis, irreps=structure.irreps, check=False
        ).embed(structure)

    def _require_matching_sectors(self, structure: BlockStructure) -> None:
        axis = structure.k_axis
        if (axis is None) != (self.kpoints is None) or (
            axis is not None
            and self.kpoints is not None
            and not axis.same_points(self.kpoints.axis())
        ):
            raise InputError(
                "the structure's k-points do not match the Hamiltonian's: build the "
                "structure from a SymmetrySpec with the Hamiltonian's k-point set"
            )
        adaptation = self.adaptation
        if (
            axis is not None
            and axis.operations is not None
            and (
                not isinstance(adaptation, SpaceGroupAdaptation)
                or axis.operations != adaptation.operations()
            )
        ):
            raise InputError(
                "the structure's point operations do not match the Hamiltonian's "
                "space-group adaptation: build the structure with Hamiltonian.structure"
            )
        if structure.irreps is not None and (
            not isinstance(adaptation, SymmetryAdaptation)
            or structure.irreps != adaptation.irreps()
        ):
            raise InputError(
                "the structure's irrep sectors do not match the Hamiltonian's "
                "symmetry adaptation: build the structure with Hamiltonian.structure"
            )


def _real_trace(density: BlockMatrix, operator: BlockMatrix) -> float:
    """``Re tr(D X)`` as a host float (the JAX milestone revisits the transfer)."""
    xp = array_namespace(*density.blocks)
    return float(xp.real(density.trace_product(operator)))

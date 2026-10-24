"""Coulomb and exchange matrices of a structured density (PLAN.md §5.4).

Only two spatial contractions are needed for every Fukutome class
(CONVENTIONS.md §3): the Coulomb matrix ``J[u, v] = Σ_wx (uv|wx) D[w, x]``
of the total density, the spin trace of the structured density, which is
the same for every spin block; and the exchange matrix
``K[u, v] = Σ_wx (uw|xv) D[w, x]`` of each spin block, applied to the four
spin blocks of a spinor matrix in turn. A :class:`CoulombExchange`
implementation provides the two contractions from its representation of
the two-electron integrals, and :func:`coulomb_exchange_matrices` assembles
the structured matrices from them.

With a k axis the contractions run over the full set of k-points
(ADR-0020): the builder takes ``[kpt, ao, ao]`` arrays, the assembly
materializes the points the structure derives, and the results are
projected back onto the stored k-points in the ring of each block, each
averaged over its orbit under time reversal and the point operations of
the space group (ADR-0024). The Bloch
integrals conjugate the first function of each pair, so
``J[u, v](k) = Σ_k' w_k' Σ_wx (u k v k | w k' x k') D[x, w](k')``. With
irrep sectors the contractions run in the atomic basis (ADR-0023): the
sector blocks of each spin block are assembled into its atomic-basis
matrix, and the results are projected back onto the sectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hafnium.backend import Array, adjoint, array_namespace
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals import (
    DensityFitting,
    IntegralEngine,
    PeriodicDensityFitting,
    density_fitting_k,
    eri,
    madelung_constant,
    overlap_k,
)
from hafnium.integrals.periodic import RangeSeparation
from hafnium.integrals.two_electron import DEFAULT_SCHWARZ_THRESHOLD
from hafnium.molecule import Cell, Molecule
from hafnium.structured import BlockMatrix, Ring, Spin
from hafnium.structured.dense import expand_k, reduce_k
from hafnium.structured.validation import cast_to_ring
from hafnium.symmetry import KPoints


class CoulombExchange(Protocol):
    """Spatial Coulomb and exchange contractions of the two-electron integrals.

    The arrays are ``[ao, ao]`` for a molecular Hamiltonian and
    ``[kpt, ao, ao]`` over the full k set for a periodic one.
    """

    @property
    def n_ao(self) -> int:
        """Number of atomic orbitals."""
        ...

    def coulomb(self, density: Array) -> Array:
        """``J[u, v] = Σ_wx (uv|wx) D[w, x]`` for a spatial ``density``."""
        ...

    def exchange(self, density: Array) -> Array:
        """``K[u, v] = Σ_wx (uw|xv) D[w, x]`` for a spatial ``density``."""
        ...


@dataclass(frozen=True, slots=True, eq=False)
class ExactCoulombExchange:
    """Contractions with the four-centre integrals held in memory.

    Parameters
    ----------
    eri : Array[ao, ao, ao, ao]
        ``(uv|wx)`` in chemists' notation.
    """

    eri: Array

    def __post_init__(self) -> None:
        n = self.eri.shape[0] if self.eri.ndim == 4 else -1
        if self.eri.shape != (n, n, n, n):
            raise InputError("the four-centre integrals have shape [ao, ao, ao, ao]")

    @classmethod
    def from_molecule(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        molecule: Molecule,
        *,
        omega: float | None = None,
        threshold: float = DEFAULT_SCHWARZ_THRESHOLD,
    ) -> ExactCoulombExchange:
        """Evaluate the integrals of a molecule with Schwarz screening.

        With ``omega`` the long-range kernel ``erf(ω r) / r`` replaces
        ``1 / r``, for the long-range exchange of a range-separated hybrid.
        """
        return cls(eri(engine, basis, molecule, omega=omega, threshold=threshold))

    @property
    def n_ao(self) -> int:
        """Number of atomic orbitals."""
        return int(self.eri.shape[0])

    def coulomb(self, density: Array) -> Array:
        """``J[u, v] = Σ_wx (uv|wx) D[w, x]``."""
        xp = array_namespace(self.eri, density)
        return xp.einsum("uvwx,wx->uv", self.eri, density)

    def exchange(self, density: Array) -> Array:
        """``K[u, v] = Σ_wx (uw|xv) D[w, x]``."""
        xp = array_namespace(self.eri, density)
        return xp.einsum("uwxv,wx->uv", self.eri, density)


@dataclass(frozen=True, slots=True, eq=False)
class DensityFittedCoulombExchange:
    """Contractions with the fitted tensor ``B[aux, ao, ao]``.

    With the metric absorbed, ``(uv|wx) ≈ Σ_P B[P, u, v] B[P, w, x]``, so
    ``J = Σ_P B[P] (B[P] · D)`` and ``K = Σ_P B[P] D B[P]``, each two matrix
    products. The exchange contraction works on the density; factorizing it
    through occupied orbitals, which lowers the cost by ``n_ao / n_occ``,
    is a later optimization behind the same interface.

    Parameters
    ----------
    df_tensor : Array[aux, ao, ao]
        The fitted tensor of :meth:`hafnium.integrals.DensityFitting.df_tensor`.
    """

    df_tensor: Array

    def __post_init__(self) -> None:
        shape = self.df_tensor.shape
        if self.df_tensor.ndim != 3 or shape[1] != shape[2]:
            raise InputError("the fitted tensor has shape [aux, ao, ao]")

    @classmethod
    def from_molecule(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        auxiliary: BasisSet,
        molecule: Molecule,
        *,
        omega: float | None = None,
    ) -> DensityFittedCoulombExchange:
        """Fit the integrals of a molecule in the auxiliary basis.

        With ``omega`` the long-range kernel ``erf(ω r) / r`` is fitted with
        its own metric, for the long-range exchange of a range-separated
        hybrid.
        """
        fitting = DensityFitting.build(engine, basis, auxiliary, molecule, omega=omega)
        return cls(fitting.df_tensor())

    @property
    def n_aux(self) -> int:
        """Number of auxiliary functions."""
        return int(self.df_tensor.shape[0])

    @property
    def n_ao(self) -> int:
        """Number of atomic orbitals."""
        return int(self.df_tensor.shape[1])

    def coulomb(self, density: Array) -> Array:
        """``J = Σ_P c_P B[P]`` with the charges ``c_P = Σ_wx B[P, w, x] D[w, x]``."""
        xp = array_namespace(self.df_tensor, density)
        n = self.n_ao
        flat = xp.reshape(self.df_tensor, (self.n_aux, n * n))
        charges = flat @ xp.reshape(density, (n * n,))
        return xp.reshape(charges @ flat, (n, n))

    def exchange(self, density: Array) -> Array:
        """``K = Σ_P B[P] D B[P]``."""
        xp = array_namespace(self.df_tensor, density)
        n = self.n_ao
        half = self.df_tensor @ density
        left = xp.reshape(xp.permute_dims(half, (1, 0, 2)), (n, self.n_aux * n))
        return left @ xp.reshape(self.df_tensor, (self.n_aux * n, n))


@dataclass(frozen=True, slots=True, eq=False)
class KPointDensityFittedCoulombExchange:
    """Contractions with the fitted tensor over k-points and the exchange correction.

    With ``B = df_tensor[k1, k2]`` of momentum transfer ``k2 − k1``
    (ADR-0019), its signs ``s = signature[k1, k2]`` (ADR-0021) and the
    weights ``w`` of the k-points,
    ``J(k) = Σ_P s_P B[k, k, P] Σ_k' w_k' ⟨B[k', k', P], D(k')⟩`` with the
    Frobenius inner product ``⟨A, D⟩ = Σ_xw conj(A[x, w]) D[x, w]``, and
    ``K(k) = Σ_k' w_k' Σ_P s_P B[k, k', P] D(k') B[k, k', P]† + ξ S(k) D(k) S(k)``,
    the last term the probe-charge correction of the exchange divergence
    with the Madelung constant ``ξ`` (ADR-0017).

    Parameters
    ----------
    df_tensor : Array[kpt, kpt, aux, ao, ao]
        The fitted tensor of
        :meth:`hafnium.integrals.PeriodicDensityFitting.df_tensor`.
    signature : Array[kpt, kpt, aux]
        The signs of the metric eigenvalues behind it,
        :meth:`hafnium.integrals.PeriodicDensityFitting.signature`.
    weights : Array[kpt]
        The k-point weights, summing to one.
    overlap : Array[kpt, ao, ao]
        The overlap matrices ``S(k)``.
    madelung
        The Madelung constant of the mesh, :func:`hafnium.integrals.madelung_constant`.
    """

    df_tensor: Array
    signature: Array
    weights: Array
    overlap: Array
    madelung: float

    def __post_init__(self) -> None:
        shape = self.df_tensor.shape
        n_kpt = shape[0] if self.df_tensor.ndim == 5 else -1
        n_ao = shape[3] if self.df_tensor.ndim == 5 else -1
        if self.df_tensor.ndim != 5 or shape[1] != n_kpt or shape[4] != n_ao:
            raise InputError("the fitted tensor has shape [kpt, kpt, aux, ao, ao]")
        if self.signature.shape != shape[:3]:
            raise InputError("the signature has shape [kpt, kpt, aux]")
        if self.weights.shape != (n_kpt,) or self.overlap.shape != (n_kpt, n_ao, n_ao):
            raise InputError("weights have shape [kpt] and the overlap [kpt, ao, ao]")

    @classmethod
    def from_cell(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        auxiliary: BasisSet,
        cell: Cell,
        kpoints: KPoints,
        *,
        separation: RangeSeparation | None = None,
    ) -> KPointDensityFittedCoulombExchange:
        """Fit the integrals of ``cell`` over ``kpoints`` in the auxiliary basis."""
        fitting = density_fitting_k(
            engine, basis, auxiliary, cell, kpoints, separation=separation
        )
        return cls.from_fitting(
            fitting,
            overlap_k(engine, basis, cell, kpoints),
            madelung_constant(cell.lattice, kpoints),
        )

    @classmethod
    def from_fitting(
        cls, fitting: PeriodicDensityFitting, overlap: Array, madelung: float
    ) -> KPointDensityFittedCoulombExchange:
        """Build from a periodic fitting, the overlaps and the Madelung constant."""
        xp = array_namespace(overlap)
        return cls(
            fitting.df_tensor(),
            fitting.signature(),
            fitting.kpoints.weights_array(xp),
            overlap,
            madelung,
        )

    @property
    def n_kpt(self) -> int:
        """Number of k-points."""
        return int(self.df_tensor.shape[0])

    @property
    def n_aux(self) -> int:
        """Number of auxiliary functions."""
        return int(self.df_tensor.shape[2])

    @property
    def n_ao(self) -> int:
        """Number of atomic orbitals."""
        return int(self.df_tensor.shape[3])

    def coulomb(self, density: Array) -> Array:
        """``J(k)`` for the total ``density[kpt, ao, ao]`` over the full k set."""
        xp = array_namespace(self.df_tensor, density)
        diagonal = xp.stack([self.df_tensor[k, k] for k in range(self.n_kpt)])
        signs = xp.stack([self.signature[k, k] for k in range(self.n_kpt)])
        inner = xp.sum(
            xp.conj(diagonal) * xp.expand_dims(density, axis=1), axis=(-2, -1)
        )
        charges = xp.sum(inner * xp.expand_dims(self.weights, axis=1), axis=0)
        signed = xp.astype(signs, xp.complex128) * charges[None, :]
        return xp.sum(diagonal * signed[:, :, None, None], axis=1)

    def exchange(self, density: Array) -> Array:
        """``K(k)`` for one spin block ``density[kpt, ao, ao]`` over the full k set."""
        xp = array_namespace(self.df_tensor, density)
        half = self.df_tensor @ density[None, :, None, :, :]
        signed = half * xp.astype(self.signature, xp.complex128)[..., None, None]
        product = xp.sum(signed @ adjoint(self.df_tensor), axis=2)
        weighted = xp.sum(product * self.weights[None, :, None, None], axis=1)
        return weighted + self.madelung * (self.overlap @ density @ self.overlap)


def coulomb_exchange_matrices(
    two_body: CoulombExchange, density: BlockMatrix
) -> tuple[BlockMatrix, BlockMatrix]:
    """Return ``(J, K)`` in the structure of a Hermitian ``density``.

    :func:`coulomb_matrix` and :func:`exchange_matrix` in one call.
    """
    return coulomb_matrix(two_body, density), exchange_matrix(two_body, density)


def coulomb_matrix(two_body: CoulombExchange, density: BlockMatrix) -> BlockMatrix:
    """Return the Coulomb matrix of a Hermitian ``density`` in its structure.

    It is built from the total density: for a molecule its real part, since
    for a Hermitian density the imaginary part is antisymmetric and drops
    out of the contraction with the symmetric integrals; over k-points the
    full set of the spin trace, projected back onto the stored points.
    """
    _check_basis(two_body, density)
    structure = density.structure
    total = density.spin_trace()
    if structure.k_axis is None:
        sectors = structure.irreps
        total_ao = (
            total.blocks[0]
            if sectors is None
            else sectors.to_atomic(total.blocks, spinor=False, orbitals=False)
        )
        xp = array_namespace(total_ao)
        coulomb_ao = two_body.coulomb(xp.astype(xp.real(total_ao), xp.float64))
        return BlockMatrix.from_spatial(coulomb_ao, irreps=sectors).embed(structure)
    (total_block,) = total.structure.stored_blocks
    total_full = expand_k(
        total.structure, total_block, total.blocks[0], structure.all_kpoints
    )
    xp = array_namespace(total_full)
    coulomb_full = two_body.coulomb(xp.astype(total_full, xp.complex128))
    ring = (
        Ring.COMPLEX
        if any(block.ring is Ring.COMPLEX for block in structure.stored_blocks)
        else Ring.REAL
    )
    return BlockMatrix.from_spatial(
        coulomb_full, structure.k_axis, ring=ring, check=False
    ).embed(structure)


def exchange_matrix(two_body: CoulombExchange, density: BlockMatrix) -> BlockMatrix:
    """Return the exchange matrix of a Hermitian ``density`` in its structure.

    The exchange of a stored block is the contraction of each of its spin
    sub-blocks, over the full k set when there is a k axis, projected back
    onto the block's stored k-points in its ring; with irrep sectors the
    sector blocks of each spin block are assembled into its atomic-basis
    matrix and the results projected back onto the sectors.
    """
    _check_basis(two_body, density)
    structure, n = density.structure, density.n_spatial
    if structure.k_axis is not None:
        return _periodic_exchange(two_body, density)
    sectors = structure.irreps
    if sectors is None:
        exchange = tuple(
            _exchange_block(two_body, block, n) for block in density.blocks
        )
        return BlockMatrix(structure, exchange)
    xp = array_namespace(*density.blocks)
    size = structure.stored_per_sector
    blocks: dict[int, Array] = {}
    for index, block in enumerate(structure.sector_structure.stored_blocks):
        spinor = block.spin is Spin.SPINOR
        parts = [
            density.blocks[sector * size + index] for sector in range(sectors.n_irreps)
        ]
        full = _exchange_block(
            two_body, sectors.to_atomic(parts, spinor=spinor, orbitals=False), n
        )
        for sector, part in enumerate(sectors.from_atomic(full, spinor=spinor)):
            blocks[sector * size + index] = cast_to_ring(
                xp, part, structure.storage_ring(block)
            )
    return BlockMatrix(
        structure, tuple(blocks[position] for position in range(structure.n_stored))
    )


def _check_basis(two_body: CoulombExchange, density: BlockMatrix) -> None:
    n = density.n_spatial
    if n != two_body.n_ao:
        raise InputError(
            f"the density has {n} atomic orbitals, the integrals {two_body.n_ao}"
        )


def _periodic_exchange(two_body: CoulombExchange, density: BlockMatrix) -> BlockMatrix:
    """Contract over the full k set and store the results at the stored points."""
    structure = density.structure
    wanted, n = structure.all_kpoints, density.n_spatial
    xp = array_namespace(*density.blocks)
    exchange: list[Array] = []
    for block, array in zip(structure.stored_blocks, density.blocks, strict=True):
        full = _exchange_block(two_body, expand_k(structure, block, array, wanted), n)
        kept = reduce_k(structure, block, full, check=False, tolerance=0.0)
        exchange.append(cast_to_ring(xp, kept, structure.storage_ring(block)))
    return BlockMatrix(structure, tuple(exchange))


def _exchange_block(two_body: CoulombExchange, block: Array, n: int) -> Array:
    """Exchange of one block, spin sub-block by spin sub-block, for either rank."""
    xp = array_namespace(block)
    m = int(block.shape[-1]) // n
    if m == 1:
        return two_body.exchange(block)

    def sub_block(s: int, t: int) -> Array:
        return block[..., s * n : (s + 1) * n, t * n : (t + 1) * n]

    rows = [
        xp.concat([two_body.exchange(sub_block(s, t)) for t in range(m)], axis=-1)
        for s in range(m)
    ]
    return xp.concat(rows, axis=-2)

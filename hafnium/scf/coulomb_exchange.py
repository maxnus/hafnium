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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hafnium.backend import Array, array_namespace
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals import DensityFitting, IntegralEngine, eri
from hafnium.integrals.two_electron import DEFAULT_SCHWARZ_THRESHOLD
from hafnium.molecule import Molecule
from hafnium.structured import BlockMatrix, BlockStructure


class CoulombExchange(Protocol):
    """Spatial Coulomb and exchange contractions of the two-electron integrals."""

    @property
    def n_ao(self) -> int:
        """Number of atomic orbitals."""
        ...

    def coulomb(self, density: Array) -> Array:
        """``J[u, v] = Σ_wx (uv|wx) D[w, x]`` for a spatial ``density[ao, ao]``."""
        ...

    def exchange(self, density: Array) -> Array:
        """``K[u, v] = Σ_wx (uw|xv) D[w, x]`` for a spatial ``density[ao, ao]``."""
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
        threshold: float = DEFAULT_SCHWARZ_THRESHOLD,
    ) -> ExactCoulombExchange:
        """Evaluate the integrals of a molecule with Schwarz screening."""
        return cls(eri(engine, basis, molecule, threshold=threshold))

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
    ) -> DensityFittedCoulombExchange:
        """Fit the integrals of a molecule in the auxiliary basis."""
        fitting = DensityFitting.build(engine, basis, auxiliary, molecule)
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


def coulomb_exchange_matrices(
    two_body: CoulombExchange, density: BlockMatrix
) -> tuple[BlockMatrix, BlockMatrix]:
    """Return ``(J, K)`` in the structure of a Hermitian ``density``.

    The Coulomb matrix is built from the real part of the total density: for
    a Hermitian density the imaginary part is antisymmetric and drops out of
    the contraction with the symmetric integrals, so ``J`` is real in every
    class and embeds into the structure without a cast. The exchange matrix
    of a stored block is the contraction of each of its spin sub-blocks.
    """
    n = density.n_spatial
    if n != two_body.n_ao:
        raise InputError(
            f"the density has {n} atomic orbitals, the integrals {two_body.n_ao}"
        )
    total = density.spin_trace().blocks[0]
    xp = array_namespace(total)
    coulomb = two_body.coulomb(xp.astype(xp.real(total), xp.float64))
    spatial = BlockMatrix(BlockStructure.spatial(), (coulomb,))
    exchange = tuple(_exchange_block(two_body, block, n) for block in density.blocks)
    return spatial.embed(density.structure), BlockMatrix(density.structure, exchange)


def _exchange_block(two_body: CoulombExchange, block: Array, n: int) -> Array:
    """Exchange of one stored block, spin sub-block by spin sub-block."""
    xp = array_namespace(block)
    m = int(block.shape[0]) // n
    if m == 1:
        return two_body.exchange(block)

    def sub_block(s: int, t: int) -> Array:
        return block[s * n : (s + 1) * n, t * n : (t + 1) * n]

    rows = [
        xp.concat([two_body.exchange(sub_block(s, t)) for t in range(m)], axis=1)
        for s in range(m)
    ]
    return xp.concat(rows, axis=0)

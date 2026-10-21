"""The molecular Hamiltonian as the self-consistent field consumes it."""

from __future__ import annotations

from dataclasses import dataclass

from hafnium.backend import Array, array_namespace
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals import IntegralEngine, core_hamiltonian, overlap
from hafnium.molecule import Molecule
from hafnium.scf.coulomb_exchange import CoulombExchange, coulomb_exchange_matrices
from hafnium.structured import BlockMatrix, BlockStructure


@dataclass(frozen=True, slots=True)
class EnergyTerms:
    """Components of a mean-field energy in Hartree.

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
    core_hamiltonian : Array[ao, ao]
        Kinetic energy plus nuclear attraction.
    two_body
        Representation of the two-electron integrals.
    nuclear_repulsion
        Coulomb repulsion of the nuclei in Hartree.
    n_alpha, n_beta
        Electron counts.
    """

    overlap: Array
    core_hamiltonian: Array
    two_body: CoulombExchange
    nuclear_repulsion: float
    n_alpha: int
    n_beta: int

    def __post_init__(self) -> None:
        n = self.two_body.n_ao
        if self.overlap.shape != (n, n) or self.core_hamiltonian.shape != (n, n):
            raise InputError(
                f"the one-electron matrices must have shape [{n}, {n}] to match "
                "the two-electron integrals"
            )
        if self.n_alpha < 0 or self.n_beta < 0:
            raise InputError("electron counts must be non-negative")

    @classmethod
    def from_molecule(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        molecule: Molecule,
        two_body: CoulombExchange,
    ) -> Hamiltonian:
        """Evaluate the one-electron integrals of a molecule."""
        return cls(
            overlap(engine, basis, molecule),
            core_hamiltonian(engine, basis, molecule),
            two_body,
            molecule.nuclear_repulsion_energy(),
            molecule.n_alpha,
            molecule.n_beta,
        )

    @property
    def n_ao(self) -> int:
        """Number of atomic orbitals."""
        return self.two_body.n_ao

    @property
    def n_electrons(self) -> int:
        """Total number of electrons."""
        return self.n_alpha + self.n_beta

    def metric(self, structure: BlockStructure) -> BlockMatrix:
        """Embed the overlap matrix into ``structure``."""
        return _embed_spatial(self.overlap, structure)

    def one_body(self, structure: BlockStructure) -> BlockMatrix:
        """Embed the core Hamiltonian into ``structure``."""
        return _embed_spatial(self.core_hamiltonian, structure)

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


def _embed_spatial(matrix: Array, structure: BlockStructure) -> BlockMatrix:
    return BlockMatrix(BlockStructure.spatial(), (matrix,)).embed(structure)


def _real_trace(density: BlockMatrix, operator: BlockMatrix) -> float:
    """``Re tr(D X)`` as a host float (the JAX milestone revisits the transfer)."""
    xp = array_namespace(*density.blocks)
    return float(xp.real(density.trace_product(operator)))

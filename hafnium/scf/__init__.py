"""Self-consistent field methods over symmetry-structured matrices.

One driver, :func:`solve_scf`, serves every Fukutome class and, through
the k axis of the structured matrices, every k-point set (ADR-0020): the
:class:`Hamiltonian` supplies the integrals, a :class:`CoulombExchange`
implementation builds the two-electron terms of a structured density, an
occupation policy fills the orbitals, and :class:`Diis` accelerates the
iteration. Guesses are densities: :func:`core_hamiltonian_guess`,
:func:`atomic_density_guess`, or a converged density embedded into another
class. :func:`hartree_fock` wraps the setup for a molecule, and
:func:`spin_expectation` reports ``⟨S⟩`` and ``⟨S²⟩`` of any state.
"""

from hafnium.scf.atoms import atomic_density, atomic_density_guess
from hafnium.scf.coulomb_exchange import (
    CoulombExchange,
    DensityFittedCoulombExchange,
    ExactCoulombExchange,
    KPointDensityFittedCoulomb,
    KPointDensityFittedCoulombExchange,
    coulomb_exchange_matrices,
    coulomb_matrix,
    exchange_matrix,
)
from hafnium.scf.diis import Diis
from hafnium.scf.driver import (
    ScfIteration,
    ScfOptions,
    ScfResult,
    core_hamiltonian_guess,
    orbital_gradient,
    solve_scf,
)
from hafnium.scf.exchange_correlation import (
    DEFAULT_BASIS_CUTOFF,
    DEFAULT_BATCH_SIZE,
    ExchangeCorrelation,
)
from hafnium.scf.hamiltonian import EnergyTerms, Hamiltonian
from hafnium.scf.hartree_fock import hartree_fock
from hafnium.scf.kohn_sham import KohnShamOptions, kohn_sham
from hafnium.scf.occupation import (
    AufbauOccupation,
    AveragedOccupation,
    OccupationPolicy,
    electrons_per_block,
)
from hafnium.scf.orthogonalization import (
    DEFAULT_OVERLAP_THRESHOLD,
    canonical_orthogonalizer,
)
from hafnium.scf.spin import SpinExpectation, spin_expectation

__all__ = [
    "DEFAULT_BASIS_CUTOFF",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_OVERLAP_THRESHOLD",
    "AufbauOccupation",
    "AveragedOccupation",
    "CoulombExchange",
    "DensityFittedCoulombExchange",
    "Diis",
    "EnergyTerms",
    "ExactCoulombExchange",
    "ExchangeCorrelation",
    "Hamiltonian",
    "KPointDensityFittedCoulomb",
    "KPointDensityFittedCoulombExchange",
    "KohnShamOptions",
    "OccupationPolicy",
    "ScfIteration",
    "ScfOptions",
    "ScfResult",
    "SpinExpectation",
    "atomic_density",
    "atomic_density_guess",
    "canonical_orthogonalizer",
    "core_hamiltonian_guess",
    "coulomb_exchange_matrices",
    "coulomb_matrix",
    "electrons_per_block",
    "exchange_matrix",
    "hartree_fock",
    "kohn_sham",
    "orbital_gradient",
    "solve_scf",
    "spin_expectation",
]

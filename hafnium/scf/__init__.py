"""Self-consistent field methods over symmetry-structured matrices.

One driver, :func:`solve_scf`, serves every Fukutome class: the
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
    coulomb_exchange_matrices,
)
from hafnium.scf.diis import Diis
from hafnium.scf.driver import (
    DEFAULT_OVERLAP_THRESHOLD,
    ScfIteration,
    ScfOptions,
    ScfResult,
    canonical_orthogonalizer,
    core_hamiltonian_guess,
    orbital_gradient,
    solve_scf,
)
from hafnium.scf.hamiltonian import EnergyTerms, Hamiltonian
from hafnium.scf.hartree_fock import hartree_fock
from hafnium.scf.occupation import (
    AufbauOccupation,
    AveragedOccupation,
    OccupationPolicy,
    electrons_per_block,
)
from hafnium.scf.spin import SpinExpectation, spin_expectation

__all__ = [
    "DEFAULT_OVERLAP_THRESHOLD",
    "AufbauOccupation",
    "AveragedOccupation",
    "CoulombExchange",
    "DensityFittedCoulombExchange",
    "Diis",
    "EnergyTerms",
    "ExactCoulombExchange",
    "Hamiltonian",
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
    "electrons_per_block",
    "hartree_fock",
    "orbital_gradient",
    "solve_scf",
    "spin_expectation",
]

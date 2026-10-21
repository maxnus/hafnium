"""Hartree–Fock for a molecule in one call."""

from __future__ import annotations

from hafnium.basis import BasisSet
from hafnium.integrals import IntegralEngine
from hafnium.molecule import Molecule
from hafnium.scf.atoms import atomic_density_guess
from hafnium.scf.coulomb_exchange import (
    CoulombExchange,
    DensityFittedCoulombExchange,
    ExactCoulombExchange,
)
from hafnium.scf.driver import ScfOptions, ScfResult, solve_scf
from hafnium.scf.hamiltonian import Hamiltonian
from hafnium.symmetry import SymmetrySpec


def hartree_fock(
    engine: IntegralEngine,
    basis: BasisSet,
    molecule: Molecule,
    *,
    symmetry: SymmetrySpec | None = None,
    auxiliary: BasisSet | None = None,
    options: ScfOptions | None = None,
) -> ScfResult:
    """Converge the Hartree–Fock state of a molecule from atomic densities.

    Parameters
    ----------
    symmetry
        Fukutome class of the state; real RHF by default.
    auxiliary
        Density-fitting basis. Without one the four-centre integrals are
        evaluated exactly and held in memory.

    To run several classes on one molecule, or to start from a guess, build
    the :class:`~hafnium.scf.Hamiltonian` once and call
    :func:`~hafnium.scf.solve_scf` directly.
    """
    two_body: CoulombExchange = (
        ExactCoulombExchange.from_molecule(engine, basis, molecule)
        if auxiliary is None
        else DensityFittedCoulombExchange.from_molecule(
            engine, basis, auxiliary, molecule
        )
    )
    hamiltonian = Hamiltonian.from_molecule(engine, basis, molecule, two_body)
    symmetry = SymmetrySpec() if symmetry is None else symmetry
    guess = atomic_density_guess(engine, basis, symmetry.structure(), options=options)
    return solve_scf(hamiltonian, symmetry, guess=guess, options=options)

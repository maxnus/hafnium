"""Hartree–Fock for a molecule or a cell in one call."""

from __future__ import annotations

from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals import IntegralEngine
from hafnium.molecule import Cell, Molecule
from hafnium.scf.atoms import atomic_density_guess
from hafnium.scf.coulomb_exchange import (
    CoulombExchange,
    DensityFittedCoulombExchange,
    ExactCoulombExchange,
    KPointDensityFittedCoulombExchange,
)
from hafnium.scf.driver import ScfOptions, ScfResult, solve_scf
from hafnium.scf.hamiltonian import Hamiltonian
from hafnium.symmetry import SymmetrySpec


def hartree_fock(
    engine: IntegralEngine,
    basis: BasisSet,
    system: Molecule | Cell,
    *,
    symmetry: SymmetrySpec | None = None,
    auxiliary: BasisSet | None = None,
    options: ScfOptions | None = None,
) -> ScfResult:
    """Converge the Hartree–Fock state of a molecule or a cell from atomic densities.

    Parameters
    ----------
    symmetry
        Fukutome class of the state; real RHF by default. For a cell it
        carries the k-point set, :meth:`SymmetrySpec.with_kpoints`, and
        may name the cell's space group, :meth:`SymmetrySpec.with_space_group`,
        to store the irreducible wedge of the set (ADR-0024); for a
        molecule it may impose an Abelian point group,
        :meth:`SymmetrySpec.with_point_group`, in whose standard frame the
        molecule must be given (ADR-0022, ADR-0023).
    auxiliary
        Density-fitting basis. Without one the four-centre integrals of a
        molecule are evaluated exactly and held in memory; a cell needs one,
        its two-electron terms being fitted (ADR-0007, ADR-0019).

    To run several classes on one system, or to start from a guess, build
    the :class:`~hafnium.scf.Hamiltonian` once and call
    :func:`~hafnium.scf.solve_scf` directly.
    """
    symmetry = SymmetrySpec() if symmetry is None else symmetry
    hamiltonian = _hamiltonian(engine, basis, system, symmetry, auxiliary)
    structure = hamiltonian.structure(symmetry)
    guess = atomic_density_guess(engine, basis, structure, options=options)
    return solve_scf(hamiltonian, symmetry, guess=guess, options=options)


def _hamiltonian(
    engine: IntegralEngine,
    basis: BasisSet,
    system: Molecule | Cell,
    symmetry: SymmetrySpec,
    auxiliary: BasisSet | None,
) -> Hamiltonian:
    if isinstance(system, Cell):
        if symmetry.kpoints is None:
            raise InputError(
                "a cell needs a k-point set: pass SymmetrySpec.with_kpoints(kpoints)"
            )
        if auxiliary is None:
            raise InputError(
                "a cell needs an auxiliary basis: its two-electron terms are "
                "density-fitted"
            )
        two_body = KPointDensityFittedCoulombExchange.from_cell(
            engine, basis, auxiliary, system, symmetry.kpoints
        )
        return Hamiltonian.from_cell(
            engine,
            basis,
            system,
            symmetry.kpoints,
            two_body,
            space_group=symmetry.space_group,
        )
    if symmetry.kpoints is not None or symmetry.space_group is not None:
        raise InputError(
            "a molecule has no k-points or space group; drop them from the symmetry"
        )
    molecular: CoulombExchange = (
        ExactCoulombExchange.from_molecule(engine, basis, system)
        if auxiliary is None
        else DensityFittedCoulombExchange.from_molecule(
            engine, basis, auxiliary, system
        )
    )
    return Hamiltonian.from_molecule(
        engine, basis, system, molecular, point_group=symmetry.point_group
    )

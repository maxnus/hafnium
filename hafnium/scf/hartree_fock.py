"""Hartree–Fock for a molecule or a cell in one call, and the Hamiltonian behind it."""

from __future__ import annotations

from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals import IntegralEngine
from hafnium.integrals.periodic import require_kpoints_for_lattice
from hafnium.molecule import Cell, Molecule
from hafnium.scf.atoms import atomic_density_guess
from hafnium.scf.coulomb_exchange import (
    CoulombExchange,
    DensityFittedCoulombExchange,
    ExactCoulombExchange,
    KPointDensityFittedCoulomb,
    KPointDensityFittedCoulombExchange,
)
from hafnium.scf.driver import ScfOptions, ScfResult, solve_scf
from hafnium.scf.exchange_correlation import ExchangeCorrelation
from hafnium.scf.hamiltonian import Hamiltonian
from hafnium.symmetry import KPoints, SymmetrySpec


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
    hamiltonian = system_hamiltonian(engine, basis, system, symmetry, auxiliary)
    structure = hamiltonian.structure(symmetry)
    guess = atomic_density_guess(engine, basis, structure, options=options)
    return solve_scf(hamiltonian, symmetry, guess=guess, options=options)


def require_periodic_inputs(
    cell: Cell, symmetry: SymmetrySpec, auxiliary: BasisSet | None
) -> tuple[KPoints, BasisSet]:
    """Return the k-points and auxiliary basis a cell's Hamiltonian needs.

    Raises
    ------
    InputError
        Without a k-point set or an auxiliary basis, or with a space group
        that does not map the k-point set onto itself.
    """
    if symmetry.kpoints is None:
        raise InputError(
            "a cell needs a k-point set: pass SymmetrySpec.with_kpoints(kpoints)"
        )
    if auxiliary is None:
        raise InputError(
            "a cell needs an auxiliary basis: its two-electron terms are density-fitted"
        )
    if symmetry.space_group is not None:
        # Fails fast on a k-point set the group does not map onto itself,
        # before the fitted integrals are built.
        symmetry.space_group.kpoint_images(symmetry.kpoints)
    require_kpoints_for_lattice(symmetry.kpoints, cell.lattice)
    return symmetry.kpoints, auxiliary


def system_hamiltonian(
    engine: IntegralEngine,
    basis: BasisSet,
    system: Molecule | Cell,
    symmetry: SymmetrySpec,
    auxiliary: BasisSet | None,
    *,
    exchange_correlation: ExchangeCorrelation | None = None,
) -> Hamiltonian:
    """Build the Hamiltonian of a molecule or a cell for ``symmetry``.

    A cell takes the k-points and, for its space group, the adaptation
    from ``symmetry`` and the fitted two-electron terms in ``auxiliary``;
    see :func:`molecular_hamiltonian` for a molecule. With
    ``exchange_correlation``, built for the same system and k-points, the
    Hamiltonian is a Kohn–Sham one, and a cell with a functional free of
    exact exchange fits the Coulomb term alone (ADR-0028).
    """
    if isinstance(system, Cell):
        kpoints, auxiliary = require_periodic_inputs(system, symmetry, auxiliary)
        two_body: CoulombExchange
        if exchange_correlation is not None and not (
            exchange_correlation.functional.is_hybrid
        ):
            # A pure functional needs the Coulomb term alone: the blocks of
            # zero momentum transfer (ADR-0028).
            two_body = KPointDensityFittedCoulomb.from_cell(
                engine, basis, auxiliary, system, kpoints
            )
        else:
            two_body = KPointDensityFittedCoulombExchange.from_cell(
                engine, basis, auxiliary, system, kpoints
            )
        return Hamiltonian.from_cell(
            engine,
            basis,
            system,
            kpoints,
            two_body,
            space_group=symmetry.space_group,
            exchange_correlation=exchange_correlation,
        )
    return molecular_hamiltonian(
        engine,
        basis,
        system,
        symmetry,
        auxiliary,
        exchange_correlation=exchange_correlation,
    )


def molecular_hamiltonian(
    engine: IntegralEngine,
    basis: BasisSet,
    molecule: Molecule,
    symmetry: SymmetrySpec,
    auxiliary: BasisSet | None,
    *,
    exchange_correlation: ExchangeCorrelation | None = None,
) -> Hamiltonian:
    """Build the Hamiltonian of a molecule for ``symmetry``, exact or density-fitted.

    With ``exchange_correlation`` the Hamiltonian is a Kohn–Sham one; the
    two-electron integrals with the long-range kernel of a range-separated
    functional are built alongside the full-range ones.
    """
    if symmetry.kpoints is not None or symmetry.space_group is not None:
        raise InputError(
            "a molecule has no k-points or space group; drop them from the symmetry"
        )
    functional = (
        None if exchange_correlation is None else exchange_correlation.functional
    )
    omegas: list[float | None] = [None]
    if functional is not None and functional.is_range_separated:
        omegas.append(functional.omega)
    two_body: list[CoulombExchange] = [
        _molecular_two_body(engine, basis, molecule, auxiliary, omega)
        for omega in omegas
    ]
    return Hamiltonian.from_molecule(
        engine,
        basis,
        molecule,
        two_body[0],
        point_group=symmetry.point_group,
        exchange_correlation=exchange_correlation,
        long_range_two_body=two_body[1] if len(two_body) > 1 else None,
    )


def _molecular_two_body(
    engine: IntegralEngine,
    basis: BasisSet,
    molecule: Molecule,
    auxiliary: BasisSet | None,
    omega: float | None,
) -> CoulombExchange:
    if auxiliary is None:
        return ExactCoulombExchange.from_molecule(engine, basis, molecule, omega=omega)
    return DensityFittedCoulombExchange.from_molecule(
        engine, basis, auxiliary, molecule, omega=omega
    )

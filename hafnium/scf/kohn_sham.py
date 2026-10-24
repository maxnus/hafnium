"""Kohn–Sham DFT for a molecule or a cell in one call (M7, ADR-0026, ADR-0027)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.grid import GridSpec
from hafnium.integrals import IntegralEngine
from hafnium.molecule import Cell, Molecule
from hafnium.scf.atoms import atomic_density_guess
from hafnium.scf.driver import ScfOptions, ScfResult, solve_scf
from hafnium.scf.exchange_correlation import (
    DEFAULT_BASIS_CUTOFF,
    DEFAULT_BATCH_SIZE,
    ExchangeCorrelation,
)
from hafnium.scf.hartree_fock import require_periodic_inputs, system_hamiltonian
from hafnium.symmetry import SymmetrySpec
from hafnium.xc import Functional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KohnShamOptions(ScfOptions):
    """The SCF options plus the numerical controls of the functional.

    Parameters
    ----------
    grid
        Sizes of the quadrature grid.
    basis_cutoff
        Basis function values below this are left out of the densities on
        the grid.
    batch_size
        Grid points evaluated at a time; for a cell the points times the
        k-points, which bounds the memory of the Bloch sums.
    """

    grid: GridSpec = field(default_factory=GridSpec)
    basis_cutoff: float = DEFAULT_BASIS_CUTOFF
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        # Named explicitly: zero-argument super() has no __class__ cell in a
        # slots dataclass before Python 3.13.
        ScfOptions.__post_init__(self)
        if not self.basis_cutoff > 0.0:
            raise InputError("the basis cutoff is a positive number")
        if self.batch_size < 1:
            raise InputError("a batch holds at least one point")


def kohn_sham(
    engine: IntegralEngine,
    basis: BasisSet,
    system: Molecule | Cell,
    functional: Functional | str,
    *,
    symmetry: SymmetrySpec | None = None,
    auxiliary: BasisSet | None = None,
    options: ScfOptions | None = None,
) -> ScfResult:
    """Converge the Kohn–Sham state of a molecule or a cell from atomic densities.

    Parameters
    ----------
    functional
        A :class:`~hafnium.xc.Functional`, or one of the names of
        :data:`~hafnium.xc.NAMED_FUNCTIONALS`.
    symmetry
        Fukutome class of the state, real RKS by default; the restricted
        and unrestricted classes are served. For a cell it carries the
        k-point set, :meth:`SymmetrySpec.with_kpoints`, and may name the
        cell's space group (ADR-0024); for a molecule it may impose an
        Abelian point group (ADR-0023).
    auxiliary
        Density-fitting basis for the Coulomb and exact exchange terms;
        without one the four-centre integrals of a molecule are evaluated
        exactly. A cell needs one, its two-electron terms being fitted.
    options
        :class:`KohnShamOptions` choose the grid as well; plain
        :class:`~hafnium.scf.ScfOptions` take the default grid.

    A range-separated functional serves molecules only, the long-range
    fitted integrals of a cell being a later addition (ADR-0027).

    To start from a guess or run several classes, build the
    :class:`~hafnium.scf.Hamiltonian` with an :class:`ExchangeCorrelation`
    term and call :func:`~hafnium.scf.solve_scf` directly.
    """
    if isinstance(functional, str):
        functional = Functional.named(functional)
    symmetry = SymmetrySpec() if symmetry is None else symmetry
    numerical = options if isinstance(options, KohnShamOptions) else KohnShamOptions()
    if isinstance(system, Cell):
        kpoints, _ = require_periodic_inputs(system, symmetry, auxiliary)
        if functional.is_range_separated:
            raise InputError(
                f"{functional.name} is range-separated, and the long-range fitted "
                "integrals of a cell are not available yet: range-separated hybrids "
                "serve molecules only (ADR-0027)"
            )
        exchange_correlation = ExchangeCorrelation.from_cell(
            engine,
            basis,
            system,
            kpoints,
            functional,
            grid=numerical.grid,
            basis_cutoff=numerical.basis_cutoff,
        )
    else:
        exchange_correlation = ExchangeCorrelation.from_molecule(
            engine,
            basis,
            system,
            functional,
            grid=numerical.grid,
            basis_cutoff=numerical.basis_cutoff,
        )
    exchange_correlation = replace(
        exchange_correlation, batch_size=numerical.batch_size
    )
    logger.info(
        "kohn-sham with %s on %d grid points",
        functional.describe(),
        exchange_correlation.grid.n_points,
    )
    hamiltonian = system_hamiltonian(
        engine,
        basis,
        system,
        symmetry,
        auxiliary,
        exchange_correlation=exchange_correlation,
    )
    structure = hamiltonian.structure(symmetry)
    guess = atomic_density_guess(engine, basis, structure, options=options)
    return solve_scf(hamiltonian, symmetry, guess=guess, options=options)

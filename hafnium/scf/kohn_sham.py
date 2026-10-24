"""Kohn–Sham DFT for a molecule in one call (M7, ADR-0026)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.grid import GridSpec
from hafnium.integrals import IntegralEngine
from hafnium.molecule import Molecule
from hafnium.scf.atoms import atomic_density_guess
from hafnium.scf.driver import ScfOptions, ScfResult, solve_scf
from hafnium.scf.exchange_correlation import (
    DEFAULT_BASIS_CUTOFF,
    DEFAULT_BATCH_SIZE,
    ExchangeCorrelation,
)
from hafnium.scf.hartree_fock import molecular_hamiltonian
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
        Grid points evaluated at a time.
    """

    grid: GridSpec = field(default_factory=GridSpec)
    basis_cutoff: float = DEFAULT_BASIS_CUTOFF
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.basis_cutoff > 0.0:
            raise InputError("the basis cutoff is a positive number")
        if self.batch_size < 1:
            raise InputError("a batch holds at least one point")


def kohn_sham(
    engine: IntegralEngine,
    basis: BasisSet,
    molecule: Molecule,
    functional: Functional | str,
    *,
    symmetry: SymmetrySpec | None = None,
    auxiliary: BasisSet | None = None,
    options: ScfOptions | None = None,
) -> ScfResult:
    """Converge the Kohn–Sham state of a molecule from atomic densities.

    Parameters
    ----------
    functional
        A :class:`~hafnium.xc.Functional`, or one of the names of
        :data:`~hafnium.xc.NAMED_FUNCTIONALS`.
    symmetry
        Fukutome class of the state, real RKS by default; the restricted
        and unrestricted classes are served, with an Abelian point group if
        the specification names one (ADR-0023).
    auxiliary
        Density-fitting basis for the Coulomb and exact exchange terms;
        without one the four-centre integrals are evaluated exactly.
    options
        :class:`KohnShamOptions` choose the grid as well; plain
        :class:`~hafnium.scf.ScfOptions` take the default grid.

    To start from a guess or run several classes, build the
    :class:`~hafnium.scf.Hamiltonian` with an :class:`ExchangeCorrelation`
    term and call :func:`~hafnium.scf.solve_scf` directly.
    """
    if isinstance(functional, str):
        functional = Functional.named(functional)
    symmetry = SymmetrySpec() if symmetry is None else symmetry
    numerical = options if isinstance(options, KohnShamOptions) else KohnShamOptions()
    exchange_correlation = ExchangeCorrelation.from_molecule(
        engine,
        basis,
        molecule,
        functional,
        grid=numerical.grid,
        basis_cutoff=numerical.basis_cutoff,
        batch_size=numerical.batch_size,
    )
    logger.info(
        "kohn-sham with %s on %d grid points",
        functional.describe(),
        exchange_correlation.grid.n_points,
    )
    hamiltonian = molecular_hamiltonian(
        engine,
        basis,
        molecule,
        symmetry,
        auxiliary,
        exchange_correlation=exchange_correlation,
    )
    structure = hamiltonian.structure(symmetry)
    guess = atomic_density_guess(engine, basis, structure, options=options)
    return solve_scf(hamiltonian, symmetry, guess=guess, options=options)

"""The periodic fitting and the two-electron builder of a cell, built once."""

from hafnium.basis import BasisSet
from hafnium.integrals import (
    IntegralEngine,
    PeriodicDensityFitting,
    density_fitting_k,
    madelung_constant,
    overlap_k,
)
from hafnium.molecule import Cell
from hafnium.scf import KPointDensityFittedCoulombExchange
from hafnium.symmetry import KPoints


def fitting_and_two_body(
    engine: IntegralEngine,
    basis: BasisSet,
    auxiliary: BasisSet,
    cell: Cell,
    mesh: KPoints,
) -> tuple[PeriodicDensityFitting, KPointDensityFittedCoulombExchange]:
    """Fit the integrals of ``cell`` over ``mesh`` and build the exchange builder."""
    fitting = density_fitting_k(engine, basis, auxiliary, cell, mesh)
    two_body = KPointDensityFittedCoulombExchange.from_fitting(
        fitting,
        overlap_k(engine, basis, cell, mesh),
        madelung_constant(cell.lattice, mesh),
    )
    return fitting, two_body

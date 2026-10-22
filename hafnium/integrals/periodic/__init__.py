"""Lattice sums for cells: image lists, Bloch sums and the Coulomb split.

The image lists and the real-space cutoffs are chosen here, in Python,
from the Gaussian tails of the basis at a precision, and the integrals
over the images run in the compiled layer with the screening of ADR-0018
(ADR-0016). The range separation of the Coulomb kernel and its
reciprocal-space part serve the nuclear attraction and, in the rest of M5,
the density fitting; the probe-charge correction of the exchange
divergence is ADR-0017.
"""

from hafnium.integrals.periodic.bloch import (
    DEFAULT_LATTICE_SUM_PRECISION,
    atom_radii,
    bloch_sum,
    function_absolute_integral,
    function_radius,
    integer_coordinates,
    kinetic_k,
    lattice_images,
    overlap_k,
    reaching_images,
)
from hafnium.integrals.periodic.coulomb import RangeSeparation
from hafnium.integrals.periodic.exchange import madelung_constant
from hafnium.integrals.periodic.fitting import (
    PeriodicDensityFitting,
    density_fitting_k,
)
from hafnium.integrals.periodic.nuclear import (
    core_hamiltonian_k,
    nuclear_attraction_images,
    nuclear_attraction_k,
)

__all__ = [
    "DEFAULT_LATTICE_SUM_PRECISION",
    "PeriodicDensityFitting",
    "RangeSeparation",
    "atom_radii",
    "bloch_sum",
    "core_hamiltonian_k",
    "density_fitting_k",
    "function_absolute_integral",
    "function_radius",
    "integer_coordinates",
    "kinetic_k",
    "lattice_images",
    "madelung_constant",
    "nuclear_attraction_images",
    "nuclear_attraction_k",
    "overlap_k",
    "reaching_images",
]

"""Lattice sums for cells: image lists, Bloch sums and the Coulomb split.

The image lists and the real-space cutoffs are chosen here, in Python,
from the exponents of the basis at a precision (ADR-0016, ADR-0025), and
the integrals over the images run in the compiled layer with the
screening of ADR-0018. The range separation of the Coulomb kernel and its
reciprocal-space part serve the nuclear attraction and the density
fitting, in one to three dimensions (ADR-0021); the probe-charge
correction of the exchange divergence is ADR-0017.
"""

from hafnium.integrals.periodic.bloch import (
    DEFAULT_LATTICE_SUM_PRECISION,
    atom_radii,
    bloch_sum,
    coulomb_reach,
    function_absolute_integral,
    function_radius,
    integer_coordinates,
    kinetic_k,
    lattice_images,
    overlap_k,
    pair_reach,
    periodic_amplitude,
    reaching_images,
    require_kpoints_for_lattice,
)
from hafnium.integrals.periodic.coulomb import (
    PerpendicularProfile,
    RangeSeparation,
    ReciprocalSum,
)
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
    "PerpendicularProfile",
    "RangeSeparation",
    "ReciprocalSum",
    "atom_radii",
    "bloch_sum",
    "core_hamiltonian_k",
    "coulomb_reach",
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
    "pair_reach",
    "periodic_amplitude",
    "reaching_images",
    "require_kpoints_for_lattice",
]

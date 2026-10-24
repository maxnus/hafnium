"""Quadrature grids of DFT (M7).

An atom-centred grid is a radial quadrature (:mod:`hafnium.grid.radial`)
times a Lebedev rule on the sphere (:mod:`hafnium.grid.angular`); the
grids of the atoms are joined by a partition of space
(:mod:`hafnium.grid.partition`) into one :class:`MolecularGrid` whose
sizes a :class:`GridSpec` chooses (ADR-0026). A molecule's grid follows
PySCF's construction (:mod:`hafnium.grid.molecular`); a cell's grid
partitions space among the atoms and their lattice images
(:mod:`hafnium.grid.periodic`, ADR-0027). The basis functions are
evaluated on the points by the compiled layer through the integral
engine's ``basis_values`` and ``bloch_basis_values``.
"""

from hafnium.grid.angular import LEBEDEV_SIZES, lebedev_degree, lebedev_grid
from hafnium.grid.build import build_grid
from hafnium.grid.molecular import (
    BRAGG_RADII,
    DEFAULT_PARTITION_RADIUS,
    GridSpec,
    MolecularGrid,
    atomic_grid,
    build_molecular_grid,
    nwchem_pruning,
)
from hafnium.grid.partition import (
    PARTITION_SCHEMES,
    PartitionScheme,
    becke_partition,
    partition_step,
    partition_weights,
    treutler_adjustment,
)
from hafnium.grid.periodic import (
    SHARED_SHELL_START,
    build_cell_grid,
    nearest_neighbour_distances,
    partition_centers,
)
from hafnium.grid.radial import TREUTLER_XI, mura_knowles, treutler_ahlrichs

__all__ = [
    "BRAGG_RADII",
    "DEFAULT_PARTITION_RADIUS",
    "LEBEDEV_SIZES",
    "PARTITION_SCHEMES",
    "SHARED_SHELL_START",
    "TREUTLER_XI",
    "GridSpec",
    "MolecularGrid",
    "PartitionScheme",
    "atomic_grid",
    "becke_partition",
    "build_cell_grid",
    "build_grid",
    "build_molecular_grid",
    "lebedev_degree",
    "lebedev_grid",
    "mura_knowles",
    "nearest_neighbour_distances",
    "nwchem_pruning",
    "partition_centers",
    "partition_step",
    "partition_weights",
    "treutler_adjustment",
    "treutler_ahlrichs",
]

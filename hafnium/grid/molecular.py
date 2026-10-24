"""Atom-centred grids of a molecule: sizes, pruning and assembly.

A :class:`GridSpec` fixes the radial and angular sizes per element by a
level, the tables of PySCF so that results compare directly; the radial
quadrature is the Treutler–Ahlrichs mapping with the element's scale, the
angular one a Lebedev rule, pruned in the NWChem fashion near the nucleus,
and the atomic grids are joined by Becke's partition with the atomic size
adjustment of Treutler (ADR-0026).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from hafnium.backend import Array, array_namespace, default_namespace, host_array
from hafnium.errors import InputError
from hafnium.grid.angular import LEBEDEV_SIZES, lebedev_degree, lebedev_grid
from hafnium.grid.partition import becke_partition, treutler_adjustment
from hafnium.grid.radial import TREUTLER_XI, treutler_ahlrichs
from hafnium.molecule import ANGSTROM_IN_BOHR, Molecule

BRAGG_RADII: tuple[float, ...] = tuple(
    radius * ANGSTROM_IN_BOHR
    for radius in (
        2.0,
        0.35, 1.40,
        1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,
        1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,
        2.20, 1.80,
        1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35,
        1.30, 1.25, 1.15, 1.15, 1.15, 1.90,
        2.35, 2.00,
        1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55,
        1.55, 1.45, 1.45, 1.40, 1.40, 2.10,
        2.60, 2.15,
        1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,
        1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
        1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50,
        1.90, 1.80, 1.60, 1.90, 1.45, 2.10,
        1.80, 2.15,
        1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    )
)  # fmt: skip
"""Bragg–Slater atomic radii in Bohr, indexed by atomic number (index 0 unused).

Slater, J. Chem. Phys. 41, 3199 (1964), with 0.35 Å for hydrogen as in
PySCF; elements beyond radon take 1.75 Å.
"""

_PERIOD_BOUNDS = (2, 10, 18, 36, 54, 86, 118)
"""Last atomic number of every period."""

_RADIAL_POINTS = (
    (10, 15, 20, 30, 35, 40, 50),
    (30, 40, 50, 60, 65, 70, 75),
    (40, 60, 65, 75, 80, 85, 90),
    (50, 75, 80, 90, 95, 100, 105),
    (60, 90, 95, 105, 110, 115, 120),
    (70, 105, 110, 120, 125, 130, 135),
    (80, 120, 125, 135, 140, 145, 150),
    (90, 135, 140, 150, 155, 160, 165),
    (100, 150, 155, 165, 170, 175, 180),
    (200, 200, 200, 200, 200, 200, 200),
)
"""Radial points ``[level][period]``."""

_ANGULAR_DEGREES = (
    (11, 15, 17, 17, 17, 17, 17),
    (17, 23, 23, 23, 23, 23, 23),
    (23, 29, 29, 29, 29, 29, 29),
    (29, 29, 35, 35, 35, 35, 35),
    (35, 41, 41, 41, 41, 41, 41),
    (41, 47, 47, 47, 47, 47, 47),
    (47, 53, 53, 53, 53, 53, 53),
    (53, 59, 59, 59, 59, 59, 59),
    (59, 59, 59, 59, 59, 59, 59),
    (65, 65, 65, 65, 65, 65, 65),
)
"""Algebraic degree of the Lebedev rule ``[level][period]``."""

_SIZE_OF_DEGREE = {lebedev_degree(size): size for size in LEBEDEV_SIZES}

_PRUNING_RADII = (
    (0.25, 0.5, 1.0, 4.5),
    (0.1667, 0.5, 0.9, 3.5),
    (0.1, 0.4, 0.8, 2.5),
)
"""Boundaries of the five pruning regions in units of the Bragg radius, for
hydrogen and helium, the first row, and heavier elements (Gill, Johnson and
Pople, Chem. Phys. Lett. 209, 506 (1993), as used by NWChem)."""


def period_index(atomic_number: int) -> int:
    """Index of the period of ``atomic_number`` into the size tables (0 to 6)."""
    if not 1 <= atomic_number <= _PERIOD_BOUNDS[-1]:
        raise InputError(f"no grid sizes for atomic number {atomic_number}")
    return sum(atomic_number > bound for bound in _PERIOD_BOUNDS)


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Sizes of the atom-centred grids.

    Parameters
    ----------
    level
        Size level 0 to 9, following PySCF's tables: level 3, the default,
        gives first-row atoms 75 radial points and Lebedev rules of 302
        points, pruned to 50 to 266 points near the nucleus.
    prune
        Reduce the angular rule in the inner regions in the NWChem fashion.
    radial_points, angular_points
        Fixed sizes for every atom instead of the level's; ``angular_points``
        must be the size of a Lebedev rule, :data:`~hafnium.grid.LEBEDEV_SIZES`.
    """

    level: int = 3
    prune: bool = True
    radial_points: int | None = None
    angular_points: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.level < len(_RADIAL_POINTS):
            raise InputError(f"grid levels run from 0 to {len(_RADIAL_POINTS) - 1}")
        if self.radial_points is not None and self.radial_points < 1:
            raise InputError("a grid needs at least one radial point")
        if self.angular_points is not None:
            lebedev_degree(self.angular_points)

    def radial_count(self, atomic_number: int) -> int:
        """Radial points of an atom."""
        if self.radial_points is not None:
            return self.radial_points
        return _RADIAL_POINTS[self.level][period_index(atomic_number)]

    def angular_count(self, atomic_number: int) -> int:
        """Points of the Lebedev rule of an atom before pruning."""
        if self.angular_points is not None:
            return self.angular_points
        return _SIZE_OF_DEGREE[
            _ANGULAR_DEGREES[self.level][period_index(atomic_number)]
        ]


def nwchem_pruning(
    atomic_number: int, radii: Array, angular_points: int
) -> tuple[int, ...]:
    """Lebedev size of every radial shell under the NWChem pruning scheme.

    The five regions bounded by :data:`_PRUNING_RADII` times the Bragg
    radius take rules of 50, 86, the size below ``angular_points``,
    ``angular_points`` and that size again, when ``angular_points`` is
    above 50; a rule of 50 points is pruned to 50, 74, 74, 74, 50, and a
    smaller rule is not pruned.
    """
    xp = array_namespace(radii)
    sizes = LEBEDEV_SIZES[LEBEDEV_SIZES.index(38) :]
    if angular_points < 50:
        return (angular_points,) * int(radii.shape[0])
    if angular_points == 50:
        regions = (50, 74, 74, 74, 50)
    else:
        index = sizes.index(angular_points)
        regions = (50, 86, sizes[index - 1], angular_points, sizes[index - 1])
    group = 0 if atomic_number <= 2 else 1 if atomic_number <= 10 else 2
    bounds = xp.asarray(_PRUNING_RADII[group], dtype=xp.float64)
    scaled = radii / BRAGG_RADII[atomic_number]
    region = xp.sum(xp.expand_dims(scaled, 1) > bounds, axis=1)
    return tuple(regions[int(r)] for r in region)


def atomic_grid(atomic_number: int, spec: GridSpec) -> tuple[Array, Array]:
    """Points ``[n, 3]`` relative to the nucleus and volume weights ``[n]`` of one atom.

    The weights carry ``4π r²`` and the radial and angular weights, so that
    they integrate functions over space before the partition.
    """
    xp = default_namespace()
    radii, radial_weights = treutler_ahlrichs(
        spec.radial_count(atomic_number), TREUTLER_XI[atomic_number]
    )
    angular = spec.angular_count(atomic_number)
    sizes = (
        nwchem_pruning(atomic_number, radii, angular)
        if spec.prune
        else (angular,) * int(radii.shape[0])
    )
    volume = 4.0 * math.pi * radii * radii * radial_weights
    points: list[Array] = []
    weights: list[Array] = []
    for shell, size in enumerate(sizes):
        directions, angular_weights = lebedev_grid(size)
        points.append(radii[shell] * directions)
        weights.append(volume[shell] * angular_weights)
    return xp.concat(points, axis=0), xp.concat(weights, axis=0)


@dataclass(frozen=True, slots=True, eq=False)
class MolecularGrid:
    """Quadrature points of a molecule with their weights.

    Parameters
    ----------
    points : Array[g, 3]
        Positions in Bohr.
    weights : Array[g]
        Quadrature weights including the partition, so that
        ``Σ_g weights[g] f(points[g])`` approximates ``∫ f dr``.
    atoms : Array[g]
        Index of the atom whose grid each point belongs to.
    """

    points: Array
    weights: Array
    atoms: Array

    def __post_init__(self) -> None:
        n = int(self.points.shape[0]) if self.points.ndim == 2 else -1
        if self.points.shape != (n, 3):
            raise InputError("grid points have shape [g, 3]")
        if self.weights.shape != (n,) or self.atoms.shape != (n,):
            raise InputError("grid weights and atom indices have shape [g]")

    @property
    def n_points(self) -> int:
        """Number of points."""
        return int(self.weights.shape[0])

    def integrate(self, values: Array) -> Array:
        """``Σ_g w_g values[..., g]`` as an array without the point axis."""
        xp = array_namespace(values, self.weights)
        return xp.sum(values * self.weights, axis=-1)

    def batches(self, size: int) -> Iterator[slice]:
        """Slices of at most ``size`` consecutive points covering the grid."""
        if size < 1:
            raise InputError("a batch holds at least one point")
        for start in range(0, self.n_points, size):
            yield slice(start, min(start + size, self.n_points))


def build_grid(molecule: Molecule, spec: GridSpec | None = None) -> MolecularGrid:
    """Build the grid of ``molecule`` with the sizes of ``spec``.

    Every atom's grid is centred on it and weighted by Becke's partition
    with Treutler's atomic size adjustment from the Bragg radii.
    """
    spec = GridSpec() if spec is None else spec
    xp = default_namespace()
    centers = molecule.positions_array(xp)
    radii = host_array([BRAGG_RADII[atom.atomic_number] for atom in molecule.atoms])
    adjustments = treutler_adjustment(radii) if molecule.n_atoms > 1 else None
    cache: dict[int, tuple[Array, Array]] = {}
    points: list[Array] = []
    weights: list[Array] = []
    owners: list[Array] = []
    for index, atom in enumerate(molecule.atoms):
        number = atom.atomic_number
        if number not in cache:
            cache[number] = atomic_grid(number, spec)
        local, volume = cache[number]
        positions = local + centers[index]
        if molecule.n_atoms > 1:
            partition = becke_partition(positions, centers, adjustments)[index]
            volume = volume * partition
        points.append(positions)
        weights.append(volume)
        owners.append(xp.full((int(volume.shape[0]),), index, dtype=xp.int64))
    return MolecularGrid(
        xp.concat(points, axis=0), xp.concat(weights, axis=0), xp.concat(owners, axis=0)
    )

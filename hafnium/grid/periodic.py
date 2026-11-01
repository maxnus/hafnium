"""Atom-centred grids of a cell (ADR-0027).

The atoms of the cell keep the atomic grids of a molecule, and the
partition of unity runs over the atoms and their lattice images within
:attr:`~hafnium.grid.GridSpec.partition_radius` of each atom, with the step
of Stratmann, Scuseria and Frisch so that it is local. For a
cell-periodic integrand ``∫_cell f = Σ_A ∫ p_A f`` over all space, one
term per atom of the cell, and each term is the atom's quadrature. Beyond
:data:`SHARED_SHELL_START` of an atom's nearest-neighbour distance the
partition weight is shared with the neighbours and fragments into ridges
between their cells, which the standard Lebedev rules resolve poorly; those
shells take the larger rule of :meth:`GridSpec.shared_angular_count`. Points
whose weight is exactly zero are dropped.
"""

from __future__ import annotations

import math

from hafnium.backend import Array, host_array, host_namespace
from hafnium.errors import InputError
from hafnium.grid.molecular import (
    BRAGG_RADII,
    GridSpec,
    MolecularGrid,
    atomic_grid,
)
from hafnium.grid.partition import partition_weights, treutler_adjustment
from hafnium.molecule import Cell

SHARED_SHELL_START = 0.6
"""Radial shells from this fraction of the nearest-neighbour distance outwards
take the shared-shell angular rule."""


def partition_centers(cell: Cell, radius: float) -> tuple[Array, Array]:
    """Return the atoms of ``cell`` and their images within ``radius`` of an atom.

    Positions ``[centre, 3]`` and atomic numbers ``[centre]``; the atoms of
    the cell itself come first, in their order, followed by the images.
    """
    if not radius > 0.0:
        raise InputError("the partition radius is a positive length")
    xp = host_namespace()
    centers = cell.positions_array(xp)
    differences = xp.expand_dims(centers, 1) - xp.expand_dims(centers, 0)
    extent = float(xp.max(xp.linalg.vector_norm(differences, axis=-1)))
    translations = cell.lattice.translations(radius + extent)
    positions = xp.reshape(
        xp.expand_dims(centers, 0) + xp.expand_dims(translations, 1), (-1, 3)
    )
    numbers = xp.tile(
        xp.asarray(cell.atomic_numbers, dtype=xp.int64), (int(translations.shape[0]),)
    )
    distances = xp.linalg.vector_norm(
        xp.expand_dims(positions, 1) - xp.expand_dims(centers, 0), axis=-1
    )
    within = xp.any(distances <= radius, axis=1)
    return positions[within], numbers[within]


def nearest_neighbour_distances(cell: Cell) -> Array:
    """Distance ``[atom]`` from every atom of ``cell`` to its nearest other centre.

    The other centre is an atom of the cell or a lattice image of any atom,
    the atom's own images included.
    """
    xp = host_namespace()
    centers = cell.positions_array(xp)
    lengths = xp.linalg.vector_norm(cell.lattice.vectors_array(xp), axis=-1)
    positions, _ = partition_centers(cell, float(xp.max(lengths)) * 2.0)
    distances = xp.linalg.vector_norm(
        xp.expand_dims(positions, 1) - xp.expand_dims(centers, 0), axis=-1
    )
    others = xp.where(distances > 0.0, distances, xp.full(distances.shape, math.inf))
    return xp.min(others, axis=0)


def build_cell_grid(cell: Cell, spec: GridSpec | None = None) -> MolecularGrid:
    """Build the grid of ``cell`` with the sizes of ``spec``.

    Every atom's grid is centred on it and weighted by its partition
    weight among the atoms and images within ``spec.partition_radius``,
    with the Stratmann step unless ``spec.partition`` says otherwise and
    with Treutler's atomic size adjustment from the Bragg radii.
    """
    spec = GridSpec() if spec is None else spec
    scheme = "stratmann" if spec.partition is None else spec.partition
    xp = host_namespace()
    centers = cell.positions_array(xp)
    positions, numbers = partition_centers(cell, spec.partition_radius)
    nearest = nearest_neighbour_distances(cell)
    cache: dict[tuple[int, float], tuple[Array, Array]] = {}
    points: list[Array] = []
    weights: list[Array] = []
    owners: list[Array] = []
    for index, atom in enumerate(cell.atoms):
        number = atom.atomic_number
        shared_from = SHARED_SHELL_START * float(nearest[index])
        key = (number, shared_from)
        if key not in cache:
            cache[key] = atomic_grid(number, spec, shared_from=shared_from)
        local, volume = cache[key]
        atom_points = local + centers[index]
        within = (
            xp.linalg.vector_norm(positions - centers[index], axis=-1)
            <= spec.partition_radius
        )
        set_centers = positions[within]
        if int(set_centers.shape[0]) > 1:
            radii = host_array([BRAGG_RADII[int(z)] for z in numbers[within]])
            # The cell's atoms lead the centre list, so the owner's position
            # among the kept centres is the number of kept atoms before it.
            owner = int(xp.sum(xp.astype(within[:index], xp.int64)))
            partition = partition_weights(
                atom_points,
                set_centers,
                owner,
                treutler_adjustment(radii),
                scheme=scheme,
            )
            volume = volume * partition
        keep = volume != 0.0
        points.append(atom_points[keep])
        weights.append(volume[keep])
        owners.append(xp.full((int(weights[-1].shape[0]),), index, dtype=xp.int64))
    return MolecularGrid(
        xp.concat(points, axis=0), xp.concat(weights, axis=0), xp.concat(owners, axis=0)
    )

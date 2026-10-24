"""Becke's partition of space between the atoms of a molecule.

Every grid point receives a weight per atom, the atom's cell function
normalized over the atoms, so that the atomic grids add up to one
quadrature of the whole molecule (Becke, J. Chem. Phys. 88, 2547 (1988)).
"""

from __future__ import annotations

from hafnium.backend import Array, array_namespace
from hafnium.errors import InputError

_SMOOTHING_STEPS = 3
"""Applications of Becke's polynomial ``f(x) = (3x − x³)/2``."""


def treutler_adjustment(radii: Array) -> Array:
    """Atomic size adjustments ``a[A, B]`` from the atoms' radii ``[atom]``.

    Becke's adjustment of the cell boundary towards the smaller atom,
    ``ν = μ + a (1 − μ²)``, with ``a_AB = (√(R_B/R_A) − √(R_A/R_B)) / 4``
    clipped to ``[−½, ½]``, as Treutler and Ahlrichs apply it
    (J. Chem. Phys. 102, 346 (1995)).
    """
    xp = array_namespace(radii)
    if radii.ndim != 1 or bool(xp.any(radii <= 0.0)):
        raise InputError("atomic radii are positive, one per atom")
    root = xp.sqrt(radii)
    ratio = xp.expand_dims(root, 1) / xp.expand_dims(root, 0)
    return xp.clip(0.25 * (xp.linalg.matrix_transpose(ratio) - ratio), -0.5, 0.5)


def becke_partition(
    points: Array, centers: Array, adjustments: Array | None = None
) -> Array:
    """Return the normalized cell functions ``[atom, g]`` of the points.

    ``points`` are ``[g, 3]`` and ``centers`` ``[atom, 3]``.

    For every pair of atoms the confocal elliptical coordinate
    ``μ_AB = (r_A − r_B) / R_AB`` of a point, adjusted by ``adjustments[A, B]``
    when given, enters the step ``s(ν) = (1 − f(f(f(ν)))) / 2``; the cell
    function of atom ``A`` is the product over ``B ≠ A``, and the weights are
    the cell functions divided by their sum, so they add up to one at every
    point.
    """
    xp = array_namespace(points, centers)
    if points.ndim != 2 or points.shape[1] != 3:
        raise InputError("points have shape [g, 3]")
    n_atoms = int(centers.shape[0]) if centers.ndim == 2 else -1
    if centers.shape != (n_atoms, 3) or n_atoms < 1:
        raise InputError("centres have shape [atom, 3]")
    if adjustments is not None and adjustments.shape != (n_atoms, n_atoms):
        raise InputError("adjustments have shape [atom, atom]")
    differences = xp.expand_dims(points, 0) - xp.expand_dims(centers, 1)
    distances = xp.sqrt(xp.sum(differences * differences, axis=-1))
    separation = xp.sqrt(
        xp.sum((xp.expand_dims(centers, 1) - xp.expand_dims(centers, 0)) ** 2, axis=-1)
    )
    eye = xp.eye(n_atoms, dtype=xp.bool)
    if bool(xp.any(separation[~eye] == 0.0)):
        raise InputError("two atoms coincide")
    safe = xp.where(eye, xp.ones_like(separation), separation)
    cells: list[Array] = []
    for atom in range(n_atoms):
        mu = (distances[atom][None, :] - distances) / safe[atom][:, None]
        if adjustments is not None:
            mu = mu + adjustments[atom][:, None] * (1.0 - mu * mu)
        step = 0.5 * (1.0 - _smoothed(mu))
        own = eye[atom][:, None]
        cells.append(xp.prod(xp.where(own, xp.ones_like(step), step), axis=0))
    stacked = xp.stack(cells, axis=0)
    return stacked / xp.sum(stacked, axis=0)


def _smoothed(mu: Array) -> Array:
    for _ in range(_SMOOTHING_STEPS):
        mu = 0.5 * mu * (3.0 - mu * mu)
    return mu

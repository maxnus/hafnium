"""Partition of space between the atoms for the atom-centred grids.

Every grid point receives a weight per atom, the atom's cell function
normalized over the atoms, so that the atomic grids add up to one
quadrature of the whole system. Two step functions join the cell
functions: Becke's (J. Chem. Phys. 88, 2547 (1988)), which molecules use,
and the compactly supported one of Stratmann, Scuseria and Frisch (Chem.
Phys. Lett. 257, 213 (1996)), which cells use so that an atom's weight
vanishes exactly near every neighbour and far away (ADR-0027).
:func:`becke_partition` is the array-code implementation over the few
centres of a molecule; :func:`partition_weights` calls the compiled kernel
for the many centres of a cell's atom and its lattice images.
"""

from __future__ import annotations

from typing import Literal

from hafnium import _kernels
from hafnium.backend import Array, array_namespace, host_array, to_host
from hafnium.errors import InputError

PartitionScheme = Literal["becke", "stratmann"]
PARTITION_SCHEMES: tuple[PartitionScheme, ...] = ("becke", "stratmann")

_SMOOTHING_STEPS = 3
"""Applications of Becke's polynomial ``f(x) = (3x − x³)/2``."""

STRATMANN_A = 0.64
"""Half-width in ``ν`` of the transition of the Stratmann–Scuseria–Frisch step."""


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
    points: Array,
    centers: Array,
    adjustments: Array | None = None,
    *,
    scheme: PartitionScheme = "becke",
) -> Array:
    """Return the normalized cell functions ``[atom, g]`` of the points.

    ``points`` are ``[g, 3]`` and ``centers`` ``[atom, 3]``.

    For every pair of atoms the confocal elliptical coordinate
    ``μ_AB = (r_A − r_B) / R_AB`` of a point, adjusted by ``adjustments[A, B]``
    when given, enters the step ``s(ν)`` of ``scheme``: Becke's
    ``(1 − f(f(f(ν)))) / 2``, or the polynomial of Stratmann, Scuseria and
    Frisch, which is one for ``ν ≤ −0.64`` and zero for ``ν ≥ 0.64``. The
    cell function of atom ``A`` is the product over ``B ≠ A``, and the
    weights are the cell functions divided by their sum, so they add up to
    one at every point.
    """
    xp = array_namespace(points, centers)
    _check_scheme(scheme)
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
        step = partition_step(mu, scheme)
        own = eye[atom][:, None]
        cells.append(xp.prod(xp.where(own, xp.ones_like(step), step), axis=0))
    stacked = xp.stack(cells, axis=0)
    return stacked / xp.sum(stacked, axis=0)


def partition_step(nu: Array, scheme: PartitionScheme) -> Array:
    """Return the step ``s(ν)`` of ``scheme``, one where the point is the first atom's.

    Zero where it is the second atom's.
    """
    xp = array_namespace(nu)
    _check_scheme(scheme)
    if scheme == "becke":
        return 0.5 * (1.0 - _smoothed(nu))
    x = xp.clip(nu / STRATMANN_A, -1.0, 1.0)
    x2 = x * x
    return 0.5 * (1.0 - x * (35.0 + x2 * (-35.0 + x2 * (21.0 - 5.0 * x2))) / 16.0)


def partition_weights(
    points: Array,
    centers: Array,
    owner: int,
    adjustments: Array | None = None,
    *,
    scheme: PartitionScheme = "becke",
) -> Array:
    """Return the weights ``[g]`` of the centre ``owner`` at ``points``.

    The row ``owner`` of :func:`becke_partition` from the compiled kernel,
    which serves the hundreds of centres of a periodic partition: it
    accumulates every cell function from the nearest centre outwards, so
    that with the Stratmann step a far centre is settled by its first
    factor.
    """
    _check_scheme(scheme)
    if points.ndim != 2 or points.shape[1] != 3:
        raise InputError("points have shape [g, 3]")
    n_centers = int(centers.shape[0]) if centers.ndim == 2 else -1
    if centers.shape != (n_centers, 3) or n_centers < 1:
        raise InputError("centres have shape [atom, 3]")
    if not 0 <= owner < n_centers:
        raise InputError("the owner is the index of one of the centres")
    if adjustments is not None and adjustments.shape != (n_centers, n_centers):
        raise InputError("adjustments have shape [atom, atom]")
    try:
        weights = _kernels.partition_weights(
            to_host(points),
            to_host(centers),
            owner,
            None if adjustments is None else to_host(adjustments),
            scheme,
        )
    except ValueError as error:
        raise InputError(str(error)) from error
    return host_array(weights)


def _check_scheme(scheme: str) -> None:
    if scheme not in PARTITION_SCHEMES:
        raise InputError(
            f"the partition scheme is one of {', '.join(PARTITION_SCHEMES)}, "
            f"not {scheme!r}"
        )


def _smoothed(mu: Array) -> Array:
    for _ in range(_SMOOTHING_STEPS):
        mu = 0.5 * mu * (3.0 - mu * mu)
    return mu

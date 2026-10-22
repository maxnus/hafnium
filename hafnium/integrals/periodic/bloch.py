"""Bloch sums of one-electron integrals over lattice images (ADR-0016).

The basis functions of a cell are the Bloch sums
``φ_μ^k(r) = Σ_T exp(i k·T) φ_μ(r − T)`` (CONVENTIONS.md §3, ADR-0007), so
a one-electron matrix at ``k`` is the lattice sum
``M(k) = Σ_T exp(i k·T) M_T`` of the matrices ``M_T`` between the cell at
the origin and its image translated by ``T``, which the integral engine
evaluates for a list of translations. The Gaussian tails of the basis
functions bound the images that contribute: an image is kept while some
atom of the cell and some atom of the image lie within the sum of their
radii, beyond which every function of an atom is below a precision.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from hafnium.backend import Array, array_namespace, default_namespace
from hafnium.basis import BasisSet, Shell
from hafnium.errors import InputError
from hafnium.integrals.engine import IntegralEngine
from hafnium.integrals.shell_table import ShellTable, require_basis_for_atoms
from hafnium.molecule import Cell, Lattice
from hafnium.symmetry import KPoints

DEFAULT_LATTICE_SUM_PRECISION = 1e-12
"""Value below which a basis function counts as zero when the images of a
lattice sum are chosen; a neglected image contributes at most of that order."""

_RADIUS_BISECTIONS = 60
"""Bisection steps of the radius equation, bringing the bracket to ``2^-60``."""


def function_radius(shell: Shell, precision: float) -> float:
    """Return the radius beyond which ``shell`` falls below ``precision``.

    A contracted function is bounded by ``Y_l Σ_i |c_i| r^l exp(−α_i r²)``
    with the bare-primitive coefficients ``c_i`` and
    ``Y_l = sqrt((2l + 1) / 4π)``, the largest value a unit-norm real solid
    harmonic takes on the sphere. Each term is driven below
    ``precision / n`` for ``n`` primitives, and the radius of the shell is
    the largest of the primitive radii.
    """
    if not (math.isfinite(precision) and precision > 0.0):
        raise InputError("the lattice-sum precision is a positive number")
    l = shell.angular_momentum
    angular = math.sqrt((2 * l + 1) / (4.0 * math.pi))
    rows = shell.primitive_coefficients()
    radius = 0.0
    for index, alpha in enumerate(shell.exponents):
        amplitude = angular * shell.n_primitives * max(abs(row[index]) for row in rows)
        if amplitude > 0.0:
            radius = max(radius, _primitive_radius(alpha, l, amplitude / precision))
    return radius


def _primitive_radius(alpha: float, l: int, ratio: float) -> float:
    """Return the largest ``r`` with ``ratio r^l exp(−α r²) = 1``, at least ``1/√α``.

    Beyond the maximum of ``r^l exp(−α r²)`` at ``sqrt(l / 2α)`` the
    function decreases, so the outer root is bracketed by doubling from
    there and found by bisection; the upper end of the bracket is returned.
    The floor ``1/√α`` keeps the radius at the natural scale of the
    primitive when the function never reaches the precision.
    """

    def excess(r: float) -> float:
        return math.log(ratio) + l * math.log(r) - alpha * r * r

    low = max(1.0 / math.sqrt(alpha), math.sqrt(l / (2.0 * alpha)))
    if excess(low) <= 0.0:
        return low
    high = 2.0 * low
    while excess(high) > 0.0:
        high *= 2.0
    for _ in range(_RADIUS_BISECTIONS):
        middle = 0.5 * (low + high)
        if excess(middle) > 0.0:
            low = middle
        else:
            high = middle
    return high


def atom_radii(basis: BasisSet, precision: float) -> tuple[float, ...]:
    """Return the radius of every atom: the largest radius among its shells."""
    radii = [0.0] * basis.n_atoms
    for shell, atom in zip(basis.shells, basis.shell_atoms, strict=True):
        radii[atom] = max(radii[atom], function_radius(shell, precision))
    return tuple(radii)


def lattice_images(
    basis: BasisSet, cell: Cell, *, precision: float = DEFAULT_LATTICE_SUM_PRECISION
) -> Array:
    """Return the translations ``[image, 3]`` in Bohr over which lattice sums run.

    An image ``T`` is kept when some atom ``A`` of the cell and some atom
    ``B`` of the image are within the sum of their radii,
    ``|A − B − T| ≤ r_A + r_B``, where every function of an atom is below
    ``precision`` beyond its radius (:func:`function_radius`). The zero
    translation comes first, the rest follow by length, and ``−T`` is kept
    with ``T``, so Bloch sums over the result are Hermitian.
    """
    require_basis_for_atoms(basis, cell)
    xp = default_namespace()
    radii = xp.asarray(atom_radii(basis, precision), dtype=xp.float64)
    positions = cell.positions_array(xp)
    separations = xp.expand_dims(positions, 1) - xp.expand_dims(positions, 0)
    diameter = float(xp.max(xp.linalg.vector_norm(separations, axis=-1)))
    candidates = cell.lattice.translations(2.0 * float(xp.max(radii)) + diameter)
    reach = xp.expand_dims(radii, 1) + xp.expand_dims(radii, 0)
    keep = xp.zeros(candidates.shape[0], dtype=xp.bool)
    for atom in range(cell.n_atoms):
        distance = xp.linalg.vector_norm(
            xp.expand_dims(separations[atom], 0) - xp.expand_dims(candidates, 1),
            axis=-1,
        )
        keep = keep | xp.any(distance <= xp.expand_dims(reach[atom], 0), axis=1)
    return candidates[keep]


def bloch_sum(
    images: Array, translations: Array, kpoints: KPoints, lattice: Lattice
) -> Array:
    """Return ``Σ_T exp(i k·T) images[T]`` at every k-point as complex ``[kpt, ...]``.

    ``images`` holds one array per translation along its first axis, as the
    engine returns them, and ``translations`` is ``[image, 3]`` in Bohr.
    """
    xp = array_namespace(images, translations)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise InputError("translations have shape [image, 3]")
    if images.shape[:1] != translations.shape[:1]:
        raise InputError("a Bloch sum takes one image per translation")
    kpts = xp.asarray(kpoints.cartesian(lattice))
    angles = kpts @ xp.linalg.matrix_transpose(translations)
    phases = xp.exp(1j * xp.astype(angles, xp.complex128))
    return xp.tensordot(phases, xp.astype(images, xp.complex128), axes=1)


def overlap_k(
    engine: IntegralEngine,
    basis: BasisSet,
    cell: Cell,
    kpoints: KPoints,
    *,
    precision: float = DEFAULT_LATTICE_SUM_PRECISION,
) -> Array:
    """Overlap matrices ``S(k)`` as complex ``[kpt, ao, ao]`` over ``kpoints``.

    Code that needs several matrices of one cell calls
    :func:`lattice_images` once and :func:`bloch_sum` on each engine
    result, so that one image list serves them all.
    """
    return _lattice_sum(engine.overlap, basis, cell, kpoints, precision)


def kinetic_k(
    engine: IntegralEngine,
    basis: BasisSet,
    cell: Cell,
    kpoints: KPoints,
    *,
    precision: float = DEFAULT_LATTICE_SUM_PRECISION,
) -> Array:
    """Kinetic matrices ``T(k)`` as complex ``[kpt, ao, ao]`` over ``kpoints``.

    See :func:`overlap_k` for sharing the image list between matrices.
    """
    return _lattice_sum(engine.kinetic, basis, cell, kpoints, precision)


def _lattice_sum(
    kernel: Callable[[ShellTable, ShellTable, Array], Array],
    basis: BasisSet,
    cell: Cell,
    kpoints: KPoints,
    precision: float,
) -> Array:
    """Bloch sums of a one-electron ``kernel`` over the images of ``cell``."""
    table = ShellTable.from_basis(basis, cell)
    translations = lattice_images(basis, cell, precision=precision)
    return bloch_sum(
        kernel(table, table, translations), translations, kpoints, cell.lattice
    )

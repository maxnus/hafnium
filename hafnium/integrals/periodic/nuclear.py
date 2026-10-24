"""The nuclear attraction of a cell: the regularized periodic potential.

The nuclei and their images carry the neutralizing background of
ADR-0012, and their potential is split by the range separation of
:class:`RangeSeparation`. The short-range part,
``Σ_{A,L} −Z_A erfc(ω|r − A − L|) / |r − A − L|``, is summed in real space
over the nuclear images ``L`` that reach the basis functions, with the
kernel's screening (ADR-0018). The long-range part is the plane-wave
expansion of the periodic potential, ``(1/Ω) Σ_G ṽ(G) S(G) exp(iG·r)``
with the structure factor ``S(G) = Σ_A −Z_A exp(−iG·A)`` and the kernel
``ṽ`` of :meth:`RangeSeparation.reciprocal_sum`, whose matrix elements
are ``(1/Ω) Σ_G ṽ(G) S(−G) ρ̃_T(G)`` over the pair transforms
``ρ̃_T(G) = ∫ φ_u(r) φ_v(r − T) exp(−iG·r) dr`` of the engine; the ``G = 0``
term carries the finite part of the kernel, which makes the matrices
independent of ``omega`` on their own. For a slab or a chain the sum over
``G`` runs over the vectors and weights of
:meth:`RangeSeparation.reciprocal_sum`, the quadrature over the open
directions of ADR-0021. The matrices at ``k`` are Bloch sums over the ket
images ``T`` of :func:`lattice_images`.
"""

from __future__ import annotations

from hafnium.backend import Array, array_namespace, default_namespace
from hafnium.basis import BasisSet
from hafnium.integrals.engine import CoulombKernel, IntegralEngine, PointCharges
from hafnium.integrals.periodic.bloch import (
    DEFAULT_LATTICE_SUM_PRECISION,
    atom_radii,
    bloch_sum,
    lattice_images,
    reaching_images,
    require_kpoints_for_lattice,
)
from hafnium.integrals.periodic.coulomb import (
    PerpendicularProfile,
    RangeSeparation,
)
from hafnium.integrals.shell_table import ShellTable
from hafnium.molecule import Cell
from hafnium.symmetry import KPoints

_TRANSFORM_CHUNK_BYTES = 128 * 2**20
"""Memory budget of one batch of pair transforms ``[image, G, ao, ao]``."""


def nuclear_attraction_k(
    engine: IntegralEngine,
    basis: BasisSet,
    cell: Cell,
    kpoints: KPoints,
    *,
    omega: float | None = None,
    precision: float = DEFAULT_LATTICE_SUM_PRECISION,
) -> Array:
    """Nuclear attraction matrices ``V(k)`` as complex ``[kpt, ao, ao]``.

    The potential of the nuclei and their images is regularized by the
    uniform background of ADR-0012, so the matrices are those of a neutral
    cell's electron–nuclear term; ``omega`` is the range-separation
    parameter, which the result does not depend on beyond ``precision``.
    See :func:`overlap_k` for sharing the image list between matrices.
    """
    require_kpoints_for_lattice(kpoints, cell.lattice)
    separation = RangeSeparation.for_lattice(
        cell.lattice, omega=omega, precision=precision
    )
    translations = lattice_images(basis, cell, precision=precision)
    images = nuclear_attraction_images(engine, basis, cell, translations, separation)
    return bloch_sum(images, translations, kpoints, cell.lattice)


def core_hamiltonian_k(
    engine: IntegralEngine,
    basis: BasisSet,
    cell: Cell,
    kpoints: KPoints,
    *,
    omega: float | None = None,
    precision: float = DEFAULT_LATTICE_SUM_PRECISION,
) -> Array:
    """Core Hamiltonian ``T(k) + V(k)`` as complex ``[kpt, ao, ao]``.

    One image list serves the kinetic and the nuclear attraction sums; see
    :func:`nuclear_attraction_k` for the regularization and the parameters.
    """
    require_kpoints_for_lattice(kpoints, cell.lattice)
    separation = RangeSeparation.for_lattice(
        cell.lattice, omega=omega, precision=precision
    )
    table = ShellTable.from_basis(basis, cell)
    translations = lattice_images(basis, cell, precision=precision)
    images = engine.kinetic(table, table, translations) + nuclear_attraction_images(
        engine, basis, cell, translations, separation
    )
    return bloch_sum(images, translations, kpoints, cell.lattice)


def nuclear_attraction_images(
    engine: IntegralEngine,
    basis: BasisSet,
    cell: Cell,
    translations: Array,
    separation: RangeSeparation,
) -> Array:
    """Return ``V_T`` ``[image, ao, ao]`` between the cell and its image at each ``T``.

    The real arrays whose Bloch sum :func:`nuclear_attraction_k` returns,
    for the ket ``translations`` ``[image, 3]`` in Bohr.
    """
    table = ShellTable.from_basis(basis, cell)
    # Every element of a Bloch sum collects one term per image, so each
    # image's sum is converged to the precision divided by their number.
    per_image = RangeSeparation(
        separation.omega, separation.precision / max(int(translations.shape[0]), 1)
    )
    return _short_range_images(
        engine, table, basis, cell, translations, per_image
    ) + _long_range_images(
        engine,
        table,
        cell,
        translations,
        per_image,
        PerpendicularProfile.from_functions(cell, basis),
    )


def _short_range_images(
    engine: IntegralEngine,
    table: ShellTable,
    basis: BasisSet,
    cell: Cell,
    translations: Array,
    separation: RangeSeparation,
) -> Array:
    """Return the erfc-attenuated attraction to every nuclear image within reach.

    An image ``A + L`` of a nucleus is included when it lies within the
    radius of some atom of the cell, beyond which that atom's functions are
    below the precision, plus the distance beyond which ``erfc(ωr)`` is;
    the kernel screens the terms of each image below the precision.
    """
    xp = default_namespace()
    positions = cell.positions_array(xp)
    radii = xp.asarray(atom_radii(basis, separation.precision), dtype=xp.float64)
    reach = xp.broadcast_to(
        xp.expand_dims(radii, 1) + separation.short_range_radius,
        (cell.n_atoms, cell.n_atoms),
    )
    images = reaching_images(cell.lattice, positions, positions, reach)
    n_images = int(images.shape[0])
    charges = xp.asarray(cell.nuclear_charges, dtype=xp.float64)
    all_charges = xp.reshape(xp.broadcast_to(charges, (n_images, cell.n_atoms)), (-1,))
    centres = xp.reshape(
        xp.expand_dims(positions, 0) + xp.expand_dims(images, 1), (-1, 3)
    )
    return engine.nuclear_attraction(
        table,
        table,
        translations,
        PointCharges(all_charges, centres),
        kernel=CoulombKernel(separation.omega, short_range=True),
        threshold=separation.precision,
    )


def _long_range_images(
    engine: IntegralEngine,
    table: ShellTable,
    cell: Cell,
    translations: Array,
    separation: RangeSeparation,
    profile: PerpendicularProfile,
) -> Array:
    """Return ``(1/Ω) Σ_G ṽ(G) S(−G) ρ̃_T(G)`` as a real array ``[image, ao, ao]``.

    The reciprocal sum reaches the cutoff for the nuclear charge against a
    pair transform of magnitude at most one. The transforms are evaluated
    in batches of images within a memory budget and contracted with the
    weights ``ṽ(G) S(−G) / Ω``; the pairs screened in the engine are those
    bounded below the precision divided by the sum of the weights'
    magnitudes, so that the neglected part of every element is below the
    precision.
    """
    xp = array_namespace(translations)
    lattice = cell.lattice
    positions = cell.positions_array(xp)
    charges = xp.asarray(cell.nuclear_charges, dtype=xp.float64)
    n_images = int(translations.shape[0])
    total = separation.reciprocal_sum(
        lattice, profile, charge_product=float(xp.sum(charges))
    )
    g_vectors = total.vectors
    angles = g_vectors @ xp.linalg.matrix_transpose(positions)
    phases = xp.exp(1j * xp.astype(angles, xp.complex128))
    structure = phases @ xp.astype(-charges, xp.complex128)
    weights = xp.astype(total.weights, xp.complex128) * structure / lattice.volume
    threshold = separation.precision / max(float(xp.sum(xp.abs(weights))), 1.0)
    n_ao = table.n_functions
    per_image = int(g_vectors.shape[0]) * n_ao * n_ao * 16
    chunk = max(1, _TRANSFORM_CHUNK_BYTES // per_image)
    parts: list[Array] = []
    for start in range(0, n_images, chunk):
        transforms = engine.pair_fourier_transform(
            table,
            table,
            translations[start : start + chunk],
            g_vectors,
            threshold=threshold,
        )
        parts.append(xp.real(xp.tensordot(weights, transforms, axes=([0], [1]))))
    return xp.concat(parts, axis=0)

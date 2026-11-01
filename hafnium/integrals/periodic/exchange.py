"""The probe-charge Ewald correction of the exchange divergence (ADR-0017)."""

from __future__ import annotations

from hafnium.backend import host_namespace
from hafnium.molecule import DEFAULT_EWALD_PRECISION, Lattice, ewald_energy
from hafnium.symmetry import KPoints


def madelung_constant(
    lattice: Lattice, kpoints: KPoints, *, precision: float = DEFAULT_EWALD_PRECISION
) -> float:
    """Return the Madelung constant ``ξ`` of the mesh's supercell, in Hartree.

    ``ξ = −2 E`` for the Ewald energy ``E`` of a unit point charge on the
    supercell lattice ``n_i a_i`` of the uniform mesh ``kpoints``, with its
    neutralizing background (ADR-0012); ``ξ`` is positive and falls off
    with the supercell size. It is the interaction the exchange term
    misses when the ``G = 0`` term of the Coulomb kernel is removed: the
    exchange matrix at every k-point gains ``ξ S(k) D(k) S(k)`` and the
    exchange energy ``−ξ N / 2`` for ``N`` electrons (ADR-0017).
    """
    supercell = lattice.supercell(kpoints.mesh_size())
    xp = host_namespace()
    energy = ewald_energy(
        supercell,
        xp.ones(1, dtype=xp.float64),
        xp.zeros((1, 3), dtype=xp.float64),
        precision=precision,
    )
    return -2.0 * energy

"""Occupation policies: how electrons fill the orbitals of each block."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from hafnium.backend import Array, array_namespace
from hafnium.errors import InputError
from hafnium.structured import BlockStructure, BlockVector


class OccupationPolicy(Protocol):
    """Assigns occupation numbers to orbitals from their energies."""

    def occupy(self, mo_energies: BlockVector) -> BlockVector:
        """Occupations ``[mo]`` per stored block, matching ``mo_energies``."""
        ...


def electrons_per_block(
    structure: BlockStructure, n_alpha: float, n_beta: float
) -> tuple[float, ...]:
    """Electrons held by each stored block of ``structure``.

    A spinor block holds all ``n_alpha + n_beta`` electrons, the alpha and
    beta blocks of a collinear structure ``n_alpha`` and ``n_beta``, and the
    single stored block of a restricted or paired-unrestricted structure
    ``n_alpha`` per spin, which requires ``n_alpha == n_beta``.
    """
    if structure.is_spinor:
        return (n_alpha + n_beta,)
    if structure.is_spatial:
        raise InputError("a spatial structure has no spin blocks to occupy")
    if structure.derived_partner is None:
        return (n_alpha, n_beta)
    if n_alpha != n_beta:
        raise InputError(
            f"a {structure.describe()} structure needs n_alpha == n_beta, got "
            f"{n_alpha} and {n_beta}"
        )
    return (n_alpha,)


@dataclass(frozen=True, slots=True)
class AufbauOccupation:
    """Fill the lowest orbitals of each block with the electrons it holds.

    Orbital energies must be in ascending order, as
    :meth:`hafnium.structured.BlockMatrix.eigh` returns them. See
    :func:`electrons_per_block` for the counts.
    """

    n_alpha: int
    n_beta: int

    def __post_init__(self) -> None:
        if self.n_alpha < 0 or self.n_beta < 0:
            raise InputError("electron counts must be non-negative")

    def occupy(self, mo_energies: BlockVector) -> BlockVector:
        """Occupations ``[mo]`` per stored block: ones for the lowest orbitals."""
        counts = electrons_per_block(mo_energies.structure, self.n_alpha, self.n_beta)
        blocks = tuple(
            _fill(energies, count, None)
            for energies, count in zip(mo_energies.blocks, counts, strict=True)
        )
        return BlockVector(mo_energies.structure, blocks)


@dataclass(frozen=True, slots=True)
class AveragedOccupation:
    """Aufbau filling with fractional counts shared within degenerate groups.

    Orbitals whose energies differ by less than ``degeneracy_tolerance``
    (relative to the larger of one and the spectral range) form a group,
    and the electrons that reach a group are spread equally over it. With
    ``n_alpha == n_beta`` this is the spin-averaged, spherically averaged
    occupation of atoms and of initial guesses.
    """

    n_alpha: float
    n_beta: float
    degeneracy_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_alpha < 0.0 or self.n_beta < 0.0:
            raise InputError("electron counts must be non-negative")
        if self.degeneracy_tolerance <= 0.0:
            raise InputError("the degeneracy tolerance must be positive")

    def occupy(self, mo_energies: BlockVector) -> BlockVector:
        """Occupations ``[mo]`` per stored block, fractional at the frontier."""
        counts = electrons_per_block(mo_energies.structure, self.n_alpha, self.n_beta)
        blocks = tuple(
            _fill(energies, count, self.degeneracy_tolerance)
            for energies, count in zip(mo_energies.blocks, counts, strict=True)
        )
        return BlockVector(mo_energies.structure, blocks)


def _fill(energies: Array, count: float, tolerance: float | None) -> Array:
    """Occupations of one block: at most one electron per orbital.

    With ``tolerance`` the degenerate groups of ``energies`` share electrons
    equally; without it ``count`` must be an integer and the lowest
    ``count`` orbitals are filled.
    """
    xp = array_namespace(energies)
    n = int(energies.shape[0])
    if count > n:
        raise InputError(f"cannot place {count} electrons in {n} orbitals")
    if tolerance is None:
        if count != int(count):
            raise InputError(f"aufbau filling needs an integer count, got {count}")
        return xp.concat(
            [
                xp.ones(int(count), dtype=xp.float64),
                xp.zeros(n - int(count), dtype=xp.float64),
            ]
        )
    values = [float(e) for e in energies]
    spread = max(1.0, values[-1] - values[0]) if values else 1.0
    occupations: list[float] = []
    remaining = count
    start = 0
    while start < n:
        end = start + 1
        while end < n and values[end] - values[end - 1] <= tolerance * spread:
            end += 1
        share = min(remaining, float(end - start)) / (end - start)
        occupations.extend([share] * (end - start))
        remaining -= share * (end - start)
        start = end
    if not math.isclose(remaining, 0.0, abs_tol=1e-12):
        raise InputError(f"cannot place {count} electrons in {n} orbitals")
    return xp.asarray(occupations, dtype=xp.float64)

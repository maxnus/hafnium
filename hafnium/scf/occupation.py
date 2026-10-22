"""Occupation policies: how electrons fill the orbitals of each block."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from hafnium.backend import Array, array_namespace
from hafnium.errors import InputError
from hafnium.structured import Block, BlockStructure, BlockVector


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
        """Occupations ``[mo]`` per stored block: ones for the lowest orbitals.

        Over k-points the levels of every stored point are filled in order
        of energy with their weights (a stored half-zone point carrying its
        partner's weight too) until the block's electron count per cell is
        reached, which must land on a whole level.
        """
        return _occupy(mo_energies, self.n_alpha, self.n_beta, None)


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
        """Occupations ``[mo]`` per stored block, fractional at the frontier.

        Over k-points the degenerate groups span the stored points, so a
        band crossing the Fermi level at several k-points is shared.
        """
        return _occupy(
            mo_energies, self.n_alpha, self.n_beta, self.degeneracy_tolerance
        )


def _occupy(
    mo_energies: BlockVector, n_alpha: float, n_beta: float, tolerance: float | None
) -> BlockVector:
    structure = mo_energies.structure
    counts = electrons_per_block(structure, n_alpha, n_beta)
    blocks = tuple(
        _fill(energies, count, tolerance, _level_weights(structure, block))
        for block, energies, count in zip(
            structure.stored_blocks, mo_energies.blocks, counts, strict=True
        )
    )
    return BlockVector(structure, blocks)


def _level_weights(structure: BlockStructure, block: Block) -> tuple[float, ...]:
    """Return the weight of every stored k-point of ``block``, partner included."""
    own, partner = structure.stored_weights(block)
    return tuple(o + p for o, p in zip(own, partner, strict=True))


FILL_TOLERANCE = 1e-12
"""Slack in the electron count when levels are filled by weight."""


def _fill(
    energies: Array, count: float, tolerance: float | None, weights: tuple[float, ...]
) -> Array:
    """Occupations of one block: at most one electron per level and k-point.

    ``energies`` is ``[mo]`` with the single weight of a molecule, or
    ``[kpt, mo]`` with one weight per stored k-point; a level at ``k`` holds
    ``weights[k]`` electrons per cell when full. Levels are filled in order
    of energy. With ``tolerance`` the degenerate groups of levels, relative
    to the larger of one and the spectral range, share electrons equally;
    without it the count must land on a whole level.
    """
    xp = array_namespace(energies)
    table = energies if energies.ndim == 2 else xp.expand_dims(energies, axis=0)
    if table.ndim != 2 or int(table.shape[0]) != len(weights):
        raise InputError("orbital energies are [mo] or [kpt, mo] with one weight per k")
    n = int(table.shape[1])
    values = [float(e) for e in xp.reshape(table, (-1,))]
    level_weights = [weights[index // n] for index in range(len(values))]
    capacity = n * sum(weights)
    if count > capacity + FILL_TOLERANCE:
        raise InputError(f"cannot place {count} electrons in {n} orbitals")
    order = sorted(range(len(values)), key=lambda index: values[index])
    occupations = (
        _fill_whole_levels(values, level_weights, order, count)
        if tolerance is None
        else _fill_degenerate_groups(values, level_weights, order, count, tolerance)
    )
    result = xp.reshape(xp.asarray(occupations, dtype=xp.float64), table.shape)
    return result if energies.ndim == 2 else result[0]


def _fill_whole_levels(
    values: list[float], weights: list[float], order: list[int], count: float
) -> list[float]:
    if count != int(count):
        raise InputError(f"aufbau filling needs an integer count, got {count}")
    occupations = [0.0] * len(values)
    remaining = count
    for index in order:
        if remaining <= FILL_TOLERANCE:
            break
        if weights[index] > remaining + FILL_TOLERANCE:
            raise InputError(
                f"{count} electrons do not fill whole levels across the k-points "
                f"(a level of weight {weights[index]:g} would be partly occupied); "
                "use AveragedOccupation to share the frontier"
            )
        occupations[index] = 1.0
        remaining -= weights[index]
    return occupations


def _fill_degenerate_groups(
    values: list[float],
    weights: list[float],
    order: list[int],
    count: float,
    tolerance: float,
) -> list[float]:
    sorted_values = [values[index] for index in order]
    spread = max(1.0, sorted_values[-1] - sorted_values[0]) if values else 1.0
    occupations = [0.0] * len(values)
    remaining = count
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and sorted_values[end] - sorted_values[end - 1] <= tolerance * spread
        ):
            end += 1
        group = order[start:end]
        capacity = sum(weights[index] for index in group)
        share = min(remaining, capacity) / capacity
        for index in group:
            occupations[index] = share
        remaining = max(remaining - share * capacity, 0.0)
        start = end
    if not math.isclose(remaining, 0.0, abs_tol=FILL_TOLERANCE):
        raise InputError(f"cannot place {count} electrons in {len(values)} levels")
    return occupations

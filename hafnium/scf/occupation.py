"""Occupation policies: how electrons fill the orbitals of each block.

The electrons of a spin block are fixed by the class and the electron
counts (:func:`electrons_per_block`). With irrep sectors (ADR-0023) the
spin block is split over the sectors, and its electrons fill the orbitals
of all sectors together in order of energy, unless a sector's count is
fixed, in which case it fills its own orbitals and the remaining electrons
share the free sectors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from hafnium.backend import Array, array_namespace
from hafnium.errors import InputError
from hafnium.structured import Block, BlockStructure, BlockVector, Spin

IrrepElectrons = Mapping[str, int | tuple[int, int]]
"""Electrons fixed per irrep label: a total, or ``(n_alpha, n_beta)``."""


class OccupationPolicy(Protocol):
    """Assigns occupation numbers to orbitals from their energies."""

    def occupy(self, mo_energies: BlockVector) -> BlockVector:
        """Occupations ``[mo]`` per stored block, matching ``mo_energies``."""
        ...


def electrons_per_block(
    structure: BlockStructure, n_alpha: float, n_beta: float
) -> tuple[float, ...]:
    """Electrons held by each stored spin block of ``structure``.

    A spinor block holds all ``n_alpha + n_beta`` electrons, the alpha and
    beta blocks of a collinear structure ``n_alpha`` and ``n_beta``, and the
    single stored block of a restricted or paired-unrestricted structure
    ``n_alpha`` per spin, which requires ``n_alpha == n_beta``. With irrep
    sectors the counts are those of the spin blocks of one sector's
    structure, shared over the sectors by the policy.
    """
    structure = structure.sector_structure
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

    Parameters
    ----------
    n_alpha, n_beta
        Electron counts.
    irrep_electrons
        Electrons fixed in irrep sectors, by Mulliken label, as PySCF's
        ``irrep_nelec``: a total, which a restricted or paired class
        splits equally between the spins and an unrestricted class with
        the odd electron in alpha, or ``(n_alpha, n_beta)`` for a
        collinear class. The sectors named fill their own lowest orbitals
        with those electrons; the remaining electrons of each spin fill the
        other sectors together in order of energy. Only with irrep sectors.
    """

    n_alpha: int
    n_beta: int
    irrep_electrons: IrrepElectrons | None = None

    def __post_init__(self) -> None:
        if self.n_alpha < 0 or self.n_beta < 0:
            raise InputError("electron counts must be non-negative")
        for label, count in (self.irrep_electrons or {}).items():
            counts = (count,) if isinstance(count, int) else count
            if len(counts) != 2 and not isinstance(count, int):
                raise InputError(
                    f"irrep {label}: give a total or (n_alpha, n_beta), got {count!r}"
                )
            if any(c < 0 for c in counts):
                raise InputError(f"irrep {label}: electron counts are non-negative")

    def occupy(self, mo_energies: BlockVector) -> BlockVector:
        """Occupations ``[mo]`` per stored block: ones for the lowest orbitals.

        Over k-points the levels of every stored point are filled in order
        of energy with their weights (a stored half-zone point carrying its
        partner's weight too) until the block's electron count per cell is
        reached, which must land on a whole level.
        """
        return _occupy(
            mo_energies, self.n_alpha, self.n_beta, None, self.irrep_electrons
        )


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
    mo_energies: BlockVector,
    n_alpha: float,
    n_beta: float,
    tolerance: float | None,
    fixed: IrrepElectrons | None = None,
) -> BlockVector:
    structure = mo_energies.structure
    base = structure.sector_structure
    counts = electrons_per_block(base, n_alpha, n_beta)
    if structure.irreps is None:
        if fixed:
            raise InputError(
                f"electrons are fixed per irrep in the {structure.describe()} "
                "structure, which has no irrep sectors"
            )
        blocks = tuple(
            _fill(energies, count, tolerance, _level_weights(base, block))
            for block, energies, count in zip(
                base.stored_blocks, mo_energies.blocks, counts, strict=True
            )
        )
        return BlockVector(structure, blocks)
    size = base.n_stored
    filled: list[Array | None] = [None] * structure.n_stored
    for index, (block, count) in enumerate(
        zip(base.stored_blocks, counts, strict=True)
    ):
        weights = _level_weights(base, block)
        positions = [sector * size + index for sector in range(structure.n_sectors)]
        fixed_counts = _fixed_counts(structure, base, index, fixed or {})
        remaining = count - sum(fixed_counts.values())
        if remaining < -FILL_TOLERANCE:
            raise InputError(
                f"the irrep sectors fix {sum(fixed_counts.values()):g} electrons in "
                f"the {block.spin.value} block, which holds {count:g}"
            )
        for position, fixed_count in fixed_counts.items():
            filled[position] = _fill(
                mo_energies.blocks[position], fixed_count, tolerance, weights
            )
        free = [position for position in positions if position not in fixed_counts]
        if not free:
            if remaining > FILL_TOLERANCE:
                raise InputError(
                    f"{remaining:g} electrons of the {block.spin.value} block are "
                    "left over after the fixed irrep sectors"
                )
            continue
        energies = [mo_energies.blocks[position] for position in free]
        xp = array_namespace(*energies)
        joined = _fill(xp.concat(energies, axis=-1), remaining, tolerance, weights)
        start = 0
        for position, part in zip(free, energies, strict=True):
            width = int(part.shape[-1])
            filled[position] = joined[..., start : start + width]
            start += width
    return BlockVector(structure, tuple(_present(block) for block in filled))


def _present(block: Array | None) -> Array:
    assert block is not None
    return block


def _fixed_counts(
    structure: BlockStructure,
    base: BlockStructure,
    index: int,
    fixed: IrrepElectrons,
) -> dict[int, float]:
    """Return the fixed electron count of each sector for stored spin block ``index``.

    Keyed by the position of the sector's block among all stored blocks.
    """
    sectors = structure.irreps
    assert sectors is not None
    block = base.stored_blocks[index]
    counts: dict[int, float] = {}
    for label, count in fixed.items():
        try:
            sector = sectors.index(label)
        except Exception as error:
            raise InputError(str(error)) from None
        if block.spin is Spin.SPINOR:
            if not isinstance(count, int):
                raise InputError(
                    f"irrep {label}: a generalized state holds a total per irrep, "
                    f"not {count!r}"
                )
            per_block = float(count)
        elif base.derived_partner is not None:
            per_block = _per_spin(label, count, paired=True)[0]
        else:
            per_block = _per_spin(label, count, paired=False)[
                0 if block.spin is Spin.ALPHA else 1
            ]
        counts[sector * base.n_stored + index] = per_block
    return counts


def _per_spin(
    label: str, count: int | tuple[int, int], *, paired: bool
) -> tuple[float, float]:
    """Return ``(n_alpha, n_beta)`` of a fixed irrep count."""
    if isinstance(count, int):
        if paired and count % 2:
            raise InputError(
                f"irrep {label}: a restricted or paired state holds an even "
                f"number of electrons per irrep, got {count}"
            )
        return (count - count // 2, count // 2)
    if paired and count[0] != count[1]:
        raise InputError(
            f"irrep {label}: a restricted or paired state holds equal alpha and "
            f"beta counts per irrep, got {count}"
        )
    return (float(count[0]), float(count[1]))


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

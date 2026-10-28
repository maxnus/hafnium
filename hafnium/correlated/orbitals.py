"""The occupied and virtual orbitals of a mean-field state, as tensor axes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from hafnium.backend import Array, array_namespace
from hafnium.errors import InputError, StructureError
from hafnium.scf.driver import ScfResult
from hafnium.structured import (
    Block,
    BlockMatrix,
    BlockStructure,
    BlockVector,
    Ring,
    TensorAxis,
)

OCCUPATION_TOLERANCE = 1e-8
"""Deviation of an occupation number from zero or one that still counts as integer."""

REALITY_TOLERANCE = 1e-10
"""Imaginary part, relative to the largest component, an orbital of a real
state may keep at a time-reversal invariant point once its phase is fixed."""


class Space(Enum):
    """The orbital spaces a correlated method distinguishes."""

    OCCUPIED = "occupied"
    VIRTUAL = "virtual"


@dataclass(frozen=True, slots=True, eq=False)
class OrbitalSpaces:
    """The occupied and virtual canonical orbitals of a mean-field state.

    Per stored block of the state's structure, the coefficients
    ``[ao, occ]`` and ``[ao, vir]`` (``[salc, ...]`` with irrep sectors,
    ``[2n, ...]`` for a spinor block, ``[kpt, ...]`` over the stored
    k-points of a cell) and the orbital energies ``[occ]`` and ``[vir]``,
    in the order of the canonical orbitals. The spaces of a derived block
    follow from its source as the block does, at a derived k-point from
    the stored partner, so the sector of every block of the structure at
    every point has a size and energies; over k-points a block holds the
    same number of occupied orbitals at every point.

    Build it with :meth:`from_scf`.
    """

    occupied: BlockMatrix
    virtual: BlockMatrix
    occupied_energies: BlockVector
    virtual_energies: BlockVector
    _stored_position: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        structure = self.occupied.structure
        if any(
            other.structure != structure
            for other in (self.virtual, self.occupied_energies, self.virtual_energies)
        ):
            raise StructureError("the orbital spaces share the structure of the state")
        if structure.k_axis is not None and structure.k_axis.operations is not None:
            raise InputError(
                "the correlated methods over the irreducible wedge of a space group "
                "are a later extension; converge the reference over the full k set"
            )
        for coefficients, energies in (
            (self.occupied, self.occupied_energies),
            (self.virtual, self.virtual_energies),
        ):
            for c, e in zip(coefficients.blocks, energies.blocks, strict=True):
                if int(c.shape[-1]) != int(e.shape[-1]):
                    raise StructureError(
                        "every orbital of a space has a coefficient column and an "
                        "energy"
                    )
        positions: list[int] = []
        count = 0
        for block in structure.blocks:
            if block.derived_from is None:
                positions.append(count)
                count += 1
            else:
                positions.append(positions[block.derived_from])
        object.__setattr__(self, "_stored_position", tuple(positions))

    @classmethod
    def from_scf(
        cls, reference: ScfResult, *, tolerance: float = OCCUPATION_TOLERANCE
    ) -> OrbitalSpaces:
        """Split the canonical orbitals of ``reference`` by their occupation.

        Over k-points the orbitals of a real block at the time-reversal
        invariant points are made real by a phase each, since the derived
        blocks of a tensor take the orbitals at ``−k`` as the conjugates
        of those at ``k`` (ADR-0031); the Fock matrix of a real state is
        real there, so its eigenvectors can be chosen real.

        Raises
        ------
        InputError
            If an orbital is fractionally occupied (within ``tolerance`` of
            neither zero nor one), as the frontier of an averaged
            occupation is: perturbation theory needs a single determinant.
            Over k-points also if the occupied orbitals of a block differ
            between its points, as in a metal, or an orbital at an
            invariant point cannot be made real by a phase.
        """
        occupied: list[Array] = []
        virtual: list[Array] = []
        occupied_energies: list[Array] = []
        virtual_energies: list[Array] = []
        structure = reference.structure
        for block, c, e, o in zip(
            structure.stored_blocks,
            reference.mo_coefficients.blocks,
            reference.mo_energies.blocks,
            reference.mo_occupations.blocks,
            strict=True,
        ):
            xp = array_namespace(c, e, o)
            c = _real_at_invariant_points(structure, block, c)
            filled, empty = _split_occupations(o, tolerance)
            for chosen, coefficients, energies in (
                (filled, occupied, occupied_energies),
                (empty, virtual, virtual_energies),
            ):
                index = xp.asarray(chosen, dtype=xp.int64)
                coefficients.append(xp.take(c, index, axis=-1))
                energies.append(xp.take(e, index, axis=-1))
        return cls(
            BlockMatrix(structure, tuple(occupied)),
            BlockMatrix(structure, tuple(virtual)),
            BlockVector(structure, tuple(occupied_energies)),
            BlockVector(structure, tuple(virtual_energies)),
        )

    @property
    def structure(self) -> BlockStructure:
        """Block structure of the state."""
        return self.occupied.structure

    @property
    def ring(self) -> Ring:
        """Ring the tensors over these orbitals are stored over."""
        return Ring.COMPLEX if self.structure.is_complex else Ring.REAL

    @property
    def n_ao(self) -> int:
        """Number of atomic orbitals."""
        return self.occupied.n_spatial

    def coefficients(self, space: Space) -> BlockMatrix:
        """Return the orbital coefficients of ``space``."""
        return self.occupied if space is Space.OCCUPIED else self.virtual

    def energies(self, space: Space) -> BlockVector:
        """Return the orbital energies of ``space``."""
        return (
            self.occupied_energies if space is Space.OCCUPIED else self.virtual_energies
        )

    def sizes(self, space: Space) -> tuple[int, ...]:
        """Count the orbitals of ``space`` in every sector, a block at a point."""
        blocks = self.coefficients(space).blocks
        n_kpt = 1 if self.structure.k_axis is None else self.structure.k_axis.n_kpt
        return tuple(
            int(blocks[position].shape[-1])
            for position in self._stored_position
            for _ in range(n_kpt)
        )

    def sector_energies(self, space: Space, sector: int) -> Array:
        """Return the orbital energies of ``space`` in a sector, a block at a point.

        A derived point takes the energies of its stored partner, which
        time reversal leaves unchanged.
        """
        structure = self.structure
        axis = structure.k_axis
        if axis is None:
            return self.energies(space).blocks[self._stored_position[sector]]
        block_index, kpoint = divmod(sector, axis.n_kpt)
        position = self._stored_position[block_index]
        block = structure.stored_blocks[position]
        stored = structure.stored_kpoints(block)
        assert stored is not None
        source = axis.derivation(kpoint, structure.half_zone(block)).source
        return self.energies(space).blocks[position][stored.index(source)]

    def axis(self, space: Space, *, conjugated: bool) -> TensorAxis:
        """Return the tensor axis running over the orbitals of ``space``."""
        return TensorAxis(self.structure, self.sizes(space), conjugated)

    def materialized(self, space: Space) -> tuple[Array, ...]:
        """Return the coefficients of ``space`` per spin block, over the full k set.

        One array per block of the spin pattern in the atomic basis,
        ``[ao, m]`` or ``[kpt, ao, m]`` over every point of the set, the
        derived blocks and points built as the dense form builds them:
        the orbitals at ``−k`` are the time-reversed orbitals at ``k``,
        which is what the derived blocks of a tensor over these spaces
        assume.
        """
        return self.coefficients(space).materialized(orbitals=True)

    def space_of(self, axis: TensorAxis) -> Space:
        """Which space ``axis`` runs over, whatever its conjugation.

        Raises
        ------
        StructureError
            If the axis runs over neither space of these orbitals.
        """
        for space in Space:
            if axis.same_space(self.axis(space, conjugated=axis.conjugated)):
                return space
        raise StructureError(
            "the axis runs over neither the occupied nor the virtual orbitals of "
            "this state"
        )


def _split_occupations(
    occupations: Array, tolerance: float
) -> tuple[list[int], list[int]]:
    """Split the orbitals of a block into occupied and empty, equal at every point."""
    xp = array_namespace(occupations)
    table = (
        occupations if occupations.ndim == 2 else xp.expand_dims(occupations, axis=0)
    )
    n = int(table.shape[1])
    result: tuple[list[int], list[int]] | None = None
    for kpoint in range(int(table.shape[0])):
        row = [float(o) for o in table[kpoint]]
        filled = [k for k in range(n) if row[k] > 1.0 - tolerance]
        empty = [k for k in range(n) if row[k] < tolerance]
        if len(filled) + len(empty) != n:
            raise InputError(
                "every orbital of the reference must be occupied by zero or one "
                "electron; a fractionally occupied frontier (an averaged "
                "occupation) is not a single determinant"
            )
        if result is None:
            result = (filled, empty)
        elif result != (filled, empty):
            raise InputError(
                "the occupied orbitals of a block differ between its k-points; "
                "perturbation theory needs an insulating reference with the same "
                "occupied bands at every point"
            )
    assert result is not None
    return result


def _real_at_invariant_points(
    structure: BlockStructure, block: Block, coefficients: Array
) -> Array:
    """Fix the phases of a real block's orbitals at the invariant k-points.

    Each column is divided by the phase of its largest component; the
    result must then be real to :data:`REALITY_TOLERANCE`.
    """
    axis = structure.k_axis
    if axis is None or block.ring is not Ring.REAL or axis.all_invariant:
        return coefficients
    xp = array_namespace(coefficients)
    if not xp.isdtype(coefficients.dtype, "complex floating"):
        return coefficients
    positions = set(structure.invariant_positions(block))
    parts: list[Array] = []
    for position in range(int(coefficients.shape[0])):
        part = coefficients[position]
        if position in positions:
            part = _with_real_phases(part)
        parts.append(part)
    return xp.stack(parts, axis=0)


def _with_real_phases(coefficients: Array) -> Array:
    """Rotate every column of ``[ao, mo]`` by a phase so that it is real."""
    xp = array_namespace(coefficients)
    n, m = int(coefficients.shape[0]), int(coefficients.shape[1])
    magnitude = xp.abs(coefficients)
    pivot_rows = xp.argmax(magnitude, axis=0)
    selector = xp.astype(
        xp.expand_dims(xp.arange(n), axis=1) == xp.expand_dims(pivot_rows, axis=0),
        coefficients.dtype,
    )
    pivots = xp.sum(coefficients * selector, axis=0)
    scale = xp.where(xp.abs(pivots) > 0.0, xp.abs(pivots), xp.ones_like(pivots))
    rotated = coefficients * xp.expand_dims(xp.conj(pivots) / scale, axis=0)
    largest = max(1.0, float(xp.max(magnitude))) if n * m else 1.0
    if float(xp.max(xp.abs(xp.imag(rotated)))) > REALITY_TOLERANCE * largest:
        raise InputError(
            "an orbital of a real state at a time-reversal invariant k-point is "
            "not real up to a phase; the tensors over k-points need real orbitals "
            "there (a degenerate level may have been mixed by the solver)"
        )
    return xp.astype(xp.real(rotated), coefficients.dtype)

"""The occupied and virtual orbitals of a mean-field state, as tensor axes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from hafnium.backend import Array, array_namespace
from hafnium.errors import InputError, StructureError
from hafnium.scf.driver import ScfResult
from hafnium.structured import (
    BlockMatrix,
    BlockStructure,
    BlockVector,
    Ring,
    TensorAxis,
)

OCCUPATION_TOLERANCE = 1e-8
"""Deviation of an occupation number from zero or one that still counts as integer."""


class Space(Enum):
    """The orbital spaces a correlated method distinguishes."""

    OCCUPIED = "occupied"
    VIRTUAL = "virtual"


@dataclass(frozen=True, slots=True, eq=False)
class OrbitalSpaces:
    """The occupied and virtual canonical orbitals of a mean-field state.

    Per stored block of the state's structure, the coefficients
    ``[ao, occ]`` and ``[ao, vir]`` (``[salc, ...]`` with irrep sectors,
    ``[2n, ...]`` for a spinor block) and the orbital energies ``[occ]``
    and ``[vir]``, in the order of the canonical orbitals. The spaces of a
    derived block follow from its source as the block does, so the sector
    of every block of the structure has a size and energies.

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
        if structure.k_axis is not None:
            raise InputError(
                "the correlated methods of a cell come with M8, part two; the orbital "
                "spaces serve molecules"
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

        Raises
        ------
        InputError
            If an orbital is fractionally occupied (within ``tolerance`` of
            neither zero nor one), as the frontier of an averaged
            occupation is: perturbation theory needs a single determinant.
        """
        occupied: list[Array] = []
        virtual: list[Array] = []
        occupied_energies: list[Array] = []
        virtual_energies: list[Array] = []
        for c, e, o in zip(
            reference.mo_coefficients.blocks,
            reference.mo_energies.blocks,
            reference.mo_occupations.blocks,
            strict=True,
        ):
            xp = array_namespace(c, e, o)
            n = int(o.shape[0])
            filled = [k for k in range(n) if float(o[k]) > 1.0 - tolerance]
            empty = [k for k in range(n) if float(o[k]) < tolerance]
            if len(filled) + len(empty) != n:
                raise InputError(
                    "every orbital of the reference must be occupied by zero or one "
                    "electron; a fractionally occupied frontier (an averaged "
                    "occupation) is not a single determinant"
                )
            for chosen, coefficients, energies in (
                (filled, occupied, occupied_energies),
                (empty, virtual, virtual_energies),
            ):
                index = xp.asarray(chosen, dtype=xp.int64)
                coefficients.append(xp.take(c, index, axis=1))
                energies.append(xp.take(e, index, axis=0))
        structure = reference.structure
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
        """Count the orbitals of ``space`` in every block of the structure."""
        blocks = self.coefficients(space).blocks
        return tuple(
            int(blocks[position].shape[-1]) for position in self._stored_position
        )

    def sector_energies(self, space: Space, sector: int) -> Array:
        """Return the orbital energies of ``space`` in the block ``sector``."""
        return self.energies(space).blocks[self._stored_position[sector]]

    def axis(self, space: Space, *, conjugated: bool) -> TensorAxis:
        """Return the tensor axis running over the orbitals of ``space``."""
        return TensorAxis(self.structure, self.sizes(space), conjugated)

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

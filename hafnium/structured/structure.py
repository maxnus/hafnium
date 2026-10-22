"""Block structures: which blocks a symmetry-constrained operator consists of.

The spin structure and, for periodic systems, the k axis exist at this
stage. Point-group sectors, added in a later milestone, multiply each block
below by the irreps of the group; non-Abelian irreps also use the
``weight`` field for their multiplicity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from hafnium.errors import StructureError
from hafnium.structured.kaxis import KAxis


class Spin(Enum):
    """Spin sector covered by a block."""

    NONE = "none"
    """A spatial operator that acts identically on both spins."""
    ALPHA = "alpha"
    BETA = "beta"
    SPINOR = "spinor"
    """Both spins in one block of dimension ``2n``, alpha components first."""


class Ring(Enum):
    """Number ring of a block's entries."""

    REAL = "real"
    COMPLEX = "complex"
    QUATERNION = "quaternion"
    """Stored as the complex ``2n × 2n`` embedding with Kramers structure."""


class Partner(Enum):
    """How a derived block is obtained from its source block."""

    COPY = "copy"
    CONJUGATE = "conjugate"


@dataclass(frozen=True, slots=True)
class Block:
    """One block of a :class:`BlockStructure`.

    Parameters
    ----------
    spin
        Spin sector of the block.
    ring
        Number ring of its entries.
    weight
        Multiplicity of the block in traces (1 at this stage).
    derived_from
        Index of the stored block this block is derived from, or ``None`` if
        the block is stored itself.
    partner
        Rule deriving the block from its source; set exactly when
        ``derived_from`` is.
    """

    spin: Spin
    ring: Ring
    weight: int = 1
    derived_from: int | None = None
    partner: Partner | None = None

    def __post_init__(self) -> None:
        if (self.derived_from is None) != (self.partner is None):
            raise StructureError("derived_from and partner must be given together")
        if self.weight < 1:
            raise StructureError("block weight must be at least 1")
        if self.ring is Ring.QUATERNION and self.spin is not Spin.SPINOR:
            raise StructureError("quaternionic blocks require the spinor sector")
        if self.spin in (Spin.NONE, Spin.SPINOR) and self.derived_from is not None:
            raise StructureError("spatial and spinor blocks cannot be derived")

    @property
    def is_stored(self) -> bool:
        """Whether the block holds its own data."""
        return self.derived_from is None


@dataclass(frozen=True, slots=True)
class BlockStructure:
    """Ordered blocks of a symmetry-constrained one-body operator.

    Use the class methods to build the structures of the Fukutome classes;
    the constructor validates arbitrary block tuples. A ``k_axis`` makes
    every block a batch over k-points (ADR-0015): blocks over the real and
    quaternionic rings then store the half zone, ``k_axis.representatives``,
    and blocks over the complex ring every point. Without a k axis the
    structure is molecular and blocks hold one matrix each.
    """

    blocks: tuple[Block, ...]
    k_axis: KAxis | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if not self.blocks:
            raise StructureError("a structure needs at least one block")
        spins = [block.spin for block in self.blocks]
        if len(set(spins)) != len(spins):
            raise StructureError("each spin sector may appear once")
        if set(spins) not in ({Spin.NONE}, {Spin.SPINOR}, {Spin.ALPHA, Spin.BETA}):
            raise StructureError(f"unsupported spin sectors {spins}")
        if spins == [Spin.BETA, Spin.ALPHA]:
            raise StructureError("the alpha block precedes the beta block")
        for index, block in enumerate(self.blocks):
            if block.derived_from is None:
                continue
            if not 0 <= block.derived_from < index:
                raise StructureError("a derived block must follow its source")
            source = self.blocks[block.derived_from]
            if not source.is_stored:
                raise StructureError("a derived block must derive from a stored block")
            if source.ring is not block.ring:
                raise StructureError("a derived block shares the ring of its source")

    @classmethod
    def spatial(cls, ring: Ring = Ring.REAL) -> BlockStructure:
        """One block acting identically on both spins (overlap, core Hamiltonian)."""
        return cls((Block(Spin.NONE, ring),))

    @classmethod
    def restricted(cls, ring: Ring = Ring.REAL) -> BlockStructure:
        """Alpha block stored, beta block a copy (RHF)."""
        return cls(
            (
                Block(Spin.ALPHA, ring),
                Block(Spin.BETA, ring, derived_from=0, partner=Partner.COPY),
            )
        )

    @classmethod
    def unrestricted(cls, ring: Ring = Ring.REAL) -> BlockStructure:
        """Independent alpha and beta blocks (UHF)."""
        return cls((Block(Spin.ALPHA, ring), Block(Spin.BETA, ring)))

    @classmethod
    def paired_unrestricted(cls) -> BlockStructure:
        """Alpha block stored, beta block its complex conjugate (paired UHF)."""
        return cls(
            (
                Block(Spin.ALPHA, Ring.COMPLEX),
                Block(
                    Spin.BETA, Ring.COMPLEX, derived_from=0, partner=Partner.CONJUGATE
                ),
            )
        )

    @classmethod
    def generalized(cls, ring: Ring = Ring.REAL) -> BlockStructure:
        """One spinor block of dimension ``2n`` (GHF)."""
        return cls((Block(Spin.SPINOR, ring),))

    @classmethod
    def paired_generalized(cls) -> BlockStructure:
        """One quaternionic spinor block (paired GHF)."""
        return cls((Block(Spin.SPINOR, Ring.QUATERNION),))

    def with_k_axis(self, k_axis: KAxis | None) -> BlockStructure:
        """Return the same block pattern over ``k_axis`` (``None``: molecular)."""
        return BlockStructure(self.blocks, k_axis=k_axis)

    def half_zone(self, block: Block) -> bool:
        """Whether ``block`` stores the half zone and derives its partners."""
        return block.ring is not Ring.COMPLEX

    def stored_kpoints(self, block: Block) -> tuple[int, ...] | None:
        """Full-set indices of the k-points ``block`` stores, in storage order.

        ``None`` without a k axis.
        """
        if self.k_axis is None:
            return None
        return self.k_axis.stored(self.half_zone(block))

    def stored_weights(
        self, block: Block
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """``(own, partner)`` weights of the stored k-points of ``block``.

        See :meth:`KAxis.stored_weights`; ``((1,), (0,))`` without a k axis.
        """
        if self.k_axis is None:
            return (1.0,), (0.0,)
        return self.k_axis.stored_weights(self.half_zone(block))

    def invariant_positions(self, block: Block) -> tuple[int, ...]:
        """Storage positions of the time-reversal invariant k-points of ``block``.

        Without a k axis the single stored matrix is invariant: ``(0,)``.
        """
        if self.k_axis is None:
            return (0,)
        invariant = self.k_axis.invariant
        stored = self.k_axis.stored(self.half_zone(block))
        return tuple(position for position, k in enumerate(stored) if invariant[k])

    def storage_ring(self, block: Block) -> Ring:
        """Ring the entries of ``block`` are stored over.

        A real block is stored as ``float64`` only when every k-point is
        time-reversal invariant, which includes the molecular case; at other
        k-points the Bloch matrices are complex and the block is stored as
        ``complex128`` with the conjugate relation between ``k`` and ``−k``.
        """
        if (
            block.ring is Ring.REAL
            and self.k_axis is not None
            and not self.k_axis.all_invariant
        ):
            return Ring.COMPLEX
        return block.ring

    @property
    def all_kpoints(self) -> tuple[int, ...] | None:
        """Every index of the k axis, or ``None`` without one."""
        if self.k_axis is None:
            return None
        return tuple(range(self.k_axis.n_kpt))

    @property
    def stored_blocks(self) -> tuple[Block, ...]:
        """The blocks that hold data, in storage order."""
        return tuple(block for block in self.blocks if block.is_stored)

    @property
    def n_stored(self) -> int:
        """Number of stored blocks."""
        return len(self.stored_blocks)

    @property
    def is_spatial(self) -> bool:
        """Whether the structure is the single spin-independent block."""
        return self.blocks[0].spin is Spin.NONE

    @property
    def is_spinor(self) -> bool:
        """Whether the structure is a single spinor block."""
        return self.blocks[0].spin is Spin.SPINOR

    @property
    def is_collinear(self) -> bool:
        """Whether the structure has separate alpha and beta blocks."""
        return not (self.is_spatial or self.is_spinor)

    @property
    def derived_partner(self) -> Partner | None:
        """Partner rule of the derived block, if the structure has one."""
        for block in self.blocks:
            if block.partner is not None:
                return block.partner
        return None

    @property
    def is_complex(self) -> bool:
        """Whether any stored block has complex entries."""
        return any(block.ring is not Ring.REAL for block in self.stored_blocks)

    def with_ring(self, ring: Ring) -> BlockStructure:
        """Return the same block pattern over another ring.

        Only real and complex rings can be chosen; the paired structures
        fix their ring themselves.
        """
        if ring is Ring.QUATERNION:
            raise StructureError(
                "use BlockStructure.paired_generalized() for quaternionic blocks"
            )
        if self.derived_partner is Partner.CONJUGATE or any(
            block.ring is Ring.QUATERNION for block in self.blocks
        ):
            raise StructureError("paired structures fix their ring")
        return BlockStructure(
            tuple(
                Block(block.spin, ring, block.weight, block.derived_from, block.partner)
                for block in self.blocks
            ),
            k_axis=self.k_axis,
        )

    def describe(self) -> str:
        """Short human-readable name, for messages and logs."""
        name = self._describe_spin()
        if self.k_axis is None:
            return name
        n = self.k_axis.n_kpt
        return f"{name} at {n} k-point{'' if n == 1 else 's'}"

    def _describe_spin(self) -> str:
        ring = self.stored_blocks[0].ring.value
        if self.is_spatial:
            return f"spatial ({ring})"
        if self.is_spinor:
            return (
                "paired generalized"
                if ring == "quaternion"
                else f"generalized ({ring})"
            )
        if self.derived_partner is Partner.COPY:
            return f"restricted ({ring})"
        if self.derived_partner is Partner.CONJUGATE:
            return "paired unrestricted"
        return f"unrestricted ({ring})"

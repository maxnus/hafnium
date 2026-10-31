"""Block tensors: block-sparse tensors over the sectors of structured matrices.

A quantity with several orbital indices, the fitted integrals in the
orbital basis, antisymmetrized integrals or amplitudes, carries one sector
label per index, and the symmetry of the state fixes which of its blocks
can be nonzero (PLAN.md §5.5, ADR-0030). Every axis of a
:class:`BlockTensor` runs over the blocks of a :class:`BlockStructure`,
stored and derived alike, each holding a number of orbitals: a sector is
labelled by the spin and the irrep of its block. A block of the tensor is
labelled by one sector per axis and is allowed when the charges of its
sectors cancel: the spin charges, ``+1`` for alpha, ``−1`` for beta and
zero for spinor and spatial sectors, counted negatively on a conjugated
index, sum to zero, the characters of the irreps multiply to the totally
symmetric irrep, and over a k axis the crystal momenta sum to zero modulo
the reciprocal lattice (ADR-0031): the sectors of an axis over a k axis
are the pairs of a block and a point of the full set. Blocks that vanish
by this rule, and blocks with an empty sector, are never stored.

The symmetries of the state derive the remaining blocks. Mapping every
index to its partner block at once, alpha to beta and back, is a symmetry
of a restricted state, and time reversal, every index to the partner
point ``−k`` with the conjugate block where the ring asks for it, is a
symmetry of a real or paired state; each maps a block onto a copy or the
complex conjugate of another, and the tensor stores the lexicographically
smallest key of every orbit. A contraction pairs a conjugated with an
unconjugated index, as a sum over a complete set of orbitals does, and
the charge rule of its result follows from those of its operands.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from hafnium.backend import Array, Namespace, array_namespace, default_namespace
from hafnium.errors import StructureError
from hafnium.structured.sectors import (
    Characters,
    is_totally_symmetric,
    product_characters,
)
from hafnium.structured.structure import BlockStructure, Partner, Ring, Spin
from hafnium.structured.validation import dtype_for

Key: TypeAlias = tuple[int, ...]
"""One sector index per axis, labelling a block of a tensor."""

Scalar: TypeAlias = int | float | complex

Momentum: TypeAlias = tuple[int, ...]
"""The crystal momentum of a sector, integers over the modulus of its k axis."""

_Charge: TypeAlias = tuple[int, Characters | None, Momentum | None]
"""The signed spin charge of a sector, the characters of its irrep, its momentum."""


class Symmetry(Enum):
    """A symmetry of the state that derives blocks of a tensor from others."""

    SPIN_FLIP = "spin flip"
    """Alpha to beta and back on every index: the blocks are copies."""
    TIME_REVERSAL = "time reversal"
    """Every index to its time-reversal partner: the blocks are conjugates."""


SPIN_CHARGE: dict[Spin, int] = {
    Spin.ALPHA: 1,
    Spin.BETA: -1,
    Spin.NONE: 0,
    Spin.SPINOR: 0,
}
"""The ``S_z`` charge a sector of each spin carries, in units of ``1/2``."""

DENSE_TOLERANCE = 1e-10
"""Relative deviation a dense tensor may show from the structure it is read into."""


@dataclass(frozen=True, slots=True)
class TensorAxis:
    """One index of a block tensor.

    Parameters
    ----------
    structure
        The structure whose blocks, stored and derived, are the sectors of
        the axis: the structure of the state for an orbital index, a
        spatial structure over the irrep sectors of the auxiliary basis or
        over the momentum transfers for a density-fitting index. Over a k
        axis the sectors are the pairs of a block and a point of the full
        set, the points of a block adjacent, and the axis must carry the
        momenta of a uniform mesh without point operations.
    sizes
        Number of functions in every sector, in the order of the sectors;
        a derived block holds as many as its source at every point.
    conjugated
        Whether the index enters conjugated, as the bra index of an
        integral does; the charge of its sector then counts negatively.
    starts
        Position of the first function of every sector within the sector
        of the axis this one is a window of (:meth:`window`); empty, the
        default, for an axis over the whole space, where every sector
        starts at zero. The related sectors of a window (:meth:`related_sectors`)
        start at the same position, as they hold as many functions.
    """

    structure: BlockStructure
    sizes: tuple[int, ...]
    conjugated: bool = False
    starts: tuple[int, ...] = ()
    _offsets: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sizes = tuple(int(size) for size in self.sizes)
        object.__setattr__(self, "sizes", sizes)
        starts = tuple(int(start) for start in self.starts) or (0,) * len(sizes)
        object.__setattr__(self, "starts", starts)
        _check_k_axis(self.structure)
        n_kpt = self.n_kpt
        blocks = self.structure.blocks
        if len(sizes) != len(blocks) * n_kpt:
            raise StructureError(
                f"an axis over the {self.structure.describe()} structure has "
                f"{len(blocks) * n_kpt} sectors, got {len(sizes)} sizes"
            )
        if len(starts) != len(sizes):
            raise StructureError("every sector of an axis has a size and a start")
        if any(size < 0 for size in sizes) or any(start < 0 for start in starts):
            raise StructureError("sector sizes and starts are non-negative")
        for index, block in enumerate(blocks):
            if block.derived_from is not None and any(
                (sizes[index * n_kpt + k], starts[index * n_kpt + k])
                != (
                    sizes[block.derived_from * n_kpt + k],
                    starts[block.derived_from * n_kpt + k],
                )
                for k in range(n_kpt)
            ):
                raise StructureError(
                    "a derived sector holds as many functions as its source"
                )
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        object.__setattr__(self, "_offsets", tuple(offsets))
        for sector in range(len(sizes)):
            image = self.image(Symmetry.TIME_REVERSAL, sector)
            if image is not None and (sizes[image], starts[image]) != (
                sizes[sector],
                starts[sector],
            ):
                raise StructureError(
                    "time reversal relates sectors that hold different numbers of "
                    "functions"
                )

    @property
    def n_kpt(self) -> int:
        """Number of k-points of the full set, one without a k axis."""
        axis = self.structure.k_axis
        return 1 if axis is None else axis.n_kpt

    @property
    def n_sectors(self) -> int:
        """Number of sectors: the blocks of the structure times the k-points."""
        return len(self.sizes)

    @property
    def extent(self) -> int:
        """Number of functions over all sectors, the length of the dense axis."""
        return self._offsets[-1]

    @property
    def modulus(self) -> Momentum | None:
        """The modulus of the momenta of the k axis; ``None`` without one."""
        axis = self.structure.k_axis
        return None if axis is None else axis.modulus

    def offset(self, sector: int) -> int:
        """Position of the first function of ``sector`` in the dense axis."""
        return self._offsets[sector]

    def block_index(self, sector: int) -> int:
        """Index in the structure's blocks of the block of ``sector``."""
        return sector // self.n_kpt

    def kpoint(self, sector: int) -> int:
        """Index in the full k set of the point of ``sector``, zero without a k axis."""
        return sector % self.n_kpt

    def sector(self, block: int, kpoint: int = 0) -> int:
        """Return the sector of ``block`` at ``kpoint``."""
        return block * self.n_kpt + kpoint

    def spin_charge(self, sector: int) -> int:
        """Return the ``S_z`` charge of ``sector``, positive for a ket index."""
        return SPIN_CHARGE[self.structure.blocks[self.block_index(sector)].spin]

    def characters(self, sector: int) -> Characters | None:
        """Return the characters of the irrep of ``sector``, ``None`` without any."""
        irreps = self.structure.irreps
        irrep = self.structure.blocks[self.block_index(sector)].irrep
        if irreps is None or irrep is None:
            return None
        return irreps.characters[irrep]

    def momentum(self, sector: int) -> Momentum | None:
        """Return the crystal momentum of ``sector``, ``None`` without a k axis."""
        axis = self.structure.k_axis
        if axis is None or axis.momenta is None:
            return None
        return axis.momenta[self.kpoint(sector)]

    def image(self, symmetry: Symmetry, sector: int) -> int | None:
        """Return the sector ``symmetry`` maps ``sector`` to, ``None`` for no symmetry.

        The spin flip maps a sector of a restricted structure to its copy
        and leaves a spatial sector alone; it is no symmetry of an
        unrestricted, paired or generalized structure. Time reversal maps a
        sector to its partner point ``−k`` in the block the structure
        derives there, a real or quaternionic block itself and a paired
        block its conjugate partner; without a k axis it is the conjugate
        partner map of a paired structure, the identity on a real
        structure, and no symmetry of a complex one.
        """
        blocks = self.structure.blocks
        block, kpoint = self.block_index(sector), self.kpoint(sector)
        partner = self._partner(block)
        if symmetry is Symmetry.SPIN_FLIP:
            if blocks[block].spin is Spin.NONE:
                return sector
            if self.structure.derived_partner is Partner.COPY:
                return self.sector(partner, kpoint)
            return None
        axis = self.structure.k_axis
        reversed_point = kpoint if axis is None else axis.partners[kpoint]
        if self.structure.derived_partner is Partner.CONJUGATE:
            return self.sector(partner, reversed_point)
        if blocks[block].ring is Ring.REAL:
            return self.sector(block, reversed_point)
        # Time reversal maps the orbitals of a quaternionic block at an
        # invariant point onto their Kramers partners, within the block,
        # which a map of sectors cannot express.
        if (
            blocks[block].ring is Ring.QUATERNION
            and axis is not None
            and not axis.invariant[kpoint]
        ):
            return self.sector(block, reversed_point)
        return None

    def _partner(self, block: int) -> int:
        """Return the block derived from ``block`` or deriving it, else ``block``."""
        derived_from = self.structure.blocks[block].derived_from
        if derived_from is not None:
            return derived_from
        for index, other in enumerate(self.structure.blocks):
            if other.derived_from == block:
                return index
        return block

    def related_sectors(self, sector: int) -> tuple[int, ...]:
        """Return the sectors that hold as many functions as ``sector``, it included.

        The partner block at the same point derives from the block of
        ``sector`` or is derived by it, and the images under the symmetries
        of the state hold copies or conjugates of the same blocks; the set
        is closed under both relations and ordered.
        """
        related = {sector}
        pending = [sector]
        while pending:
            current = pending.pop()
            images: list[int | None] = [
                self.sector(
                    self._partner(self.block_index(current)), self.kpoint(current)
                )
            ]
            images.extend(self.image(symmetry, current) for symmetry in Symmetry)
            for image in images:
                if image is not None and image not in related:
                    related.add(image)
                    pending.append(image)
        return tuple(sorted(related))

    def window(self, sector: int, start: int, stop: int) -> TensorAxis:
        """Restrict the axis to the functions ``start:stop`` of ``sector``.

        The related sectors of ``sector`` are restricted to the same
        functions and every other sector is left empty, so a tensor over
        the window holds the blocks of those functions alone; the
        ``starts`` of the window record the position of the functions
        within the sectors of this axis.
        """
        related = self.related_sectors(sector)
        if not 0 <= start <= stop <= self.sizes[sector]:
            raise StructureError(
                f"a window of a sector of {self.sizes[sector]} functions lies "
                f"within it, got {start}:{stop}"
            )
        return TensorAxis(
            self.structure,
            tuple(stop - start if s in related else 0 for s in range(self.n_sectors)),
            self.conjugated,
            tuple(
                self.starts[s] + start if s in related else self.starts[s]
                for s in range(self.n_sectors)
            ),
        )

    def windows(self, size: int) -> tuple[TensorAxis, ...]:
        """Partition the axis into windows of at most ``size`` functions per sector.

        Each set of related sectors is cut into consecutive windows of
        ``size`` functions, the last one shorter, in the order of the
        sectors; the windows cover every function of the axis once.
        """
        if size < 1:
            raise StructureError("a window holds at least one function")
        result: list[TensorAxis] = []
        seen: set[int] = set()
        for sector in range(self.n_sectors):
            if sector in seen:
                continue
            seen.update(self.related_sectors(sector))
            result.extend(
                self.window(sector, start, min(start + size, self.sizes[sector]))
                for start in range(0, self.sizes[sector], size)
            )
        return tuple(result)

    def is_window_of(self, other: TensorAxis) -> bool:
        """Whether every sector is a slice of the same sector of ``other``.

        An axis is a window of itself; the conjugation must agree.
        """
        return (
            self.structure == other.structure
            and self.conjugated == other.conjugated
            and all(
                start >= other_start and start + size <= other_start + other_size
                for start, size, other_start, other_size in zip(
                    self.starts, self.sizes, other.starts, other.sizes, strict=True
                )
            )
        )

    def conjugate(self) -> TensorAxis:
        """Return the same axis with the opposite conjugation."""
        return TensorAxis(self.structure, self.sizes, not self.conjugated, self.starts)

    def same_space(self, other: TensorAxis) -> bool:
        """Whether both axes run over the same sectors, sizes and windows."""
        return (
            self.structure == other.structure
            and self.sizes == other.sizes
            and self.starts == other.starts
        )


def _check_k_axis(structure: BlockStructure) -> None:
    """Raise unless the k axis of ``structure``, if any, admits tensors."""
    axis = structure.k_axis
    if axis is None:
        return
    if axis.operations is not None:
        raise StructureError(
            "tensors over the irreducible wedge of a space group are a later "
            "extension; build the state over the full k-point set"
        )
    if axis.momenta is None:
        raise StructureError("tensors conserve crystal momentum on a uniform mesh only")


@dataclass(frozen=True, slots=True)
class TensorStructure:
    """The axes of a block tensor and the blocks they allow.

    Parameters
    ----------
    axes
        One :class:`TensorAxis` per index, in order; a tensor without axes
        holds one scalar block. The axes carry irrep sectors of one group
        either all or none, and k axes of one mesh either all or none; a
        symmetry of the state derives blocks when every axis has an image
        under it.
    ring
        Ring the blocks are stored over, ``float64`` or ``complex128``;
        it must be complex when any axis holds complex blocks, which the
        blocks over a k axis are away from the time-reversal invariant
        points.
    """

    axes: tuple[TensorAxis, ...]
    ring: Ring = field(default=Ring.REAL, kw_only=True)
    _generators: tuple[_Generator, ...] = field(init=False, repr=False, compare=False)
    _modulus: Momentum | None = field(init=False, repr=False, compare=False)
    _keys: tuple[Key, ...] = field(init=False, repr=False, compare=False)
    _stored: tuple[Key, ...] = field(init=False, repr=False, compare=False)
    _positions: dict[Key, int] = field(init=False, repr=False, compare=False)
    _derivations: dict[Key, tuple[Key, bool]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        object.__setattr__(self, "axes", axes)
        _check_axes(axes, self.ring)
        object.__setattr__(self, "_generators", _generators(axes))
        object.__setattr__(
            self,
            "_modulus",
            next((axis.modulus for axis in axes if axis.modulus is not None), None),
        )
        keys, stored, positions, derivations = _tables(axes)
        object.__setattr__(self, "_keys", keys)
        object.__setattr__(self, "_stored", stored)
        object.__setattr__(self, "_positions", positions)
        object.__setattr__(self, "_derivations", derivations)

    @property
    def n_axes(self) -> int:
        """Number of indices."""
        return len(self.axes)

    @property
    def symmetries(self) -> tuple[Symmetry, ...]:
        """The symmetries that derive blocks of the tensor from others."""
        return tuple(generator.symmetry for generator in self._generators)

    @property
    def keys(self) -> tuple[Key, ...]:
        """Every block the rule allows with no empty sector, stored or derived."""
        return self._keys

    @property
    def stored_keys(self) -> tuple[Key, ...]:
        """The blocks that hold data, in storage order."""
        return self._stored

    @property
    def n_stored(self) -> int:
        """Number of stored blocks."""
        return len(self._stored)

    def position(self, key: Key) -> int | None:
        """Storage position of ``key``, or ``None`` if it is not stored."""
        return self._positions.get(key)

    def shape(self, key: Key) -> tuple[int, ...]:
        """Shape of the block ``key``."""
        return tuple(axis.sizes[s] for axis, s in zip(self.axes, key, strict=True))

    def is_allowed(self, key: Key) -> bool:
        """Whether the charges of the sectors of ``key`` cancel."""
        if len(key) != len(self.axes):
            raise StructureError(
                f"a key of a {len(self.axes)}-index tensor has {len(self.axes)} "
                f"sectors, got {len(key)}"
            )
        total: _Charge = (0, None, None)
        for axis, s in zip(self.axes, key, strict=True):
            total = _combine(total, _charge(axis, s), self._modulus)
        return _is_neutral(total)

    def is_self_conjugate(self, key: Key) -> bool:
        """Whether a symmetry maps ``key`` onto its own conjugate, so its block is real.

        A real state's blocks at the time-reversal invariant points are
        such blocks; the orbitals there must be real for the derived blocks
        to follow (ADR-0031).
        """
        return any(
            generator.conjugates
            and all(generator.maps[axis][s] == s for axis, s in enumerate(key))
            for generator in self._generators
        )

    def orbit(self, key: Key) -> tuple[tuple[Key, bool], ...]:
        """Return the images of ``key`` under the symmetries, each with its conjugation.

        The key itself comes first, unconjugated; the other entries are
        distinct keys with whether their block is the conjugate of the
        block ``key``. A symmetry that maps ``key`` onto itself adds nothing.
        """
        return _orbit(self._generators, key)

    def derivation(self, key: Key) -> tuple[Key, bool]:
        """Return the stored key the block ``key`` follows from, and whether conjugated.

        The stored member of an orbit is the lexicographically smallest
        key; a block mapped onto itself by a symmetry is its own source.
        The derivations of the allowed keys are computed once.
        """
        cached = self._derivations.get(key)
        return self._derive(key) if cached is None else cached

    def _derive(self, key: Key) -> tuple[Key, bool]:
        return _derive(self._generators, key)

    def conjugated(self) -> TensorStructure:
        """Return the structure of the complex conjugate: every axis conjugated."""
        return TensorStructure(
            tuple(axis.conjugate() for axis in self.axes), ring=self.ring
        )

    def permuted(self, order: Sequence[int]) -> TensorStructure:
        """Reorder the axes: the new axis ``k`` is the old axis ``order[k]``."""
        return TensorStructure(tuple(self.axes[o] for o in order), ring=self.ring)

    def describe(self) -> str:
        """Short human-readable name, for messages."""
        names = sorted({axis.structure.describe() for axis in self.axes})
        return f"{self.n_axes}-index tensor over " + (
            ", ".join(names) if names else "no axes"
        )


@dataclass(frozen=True, slots=True)
class _Generator:
    """A symmetry acting on the keys: the image of every sector of every axis."""

    symmetry: Symmetry
    maps: tuple[tuple[int, ...], ...]
    conjugates: bool


def _generators(axes: tuple[TensorAxis, ...]) -> tuple[_Generator, ...]:
    """Collect the symmetries every axis has an image under that move some sector."""
    generators: list[_Generator] = []
    for symmetry, conjugates in (
        (Symmetry.SPIN_FLIP, False),
        (Symmetry.TIME_REVERSAL, True),
    ):
        maps: list[tuple[int, ...]] = []
        for axis in axes:
            images = [axis.image(symmetry, s) for s in range(axis.n_sectors)]
            if any(image is None for image in images):
                break
            maps.append(tuple(image for image in images if image is not None))
        else:
            if any(image != s for row in maps for s, image in enumerate(row)):
                generators.append(_Generator(symmetry, tuple(maps), conjugates))
    return tuple(generators)


def _charge(axis: TensorAxis, sector: int) -> _Charge:
    """Return the signed charges of ``sector`` on ``axis``."""
    sign = -1 if axis.conjugated else 1
    momentum, modulus = axis.momentum(sector), axis.modulus
    if momentum is not None and modulus is not None:
        momentum = tuple((sign * m) % n for m, n in zip(momentum, modulus, strict=True))
    return (sign * axis.spin_charge(sector), axis.characters(sector), momentum)


def _combine(a: _Charge, b: _Charge, modulus: Momentum | None) -> _Charge:
    """Add two charges: spins add, characters multiply, momenta add modulo."""
    characters = (
        a[1]
        if b[1] is None
        else b[1]
        if a[1] is None
        else product_characters([a[1], b[1]])
    )
    if a[2] is None or b[2] is None:
        momentum = a[2] if b[2] is None else b[2]
    else:
        assert modulus is not None
        momentum = tuple(
            (x + y) % n for x, y, n in zip(a[2], b[2], modulus, strict=True)
        )
    return (a[0] + b[0], characters, momentum)


def _negated(charge: _Charge, modulus: Momentum | None) -> _Charge:
    """Return the charge that cancels ``charge``; characters are their own inverse."""
    momentum = charge[2]
    if momentum is not None:
        assert modulus is not None
        momentum = tuple((-m) % n for m, n in zip(momentum, modulus, strict=True))
    return (-charge[0], charge[1], momentum)


def _is_neutral(charge: _Charge) -> bool:
    return (
        charge[0] == 0
        and (charge[1] is None or is_totally_symmetric(charge[1]))
        and (charge[2] is None or not any(charge[2]))
    )


def _orbit(
    generators: tuple[_Generator, ...], key: Key
) -> tuple[tuple[Key, bool], ...]:
    images: dict[Key, bool] = {key: False}
    for generator in generators:
        for current, conjugated in list(images.items()):
            moved = tuple(generator.maps[axis][s] for axis, s in enumerate(current))
            images.setdefault(moved, conjugated != generator.conjugates)
    return tuple(images.items())


def _derive(generators: tuple[_Generator, ...], key: Key) -> tuple[Key, bool]:
    orbit = _orbit(generators, key)
    source = min(image for image, _ in orbit)
    if source == key:
        return key, False
    for image, conjugated in orbit:
        if image == source:
            return source, conjugated
    raise AssertionError("the source is in the orbit")


@functools.lru_cache(maxsize=1024)
def _tables(
    axes: tuple[TensorAxis, ...],
) -> tuple[
    tuple[Key, ...], tuple[Key, ...], dict[Key, int], dict[Key, tuple[Key, bool]]
]:
    """Enumerate the keys, stored keys, positions and derivations of ``axes``.

    Structures over the same axes recur at every iteration of a correlated
    method, so the tables are kept for the last ones; the tables are
    shared, never mutated.
    """
    generators = _generators(axes)
    keys = _allowed_keys(axes)
    derivations = {key: _derive(generators, key) for key in keys}
    stored = tuple(key for key in keys if derivations[key][0] == key)
    positions = {key: index for index, key in enumerate(stored)}
    return keys, stored, positions, derivations


def _allowed_keys(axes: tuple[TensorAxis, ...]) -> tuple[Key, ...]:
    """Enumerate the keys whose charges cancel, with no empty sector, in order.

    The sectors of the last axis are looked up by the charge the other
    sectors leave to cancel, so the cost is that of the product over the
    other axes times the matches rather than of the full product.
    """
    if not axes:
        return ((),)
    modulus = next((axis.modulus for axis in axes if axis.modulus is not None), None)
    last = axes[-1]
    by_charge: dict[_Charge, list[int]] = {}
    for s in range(last.n_sectors):
        if last.sizes[s] > 0:
            by_charge.setdefault(_charge(last, s), []).append(s)
    # The charges of the other axes may lack a component the last axis
    # carries; the neutral element of that component fills it in.
    neutral: _Charge = (
        0,
        None
        if last.characters(0) is None
        else tuple([1] * len(last.characters(0) or ())),
        None if modulus is None else tuple([0] * len(modulus)),
    )
    charges = [
        {s: _charge(axis, s) for s in range(axis.n_sectors) if axis.sizes[s] > 0}
        for axis in axes[:-1]
    ]
    keys: list[Key] = []
    for prefix in itertools.product(*[list(table) for table in charges]):
        total = neutral
        for table, s in zip(charges, prefix, strict=True):
            total = _combine(total, table[s], modulus)
        keys.extend((*prefix, s) for s in by_charge.get(_negated(total, modulus), []))
    return tuple(keys)


def _check_axes(axes: tuple[TensorAxis, ...], ring: Ring) -> None:
    """Validate the axes of a structure against each other and the ring."""
    if ring is Ring.QUATERNION:
        raise StructureError("tensors are stored over the real or the complex ring")
    if ring is Ring.REAL and not all(_is_real(axis) for axis in axes):
        raise StructureError(
            "an axis holds complex blocks; build the structure with ring=Ring.COMPLEX"
        )
    sectors = [axis.structure.irreps for axis in axes]
    present = [s for s in sectors if s is not None]
    if present and len(present) != len(sectors):
        raise StructureError("every axis carries irrep sectors, or none does")
    if present and any(
        s.group != present[0].group or s.n_operations != present[0].n_operations
        for s in present
    ):
        raise StructureError("the irrep sectors of every axis belong to one group")
    moduli = [axis.modulus for axis in axes]
    meshes = [m for m in moduli if m is not None]
    if meshes and len(meshes) != len(moduli):
        raise StructureError("every axis runs over k-points, or none does")
    if meshes and any(m != meshes[0] for m in meshes):
        raise StructureError("the k axes of every axis belong to one mesh")


def _is_real(axis: TensorAxis) -> bool:
    """Whether every block of the axis is stored as real."""
    structure = axis.structure
    k_axis = structure.k_axis
    return not structure.is_complex and (k_axis is None or k_axis.all_invariant)


@dataclass(frozen=True, slots=True, eq=False)
class BlockTensor:
    """A block-sparse tensor in a :class:`TensorStructure`.

    ``blocks`` holds one array per stored key of ``structure``, in storage
    order, each of the shape the sectors of its key give and of the dtype
    of the structure's ring. Derived blocks are materialized on request by
    :meth:`block`.
    """

    structure: TensorStructure
    blocks: tuple[Array, ...]

    def __post_init__(self) -> None:
        stored = self.structure.stored_keys
        if len(self.blocks) != len(stored):
            raise StructureError(
                f"the {self.structure.describe()} stores {len(stored)} blocks, "
                f"got {len(self.blocks)}"
            )
        if not self.blocks:
            return
        xp = array_namespace(*self.blocks)
        dtype = dtype_for(xp, self.structure.ring)
        for key, array in zip(stored, self.blocks, strict=True):
            if tuple(int(n) for n in array.shape) != self.structure.shape(key):
                raise StructureError(
                    f"block {key} has shape {list(self.structure.shape(key))}, got "
                    f"{list(array.shape)}"
                )
            if array.dtype != dtype:
                expected = (
                    "float64" if self.structure.ring is Ring.REAL else "complex128"
                )
                raise StructureError(
                    f"blocks over the {self.structure.ring.value} ring must be "
                    f"{expected}, got {array.dtype}"
                )

    # ------------------------------------------------------------------ building

    @classmethod
    def from_blocks(
        cls,
        structure: TensorStructure,
        blocks: Mapping[Key, Array],
        *,
        xp: Namespace | None = None,
    ) -> BlockTensor:
        """Build a tensor from stored blocks by key; blocks left out are zero.

        Real blocks are cast to complex when the structure's ring is
        complex. Keys the structure does not store, because they vanish by
        symmetry or are derived from another block, are rejected.
        """
        for key in blocks:
            if structure.position(key) is None:
                source = (
                    f"is derived from {structure.derivation(key)[0]}"
                    if structure.is_allowed(key)
                    else "vanishes by symmetry or has an empty sector"
                )
                raise StructureError(f"block {key} is not stored: it {source}")
        if xp is None:
            xp = array_namespace(*blocks.values()) if blocks else default_namespace()
        dtype = dtype_for(xp, structure.ring)
        arrays: list[Array] = []
        for key in structure.stored_keys:
            array = blocks.get(key)
            if array is None:
                arrays.append(xp.zeros(structure.shape(key), dtype=dtype))
                continue
            if array.dtype != dtype:
                if xp.isdtype(array.dtype, "complex floating"):
                    raise StructureError(
                        "a block has complex entries; build the structure with "
                        "ring=Ring.COMPLEX"
                    )
                array = xp.astype(array, dtype)
            arrays.append(array)
        return cls(structure, tuple(arrays))

    @classmethod
    def from_function(
        cls, structure: TensorStructure, builder: Callable[[Key], Array]
    ) -> BlockTensor:
        """Build a tensor with ``builder(key)`` as every stored block."""
        return cls.from_blocks(
            structure, {key: builder(key) for key in structure.stored_keys}
        )

    @classmethod
    def zeros(
        cls, structure: TensorStructure, *, xp: Namespace | None = None
    ) -> BlockTensor:
        """Build the zero tensor."""
        return cls.from_blocks(structure, {}, xp=xp)

    @property
    def n_axes(self) -> int:
        """Number of indices."""
        return self.structure.n_axes

    @property
    def scalar(self) -> Array:
        """The value of a tensor without axes, as a 0-d array."""
        if self.n_axes:
            raise StructureError("only a tensor without axes is a scalar")
        return self.blocks[0]

    def block(self, key: Key) -> Array:
        """Return the block ``key``, materialized from its source if derived.

        Raises
        ------
        StructureError
            If the block vanishes by symmetry or has an empty sector.
        """
        position = self.structure.position(key)
        if position is not None:
            return self.blocks[position]
        if not self.structure.is_allowed(key) or any(
            n == 0 for n in self.structure.shape(key)
        ):
            raise StructureError(
                f"block {key} vanishes by symmetry or has an empty sector"
            )
        source, conjugate = self.structure.derivation(key)
        position = self.structure.position(source)
        if position is None:
            raise StructureError(
                f"block {key} derives from {source}, which is not stored"
            )
        array = self.blocks[position]
        if not conjugate:
            return array
        xp = array_namespace(array)
        return xp.conj(array)

    def map_blocks(self, function: Callable[[Array], Array]) -> BlockTensor:
        """Apply ``function`` to every stored block."""
        return BlockTensor(
            self.structure, tuple(function(block) for block in self.blocks)
        )

    def window(self, position: int, axis: TensorAxis) -> BlockTensor:
        """Restrict the index ``position`` to ``axis``, a window of its axis.

        The blocks over the window are views of the stored blocks of this
        tensor sliced along ``position``, conjugated after the slice when
        the block derives from its source by conjugation.

        Raises
        ------
        StructureError
            If ``axis`` is not a window of the axis at ``position``
            (:meth:`TensorAxis.is_window_of`).
        """
        full = self.structure.axes[position]
        if not axis.is_window_of(full):
            raise StructureError(
                f"the axis is not a window of index {position} of the "
                f"{self.structure.describe()}"
            )
        axes = list(self.structure.axes)
        axes[position] = axis
        structure = TensorStructure(tuple(axes), ring=self.structure.ring)
        if not self.blocks:
            return BlockTensor.from_blocks(structure, {})
        xp = array_namespace(*self.blocks)
        blocks: dict[Key, Array] = {}
        for key in structure.stored_keys:
            # The related sectors of the window share its slice, so the
            # slice of the key applies to its source.
            source, conjugate = self.structure.derivation(key)
            stored = self.structure.position(source)
            assert stored is not None, "the source of an allowed key is stored"
            sector = key[position]
            start = axis.starts[sector] - full.starts[sector]
            slices = [slice(None)] * self.n_axes
            slices[position] = slice(start, start + axis.sizes[sector])
            array = self.blocks[stored][tuple(slices)]
            blocks[key] = xp.conj(array) if conjugate else array
        return BlockTensor.from_blocks(structure, blocks, xp=xp)

    # ------------------------------------------------------------------ algebra

    def conj(self) -> BlockTensor:
        """Return the complex conjugate: every block conjugated, every axis flipped."""
        if not self.blocks:
            return BlockTensor(self.structure.conjugated(), ())
        xp = array_namespace(*self.blocks)
        return BlockTensor(
            self.structure.conjugated(), tuple(xp.conj(block) for block in self.blocks)
        )

    def permute(self, subscripts: str) -> BlockTensor:
        """Reorder the axes as ``"iajb->ijab"`` says, one letter per axis."""
        order = _parse_permutation(subscripts, self.n_axes)
        structure = self.structure.permuted(order)
        if not self.blocks:
            return BlockTensor(structure, ())
        xp = array_namespace(*self.blocks)
        blocks: dict[Key, Array] = {}
        for key in structure.stored_keys:
            source = [0] * self.n_axes
            for new, old in enumerate(order):
                source[old] = key[new]
            blocks[key] = xp.permute_dims(self.block(tuple(source)), order)
        return BlockTensor.from_blocks(structure, blocks, xp=xp)

    def _require_same_structure(self, other: BlockTensor, operation: str) -> None:
        if self.structure != other.structure:
            raise StructureError(
                f"cannot {operation} tensors of different structures: "
                f"{self.structure.describe()} and {other.structure.describe()}"
            )

    def _zip(
        self,
        other: BlockTensor,
        operation: str,
        function: Callable[[Array, Array], Array],
    ) -> BlockTensor:
        self._require_same_structure(other, operation)
        return BlockTensor(
            self.structure,
            tuple(
                function(a, b) for a, b in zip(self.blocks, other.blocks, strict=True)
            ),
        )

    def __add__(self, other: BlockTensor) -> BlockTensor:
        return self._zip(other, "add", lambda a, b: a + b)

    def __sub__(self, other: BlockTensor) -> BlockTensor:
        return self._zip(other, "subtract", lambda a, b: a - b)

    def __neg__(self) -> BlockTensor:
        return self.map_blocks(lambda block: -block)

    def __mul__(self, other: Scalar | BlockTensor) -> BlockTensor:
        """Elementwise product with a tensor of the same structure, or a scalar."""
        if isinstance(other, BlockTensor):
            return self._zip(other, "multiply", lambda a, b: a * b)
        return self.map_blocks(lambda block: block * other)

    def __rmul__(self, other: Scalar) -> BlockTensor:
        return self.map_blocks(lambda block: other * block)

    def __truediv__(self, other: Scalar | BlockTensor) -> BlockTensor:
        """Elementwise quotient by a tensor of the same structure, or a scalar."""
        if isinstance(other, BlockTensor):
            return self._zip(other, "divide", lambda a, b: a / b)
        return self.map_blocks(lambda block: block / other)

    def sum(self) -> Array:
        """Sum every element over all blocks, derived ones included.

        Returned as a 0-d array.
        """
        xp = array_namespace(*self.blocks) if self.blocks else default_namespace()
        total = xp.zeros((), dtype=dtype_for(xp, self.structure.ring))
        for key, block in zip(self.structure.stored_keys, self.blocks, strict=True):
            part = xp.sum(block)
            for _, conjugated in self.structure.orbit(key):
                total = total + (xp.conj(part) if conjugated else part)
        return total

    # ------------------------------------------------------------------ dense form

    def to_dense(self) -> Array:
        """Expand into one dense array, the sectors of every axis in order."""
        xp = array_namespace(*self.blocks) if self.blocks else default_namespace()
        return self._dense(xp, ())

    def _dense(self, xp: Namespace, prefix: Key) -> Array:
        structure = self.structure
        if len(prefix) == structure.n_axes:
            if structure.is_allowed(prefix) and all(structure.shape(prefix)):
                return self.block(prefix)
            return xp.zeros(
                structure.shape(prefix), dtype=dtype_for(xp, structure.ring)
            )
        axis = structure.axes[len(prefix)]
        parts = [self._dense(xp, (*prefix, s)) for s in range(axis.n_sectors)]
        return xp.concat(parts, axis=len(prefix))

    @classmethod
    def from_dense(
        cls,
        structure: TensorStructure,
        dense: Array,
        *,
        check: bool = True,
        tolerance: float = DENSE_TOLERANCE,
    ) -> BlockTensor:
        """Read the stored blocks of ``structure`` off a dense array.

        With ``check`` the array must be a member of the structure to
        within ``tolerance`` relative to its largest entry: the blocks the
        rule forbids vanish, the derived blocks follow from their sources,
        and the blocks a symmetry maps onto their own conjugate are real.
        Without it the other blocks are dropped and those blocks keep
        their imaginary parts.
        """
        shape = tuple(axis.extent for axis in structure.axes)
        if tuple(int(n) for n in dense.shape) != shape:
            raise StructureError(
                f"a dense {structure.describe()} has shape {list(shape)}, got "
                f"{list(dense.shape)}"
            )
        xp = array_namespace(dense)
        blocks: dict[Key, Array] = {}
        for key in structure.stored_keys:
            slices = tuple(
                slice(axis.offset(s), axis.offset(s) + axis.sizes[s])
                for axis, s in zip(structure.axes, key, strict=True)
            )
            blocks[key] = dense[slices]
        result = cls.from_blocks(structure, blocks, xp=xp)
        if check and dense.size:
            scale = max(1.0, float(xp.max(xp.abs(dense))))
            deviation = float(xp.max(xp.abs(result.to_dense() - dense)))
            if deviation > tolerance * scale:
                raise StructureError(
                    "the dense tensor is not a member of the structure: a block "
                    "the rule forbids is nonzero or a derived block does not follow "
                    f"from its source (deviation {deviation:.2e})"
                )
            for key, block in zip(structure.stored_keys, result.blocks, strict=True):
                if structure.is_self_conjugate(key) and (
                    float(xp.max(xp.abs(xp.imag(block)))) > tolerance * scale
                ):
                    raise StructureError(
                        f"block {key} is mapped onto its own conjugate by a symmetry "
                        "of the structure and must be real"
                    )
        return result


# -------------------------------------------------------------------- contraction


def contract(subscripts: str, first: BlockTensor, second: BlockTensor) -> BlockTensor:
    """Contract two tensors as ``subscripts`` says, in einsum notation.

    ``"Pia,Pbj->iajb"`` names the axes of both operands and of the result
    with one letter each. A letter of the result appears in exactly one
    operand; every other letter appears in both and is summed, and must
    pair a conjugated with an unconjugated axis over the same sectors, as
    the completeness of an orbital set requires; the two operands are not
    summed over a letter they share with the result. Chains of
    contractions are nested calls. The result is complex when either
    operand is.
    """
    letters_first, letters_second, output = _parse_contraction(
        subscripts, first.n_axes, second.n_axes
    )
    contracted = [letter for letter in letters_first if letter in letters_second]
    sources: list[tuple[int, int]] = []
    for letter in output:
        in_first, in_second = letter in letters_first, letter in letters_second
        if in_first == in_second:
            raise StructureError(
                f"index {letter!r} of the result appears in exactly one operand"
            )
        sources.append(
            (0, letters_first.index(letter))
            if in_first
            else (1, letters_second.index(letter))
        )
    for letter in (*letters_first, *letters_second):
        if letter not in output and letter not in contracted:
            raise StructureError(
                f"index {letter!r} is neither summed with the other operand nor "
                "kept in the result"
            )
    for letter in contracted:
        if letter in output:
            raise StructureError(
                f"index {letter!r} appears in both operands and in the result"
            )
        a = first.structure.axes[letters_first.index(letter)]
        b = second.structure.axes[letters_second.index(letter)]
        if a.conjugated == b.conjugated:
            raise StructureError(
                f"summing over {letter!r} pairs a conjugated with an unconjugated "
                "index; conjugate one operand"
            )
        if not a.same_space(b):
            raise StructureError(
                f"the operands run over different sectors along {letter!r}"
            )
    axes = tuple(
        (first if operand == 0 else second).structure.axes[position]
        for operand, position in sources
    )
    ring = (
        Ring.COMPLEX
        if Ring.COMPLEX in (first.structure.ring, second.structure.ring)
        else Ring.REAL
    )
    structure = TensorStructure(axes, ring=ring)
    positions = (
        [letters_first.index(letter) for letter in contracted],
        [letters_second.index(letter) for letter in contracted],
    )
    # tensordot leaves the free axes of its first operand, then the second's;
    # the operands go in the order that spares a permutation of every product.
    free = (
        [(0, k) for k in range(first.n_axes) if k not in positions[0]],
        [(1, k) for k in range(second.n_axes) if k not in positions[1]],
    )
    order = tuple((free[0] + free[1]).index(source) for source in sources)
    swapped = tuple((free[1] + free[0]).index(source) for source in sources)
    identity = tuple(range(len(order)))
    operand_order = (0, 1) if order == identity or swapped != identity else (1, 0)
    return BlockTensor.from_blocks(
        structure,
        _accumulate(
            first,
            second,
            positions,
            (order if operand_order == (0, 1) else swapped, operand_order),
            sources,
            structure,
        ),
        xp=array_namespace(*first.blocks, *second.blocks)
        if first.blocks or second.blocks
        else None,
    )


def _accumulate(
    first: BlockTensor,
    second: BlockTensor,
    positions: tuple[list[int], list[int]],
    product: tuple[tuple[int, ...], tuple[int, int]],
    sources: list[tuple[int, int]],
    structure: TensorStructure,
) -> dict[Key, Array]:
    """Sum the block products of a contraction into the stored result keys.

    ``positions`` holds the axes of the contracted letters in each operand;
    ``product`` the permutation from the axes ``tensordot`` leaves to the
    result's and the order in which the operands enter it. Each product
    is one ``tensordot``, a matrix product.
    """
    positions_first, positions_second = positions
    order, operand_order = product
    identity = tuple(range(len(order)))
    by_contracted: dict[Key, list[Key]] = {}
    for key in second.structure.keys:
        by_contracted.setdefault(tuple(key[p] for p in positions_second), []).append(
            key
        )
    if not first.blocks or not second.blocks:
        return {}
    xp = array_namespace(*first.blocks, *second.blocks)
    accumulated: dict[Key, Array] = {}
    second_blocks: dict[Key, Array] = {}
    for key_first in first.structure.keys:
        matches = by_contracted.get(tuple(key_first[p] for p in positions_first))
        if not matches:
            continue
        block_first = first.block(key_first)
        for key_second in matches:
            result_key = tuple(
                key_first[position] if operand == 0 else key_second[position]
                for operand, position in sources
            )
            if structure.derivation(result_key)[0] != result_key:
                continue
            block_second = second_blocks.get(key_second)
            if block_second is None:
                block_second = second_blocks[key_second] = second.block(key_second)
            value = (
                xp.tensordot(
                    block_first, block_second, axes=(positions_first, positions_second)
                )
                if operand_order == (0, 1)
                else xp.tensordot(
                    block_second, block_first, axes=(positions_second, positions_first)
                )
            )
            if order != identity:
                value = xp.permute_dims(value, order)
            previous = accumulated.get(result_key)
            accumulated[result_key] = value if previous is None else previous + value
    return accumulated


def _parse_permutation(subscripts: str, n_axes: int) -> tuple[int, ...]:
    """Return ``order`` with the new axis ``k`` the old axis ``order[k]``."""
    source, target = _split_arrow(subscripts)
    if len(source) != n_axes or len(set(source)) != n_axes:
        raise StructureError(
            f"a permutation of a {n_axes}-index tensor names {n_axes} distinct "
            f"axes, got {source!r}"
        )
    if sorted(target) != sorted(source):
        raise StructureError(
            f"the permutation {subscripts!r} does not reorder the same axes"
        )
    return tuple(source.index(letter) for letter in target)


def _parse_contraction(
    subscripts: str, n_first: int, n_second: int
) -> tuple[str, str, str]:
    """Split ``"ab,bc->ac"`` into the letters of both operands and the result."""
    inputs, output = _split_arrow(subscripts)
    parts = inputs.split(",")
    if len(parts) != 2:
        raise StructureError(
            f"a contraction takes two operands, got {subscripts!r}; nest calls for "
            "longer chains"
        )
    for letters, n in zip(parts, (n_first, n_second), strict=True):
        if len(letters) != n or len(set(letters)) != n:
            raise StructureError(
                f"an operand with {n} axes needs {n} distinct letters, got {letters!r}"
            )
    if len(set(output)) != len(output):
        raise StructureError(f"the result of {subscripts!r} repeats an index")
    return parts[0], parts[1], output


def _split_arrow(subscripts: str) -> tuple[str, str]:
    cleaned = subscripts.replace(" ", "")
    if cleaned.count("->") != 1:
        raise StructureError(f"subscripts need one arrow, got {subscripts!r}")
    before, after = cleaned.split("->")
    if not all(letter.isalpha() for letter in before.replace(",", "") + after):
        raise StructureError(f"subscripts are letters, got {subscripts!r}")
    return before, after

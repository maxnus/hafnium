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
index, sum to zero, and the characters of the irreps multiply to the
totally symmetric irrep. Blocks that vanish by this rule, and blocks with
an empty sector, are never stored.

The partner rules of the structures derive the remaining blocks: mapping
every index to its partner block at once, alpha to beta and back, is a
symmetry of the state, so a block and its image are copies or complex
conjugates of each other, and the tensor stores the lexicographically
smaller key of every pair. A contraction pairs a conjugated with an
unconjugated index, as a sum over a complete set of orbitals does, and
the charge rule of its result follows from those of its operands.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
        spatial structure over the irrep sectors of the auxiliary basis for
        a density-fitting index. Structures over a k axis wait for the
        periodic correlated methods.
    sizes
        Number of functions in every sector, one per block of the
        structure in its order; a derived block holds as many as its
        source.
    conjugated
        Whether the index enters conjugated, as the bra index of an
        integral does; the charge of its sector then counts negatively.
    """

    structure: BlockStructure
    sizes: tuple[int, ...]
    conjugated: bool = False
    _partners: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _offsets: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sizes = tuple(int(size) for size in self.sizes)
        object.__setattr__(self, "sizes", sizes)
        blocks = self.structure.blocks
        if self.structure.k_axis is not None:
            raise StructureError(
                "tensors over k-points come with the periodic correlated methods "
                "(M8, part two)"
            )
        if len(sizes) != len(blocks):
            raise StructureError(
                f"an axis over the {self.structure.describe()} structure has "
                f"{len(blocks)} sectors, got {len(sizes)} sizes"
            )
        if any(size < 0 for size in sizes):
            raise StructureError("sector sizes are non-negative")
        partners = list(range(len(blocks)))
        for index, block in enumerate(blocks):
            if block.derived_from is None:
                continue
            if sizes[index] != sizes[block.derived_from]:
                raise StructureError(
                    "a derived sector holds as many functions as its source"
                )
            partners[index] = block.derived_from
            partners[block.derived_from] = index
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        object.__setattr__(self, "_partners", tuple(partners))
        object.__setattr__(self, "_offsets", tuple(offsets))

    @property
    def n_sectors(self) -> int:
        """Number of sectors, the number of blocks of the structure."""
        return len(self.sizes)

    @property
    def extent(self) -> int:
        """Number of functions over all sectors, the length of the dense axis."""
        return self._offsets[-1]

    def offset(self, sector: int) -> int:
        """Position of the first function of ``sector`` in the dense axis."""
        return self._offsets[sector]

    def partner(self, sector: int) -> int:
        """Return the sector the partner rule of the structure maps ``sector`` to.

        A stored block maps to the block derived from it and back; a block
        without a partner maps to itself.
        """
        return self._partners[sector]

    def spin_charge(self, sector: int) -> int:
        """Return the ``S_z`` charge of ``sector``, positive for a ket index."""
        return SPIN_CHARGE[self.structure.blocks[sector].spin]

    def characters(self, sector: int) -> Characters | None:
        """Return the characters of the irrep of ``sector``, ``None`` without any."""
        irreps = self.structure.irreps
        irrep = self.structure.blocks[sector].irrep
        if irreps is None or irrep is None:
            return None
        return irreps.characters[irrep]

    def conjugate(self) -> TensorAxis:
        """Return the same axis with the opposite conjugation."""
        return TensorAxis(self.structure, self.sizes, not self.conjugated)

    def same_space(self, other: TensorAxis) -> bool:
        """Whether both axes run over the same sectors and sizes."""
        return self.structure == other.structure and self.sizes == other.sizes


@dataclass(frozen=True, slots=True)
class TensorStructure:
    """The axes of a block tensor and the blocks they allow.

    Parameters
    ----------
    axes
        One :class:`TensorAxis` per index, in order; a tensor without axes
        holds one scalar block. The axes share the partner rule of their
        structures, so that the simultaneous map to the partner sectors is
        a symmetry of every block, and carry irrep sectors of one group
        either all or none.
    ring
        Ring the blocks are stored over, ``float64`` or ``complex128``;
        it must be complex when any axis holds complex blocks.
    """

    axes: tuple[TensorAxis, ...]
    ring: Ring = field(default=Ring.REAL, kw_only=True)
    _partner: Partner | None = field(init=False, repr=False, compare=False)
    _keys: tuple[Key, ...] = field(init=False, repr=False, compare=False)
    _stored: tuple[Key, ...] = field(init=False, repr=False, compare=False)
    _positions: dict[Key, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "_partner", _check_axes(axes, self.ring))
        keys = tuple(
            key
            for key in itertools.product(
                *[
                    [s for s in range(axis.n_sectors) if axis.sizes[s] > 0]
                    for axis in axes
                ]
            )
            if self.is_allowed(key)
        )
        stored = tuple(key for key in keys if self.derivation(key)[0] == key)
        object.__setattr__(self, "_keys", keys)
        object.__setattr__(self, "_stored", stored)
        object.__setattr__(
            self, "_positions", {key: index for index, key in enumerate(stored)}
        )

    @property
    def n_axes(self) -> int:
        """Number of indices."""
        return len(self.axes)

    @property
    def partner(self) -> Partner | None:
        """How a derived block follows from its source; ``None`` without partners."""
        return self._partner

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
        spin = 0
        rows: list[Characters] = []
        for axis, s in zip(self.axes, key, strict=True):
            spin += -axis.spin_charge(s) if axis.conjugated else axis.spin_charge(s)
            characters = axis.characters(s)
            if characters is not None:
                rows.append(characters)
        if spin != 0:
            return False
        return not rows or is_totally_symmetric(product_characters(rows))

    def flip(self, key: Key) -> Key:
        """Map every sector of ``key`` to its partner at once."""
        return tuple(axis.partner(s) for axis, s in zip(self.axes, key, strict=True))

    def derivation(self, key: Key) -> tuple[Key, bool]:
        """Return the stored key the block ``key`` follows from, and whether conjugated.

        The stored member of a pair is the lexicographically smaller key;
        a block whose image is itself is stored.
        """
        image = self.flip(key)
        if image < key:
            return image, self._partner is Partner.CONJUGATE
        return key, False

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


def _check_axes(axes: tuple[TensorAxis, ...], ring: Ring) -> Partner | None:
    """Validate the axes of a structure and return their shared partner rule."""
    if ring is Ring.QUATERNION:
        raise StructureError("tensors are stored over the real or the complex ring")
    if ring is Ring.REAL and any(axis.structure.is_complex for axis in axes):
        raise StructureError(
            "an axis holds complex blocks; build the structure with ring=Ring.COMPLEX"
        )
    rules = {
        axis.structure.derived_partner
        for axis in axes
        if axis.structure.derived_partner is not None
    }
    if len(rules) > 1:
        raise StructureError(
            "the axes of a tensor share one partner rule; copies and conjugates "
            "cannot be mixed"
        )
    partner = rules.pop() if rules else None
    if partner is Partner.CONJUGATE and any(
        axis.structure.derived_partner is None and axis.structure.is_complex
        for axis in axes
    ):
        raise StructureError(
            "a conjugate partner rule derives blocks by complex conjugation, so "
            "every axis without partners must hold real blocks"
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
    return partner


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
        array = self.blocks[self.structure.position(source) or 0]
        if not conjugate:
            return array
        xp = array_namespace(array)
        return xp.conj(array)

    def map_blocks(self, function: Callable[[Array], Array]) -> BlockTensor:
        """Apply ``function`` to every stored block."""
        return BlockTensor(
            self.structure, tuple(function(block) for block in self.blocks)
        )

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
        conjugate = self.structure.partner is Partner.CONJUGATE
        for key, block in zip(self.structure.stored_keys, self.blocks, strict=True):
            part = xp.sum(block)
            total = total + part
            if self.structure.flip(key) != key:
                total = total + (xp.conj(part) if conjugate else part)
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
        rule forbids vanish and the derived blocks follow from their
        sources. Without it the other blocks are dropped.
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
    return BlockTensor.from_blocks(
        structure,
        _accumulate(subscripts, first, second, contracted, sources, structure),
        xp=array_namespace(*first.blocks, *second.blocks)
        if first.blocks or second.blocks
        else None,
    )


def _accumulate(
    subscripts: str,
    first: BlockTensor,
    second: BlockTensor,
    contracted: list[str],
    sources: list[tuple[int, int]],
    structure: TensorStructure,
) -> dict[Key, Array]:
    """Sum the block products of a contraction into the stored result keys."""
    letters_first, letters_second, _ = _parse_contraction(
        subscripts, first.n_axes, second.n_axes
    )
    positions_first = [letters_first.index(letter) for letter in contracted]
    positions_second = [letters_second.index(letter) for letter in contracted]
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
            value = xp.einsum(subscripts, block_first, block_second)
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

"""Abelian point groups in the standard frame: sign operations and parity irreps.

The point groups a calculation can impose are D2h and its subgroups
(PLAN.md §5.8). In the standard frame (CONVENTIONS.md §3, ADR-0022) every
operation of these groups is a diagonal matrix of signs: the identity,
the twofold rotations about the axes, the inversion and the reflections in
the coordinate planes. A function with a definite parity under each of the
three coordinate reflections, such as a real solid harmonic, spans one
irrep, and the character of that irrep under an operation is the product
of the operation's signs raised to the parities. Every irrep is therefore
represented by a parity vector, and the direct product of two irreps is
the irrep of the sum of their parities modulo two.

Labels are the Mulliken symbols in Cotton's order; the operations of each
group are listed in Cotton's order as well, and agree with the tables of
PySCF (``pyscf.symm.param``) up to that order.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hafnium.errors import InputError, SymmetryError

Parity = tuple[int, int, int]
"""Parity (0 even, 1 odd) under the reflections ``x → −x``, ``y → −y``, ``z → −z``."""


class Operation(Enum):
    """A symmetry operation of D2h, a diagonal sign matrix in the standard frame.

    The value is the Cotton symbol; :attr:`signs` is the diagonal.
    """

    IDENTITY = "E"
    C2_Z = "C2(z)"
    C2_Y = "C2(y)"
    C2_X = "C2(x)"
    INVERSION = "i"
    SIGMA_XY = "σ(xy)"
    """Reflection in the ``xy`` plane, ``z → −z``."""
    SIGMA_XZ = "σ(xz)"
    """Reflection in the ``xz`` plane, ``y → −y``."""
    SIGMA_YZ = "σ(yz)"
    """Reflection in the ``yz`` plane, ``x → −x``."""

    @property
    def signs(self) -> tuple[int, int, int]:
        """Diagonal of the operation's matrix, ``(s_x, s_y, s_z)``."""
        return _SIGNS[self]

    @property
    def symbol(self) -> str:
        """Cotton's symbol of the operation."""
        return self.value

    def character(self, parity: Parity) -> int:
        """Return ``±1``: how a function of ``parity`` transforms under it."""
        result = 1
        for sign, odd in zip(self.signs, parity, strict=True):
            if odd:
                result *= sign
        return result

    def compose(self, other: Operation) -> Operation:
        """Return the product of two operations, their signs multiplied."""
        signs = tuple(a * b for a, b in zip(self.signs, other.signs, strict=True))
        return _BY_SIGNS[signs]


_SIGNS: dict[Operation, tuple[int, int, int]] = {
    Operation.IDENTITY: (1, 1, 1),
    Operation.C2_Z: (-1, -1, 1),
    Operation.C2_Y: (-1, 1, -1),
    Operation.C2_X: (1, -1, -1),
    Operation.INVERSION: (-1, -1, -1),
    Operation.SIGMA_XY: (1, 1, -1),
    Operation.SIGMA_XZ: (1, -1, 1),
    Operation.SIGMA_YZ: (-1, 1, 1),
}
_BY_SIGNS: dict[tuple[int, ...], Operation] = {
    signs: operation for operation, signs in _SIGNS.items()
}


@dataclass(frozen=True, slots=True)
class Irrep:
    """An irreducible representation: its Mulliken label and a parity spanning it.

    The parity is one representative; within a group several parities span
    the same irrep when the group lacks the reflections that tell them
    apart (``z`` and ``1`` both span ``A1`` of C2v).
    """

    label: str
    parity: Parity


@dataclass(frozen=True, slots=True)
class PointGroup:
    """An Abelian point group in the standard frame.

    Build one with :meth:`named`; the constructor validates that the
    operations form a group and that the irreps are its distinct
    characters, in the order given.
    """

    name: str
    operations: tuple[Operation, ...]
    irreps: tuple[Irrep, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "irreps", tuple(self.irreps))
        operations = set(self.operations)
        if not self.operations or self.operations[0] is not Operation.IDENTITY:
            raise SymmetryError("the identity is the first operation of a group")
        if len(operations) != len(self.operations):
            raise SymmetryError("operations are listed once each")
        if any(a.compose(b) not in operations for a in operations for b in operations):
            raise SymmetryError("the operations are not closed under composition")
        rows = {self._row(irrep.parity) for irrep in self.irreps}
        if len(rows) != len(self.irreps) or len(self.irreps) != len(self.operations):
            raise SymmetryError(
                "an Abelian group has one irrep per operation, with distinct characters"
            )

    @classmethod
    def named(cls, name: str) -> PointGroup:
        """Return the group of Schoenflies symbol ``name`` (case-insensitive).

        The Abelian groups are ``C1``, ``Ci``, ``Cs``, ``C2``, ``C2h``,
        ``C2v``, ``D2`` and ``D2h``.
        """
        try:
            operations, irreps = _TABLES[name.strip().lower()]
        except KeyError:
            raise InputError(
                f"unknown or non-Abelian point group {name!r}; the Abelian groups "
                f"are {', '.join(ABELIAN_GROUP_NAMES)}"
            ) from None
        return cls(
            _CANONICAL_NAMES[name.strip().lower()],
            operations,
            tuple(Irrep(label, parity) for label, parity in irreps),
        )

    @property
    def order(self) -> int:
        """Number of operations."""
        return len(self.operations)

    @property
    def n_irreps(self) -> int:
        """Number of irreps, equal to the order for an Abelian group."""
        return len(self.irreps)

    @property
    def labels(self) -> tuple[str, ...]:
        """Mulliken label of every irrep, in order."""
        return tuple(irrep.label for irrep in self.irreps)

    def character(self, irrep: int, operation: Operation) -> int:
        """Character ``±1`` of irrep ``irrep`` under ``operation``."""
        return operation.character(self.irreps[irrep].parity)

    def characters(self) -> tuple[tuple[int, ...], ...]:
        """Return the character table ``[irrep, operation]``."""
        return tuple(self._row(irrep.parity) for irrep in self.irreps)

    def irrep_of(self, parity: Parity) -> int:
        """Index of the irrep spanned by a function of ``parity``."""
        row = self._row(parity)
        for index, irrep in enumerate(self.irreps):
            if self._row(irrep.parity) == row:
                return index
        raise SymmetryError(f"no irrep of {self.name} has the characters {row}")

    def product(self, first: int, second: int) -> int:
        """Index of the direct product of two irreps."""
        a, b = self.irreps[first].parity, self.irreps[second].parity
        return self.irrep_of((a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2]))

    def _row(self, parity: Parity) -> tuple[int, ...]:
        return tuple(operation.character(parity) for operation in self.operations)


_E, _C2Z, _C2Y, _C2X = (
    Operation.IDENTITY,
    Operation.C2_Z,
    Operation.C2_Y,
    Operation.C2_X,
)
_I, _SXY, _SXZ, _SYZ = (
    Operation.INVERSION,
    Operation.SIGMA_XY,
    Operation.SIGMA_XZ,
    Operation.SIGMA_YZ,
)
_TABLES: dict[str, tuple[tuple[Operation, ...], tuple[tuple[str, Parity], ...]]] = {
    "c1": ((_E,), (("A", (0, 0, 0)),)),
    "ci": ((_E, _I), (("Ag", (0, 0, 0)), ("Au", (0, 0, 1)))),
    "cs": ((_E, _SXY), (("A'", (0, 0, 0)), ('A"', (0, 0, 1)))),
    "c2": ((_E, _C2Z), (("A", (0, 0, 0)), ("B", (1, 0, 0)))),
    "c2h": (
        (_E, _C2Z, _I, _SXY),
        (("Ag", (0, 0, 0)), ("Bg", (1, 0, 1)), ("Au", (0, 0, 1)), ("Bu", (1, 0, 0))),
    ),
    "c2v": (
        (_E, _C2Z, _SXZ, _SYZ),
        (("A1", (0, 0, 0)), ("A2", (1, 1, 0)), ("B1", (1, 0, 0)), ("B2", (0, 1, 0))),
    ),
    "d2": (
        (_E, _C2Z, _C2Y, _C2X),
        (("A", (0, 0, 0)), ("B1", (0, 0, 1)), ("B2", (0, 1, 0)), ("B3", (1, 0, 0))),
    ),
    "d2h": (
        (_E, _C2Z, _C2Y, _C2X, _I, _SXY, _SXZ, _SYZ),
        (
            ("Ag", (0, 0, 0)),
            ("B1g", (1, 1, 0)),
            ("B2g", (1, 0, 1)),
            ("B3g", (0, 1, 1)),
            ("Au", (1, 1, 1)),
            ("B1u", (0, 0, 1)),
            ("B2u", (0, 1, 0)),
            ("B3u", (1, 0, 0)),
        ),
    ),
}
_CANONICAL_NAMES = {
    "c1": "C1",
    "ci": "Ci",
    "cs": "Cs",
    "c2": "C2",
    "c2h": "C2h",
    "c2v": "C2v",
    "d2": "D2",
    "d2h": "D2h",
}
ABELIAN_GROUP_NAMES: tuple[str, ...] = tuple(_CANONICAL_NAMES.values())
"""Schoenflies symbols of D2h and its subgroups, see :meth:`PointGroup.named`."""


def solid_harmonic_parity(angular_momentum: int, m: int) -> Parity:
    """Parity of the real solid harmonic ``S_lm`` under the three reflections.

    ``m > 0`` is the cosine-type harmonic, ``m < 0`` the sine-type one, in
    the convention of ADR-0006; the parity does not depend on the phase.
    A p shell stored as ``x, y, z`` (ADR-0009) carries ``m = 1, −1, 0``
    in that order, so the rule applies to it unchanged.
    """
    if angular_momentum < 0 or abs(m) > angular_momentum:
        raise InputError(f"no harmonic with l = {angular_momentum} and m = {m}")
    order = abs(m)
    along_z = (angular_momentum - order) % 2
    if m > 0:
        return (order % 2, 0, along_z)
    if m < 0:
        return ((order + 1) % 2, 1, along_z)
    return (0, 0, along_z)

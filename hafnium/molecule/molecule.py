"""Immutable molecular structures."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from hafnium.backend import Array, Namespace, default_namespace
from hafnium.errors import InputError
from hafnium.molecule.elements import SYMBOLS, atomic_number
from hafnium.molecule.units import ANGSTROM_IN_BOHR

Position = tuple[float, float, float]

_BOHR_PER_UNIT = {"bohr": 1.0, "angstrom": ANGSTROM_IN_BOHR}


def bohr_per_unit(unit: str) -> float:
    """Length of one ``unit`` in Bohr; ``unit`` is ``'angstrom'`` or ``'bohr'``."""
    try:
        return _BOHR_PER_UNIT[unit.strip().lower()]
    except KeyError:
        raise InputError(
            f"unknown length unit {unit!r}; use 'angstrom' or 'bohr'"
        ) from None


def _to_bohr(symbol: str, position: Iterable[float], unit: str) -> Position:
    values = tuple(float(value) for value in position)
    if len(values) != 3:
        raise InputError(f"atom {symbol!r} needs three coordinates, got {values}")
    scale = bohr_per_unit(unit)
    return (values[0] * scale, values[1] * scale, values[2] * scale)


@dataclass(frozen=True, slots=True)
class Atom:
    """An atom at a position in Bohr."""

    symbol: str
    position: Position

    def __post_init__(self) -> None:
        number = atomic_number(self.symbol)
        if number == 0:
            raise InputError("ghost atoms ('X') are not supported yet")
        object.__setattr__(self, "symbol", SYMBOLS[number])
        object.__setattr__(self, "position", tuple(float(x) for x in self.position))
        if len(self.position) != 3:
            raise InputError("an atom position has three coordinates")

    @property
    def atomic_number(self) -> int:
        """Atomic number of the element."""
        return atomic_number(self.symbol)

    @property
    def nuclear_charge(self) -> float:
        """Charge of the nucleus seen by the electrons (the atomic number for now)."""
        return float(self.atomic_number)


@dataclass(frozen=True, slots=True)
class Molecule:
    """Atoms, total charge and spin of a molecule.

    Parameters
    ----------
    atoms
        Atoms with positions in Bohr.
    charge
        Total charge in units of the elementary charge.
    spin
        ``n_alpha - n_beta``, never the multiplicity (CONVENTIONS.md §3).
    """

    atoms: tuple[Atom, ...]
    charge: int = 0
    spin: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        if not self.atoms:
            raise InputError("a molecule needs at least one atom")
        if self.n_electrons < 0:
            raise InputError(f"charge {self.charge} leaves a negative electron count")
        if self.spin < 0:
            raise InputError(
                "spin is n_alpha - n_beta with n_alpha >= n_beta; use spin >= 0"
            )
        if self.spin > self.n_electrons or (self.n_electrons - self.spin) % 2:
            raise InputError(
                f"spin {self.spin} is inconsistent with {self.n_electrons} electrons; "
                "spin is n_alpha - n_beta, the number of unpaired electrons, and the "
                f"smallest consistent value is {self.n_electrons % 2}"
            )

    @classmethod
    def from_atoms(
        cls,
        atoms: Iterable[tuple[str, Iterable[float]]],
        *,
        charge: int = 0,
        spin: int = 0,
        unit: str = "angstrom",
    ) -> Molecule:
        """Build from ``(symbol, (x, y, z))`` pairs, in Ångström by default."""
        return cls(
            tuple(
                Atom(symbol, _to_bohr(symbol, position, unit))
                for symbol, position in atoms
            ),
            charge=charge,
            spin=spin,
        )

    @classmethod
    def from_xyz(
        cls, text: str, *, charge: int = 0, spin: int = 0, unit: str = "angstrom"
    ) -> Molecule:
        """Parse XYZ-format text, with or without the count and comment lines.

        A file that starts with a line holding a single integer is taken to
        have the standard header: that count, then one comment line of any
        content, then the atoms.
        """
        raw = text.splitlines()
        first = raw[0].split() if raw else []
        if len(first) == 1 and first[0].isdigit():
            count = int(first[0])
            lines = [line for line in raw[2:] if line.strip()]
            if len(lines) != count:
                raise InputError(
                    f"XYZ header announces {count} atoms, found {len(lines)}"
                )
        else:
            lines = [line for line in raw if line.strip()]
        atoms: list[tuple[str, tuple[float, float, float]]] = []
        for line in lines:
            fields = line.split()
            if len(fields) != 4:
                raise InputError(f"cannot parse atom line {line!r}")
            atoms.append(
                (fields[0], (float(fields[1]), float(fields[2]), float(fields[3])))
            )
        return cls.from_atoms(atoms, charge=charge, spin=spin, unit=unit)

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return len(self.atoms)

    @property
    def symbols(self) -> tuple[str, ...]:
        """Element symbol of each atom."""
        return tuple(atom.symbol for atom in self.atoms)

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        """Atomic number of each atom."""
        return tuple(atom.atomic_number for atom in self.atoms)

    @property
    def nuclear_charges(self) -> tuple[float, ...]:
        """Nuclear charge of each atom."""
        return tuple(atom.nuclear_charge for atom in self.atoms)

    @property
    def positions(self) -> tuple[Position, ...]:
        """Position of each atom in Bohr."""
        return tuple(atom.position for atom in self.atoms)

    @property
    def n_electrons(self) -> int:
        """Total number of electrons."""
        return sum(atom.atomic_number for atom in self.atoms) - self.charge

    @property
    def n_alpha(self) -> int:
        """Number of alpha electrons."""
        return (self.n_electrons + self.spin) // 2

    @property
    def n_beta(self) -> int:
        """Number of beta electrons."""
        return (self.n_electrons - self.spin) // 2

    def positions_array(self, xp: Namespace | None = None) -> Array:
        """Positions as an array ``[atom, 3]`` in Bohr."""
        xp = default_namespace() if xp is None else xp
        return xp.asarray(self.positions, dtype=xp.float64)

    def nuclear_repulsion_energy(self) -> float:
        """Coulomb repulsion of the nuclei in Hartree."""
        energy = 0.0
        for i, first in enumerate(self.atoms):
            for second in self.atoms[i + 1 :]:
                distance = math.dist(first.position, second.position)
                if distance == 0.0:
                    raise InputError(
                        f"atoms {first.symbol} and {second.symbol} coincide"
                    )
                energy += first.nuclear_charge * second.nuclear_charge / distance
        return energy

    def to_xyz(self, *, comment: str = "", unit: str = "angstrom") -> str:
        """Format as XYZ text, in Ångström by default."""
        scale = 1.0 / bohr_per_unit(unit)
        if "\n" in comment:
            raise InputError("the XYZ comment is a single line")
        lines = [str(self.n_atoms), comment]
        for atom in self.atoms:
            x, y, z = (value * scale for value in atom.position)
            lines.append(f"{atom.symbol:<2} {x:20.12f} {y:20.12f} {z:20.12f}")
        return "\n".join(lines) + "\n"

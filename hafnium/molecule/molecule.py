"""Immutable molecular structures."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from hafnium.errors import InputError
from hafnium.molecule.system import AtomicSystem, atoms_in_bohr, parse_xyz


@dataclass(frozen=True, slots=True)
class Molecule(AtomicSystem):
    """Atoms, total charge and spin of a molecule: a system without a lattice.

    See :class:`~hafnium.molecule.AtomicSystem` for the fields.
    """

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
        return cls(atoms_in_bohr(atoms, unit), charge=charge, spin=spin)

    @classmethod
    def from_xyz(
        cls, text: str, *, charge: int = 0, spin: int = 0, unit: str = "angstrom"
    ) -> Molecule:
        """Parse XYZ-format text, with or without the count and comment lines.

        A file that starts with a line holding a single integer is taken to
        have the standard header: that count, then one comment line of any
        content, then the atoms.
        """
        _, atoms = parse_xyz(text)
        return cls.from_atoms(atoms, charge=charge, spin=spin, unit=unit)

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

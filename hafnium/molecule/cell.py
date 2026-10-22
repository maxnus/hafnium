"""Periodic systems: atoms repeated on a lattice (ADR-0007, ADR-0013)."""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from hafnium.backend import host_array
from hafnium.errors import InputError
from hafnium.molecule.ewald import DEFAULT_EWALD_PRECISION, ewald_energy
from hafnium.molecule.lattice import Lattice
from hafnium.molecule.molecule import Molecule
from hafnium.molecule.system import (
    Atom,
    AtomicSystem,
    atoms_in_bohr,
    bohr_per_unit,
    parse_xyz,
)

_LATTICE_FIELD = re.compile(r'Lattice="([^"]*)"')


@dataclass(frozen=True, slots=True)
class Cell(AtomicSystem):
    """A unit cell: atoms with charge and spin per cell, and a lattice.

    Build it with :meth:`from_atoms`, :meth:`from_molecule`, or
    :meth:`from_xyz` from extended-XYZ text whose comment line carries the
    lattice; the constructor takes positions and lattice in Bohr. Charge
    and spin are per cell; a charged cell is neutralized by a uniform
    background, and the nuclear repulsion is an Ewald sum.
    """

    lattice: Lattice = field(kw_only=True)

    @classmethod
    def from_atoms(
        cls,
        atoms: Iterable[tuple[str, Iterable[float]]],
        lattice: Lattice | Iterable[Iterable[float]],
        *,
        charge: int = 0,
        spin: int = 0,
        unit: str = "angstrom",
    ) -> Cell:
        """Build from ``(symbol, (x, y, z))`` pairs and lattice vectors.

        Positions and vectors are in ``unit``, Ångström by default; a
        :class:`Lattice` is taken as it is, in Bohr.
        """
        if not isinstance(lattice, Lattice):
            lattice = Lattice.from_vectors(lattice, unit=unit)
        return cls(
            atoms_in_bohr(atoms, unit), charge=charge, spin=spin, lattice=lattice
        )

    @classmethod
    def from_molecule(cls, molecule: Molecule, lattice: Lattice) -> Cell:
        """Repeat the atoms of ``molecule`` on ``lattice``, keeping charge and spin."""
        return cls(
            molecule.atoms, charge=molecule.charge, spin=molecule.spin, lattice=lattice
        )

    @classmethod
    def from_xyz(
        cls, text: str, *, charge: int = 0, spin: int = 0, unit: str = "angstrom"
    ) -> Cell:
        """Parse extended-XYZ text whose comment line holds ``Lattice="..."``.

        The field lists the periodic vectors row by row, three, six or nine
        numbers in ``unit``, as :meth:`to_xyz` writes them.
        """
        comment, atoms = parse_xyz(text)
        match = _LATTICE_FIELD.search(comment)
        if match is None:
            raise InputError(
                'a cell in XYZ format carries Lattice="..." on its comment line'
            )
        values = [float(value) for value in match.group(1).split()]
        if len(values) not in (3, 6, 9):
            raise InputError(
                "the Lattice field holds three, six or nine numbers, one to three "
                f"vectors, got {len(values)}"
            )
        vectors = [values[start : start + 3] for start in range(0, len(values), 3)]
        return cls.from_atoms(atoms, vectors, charge=charge, spin=spin, unit=unit)

    @property
    def dimension(self) -> int:
        """Number of periodic directions."""
        return self.lattice.dimension

    def supercell(self, multiples: tuple[int, ...]) -> Cell:
        """Repeat the cell ``multiples[i]`` times along lattice vector ``i``.

        Atoms are ordered image by image, charge and spin are multiplied by
        the number of images.
        """
        lattice = self.lattice.supercell(multiples)
        n_images = math.prod(multiples)
        atoms: list[Atom] = []
        for image in itertools.product(*(range(m) for m in multiples)):
            shift = [
                sum(
                    n * vector[k]
                    for n, vector in zip(image, self.lattice.vectors, strict=True)
                )
                for k in range(3)
            ]
            atoms.extend(
                Atom(
                    atom.symbol,
                    (
                        atom.position[0] + shift[0],
                        atom.position[1] + shift[1],
                        atom.position[2] + shift[2],
                    ),
                )
                for atom in self.atoms
            )
        return Cell(
            tuple(atoms),
            charge=self.charge * n_images,
            spin=self.spin * n_images,
            lattice=lattice,
        )

    def nuclear_repulsion_energy(
        self,
        *,
        omega: float | None = None,
        precision: float = DEFAULT_EWALD_PRECISION,
    ) -> float:
        """Ewald energy of the nuclei per cell in Hartree, see :func:`ewald_energy`.

        The nuclei alone are not neutral; their net charge is compensated by
        a uniform background, as the electrons' is.
        """
        return ewald_energy(
            self.lattice,
            host_array(self.nuclear_charges),
            self.positions_array(),
            omega=omega,
            precision=precision,
        )

    def to_xyz(self, *, comment: str = "", unit: str = "angstrom") -> str:
        """Format as extended XYZ with ``Lattice="..."`` on the comment line."""
        scale = 1.0 / bohr_per_unit(unit)
        numbers = " ".join(
            f"{component * scale:.12f}"
            for vector in self.lattice.vectors
            for component in vector
        )
        header = f'Lattice="{numbers}" {comment}'.rstrip()
        return AtomicSystem.to_xyz(self, comment=header, unit=unit)

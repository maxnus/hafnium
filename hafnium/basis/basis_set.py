"""A basis set assigned to the atoms of a molecule."""

from __future__ import annotations

from dataclasses import dataclass

from hafnium.basis.library import BasisLibrary
from hafnium.basis.shell import Shell
from hafnium.errors import BasisSetError
from hafnium.molecule import Molecule


@dataclass(frozen=True, slots=True)
class FunctionLabel:
    """Identity of one basis function: centre, shell, contraction and ``m``."""

    atom: int
    symbol: str
    shell: int
    angular_momentum: int
    contraction: int
    m: int


@dataclass(frozen=True, slots=True)
class BasisSet:
    """Shells of a basis set in atom order (ADR-0006).

    Parameters
    ----------
    name
        Basis set name as given by the user.
    atom_symbols
        Element symbol of each centre, in atom order.
    shells
        All shells, ordered by atom and then as listed in the basis set.
    shell_atoms
        Index of the centre of each shell.
    """

    name: str
    atom_symbols: tuple[str, ...]
    shells: tuple[Shell, ...]
    shell_atoms: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_symbols", tuple(self.atom_symbols))
        object.__setattr__(self, "shells", tuple(self.shells))
        object.__setattr__(self, "shell_atoms", tuple(int(i) for i in self.shell_atoms))
        if not self.shells:
            raise BasisSetError("a basis set needs at least one shell")
        if len(self.shells) != len(self.shell_atoms):
            raise BasisSetError("each shell needs a centre")
        if any(not 0 <= atom < len(self.atom_symbols) for atom in self.shell_atoms):
            raise BasisSetError("shell centres refer to atoms that do not exist")
        if any(
            b < a for a, b in zip(self.shell_atoms, self.shell_atoms[1:], strict=False)
        ):
            raise BasisSetError("shells are ordered by atom")

    @classmethod
    def for_molecule(
        cls, molecule: Molecule, name: str, *, library: BasisLibrary | None = None
    ) -> BasisSet:
        """Assign basis set ``name`` to every atom of ``molecule``."""
        library = BasisLibrary() if library is None else library
        by_element = library.shells(name, molecule.atomic_numbers)
        shells: list[Shell] = []
        centres: list[int] = []
        for index, atom in enumerate(molecule.atoms):
            element_shells = by_element[atom.atomic_number]
            if not element_shells:
                raise BasisSetError(
                    f"basis set {name!r} has no functions for {atom.symbol}"
                )
            shells.extend(element_shells)
            centres.extend([index] * len(element_shells))
        return cls(name, molecule.symbols, tuple(shells), tuple(centres))

    @property
    def n_atoms(self) -> int:
        """Number of centres."""
        return len(self.atom_symbols)

    @property
    def n_shells(self) -> int:
        """Number of shells."""
        return len(self.shells)

    @property
    def n_functions(self) -> int:
        """Number of basis functions."""
        return sum(shell.n_functions for shell in self.shells)

    @property
    def max_angular_momentum(self) -> int:
        """Highest angular momentum among the shells."""
        return max(shell.angular_momentum for shell in self.shells)

    def shell_offsets(self) -> tuple[int, ...]:
        """Index of the first basis function of each shell."""
        offsets: list[int] = []
        total = 0
        for shell in self.shells:
            offsets.append(total)
            total += shell.n_functions
        return tuple(offsets)

    def function_labels(self) -> tuple[FunctionLabel, ...]:
        """One label per basis function, in basis function order."""
        labels: list[FunctionLabel] = []
        for shell_index, (shell, atom) in enumerate(
            zip(self.shells, self.shell_atoms, strict=True)
        ):
            l = shell.angular_momentum
            labels.extend(
                FunctionLabel(atom, self.atom_symbols[atom], shell_index, l, c, m)
                for c in range(shell.n_contractions)
                for m in m_values(l)
            )
        return tuple(labels)


def m_values(angular_momentum: int) -> tuple[int, ...]:
    """The ``m`` of each function of a shell, in storage order (ADR-0009).

    ``m = -l, ..., +l`` except for p shells, which are ordered ``x, y, z``,
    that is ``m = +1, -1, 0``, as in libcint, PySCF and Molden.
    """
    if angular_momentum == 1:
        return (1, -1, 0)
    return tuple(range(-angular_momentum, angular_momentum + 1))

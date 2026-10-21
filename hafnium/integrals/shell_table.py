"""Shells in the structure-of-arrays form the compiled kernels consume."""

from __future__ import annotations

from dataclasses import dataclass

from hafnium.backend import Array, host_array, to_host
from hafnium.basis import BasisSet
from hafnium.errors import InputError


@dataclass(frozen=True, slots=True)
class ShellTable:
    """Contracted shells with their centres, one entry per contraction.

    General contractions of a :class:`~hafnium.basis.Shell` are split into
    consecutive entries sharing the exponents, which keeps the function
    order of ADR-0006 (contraction-major, then ``m``). Coefficients are
    those of bare primitives from :meth:`Shell.primitive_coefficients`.

    Parameters
    ----------
    angular_momenta : Array[shell]
        ``l`` of each entry.
    centers : Array[shell, 3]
        Centre of each entry in Bohr.
    primitive_offsets : Array[shell + 1]
        Entry ``i`` owns primitives ``offsets[i]:offsets[i + 1]``.
    exponents, coefficients : Array[primitive]
        Primitive exponents and bare-primitive coefficients.
    """

    angular_momenta: Array
    centers: Array
    primitive_offsets: Array
    exponents: Array
    coefficients: Array

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "angular_momenta", host_array(self.angular_momenta, "int64")
        )
        object.__setattr__(self, "centers", host_array(self.centers))
        object.__setattr__(
            self, "primitive_offsets", host_array(self.primitive_offsets, "int64")
        )
        object.__setattr__(self, "exponents", host_array(self.exponents))
        object.__setattr__(self, "coefficients", host_array(self.coefficients))
        n_shells = self.angular_momenta.shape[0]
        if self.centers.shape != (n_shells, 3):
            raise InputError("a shell table needs one centre per shell")
        if self.primitive_offsets.shape != (n_shells + 1,):
            raise InputError("primitive offsets have one entry per shell plus one")
        if self.exponents.shape != self.coefficients.shape:
            raise InputError("exponents and coefficients have the same length")

    @classmethod
    def from_basis(cls, basis: BasisSet, positions: Array) -> ShellTable:
        """Lay out ``basis`` on centres ``positions`` (``[atom, 3]``, Bohr)."""
        positions = to_host(positions)
        if positions.shape != (basis.n_atoms, 3):
            raise InputError(
                f"expected positions for {basis.n_atoms} atoms, got shape "
                f"{positions.shape}"
            )
        angular_momenta: list[int] = []
        centers: list[Array] = []
        offsets = [0]
        exponents: list[float] = []
        coefficients: list[float] = []
        for shell, atom in zip(basis.shells, basis.shell_atoms, strict=True):
            for row in shell.primitive_coefficients():
                angular_momenta.append(shell.angular_momentum)
                centers.append(positions[atom])
                exponents.extend(shell.exponents)
                coefficients.extend(row)
                offsets.append(len(exponents))
        return cls(
            host_array(angular_momenta, "int64"),
            host_array(centers),
            host_array(offsets, "int64"),
            host_array(exponents),
            host_array(coefficients),
        )

    @property
    def n_shells(self) -> int:
        """Number of entries (contractions)."""
        return int(self.angular_momenta.shape[0])

    @property
    def n_functions(self) -> int:
        """Number of spherical basis functions."""
        return int(sum(2 * int(l) + 1 for l in self.angular_momenta))

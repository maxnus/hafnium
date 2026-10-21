"""Shells in the structure-of-arrays form the compiled kernels consume."""

from __future__ import annotations

from dataclasses import dataclass

from hafnium.backend import Array, host_array, to_host
from hafnium.basis import BasisSet
from hafnium.errors import InputError


@dataclass(frozen=True, slots=True)
class ShellTable:
    """Contracted shells with their centres, one entry per shell.

    A general contraction of a :class:`~hafnium.basis.Shell` stays one
    entry with several coefficient rows over the same exponents, and the
    kernels evaluate its primitives once for all rows (ADR-0011); the
    function order of ADR-0006 (contraction-major, then ``m``) is kept.
    Coefficients are those of bare primitives from
    :meth:`Shell.primitive_coefficients`.

    Parameters
    ----------
    angular_momenta : Array[shell]
        ``l`` of each entry.
    centers : Array[shell, 3]
        Centre of each entry in Bohr.
    primitive_offsets : Array[shell + 1]
        Entry ``i`` owns primitives ``offsets[i]:offsets[i + 1]``.
    exponents : Array[primitive]
        Primitive exponents.
    coefficients : Array
        Bare-primitive coefficients, shell by shell, one row per
        contraction and one entry per primitive of the shell.
    n_contractions : Array[shell]
        Contractions of each entry; one each when omitted.
    """

    angular_momenta: Array
    centers: Array
    primitive_offsets: Array
    exponents: Array
    coefficients: Array
    n_contractions: Array | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "angular_momenta", _integers(self.angular_momenta, "angular momenta")
        )
        object.__setattr__(self, "centers", host_array(self.centers))
        object.__setattr__(
            self,
            "primitive_offsets",
            _integers(self.primitive_offsets, "primitive offsets"),
        )
        object.__setattr__(self, "exponents", host_array(self.exponents))
        object.__setattr__(self, "coefficients", host_array(self.coefficients))
        if self.angular_momenta.ndim != 1:
            raise InputError("angular momenta form a one-dimensional array")
        n_shells = self.angular_momenta.shape[0]
        if self.centers.shape != (n_shells, 3):
            raise InputError("a shell table needs one centre per shell")
        if self.primitive_offsets.shape != (n_shells + 1,):
            raise InputError("primitive offsets have one entry per shell plus one")
        counts = (
            host_array([1] * n_shells, "int64")
            if self.n_contractions is None
            else _integers(self.n_contractions, "contraction counts")
        )
        object.__setattr__(self, "n_contractions", counts)
        if counts.shape != (n_shells,) or bool((counts < 1).any()):
            raise InputError("every shell has at least one contraction")
        primitives = self.primitive_offsets[1:] - self.primitive_offsets[:-1]
        if self.exponents.shape != (int(self.primitive_offsets[-1]),):
            raise InputError("exponents cover exactly the primitives of the shells")
        if self.coefficients.shape != (int((counts * primitives).sum()),):
            raise InputError(
                "coefficients hold one row per contraction, one entry per primitive"
            )

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
        n_contractions: list[int] = []
        for shell, atom in zip(basis.shells, basis.shell_atoms, strict=True):
            rows = shell.primitive_coefficients()
            angular_momenta.append(shell.angular_momentum)
            centers.append(positions[atom])
            exponents.extend(shell.exponents)
            offsets.append(len(exponents))
            for row in rows:
                coefficients.extend(row)
            n_contractions.append(len(rows))
        return cls(
            host_array(angular_momenta, "int64"),
            host_array(centers),
            host_array(offsets, "int64"),
            host_array(exponents),
            host_array(coefficients),
            host_array(n_contractions, "int64"),
        )

    @property
    def n_shells(self) -> int:
        """Number of entries (shells)."""
        return int(self.angular_momenta.shape[0])

    @property
    def contraction_counts(self) -> Array:
        """Contractions of each entry, ``int64 [shell]``; never ``None`` once built."""
        assert self.n_contractions is not None
        return self.n_contractions

    @property
    def n_functions(self) -> int:
        """Number of spherical basis functions, ``2l + 1`` per contraction."""
        counts = self.contraction_counts
        return int(
            sum(
                int(k) * (2 * int(l) + 1)
                for k, l in zip(counts, self.angular_momenta, strict=True)
            )
        )


def _integers(values: Array, name: str) -> Array:
    """Host ``int64`` array of ``values``, rejecting anything non-integral."""
    integers = host_array(values, "int64")
    if integers.shape != host_array(values).shape or not bool(
        (integers == host_array(values)).all()
    ):
        raise InputError(f"{name} are integers")
    return integers

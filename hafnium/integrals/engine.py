"""The interface between method code and integral evaluation."""

from __future__ import annotations

from typing import Protocol

from hafnium.backend import Array
from hafnium.integrals.shell_table import ShellTable


class IntegralEngine(Protocol):
    """Evaluates integrals over shell tables.

    Every method takes a ``bra`` and a ``ket`` table and ``translations``
    of shape ``[image, 3]`` in Bohr; image ``T`` of the result holds the
    integrals between the bra functions and the ket functions translated by
    ``translations[T]``. A molecule uses a single zero translation; a cell
    passes its lattice images and forms Bloch sums from the result.
    """

    def overlap(self, bra: ShellTable, ket: ShellTable, translations: Array) -> Array:
        """Overlap integrals ``[image, ao, ao]``."""
        ...

    def kinetic(self, bra: ShellTable, ket: ShellTable, translations: Array) -> Array:
        """Kinetic energy integrals ``[image, ao, ao]``."""
        ...

    def nuclear_attraction(
        self,
        bra: ShellTable,
        ket: ShellTable,
        translations: Array,
        charges: Array,
        positions: Array,
        *,
        omega: float | None = None,
    ) -> Array:
        """Attraction ``[image, ao, ao]`` to point ``charges`` at ``positions``.

        The potential of each charge is ``-Z / r``, or ``-Z erf(ω r) / r``
        when ``omega`` is given (the long-range part of a range-separated
        kernel).
        """
        ...

    def coulomb_two_center(
        self,
        bra: ShellTable,
        ket: ShellTable,
        translations: Array,
        *,
        omega: float | None = None,
    ) -> Array:
        """Two-centre Coulomb integrals ``(P|Q)`` as ``[image, aux, aux]``.

        With ``omega`` the kernel is ``erf(ω r) / r`` instead of ``1 / r``.
        """
        ...

    def coulomb_three_center(
        self,
        auxiliary: ShellTable,
        bra: ShellTable,
        ket: ShellTable,
        translations: Array,
        *,
        omega: float | None = None,
    ) -> Array:
        """Three-centre Coulomb integrals ``(P|uv)`` as ``[image, aux, ao, ao]``.

        ``translations`` has shape ``[image, 2, 3]``: the first vector
        translates the auxiliary function, the second the ket function.
        """
        ...

    def coulomb_four_center(
        self,
        bra: tuple[ShellTable, ShellTable],
        ket: tuple[ShellTable, ShellTable],
        translations: Array,
        *,
        omega: float | None = None,
        threshold: float = 0.0,
    ) -> Array:
        """Four-centre Coulomb integrals ``(uv|wx)`` as ``[image, ao, ao, ao, ao]``.

        Chemists' notation: the bra tables give the first two indices, the
        ket tables the last two. ``translations`` has shape ``[image, 3, 3]``:
        the vectors applied to the second bra function and to the two ket
        functions, the first bra function staying at its centre. Shell
        quartets whose Schwarz bound, the product of the two pair bounds of
        :meth:`schwarz_bounds`, is below ``threshold`` are not evaluated and
        left at zero.
        """
        ...

    def schwarz_bounds(
        self,
        bra: ShellTable,
        ket: ShellTable,
        translations: Array,
        *,
        omega: float | None = None,
    ) -> Array:
        """Schwarz bounds of the shell pairs as ``[image, shell, shell]``.

        The bound of a pair is the largest ``sqrt((uv|uv))`` over its
        functions, so that ``|(uv|wx)| ≤ Q_uv Q_wx`` for every integral with
        the same kernel.
        """
        ...

    def fourier_transform(self, table: ShellTable, g_vectors: Array) -> Array:
        """Transform every function: ``∫ χ_P(r) exp(−i G·r) dr``, complex ``[G, ao]``.

        ``g_vectors`` has shape ``[G, 3]`` in inverse Bohr. A function
        translated by ``T`` has its transform multiplied by ``exp(−i G·T)``,
        so no translations are taken.
        """
        ...

    def pair_fourier_transform(
        self, bra: ShellTable, ket: ShellTable, translations: Array, g_vectors: Array
    ) -> Array:
        """Transform every pair: ``∫ φ_u(r) φ_v(r − T) exp(−i G·r) dr``.

        Complex ``[image, G, ao, ao]`` for the translations ``T`` of the ket
        and the ``g_vectors`` ``[G, 3]`` in inverse Bohr. At ``G = 0`` the
        transform is the overlap matrix of the same translation.
        """
        ...

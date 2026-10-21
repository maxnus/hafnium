"""The native Rust integral engine (ADR-0008)."""

from __future__ import annotations

from hafnium import _kernels
from hafnium.backend import Array, to_host
from hafnium.integrals.shell_table import ShellTable


def _kernel_table(table: ShellTable) -> _kernels.ShellTable:
    return _kernels.ShellTable(
        table.angular_momenta,
        table.centers,
        table.primitive_offsets,
        table.exponents,
        table.coefficients,
    )


class RustEngine:
    """Integrals from the compiled McMurchie–Davidson kernels.

    Results are host arrays; conversion to the active backend happens in
    the caller.
    """

    def overlap(self, bra: ShellTable, ket: ShellTable, translations: Array) -> Array:
        """Overlap integrals ``[image, ao, ao]``."""
        return _kernels.overlap(
            _kernel_table(bra), _kernel_table(ket), to_host(translations)
        )

    def kinetic(self, bra: ShellTable, ket: ShellTable, translations: Array) -> Array:
        """Kinetic energy integrals ``[image, ao, ao]``."""
        return _kernels.kinetic(
            _kernel_table(bra), _kernel_table(ket), to_host(translations)
        )

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
        """Attraction ``[image, ao, ao]`` to point ``charges`` at ``positions``."""
        return _kernels.nuclear_attraction(
            _kernel_table(bra),
            _kernel_table(ket),
            to_host(translations),
            to_host(charges),
            to_host(positions),
            omega,
        )

    def coulomb_two_center(
        self,
        bra: ShellTable,
        ket: ShellTable,
        translations: Array,
        *,
        omega: float | None = None,
    ) -> Array:
        """Two-centre Coulomb integrals ``[image, aux, aux]``."""
        return _kernels.coulomb_two_center(
            _kernel_table(bra), _kernel_table(ket), to_host(translations), omega
        )

    def coulomb_three_center(
        self,
        auxiliary: ShellTable,
        bra: ShellTable,
        ket: ShellTable,
        translations: Array,
        *,
        omega: float | None = None,
    ) -> Array:
        """Three-centre Coulomb integrals ``[image, aux, ao, ao]``."""
        return _kernels.coulomb_three_center(
            _kernel_table(auxiliary),
            _kernel_table(bra),
            _kernel_table(ket),
            to_host(translations),
            omega,
        )

    def coulomb_four_center(
        self,
        bra: tuple[ShellTable, ShellTable],
        ket: tuple[ShellTable, ShellTable],
        translations: Array,
        *,
        omega: float | None = None,
        threshold: float = 0.0,
    ) -> Array:
        """Four-centre Coulomb integrals ``[image, ao, ao, ao, ao]``."""
        return _kernels.coulomb_four_center(
            (_kernel_table(bra[0]), _kernel_table(bra[1])),
            (_kernel_table(ket[0]), _kernel_table(ket[1])),
            to_host(translations),
            omega,
            threshold,
        )

    def schwarz_bounds(
        self,
        bra: ShellTable,
        ket: ShellTable,
        translations: Array,
        *,
        omega: float | None = None,
    ) -> Array:
        """Schwarz bounds of the shell pairs ``[image, shell, shell]``."""
        return _kernels.schwarz_bounds(
            _kernel_table(bra), _kernel_table(ket), to_host(translations), omega
        )

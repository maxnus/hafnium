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

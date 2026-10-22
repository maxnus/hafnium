"""The native Rust integral engine (ADR-0008)."""

from __future__ import annotations

from hafnium import _kernels
from hafnium.backend import Array, to_host
from hafnium.integrals.engine import (
    COULOMB,
    BlochImages,
    CoulombKernel,
    PointCharges,
)
from hafnium.integrals.shell_table import ShellTable


def _kernel_table(table: ShellTable) -> _kernels.ShellTable:
    return _kernels.ShellTable(
        table.angular_momenta,
        table.centers,
        table.primitive_offsets,
        table.exponents,
        table.coefficients,
        table.contraction_counts,
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
        charges: PointCharges,
        *,
        kernel: CoulombKernel = COULOMB,
        threshold: float = 0.0,
    ) -> Array:
        """Attraction ``[image, ao, ao]`` to the point ``charges``."""
        return _kernels.nuclear_attraction(
            _kernel_table(bra),
            _kernel_table(ket),
            to_host(translations),
            to_host(charges.charges),
            to_host(charges.positions),
            kernel.omega,
            kernel.short_range,
            threshold,
        )

    def coulomb_two_center(
        self,
        bra: ShellTable,
        ket: ShellTable,
        translations: Array,
        *,
        kernel: CoulombKernel = COULOMB,
        threshold: float = 0.0,
    ) -> Array:
        """Two-centre Coulomb integrals ``[image, aux, aux]``."""
        return _kernels.coulomb_two_center(
            _kernel_table(bra),
            _kernel_table(ket),
            to_host(translations),
            kernel.omega,
            kernel.short_range,
            threshold,
        )

    def coulomb_three_center(
        self,
        auxiliary: ShellTable,
        bra: ShellTable,
        ket: ShellTable,
        translations: Array,
        *,
        kernel: CoulombKernel = COULOMB,
        threshold: float = 0.0,
    ) -> Array:
        """Three-centre Coulomb integrals ``[image, aux, ao, ao]``."""
        return _kernels.coulomb_three_center(
            _kernel_table(auxiliary),
            _kernel_table(bra),
            _kernel_table(ket),
            to_host(translations),
            kernel.omega,
            kernel.short_range,
            threshold,
        )

    def coulomb_three_center_bloch(
        self,
        auxiliary: ShellTable,
        bra: ShellTable,
        ket: ShellTable,
        images: BlochImages,
        *,
        kernel: CoulombKernel = COULOMB,
        threshold: float = 0.0,
    ) -> Array:
        """Bloch-summed three-centre integrals, complex ``[kpt, kpt, aux, ao, ao]``."""
        return _kernels.coulomb_three_center_bloch(
            _kernel_table(auxiliary),
            _kernel_table(bra),
            _kernel_table(ket),
            to_host(images.images),
            to_host(images.coordinates[0], "int64"),
            to_host(images.steps),
            to_host(images.coordinates[1], "int64"),
            to_host(images.phases[0], "complex128"),
            to_host(images.phases[1], "complex128"),
            kernel.omega,
            kernel.short_range,
            threshold,
        )

    def coulomb_four_center(
        self,
        bra: tuple[ShellTable, ShellTable],
        ket: tuple[ShellTable, ShellTable],
        translations: Array,
        *,
        kernel: CoulombKernel = COULOMB,
        threshold: float = 0.0,
    ) -> Array:
        """Four-centre Coulomb integrals ``[image, ao, ao, ao, ao]``."""
        return _kernels.coulomb_four_center(
            (_kernel_table(bra[0]), _kernel_table(bra[1])),
            (_kernel_table(ket[0]), _kernel_table(ket[1])),
            to_host(translations),
            kernel.omega,
            kernel.short_range,
            threshold,
        )

    def schwarz_bounds(
        self,
        bra: ShellTable,
        ket: ShellTable,
        translations: Array,
        *,
        kernel: CoulombKernel = COULOMB,
    ) -> Array:
        """Schwarz bounds of the shell pairs ``[image, shell, shell]``."""
        return _kernels.schwarz_bounds(
            _kernel_table(bra),
            _kernel_table(ket),
            to_host(translations),
            kernel.omega,
            kernel.short_range,
        )

    def fourier_transform(self, table: ShellTable, g_vectors: Array) -> Array:
        """Transform every function, complex ``[G, ao]``."""
        return _kernels.fourier_transform(_kernel_table(table), to_host(g_vectors))

    def pair_fourier_transform(
        self,
        bra: ShellTable,
        ket: ShellTable,
        translations: Array,
        g_vectors: Array,
        *,
        threshold: float = 0.0,
    ) -> Array:
        """Transform every pair, complex ``[image, G, ao, ao]``."""
        return _kernels.pair_fourier_transform(
            _kernel_table(bra),
            _kernel_table(ket),
            to_host(translations),
            to_host(g_vectors),
            threshold,
        )

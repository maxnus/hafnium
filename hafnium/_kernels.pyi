"""Type stubs for the compiled kernel module built from crates/hafnium-py."""

import numpy as np
from numpy.typing import NDArray

def kernel_version() -> str:
    """Return the version of the compiled kernel crate."""

def dot(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Return the dot product of two contiguous one-dimensional float64 arrays.

    Bootstrap smoke test of the array boundary between Python and Rust.
    Raises ``ValueError`` if the lengths differ and ``TypeError`` if an array
    is not contiguous.
    """

def boys_function(n_max: int, t: float) -> NDArray[np.float64]:
    """Return the Boys function ``F_n(t)`` for ``n = 0, ..., n_max``."""

def spherical_transformation(l: int) -> NDArray[np.float64]:
    """Return the Cartesian-to-solid-harmonic matrix ``[2l + 1, n_cartesian]``."""

class ShellTable:
    """Contracted shells in structure-of-arrays form, one entry per contraction."""

    def __init__(
        self,
        angular_momenta: NDArray[np.int64],
        centers: NDArray[np.float64],
        primitive_offsets: NDArray[np.int64],
        exponents: NDArray[np.float64],
        coefficients: NDArray[np.float64],
    ) -> None: ...
    @property
    def n_shells(self) -> int:
        """Number of shells."""
    @property
    def n_functions(self) -> int:
        """Number of spherical basis functions."""

def overlap(
    bra: ShellTable, ket: ShellTable, translations: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return overlap integrals ``[image, bra, ket]`` for each ket translation."""

def kinetic(
    bra: ShellTable, ket: ShellTable, translations: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return kinetic energy integrals ``[image, bra, ket]``."""

def nuclear_attraction(
    bra: ShellTable,
    ket: ShellTable,
    translations: NDArray[np.float64],
    charges: NDArray[np.float64],
    positions: NDArray[np.float64],
    omega: float | None = None,
) -> NDArray[np.float64]:
    """Return nuclear attraction integrals ``[image, bra, ket]`` of point charges.

    ``omega`` selects the long-range kernel ``erf(ωr)/r`` instead of ``1/r``.
    """

def coulomb_two_center(
    bra: ShellTable,
    ket: ShellTable,
    translations: NDArray[np.float64],
    omega: float | None = None,
) -> NDArray[np.float64]:
    """Return two-centre Coulomb integrals ``[image, bra, ket]``."""

def coulomb_three_center(
    auxiliary: ShellTable,
    bra: ShellTable,
    ket: ShellTable,
    translations: NDArray[np.float64],
    omega: float | None = None,
) -> NDArray[np.float64]:
    """Return three-centre Coulomb integrals ``[image, aux, bra, ket]``.

    ``translations`` has shape ``[image, 2, 3]``: auxiliary and ket vectors.
    """

def coulomb_four_center(
    bra: tuple[ShellTable, ShellTable],
    ket: tuple[ShellTable, ShellTable],
    translations: NDArray[np.float64],
    omega: float | None = None,
    threshold: float = 0.0,
) -> NDArray[np.float64]:
    """Return four-centre Coulomb integrals ``(uv|wx)`` as ``[image, ao, ao, ao, ao]``.

    ``translations`` has shape ``[image, 3, 3]``: the vectors applied to the
    second bra function and to the two ket functions. Shell quartets whose
    Schwarz bound is below ``threshold`` are left at zero.
    """

def schwarz_bounds(
    bra: ShellTable,
    ket: ShellTable,
    translations: NDArray[np.float64],
    omega: float | None = None,
) -> NDArray[np.float64]:
    """Return the Schwarz bounds of the shell pairs, ``[image, shell, shell]``."""

def configure_threads(n_threads: int) -> None:
    """Set the size of the kernel thread pool once per process.

    Raises ``ValueError`` if the pool was already configured.
    """

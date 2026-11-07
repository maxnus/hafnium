"""Type stubs for the compiled kernel module built from crates/hafnium-py."""

from typing import Any

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
    """Contracted shells in structure-of-arrays form, one entry per shell."""

    def __init__(
        self,
        angular_momenta: NDArray[np.int64],
        centers: NDArray[np.float64],
        primitive_offsets: NDArray[np.int64],
        exponents: NDArray[np.float64],
        coefficients: NDArray[np.float64],
        n_contractions: NDArray[np.int64],
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
    short_range: bool = False,
    threshold: float = 0.0,
) -> NDArray[np.float64]:
    """Return nuclear attraction integrals ``[image, bra, ket]`` of point charges.

    ``omega`` selects the long-range kernel ``erf(ωr)/r`` instead of ``1/r``,
    or the short-range kernel ``erfc(ωr)/r`` with ``short_range``; terms
    bounded below ``threshold`` are skipped.
    """

def overlap_derivative(
    bra: ShellTable, ket: ShellTable, translations: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the derivative of the overlap with respect to the bra centres.

    ``[image, 3, bra, ket]``.
    """

def kinetic_derivative(
    bra: ShellTable, ket: ShellTable, translations: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the derivative of the kinetic integrals with respect to the bra centres.

    ``[image, 3, bra, ket]``.
    """

def nuclear_attraction_derivative(
    bra: ShellTable,
    ket: ShellTable,
    translations: NDArray[np.float64],
    charges: NDArray[np.float64],
    positions: NDArray[np.float64],
    omega: float | None = None,
    short_range: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the derivatives of the nuclear attraction.

    With respect to the bra centres as ``[image, 3, bra, ket]`` and with
    respect to the charge positions as ``[image, charge, 3, bra, ket]``.
    """

def coulomb_two_center(
    bra: ShellTable,
    ket: ShellTable,
    translations: NDArray[np.float64],
    omega: float | None = None,
    short_range: bool = False,
    threshold: float = 0.0,
) -> NDArray[np.float64]:
    """Return two-centre Coulomb integrals ``[image, bra, ket]``.

    ``omega`` selects ``erf(ωr)/r``, or ``erfc(ωr)/r`` with ``short_range``;
    terms bounded below ``threshold`` are skipped.
    """

def coulomb_two_center_derivative(
    bra: ShellTable,
    ket: ShellTable,
    translations: NDArray[np.float64],
    omega: float | None = None,
    short_range: bool = False,
) -> NDArray[np.float64]:
    """Return the derivative of ``(P|Q)`` with respect to the bra centres.

    ``[image, 3, bra, ket]``.
    """

def coulomb_three_center(
    auxiliary: ShellTable,
    bra: ShellTable,
    ket: ShellTable,
    translations: NDArray[np.float64],
    omega: float | None = None,
    short_range: bool = False,
    threshold: float = 0.0,
) -> NDArray[np.float64]:
    """Return three-centre Coulomb integrals ``(P|uv)`` as ``[image, aux, bra, ket]``.

    ``translations`` has shape ``[image, 2, 3]`` (auxiliary, ket).
    """

def coulomb_three_center_derivative(
    auxiliary: ShellTable,
    bra: ShellTable,
    ket: ShellTable,
    translations: NDArray[np.float64],
    omega: float | None = None,
    short_range: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the derivatives of ``(P|uv)``.

    With respect to the auxiliary and to the bra centres, each as
    ``[image, 3, aux, bra, ket]``.
    """

def coulomb_three_center_bloch(
    auxiliary: ShellTable,
    bra: ShellTable,
    ket: ShellTable,
    images: NDArray[np.float64],
    image_coordinates: NDArray[np.int64],
    steps: NDArray[np.float64],
    step_coordinates: NDArray[np.int64],
    image_classes: NDArray[np.int64],
    step_classes: NDArray[np.int64],
    kpoints: NDArray[np.float64],
    omega: float | None = None,
    short_range: bool = False,
    threshold: float = 0.0,
    coulomb_only: bool = False,
) -> NDArray[np.complex128]:
    """Return the Bloch-summed three-centre integrals ``[kpt, kpt, aux, bra, ket]``.

    ``Σ_ij exp(−i k1·T_i) exp(i k2·T_j) (P | u(T_i) v(T_j))`` over the image
    pairs whose difference is a step, found by the lattice coordinates, the
    auxiliary function in the home cell; ``exp(i (k2 − k1)·T_i)`` is taken
    from the class of every image and ``exp(i k2·T_s)`` from the class of
    every step (ADR-0025, ADR-0028). With ``coulomb_only`` only the blocks
    of zero transfer are formed, ``[kpt, aux, bra, ket]``.
    """

def coulomb_four_center(
    bra: tuple[ShellTable, ShellTable],
    ket: tuple[ShellTable, ShellTable],
    translations: NDArray[np.float64],
    omega: float | None = None,
    short_range: bool = False,
    threshold: float = 0.0,
) -> NDArray[np.float64]:
    """Return four-centre Coulomb integrals ``(uv|wx)`` as ``[image, u, v, w, x]``.

    ``translations`` has shape ``[image, 3, 3]`` (second bra, first ket,
    second ket); shell quartets with a Schwarz bound below ``threshold`` are
    left at zero.
    """

def schwarz_bounds(
    bra: ShellTable,
    ket: ShellTable,
    translations: NDArray[np.float64],
    omega: float | None = None,
    short_range: bool = False,
) -> NDArray[np.float64]:
    """Return the Schwarz bounds of the shell pairs as ``[image, shell, shell]``."""

def fourier_transform(
    table: ShellTable, g_vectors: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Return ``∫ χ(r) exp(−i G·r) dr`` of every function as ``[G, ao]``."""

def pair_fourier_transform(
    bra: ShellTable,
    ket: ShellTable,
    translations: NDArray[np.float64],
    g_vectors: NDArray[np.float64],
    threshold: float = 0.0,
) -> NDArray[np.complex128]:
    """Return ``∫ φ_u(r) φ_v(r − T) exp(−i G·r) dr`` as ``[image, G, ao, ao]``.

    Pairs bounded below ``threshold`` are skipped.
    """

def ewald_energy(
    vectors: NDArray[np.float64],
    charges: NDArray[np.float64],
    positions: NDArray[np.float64],
    translations: NDArray[np.float64],
    g_vectors: NDArray[np.float64],
    omega: float,
) -> float:
    """Return the Ewald energy per cell of point charges on a lattice.

    ``vectors`` ``[d, 3]`` span the lattice, ``translations`` ``[T, 3]`` are
    the real-space images summed over and ``g_vectors`` ``[G, 3]`` the
    non-zero reciprocal vectors; ``omega`` is the range-separation parameter.
    """

def basis_values(
    table: ShellTable,
    points: NDArray[np.float64],
    radii: NDArray[np.float64],
    order: int = 0,
) -> NDArray[np.float64]:
    """Return the basis functions at ``points`` ``[g, 3]`` as ``[g, component, ao]``.

    The components are the value and, for ``order`` 1 and 2, the derivatives
    ``x, y, z`` and ``xx, xy, xz, yy, yz, zz``; a shell's functions are zero
    farther than its entry of ``radii`` from its centre.
    """

def bloch_basis_values(
    table: ShellTable,
    points: NDArray[np.float64],
    radii: NDArray[np.float64],
    translations: NDArray[np.float64],
    phases: NDArray[np.complex128],
    order: int = 0,
) -> NDArray[np.complex128]:
    """Return the Bloch sums of the basis functions at ``points``.

    The result is ``[kpt, g, component, ao]``: ``Σ_T exp(i k·T) φ(r − T)``
    over the ``translations`` ``[image, 3]``
    with the ``phases`` ``exp(i k·T)`` given as ``[kpt, image]``; the
    components and the cutoff ``radii`` are those of :func:`basis_values`,
    the radii applying to every image of a shell.
    """

def partition_weights(
    points: NDArray[np.float64],
    centers: NDArray[np.float64],
    owner: int,
    adjustments: NDArray[np.float64] | None = None,
    scheme: str = "becke",
) -> NDArray[np.float64]:
    """Return the partition weights ``[g]`` of the centre ``owner`` at ``points``.

    The normalized cell function of ``centers[owner]`` among all ``centers``
    ``[centre, 3]``, with Becke's step (``scheme`` ``"becke"``) or the
    compactly supported step of Stratmann, Scuseria and Frisch
    (``"stratmann"``), and the atomic size ``adjustments`` ``[centre, centre]``
    when given.
    """

def functional_info(name: str) -> dict[str, Any]:
    """Return the metadata of the functional with the libxc ``name``.

    Keys: ``id``, ``name``, ``rung`` (``"lda"``, ``"gga"``, ``"meta-gga"``),
    ``needs_laplacian``, ``exact_exchange``, ``range_separation`` (``(omega,
    alpha, beta)`` or ``None``), ``vv10`` (``(b, C)`` or ``None``) and
    ``density_threshold``.
    """

def evaluate_functional(
    names: list[str],
    weights: NDArray[np.float64],
    rho: NDArray[np.float64],
    sigma: NDArray[np.float64] | None = None,
    tau: NDArray[np.float64] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    """Return ``(exc, vrho, vsigma, vtau)`` of ``Σ weights[i] names[i]``.

    ``rho`` is ``[2, g]``, ``sigma`` ``[3, g]`` and ``tau`` ``[2, g]``; the
    energy per particle is ``[g]`` and the derivatives of the energy density
    follow the input layouts, ``None`` below the rung that needs them.
    """

def evaluate_functional_second(
    names: list[str],
    weights: NDArray[np.float64],
    rho: NDArray[np.float64],
    sigma: NDArray[np.float64] | None = None,
    tau: NDArray[np.float64] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    """Return ``(v2rho2, v2rhosigma, v2sigma2, v2rhotau, v2sigmatau, v2tau2)``.

    In libxc's packed layouts over the first axis, ``[3, g]``, ``[6, g]``,
    ``[6, g]``, ``[4, g]``, ``[6, g]`` and ``[3, g]``; ``None`` for the rows
    the rung does not read.
    """

def evaluate_noncollinear(
    names: list[str],
    weights: NDArray[np.float64],
    rho: NDArray[np.float64],
    m: NDArray[np.float64],
    grad_rho: NDArray[np.float64] | None = None,
    grad_m: NDArray[np.float64] | None = None,
    tau: NDArray[np.float64] | None = None,
    tau_m: NDArray[np.float64] | None = None,
    directions: NDArray[np.float64] | None = None,
    direction_weights: NDArray[np.float64] | None = None,
    precision: float = 1e-14,
    max_level: int = 6,
    collinear_threshold: float | None = None,
    point_weights: NDArray[np.float64] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    int,
]:
    """Multi-collinear evaluation of ``Σ weights[i] names[i]`` (ADR-0029).

    ``rho`` is ``[g]``, ``m`` ``[3, g]``, ``grad_rho`` ``[3, g]``, ``grad_m``
    ``[3, 3, g]`` (spin direction before spatial direction), ``tau`` ``[g]``
    and ``tau_m`` ``[3, g]``. The sphere is sampled at ``directions [k, 3]``
    with ``direction_weights [k]`` summing to one when both are given, else
    with product rules aligned with the magnetization and refined until the
    energy density and the first-order energy response change by less than
    ``precision`` divided by the point's weight (``point_weights [g]``, one
    without), up to ``max_level``; an LDA samples nothing. Points with
    ``|m| ≥ collinear_threshold · rho`` take the locally collinear
    functional. Returns the energy density ``[g]``, the derivatives with
    respect to ``rho, m, grad_rho, grad_m, tau, tau_m`` in the input
    layouts, ``None`` below the rung that reads a variable, and the number
    of points whose rule did not converge.
    """

def configure_threads(n_threads: int) -> None:
    """Set the size of the kernel thread pool once per process.

    Raises ``ValueError`` if the pool was already configured.
    """

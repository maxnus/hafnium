"""Density fitting with momentum transfer (ADR-0019).

The pair density ``conj(φ_u^{k1}) φ_v^{k2}`` of two Bloch functions has
the character of the momentum transfer ``q = k2 − k1`` and is fitted in
the Bloch sums ``χ_P^q`` of the auxiliary functions with the Coulomb
metric, so that ``(u k1 v k2 | w k3 x k4) ≈ Σ_P B[k1, k2, P, u, v]
conj(B[k4, k3, P, x, w])`` for ``k3 − k4 = q`` (chemists' notation with the
first function of each pair conjugated, CONVENTIONS.md §3) and
``B = (M^q)^{-1/2} b`` with the metric ``M^q = (P^q | Q^q)`` and the
three-centre integrals ``b[k1, k2] = (P^q | u k1 v k2)``.

Both are lattice sums split by the range separation of
:class:`~hafnium.integrals.periodic.RangeSeparation`. The short-range
parts run in the compiled layer over the images the functions reach: the
metric as the Bloch sum over the images of one auxiliary function, the
three-centre integrals as the sum over the image pairs of the orbital
functions with the auxiliary function in the home cell. The long-range
parts are reciprocal-space sums over ``q + G`` of the kernel of ADR-0012
with the Fourier transforms of the auxiliary functions and of the Bloch
pair densities, the finite part at ``q + G = 0`` included.
"""

from __future__ import annotations

from dataclasses import dataclass

from hafnium.backend import Array, array_namespace, default_namespace
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals.density_fitting import (
    DEFAULT_METRIC_THRESHOLD,
    inverse_square_root,
)
from hafnium.integrals.engine import BlochImages, CoulombKernel, IntegralEngine
from hafnium.integrals.periodic.bloch import (
    atom_radii,
    bloch_sum,
    function_absolute_integral,
    integer_coordinates,
    lattice_images,
    reaching_images,
    require_kpoints_for_lattice,
)
from hafnium.integrals.periodic.coulomb import (
    RangeSeparation,
    require_three_dimensions,
)
from hafnium.integrals.shell_table import ShellTable
from hafnium.molecule import Cell
from hafnium.symmetry import KPoints

_TRANSFORM_CHUNK_BYTES = 128 * 2**20
"""Memory budget of one batch of pair transforms ``[image, G, ao, ao]``."""


@dataclass(frozen=True, slots=True, eq=False)
class PeriodicDensityFitting:
    """Metric and three-centre integrals of a cell's auxiliary basis over k-points.

    Parameters
    ----------
    kpoints
        The k-point set, a uniform mesh.
    transfers
        The momentum transfers ``q = k2 − k1`` of the mesh, see
        :meth:`hafnium.symmetry.KPoints.momentum_transfers`.
    transfer_index : Array[kpt, kpt]
        Index in ``transfers`` of the transfer of every pair ``(k1, k2)``.
    metric : Array[q, aux, aux]
        ``(P^q | Q^q)``, complex and Hermitian at every transfer.
    three_center : Array[kpt, kpt, aux, ao, ao]
        ``(P^q | u k1 v k2)`` with ``q`` the transfer of ``(k1, k2)``.
    """

    kpoints: KPoints
    transfers: KPoints
    transfer_index: Array
    metric: Array
    three_center: Array

    def __post_init__(self) -> None:
        n_kpt, n_q = self.kpoints.n_kpt, self.transfers.n_kpt
        n_aux = self.metric.shape[1] if self.metric.ndim == 3 else -1
        if self.metric.shape != (n_q, n_aux, n_aux):
            raise InputError("the metric has shape [q, aux, aux]")
        shape = self.three_center.shape
        if (
            self.three_center.ndim != 5
            or shape[:3] != (n_kpt, n_kpt, n_aux)
            or shape[3] != shape[4]
        ):
            raise InputError(
                "three-centre integrals have shape [kpt, kpt, aux, ao, ao]"
            )
        if self.transfer_index.shape != (n_kpt, n_kpt):
            raise InputError("the transfer table has shape [kpt, kpt]")

    @property
    def n_aux(self) -> int:
        """Number of auxiliary functions."""
        return int(self.metric.shape[1])

    @property
    def n_ao(self) -> int:
        """Number of orbital basis functions."""
        return int(self.three_center.shape[3])

    def df_tensor(self, *, threshold: float = DEFAULT_METRIC_THRESHOLD) -> Array:
        """``df_tensor[kpt, kpt, aux, ao, ao]`` with the metric absorbed.

        ``(M^q)^{-1/2} b[k1, k2]`` for the transfer ``q`` of each pair; metric
        eigenvalues below ``threshold`` times the largest are dropped, see
        :func:`hafnium.integrals.density_fitting.inverse_square_root`.
        """
        xp = array_namespace(self.metric)
        n_kpt, n_aux = self.kpoints.n_kpt, self.n_aux
        halves = [
            inverse_square_root(self.metric[q], threshold)
            for q in range(self.transfers.n_kpt)
        ]
        rows: list[Array] = []
        for k1 in range(n_kpt):
            fitted: list[Array] = []
            for k2 in range(n_kpt):
                half = halves[int(self.transfer_index[k1, k2])]
                flat = xp.reshape(self.three_center[k1, k2], (n_aux, -1))
                fitted.append(xp.reshape(half @ flat, self.three_center.shape[2:]))
            rows.append(xp.stack(fitted))
        return xp.stack(rows)

    def eri(
        self,
        kpts: tuple[int, int, int, int],
        *,
        threshold: float = DEFAULT_METRIC_THRESHOLD,
    ) -> Array:
        """Fitted ``(u k1 v k2 | w k3 x k4)`` as ``eri[ao, ao, ao, ao]``.

        Chemists' notation with the first function of each pair conjugated;
        the quadruple must conserve crystal momentum, ``k2 − k1 = k3 − k4``.
        Stored in full; for tests and small systems.
        """
        k1, k2, k3, k4 = kpts
        if int(self.transfer_index[k1, k2]) != int(self.transfer_index[k4, k3]):
            raise InputError(
                f"the k-points {kpts} do not conserve crystal momentum: "
                "k2 − k1 must equal k3 − k4"
            )
        xp = array_namespace(self.metric)
        half = inverse_square_root(
            self.metric[int(self.transfer_index[k1, k2])], threshold
        )
        n_aux = self.n_aux
        first = xp.reshape(
            half @ xp.reshape(self.three_center[k1, k2], (n_aux, -1)),
            (n_aux, self.n_ao, self.n_ao),
        )
        second = xp.reshape(
            half @ xp.reshape(self.three_center[k4, k3], (n_aux, -1)),
            (n_aux, self.n_ao, self.n_ao),
        )
        return xp.einsum("Puv,Pxw->uvwx", first, xp.conj(second))


def density_fitting_k(
    engine: IntegralEngine,
    basis: BasisSet,
    auxiliary: BasisSet,
    cell: Cell,
    kpoints: KPoints,
    separation: RangeSeparation | None = None,
) -> PeriodicDensityFitting:
    """Evaluate the metric and three-centre integrals of ``cell`` over ``kpoints``.

    ``separation`` carries the range-separation parameter and the precision
    of the lattice sums; by default :meth:`RangeSeparation.for_lattice` of
    the cell's lattice.
    """
    require_three_dimensions(cell.lattice)
    require_kpoints_for_lattice(kpoints, cell.lattice)
    if separation is None:
        separation = RangeSeparation.for_lattice(cell.lattice)
    transfers, table = kpoints.momentum_transfers()
    orbital = ShellTable.from_basis(basis, cell)
    fitting = ShellTable.from_basis(auxiliary, cell)
    builder = _Builder(engine, cell, separation, kpoints, transfers)
    metric = builder.metric(fitting, auxiliary)
    three_center = builder.three_center(fitting, auxiliary, orbital, basis, table)
    return PeriodicDensityFitting(kpoints, transfers, table, metric, three_center)


@dataclass(frozen=True, slots=True)
class _Builder:
    """The lattice sums of one cell, shared by the metric and the tensor."""

    engine: IntegralEngine
    cell: Cell
    separation: RangeSeparation
    kpoints: KPoints
    transfers: KPoints

    def metric(self, fitting: ShellTable, auxiliary: BasisSet) -> Array:
        """``(P^q | Q^q)`` as complex ``[q, aux, aux]``."""
        xp = default_namespace()
        lattice = self.cell.lattice
        positions = self.cell.positions_array(xp)
        radii = atom_radii(auxiliary, self.separation.precision)
        reach = tuple(r + self.separation.short_range_radius for r in radii)
        images = reaching_images(lattice, positions, radii, positions, reach)
        per_image = self.separation.precision / max(int(images.shape[0]), 1)
        short = self.engine.coulomb_two_center(
            fitting,
            fitting,
            images,
            kernel=CoulombKernel(self.separation.omega, short_range=True),
            threshold=per_image,
        )
        total = bloch_sum(short, images, self.transfers, lattice)
        charge = max(function_absolute_integral(shell) for shell in auxiliary.shells)
        rows: list[Array] = []
        for q in range(self.transfers.n_kpt):
            vectors = self._vectors(q, charge * charge)
            weights, transform = self._auxiliary_weights(fitting, vectors)
            rows.append(total[q] + weights @ transform)
        return xp.stack(rows)

    def three_center(
        self,
        fitting: ShellTable,
        auxiliary: BasisSet,
        orbital: ShellTable,
        basis: BasisSet,
        table: Array,
    ) -> Array:
        """``(P^q | u k1 v k2)`` as complex ``[kpt, kpt, aux, ao, ao]``."""
        xp = default_namespace()
        lattice = self.cell.lattice
        positions = self.cell.positions_array(xp)
        precision = self.separation.precision
        pair_images = lattice_images(basis, self.cell, precision=precision)
        aux_reach = tuple(
            r + self.separation.short_range_radius
            for r in atom_radii(auxiliary, precision)
        )
        images = reaching_images(
            lattice, positions, aux_reach, positions, atom_radii(basis, precision)
        )
        # Every element collects one term per step of every image; the
        # terms fall off with the step, so the step count sets the threshold.
        per_pair = precision / max(int(pair_images.shape[0]), 1)
        kpts = xp.asarray(self.kpoints.cartesian(lattice))
        angles = xp.astype(kpts @ xp.linalg.matrix_transpose(images), xp.complex128)
        bloch = BlochImages(
            images,
            pair_images,
            (
                integer_coordinates(lattice, images),
                integer_coordinates(lattice, pair_images),
            ),
            (xp.exp(-1j * angles), xp.exp(1j * angles)),
        )
        short = self.engine.coulomb_three_center_bloch(
            fitting,
            orbital,
            orbital,
            bloch,
            kernel=CoulombKernel(self.separation.omega, short_range=True),
            threshold=per_pair,
        )
        charge = max(function_absolute_integral(shell) for shell in auxiliary.shells)
        n_images = int(pair_images.shape[0])
        pair_angles = xp.astype(
            kpts @ xp.linalg.matrix_transpose(pair_images), xp.complex128
        )
        pair_phases = xp.exp(1j * pair_angles)
        parts = [
            [short[k1, k2] for k2 in range(self.kpoints.n_kpt)]
            for k1 in range(self.kpoints.n_kpt)
        ]
        for q in range(self.transfers.n_kpt):
            vectors = self._vectors(q, charge)
            weights, _ = self._auxiliary_weights(fitting, vectors)
            summed = self._pair_transforms(
                orbital,
                pair_images,
                pair_phases,
                vectors,
                precision / (n_images * self._weight_norm(weights)),
            )
            for k2 in range(self.kpoints.n_kpt):
                k1 = int(xp.argmax(xp.astype(table[:, k2] == q, xp.int64)))
                parts[k1][k2] = parts[k1][k2] + xp.tensordot(
                    weights, summed[k2], axes=1
                )
        return xp.stack([xp.stack(row) for row in parts])

    def _vectors(self, q: int, charge_product: float) -> Array:
        """Return the vectors ``q + G`` of the long-range sum at transfer ``q``."""
        xp = default_namespace()
        lattice = self.cell.lattice
        q_cartesian = xp.asarray(self.transfers.cartesian(lattice))[q]
        reciprocal = self.separation.reciprocal_vectors(
            lattice,
            charge_product=charge_product,
            margin=float(xp.linalg.vector_norm(q_cartesian)),
        )
        return reciprocal + q_cartesian

    def _auxiliary_weights(
        self, fitting: ShellTable, vectors: Array
    ) -> tuple[Array, Array]:
        """Return the weights ``ṽ(q+G) conj(χ̂_P(q+G)) / Ω`` ``[aux, G]`` and ``χ̂``.

        The transform ``χ̂_Q(q+G)`` comes as ``[G, aux]``.
        """
        xp = array_namespace(vectors)
        kernel = self.separation.long_range_kernel(self.cell.lattice, vectors)
        transform = self.engine.fourier_transform(fitting, vectors)
        weights = (
            xp.linalg.matrix_transpose(xp.conj(transform))
            * xp.astype(kernel, xp.complex128)
            / self.cell.lattice.volume
        )
        return weights, transform

    @staticmethod
    def _weight_norm(weights: Array) -> float:
        xp = array_namespace(weights)
        return max(float(xp.max(xp.sum(xp.abs(weights), axis=1))), 1.0)

    def _pair_transforms(
        self,
        orbital: ShellTable,
        pair_images: Array,
        pair_phases: Array,
        vectors: Array,
        threshold: float,
    ) -> Array:
        """Return ``Σ_T exp(i k2·T) ρ̃_T(q+G)`` as ``[kpt, G, ao, ao]``, in batches."""
        xp = array_namespace(vectors)
        n_ao = orbital.n_functions
        per_image = int(vectors.shape[0]) * n_ao * n_ao * 16
        chunk = max(1, _TRANSFORM_CHUNK_BYTES // per_image)
        total = xp.zeros(
            (self.kpoints.n_kpt, int(vectors.shape[0]), n_ao, n_ao), dtype=xp.complex128
        )
        for start in range(0, int(pair_images.shape[0]), chunk):
            transforms = self.engine.pair_fourier_transform(
                orbital,
                orbital,
                pair_images[start : start + chunk],
                vectors,
                threshold=threshold,
            )
            total = total + xp.tensordot(
                pair_phases[:, start : start + chunk], transforms, axes=1
            )
        return total

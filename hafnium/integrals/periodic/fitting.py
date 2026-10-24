"""Density fitting with momentum transfer (ADR-0019).

The pair density ``conj(φ_u^{k1}) φ_v^{k2}`` of two Bloch functions has
the character of the momentum transfer ``q = k2 − k1`` and is fitted in
the Bloch sums ``χ_P^q`` of the auxiliary functions with the Coulomb
metric, so that ``(u k1 v k2 | w k3 x k4) ≈ Σ_P s_P B[k1, k2, P, u, v]
conj(B[k4, k3, P, x, w])`` for ``k3 − k4 = q`` (chemists' notation with the
first function of each pair conjugated, CONVENTIONS.md §3) and
``B = |Λ^q|^{-1/2} (U^q)† b`` in the eigenbasis ``M^q = U^q Λ^q (U^q)†`` of
the metric ``M^q = (P^q | Q^q)``, with the three-centre integrals
``b[k1, k2] = (P^q | u k1 v k2)`` and the signs ``s_P`` of the eigenvalues.
The signs matter for a slab or a chain, whose metric at ``q = 0`` has one
negative eigenvalue along the charge of the auxiliary functions
(ADR-0021); in three dimensions every sign is positive.

Both are lattice sums split by the range separation of
:class:`~hafnium.integrals.periodic.RangeSeparation`. The short-range
parts run in the compiled layer over the images the short-range kernel
reaches (:func:`~hafnium.integrals.periodic.coulomb_reach`, ADR-0025):
the metric as the Bloch sum over the images of one auxiliary function,
the three-centre integrals as the sum over the image pairs of the orbital
functions with the auxiliary function in the home cell. The long-range
parts are reciprocal-space sums over ``q + G`` of the kernel of ADR-0012
with the Fourier transforms of the auxiliary functions and of the Bloch
pair densities, the finite part at ``q + G = 0`` included; for a slab or a
chain they run over the vectors and weights of
:meth:`RangeSeparation.reciprocal_sum`, the quadrature over the open
directions of ADR-0021.
"""

from __future__ import annotations

from dataclasses import dataclass

from hafnium.backend import Array, array_namespace, default_namespace
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals.density_fitting import (
    DEFAULT_METRIC_THRESHOLD,
    signed_inverse_square_root,
)
from hafnium.integrals.engine import BlochImages, CoulombKernel, IntegralEngine
from hafnium.integrals.periodic.bloch import (
    bloch_sum,
    coulomb_reach,
    function_absolute_integral,
    integer_coordinates,
    lattice_images,
    periodic_amplitude,
    reaching_images,
    require_kpoints_for_lattice,
)
from hafnium.integrals.periodic.coulomb import (
    PerpendicularProfile,
    RangeSeparation,
    ReciprocalSum,
)
from hafnium.integrals.shell_table import ShellTable
from hafnium.molecule import Cell
from hafnium.symmetry import KPoints

_TRANSFORM_CHUNK_BYTES = 128 * 2**20
"""Memory budget of one batch of pair transforms ``[image, G, ao, ao]``."""

_SHIFT_TOLERANCE = 1e-8
"""A mesh counts as shifted along a direction when its points sit this far
from the multiples of ``1/n``."""


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

        ``|Λ^q|^{-1/2} (U^q)† b[k1, k2]`` for the transfer ``q`` of each pair,
        in the eigenbasis of the metric; eigenvalues of magnitude below
        ``threshold`` times the largest are dropped, see
        :func:`hafnium.integrals.density_fitting.signed_inverse_square_root`.
        Contractions carry the :meth:`signature` of the same threshold.
        """
        xp = array_namespace(self.metric)
        n_kpt, n_aux = self.kpoints.n_kpt, self.n_aux
        halves = [half for half, _ in self._halves(threshold)]
        rows: list[Array] = []
        for k1 in range(n_kpt):
            fitted: list[Array] = []
            for k2 in range(n_kpt):
                half = halves[int(self.transfer_index[k1, k2])]
                flat = xp.reshape(self.three_center[k1, k2], (n_aux, -1))
                fitted.append(xp.reshape(half @ flat, self.three_center.shape[2:]))
            rows.append(xp.stack(fitted))
        return xp.stack(rows)

    def signature(self, *, threshold: float = DEFAULT_METRIC_THRESHOLD) -> Array:
        """Signs ``[kpt, kpt, aux]`` of the metric eigenvalues behind :meth:`df_tensor`.

        ``+1``, ``−1`` or ``0`` for a dropped eigenvalue, at the transfer of
        every pair of k-points; a fitted integral is
        ``Σ_P signature[k1, k2, P] B[k1, k2, P] conj(B[k4, k3, P])``.
        """
        xp = array_namespace(self.metric)
        signs = xp.stack([sign for _, sign in self._halves(threshold)])
        n_kpt = self.kpoints.n_kpt
        flat = xp.take(signs, xp.reshape(self.transfer_index, (-1,)), axis=0)
        return xp.reshape(flat, (n_kpt, n_kpt, self.n_aux))

    def _halves(self, threshold: float) -> list[tuple[Array, Array]]:
        """Return the signed inverse square root of the metric at every transfer."""
        return [
            signed_inverse_square_root(self.metric[q], threshold)
            for q in range(self.transfers.n_kpt)
        ]

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
        half, signs = signed_inverse_square_root(
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
        return xp.einsum(
            "P,Puv,Pxw->uvwx", xp.astype(signs, xp.complex128), first, xp.conj(second)
        )


@dataclass(frozen=True, slots=True, eq=False)
class PeriodicCoulombFitting:
    """The blocks of zero momentum transfer of a cell's density fitting (ADR-0028).

    The Coulomb matrix ``J(k)`` needs only ``(P^0 | u k v k)`` and the metric
    at ``q = 0``, one block per k-point instead of one per pair, so a pure
    functional fits the Coulomb term alone and skips the exchange blocks.

    Parameters
    ----------
    kpoints
        The k-point set, a uniform mesh.
    metric : Array[aux, aux]
        ``(P^0 | Q^0)``, complex and Hermitian.
    three_center : Array[kpt, aux, ao, ao]
        ``(P^0 | u k v k)`` at every k-point.
    """

    kpoints: KPoints
    metric: Array
    three_center: Array

    def __post_init__(self) -> None:
        n_aux = self.metric.shape[0] if self.metric.ndim == 2 else -1
        if self.metric.shape != (n_aux, n_aux):
            raise InputError("the metric has shape [aux, aux]")
        shape = self.three_center.shape
        if (
            self.three_center.ndim != 4
            or shape[:2] != (self.kpoints.n_kpt, n_aux)
            or shape[2] != shape[3]
        ):
            raise InputError("three-centre integrals have shape [kpt, aux, ao, ao]")

    @property
    def n_aux(self) -> int:
        """Number of auxiliary functions."""
        return int(self.metric.shape[0])

    @property
    def n_ao(self) -> int:
        """Number of orbital basis functions."""
        return int(self.three_center.shape[2])

    def df_tensor(self, *, threshold: float = DEFAULT_METRIC_THRESHOLD) -> Array:
        """``df_tensor[kpt, aux, ao, ao]`` with the metric at zero transfer absorbed.

        See :meth:`PeriodicDensityFitting.df_tensor`; contractions carry the
        :meth:`signature` of the same threshold.
        """
        xp = array_namespace(self.metric)
        half, _ = signed_inverse_square_root(self.metric, threshold)
        n_aux = self.n_aux
        return xp.stack(
            [
                xp.reshape(
                    half @ xp.reshape(self.three_center[k], (n_aux, -1)),
                    self.three_center.shape[1:],
                )
                for k in range(self.kpoints.n_kpt)
            ]
        )

    def signature(self, *, threshold: float = DEFAULT_METRIC_THRESHOLD) -> Array:
        """Signs ``[aux]`` of the metric eigenvalues behind :meth:`df_tensor`."""
        _, signs = signed_inverse_square_root(self.metric, threshold)
        return signs


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
    builder, tables, table = _prepare(
        engine, basis, auxiliary, cell, kpoints, separation
    )
    fitting, orbital = tables
    metric = builder.metric(fitting, auxiliary)
    three_center = builder.three_center(fitting, auxiliary, orbital, basis, table)
    return PeriodicDensityFitting(
        kpoints, builder.transfers, table, metric, three_center
    )


def coulomb_fitting_k(
    engine: IntegralEngine,
    basis: BasisSet,
    auxiliary: BasisSet,
    cell: Cell,
    kpoints: KPoints,
    separation: RangeSeparation | None = None,
) -> PeriodicCoulombFitting:
    """Evaluate the zero-transfer metric and three-centre integrals of ``cell``.

    The fitting of the Coulomb term alone (ADR-0028): the same lattice sums
    as :func:`density_fitting_k` restricted to ``q = 0``, one three-centre
    block per k-point.
    """
    builder, tables, table = _prepare(
        engine, basis, auxiliary, cell, kpoints, separation
    )
    fitting, orbital = tables
    metric = builder.metric(fitting, auxiliary)[_ZERO_TRANSFER]
    three_center = builder.three_center(
        fitting, auxiliary, orbital, basis, table, coulomb_only=True
    )
    return PeriodicCoulombFitting(kpoints, metric, three_center)


_ZERO_TRANSFER = 0
"""Index of the zero momentum transfer in :meth:`KPoints.momentum_transfers`."""


def _prepare(
    engine: IntegralEngine,
    basis: BasisSet,
    auxiliary: BasisSet,
    cell: Cell,
    kpoints: KPoints,
    separation: RangeSeparation | None,
) -> tuple[_Builder, tuple[ShellTable, ShellTable], Array]:
    """Return the builder, the (auxiliary, orbital) tables and the transfer table."""
    require_kpoints_for_lattice(kpoints, cell.lattice)
    if separation is None:
        separation = RangeSeparation.for_lattice(cell.lattice)
    transfers, table = kpoints.momentum_transfers()
    if any(coordinate != 0.0 for coordinate in transfers.fractional[_ZERO_TRANSFER]):
        raise InputError("the first momentum transfer of a mesh is zero")
    orbital = ShellTable.from_basis(basis, cell)
    fitting = ShellTable.from_basis(auxiliary, cell)
    builder = _Builder(engine, cell, separation, kpoints, transfers)
    return builder, (fitting, orbital), table


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
        reach = coulomb_reach(
            auxiliary,
            auxiliary,
            lattice,
            omega=self.separation.omega,
            precision=self.separation.precision,
        )
        images = reaching_images(lattice, positions, positions, reach)
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
        profile = PerpendicularProfile.from_functions(self.cell, auxiliary)
        rows: list[Array] = []
        for q in range(self.transfers.n_kpt):
            reciprocal = self._reciprocal_sum(q, charge * charge, profile)
            weights, transform = self._auxiliary_weights(fitting, reciprocal)
            rows.append(total[q] + weights @ transform)
        return xp.stack(rows)

    def three_center(
        self,
        fitting: ShellTable,
        auxiliary: BasisSet,
        orbital: ShellTable,
        basis: BasisSet,
        table: Array,
        *,
        coulomb_only: bool = False,
    ) -> Array:
        """``(P^q | u k1 v k2)`` as complex ``[kpt, kpt, aux, ao, ao]``.

        The orbital functions are imaged where the short-range kernel
        reaches them from the home-cell auxiliary functions, with the
        third function of the triple bounded by its periodic amplitude.
        With ``coulomb_only`` the blocks of zero transfer alone,
        ``[kpt, aux, ao, ao]`` (ADR-0028).
        """
        xp = default_namespace()
        lattice = self.cell.lattice
        positions = self.cell.positions_array(xp)
        precision = self.separation.precision
        pair_images = lattice_images(basis, self.cell, precision=precision)
        reach = coulomb_reach(
            auxiliary,
            basis,
            lattice,
            omega=self.separation.omega,
            precision=precision,
            spectator=periodic_amplitude(basis, lattice),
        )
        images = reaching_images(lattice, positions, positions, reach)
        # Every element collects one term per step of every image; the
        # terms fall off with the step, so the step count sets the threshold.
        per_pair = precision / max(int(pair_images.shape[0]), 1)
        kpts = xp.asarray(self.kpoints.cartesian(lattice))
        coordinates = integer_coordinates(lattice, images)
        step_coordinates = integer_coordinates(lattice, pair_images)
        step_classes = self._step_classes(step_coordinates)
        bloch = BlochImages(
            images,
            pair_images,
            (coordinates, step_coordinates),
            self._image_classes(coordinates),
            step_classes,
            kpts,
        )
        total = self.engine.coulomb_three_center_bloch(
            fitting,
            orbital,
            orbital,
            bloch,
            kernel=CoulombKernel(self.separation.omega, short_range=True),
            threshold=per_pair,
            coulomb_only=coulomb_only,
        )
        charge = max(function_absolute_integral(shell) for shell in auxiliary.shells)
        profile = PerpendicularProfile.from_functions(self.cell, basis, auxiliary)
        n_images = int(pair_images.shape[0])
        steps = (
            pair_images,
            step_classes,
            self._class_phases(kpts, pair_images, step_classes),
        )
        transfers = [_ZERO_TRANSFER] if coulomb_only else range(self.transfers.n_kpt)
        # The reciprocal-space part is added block by block into the array
        # the engine returned, which holds the whole tensor already.
        for q in transfers:
            reciprocal = self._reciprocal_sum(q, charge, profile)
            weights, _ = self._auxiliary_weights(fitting, reciprocal)
            summed = self._pair_transforms(
                orbital,
                steps,
                reciprocal.vectors,
                precision / (n_images * self._weight_norm(weights)),
            )
            for k2 in range(self.kpoints.n_kpt):
                addition = xp.tensordot(weights, summed[k2], axes=1)
                if coulomb_only:
                    total[k2] = total[k2] + addition
                else:
                    k1 = int(xp.argmax(xp.astype(table[:, k2] == q, xp.int64)))
                    total[k1, k2] = total[k1, k2] + addition
        return total

    @staticmethod
    def _class_phases(kpts: Array, images: Array, classes: Array) -> Array:
        """``exp(i k·T_c)`` as ``[kpt, class]`` from the first image of every class."""
        xp = array_namespace(images)
        n_classes = int(xp.max(classes)) + 1
        representatives = xp.stack(
            [
                images[xp.argmax(xp.astype(classes == c, xp.int64))]
                for c in range(n_classes)
            ]
        )
        angles = xp.astype(
            kpts @ xp.linalg.matrix_transpose(representatives), xp.complex128
        )
        return xp.exp(1j * angles)

    def _image_classes(self, coordinates: Array) -> Array:
        """Return the Born–von Kármán class of every image.

        The lattice ``coordinates`` ``[image, 3]`` modulo the mesh sizes,
        flattened: ``exp(i q·T)`` is the same for every image of a class at
        every momentum transfer ``q`` of the mesh.
        """
        return self._classes(coordinates, (1, 1, 1))

    def _step_classes(self, coordinates: Array) -> Array:
        """Return the class of every step for the phases ``exp(i k·T_s)`` (ADR-0028).

        A k-point ``(m + ½) / n`` of a shifted mesh changes sign over ``n``
        cells, so along a shifted direction the period of the classes is
        twice the mesh size.
        """
        xp = array_namespace(coordinates)
        sizes = self.kpoints.mesh_size()
        fractional = self.kpoints.fractional_array(xp)
        offsets = fractional * xp.asarray(sizes, dtype=xp.float64)
        offsets = offsets - xp.floor(offsets + 0.5)
        shifted = xp.any(xp.abs(offsets) > _SHIFT_TOLERANCE, axis=0)
        multipliers = tuple(2 if bool(flag) else 1 for flag in shifted)
        return self._classes(coordinates, multipliers + (1,) * (3 - len(sizes)))

    def _classes(self, coordinates: Array, multipliers: tuple[int, ...]) -> Array:
        """Flattened ``coordinates`` modulo the mesh sizes times ``multipliers``."""
        xp = array_namespace(coordinates)
        sizes = self.kpoints.mesh_size() + (1,) * (3 - self.cell.lattice.dimension)
        sizes = tuple(
            size * factor for size, factor in zip(sizes, multipliers, strict=True)
        )
        n = xp.asarray(sizes, dtype=xp.int64)
        residues = coordinates - n * xp.floor_divide(coordinates, n)
        return (residues[:, 0] * sizes[1] + residues[:, 1]) * sizes[2] + residues[:, 2]

    def _reciprocal_sum(
        self, q: int, charge_product: float, profile: PerpendicularProfile
    ) -> ReciprocalSum:
        """Return the vectors ``q + G`` and weights of the sum at transfer ``q``."""
        xp = default_namespace()
        lattice = self.cell.lattice
        q_cartesian = xp.asarray(self.transfers.cartesian(lattice))[q]
        return self.separation.reciprocal_sum(
            lattice, profile, charge_product=charge_product, transfer=q_cartesian
        )

    def _auxiliary_weights(
        self, fitting: ShellTable, reciprocal: ReciprocalSum
    ) -> tuple[Array, Array]:
        """Return the weights ``ṽ(q+G) conj(χ̂_P(q+G)) / Ω`` ``[aux, G]`` and ``χ̂``.

        The transform ``χ̂_Q(q+G)`` comes as ``[G, aux]``; the kernel carries
        the quadrature weights of the open directions.
        """
        xp = array_namespace(reciprocal.vectors)
        transform = self.engine.fourier_transform(fitting, reciprocal.vectors)
        weights = (
            xp.linalg.matrix_transpose(xp.conj(transform))
            * xp.astype(reciprocal.weights, xp.complex128)
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
        steps: tuple[Array, Array, Array],
        vectors: Array,
        threshold: float,
    ) -> Array:
        """Return ``Σ_T exp(i k·T) ρ̃_T(q+G)`` as ``[kpt, G, ao, ao]``, in batches.

        ``steps`` holds the pair images ``[image, 3]``, their classes
        ``[image]`` and the phases ``exp(i k·T_c)`` ``[kpt, class]`` of one
        representative per class: the transforms are summed per class
        first, since the phase is the same within a class, and the phases
        multiply the class sums (ADR-0028).
        """
        xp = array_namespace(vectors)
        pair_images, classes, class_phases = steps
        n_ao = orbital.n_functions
        n_classes = int(class_phases.shape[1])
        per_image = int(vectors.shape[0]) * n_ao * n_ao * 16
        chunk = max(1, _TRANSFORM_CHUNK_BYTES // per_image)
        order = xp.argsort(classes, stable=True)
        sorted_classes = classes[order]
        sorted_images = pair_images[order]
        # The images of a class are contiguous after the sort; the class
        # sums are accumulated into a preallocated array, batch by batch.
        totals = xp.zeros(
            (n_classes, int(vectors.shape[0]), n_ao, n_ao), dtype=xp.complex128
        )
        for start in range(0, int(pair_images.shape[0]), chunk):
            stop = min(start + chunk, int(pair_images.shape[0]))
            transforms = self.engine.pair_fourier_transform(
                orbital,
                orbital,
                sorted_images[start:stop],
                vectors,
                threshold=threshold,
            )
            batch_classes = sorted_classes[start:stop]
            first = int(batch_classes[0])
            last = int(batch_classes[-1])
            for c in range(first, last + 1):
                members = batch_classes == c
                if bool(xp.any(members)):
                    totals[c] = totals[c] + xp.sum(transforms[members], axis=0)
        return xp.tensordot(class_phases, totals, axes=1)

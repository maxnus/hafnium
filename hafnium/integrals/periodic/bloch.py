"""Bloch sums of one-electron integrals over lattice images (ADR-0016, ADR-0025).

The basis functions of a cell are the Bloch sums
``φ_μ^k(r) = Σ_T exp(i k·T) φ_μ(r − T)`` (CONVENTIONS.md §3, ADR-0007), so
a one-electron matrix at ``k`` is the lattice sum
``M(k) = Σ_T exp(i k·T) M_T`` of the matrices ``M_T`` between the cell at
the origin and its image translated by ``T``, which the integral engine
evaluates for a list of translations. The exponents of the functions
bound the images that contribute: an image is kept while some atom of
the cell and some atom of the image lie within their reach, the distance
beyond which the products of their functions, summed over every farther
image, fall below a precision (:func:`pair_reach`). The Coulomb sums of
the density fitting choose their images by the reach of the short-range
kernel between two functions in the same way (:func:`coulomb_reach`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from hafnium.backend import Array, array_namespace, default_namespace
from hafnium.basis import BasisSet, Shell
from hafnium.errors import InputError
from hafnium.integrals.engine import IntegralEngine
from hafnium.integrals.shell_table import ShellTable, require_basis_for_atoms
from hafnium.molecule import Cell, Lattice
from hafnium.symmetry import KPoints

DEFAULT_LATTICE_SUM_PRECISION = 1e-12
"""Bound on what the images left out of a lattice sum contribute to any
matrix element: the tail of the sum beyond the images kept."""

_REACH_ELEMENTS = 4_000_000
"""Elements of the largest temporary array of :func:`reaching_images`."""

_RADIUS_BISECTIONS = 60
"""Bisection steps of a radius equation, bringing the bracket to ``2^-60``."""

_SPLIT = 0.1
"""Fraction ``δ`` of an exponent that absorbs the polynomial of a function,
``r^l exp(−δ α r²) ≤ (l / (2 e δ α))^{l/2}``, so that the Gaussian which
sets a reach keeps ``(1 − δ) α``. The split enters a reach logarithmically."""

_TAIL_MARGIN = 2.0
"""Factor covering a lattice sum that exceeds the continuum estimate of its tail."""

_LATTICE_SUM_LOG = 37.0
"""``exp(−α T²)`` is summed over the lattice vectors with ``α T²`` below
this, where the terms are below ``1e-16``."""


def function_radius(shell: Shell, precision: float) -> float:
    """Return the radius beyond which ``shell`` falls below ``precision``.

    A contracted function is bounded by ``Y_l Σ_i |c_i| r^l exp(−α_i r²)``
    with the bare-primitive coefficients ``c_i`` and
    ``Y_l = sqrt((2l + 1) / 4π)``, the largest value a unit-norm real solid
    harmonic takes on the sphere. Each term is driven below
    ``precision / n`` for ``n`` primitives, and the radius of the shell is
    the largest of the primitive radii.
    """
    _check_precision(precision)
    l = shell.angular_momentum
    angular = math.sqrt((2 * l + 1) / (4.0 * math.pi))
    rows = shell.primitive_coefficients()
    radius = 0.0
    for index, alpha in enumerate(shell.exponents):
        amplitude = angular * shell.n_primitives * max(abs(row[index]) for row in rows)
        if amplitude > 0.0:
            radius = max(radius, _primitive_radius(alpha, l, amplitude / precision))
    return radius


def _check_precision(precision: float) -> None:
    if not (math.isfinite(precision) and precision > 0.0):
        raise InputError("the lattice-sum precision is a positive number")


def _primitive_radius(alpha: float, l: int, ratio: float) -> float:
    """Return the largest ``r`` with ``ratio r^l exp(−α r²) = 1``, at least ``1/√α``.

    Beyond the maximum of ``r^l exp(−α r²)`` at ``sqrt(l / 2α)`` the
    function decreases, so the outer root is bracketed by doubling from
    there and found by bisection; the upper end of the bracket is returned.
    The floor ``1/√α`` keeps the radius at the natural scale of the
    primitive when the function never reaches the precision.
    """

    def excess(r: float) -> float:
        return math.log(ratio) + l * math.log(r) - alpha * r * r

    low = max(1.0 / math.sqrt(alpha), math.sqrt(l / (2.0 * alpha)))
    return _outer_root(excess, low)


def _outer_root(excess: Callable[[float], float], low: float) -> float:
    """Return the root of ``excess``, positive at ``low`` and decreasing beyond.

    ``low`` itself when ``excess`` is already non-positive there; otherwise
    the upper end of the bisected bracket found by doubling.
    """
    if excess(low) <= 0.0:
        return low
    high = 2.0 * low
    while excess(high) > 0.0:
        high *= 2.0
    for _ in range(_RADIUS_BISECTIONS):
        middle = 0.5 * (low + high)
        if excess(middle) > 0.0:
            low = middle
        else:
            high = middle
    return high


def function_absolute_integral(shell: Shell) -> float:
    """Return a bound on ``∫ |χ(r)| dr`` over the functions of ``shell``.

    With ``|Y_lm| ≤ Y_l = sqrt((2l + 1) / 4π)`` on the sphere and the radial
    part bounded by ``Σ_i |c_i| r^l exp(−α_i r²)``, the integral is at most
    ``4π Y_l Σ_i |c_i| Γ((l + 3) / 2) / (2 α_i^{(l + 3) / 2})``; the largest
    coefficients over the contractions are taken. It bounds the Fourier
    transform of the function at every reciprocal vector.
    """
    l = shell.angular_momentum
    angular = 4.0 * math.pi * math.sqrt((2 * l + 1) / (4.0 * math.pi))
    rows = shell.primitive_coefficients()
    total = 0.0
    for index, alpha in enumerate(shell.exponents):
        coefficient = max(abs(row[index]) for row in rows)
        total += coefficient * math.gamma((l + 3) / 2) / (2.0 * alpha ** ((l + 3) / 2))
    return angular * total


def atom_radii(basis: BasisSet, precision: float) -> tuple[float, ...]:
    """Return the radius of every atom: the largest radius among its shells."""
    radii = [0.0] * basis.n_atoms
    for shell, atom in zip(basis.shells, basis.shell_atoms, strict=True):
        radii[atom] = max(radii[atom], function_radius(shell, precision))
    return tuple(radii)


def _primitive_terms(shells: Iterable[Shell]) -> tuple[Array, Array]:
    """Return the amplitudes and reduced exponents of the primitives of ``shells``.

    Every function of a shell is bounded by ``Σ_i A_i exp(−α'_i r²)`` with
    the amplitude ``A_i = Y_l |c_i| (l / (2 e δ α_i))^{l/2}`` of the largest
    coefficient over the contractions and the reduced exponent
    ``α'_i = (1 − δ) α_i``, the polynomial of the function absorbed by the
    split ``δ`` of the exponent.
    """
    xp = default_namespace()
    amplitudes: list[float] = []
    exponents: list[float] = []
    for shell in shells:
        l = shell.angular_momentum
        angular = math.sqrt((2 * l + 1) / (4.0 * math.pi))
        polynomial = (l / (2.0 * math.e * _SPLIT)) ** (0.5 * l)
        rows = shell.primitive_coefficients()
        for index, alpha in enumerate(shell.exponents):
            coefficient = max(abs(row[index]) for row in rows)
            amplitudes.append(angular * coefficient * polynomial / alpha ** (0.5 * l))
            exponents.append((1.0 - _SPLIT) * alpha)
    return (
        xp.asarray(amplitudes, dtype=xp.float64),
        xp.asarray(exponents, dtype=xp.float64),
    )


def _atom_types(basis: BasisSet) -> tuple[list[tuple[Shell, ...]], list[int]]:
    """Return the distinct shell lists of the atoms and the type of every atom."""
    shells: list[list[Shell]] = [[] for _ in range(basis.n_atoms)]
    for shell, atom in zip(basis.shells, basis.shell_atoms, strict=True):
        shells[atom].append(shell)
    types: dict[tuple[Shell, ...], int] = {}
    atom_types: list[int] = []
    for atom_shells in shells:
        key = tuple(atom_shells)
        atom_types.append(types.setdefault(key, len(types)))
    return list(types), atom_types


def _reach_matrix(
    first: BasisSet,
    second: BasisSet,
    reach: Callable[[tuple[Shell, ...], tuple[Shell, ...]], float],
) -> Array:
    """Evaluate ``reach`` per pair of atom types over ``[n_first, n_second]``.

    A pair of types is evaluated once in either order, so the matrix of a
    basis with itself is exactly symmetric.
    """
    xp = default_namespace()
    types_a, atoms_a = _atom_types(first)
    types_b, atoms_b = _atom_types(second)
    same = types_a == types_b
    table = [[0.0] * len(types_b) for _ in types_a]
    for ta, a in enumerate(types_a):
        for tb, b in enumerate(types_b):
            table[ta][tb] = table[tb][ta] if same and tb < ta else reach(a, b)
    rows = [[table[ta][tb] for tb in atoms_b] for ta in atoms_a]
    return xp.asarray(rows, dtype=xp.float64)


def _tail_prefactor(
    lattice: Lattice, decay: Array, radius: float, coulomb: bool
) -> Array:
    """Return the tail factor beyond ``radius`` per ``decay`` without its Gaussian.

    The continuum estimate ``(S_d / V_d) ∫_R^∞ t^{d−1} f(t) dt`` of a sum over
    the lattice points farther than ``R`` of ``f(t) = exp(−μ t²)``, or of
    ``exp(−μ t²) / t`` for a Coulomb term, with the surface ``S_d`` of the
    unit sphere and the measure ``V_d`` of the cell in ``d`` dimensions;
    ``∫_R^∞ exp(−μ t²) dt ≤ exp(−μ R²) / (2 μ R)`` bounds the incomplete
    integrals. The margin covers a lattice sum exceeding the estimate.
    """
    xp = array_namespace(decay)
    volume = lattice.volume
    if lattice.dimension == 3:
        factor = (
            2.0 * math.pi / (volume * decay)
            if coulomb
            else 4.0
            * math.pi
            / volume
            * (radius / (2.0 * decay) + 1.0 / (4.0 * decay**2 * radius))
        )
    elif lattice.dimension == 2:
        factor = (
            math.pi / (volume * decay * radius)
            if coulomb
            else math.pi / (volume * decay)
        )
    else:
        factor = (
            1.0 / (volume * decay * radius**2)
            if coulomb
            else 1.0 / (volume * decay * radius)
        )
    return _TAIL_MARGIN * xp.asarray(factor, dtype=xp.float64)


def _solve_reach(
    lattice: Lattice,
    amplitudes: Array,
    decays: Array,
    precision: float,
    coulomb: bool,
) -> float:
    """Return the smallest ``R`` beyond which the terms' tail is below ``precision``.

    The terms ``amplitudes[k] exp(−decays[k] t²)`` (times ``1/t`` for
    Coulomb terms) are summed over the lattice points farther than ``R``
    by the estimate of :func:`_tail_prefactor`; ``R`` is at least the
    natural scale ``1/sqrt(min decay)``, beyond which the tail decreases.
    """
    xp = array_namespace(amplitudes, decays)

    def excess(radius: float) -> float:
        tail = _tail_prefactor(lattice, decays, radius, coulomb) * xp.exp(
            -decays * radius * radius
        )
        return float(xp.sum(amplitudes * tail)) - precision

    return _outer_root(excess, 1.0 / math.sqrt(float(xp.min(decays))))


def pair_reach(
    first: BasisSet,
    second: BasisSet,
    lattice: Lattice,
    precision: float = DEFAULT_LATTICE_SUM_PRECISION,
) -> Array:
    """Return the reach ``[n_first, n_second]`` of every pair of atoms, in Bohr.

    The products of a function ``a`` of an atom of ``first`` and a function
    ``b`` of an atom of ``second`` at the separation ``X`` have
    ``∫ |a b| dr ≤ Σ_ij A_i A_j (π / (α'_i + α'_j))^{3/2} exp(−μ'_ij X²)``
    over the primitives with the amplitudes and reduced exponents of
    :func:`_primitive_terms` and ``μ'_ij = α'_i α'_j / (α'_i + α'_j)``, by
    the Gaussian product rule. The reach of the pair of atoms is the
    distance beyond which these integrals, summed over every image of the
    second atom farther than it, fall below ``precision``: any matrix
    element of a bounded one-electron operator between the two atoms then
    changes by less than the precision when those images are left out.
    """
    _check_precision(precision)
    xp = default_namespace()

    def reach(shells_a: tuple[Shell, ...], shells_b: tuple[Shell, ...]) -> float:
        amplitudes_a, exponents_a = _primitive_terms(shells_a)
        amplitudes_b, exponents_b = _primitive_terms(shells_b)
        total = xp.expand_dims(exponents_a, 1) + xp.expand_dims(exponents_b, 0)
        reduced = (
            xp.expand_dims(exponents_a, 1) * xp.expand_dims(exponents_b, 0) / total
        )
        amplitudes = (
            xp.expand_dims(amplitudes_a, 1)
            * xp.expand_dims(amplitudes_b, 0)
            * (math.pi / total) ** 1.5
        )
        return _solve_reach(
            lattice,
            xp.reshape(amplitudes, (-1,)),
            xp.reshape(reduced, (-1,)),
            precision,
            False,
        )

    return _reach_matrix(first, second, reach)


def coulomb_reach(
    first: BasisSet,
    second: BasisSet,
    lattice: Lattice,
    *,
    omega: float,
    precision: float = DEFAULT_LATTICE_SUM_PRECISION,
    spectator: float = 1.0,
) -> Array:
    """Return the reach ``[n_first, n_second]`` of the short-range kernel between atoms.

    Two functions ``a`` and ``b`` of atoms at the distance ``D`` interact
    through ``erfc(ωr)/r`` by at most
    ``Σ_ik A_i A_k π³ exp(−θ_ik D²) / (α'_i β'_k sqrt(α'_i + β'_k) sqrt(ρ_ik) D)``
    over the primitives with the amplitudes and reduced exponents of
    :func:`_primitive_terms`, ``ρ_ik = α'_i β'_k / (α'_i + β'_k)`` and
    ``1/θ_ik = 1/ρ_ik + 1/ω²``: the short-range integral of two s Gaussians
    is ``π³ [erf(sqrt(ρ) D) − erf(sqrt(θ) D)] / (αβ sqrt(α + β) sqrt(ρ) D)``
    and ``erf(x) − erf(y) ≤ exp(−y²)``. The reach of a pair of atoms is the
    distance beyond which these bounds, summed over every image of the
    second atom farther than it, fall below ``precision``. A ``spectator``
    multiplies the bounds: for the three-centre integrals ``(P | a b)`` it
    is a bound on ``Σ_T |b(r − T)|`` over the images of the third function,
    :func:`periodic_amplitude`.
    """
    _check_precision(precision)
    if not (math.isfinite(omega) and omega > 0.0):
        raise InputError("the range-separation parameter omega is positive")
    if not (math.isfinite(spectator) and spectator > 0.0):
        raise InputError("the spectator amplitude is a positive number")
    xp = default_namespace()

    def reach(shells_a: tuple[Shell, ...], shells_b: tuple[Shell, ...]) -> float:
        amplitudes_a, exponents_a = _primitive_terms(shells_a)
        amplitudes_b, exponents_b = _primitive_terms(shells_b)
        alpha = xp.expand_dims(exponents_a, 1)
        beta = xp.expand_dims(exponents_b, 0)
        rho = alpha * beta / (alpha + beta)
        theta = rho * omega**2 / (rho + omega**2)
        amplitudes = (
            spectator
            * xp.expand_dims(amplitudes_a, 1)
            * xp.expand_dims(amplitudes_b, 0)
            * math.pi**3
            / (alpha * beta * xp.sqrt(alpha + beta) * xp.sqrt(rho))
        )
        return _solve_reach(
            lattice,
            xp.reshape(amplitudes, (-1,)),
            xp.reshape(theta, (-1,)),
            precision,
            True,
        )

    return _reach_matrix(first, second, reach)


def periodic_amplitude(basis: BasisSet, lattice: Lattice) -> float:
    """Return a bound on ``max_r Σ_T |χ(r − T)|`` over the functions of ``basis``.

    Every function is bounded by ``Σ_i A_i exp(−α'_i r²)`` as in
    :func:`_primitive_terms`, and the lattice sum of a Gaussian takes its
    largest value on a lattice point: by the Poisson summation formula
    ``Σ_T exp(−α |r − T|²) = (π/α)^{d/2} Σ_G exp(−G²/4α) exp(i G·r) / V_d``
    over the ``d`` periodic directions, which is at most its value at
    ``r = 0``, ``Σ_T exp(−α T²)``, summed where its terms are above ``1e-16``.
    """
    xp = default_namespace()
    largest = 0.0
    for shell in basis.shells:
        amplitudes, exponents = _primitive_terms([shell])
        total = 0.0
        for amplitude, alpha in zip(
            amplitudes.tolist(), exponents.tolist(), strict=True
        ):
            vectors = lattice.translations(math.sqrt(_LATTICE_SUM_LOG / alpha))
            total += amplitude * float(
                xp.sum(xp.exp(-alpha * xp.sum(vectors * vectors, axis=-1)))
            )
        largest = max(largest, total)
    return largest


def lattice_images(
    basis: BasisSet, cell: Cell, *, precision: float = DEFAULT_LATTICE_SUM_PRECISION
) -> Array:
    """Return the translations ``[image, 3]`` in Bohr over which lattice sums run.

    An image ``T`` is kept when some atom ``A`` of the cell and some atom
    ``B`` of the image are within their reach, ``|A − B − T| ≤ R_AB``, the
    distance beyond which the products of their functions over all farther
    images are below ``precision`` (:func:`pair_reach`). The zero
    translation comes first, the rest follow by length, and ``−T`` is kept
    with ``T``, so Bloch sums over the result are Hermitian.
    """
    require_basis_for_atoms(basis, cell)
    positions = cell.positions_array()
    reach = pair_reach(basis, basis, cell.lattice, precision)
    return reaching_images(cell.lattice, positions, positions, reach)


def reaching_images(
    lattice: Lattice, positions: Array, targets: Array, reach: Array
) -> Array:
    """Return the translations ``T`` ``[image, 3]`` at which the sets touch.

    ``T`` is kept when some position ``p_i`` and some target ``q_j`` satisfy
    ``|p_i − q_j − T| ≤ reach[i, j]``, both sets given as ``[n, 3]`` in
    Bohr with the reach ``[n, m]`` of every pair. The zero translation
    comes first when it is kept, the rest follow by length, and ``−T`` is
    kept with ``T`` whenever the two sets coincide and the reach is
    symmetric.
    """
    xp = array_namespace(positions, targets, reach)
    if positions.ndim != 2 or targets.ndim != 2 or reach.ndim != 2:
        raise InputError("positions have shape [n, 3] with one reach per pair")
    n, m = int(positions.shape[0]), int(targets.shape[0])
    if positions.shape != (n, 3) or targets.shape != (m, 3) or reach.shape != (n, m):
        raise InputError("positions have shape [n, 3] with one reach per pair")
    if n == 0 or m == 0:
        return xp.zeros((0, 3), dtype=xp.float64)
    separations = xp.expand_dims(positions, 1) - xp.expand_dims(targets, 0)
    extent = float(xp.max(xp.linalg.vector_norm(separations, axis=-1)))
    candidates = lattice.translations(float(xp.max(reach)) + extent)
    n_candidates = int(candidates.shape[0])
    # The distances of every candidate to every target are taken in chunks
    # of candidates, so that the memory stays bounded for many targets (the
    # points of a grid) and many candidates (a small cell).
    chunk = max(1, _REACH_ELEMENTS // (3 * m))
    kept: list[Array] = []
    for start in range(0, n_candidates, chunk):
        part = candidates[start : start + chunk]
        keep = xp.zeros(part.shape[0], dtype=xp.bool)
        for index in range(n):
            distance = xp.linalg.vector_norm(
                xp.expand_dims(separations[index], 0) - xp.expand_dims(part, 1),
                axis=-1,
            )
            keep = keep | xp.any(distance <= xp.expand_dims(reach[index], 0), axis=1)
        kept.append(part[keep])
    return xp.concat(kept, axis=0)


def integer_coordinates(lattice: Lattice, translations: Array) -> Array:
    """Return the integer lattice coordinates ``[image, 3]`` of ``translations``.

    The translations ``[image, 3]`` in Bohr are lattice vectors; their
    fractional coordinates are rounded to the nearest integers, which label
    the images for the pair lookup of the compiled layer (ADR-0019). The
    open directions of a slab or a chain carry zeros.
    """
    xp = array_namespace(translations)
    fractional = lattice.fractional_coordinates(translations)
    coordinates = xp.astype(xp.floor(fractional + 0.5), xp.int64)
    padding = xp.zeros((coordinates.shape[0], 3 - lattice.dimension), dtype=xp.int64)
    return xp.concat([coordinates, padding], axis=1)


def require_kpoints_for_lattice(kpoints: KPoints, lattice: Lattice) -> None:
    """Raise unless ``kpoints`` has one coordinate per direction of ``lattice``.

    The check that :meth:`KPoints.cartesian` makes, taken before a lattice
    sum rather than after it.
    """
    if kpoints.dimension != lattice.dimension:
        raise InputError(
            f"{kpoints.dimension}-dimensional k-points do not fit a "
            f"{lattice.dimension}-dimensional lattice"
        )


def bloch_sum(
    images: Array, translations: Array, kpoints: KPoints, lattice: Lattice
) -> Array:
    """Return ``Σ_T exp(i k·T) images[T]`` at every k-point as complex ``[kpt, ...]``.

    ``images`` holds one array per translation along its first axis, as the
    engine returns them, and ``translations`` is ``[image, 3]`` in Bohr.
    """
    xp = array_namespace(images, translations)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise InputError("translations have shape [image, 3]")
    if images.shape[:1] != translations.shape[:1]:
        raise InputError("a Bloch sum takes one image per translation")
    kpts = xp.asarray(kpoints.cartesian(lattice))
    angles = kpts @ xp.linalg.matrix_transpose(translations)
    phases = xp.exp(1j * xp.astype(angles, xp.complex128))
    return xp.tensordot(phases, xp.astype(images, xp.complex128), axes=1)


def overlap_k(
    engine: IntegralEngine,
    basis: BasisSet,
    cell: Cell,
    kpoints: KPoints,
    *,
    precision: float = DEFAULT_LATTICE_SUM_PRECISION,
) -> Array:
    """Overlap matrices ``S(k)`` as complex ``[kpt, ao, ao]`` over ``kpoints``.

    Code that needs several matrices of one cell calls
    :func:`lattice_images` once and :func:`bloch_sum` on each engine
    result, so that one image list serves them all.
    """
    return _lattice_sum(engine.overlap, basis, cell, kpoints, precision)


def kinetic_k(
    engine: IntegralEngine,
    basis: BasisSet,
    cell: Cell,
    kpoints: KPoints,
    *,
    precision: float = DEFAULT_LATTICE_SUM_PRECISION,
) -> Array:
    """Kinetic matrices ``T(k)`` as complex ``[kpt, ao, ao]`` over ``kpoints``.

    See :func:`overlap_k` for sharing the image list between matrices.
    """
    return _lattice_sum(engine.kinetic, basis, cell, kpoints, precision)


def _lattice_sum(
    kernel: Callable[[ShellTable, ShellTable, Array], Array],
    basis: BasisSet,
    cell: Cell,
    kpoints: KPoints,
    precision: float,
) -> Array:
    """Bloch sums of a one-electron ``kernel`` over the images of ``cell``."""
    require_kpoints_for_lattice(kpoints, cell.lattice)
    table = ShellTable.from_basis(basis, cell)
    translations = lattice_images(basis, cell, precision=precision)
    return bloch_sum(
        kernel(table, table, translations), translations, kpoints, cell.lattice
    )

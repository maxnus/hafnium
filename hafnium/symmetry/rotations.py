"""Rotation matrices of real solid harmonics and of a basis set.

A point operation ``R`` of a crystal is an orthogonal matrix in a general
frame, not a sign operation as in the standard frame of ADR-0022, so it
mixes the functions of a shell: ``S_lm(R⁻¹ r) = Σ_m' D^l[m', m] S_lm'(r)``
with the Wigner matrix ``D^l(R)`` of the real solid harmonics. The matrix
is obtained exactly from the polynomial form of the harmonics
(:mod:`hafnium.basis.harmonics`): the monomials of degree ``l`` transform
among themselves under ``R⁻¹``, and the harmonics span an invariant
subspace of them. The same convention holds for improper operations, so
an inversion gives ``(−1)^l`` without a special case.
"""

from __future__ import annotations

from collections.abc import Sequence

from hafnium.backend import Array, array_namespace, host_namespace
from hafnium.basis import BasisSet, cartesian_components, solid_harmonic_coefficients
from hafnium.errors import InputError, SymmetryError

ORTHOGONALITY_TOLERANCE = 1e-8
"""Largest deviation of a point operation from an orthogonal matrix."""

Exponents = tuple[int, int, int]


def solid_harmonic_rotation(angular_momentum: int, matrix: Array) -> Array:
    """Return the Wigner matrix ``D^l(R)`` of the real solid harmonics.

    ``D`` is ``[2l + 1, 2l + 1]`` with ``S_lm(R⁻¹ r) = Σ_m' D[m', m] S_lm'(r)``.
    Rows and columns follow the storage order of a shell (ADR-0009), so
    for ``l = 1`` the matrix is ``R`` itself, the p functions rotating like
    ``x, y, z``. ``D`` is orthogonal, and ``D(R₁ R₂) = D(R₁) D(R₂)``.

    Raises
    ------
    InputError
        If ``matrix`` is not a ``[3, 3]`` orthogonal matrix.
    """
    xp = array_namespace(matrix)
    rotation = xp.asarray(matrix, dtype=xp.float64)
    if rotation.shape != (3, 3) or (
        float(
            xp.max(xp.abs(xp.linalg.matrix_transpose(rotation) @ rotation - xp.eye(3)))
        )
        > ORTHOGONALITY_TOLERANCE
    ):
        raise InputError("a point operation is a [3, 3] orthogonal matrix")
    table = xp.asarray(solid_harmonic_coefficients(angular_momentum))
    monomials = _monomial_rotation(angular_momentum, rotation)
    # S(R⁻¹ r) = T M m(r) and Σ_m' D[m', m] S_m'(r) = (Dᵀ T) m(r), so
    # T M = Dᵀ T and D = (T Tᵀ)⁻¹ T Mᵀ Tᵀ.
    transposed = xp.linalg.matrix_transpose(table)
    gram = table @ transposed
    return xp.linalg.solve(
        gram, table @ xp.linalg.matrix_transpose(monomials) @ transposed
    )


def _monomial_rotation(angular_momentum: int, rotation: Array) -> Array:
    """``M[c, c']`` with ``m_c(R⁻¹ r) = Σ_c' M[c, c'] m_c'(r)`` over the monomials."""
    xp = array_namespace(rotation)
    components = cartesian_components(angular_momentum)
    index = {exponents: position for position, exponents in enumerate(components)}
    # (R⁻¹ r)_a = Σ_b R[b, a] r_b: the a-th linear form has the coefficients
    # of column a of R.
    forms = [[float(rotation[b, a]) for b in range(3)] for a in range(3)]
    rows: list[list[float]] = []
    for exponents in components:
        polynomial: dict[Exponents, float] = {(0, 0, 0): 1.0}
        for axis, power in enumerate(exponents):
            for _ in range(power):
                polynomial = _times_linear_form(polynomial, forms[axis])
        row = [0.0] * len(components)
        for monomial, coefficient in polynomial.items():
            row[index[monomial]] = coefficient
        rows.append(row)
    return xp.asarray(rows, dtype=xp.float64)


def _times_linear_form(
    polynomial: dict[Exponents, float], form: list[float]
) -> dict[Exponents, float]:
    """Multiply a polynomial by ``form[0] x + form[1] y + form[2] z``."""
    product: dict[Exponents, float] = {}
    for (i, j, k), coefficient in polynomial.items():
        for axis, weight in enumerate(form):
            if weight == 0.0:
                continue
            exponents = (i + (axis == 0), j + (axis == 1), k + (axis == 2))
            product[exponents] = product.get(exponents, 0.0) + coefficient * weight
    return product


def basis_rotation(basis: BasisSet, matrix: Array, atom_images: Sequence[int]) -> Array:
    """Return ``Π [ao, ao]``: the action of a point operation on the basis functions.

    The function ``μ`` of a shell on atom ``A`` is mapped to the functions
    of the same shell on the image atom ``atom_images[A]``,
    ``Π[ν, μ] = D^l(R)[m', m]``, so that ``Π`` is orthogonal and an
    operator commuting with the operation satisfies ``Π M Πᵀ = M`` for a
    molecule (the Bloch phases of a crystal are applied by the caller).
    Atoms related by the operation must carry equal shells. ``Π`` is the
    block-diagonal matrix of the Wigner matrices of every shell with its
    rows moved to the functions of the image atoms.

    Raises
    ------
    SymmetryError
        If an atom and its image carry different shells.
    """
    xp = host_namespace()
    n = basis.n_functions
    offsets = basis.shell_offsets()
    shells_of = [
        [index for index, centre in enumerate(basis.shell_atoms) if centre == atom]
        for atom in range(basis.n_atoms)
    ]
    rotations = {
        l: solid_harmonic_rotation(l, matrix)
        for l in {shell.angular_momentum for shell in basis.shells}
    }
    targets = [0] * basis.n_shells
    for atom, image in enumerate(atom_images):
        own, other = shells_of[atom], shells_of[image]
        if [basis.shells[s] for s in own] != [basis.shells[s] for s in other]:
            raise SymmetryError(
                f"atoms {atom} and {image} are related by the symmetry but carry "
                "different basis functions"
            )
        for source, target in zip(own, other, strict=True):
            targets[source] = target
    # Rows of the block-diagonal matrix in the order of the functions, and
    # the function each row lands on.
    rows: list[Array] = []
    landing: list[int] = []
    for index, shell in enumerate(basis.shells):
        width = 2 * shell.angular_momentum + 1
        block = rotations[shell.angular_momentum]
        for contraction in range(shell.n_contractions):
            start = offsets[index] + contraction * width
            rows.append(
                xp.concat(
                    [
                        xp.zeros((width, start), dtype=xp.float64),
                        block,
                        xp.zeros((width, n - start - width), dtype=xp.float64),
                    ],
                    axis=1,
                )
            )
            target = offsets[targets[index]] + contraction * width
            landing.extend(range(target, target + width))
    stacked = xp.concat(rows, axis=0)
    order = sorted(range(n), key=lambda position: landing[position])
    return xp.take(stacked, xp.asarray(order, dtype=xp.int64), axis=0)

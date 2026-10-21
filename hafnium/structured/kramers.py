"""Time reversal on the spinor representation.

The time-reversal operator is Θ = −iσ_y K with K complex conjugation. On a
spinor ``(v_alpha, v_beta)`` it acts as ``(-conj(v_beta), conj(v_alpha))``,
and ``Θ² = −1``. Operators that commute with Θ have the quaternionic form
``[[A, B], [-conj(B), conj(A)]]`` and doubly degenerate (Kramers) spectra.
"""

from __future__ import annotations

from hafnium.backend import Array, adjoint, array_namespace
from hafnium.errors import StructureError


def time_reverse(x: Array) -> Array:
    """Apply Θ to each column of a ``(2n, k)`` spinor array."""
    xp = array_namespace(x)
    n = _half(x.shape[0])
    return xp.concat([-xp.conj(x[n:, ...]), xp.conj(x[:n, ...])], axis=0)


def time_reverse_operator(x: Array) -> Array:
    """Return ``Θ X Θ⁻¹`` for a square ``(2n, 2n)`` operator."""
    xp = array_namespace(x)
    n = _half(x.shape[0])
    if x.shape[1] != 2 * n:
        raise StructureError("time reversal of an operator needs a square matrix")
    a, b = x[:n, :n], x[:n, n:]
    c, d = x[n:, :n], x[n:, n:]
    top = xp.concat([xp.conj(d), -xp.conj(c)], axis=1)
    bottom = xp.concat([-xp.conj(b), xp.conj(a)], axis=1)
    return xp.concat([top, bottom], axis=0)


def kramers_symmetrize(x: Array) -> Array:
    """Project a square operator onto the Θ-symmetric (quaternionic) subspace."""
    return (x + time_reverse_operator(x)) / 2


def is_kramers_symmetric(x: Array, *, tolerance: float = 1e-10) -> bool:
    """Whether a square operator commutes with Θ to within ``tolerance``."""
    xp = array_namespace(x)
    scale = max(1.0, float(xp.max(xp.abs(x))))
    deviation = float(xp.max(xp.abs(x - time_reverse_operator(x))))
    return deviation <= tolerance * scale


def kramers_adapt_eigenvectors(
    eigenvalues: Array,
    eigenvectors: Array,
    *,
    tolerance: float = 1e-8,
) -> Array:
    """Rotate eigenvectors of a Θ-symmetric operator into Kramers pairs.

    Within each degenerate eigenspace the returned columns come in adjacent
    pairs ``(u, Θu)``. Eigenvalues closer than ``tolerance`` (relative to the
    spectral range) are treated as degenerate.

    This function uses data-dependent control flow and cannot be traced by
    JAX; the JAX backend milestone will provide a traceable variant.

    Raises
    ------
    StructureError
        If a degenerate subspace has odd dimension, which means the operator
        was not Θ-symmetric.
    """
    xp = array_namespace(eigenvalues, eigenvectors)
    count = eigenvectors.shape[1]
    scale = max(1.0, float(xp.max(xp.abs(eigenvalues))))
    columns: list[Array] = []
    start = 0
    while start < count:
        end = start + 1
        while (
            end < count
            and float(xp.abs(eigenvalues[end] - eigenvalues[end - 1]))
            <= tolerance * scale
        ):
            end += 1
        if (end - start) % 2:
            raise StructureError(
                "Kramers degeneracy broken: the operator is not Θ-symmetric"
            )
        columns.extend(_pair_cluster(eigenvectors[:, start:end], tolerance))
        start = end
    return xp.concat(columns, axis=1)


def _pair_cluster(basis: Array, tolerance: float) -> list[Array]:
    """Split an orthonormal basis of a Θ-invariant subspace into Kramers pairs."""
    xp = array_namespace(basis)
    pairs: list[Array] = []
    remaining = basis
    while remaining.shape[1] > 0:
        u = remaining[:, :1]
        u = u / xp.linalg.vector_norm(u)
        pair = xp.concat([u, time_reverse(u)], axis=1)
        pairs.append(pair)
        rest = remaining[:, 1:]
        if rest.shape[1] == 0:
            break
        rest = rest - pair @ (adjoint(pair) @ rest)
        left, singular, _ = xp.linalg.svd(rest, full_matrices=False)
        remaining = left[:, singular > tolerance]
    return pairs


def _half(dimension: int) -> int:
    if dimension % 2:
        raise StructureError("spinor arrays have an even number of rows")
    return dimension // 2

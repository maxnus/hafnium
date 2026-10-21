"""Linear algebra helpers beyond the array API standard."""

from collections.abc import Callable

from hafnium.backend.namespace import array_namespace
from hafnium.backend.types import Array


def adjoint(x: Array) -> Array:
    """Return the conjugate transpose of the last two axes of ``x``."""
    xp = array_namespace(x)
    return xp.conj(xp.linalg.matrix_transpose(x))


def eigh(a: Array) -> tuple[Array, Array]:
    """Diagonalize a Hermitian matrix.

    Returns
    -------
    eigenvalues : Array[p]
        In ascending order.
    eigenvectors : Array[p, p]
        Orthonormal, one eigenvector per column.
    """
    xp = array_namespace(a)
    result = xp.linalg.eigh(a)
    return result.eigenvalues, result.eigenvectors


def generalized_eigh(
    a: Array,
    b: Array,
    *,
    adapt: Callable[[Array, Array], Array] | None = None,
) -> tuple[Array, Array]:
    """Solve ``a @ v = w * (b @ v)`` for Hermitian ``a`` and positive-definite ``b``.

    The problem is reduced to a standard one with the Cholesky factor
    ``b = L @ adjoint(L)``. The optional ``adapt`` hook receives the
    eigenvalues and the eigenvectors of the reduced problem, which are
    orthonormal in the Euclidean sense, and returns a rotated set of
    eigenvectors; it is used to impose Kramers pairing within degenerate
    subspaces before transforming back.

    Returns
    -------
    eigenvalues : Array[p]
        In ascending order.
    eigenvectors : Array[p, p]
        Normalized so that ``adjoint(v) @ b @ v`` is the identity.
    """
    xp = array_namespace(a, b)
    lower = xp.linalg.cholesky(b)
    half = xp.linalg.solve(lower, a)
    reduced = xp.linalg.solve(lower, adjoint(half))
    reduced = (reduced + adjoint(reduced)) / 2
    eigenvalues, reduced_vectors = eigh(reduced)
    if adapt is not None:
        reduced_vectors = adapt(eigenvalues, reduced_vectors)
    eigenvectors = xp.linalg.solve(adjoint(lower), reduced_vectors)
    return eigenvalues, eigenvectors

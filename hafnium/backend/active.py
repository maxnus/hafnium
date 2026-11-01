"""The active backend: the array library that holds the arrays of a calculation.

Method code never names an array library. Arrays derived from inputs take
the namespace of those inputs (:func:`array_namespace`); arrays that start a
calculation, the integrals the compiled kernels return and the matrices a
structure is filled with, are created in the *active backend*, which the
caller selects with :func:`use_backend` or :func:`set_backend` (ADR-0036).
The compiled kernels and the geometric setup around them, shell tables,
grids, lattices and symmetry detection, work on host arrays whatever the
active backend, through :func:`host_namespace`, and their results cross
into the active backend with :func:`from_host`.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass

import array_api_compat
import numpy

from hafnium.backend.types import Array, Namespace
from hafnium.errors import BackendError

HOST_NAMESPACE: Namespace = array_api_compat.array_namespace(numpy.empty(0))
"""The namespace of host arrays, what :func:`array_namespace` returns for NumPy."""


@dataclass(frozen=True, slots=True, eq=False)
class Backend:
    """An array library that holds the arrays of a calculation.

    Parameters
    ----------
    name
        ``"numpy"``, ``"jax"`` or ``"cupy"``.
    namespace
        The array API namespace of the library's arrays, the object
        :func:`array_namespace` returns for them.
    """

    name: str
    namespace: Namespace

    @classmethod
    def named(cls, name: str) -> Backend:
        """Return the backend of the array library called ``name``.

        The library is imported, and configured where it needs to be, the
        first time it is asked for: JAX is switched to 64-bit arrays and the
        structured types are registered as pytrees.

        Raises
        ------
        BackendError
            For an unknown name, or when the library is not installed.
        """
        if name not in _LOADERS:
            raise BackendError(
                f"unknown backend {name!r}; the backends are {sorted(_LOADERS)}"
            )
        if name not in _LOADED:
            _LOADED[name] = _LOADERS[name]()
        return _LOADED[name]

    def from_host(self, array: Array) -> Array:
        """Return a host array as an array of this backend."""
        return self.namespace.asarray(array)

    def __repr__(self) -> str:
        return f"Backend({self.name!r})"


NUMPY = Backend("numpy", HOST_NAMESPACE)
"""The host backend, active unless another is selected."""


def _load_jax() -> Backend:
    from hafnium.backend import _jax

    return _jax.load()


def _load_cupy() -> Backend:
    try:
        import cupy  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise BackendError(
            "the cupy backend needs CuPy installed (pip install hafnium[cuda])"
        ) from error
    namespace = array_api_compat.array_namespace(cupy.empty(0))  # pyright: ignore[reportUnknownMemberType]
    return Backend("cupy", namespace)


_LOADERS: dict[str, Callable[[], Backend]] = {
    "numpy": lambda: NUMPY,
    "jax": _load_jax,
    "cupy": _load_cupy,
}
_LOADED: dict[str, Backend] = {"numpy": NUMPY}

_ACTIVE: contextvars.ContextVar[Backend] = contextvars.ContextVar(
    "hafnium_backend", default=NUMPY
)
"""The selection lives in a context variable, not in module state
(CONVENTIONS.md §4): a ``with`` block scopes it, and a thread or task
started inside the block inherits it as the context rules say."""


def _resolve(backend: Backend | str) -> Backend:
    return backend if isinstance(backend, Backend) else Backend.named(backend)


def active_backend() -> Backend:
    """Return the backend new arrays are created in.

    The backend of the innermost :func:`use_backend` block, or the one
    :func:`set_backend` selected for the current context, NumPy to begin
    with.
    """
    return _ACTIVE.get()


def set_backend(backend: Backend | str) -> Backend:
    """Select ``backend`` for the rest of the current context, at the entry point.

    Returns the backend that was active, so a caller can restore it.
    """
    previous = _ACTIVE.get()
    _ACTIVE.set(_resolve(backend))
    return previous


@contextmanager
def use_backend(backend: Backend | str) -> Generator[Backend, None, None]:
    """Make ``backend`` active inside a ``with`` block.

    >>> with use_backend("jax") as backend:
    ...     result = hartree_fock(...)  # doctest: +SKIP

    The arrays of a result built in a block keep belonging to its backend
    after the block ends; operations derived from them stay there too.
    """
    token = _ACTIVE.set(_resolve(backend))
    try:
        yield _ACTIVE.get()
    finally:
        _ACTIVE.reset(token)


def host_namespace() -> Namespace:
    """Return the namespace of host arrays, for the setup around the kernels."""
    return HOST_NAMESPACE


def is_jax_array(array: object) -> bool:
    """Whether ``array`` is a JAX array or tracer, which takes the JAX rules."""
    return array_api_compat.is_jax_array(array)


def from_host(array: Array) -> Array:
    """Return a host array as an array of the active backend."""
    return active_backend().from_host(array)

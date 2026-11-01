"""Discovery of the array namespace behind a set of arrays."""

import array_api_compat

from hafnium.backend.active import active_backend
from hafnium.backend.types import Array, Namespace


def array_namespace(*arrays: Array) -> Namespace:
    """Return the array API namespace shared by ``arrays``.

    Raises
    ------
    TypeError
        If the arrays belong to different backends.
    """
    return array_api_compat.array_namespace(*arrays)


def default_namespace() -> Namespace:
    """Return the namespace of the active backend, for arrays not derived from inputs.

    The same object :func:`array_namespace` returns for that backend's
    arrays; see :func:`hafnium.backend.use_backend`.
    """
    return active_backend().namespace

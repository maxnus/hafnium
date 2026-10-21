"""Discovery of the array namespace behind a set of arrays."""

import array_api_compat
import numpy

from hafnium.backend.types import Array, Namespace


def array_namespace(*arrays: Array) -> Namespace:
    """Return the array API namespace shared by ``arrays``.

    Raises
    ------
    TypeError
        If the arrays belong to different backends.
    """
    return array_api_compat.array_namespace(*arrays)


_NUMPY_NAMESPACE: Namespace = array_api_compat.array_namespace(numpy.empty(0))


def default_namespace() -> Namespace:
    """Return the namespace used for arrays that are not derived from inputs.

    This is the same object that :func:`array_namespace` returns for NumPy
    arrays. A configurable default is part of the backend milestone.
    """
    return _NUMPY_NAMESPACE

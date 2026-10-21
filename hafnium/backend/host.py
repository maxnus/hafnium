"""Host (CPU, NumPy) arrays at the boundary to the compiled layer.

The compiled kernels read and write contiguous NumPy arrays. These helpers
are the only place outside tests where method code obtains such arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hafnium.backend.types import Array
from hafnium.errors import BackendError


def host_array(values: Any, dtype: str = "float64") -> Array:
    """Return a contiguous host array of ``dtype`` built from ``values``.

    Nested sequences and host arrays are accepted. The result is the input
    itself when that already is a contiguous host array of ``dtype``, so
    callers must not mutate it.

    Raises
    ------
    BackendError
        If ``values`` live on a device the host cannot read implicitly
        (for example a CuPy array); the caller moves them to the host first.
    """
    try:
        return np.ascontiguousarray(np.asarray(values, dtype=dtype))
    except TypeError as error:
        raise BackendError(
            f"cannot build a host array from {type(values).__name__}: the compiled "
            "kernels read host (CPU) arrays, so move device arrays to the host first"
        ) from error


def to_host(array: Array, dtype: str = "float64") -> Array:
    """Return ``array`` as a contiguous host array of ``dtype``.

    See :func:`host_array` for the aliasing and device rules.
    """
    return host_array(array, dtype)

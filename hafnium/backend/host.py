"""Host (CPU, NumPy) arrays at the boundary to the compiled layer.

The compiled kernels read and write contiguous NumPy arrays. These helpers
are the only place outside tests where method code obtains such arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hafnium.backend.types import Array


def host_array(values: Any, dtype: str = "float64") -> Array:
    """Build a contiguous host array of ``dtype`` from nested sequences or arrays."""
    return np.ascontiguousarray(np.asarray(values, dtype=dtype))


def to_host(array: Array, dtype: str = "float64") -> Array:
    """Copy an array of the active backend to a contiguous host array."""
    return host_array(array, dtype)

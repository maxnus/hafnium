"""Type stubs for the compiled kernel module built from crates/hafnium-py."""

import numpy as np
from numpy.typing import NDArray

def kernel_version() -> str:
    """Return the version of the compiled kernel crate."""

def dot(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Return the dot product of two contiguous one-dimensional float64 arrays.

    Bootstrap smoke test of the array boundary between Python and Rust.
    Raises ``ValueError`` if the lengths differ and ``TypeError`` if an array
    is not contiguous.
    """

"""Helpers over density matrices shared by the tests."""

import numpy as np

from hafnium.structured import BlockMatrix, natural_orbitals


def natural_occupations(density: BlockMatrix, overlap: BlockMatrix) -> np.ndarray:
    """The natural occupations of ``density``, descending, over all spin orbitals."""
    occupations, _ = natural_orbitals(density, overlap)
    return np.sort(np.asarray(occupations.to_dense()).ravel())[::-1]


def electron_count(density: BlockMatrix, overlap: BlockMatrix) -> float:
    """The real trace of ``density`` with the overlap, the electrons per cell."""
    return float(np.real(np.asarray(density.trace_product(overlap))))

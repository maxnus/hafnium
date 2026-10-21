"""Random matrices with a given symmetry, shared by the test suites."""

import numpy as np


def random_hermitian(rng: np.random.Generator, size: int) -> np.ndarray:
    """A complex Hermitian matrix with standard-normal entries."""
    x = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    return (x + x.conj().T) / 2


def random_positive_definite(rng: np.random.Generator, size: int) -> np.ndarray:
    """A real symmetric matrix with eigenvalues of at least ``size``."""
    x = rng.standard_normal((size, size))
    return x @ x.T + size * np.eye(size)


def random_antisymmetric(rng: np.random.Generator, size: int) -> np.ndarray:
    """A real antisymmetric matrix, so that ``i`` times it is Hermitian."""
    x = rng.standard_normal((size, size))
    return x - x.T

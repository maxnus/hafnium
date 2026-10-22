"""Spectra shared by the test suites."""

import numpy as np


def generalized_eigenvalues(a: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Eigenvalues of ``a c = ε s c`` for Hermitian ``a`` and positive ``s``."""
    inverse = np.linalg.inv(np.linalg.cholesky(s))
    return np.linalg.eigvalsh(inverse @ a @ inverse.conj().T)

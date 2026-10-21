"""Smoke tests for the compiled kernel module."""

import numpy as np
import pytest

from hafnium import _kernels, kernel_version
from tests.tolerances import KERNEL_REL_TOL


def test_kernel_version_is_a_version_string() -> None:
    parts = kernel_version().split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_dot_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1000)
    b = rng.standard_normal(1000)
    assert _kernels.dot(a, b) == pytest.approx(float(a @ b), rel=KERNEL_REL_TOL)


def test_dot_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        _kernels.dot(np.ones(3), np.ones(4))


def test_dot_rejects_non_contiguous_input() -> None:
    strided = np.ones(8)[::2]
    with pytest.raises(TypeError):
        _kernels.dot(strided, np.ones(4))

"""The Boys function against an arbitrary-precision reference."""

import mpmath
import numpy as np
import pytest

from hafnium import _kernels
from tests.tolerances import BOYS_REL_TOL

ORDERS = [0, 1, 2, 5, 10, 16, 20]
ARGUMENTS = [0.0, 1e-9, 0.05, 0.7, 3.0, 12.5, 34.999, 35.0, 36.0, 60.0, 250.0, 1e4]


def reference(n: int, t: float) -> float:
    """``F_n(t)`` from the lower incomplete gamma function at 30 digits.

    ``F_n(t) = gamma(n + 1/2, t) / (2 t^(n + 1/2))``, and ``1 / (2n + 1)`` at zero.
    """
    mpmath.mp.dps = 30
    if t == 0.0:
        return 1.0 / (2 * n + 1)
    half = mpmath.mpf(n) + mpmath.mpf(1) / 2
    value = mpmath.gammainc(half, 0, t) / (2 * mpmath.mpf(t) ** half)
    return float(value)


@pytest.mark.parametrize("t", ARGUMENTS)
def test_boys_function_matches_quadrature(t: float) -> None:
    values = _kernels.boys_function(max(ORDERS), t)
    for n in ORDERS:
        expected = reference(n, t)
        assert values[n] == pytest.approx(expected, rel=BOYS_REL_TOL, abs=0.0)


def test_boys_function_returns_every_order() -> None:
    values = _kernels.boys_function(4, 0.0)
    assert values.shape == (5,)
    np.testing.assert_allclose(values, 1.0 / (2 * np.arange(5) + 1), rtol=1e-15)


def test_boys_function_rejects_negative_argument() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _kernels.boys_function(2, -0.5)

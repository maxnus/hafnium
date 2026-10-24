"""Radial quadratures for atom-centred grids.

Each scheme returns ``n`` radii ``r_i`` in Bohr in ascending order and the
weights ``w_i`` of ``∫_0^∞ f(r) dr ≈ Σ_i w_i f(r_i)``; the volume element
``4π r²`` is applied by the molecular grid.
"""

from __future__ import annotations

import math

from hafnium.backend import Array, default_namespace
from hafnium.errors import InputError

TREUTLER_XI: tuple[float, ...] = (
    1.0,
    0.8, 0.9,
    1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9,
    1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0,
    1.5, 1.4,
    1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1,
    1.1, 1.0, 0.9, 0.9, 0.9, 0.9,
    2.0, 1.7,
    1.5, 1.5, 1.35, 1.35, 1.25, 1.2, 1.25, 1.3, 1.5, 1.5,
    1.3, 1.2, 1.2, 1.15, 1.15, 1.15,
    2.5, 2.2,
    2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    2.5, 2.1,
    3.685, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
)  # fmt: skip
"""The radial scale ``ξ`` of every element in the Treutler–Ahlrichs mapping.

Indexed by atomic number (index 0 unused). Hydrogen through krypton are the
values of Treutler and Ahlrichs, J. Chem. Phys. 102, 346 (1995), Table 1;
the heavier elements follow PySCF, which took them from Psi4.
"""


def _check_count(n_points: int) -> None:
    if n_points < 1:
        raise InputError("a radial quadrature needs at least one point")


def treutler_ahlrichs(n_points: int, xi: float = 1.0) -> tuple[Array, Array]:
    """Radii and weights of the Treutler–Ahlrichs mapping M4 with ``α = 0.6``.

    The Gauss–Chebyshev quadrature of the second kind on ``x ∈ (−1, 1)``,
    ``x_i = cos(iπ/(n + 1))``, is mapped to ``r = (ξ/ln 2) (1 + x)^0.6
    ln(2/(1 − x))`` (Treutler and Ahlrichs, J. Chem. Phys. 102, 346 (1995));
    ``xi`` scales the radii, :data:`TREUTLER_XI` per element.
    """
    _check_count(n_points)
    if not (math.isfinite(xi) and xi > 0.0):
        raise InputError("the radial scale is a positive number")
    xp = default_namespace()
    i = xp.arange(1, n_points + 1, dtype=xp.float64)
    step = math.pi / (n_points + 1)
    x = xp.cos(i * step)
    scale = xi / math.log(2.0)
    stretch = (1.0 + x) ** 0.6
    logarithm = xp.log((1.0 - x) / 2.0)
    radii = -scale * stretch * logarithm
    derivative = scale * stretch * (-0.6 / (1.0 + x) * logarithm + 1.0 / (1.0 - x))
    weights = step * xp.sin(i * step) * derivative
    return xp.flip(radii), xp.flip(weights)


def mura_knowles(n_points: int, far: float = 5.2) -> tuple[Array, Array]:
    """Radii and weights of the Mura–Knowles log3 mapping.

    Midpoint nodes ``x_i = (i + ½)/n`` on ``(0, 1)`` are mapped to
    ``r = −far ln(1 − x³)`` (Mura and Knowles, J. Chem. Phys. 104, 9848
    (1996)); ``far`` is 7 for the alkali and alkaline earth metals of the
    first periods and 5.2 otherwise in the original scheme.
    """
    _check_count(n_points)
    if not (math.isfinite(far) and far > 0.0):
        raise InputError("the radial scale is a positive number")
    xp = default_namespace()
    x = (xp.arange(n_points, dtype=xp.float64) + 0.5) / n_points
    cube = x**3
    radii = -far * xp.log(1.0 - cube)
    weights = far * 3.0 * x * x / ((1.0 - cube) * n_points)
    return radii, weights

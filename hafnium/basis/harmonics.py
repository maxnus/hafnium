"""Real solid harmonics as polynomials: the table behind their rotation.

The compiled layer transforms Cartesian Gaussians to the real solid
harmonics of ADR-0006 (Helgaker, Jørgensen and Olsen equation 6.4.47,
normalized on the sphere). The same table, read from Python, gives each
harmonic ``S_lm`` as a polynomial in ``x``, ``y`` and ``z``, which is what
a rotation of a shell needs (:mod:`hafnium.symmetry.rotations`).
"""

from __future__ import annotations

from hafnium import _kernels
from hafnium.backend import Array, host_array
from hafnium.errors import InputError

Exponents = tuple[int, int, int]
"""Powers of ``x``, ``y`` and ``z`` of a Cartesian monomial."""


def cartesian_components(angular_momentum: int) -> tuple[Exponents, ...]:
    """Exponents of the monomials ``x^i y^j z^k`` of degree ``angular_momentum``.

    The order of the compiled layer: ``x^l`` first, ``z^l`` last, ``i``
    descending and ``j`` descending within each ``i``.
    """
    l = angular_momentum
    if l < 0:
        raise InputError("angular momentum is non-negative")
    return tuple(
        (i, j, l - i - j) for i in range(l, -1, -1) for j in range(l - i, -1, -1)
    )


def solid_harmonic_coefficients(angular_momentum: int) -> Array:
    """Coefficients ``[2l + 1, n_cartesian]`` of each harmonic over the monomials.

    Row ``m`` holds the polynomial of the harmonic stored at position ``m``
    of a shell (``m = −l, ..., l``, a p shell as ``x, y, z``), column ``c``
    the monomial :func:`cartesian_components` lists at ``c``, so that
    ``S_m(r) = Σ_c T[m, c] x^i y^j z^k``.
    """
    if angular_momentum < 0:
        raise InputError("angular momentum is non-negative")
    return host_array(_kernels.spherical_transformation(angular_momentum))

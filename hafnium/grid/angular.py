"""Lebedev–Laikov quadrature rules on the unit sphere.

A rule with ``n`` points integrates every polynomial up to its algebraic
degree exactly; the weights sum to one, so ``∫ f dΩ ≈ 4π Σ_i w_i f(n_i)``.
The rules are stored as orbit generators under the octahedral group in
``data/lebedev.json`` and expanded here (ADR-0026).
"""

from __future__ import annotations

import functools
import itertools
import json
import math
from importlib import resources
from typing import Any

from hafnium.backend import Array, array_namespace, host_array
from hafnium.errors import InputError


@functools.cache
def _rules() -> dict[int, tuple[int, tuple[tuple[int, float, float, float], ...]]]:
    """Degree and orbit generators of every rule, by number of points."""
    resource = resources.files("hafnium.grid") / "data" / "lebedev.json"
    data: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    rules: dict[int, tuple[int, tuple[tuple[int, float, float, float], ...]]] = {}
    for size, rule in data["rules"].items():
        orbits = tuple(
            (int(code), float(a), float(b), float(weight))
            for code, a, b, weight in rule["orbits"]
        )
        rules[int(size)] = (int(rule["degree"]), orbits)
    return rules


LEBEDEV_SIZES: tuple[int, ...] = tuple(sorted(_rules()))
"""Numbers of points of the available rules, 6 to 5810."""


def lebedev_degree(n_points: int) -> int:
    """Algebraic degree of the rule with ``n_points`` points.

    Raises
    ------
    InputError
        If no rule has that many points; see :data:`LEBEDEV_SIZES`.
    """
    try:
        return _rules()[n_points][0]
    except KeyError:
        raise InputError(
            f"no Lebedev rule has {n_points} points; the rules have "
            f"{', '.join(str(size) for size in LEBEDEV_SIZES)} points"
        ) from None


def _signs(count: int) -> list[tuple[int, ...]]:
    return list(itertools.product((1, -1), repeat=count))


def _placed(
    positions: tuple[int, ...], values: tuple[float, ...], signs: tuple[int, ...]
) -> tuple[float, float, float]:
    """Return the point with ``values`` at ``positions``, signed, zero elsewhere."""
    point = [0.0, 0.0, 0.0]
    for position, value, sign in zip(positions, values, signs, strict=True):
        point[position] = sign * value
    return (point[0], point[1], point[2])


def _orbit(code: int, a: float, b: float) -> list[tuple[float, float, float]]:
    """Return the points of one orbit type of Lebedev and Laikov's ``gen_oh``.

    ``code`` 0: the 6 points ``(±1, 0, 0)``; 1: the 12 points ``(0, ±a, ±a)``
    with ``a = 1/√2``; 2: the 8 points ``(±a, ±a, ±a)`` with ``a = 1/√3``;
    3: the 24 points ``(±a, ±a, ±b)``, ``b = √(1 − 2a²)``; 4: the 24 points
    ``(±a, ±b, 0)``, ``b = √(1 − a²)``; 5: the 48 points ``(±a, ±b, ±c)``,
    ``c = √(1 − a² − b²)``, each with every placement of the components.
    """
    axes = (0, 1, 2)
    if code == 0:
        return [_placed((axis,), (1.0,), signs) for axis in axes for signs in _signs(1)]
    if code == 1:
        a = math.sqrt(0.5)
        return [
            _placed(pair, (a, a), signs)
            for pair in itertools.combinations(axes, 2)
            for signs in _signs(2)
        ]
    if code == 2:
        a = math.sqrt(1.0 / 3.0)
        return [_placed(axes, (a, a, a), signs) for signs in _signs(3)]
    if code == 3:
        b = math.sqrt(max(1.0 - 2.0 * a * a, 0.0))
        return [
            _placed(axes, tuple(b if k == odd else a for k in axes), signs)
            for odd in axes
            for signs in _signs(3)
        ]
    if code == 4:
        b = math.sqrt(max(1.0 - a * a, 0.0))
        return [
            _placed(pair, (a, b), signs)
            for pair in itertools.permutations(axes, 2)
            for signs in _signs(2)
        ]
    if code == 5:
        c = math.sqrt(max(1.0 - a * a - b * b, 0.0))
        return [
            _placed(axes, values, signs)
            for values in itertools.permutations((a, b, c))
            for signs in _signs(3)
        ]
    raise InputError(f"unknown Lebedev orbit code {code}")


@functools.cache
def _grid(n_points: int) -> tuple[Array, Array]:
    degree, orbits = _rules()[n_points]
    del degree
    points: list[tuple[float, float, float]] = []
    weights: list[float] = []
    for code, a, b, weight in orbits:
        orbit = _orbit(code, a, b)
        points.extend(orbit)
        weights.extend([weight] * len(orbit))
    if len(points) != n_points:
        raise InputError(f"the Lebedev rule of {n_points} points is corrupt")
    return host_array(points), host_array(weights)


def lebedev_grid(n_points: int) -> tuple[Array, Array]:
    """Return the unit vectors ``[n, 3]`` and weights ``[n]`` of a rule.

    The rule is the one with ``n_points`` points.

    The weights sum to one; the arrays are the caller's to modify. Raises
    :class:`~hafnium.errors.InputError` for a size that is not in
    :data:`LEBEDEV_SIZES`.
    """
    lebedev_degree(n_points)
    points, weights = _grid(n_points)
    xp = array_namespace(points, weights)
    return xp.asarray(points, copy=True), xp.asarray(weights, copy=True)

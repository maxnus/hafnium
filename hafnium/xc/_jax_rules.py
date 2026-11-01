"""The JAX rule of the functional evaluation (ADR-0037).

The energy density and its first derivatives are a host callback whose
Jacobian-vector product comes from the second derivatives of the energy
density that ``xcx`` supplies, in libxc's packed layouts.
"""

# JAX's own annotations leave parts of its API unknown to a strict checker.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax

from hafnium.backend import Array, array_namespace, to_host
from hafnium.xc.evaluation import host_second_derivatives, host_values
from hafnium.xc.functional import Functional

Values = tuple[Array, Array, Array | None, Array | None]

_UPPER_THREE = ((0, 1, 2), (1, 3, 4), (2, 4, 5))
"""Index of ``(k, l)`` in libxc's upper triangle of three variables."""
_UPPER_TWO = ((0, 1), (1, 2))


def _callback(function: Callable[..., Any], shapes: Any, *args: Array | None) -> Any:
    structs = jax.tree_util.tree_map(
        lambda shape: jax.ShapeDtypeStruct(shape, jax.numpy.float64),
        shapes,
        is_leaf=lambda node: (
            isinstance(node, tuple) and all(isinstance(i, int) for i in node)
        ),
    )
    return jax.pure_callback(function, structs, *args, vmap_method="sequential")


def _shapes(functional: Functional, n: int) -> tuple[Any, ...]:
    return (
        (n,),
        (2, n),
        (3, n) if functional.needs_gradient else None,
        (2, n) if functional.needs_tau else None,
    )


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def functional_values(
    functional: Functional, rho: Array, sigma: Array | None, tau: Array | None
) -> Values:
    """``(energy_density, vrho, vsigma, vtau)`` at every point, with the rule.

    ``sigma`` and ``tau`` are ``None`` when the rung does not read them.
    """
    n = int(rho.shape[1])
    return _callback(
        lambda r, s, t: host_values(functional, to_host(r), _host(s), _host(t)),
        _shapes(functional, n),
        rho,
        sigma,
        tau,
    )


@functional_values.defjvp
def _functional_values_jvp(
    functional: Functional, primals: tuple[Any, ...], tangents: tuple[Any, ...]
) -> tuple[Values, Values]:
    rho, sigma, tau = primals
    rho_dot, sigma_dot, tau_dot = tangents
    xp = array_namespace(rho)
    energy, vrho, vsigma, vtau = functional_values(functional, rho, sigma, tau)
    n = int(rho.shape[1])
    gradient, meta = functional.needs_gradient, functional.needs_tau
    second_shapes = (
        (3, n),
        (6, n) if gradient else None,
        (6, n) if gradient else None,
        (4, n) if meta else None,
        (6, n) if meta else None,
        (3, n) if meta else None,
    )
    rr, rs, ss, rt, st, tt = _callback(
        lambda r, s, t: host_second_derivatives(
            functional, to_host(r), _host(s), _host(t)
        ),
        second_shapes,
        rho,
        sigma,
        tau,
    )
    rho_dot = _tangent(xp, rho_dot, (2, n))
    energy_dot = xp.sum(vrho * rho_dot, axis=0)
    vrho_dot = xp.stack([rr[a] * rho_dot[0] + rr[a + 1] * rho_dot[1] for a in range(2)])
    vsigma_dot = None
    vtau_dot = None
    if gradient and vsigma is not None and rs is not None and ss is not None:
        sigma_dot = _tangent(xp, sigma_dot, (3, n))
        energy_dot = energy_dot + xp.sum(vsigma * sigma_dot, axis=0)
        vrho_dot = vrho_dot + xp.stack(
            [sum(rs[3 * a + k] * sigma_dot[k] for k in range(3)) for a in range(2)]
        )
        vsigma_dot = xp.stack(
            [
                sum(rs[3 * a + k] * rho_dot[a] for a in range(2))
                + sum(ss[_UPPER_THREE[k][l]] * sigma_dot[l] for l in range(3))
                for k in range(3)
            ]
        )
    if (
        meta
        and vtau is not None
        and rt is not None
        and st is not None
        and tt is not None
    ):
        tau_dot = _tangent(xp, tau_dot, (2, n))
        sigma_dot = _tangent(xp, sigma_dot, (3, n))
        energy_dot = energy_dot + xp.sum(vtau * tau_dot, axis=0)
        vrho_dot = vrho_dot + xp.stack(
            [sum(rt[2 * a + b] * tau_dot[b] for b in range(2)) for a in range(2)]
        )
        assert vsigma_dot is not None
        vsigma_dot = vsigma_dot + xp.stack(
            [sum(st[2 * k + a] * tau_dot[a] for a in range(2)) for k in range(3)]
        )
        vtau_dot = xp.stack(
            [
                sum(rt[2 * b + a] * rho_dot[b] for b in range(2))
                + sum(st[2 * k + a] * sigma_dot[k] for k in range(3))
                + sum(tt[_UPPER_TWO[a][b]] * tau_dot[b] for b in range(2))
                for a in range(2)
            ]
        )
    return (energy, vrho, vsigma, vtau), (energy_dot, vrho_dot, vsigma_dot, vtau_dot)


def _host(array: Array | None) -> Array | None:
    """Copy a callback argument to the host; eager JAX passes its own arrays."""
    return None if array is None else to_host(array)


def _tangent(xp: Any, tangent: Any, shape: tuple[int, ...]) -> Array:
    """Return a tangent as an array; JAX passes symbolic zeros for constants."""
    if tangent is None or isinstance(tangent, jax.custom_derivatives.SymbolicZero):
        return xp.zeros(shape, dtype=xp.float64)
    return tangent

"""The JAX rules of :class:`~hafnium.integrals.DifferentiableIntegrals` (ADR-0037).

Each integral type is a host callback, so it traces under ``jit``, with a
custom vector-Jacobian product that contracts the cotangent with the
derivative integrals of ADR-0035 on the host and returns one cotangent per
shell centre. The bra derivative of a symmetric integral is paired with the
symmetrized cotangent, which supplies the ket term, and the per-charge
derivative of the nuclear attraction with the cotangent itself.
"""

# JAX's own annotations leave parts of its API unknown to a strict checker.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax

from hafnium.backend import Array, host_array, host_namespace, to_host
from hafnium.integrals.differentiable import (
    DifferentiableIntegrals,
    no_pair_translation,
    no_translation,
)
from hafnium.integrals.engine import CoulombKernel, PointCharges

Charges = tuple[float, ...]


def _callback(function: Callable[..., Any], shapes: Any, *args: Array) -> Any:
    """Run ``function`` on host copies of ``args`` with the declared ``shapes``."""
    structs = jax.tree_util.tree_map(
        lambda shape: jax.ShapeDtypeStruct(shape, jax.numpy.float64),
        shapes,
        is_leaf=lambda node: (
            isinstance(node, tuple) and all(isinstance(i, int) for i in node)
        ),
    )
    return jax.pure_callback(function, structs, *args, vmap_method="sequential")


def _on_bra(derivative: Array, cotangent: Array, shells: Array) -> Array:
    """``Σ_{u ∈ s} Σ_v derivative[x, u, v] (c + cᵀ)[u, v]`` as ``[shell, 3]``.

    The symmetrized cotangent adds the ket term of a symmetric integral.
    """
    xp = host_namespace()
    cotangent = to_host(cotangent)
    symmetric = cotangent + xp.linalg.matrix_transpose(cotangent)
    rows = xp.einsum("xuv,uv->xu", derivative, symmetric)
    return xp.linalg.matrix_transpose(rows @ xp.linalg.matrix_transpose(shells))


# ----------------------------------------------------------------- overlap


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def overlap(integrals: DifferentiableIntegrals, centers: Array) -> Array:
    """Return the overlap ``[ao, ao]`` at ``centers``; differentiable."""
    n = integrals.orbital.n_functions
    return _callback(lambda c: to_host(integrals.host_overlap(c)), (n, n), centers)


def _overlap_fwd(
    integrals: DifferentiableIntegrals, centers: Array
) -> tuple[Array, Array]:
    return overlap(integrals, centers), centers


def _overlap_bwd(
    integrals: DifferentiableIntegrals, centers: Array, cotangent: Array
) -> tuple[Array]:
    def host(c: Array, cot: Array) -> Array:
        table = integrals.orbital_table(c)
        derivative = integrals.engine.overlap_derivative(table, table, no_translation())
        return _on_bra(derivative[0], cot, integrals.orbital_shells())

    return (_callback(host, (integrals.orbital.n_shells, 3), centers, cotangent),)


overlap.defvjp(_overlap_fwd, _overlap_bwd)


# ----------------------------------------------------------------- kinetic


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def kinetic(integrals: DifferentiableIntegrals, centers: Array) -> Array:
    """Return the kinetic energy ``[ao, ao]`` at ``centers``; differentiable."""
    n = integrals.orbital.n_functions
    return _callback(lambda c: to_host(integrals.host_kinetic(c)), (n, n), centers)


def _kinetic_fwd(
    integrals: DifferentiableIntegrals, centers: Array
) -> tuple[Array, Array]:
    return kinetic(integrals, centers), centers


def _kinetic_bwd(
    integrals: DifferentiableIntegrals, centers: Array, cotangent: Array
) -> tuple[Array]:
    def host(c: Array, cot: Array) -> Array:
        table = integrals.orbital_table(c)
        derivative = integrals.engine.kinetic_derivative(table, table, no_translation())
        return _on_bra(derivative[0], cot, integrals.orbital_shells())

    return (_callback(host, (integrals.orbital.n_shells, 3), centers, cotangent),)


kinetic.defvjp(_kinetic_fwd, _kinetic_bwd)


# ------------------------------------------------------- nuclear attraction


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def nuclear_attraction(
    integrals: DifferentiableIntegrals,
    kernel: CoulombKernel,
    charges: Charges,
    centers: Array,
    charge_positions: Array,
) -> Array:
    """Return the attraction ``[ao, ao]`` to ``charges``; differentiable."""
    n = integrals.orbital.n_functions
    return _callback(
        lambda c, r: to_host(
            integrals.host_nuclear_attraction(c, r, host_array(charges), kernel)
        ),
        (n, n),
        centers,
        charge_positions,
    )


def _attraction_fwd(
    integrals: DifferentiableIntegrals,
    kernel: CoulombKernel,
    charges: Charges,
    centers: Array,
    charge_positions: Array,
) -> tuple[Array, tuple[Array, Array]]:
    value = nuclear_attraction(integrals, kernel, charges, centers, charge_positions)
    return value, (centers, charge_positions)


def _attraction_bwd(
    integrals: DifferentiableIntegrals,
    kernel: CoulombKernel,
    charges: Charges,
    residuals: tuple[Array, Array],
    cotangent: Array,
) -> tuple[Array, Array]:
    centers, charge_positions = residuals

    def host(c: Array, r: Array, cot: Array) -> tuple[Array, Array]:
        table = integrals.orbital_table(c)
        by_center, by_charge = integrals.engine.nuclear_attraction_derivative(
            table,
            table,
            no_translation(),
            PointCharges(host_array(charges), to_host(r)),
            kernel=kernel,
        )
        xp = host_namespace()
        on_charges = xp.einsum("kxuv,uv->kx", by_charge[0], to_host(cot))
        return _on_bra(by_center[0], cot, integrals.orbital_shells()), on_charges

    shapes = ((integrals.orbital.n_shells, 3), (len(charges), 3))
    return _callback(host, shapes, centers, charge_positions, cotangent)


nuclear_attraction.defvjp(_attraction_fwd, _attraction_bwd)


# ------------------------------------------------------------------ metric


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def metric(
    integrals: DifferentiableIntegrals, kernel: CoulombKernel, auxiliary_centers: Array
) -> Array:
    """Return the two-centre integrals ``[aux, aux]``; differentiable."""
    n = integrals.fitting().n_functions
    return _callback(
        lambda a: to_host(integrals.host_metric(a, kernel)), (n, n), auxiliary_centers
    )


def _metric_fwd(
    integrals: DifferentiableIntegrals, kernel: CoulombKernel, auxiliary_centers: Array
) -> tuple[Array, Array]:
    return metric(integrals, kernel, auxiliary_centers), auxiliary_centers


def _metric_bwd(
    integrals: DifferentiableIntegrals,
    kernel: CoulombKernel,
    auxiliary_centers: Array,
    cotangent: Array,
) -> tuple[Array]:
    def host(a: Array, cot: Array) -> Array:
        table = integrals.auxiliary_table(a)
        derivative = integrals.engine.coulomb_two_center_derivative(
            table, table, no_translation(), kernel=kernel
        )
        return _on_bra(derivative[0], cot, integrals.auxiliary_shells())

    n_shells = integrals.fitting().n_shells
    return (_callback(host, (n_shells, 3), auxiliary_centers, cotangent),)


metric.defvjp(_metric_fwd, _metric_bwd)


# ------------------------------------------------------------ three-centre


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def three_center(
    integrals: DifferentiableIntegrals,
    kernel: CoulombKernel,
    auxiliary_centers: Array,
    centers: Array,
) -> Array:
    """Return the three-centre integrals ``[aux, ao, ao]``; differentiable."""
    n_aux = integrals.fitting().n_functions
    n = integrals.orbital.n_functions
    return _callback(
        lambda a, c: to_host(integrals.host_three_center(a, c, kernel)),
        (n_aux, n, n),
        auxiliary_centers,
        centers,
    )


def _three_center_fwd(
    integrals: DifferentiableIntegrals,
    kernel: CoulombKernel,
    auxiliary_centers: Array,
    centers: Array,
) -> tuple[Array, tuple[Array, Array]]:
    value = three_center(integrals, kernel, auxiliary_centers, centers)
    return value, (auxiliary_centers, centers)


def _three_center_bwd(
    integrals: DifferentiableIntegrals,
    kernel: CoulombKernel,
    residuals: tuple[Array, Array],
    cotangent: Array,
) -> tuple[Array, Array]:
    auxiliary_centers, centers = residuals

    def host(a: Array, c: Array, cot: Array) -> tuple[Array, Array]:
        fitting = integrals.auxiliary_table(a)
        table = integrals.orbital_table(c)
        by_auxiliary, by_orbital = integrals.engine.coulomb_three_center_derivative(
            fitting, table, table, no_pair_translation(), kernel=kernel
        )
        xp = host_namespace()
        cot = to_host(cot)
        on_auxiliary = xp.einsum("xPuv,Puv->xP", by_auxiliary[0], cot)
        symmetric = cot + xp.permute_dims(cot, (0, 2, 1))
        on_orbital = xp.einsum("xPuv,Puv->xu", by_orbital[0], symmetric)
        aux_shells = xp.linalg.matrix_transpose(integrals.auxiliary_shells())
        shells = xp.linalg.matrix_transpose(integrals.orbital_shells())
        return (
            xp.linalg.matrix_transpose(on_auxiliary @ aux_shells),
            xp.linalg.matrix_transpose(on_orbital @ shells),
        )

    shapes = (
        (integrals.fitting().n_shells, 3),
        (integrals.orbital.n_shells, 3),
    )
    return _callback(host, shapes, auxiliary_centers, centers, cotangent)


three_center.defvjp(_three_center_fwd, _three_center_bwd)

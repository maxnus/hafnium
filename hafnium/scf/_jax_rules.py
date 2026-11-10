"""The JAX rules of :class:`~hafnium.scf.DifferentiableScf` (ADR-0038).

The solver is a host callback with a custom vector-Jacobian product that
differentiates its solution implicitly. With the converged orbitals ``C``
and the rotations ``κ`` of :meth:`DifferentiableScf.rotate`, the solution
at positions ``R`` is ``P(κ*(R), R) = rotate(connect(C, S(R)), κ*(R))``
with ``κ*`` the root of the residual ``r(κ, R)``, so a cotangent ``C̄``
pulls back to ``R̄ = ∂_R Pᵀ C̄ − ∂_R rᵀ z`` with ``z`` solving
``∂_κ rᵀ z = ∂_κ Pᵀ C̄``: the transposed coupled-perturbed equations, solved
by conjugate gradients with the vector-Jacobian products of the residual,
one Fock build each, preconditioned by the orbital-energy differences.
Every transpose is JAX's own, so the complex classes follow its
conventions, and the derivative integrals are evaluated once, for the
combined cotangent of the integrals.

Whatever can fail on the host, the field, the checks of the state and the
response solve, raises in Python where the values are concrete, in a plain
call and in ``grad`` outside ``jit``, so the caller gets the library's own
error. On abstract values, under ``jit`` and ``vmap``, the same code runs
as a callback, and JAX reports what it raises as its own runtime error.
"""

# JAX's own annotations leave parts of its API unknown to a strict checker.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from hafnium.backend import Array, host_namespace, is_traced
from hafnium.backend._jax import register_pytrees
from hafnium.errors import ConvergenceError
from hafnium.scf.differentiable import DifferentiableScf, FittedIntegrals
from hafnium.structured import BlockMatrix, BlockVector

Rotations = tuple[Array, ...]

conjugate_gradients: Any = cg
"""The solver, whose annotations do not admit the pytrees it accepts."""

as_jax: Any = jnp.asarray
"""The conversion of a host array, whose result JAX annotates as its own type."""

PRECONDITIONER_FLOOR = 1e-3
"""Orbital-energy differences below this are raised to it in the preconditioner."""

register_pytrees()


def solve(scf: DifferentiableScf, positions: Array) -> BlockMatrix:
    """Return the converged orbitals at ``positions``; differentiable implicitly."""
    coefficients, _ = _solve(scf, positions)
    return coefficients


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _solve(scf: DifferentiableScf, positions: Array) -> tuple[BlockMatrix, BlockVector]:
    """Return the orbitals of the host solve and, for the preconditioner, its energies.

    The energies are constants of the backward pass; the state's own
    orbital energies come from :meth:`DifferentiableScf.complete`.
    Concrete positions are solved in Python, so that an error of the host
    solver reaches the caller as it is; abstract ones in a callback.
    """
    if is_traced(positions):
        structs = (
            tuple(jax.ShapeDtypeStruct(b.shape, b.dtype) for b in scf.guess.blocks),
            tuple(
                jax.ShapeDtypeStruct(b.shape, jnp.float64)
                for b in scf.occupations.blocks
            ),
        )
        coefficients, energies = jax.pure_callback(
            lambda p: _host(scf, p), structs, positions, vmap_method="sequential"
        )
    else:
        coefficients, energies = (
            tuple(as_jax(block) for block in blocks) for blocks in _host(scf, positions)
        )
    return (
        BlockMatrix(scf.structure, tuple(coefficients)),
        jax.lax.stop_gradient(BlockVector(scf.structure, tuple(energies))),
    )


def _host(
    scf: DifferentiableScf, positions: Array
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    xp = host_namespace()
    result = scf.host_solve(positions)
    return (
        tuple(xp.asarray(block) for block in result.mo_coefficients.blocks),
        tuple(xp.asarray(block) for block in result.mo_energies.blocks),
    )


def _solve_fwd(
    scf: DifferentiableScf, positions: Array
) -> tuple[tuple[BlockMatrix, BlockVector], tuple[Array, BlockMatrix, BlockVector]]:
    coefficients, energies = _solve(scf, positions)
    return (coefficients, energies), (positions, coefficients, energies)


def _solve_bwd(
    scf: DifferentiableScf,
    residuals: tuple[Array, BlockMatrix, BlockVector],
    cotangents: tuple[BlockMatrix, BlockVector],
) -> tuple[Array]:
    positions, coefficients, energies = residuals
    coefficient_bar, _ = cotangents
    integrals, pull_integrals = jax.vjp(scf.fitted_integrals, positions)
    hamiltonian = scf.hamiltonian_from(integrals)
    zero = scf.zero_rotations(coefficients)

    def connected(at: FittedIntegrals) -> BlockMatrix:
        return scf.connect(coefficients, scf.overlap_from(at))

    # The connection carries the cotangent of the orbitals to the overlap.
    _, pull_connection = jax.vjp(connected, integrals)
    (direct,) = pull_connection(coefficient_bar)
    # The rotations carry it to the stationarity condition.
    _, pull_rotation = jax.vjp(lambda k: scf.rotate(coefficients, k), zero)
    (rotation_bar,) = pull_rotation(coefficient_bar)
    _, pull_hessian = jax.vjp(
        lambda k: scf.residual(scf.rotate(coefficients, k), hamiltonian, check=False),
        zero,
    )
    response = solve_response(scf, lambda z: pull_hessian(z)[0], rotation_bar, energies)
    # The residual's own dependence on the integrals, at zero rotation.
    _, pull_mixed = jax.vjp(
        lambda at: scf.residual(connected(at), scf.hamiltonian_from(at), check=False),
        integrals,
    )
    (mixed,) = pull_mixed(response)
    combined = jax.tree_util.tree_map(lambda a, b: a - b, direct, mixed)
    (positions_bar,) = pull_integrals(combined)
    return (positions_bar,)


_solve.defvjp(_solve_fwd, _solve_bwd)


def solve_response(
    scf: DifferentiableScf,
    operator: Callable[[Rotations], Rotations],
    right_hand_side: Rotations,
    energies: BlockVector,
) -> Rotations:
    """Solve ``operator(z) = right_hand_side`` by preconditioned conjugate gradients.

    The operator is the transposed electronic Hessian, self-adjoint and
    positive definite at a stable state; the preconditioner divides by the
    orbital-energy differences, its diagonal in the canonical orbitals.
    The residual of the solution is verified on the host: in Python where
    the solution is concrete, and in a callback the solution passes
    through where it is abstract, under ``jit``, ``vmap`` and ``jacrev``.

    Raises
    ------
    ConvergenceError
        If the residual is above the tolerance after the allowed
        iterations, or not finite, as for an operator that is singular
        along the right-hand side; ``last`` holds the rotations reached.
    """
    gaps = tuple(jnp.maximum(gap, PRECONDITIONER_FLOOR) for gap in scf.gaps(energies))

    def precondition(z: Rotations) -> Rotations:
        return tuple(zb / gap for zb, gap in zip(z, gaps, strict=True))

    tolerance = scf.response.tolerance
    solution: Rotations
    solution, _ = conjugate_gradients(
        operator,
        right_hand_side,
        M=precondition,
        tol=tolerance,
        atol=tolerance,
        maxiter=scf.response.iterations,
    )
    residual: Array = _norm(
        jax.tree_util.tree_map(lambda a, b: a - b, operator(solution), right_hand_side)
    )
    bound: Array = as_jax(jnp.maximum(tolerance * _norm(right_hand_side), tolerance))

    def verified(z: Rotations, norm: Array, limit: Array) -> Rotations:
        # Written so that a residual that is not a number fails as well.
        if not float(norm) <= float(limit):
            raise ConvergenceError(
                "the response equations did not converge in "
                f"{scf.response.iterations} iterations (residual {float(norm):.2e}, "
                f"tolerance {float(limit):.2e})",
                z,
            )
        return z

    if not any(is_traced(value) for value in (*solution, residual, bound)):
        return verified(solution, residual, bound)
    structs = tuple(jax.ShapeDtypeStruct(z.shape, z.dtype) for z in solution)
    checked: Rotations = jax.pure_callback(
        verified, structs, solution, residual, bound, vmap_method="sequential"
    )
    return checked


def _norm(rotations: Rotations) -> Any:
    """Return the Euclidean norm over all blocks as a 0-d array, which traces."""
    return jnp.sqrt(sum(jnp.sum(jnp.abs(block) ** 2) for block in rotations))

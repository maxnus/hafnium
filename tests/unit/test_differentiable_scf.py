"""The SCF differentiates implicitly through its stationarity condition (ADR-0038)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from hafnium.backend import Array, Backend, array_namespace, use_backend
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.molecule import Molecule
from hafnium.properties import nuclear_gradient
from hafnium.scf import (
    DifferentiableScf,
    ScfOptions,
    ScfResult,
    ScfSolution,
    hartree_fock,
)
from hafnium.symmetry import KPoints, SymmetrySpec
from tests.gradients import finite_differences
from tests.hamiltonians import ENGINE
from tests.molecules import water, water_cation
from tests.shell_tables import LIBRARY
from tests.tolerances import (
    FITTED_ENERGY_SOLVE_TOL,
    GRADIENT_FINITE_DIFFERENCE_TOL,
    GRADIENT_TOL,
    MO_ENERGY_TOL,
    OPERATOR_REL_TOL,
    RESPONSE_DERIVATIVE_FLOOR,
    RESPONSE_FINITE_DIFFERENCE_TOL,
    TRACED_TOL,
)

AUXILIARY = "def2-universal-jkfit"
# Tight convergence, so that the finite differences are limited by the
# rounding of the energies and not by the residual orbital error.
OPTIONS = ScfOptions(
    energy_tolerance=1e-13, gradient_tolerance=1e-10, max_iterations=200
)
CLASSES = [
    "RHF",
    "complex RHF",
    "UHF",
    "complex UHF",
    "paired UHF",
    "GHF",
    "complex GHF",
    "paired GHF",
]
RESPONSE_CLASSES = ["RHF", "UHF", "complex GHF"]


def bases(molecule: Molecule) -> tuple[BasisSet, BasisSet]:
    return (
        BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY),
        BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY),
    )


def molecule_for(name: str) -> Molecule:
    return water_cation() if name == "UHF" else water()


def converge(molecule: Molecule, symmetry: SymmetrySpec) -> ScfResult:
    basis, auxiliary = bases(molecule)
    return hartree_fock(
        ENGINE, basis, molecule, symmetry=symmetry, auxiliary=auxiliary, options=OPTIONS
    )


def differentiable(molecule: Molecule, symmetry: SymmetrySpec) -> DifferentiableScf:
    basis, auxiliary = bases(molecule)
    return DifferentiableScf.from_molecule(
        ENGINE, basis, molecule, auxiliary, symmetry=symmetry, options=OPTIONS
    )


def gradient_of(
    scf: DifferentiableScf,
    molecule: Molecule,
    function: Callable[[ScfSolution], Array],
    *,
    jit: bool = False,
) -> Array:
    """``d function(solve(positions)) / d positions`` under JAX, as a host array."""
    jax = pytest.importorskip("jax")
    with use_backend("jax") as backend:
        positions = backend.namespace.asarray(molecule.positions_array())

        def objective(at: Array) -> Array:
            return function(scf.solve(at))

        derivative: Any = jax.grad(objective)
        if jit:
            derivative = jax.jit(derivative)
        return np.asarray(derivative(positions))


def homo_energy(solution: ScfSolution, occupied: int) -> Array:
    return solution.mo_energies.blocks[0][occupied - 1]


def host_homo_energy(result: ScfResult, occupied: int) -> float:
    return float(result.mo_energies.blocks[0][occupied - 1])


def test_the_solution_is_the_host_state(backend: Backend) -> None:
    molecule = water()
    symmetry = SymmetrySpec.from_name("RHF")
    expected = converge(molecule, symmetry)
    scf = differentiable(molecule, symmetry)
    positions = backend.namespace.asarray(molecule.positions_array())
    solution = scf.solve(positions)
    assert float(solution.energy) == pytest.approx(
        expected.energy, abs=FITTED_ENERGY_SOLVE_TOL
    )
    host = scf.host_solve(positions)
    assert host.converged
    assert host.energy == pytest.approx(expected.energy, abs=FITTED_ENERGY_SOLVE_TOL)
    assert host.history[-1].energy == pytest.approx(
        host.energy, abs=OPTIONS.energy_tolerance
    )
    np.testing.assert_allclose(
        np.asarray(solution.mo_energies.blocks[0]),
        np.asarray(expected.mo_energies.blocks[0]),
        atol=MO_ENERGY_TOL,
    )
    for block in (*solution.density.blocks, *solution.mo_coefficients.blocks):
        assert array_namespace(block) is backend.namespace
    assert array_namespace(solution.energy) is backend.namespace


@pytest.mark.parametrize("name", CLASSES)
def test_the_energy_gradient_is_the_analytic_gradient(name: str) -> None:
    """``grad`` of the energy carries the overlap term, in every class."""
    molecule = molecule_for(name)
    symmetry = SymmetrySpec.from_name(name)
    basis, auxiliary = bases(molecule)
    expected = nuclear_gradient(
        ENGINE, basis, auxiliary, molecule, converge(molecule, symmetry)
    ).total
    scf = differentiable(molecule, symmetry)
    gradient = gradient_of(scf, molecule, lambda solution: solution.energy)
    np.testing.assert_allclose(gradient, np.asarray(expected), atol=GRADIENT_TOL)


def test_the_energy_gradient_matches_finite_differences() -> None:
    molecule = water()
    symmetry = SymmetrySpec.from_name("RHF")
    numeric = finite_differences(molecule, lambda m: converge(m, symmetry).energy)
    scf = differentiable(molecule, symmetry)
    gradient = gradient_of(scf, molecule, lambda solution: solution.energy)
    np.testing.assert_allclose(gradient, numeric, atol=GRADIENT_FINITE_DIFFERENCE_TOL)


def test_point_group_sectors_give_the_same_gradient() -> None:
    molecule = water()
    plain = differentiable(molecule, SymmetrySpec.from_name("RHF"))
    adapted = differentiable(
        molecule, SymmetrySpec.from_name("RHF").with_point_group("C2v")
    )
    assert adapted.structure.irreps is not None
    expected = gradient_of(plain, molecule, lambda solution: solution.energy)
    gradient = gradient_of(adapted, molecule, lambda solution: solution.energy)
    np.testing.assert_allclose(gradient, expected, atol=GRADIENT_TOL)


@pytest.mark.parametrize("name", RESPONSE_CLASSES)
def test_orbital_energies_differentiate_through_the_response(name: str) -> None:
    """A non-stationary quantity needs the coupled-perturbed response, and gets it."""
    molecule = molecule_for(name)
    symmetry = SymmetrySpec.from_name(name)
    scf = differentiable(molecule, symmetry)
    occupied = scf.occupied[0]
    numeric = finite_differences(
        molecule, lambda m: host_homo_energy(converge(m, symmetry), occupied)
    )
    gradient = gradient_of(
        scf, molecule, lambda solution: homo_energy(solution, occupied)
    )
    assert np.abs(numeric).max() > RESPONSE_DERIVATIVE_FLOOR
    np.testing.assert_allclose(gradient, numeric, atol=RESPONSE_FINITE_DIFFERENCE_TOL)


def test_jit_gives_the_same_gradient() -> None:
    molecule = water()
    scf = differentiable(molecule, SymmetrySpec.from_name("RHF"))
    occupied = scf.occupied[0]

    def observable(solution: ScfSolution) -> Array:
        return solution.energy + homo_energy(solution, occupied)

    eager = gradient_of(scf, molecule, observable)
    traced = gradient_of(scf, molecule, observable, jit=True)
    np.testing.assert_allclose(traced, eager, atol=TRACED_TOL)


def test_the_response_operator_is_self_adjoint_and_positive() -> None:
    """The residual's Jacobian in the rotations, transposed as the solver uses it."""
    jax = pytest.importorskip("jax")
    molecule = water()
    scf = differentiable(molecule, SymmetrySpec.from_name("complex GHF"))
    rng = np.random.default_rng(3)
    with use_backend("jax") as backend:
        xp = backend.namespace
        positions = xp.asarray(molecule.positions_array())
        solution = scf.solve(positions)
        coefficients = solution.mo_coefficients
        hamiltonian = scf.hamiltonian(positions)
        zero = scf.zero_rotations(coefficients)

        def residual(rotations: tuple[Array, ...]) -> tuple[Array, ...]:
            return scf.residual(scf.rotate(coefficients, rotations), hamiltonian)

        pull: Any = jax.vjp(residual, zero)[1]

        def apply(z: tuple[Array, ...]) -> tuple[Array, ...]:
            result: tuple[Array, ...] = pull(z)[0]
            return result

        def random() -> tuple[Array, ...]:
            return tuple(
                xp.asarray(
                    rng.standard_normal(block.shape)
                    + 1j * rng.standard_normal(block.shape)
                )
                for block in zero
            )

        def inner(a: tuple[Array, ...], b: tuple[Array, ...]) -> float:
            return float(
                sum(xp.real(xp.sum(xp.conj(x) * y)) for x, y in zip(a, b, strict=True))
            )

        u, v = random(), random()
        left, right = inner(u, apply(v)), inner(apply(u), v)
        assert left == pytest.approx(right, rel=OPERATOR_REL_TOL)
        assert inner(v, apply(v)) > 0.0


def test_wrong_inputs_are_rejected() -> None:
    molecule = water()
    scf = differentiable(molecule, SymmetrySpec.from_name("RHF"))
    with pytest.raises(InputError, match=r"\[atom, 3\]"):
        scf.solve(np.zeros((2, 3)))
    basis, auxiliary = bases(molecule)
    with pytest.raises(InputError, match="k-points"):
        DifferentiableScf.from_molecule(
            ENGINE,
            basis,
            molecule,
            auxiliary,
            symmetry=SymmetrySpec.from_name("RHF").with_kpoints(
                KPoints.mesh((1, 1, 1))
            ),
        )

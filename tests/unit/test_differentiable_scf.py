"""The SCF differentiates implicitly through its stationarity condition (ADR-0038)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from hafnium.backend import Array, Backend, array_namespace, use_backend
from hafnium.errors import BackendError, ConvergenceError, InputError, SymmetryError
from hafnium.molecule import Molecule
from hafnium.properties import nuclear_gradient
from hafnium.scf import (
    DifferentiableScf,
    ResponseOptions,
    ScfOptions,
    ScfResult,
    ScfSolution,
    hartree_fock,
)
from hafnium.symmetry import KPoints, SymmetrySpec
from tests.gradients import CLASSES, OPTIONS, bases, finite_differences
from tests.hamiltonians import ENGINE
from tests.molecules import water, water_cation
from tests.tolerances import (
    ENERGY_TOL,
    FITTED_ENERGY_SOLVE_TOL,
    GRADIENT_FINITE_DIFFERENCE_TOL,
    GRADIENT_TOL,
    MO_ENERGY_TOL,
    OPERATOR_REL_TOL,
    RESPONSE_DERIVATIVE_FLOOR,
    RESPONSE_FINITE_DIFFERENCE_TOL,
    TRACED_TOL,
)

RESPONSE_CLASSES = ["RHF", "UHF", "complex GHF"]
C2V = SymmetrySpec.from_name("RHF").with_point_group("C2v")


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
    adapted = differentiable(molecule, C2V)
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


def moved(molecule: Molecule, atom: int, shift: tuple[float, float, float]) -> Array:
    """The positions of ``molecule`` with ``atom`` moved by ``shift`` Bohr."""
    positions = np.array(molecule.positions_array())
    positions[atom] += shift
    return positions


def test_a_geometry_that_breaks_the_point_group_is_rejected() -> None:
    """The sectors of the reference geometry describe no other symmetry."""
    jax = pytest.importorskip("jax")
    molecule = water()
    scf = differentiable(molecule, C2V)
    broken = moved(molecule, 1, (0.1, 0.2, 0.0))
    with pytest.raises(SymmetryError, match="has left C2v"):
        scf.solve(broken)
    with pytest.raises(SymmetryError, match="has left C2v"):
        scf.hamiltonian(broken)
    with use_backend("jax") as backend:
        positions = backend.namespace.asarray(broken)
        with pytest.raises(SymmetryError, match="has left C2v"):
            scf.solve(positions)
        with pytest.raises(SymmetryError, match="has left C2v"):
            jax.grad(scf.energy)(positions)
        with pytest.raises(jax.errors.JaxRuntimeError, match="has left C2v"):
            jax.jit(jax.grad(scf.energy))(positions)


def test_a_geometry_that_keeps_the_point_group_is_solved_in_its_sectors() -> None:
    jax = pytest.importorskip("jax")
    molecule = water()
    plain = differentiable(molecule, SymmetrySpec.from_name("RHF"))
    adapted = differentiable(molecule, C2V)
    along_the_axis = moved(molecule, 0, (0.0, 0.0, 0.05))
    assert float(adapted.solve(along_the_axis).energy) == pytest.approx(
        float(plain.solve(along_the_axis).energy), abs=ENERGY_TOL
    )
    with use_backend("jax") as backend:
        positions = backend.namespace.asarray(along_the_axis)
        expected = np.asarray(jax.grad(plain.energy)(positions))
        gradient = np.asarray(jax.grad(adapted.energy)(positions))
    np.testing.assert_allclose(gradient, expected, atol=GRADIENT_TOL)


def test_an_unconverged_response_raises_eagerly_and_under_jit() -> None:
    jax = pytest.importorskip("jax")
    molecule = water()
    scf = dataclasses.replace(
        differentiable(molecule, SymmetrySpec.from_name("RHF")),
        response=ResponseOptions(iterations=1),
    )
    occupied = scf.occupied[0]

    def objective(at: Array) -> Array:
        return homo_energy(scf.solve(at), occupied)

    def lowest_two(at: Array) -> Array:
        return scf.solve(at).mo_energies.blocks[0][:2]

    with use_backend("jax") as backend:
        positions = backend.namespace.asarray(molecule.positions_array())
        with pytest.raises(ConvergenceError, match="response equations") as error:
            jax.grad(objective)(positions)
        rotations = error.value.last
        assert [block.shape for block in rotations] == [
            block.shape
            for block in scf.zero_rotations(scf.solve(positions).mo_coefficients)
        ]
        with pytest.raises(jax.errors.JaxRuntimeError, match="response equations"):
            jax.jit(jax.grad(objective))(positions)
        with pytest.raises(jax.errors.JaxRuntimeError, match="response equations"):
            jax.jacrev(lowest_two)(positions)


def test_an_unconverged_field_raises_the_same_error_for_jax_positions() -> None:
    """The host solver's errors reach the caller as themselves outside ``jit``."""
    jax = pytest.importorskip("jax")
    molecule = water()
    scf = dataclasses.replace(
        differentiable(molecule, SymmetrySpec.from_name("RHF")),
        options=ScfOptions(max_iterations=3),
    )
    host = molecule.positions_array()
    with pytest.raises(ConvergenceError, match="SCF did not converge"):
        scf.solve(host)
    with use_backend("jax") as backend:
        positions = backend.namespace.asarray(host)
        with pytest.raises(ConvergenceError, match="SCF did not converge") as error:
            scf.solve(positions)
        assert isinstance(error.value.last, ScfResult)
        with pytest.raises(ConvergenceError, match="SCF did not converge"):
            jax.grad(scf.energy)(positions)
        with pytest.raises(jax.errors.JaxRuntimeError, match="ConvergenceError"):
            jax.jit(scf.energy)(positions)


def test_the_hamiltonian_carries_the_nuclear_repulsion(backend: Backend) -> None:
    molecule = water()
    scf = differentiable(molecule, SymmetrySpec.from_name("RHF"))
    positions = backend.namespace.asarray(molecule.positions_array())
    hamiltonian = scf.hamiltonian(positions)
    assert float(hamiltonian.nuclear_repulsion) == pytest.approx(
        molecule.nuclear_repulsion_energy(), abs=ENERGY_TOL
    )
    solution = scf.solve(positions)
    _, terms = hamiltonian.fock(solution.density)
    assert float(terms.total) == pytest.approx(float(solution.energy), abs=ENERGY_TOL)


def test_the_occupied_counts_follow_the_occupations() -> None:
    scf = differentiable(water(), SymmetrySpec.from_name("RHF"))
    (count,) = scf.occupied
    n_mo = int(scf.occupations.blocks[0].shape[0])
    highest: Array = np.arange(n_mo) == count - 1
    emptied = scf.occupations.map_blocks(lambda block: np.where(highest, 0.0, block))
    assert dataclasses.replace(scf, occupations=emptied).occupied == (count - 1,)
    halved = scf.occupations.map_blocks(
        lambda block: np.where(highest, block / 2, block)
    )
    with pytest.raises(InputError, match="fractional occupations"):
        dataclasses.replace(scf, occupations=halved)
    other = differentiable(water_cation(), SymmetrySpec.from_name("UHF"))
    with pytest.raises(InputError, match="another structure"):
        dataclasses.replace(scf, occupations=other.occupations)


def test_the_state_lives_in_the_library_of_the_positions() -> None:
    """Host positions give a host state under any backend; JAX ones are 64-bit."""
    pytest.importorskip("jax")
    molecule = water()
    scf = differentiable(molecule, SymmetrySpec.from_name("RHF"))
    host = molecule.positions_array()
    with use_backend("jax") as backend:
        solution = scf.solve(host)
        assert array_namespace(solution.energy) is array_namespace(host)
        single = backend.namespace.asarray(host, dtype=backend.namespace.float32)
        with pytest.raises(BackendError, match="float64"):
            scf.solve(single)


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

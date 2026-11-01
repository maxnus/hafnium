"""Integrals as differentiable functions of the shell centres (ADR-0037)."""

# JAX's own annotations leave parts of its API unknown to a strict checker.
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownLambdaType=false

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from hafnium.backend import Array, Backend, array_namespace, use_backend
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals import (
    CoulombKernel,
    DensityFitting,
    DensityFittingDerivatives,
    DifferentiableIntegrals,
    kinetic,
    kinetic_derivative,
    nuclear_attraction,
    nuclear_attraction_derivative,
    nuclear_repulsion,
    overlap,
    overlap_derivative,
)
from hafnium.molecule import Molecule
from hafnium.properties import nuclear_gradient
from hafnium.scf import ScfOptions, hartree_fock
from hafnium.symmetry import SymmetrySpec
from tests.hamiltonians import ENGINE
from tests.molecules import ammonia_tilted, water
from tests.shell_tables import LIBRARY
from tests.tolerances import (
    FITTED_ENERGY_SOLVE_TOL,
    GRADIENT_FINITE_DIFFERENCE_TOL,
    GRADIENT_TOL,
    INTEGRAL_TOL,
)

AUXILIARY = "def2-universal-jkfit"
STEP = 1e-4
KERNELS = ["overlap", "kinetic", "attraction", "metric", "three_center", "repulsion"]


def setup(
    molecule: Molecule, basis_name: str
) -> tuple[DifferentiableIntegrals, BasisSet, BasisSet]:
    basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
    auxiliary = BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY)
    return (
        DifferentiableIntegrals.from_basis(ENGINE, basis, molecule, auxiliary),
        basis,
        auxiliary,
    )


def weighted(
    integrals: DifferentiableIntegrals,
    molecule: Molecule,
    weights: dict[str, Array],
    which: str,
) -> Callable[[Array], Array]:
    """A scalar function of the positions: a kernel contracted with ``weights``."""
    charges = np.asarray(molecule.nuclear_charges)

    def function(positions: Array) -> Array:
        xp = array_namespace(positions)
        centers = integrals.orbital_centers(positions)
        if which == "overlap":
            return xp.sum(weights["ao"] * integrals.overlap(centers))
        if which == "kinetic":
            return xp.sum(weights["ao"] * integrals.kinetic(centers))
        if which == "attraction":
            attraction = integrals.nuclear_attraction(centers, positions, charges)
            return xp.sum(weights["ao"] * attraction)
        auxiliary = integrals.auxiliary_centers(positions)
        if which == "metric":
            return xp.sum(weights["aux"] * integrals.metric(auxiliary))
        if which == "three_center":
            return xp.sum(weights["three"] * integrals.three_center(auxiliary, centers))
        return nuclear_repulsion(positions, charges)

    return function


def random_weights(
    rng: np.random.Generator, basis: BasisSet, auxiliary: BasisSet, xp: object
) -> dict[str, Array]:
    n, n_aux = basis.n_functions, auxiliary.n_functions
    return {
        "ao": xp.asarray(rng.standard_normal((n, n))),  # type: ignore[attr-defined]
        "aux": xp.asarray(rng.standard_normal((n_aux, n_aux))),  # type: ignore[attr-defined]
        "three": xp.asarray(rng.standard_normal((n_aux, n, n))),  # type: ignore[attr-defined]
    }


def test_values_match_the_molecular_helpers(backend: Backend) -> None:
    molecule = water()
    integrals, basis, auxiliary = setup(molecule, "cc-pvdz")
    xp = backend.namespace
    positions = xp.asarray(molecule.positions_array())
    centers = integrals.orbital_centers(positions)
    fitting = DensityFitting.build(ENGINE, basis, auxiliary, molecule)
    expected = {
        "overlap": overlap(ENGINE, basis, molecule),
        "kinetic": kinetic(ENGINE, basis, molecule),
        "attraction": nuclear_attraction(ENGINE, basis, molecule),
        "metric": fitting.metric,
        "three_center": fitting.three_center,
    }
    charges = np.asarray(molecule.nuclear_charges)
    auxiliary_centers = integrals.auxiliary_centers(positions)
    values = {
        "overlap": integrals.overlap(centers),
        "kinetic": integrals.kinetic(centers),
        "attraction": integrals.nuclear_attraction(centers, positions, charges),
        "metric": integrals.metric(auxiliary_centers),
        "three_center": integrals.three_center(auxiliary_centers, centers),
    }
    for name, value in values.items():
        assert array_namespace(value) is xp
        np.testing.assert_allclose(
            np.asarray(value), np.asarray(expected[name]), atol=INTEGRAL_TOL
        )
    assert float(nuclear_repulsion(positions, charges)) == pytest.approx(
        molecule.nuclear_repulsion_energy(), abs=INTEGRAL_TOL
    )


def test_wrong_inputs_are_rejected() -> None:
    molecule = water()
    basis = BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY)
    integrals = DifferentiableIntegrals.from_basis(ENGINE, basis, molecule)
    with pytest.raises(InputError, match="auxiliary"):
        integrals.metric(np.zeros((1, 3)))
    with pytest.raises(InputError, match=r"\[shell, 3\]"):
        integrals.overlap(np.zeros((2, 3)))
    with pytest.raises(InputError, match="one atom per orbital shell"):
        DifferentiableIntegrals(ENGINE, integrals.orbital, np.zeros(2, dtype=np.int64))


@pytest.mark.parametrize("which", KERNELS)
def test_gradients_match_finite_differences_and_trace(which: str) -> None:
    jax = pytest.importorskip("jax")
    molecule = ammonia_tilted()
    rng = np.random.default_rng(7)
    with use_backend("jax") as backend:
        integrals, basis, auxiliary = setup(molecule, "sto-3g")
        weights = random_weights(rng, basis, auxiliary, backend.namespace)
        function = weighted(integrals, molecule, weights, which)
        positions = backend.namespace.asarray(molecule.positions_array())
        gradient = jax.grad(function)(positions)
        traced = jax.jit(jax.grad(function))(positions)
        host = np.asarray(positions)
        numeric = np.zeros_like(host)
        for atom in range(host.shape[0]):
            for axis in range(3):
                shift = np.zeros_like(host)
                shift[atom, axis] = STEP
                up = float(function(backend.namespace.asarray(host + shift)))
                down = float(function(backend.namespace.asarray(host - shift)))
                numeric[atom, axis] = (up - down) / (2.0 * STEP)
    np.testing.assert_allclose(
        np.asarray(gradient), numeric, atol=GRADIENT_FINITE_DIFFERENCE_TOL
    )
    np.testing.assert_allclose(np.asarray(traced), np.asarray(gradient), atol=1e-13)


def atom_map(basis: BasisSet, n_atoms: int) -> Array:
    labels = basis.function_labels()
    return np.asarray(
        [
            [1.0 if label.atom == atom else 0.0 for label in labels]
            for atom in range(n_atoms)
        ]
    )


def test_one_electron_gradients_equal_the_contracted_derivative_integrals() -> None:
    """The rules reproduce the derivative kernels exactly, not just numerically."""
    jax = pytest.importorskip("jax")
    molecule = water()
    rng = np.random.default_rng(3)
    with use_backend("jax") as backend:
        xp = backend.namespace
        integrals, basis, auxiliary = setup(molecule, "cc-pvdz")
        weights = random_weights(rng, basis, auxiliary, xp)
        positions = xp.asarray(molecule.positions_array())
        atoms = atom_map(basis, molecule.n_atoms)
        w = np.asarray(weights["ao"])
        symmetric = w + w.T
        by_center, by_nucleus = nuclear_attraction_derivative(ENGINE, basis, molecule)
        derivatives = {
            "overlap": overlap_derivative(ENGINE, basis, molecule),
            "kinetic": kinetic_derivative(ENGINE, basis, molecule),
            "attraction": by_center,
        }
        for which, derivative in derivatives.items():
            function = weighted(integrals, molecule, weights, which)
            gradient = np.asarray(jax.grad(function)(positions))
            rows = np.einsum("xuv,uv->xu", np.asarray(derivative), symmetric)
            expected = (rows @ atoms.T).T
            if which == "attraction":
                expected = expected + np.einsum(
                    "kxuv,uv->kx", np.asarray(by_nucleus), w
                )
            np.testing.assert_allclose(gradient, expected, atol=1e-12)


def test_fitting_gradients_equal_the_contracted_derivative_integrals() -> None:
    jax = pytest.importorskip("jax")
    molecule = water()
    rng = np.random.default_rng(3)
    with use_backend("jax") as backend:
        xp = backend.namespace
        integrals, basis, auxiliary = setup(molecule, "cc-pvdz")
        weights = random_weights(rng, basis, auxiliary, xp)
        positions = xp.asarray(molecule.positions_array())
        atoms = atom_map(basis, molecule.n_atoms)
        aux_atoms = atom_map(auxiliary, molecule.n_atoms)
        derivatives = DensityFittingDerivatives.build(
            ENGINE, basis, auxiliary, molecule
        )
        metric = jax.grad(weighted(integrals, molecule, weights, "metric"))(positions)
        wa = np.asarray(weights["aux"])
        rows = np.einsum("xPQ,PQ->xP", np.asarray(derivatives.metric), wa + wa.T)
        np.testing.assert_allclose(
            np.asarray(metric), (rows @ aux_atoms.T).T, atol=1e-12
        )
        function = weighted(integrals, molecule, weights, "three_center")
        three = np.asarray(jax.grad(function)(positions))
        w3 = np.asarray(weights["three"])
        on_aux = np.einsum("xPuv,Puv->xP", np.asarray(derivatives.auxiliary), w3)
        symmetric = w3 + np.transpose(w3, (0, 2, 1))
        on_orbital = np.einsum(
            "xPuv,Puv->xu", np.asarray(derivatives.orbital), symmetric
        )
        expected = (on_aux @ aux_atoms.T).T + (on_orbital @ atoms.T).T
        np.testing.assert_allclose(three, expected, atol=1e-12)


def fixed_density_energy(
    integrals: DifferentiableIntegrals, spinor: Array, charges: Array, xp: object
) -> Callable[[Array], Array]:
    """The density-fitted Hartree–Fock energy at the fixed spin-orbital density."""
    n = int(spinor.shape[0]) // 2
    total = spinor[:n, :n] + spinor[n:, n:]

    def energy(positions: Array) -> Array:
        centers = integrals.orbital_centers(positions)
        auxiliary_centers = integrals.auxiliary_centers(positions)
        core = integrals.kinetic(centers) + integrals.nuclear_attraction(
            centers, positions, charges
        )
        metric = integrals.metric(auxiliary_centers)
        three = integrals.three_center(auxiliary_centers, centers)
        fitted = xp.einsum("Puv,vu->P", three, total)  # type: ignore[attr-defined]
        coefficients = xp.linalg.solve(metric, fitted)  # type: ignore[attr-defined]
        coulomb = 0.5 * xp.sum(fitted * coefficients)  # type: ignore[attr-defined]
        # Exchange over the spin-orbital density: L̃_P = block_diag(L_P, L_P).
        zero = xp.zeros_like(three)  # type: ignore[attr-defined]
        doubled = xp.concat(  # type: ignore[attr-defined]
            [
                xp.concat([three, zero], axis=2),  # type: ignore[attr-defined]
                xp.concat([zero, three], axis=2),  # type: ignore[attr-defined]
            ],
            axis=1,
        )
        pairs = xp.einsum(  # type: ignore[attr-defined]
            "Pij,jk,Qkl,li->PQ", doubled, spinor, doubled, spinor
        )
        solved = xp.linalg.solve(metric, pairs)  # type: ignore[attr-defined]
        exchange = -0.5 * xp.linalg.trace(solved)  # type: ignore[attr-defined]
        one_electron = xp.sum(core * total)  # type: ignore[attr-defined]
        return nuclear_repulsion(positions, charges) + one_electron + coulomb + exchange

    return energy


def test_fixed_density_energy_differentiates_to_the_analytic_gradient() -> None:
    """``grad`` at fixed density is the analytic gradient without its overlap term."""
    jax = pytest.importorskip("jax")
    molecule = water()
    basis = BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY)
    auxiliary = BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY)
    options = ScfOptions(energy_tolerance=1e-13, gradient_tolerance=1e-10)
    state = hartree_fock(
        ENGINE,
        basis,
        molecule,
        symmetry=SymmetrySpec.from_name("RHF"),
        auxiliary=auxiliary,
        options=options,
    )
    analytic = nuclear_gradient(ENGINE, basis, auxiliary, molecule, state)
    expected = np.asarray(analytic.total) - np.asarray(analytic.overlap)
    with use_backend("jax") as backend:
        xp = backend.namespace
        integrals = DifferentiableIntegrals.from_basis(
            ENGINE, basis, molecule, auxiliary
        )
        spinor = xp.asarray(np.real(np.asarray(state.density.to_dense())))
        charges = np.asarray(molecule.nuclear_charges)
        energy = fixed_density_energy(integrals, spinor, charges, xp)
        positions = xp.asarray(molecule.positions_array())
        assert float(energy(positions)) == pytest.approx(
            state.energy, abs=FITTED_ENERGY_SOLVE_TOL
        )
        gradient = jax.grad(energy)(positions)
    np.testing.assert_allclose(np.asarray(gradient), expected, atol=GRADIENT_TOL)


def test_range_separated_gradients_match_finite_differences() -> None:
    jax = pytest.importorskip("jax")
    molecule = water()
    kernel = CoulombKernel(0.4)
    rng = np.random.default_rng(5)
    with use_backend("jax") as backend:
        xp = backend.namespace
        integrals, basis, auxiliary = setup(molecule, "sto-3g")
        weights = random_weights(rng, basis, auxiliary, xp)
        charges = np.asarray(molecule.nuclear_charges)

        def function(positions: Array) -> Array:
            centers = integrals.orbital_centers(positions)
            auxiliary_centers = integrals.auxiliary_centers(positions)
            attraction = integrals.nuclear_attraction(
                centers, positions, charges, kernel=kernel
            )
            three = integrals.three_center(auxiliary_centers, centers, kernel=kernel)
            return (
                xp.sum(weights["ao"] * attraction)
                + xp.sum(
                    weights["aux"] * integrals.metric(auxiliary_centers, kernel=kernel)
                )
                + xp.sum(weights["three"] * three)
            )

        positions = xp.asarray(molecule.positions_array())
        gradient = np.asarray(jax.grad(function)(positions))
        host = np.asarray(positions)
        numeric = np.zeros_like(host)
        for atom in range(host.shape[0]):
            for axis in range(3):
                shift = np.zeros_like(host)
                shift[atom, axis] = STEP
                up = float(function(xp.asarray(host + shift)))
                down = float(function(xp.asarray(host - shift)))
                numeric[atom, axis] = (up - down) / (2.0 * STEP)
    np.testing.assert_allclose(gradient, numeric, atol=GRADIENT_FINITE_DIFFERENCE_TOL)


def test_traced_charges_are_rejected() -> None:
    jax = pytest.importorskip("jax")
    molecule = water()
    with use_backend("jax") as backend:
        xp = backend.namespace
        integrals, _, _ = setup(molecule, "sto-3g")
        positions = xp.asarray(molecule.positions_array())
        centers = integrals.orbital_centers(positions)

        def function(charges: Array) -> Array:
            return xp.sum(integrals.nuclear_attraction(centers, positions, charges))

        with pytest.raises(InputError, match="constants"):
            jax.grad(function)(xp.asarray(np.asarray(molecule.nuclear_charges)))

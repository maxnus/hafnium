"""Derivative integrals against finite differences of the integrals (ADR-0035)."""

from collections.abc import Callable

import numpy as np
import pytest

from hafnium.backend import Array
from hafnium.basis import BasisLibrary, BasisSet
from hafnium.integrals import (
    CoulombKernel,
    DensityFitting,
    DensityFittingDerivatives,
    PointCharges,
    RustEngine,
    ShellTable,
    kinetic,
    kinetic_derivative,
    nuclear_attraction,
    nuclear_attraction_derivative,
    overlap,
    overlap_derivative,
)
from hafnium.molecule import Molecule
from tests.molecules import ammonia_tilted, displaced, water
from tests.tolerances import FINITE_DIFFERENCE_TOL, INTEGRAL_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
STEP = 1e-5
AUXILIARY = "def2-universal-jkfit"


def basis_for(molecule: Molecule, name: str) -> BasisSet:
    return BasisSet.for_atoms(molecule, name, library=LIBRARY)


def central_difference(
    evaluate: Callable[[Molecule], Array], molecule: Molecule, atom: int, axis: int
) -> Array:
    """Central difference of ``evaluate`` under a move of ``atom`` along ``axis``."""
    plus = evaluate(displaced(molecule, atom, axis, STEP))
    minus = evaluate(displaced(molecule, atom, axis, -STEP))
    return (plus - minus) / (2 * STEP)


def on_atom(basis: BasisSet, atom: int) -> Array:
    """Mask of the functions of ``basis`` centred on ``atom``."""
    return np.array([label.atom == atom for label in basis.function_labels()])


def symmetric_total(bra_derivative: Array, mask: Array) -> Array:
    """Total derivative of a symmetric two-index integral under a move of the atom.

    The bra rows on the atom plus the ket columns, the transpose of the
    same rows.
    """
    total = np.zeros_like(bra_derivative)
    total[mask, :] += bra_derivative[mask, :]
    total[:, mask] += bra_derivative[mask, :].T
    return total


def three_center_total(
    derivatives: DensityFittingDerivatives, axis: int, aux_mask: Array, mask: Array
) -> Array:
    """Total derivative of ``(P|uv)`` under a move of the atom."""
    total = np.zeros_like(derivatives.auxiliary[axis])
    total[aux_mask] += derivatives.auxiliary[axis][aux_mask]
    total[:, mask, :] += derivatives.orbital[axis][:, mask, :]
    total[:, :, mask] += np.transpose(derivatives.orbital[axis][:, mask, :], (0, 2, 1))
    return total


@pytest.mark.parametrize(
    ("make", "basis_name"), [(water, "cc-pvdz"), (ammonia_tilted, "sto-3g")]
)
def test_one_electron_derivatives_match_finite_differences(
    make: Callable[[], Molecule], basis_name: str
) -> None:
    molecule = make()
    basis = basis_for(molecule, basis_name)
    s_x = overlap_derivative(ENGINE, basis, molecule)
    t_x = kinetic_derivative(ENGINE, basis, molecule)
    v_x, v_nuclei = nuclear_attraction_derivative(ENGINE, basis, molecule)
    assert s_x.shape == (3, basis.n_functions, basis.n_functions)
    assert v_nuclei.shape == (molecule.n_atoms, 3, basis.n_functions, basis.n_functions)
    integrals: list[Callable[[Molecule], Array]] = [
        lambda m: overlap(ENGINE, basis, m),
        lambda m: kinetic(ENGINE, basis, m),
        lambda m: nuclear_attraction(ENGINE, basis, m),
    ]
    for atom in range(molecule.n_atoms):
        mask = on_atom(basis, atom)
        for axis in range(3):
            totals = [
                symmetric_total(s_x[axis], mask),
                symmetric_total(t_x[axis], mask),
                symmetric_total(v_x[axis], mask) + v_nuclei[atom, axis],
            ]
            for analytic, evaluate in zip(totals, integrals, strict=True):
                numeric = central_difference(evaluate, molecule, atom, axis)
                np.testing.assert_allclose(
                    analytic, numeric, atol=FINITE_DIFFERENCE_TOL
                )


@pytest.mark.parametrize("omega", [None, 0.4])
def test_fitting_derivatives_match_finite_differences(omega: float | None) -> None:
    molecule = water()
    basis = basis_for(molecule, "cc-pvdz" if omega is None else "sto-3g")
    auxiliary = basis_for(molecule, AUXILIARY)
    derivatives = DensityFittingDerivatives.build(
        ENGINE, basis, auxiliary, molecule, omega=omega
    )
    assert derivatives.metric.shape == (3, auxiliary.n_functions, auxiliary.n_functions)
    for atom in range(molecule.n_atoms):
        mask = on_atom(basis, atom)
        aux_mask = on_atom(auxiliary, atom)
        for axis in range(3):
            fitting = central_difference(
                lambda m: (
                    DensityFitting.build(
                        ENGINE, basis, auxiliary, m, omega=omega
                    ).three_center
                ),
                molecule,
                atom,
                axis,
            )
            np.testing.assert_allclose(
                three_center_total(derivatives, axis, aux_mask, mask),
                fitting,
                atol=FINITE_DIFFERENCE_TOL,
            )
            metric = central_difference(
                lambda m: (
                    DensityFitting.build(
                        ENGINE, basis, auxiliary, m, omega=omega
                    ).metric
                ),
                molecule,
                atom,
                axis,
            )
            np.testing.assert_allclose(
                symmetric_total(derivatives.metric[axis], aux_mask),
                metric,
                atol=FINITE_DIFFERENCE_TOL,
            )


def test_long_range_attraction_derivatives_match_finite_differences() -> None:
    molecule = ammonia_tilted()
    basis = basis_for(molecule, "sto-3g")
    omega = 0.6
    v_x, v_nuclei = nuclear_attraction_derivative(ENGINE, basis, molecule, omega=omega)
    for atom in range(molecule.n_atoms):
        mask = on_atom(basis, atom)
        for axis in range(3):
            numeric = central_difference(
                lambda m: nuclear_attraction(ENGINE, basis, m, omega=omega),
                molecule,
                atom,
                axis,
            )
            np.testing.assert_allclose(
                symmetric_total(v_x[axis], mask) + v_nuclei[atom, axis],
                numeric,
                atol=FINITE_DIFFERENCE_TOL,
            )


def test_derivatives_are_invariant_under_translation() -> None:
    """Moving every atom together leaves every integral unchanged."""
    molecule = water()
    basis = basis_for(molecule, "def2-svp")
    auxiliary = basis_for(molecule, AUXILIARY)
    derivatives = DensityFittingDerivatives.build(ENGINE, basis, auxiliary, molecule)
    v_x, v_nuclei = nuclear_attraction_derivative(ENGINE, basis, molecule)
    everything = np.ones(basis.n_functions, dtype=bool)
    every_aux = np.ones(auxiliary.n_functions, dtype=bool)
    for axis in range(3):
        for bra in (
            overlap_derivative(ENGINE, basis, molecule)[axis],
            kinetic_derivative(ENGINE, basis, molecule)[axis],
        ):
            np.testing.assert_allclose(
                symmetric_total(bra, everything), 0.0, atol=INTEGRAL_TOL
            )
        np.testing.assert_allclose(
            symmetric_total(v_x[axis], everything) + np.sum(v_nuclei[:, axis], axis=0),
            0.0,
            atol=INTEGRAL_TOL,
        )
        np.testing.assert_allclose(
            symmetric_total(derivatives.metric[axis], every_aux), 0.0, atol=INTEGRAL_TOL
        )
        np.testing.assert_allclose(
            three_center_total(derivatives, axis, every_aux, everything),
            0.0,
            atol=INTEGRAL_TOL,
        )


def test_engine_derivatives_translate_the_ket() -> None:
    """A translation of the ket equals the ket table built on the moved molecule."""
    molecule = water()
    basis = basis_for(molecule, "sto-3g")
    auxiliary = basis_for(molecule, AUXILIARY)
    shift = np.array([0.3, -0.2, 0.5])
    moved = molecule
    for atom in range(molecule.n_atoms):
        for axis in range(3):
            moved = displaced(moved, atom, axis, float(shift[axis]))
    table = ShellTable.from_basis(basis, molecule)
    moved_table = ShellTable.from_basis(basis, moved)
    aux_table = ShellTable.from_basis(auxiliary, molecule)
    zero = np.zeros((1, 3))
    both = np.stack([np.zeros(3), shift])
    translated = ENGINE.overlap_derivative(table, table, both)
    direct = ENGINE.overlap_derivative(table, moved_table, zero)
    np.testing.assert_allclose(translated[1], direct[0], atol=INTEGRAL_TOL)
    charges = PointCharges.from_system(molecule)
    kernel = CoulombKernel(0.5)
    translated_v, translated_nuclei = ENGINE.nuclear_attraction_derivative(
        table, table, both, charges, kernel=kernel
    )
    direct_v, direct_nuclei = ENGINE.nuclear_attraction_derivative(
        table, moved_table, zero, charges, kernel=kernel
    )
    np.testing.assert_allclose(translated_v[1], direct_v[0], atol=INTEGRAL_TOL)
    np.testing.assert_allclose(
        translated_nuclei[1], direct_nuclei[0], atol=INTEGRAL_TOL
    )
    pairs = np.stack([np.zeros((2, 3)), np.stack([np.zeros(3), shift])])
    translated_aux, translated_bra = ENGINE.coulomb_three_center_derivative(
        aux_table, table, table, pairs
    )
    direct_aux, direct_bra = ENGINE.coulomb_three_center_derivative(
        aux_table, table, moved_table, np.zeros((1, 2, 3))
    )
    np.testing.assert_allclose(translated_aux[1], direct_aux[0], atol=INTEGRAL_TOL)
    np.testing.assert_allclose(translated_bra[1], direct_bra[0], atol=INTEGRAL_TOL)

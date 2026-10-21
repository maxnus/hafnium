"""Two- and three-centre Coulomb integrals and the fitted tensor."""

import math

import numpy as np
import pytest

from hafnium.basis import BasisLibrary, BasisSet, Shell
from hafnium.integrals import DensityFitting, RustEngine, ShellTable
from hafnium.molecule import Molecule
from tests.tolerances import DENSITY_FITTING_TOL, INTEGRAL_TOL, SPECTRUM_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
ZERO = np.zeros((1, 3))
ZERO_PAIR = np.zeros((1, 2, 3))


def water() -> Molecule:
    return Molecule.from_atoms(
        [
            ("O", (0.0, 0.0, 0.1173)),
            ("H", (0.0, 0.7572, -0.4692)),
            ("H", (0.0, -0.7572, -0.4692)),
        ]
    )


def tables(
    molecule: Molecule, orbital: str, auxiliary: str
) -> tuple[ShellTable, ShellTable]:
    positions = molecule.positions_array()
    basis = BasisSet.for_molecule(molecule, orbital, library=LIBRARY)
    aux = BasisSet.for_molecule(molecule, auxiliary, library=LIBRARY)
    return ShellTable.from_basis(basis, positions), ShellTable.from_basis(
        aux, positions
    )


def test_two_center_s_integral_matches_the_closed_form() -> None:
    a, b, distance = 0.9, 0.4, 1.3
    molecule = Molecule.from_atoms(
        [("He", (0.0, 0.0, 0.0)), ("He", (0.0, 0.0, distance))], unit="bohr"
    )
    basis = BasisSet(
        "s",
        ("He", "He"),
        (Shell(0, (a,), ((1.0,),)), Shell(0, (b,), ((1.0,),))),
        (0, 1),
    )
    table = ShellTable.from_basis(basis, molecule.positions_array())
    metric = ENGINE.coulomb_two_center(table, table, ZERO)[0]
    alpha = a * b / (a + b)
    boys = float(
        np.sqrt(np.pi / (4 * alpha * distance**2)) * math.erf(np.sqrt(alpha) * distance)
    )
    norms = (2 * a / np.pi) ** 0.75 * (2 * b / np.pi) ** 0.75
    expected = norms * 2 * np.pi**2.5 / (a * b * np.sqrt(a + b)) * boys
    assert metric[0, 1] == pytest.approx(expected, abs=INTEGRAL_TOL)
    assert metric[1, 0] == pytest.approx(expected, abs=INTEGRAL_TOL)


def test_metric_is_symmetric_positive_and_rotation_invariant() -> None:
    molecule = water()
    _, aux = tables(molecule, "sto-3g", "def2-universal-jkfit")
    metric = ENGINE.coulomb_two_center(aux, aux, ZERO)[0]
    np.testing.assert_allclose(metric, metric.T, atol=INTEGRAL_TOL)
    eigenvalues = np.linalg.eigvalsh(metric)
    assert eigenvalues.min() > 0.0
    rotation, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((3, 3)))
    rotated = Molecule.from_atoms(
        zip(
            molecule.symbols,
            (np.asarray(molecule.positions) @ rotation.T).tolist(),
            strict=True,
        ),
        unit="bohr",
    )
    _, aux_rotated = tables(rotated, "sto-3g", "def2-universal-jkfit")
    metric_rotated = ENGINE.coulomb_two_center(aux_rotated, aux_rotated, ZERO)[0]
    np.testing.assert_allclose(
        eigenvalues, np.linalg.eigvalsh(metric_rotated), atol=SPECTRUM_TOL
    )


def test_three_center_is_symmetric_in_the_orbital_pair() -> None:
    molecule = water()
    orbital, aux = tables(molecule, "def2-svp", "def2-universal-jkfit")
    three = ENGINE.coulomb_three_center(aux, orbital, orbital, ZERO_PAIR)[0]
    assert three.shape == (aux.n_functions, orbital.n_functions, orbital.n_functions)
    np.testing.assert_allclose(three, np.swapaxes(three, 1, 2), atol=INTEGRAL_TOL)


def test_three_center_with_s_auxiliary_is_an_attenuated_attraction() -> None:
    # A unit-norm s auxiliary function of exponent q is the unit Gaussian
    # charge divided by (q / 2π)^{3/4}, whose potential is erf(√q r) / r.
    molecule = water()
    basis = BasisSet.for_molecule(molecule, "cc-pvdz", library=LIBRARY)
    q = 0.6
    aux_basis = BasisSet("s", ("O",), (Shell(0, (q,), ((1.0,),)),), (0,))
    probe = Molecule.from_atoms([("O", molecule.positions[0])], unit="bohr")
    orbital = ShellTable.from_basis(basis, molecule.positions_array())
    aux = ShellTable.from_basis(aux_basis, probe.positions_array())
    three = ENGINE.coulomb_three_center(aux, orbital, orbital, ZERO_PAIR)[0, 0]
    attraction = ENGINE.nuclear_attraction(
        orbital,
        orbital,
        ZERO,
        np.array([1.0]),
        probe.positions_array(),
        omega=float(np.sqrt(q)),
    )[0]
    np.testing.assert_allclose(
        three, -attraction / (q / (2 * np.pi)) ** 0.75, atol=INTEGRAL_TOL
    )


def test_translations_move_the_auxiliary_and_the_ket() -> None:
    molecule = water()
    orbital, aux = tables(molecule, "sto-3g", "def2-universal-jkfit")
    shift = np.array([0.7, -0.2, 0.4])
    moved = Molecule.from_atoms(
        zip(
            molecule.symbols,
            (np.asarray(molecule.positions) + shift).tolist(),
            strict=True,
        ),
        unit="bohr",
    )
    orbital_moved, aux_moved = tables(moved, "sto-3g", "def2-universal-jkfit")
    translations = np.array([[shift, np.zeros(3)], [np.zeros(3), shift]])
    both = ENGINE.coulomb_three_center(aux, orbital, orbital, translations)
    np.testing.assert_allclose(
        both[0],
        ENGINE.coulomb_three_center(aux_moved, orbital, orbital, ZERO_PAIR)[0],
        atol=INTEGRAL_TOL,
    )
    np.testing.assert_allclose(
        both[1],
        ENGINE.coulomb_three_center(aux, orbital, orbital_moved, ZERO_PAIR)[0],
        atol=INTEGRAL_TOL,
    )
    two = ENGINE.coulomb_two_center(aux, aux, shift[None, :])[0]
    np.testing.assert_allclose(
        two, ENGINE.coulomb_two_center(aux, aux_moved, ZERO)[0], atol=INTEGRAL_TOL
    )


def test_attenuated_integrals_are_bounded_by_the_full_kernel() -> None:
    molecule = water()
    orbital, aux = tables(molecule, "sto-3g", "def2-universal-jkfit")
    full = ENGINE.coulomb_two_center(aux, aux, ZERO)[0]
    long_range = ENGINE.coulomb_two_center(aux, aux, ZERO, omega=0.5)[0]
    nearly = ENGINE.coulomb_two_center(aux, aux, ZERO, omega=1e5)[0]
    assert np.all(np.diag(long_range) < np.diag(full))
    assert np.all(np.diag(long_range) > 0.0)
    np.testing.assert_allclose(nearly, full, atol=1e-8)
    three_full = ENGINE.coulomb_three_center(aux, orbital, orbital, ZERO_PAIR)[0]
    three_long = ENGINE.coulomb_three_center(
        aux, orbital, orbital, ZERO_PAIR, omega=0.5
    )[0]
    assert np.abs(three_long).max() < np.abs(three_full).max()


def test_fitted_tensor_reproduces_the_fitted_repulsion() -> None:
    molecule = water()
    basis = BasisSet.for_molecule(molecule, "sto-3g", library=LIBRARY)
    aux = BasisSet.for_molecule(molecule, "def2-universal-jkfit", library=LIBRARY)
    fitting = DensityFitting.build(ENGINE, basis, aux, molecule)
    assert fitting.n_aux == aux.n_functions
    assert fitting.n_ao == basis.n_functions
    fitted = fitting.fitted_tensor()
    # The metric is absorbed: Σ_P B B equals (uv|P) M^{-1} (Q|wx).
    three = fitting.three_center.reshape(fitting.n_aux, -1)
    expected = three.T @ np.linalg.solve(fitting.metric, three)
    eri = fitting.electron_repulsion()
    np.testing.assert_allclose(
        eri.reshape(expected.shape), expected, atol=DENSITY_FITTING_TOL
    )
    np.testing.assert_allclose(fitted, np.swapaxes(fitted, 1, 2), atol=INTEGRAL_TOL)
    # Eight-fold permutational symmetry and positivity of the diagonal.
    np.testing.assert_allclose(eri, np.transpose(eri, (2, 3, 0, 1)), atol=INTEGRAL_TOL)
    assert all(eri[u, u, u, u] > 0.0 for u in range(fitting.n_ao))


def test_density_fitting_validates_shapes() -> None:
    from hafnium.errors import InputError

    with pytest.raises(InputError, match="square"):
        DensityFitting(np.zeros((2, 3)), np.zeros((2, 1, 1)))
    with pytest.raises(InputError, match=r"\[aux, ao, ao\]"):
        DensityFitting(np.eye(2), np.zeros((3, 1, 1)))
    orbital, aux = tables(water(), "sto-3g", "def2-universal-jkfit")
    with pytest.raises(ValueError, match=r"\[n, 2, 3\]"):
        ENGINE.coulomb_three_center(aux, orbital, orbital, np.zeros((1, 3, 2)))

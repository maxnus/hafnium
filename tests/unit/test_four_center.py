"""Four-centre integrals: closed forms, symmetry paths, screening and bounds."""

import numpy as np
import pytest

from hafnium.basis import BasisLibrary, BasisSet
from hafnium.integrals import DensityFitting, RustEngine, ShellTable, eri
from hafnium.molecule import Molecule
from tests.shell_tables import reversed_table
from tests.tolerances import FITTED_INTEGRAL_TOL, INTEGRAL_TOL, LITERATURE_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
ZERO = np.zeros((1, 3))
ZERO_TRIPLE = np.zeros((1, 3, 3))
# Screening threshold of the tests that check what screening neglects.
SCREENING_THRESHOLD = 1e-8
# All eight index permutations of (uv|wx) except the identity.
PERMUTATIONS = [
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
]


def water() -> Molecule:
    return Molecule.from_atoms(
        [
            ("O", (0.0, 0.0, 0.1173)),
            ("H", (0.0, 0.7572, -0.4692)),
            ("H", (0.0, -0.7572, -0.4692)),
        ]
    )


def basis_for(molecule: Molecule, name: str) -> BasisSet:
    return BasisSet.for_molecule(molecule, name, library=LIBRARY)


def shifted(molecule: Molecule, shift: np.ndarray) -> Molecule:
    positions = np.asarray(molecule.positions) + shift
    return Molecule.from_atoms(
        zip(molecule.symbols, positions.tolist(), strict=True), unit="bohr"
    )


def test_h2_sto3g_matches_szabo_and_ostlund() -> None:
    # Szabo and Ostlund, Modern Quantum Chemistry, section 3.5.2 (R = 1.4 Bohr).
    molecule = Molecule.from_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))], unit="bohr"
    )
    integrals = eri(ENGINE, basis_for(molecule, "sto-3g"), molecule)
    assert integrals.shape == (2, 2, 2, 2)
    assert integrals[0, 0, 0, 0] == pytest.approx(0.7746, abs=LITERATURE_TOL)
    assert integrals[0, 0, 1, 1] == pytest.approx(0.5697, abs=LITERATURE_TOL)
    assert integrals[1, 0, 0, 0] == pytest.approx(0.4441, abs=LITERATURE_TOL)
    assert integrals[1, 0, 1, 0] == pytest.approx(0.2970, abs=LITERATURE_TOL)


def test_eri_has_eightfold_symmetry_and_a_positive_diagonal() -> None:
    molecule = water()
    integrals = eri(ENGINE, basis_for(molecule, "def2-svp"), molecule)
    for permutation in PERMUTATIONS:
        np.testing.assert_allclose(
            integrals, np.transpose(integrals, permutation), atol=INTEGRAL_TOL
        )
    assert np.all(np.einsum("uvuv->uv", integrals) >= 0.0)
    assert np.all(np.einsum("uuuu->u", integrals) > 0.0)


def test_symmetry_paths_match_the_generic_path() -> None:
    # Equal tables take the symmetric shortcuts; a reversed copy of the
    # table holds the same functions and forces the generic evaluation.
    molecule = water()
    basis = basis_for(molecule, "def2-svp")
    table = ShellTable.from_basis(basis, molecule.positions_array())
    reordered, permutation = reversed_table(table)
    full = eri(ENGINE, basis, molecule, threshold=0.0)

    def mixed(bra: tuple[ShellTable, ShellTable], ket: tuple[ShellTable, ShellTable]):
        return ENGINE.coulomb_four_center(bra, ket, ZERO_TRIPLE)[0]

    # (a b'|a b'): only the pair swap applies.
    np.testing.assert_allclose(
        mixed((table, reordered), (table, reordered)),
        full[:, permutation][:, :, :, permutation],
        atol=INTEGRAL_TOL,
    )
    # (a a|b' b'): the bra and the ket swap apply, the pair swap does not.
    np.testing.assert_allclose(
        mixed((table, table), (reordered, reordered)),
        full[:, :, permutation][:, :, :, permutation],
        atol=INTEGRAL_TOL,
    )
    # (a b'|b' a): no symmetry at all.
    np.testing.assert_allclose(
        mixed((table, reordered), (reordered, table)),
        full[:, permutation][:, :, permutation],
        atol=INTEGRAL_TOL,
    )


def test_translations_move_three_of_the_four_functions() -> None:
    molecule = water()
    basis = basis_for(molecule, "sto-3g")
    table = ShellTable.from_basis(basis, molecule.positions_array())
    shifts = np.array([[0.7, -0.2, 0.4], [-0.3, 0.5, 0.1], [0.2, 0.2, -0.6]])
    moved = [
        ShellTable.from_basis(basis, shifted(molecule, shift).positions_array())
        for shift in shifts
    ]
    translated = ENGINE.coulomb_four_center(
        (table, table), (table, table), np.concatenate([ZERO_TRIPLE, shifts[None]])
    )
    direct = ENGINE.coulomb_four_center(
        (table, moved[0]), (moved[1], moved[2]), ZERO_TRIPLE
    )[0]
    np.testing.assert_allclose(translated[1], direct, atol=INTEGRAL_TOL)
    np.testing.assert_allclose(
        translated[0], eri(ENGINE, basis, molecule), atol=INTEGRAL_TOL
    )


def test_screening_neglects_only_quartets_below_the_threshold() -> None:
    molecule = Molecule.from_atoms(
        [
            ("H", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.4)),
            ("H", (12.0, 0.0, 0.0)),
            ("H", (12.0, 0.0, 1.4)),
        ],
        unit="bohr",
    )
    basis = basis_for(molecule, "def2-svp")
    exact = eri(ENGINE, basis, molecule, threshold=0.0)
    screened = eri(ENGINE, basis, molecule, threshold=SCREENING_THRESHOLD)
    assert np.count_nonzero(screened == 0.0) > np.count_nonzero(exact == 0.0)
    assert np.all((screened == exact) | (screened == 0.0))
    assert np.abs(exact - screened).max() < SCREENING_THRESHOLD
    # The default threshold changes nothing at the integral tolerance.
    np.testing.assert_allclose(eri(ENGINE, basis, molecule), exact, atol=INTEGRAL_TOL)


def test_schwarz_bounds_bound_every_quartet() -> None:
    molecule = water()
    basis = basis_for(molecule, "def2-svp")
    table = ShellTable.from_basis(basis, molecule.positions_array())
    bounds = ENGINE.schwarz_bounds(table, table, ZERO)[0]
    assert bounds.shape == (table.n_shells, table.n_shells)
    integrals = eri(ENGINE, basis, molecule, threshold=0.0)
    sizes = 2 * np.asarray(table.angular_momenta) + 1
    shell_of = np.repeat(np.arange(table.n_shells), sizes)
    pair_bound = bounds[shell_of[:, None], shell_of[None, :]]
    product = pair_bound[:, :, None, None] * pair_bound[None, None, :, :]
    assert np.all(np.abs(integrals) <= product + INTEGRAL_TOL)
    # The bound of a shell pair is attained by one of its own diagonal integrals.
    diagonal = np.sqrt(np.einsum("uvuv->uv", integrals))
    attained = np.zeros_like(bounds)
    for u in range(table.n_functions):
        for v in range(table.n_functions):
            index = shell_of[u], shell_of[v]
            attained[index] = max(attained[index], diagonal[u, v])
    np.testing.assert_allclose(bounds, attained, atol=INTEGRAL_TOL)
    long_range = ENGINE.schwarz_bounds(table, table, ZERO, omega=0.5)[0]
    assert np.all(long_range <= bounds + INTEGRAL_TOL)


def test_attenuated_eri_is_bounded_by_the_full_kernel() -> None:
    molecule = water()
    basis = basis_for(molecule, "sto-3g")
    full = eri(ENGINE, basis, molecule)
    long_range = eri(ENGINE, basis, molecule, omega=0.5)
    nearly = eri(ENGINE, basis, molecule, omega=1e5)
    diagonal_full = np.einsum("uuuu->u", full)
    diagonal_long = np.einsum("uuuu->u", long_range)
    assert np.all(diagonal_long < diagonal_full)
    assert np.all(diagonal_long > 0.0)
    np.testing.assert_allclose(nearly, full, atol=1e-8)


def test_density_fitting_approximates_the_exact_integrals() -> None:
    molecule = water()
    basis = basis_for(molecule, "sto-3g")
    auxiliary = basis_for(molecule, "def2-universal-jkfit")
    fitted = DensityFitting.build(ENGINE, basis, auxiliary, molecule).eri()
    exact = eri(ENGINE, basis, molecule)
    assert np.abs(fitted - exact).max() < FITTED_INTEGRAL_TOL


def test_four_center_rejects_bad_arguments() -> None:
    molecule = water()
    basis = basis_for(molecule, "sto-3g")
    table = ShellTable.from_basis(basis, molecule.positions_array())
    with pytest.raises(ValueError, match="threshold"):
        eri(ENGINE, basis, molecule, threshold=-1.0)
    with pytest.raises(ValueError, match="positive"):
        eri(ENGINE, basis, molecule, omega=0.0)
    with pytest.raises(ValueError, match=r"\[n, 3, 3\]"):
        ENGINE.coulomb_four_center((table, table), (table, table), np.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match=r"\[n, 3\]"):
        ENGINE.schwarz_bounds(table, table, np.zeros((1, 2)))

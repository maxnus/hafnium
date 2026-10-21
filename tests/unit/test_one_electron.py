"""One-electron integrals: closed forms, invariances and literature values."""

import numpy as np
import pytest

from hafnium import _kernels
from hafnium.basis import BasisLibrary, BasisSet, Shell
from hafnium.errors import InputError
from hafnium.integrals import (
    RustEngine,
    ShellTable,
    core_hamiltonian,
    kinetic,
    nuclear_attraction,
    overlap,
)
from hafnium.molecule import Molecule
from tests.molecules import hydrogen, water
from tests.tolerances import (
    ATTENUATION_LIMIT_TOL,
    INTEGRAL_TOL,
    LITERATURE_TOL,
    SPECTRUM_TOL,
)

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()


def basis_for(molecule: Molecule, name: str) -> BasisSet:
    return BasisSet.for_atoms(molecule, name, library=LIBRARY)


def hydrogen_molecule() -> tuple[Molecule, BasisSet]:
    molecule = hydrogen()
    return molecule, basis_for(molecule, "sto-3g")


def test_h2_sto3g_matches_szabo_and_ostlund() -> None:
    # Szabo and Ostlund, Modern Quantum Chemistry, table 3.5 (R = 1.4 Bohr).
    molecule, basis = hydrogen_molecule()
    s = overlap(ENGINE, basis, molecule)
    t = kinetic(ENGINE, basis, molecule)
    v = nuclear_attraction(ENGINE, basis, molecule)
    assert s[0, 1] == pytest.approx(0.6593, abs=LITERATURE_TOL)
    assert t[0, 0] == pytest.approx(0.7600, abs=LITERATURE_TOL)
    assert t[0, 1] == pytest.approx(0.2365, abs=LITERATURE_TOL)
    # V = V_A + V_B with V_A,11 = -1.2266, V_A,12 = -0.5974, V_A,22 = -0.6538.
    assert v[0, 0] == pytest.approx(-1.2266 - 0.6538, abs=LITERATURE_TOL)
    assert v[0, 1] == pytest.approx(2 * -0.5974, abs=LITERATURE_TOL)
    h = core_hamiltonian(ENGINE, basis, molecule)
    np.testing.assert_allclose(h, t + v, atol=INTEGRAL_TOL)


@pytest.mark.parametrize(
    "name", ["sto-3g", "def2-svp", "cc-pvdz", "def2-universal-jkfit"]
)
def test_contracted_functions_are_normalized(name: str) -> None:
    molecule = water()
    basis = basis_for(molecule, name)
    s = overlap(ENGINE, basis, molecule)
    np.testing.assert_allclose(np.diag(s), 1.0, atol=INTEGRAL_TOL)
    np.testing.assert_allclose(s, s.T, atol=INTEGRAL_TOL)
    assert s.shape == (basis.n_functions, basis.n_functions)


def test_matrices_are_symmetric_and_positive() -> None:
    molecule = water()
    basis = basis_for(molecule, "def2-universal-jkfit")
    for matrix in (
        kinetic(ENGINE, basis, molecule),
        nuclear_attraction(ENGINE, basis, molecule),
    ):
        np.testing.assert_allclose(matrix, matrix.T, atol=INTEGRAL_TOL)
    s = overlap(ENGINE, basis, molecule)
    t = kinetic(ENGINE, basis, molecule)
    assert np.linalg.eigvalsh(s).min() > 0.0
    assert np.linalg.eigvalsh(t).min() > 0.0


def test_integrals_are_invariant_under_translation_and_rotation() -> None:
    molecule = water()
    basis = basis_for(molecule, "cc-pvdz")
    reference = [
        f(ENGINE, basis, molecule) for f in (overlap, kinetic, nuclear_attraction)
    ]
    rng = np.random.default_rng(7)
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    shift = np.array([1.3, -0.4, 2.2])
    positions = np.asarray(molecule.positions) @ rotation.T + shift
    moved = Molecule.from_atoms(
        zip(molecule.symbols, positions.tolist(), strict=True), unit="bohr"
    )
    transformed = [
        f(ENGINE, basis, moved) for f in (overlap, kinetic, nuclear_attraction)
    ]
    # Functions with l > 0 rotate among themselves, so compare the spectra of
    # the generalized eigenvalue problems, which are invariant.
    s_ref, s_new = reference[0], transformed[0]
    np.testing.assert_allclose(
        np.linalg.eigvalsh(s_ref), np.linalg.eigvalsh(s_new), atol=SPECTRUM_TOL
    )
    for before, after in zip(reference[1:], transformed[1:], strict=True):
        np.testing.assert_allclose(
            generalized_eigenvalues(before, s_ref),
            generalized_eigenvalues(after, s_new),
            atol=SPECTRUM_TOL,
        )


def generalized_eigenvalues(a: np.ndarray, s: np.ndarray) -> np.ndarray:
    lower = np.linalg.cholesky(s)
    inverse = np.linalg.inv(lower)
    return np.linalg.eigvalsh(inverse @ a @ inverse.T)


def test_images_translate_the_ket() -> None:
    molecule, basis = hydrogen_molecule()
    table = ShellTable.from_basis(basis, molecule.positions_array())
    translations = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    images = ENGINE.overlap(table, table, translations)
    assert images.shape == (2, 2, 2)
    moved = Molecule.from_atoms(
        [("H", (0.0, 0.0, 3.0)), ("H", (0.0, 0.0, 4.4))], unit="bohr"
    )
    moved_table = ShellTable.from_basis(basis, moved.positions_array())
    direct = ENGINE.overlap(table, moved_table, translations[:1])
    np.testing.assert_allclose(images[1], direct[0], atol=INTEGRAL_TOL)
    np.testing.assert_allclose(
        images[0], overlap(ENGINE, basis, molecule), atol=INTEGRAL_TOL
    )


def test_attenuated_attraction_is_bounded_by_the_full_kernel() -> None:
    molecule = water()
    basis = basis_for(molecule, "def2-svp")
    full = nuclear_attraction(ENGINE, basis, molecule)
    long_range = nuclear_attraction(ENGINE, basis, molecule, omega=0.4)
    nearly_full = nuclear_attraction(ENGINE, basis, molecule, omega=1e5)
    assert np.all(np.diag(long_range) > np.diag(full))
    np.testing.assert_allclose(nearly_full, full, atol=ATTENUATION_LIMIT_TOL)


def test_spherical_transformation_rows_are_orthonormal_on_the_sphere() -> None:
    # Overlap of the l = 3 functions of a single primitive is the identity.
    molecule = Molecule.from_atoms([("He", (0.0, 0.0, 0.0))])
    shell = Shell(3, (0.8,), ((1.0,),))
    basis = BasisSet("f", ("He",), (shell,), (0,))
    s = overlap(ENGINE, basis, molecule)
    np.testing.assert_allclose(s, np.eye(7), atol=INTEGRAL_TOL)
    matrix = _kernels.spherical_transformation(3)
    assert matrix.shape == (7, 10)


def test_shell_table_validates_its_input() -> None:
    with pytest.raises(InputError, match="centre"):
        ShellTable(
            np.array([0]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([0, 1]),
            np.array([1.0]),
            np.array([1.0]),
        )
    molecule, basis = hydrogen_molecule()
    with pytest.raises(InputError, match="positions"):
        ShellTable.from_basis(basis, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="positive"):
        nuclear_attraction(ENGINE, basis, molecule, omega=-1.0)


def test_binding_rejects_malformed_input() -> None:
    molecule, basis = hydrogen_molecule()
    table = ShellTable.from_basis(basis, molecule.positions_array())
    with pytest.raises(ValueError, match=r"shape \[n, 3\]"):
        ENGINE.overlap(table, table, np.zeros((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        ENGINE.overlap(table, table, np.array([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        ENGINE.nuclear_attraction(
            table, table, np.zeros((1, 3)), np.array([1.0]), np.array([[np.inf, 0, 0]])
        )
    too_high = ShellTable(
        np.array([17]), np.zeros((1, 3)), np.array([0, 1]), np.ones(1), np.ones(1)
    )
    with pytest.raises(ValueError, match="maximum"):
        ENGINE.overlap(too_high, table, np.zeros((1, 3)))
    with pytest.raises(InputError, match="integers"):
        ShellTable(
            np.array([1.5]), np.zeros((1, 3)), np.array([0, 1]), np.ones(1), np.ones(1)
        )

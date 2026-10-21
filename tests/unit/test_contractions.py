"""General contractions: one table entry per shell, same integrals as split shells."""

import numpy as np

from hafnium.basis import BasisSet
from hafnium.integrals import RustEngine
from hafnium.molecule import Molecule
from tests.molecules import water
from tests.shell_tables import LIBRARY, split_table, table_for
from tests.tolerances import INTEGRAL_TOL

ENGINE = RustEngine()
ZERO = np.zeros((1, 3))


def test_from_basis_keeps_general_contractions_together() -> None:
    molecule = water()
    basis = BasisSet.for_molecule(molecule, "cc-pvdz", library=LIBRARY)
    table = table_for(molecule, "cc-pvdz")
    assert table.n_contractions is not None
    assert table.n_shells == basis.n_shells
    assert table.n_functions == basis.n_functions
    assert [int(k) for k in table.n_contractions] == [
        shell.n_contractions for shell in basis.shells
    ]
    assert max(int(k) for k in table.n_contractions) > 1
    split = split_table(table)
    assert split.n_shells > table.n_shells
    assert split.n_functions == table.n_functions


def test_every_kernel_matches_the_split_table() -> None:
    molecule = water()
    table = table_for(molecule, "cc-pvdz")
    auxiliary = table_for(molecule, "def2-universal-jkfit")
    split = split_table(table)
    charges = np.asarray(molecule.nuclear_charges)
    positions = molecule.positions_array()
    g_vectors = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.5]])
    pairs = [
        (ENGINE.overlap(table, table, ZERO), ENGINE.overlap(split, split, ZERO)),
        (ENGINE.kinetic(table, split, ZERO), ENGINE.kinetic(split, split, ZERO)),
        (
            ENGINE.nuclear_attraction(
                table, table, ZERO, charges, positions, omega=0.5
            ),
            ENGINE.nuclear_attraction(
                split, split, ZERO, charges, positions, omega=0.5
            ),
        ),
        (
            ENGINE.coulomb_three_center(auxiliary, table, table, np.zeros((1, 2, 3))),
            ENGINE.coulomb_three_center(auxiliary, split, split, np.zeros((1, 2, 3))),
        ),
        (
            ENGINE.coulomb_four_center(
                (table, table), (table, table), np.zeros((1, 3, 3))
            ),
            ENGINE.coulomb_four_center(
                (split, split), (split, split), np.zeros((1, 3, 3))
            ),
        ),
        (
            ENGINE.fourier_transform(table, g_vectors),
            ENGINE.fourier_transform(split, g_vectors),
        ),
        (
            ENGINE.pair_fourier_transform(table, table, ZERO, g_vectors),
            ENGINE.pair_fourier_transform(split, split, ZERO, g_vectors),
        ),
    ]
    for general, reference in pairs:
        np.testing.assert_allclose(general, reference, atol=INTEGRAL_TOL)


def test_schwarz_bounds_cover_the_contractions_of_a_shell() -> None:
    molecule = Molecule.from_atoms([("O", (0.0, 0.0, 0.0))])
    table = table_for(molecule, "cc-pvdz")
    assert table.n_contractions is not None
    split = split_table(table)
    coarse = ENGINE.schwarz_bounds(table, table, ZERO)[0]
    fine = ENGINE.schwarz_bounds(split, split, ZERO)[0]
    members = np.repeat(np.arange(table.n_shells), table.n_contractions)
    for i in range(table.n_shells):
        for j in range(table.n_shells):
            expected = fine[np.ix_(members == i, members == j)].max()
            assert abs(coarse[i, j] - expected) < INTEGRAL_TOL

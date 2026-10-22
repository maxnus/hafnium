"""Coulomb and exchange matrices of structured densities."""

import numpy as np
import pytest

from hafnium.backend import adjoint
from hafnium.basis import BasisLibrary, BasisSet
from hafnium.errors import InputError
from hafnium.integrals import RustEngine
from hafnium.scf import (
    DensityFittedCoulombExchange,
    ExactCoulombExchange,
    coulomb_exchange_matrices,
)
from hafnium.structured import BlockMatrix, BlockStructure, Ring
from tests.molecules import water
from tests.random_matrices import random_hermitian
from tests.tolerances import FITTED_INTEGRAL_TOL, STRUCTURE_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
MOLECULE = water()
BASIS = BasisSet.for_atoms(MOLECULE, "sto-3g", library=LIBRARY)
AUXILIARY = BasisSet.for_atoms(MOLECULE, "def2-universal-jkfit", library=LIBRARY)
EXACT = ExactCoulombExchange.from_molecule(ENGINE, BASIS, MOLECULE)
FITTED = DensityFittedCoulombExchange.from_molecule(ENGINE, BASIS, AUXILIARY, MOLECULE)


def looped_coulomb_exchange(
    eri: np.ndarray, density: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The definitions of CONVENTIONS.md §3 written out as loops."""
    n = eri.shape[0]
    coulomb = np.zeros((n, n), dtype=density.dtype)
    exchange = np.zeros((n, n), dtype=density.dtype)
    for u in range(n):
        for v in range(n):
            for w in range(n):
                for x in range(n):
                    coulomb[u, v] += eri[u, v, w, x] * density[w, x]
                    exchange[u, v] += eri[u, w, x, v] * density[w, x]
    return coulomb, exchange


def test_exact_contractions_match_the_looped_definitions() -> None:
    density = random_hermitian(np.random.default_rng(1), EXACT.n_ao)
    coulomb, exchange = looped_coulomb_exchange(EXACT.eri, density)
    np.testing.assert_allclose(EXACT.coulomb(density), coulomb, atol=STRUCTURE_TOL)
    np.testing.assert_allclose(EXACT.exchange(density), exchange, atol=STRUCTURE_TOL)
    # J of a Hermitian density is real: the imaginary part is antisymmetric.
    np.testing.assert_allclose(np.imag(coulomb), 0.0, atol=STRUCTURE_TOL)


def test_fitted_contractions_approximate_the_exact_ones() -> None:
    density = random_hermitian(np.random.default_rng(2), EXACT.n_ao)
    np.testing.assert_allclose(
        FITTED.coulomb(density), EXACT.coulomb(density), atol=FITTED_INTEGRAL_TOL
    )
    np.testing.assert_allclose(
        FITTED.exchange(density), EXACT.exchange(density), atol=FITTED_INTEGRAL_TOL
    )
    assert FITTED.n_aux == AUXILIARY.n_functions


def test_collinear_structure_uses_the_total_density_for_coulomb() -> None:
    rng = np.random.default_rng(3)
    n = EXACT.n_ao
    alpha, beta = random_hermitian(rng, n), random_hermitian(rng, n)
    density = BlockMatrix(BlockStructure.unrestricted(Ring.COMPLEX), (alpha, beta))
    coulomb, exchange = coulomb_exchange_matrices(EXACT, density)
    total = EXACT.coulomb(np.real(alpha + beta))
    for block in coulomb.blocks:
        np.testing.assert_allclose(block, total, atol=STRUCTURE_TOL)
    np.testing.assert_allclose(exchange.blocks[0], EXACT.exchange(alpha))
    np.testing.assert_allclose(exchange.blocks[1], EXACT.exchange(beta))


def test_restricted_structure_doubles_the_stored_block() -> None:
    alpha = np.real(random_hermitian(np.random.default_rng(4), EXACT.n_ao))
    density = BlockMatrix(BlockStructure.restricted(), (alpha,))
    coulomb, exchange = coulomb_exchange_matrices(EXACT, density)
    np.testing.assert_allclose(coulomb.blocks[0], EXACT.coulomb(2 * alpha))
    np.testing.assert_allclose(exchange.blocks[0], EXACT.exchange(alpha))


def test_spinor_structure_contracts_every_spin_block() -> None:
    n = EXACT.n_ao
    spinor = random_hermitian(np.random.default_rng(5), 2 * n)
    density = BlockMatrix(BlockStructure.generalized(Ring.COMPLEX), (spinor,))
    coulomb, exchange = coulomb_exchange_matrices(EXACT, density)
    total = EXACT.coulomb(np.real(spinor[:n, :n] + spinor[n:, n:]))
    np.testing.assert_allclose(
        coulomb.blocks[0], np.kron(np.eye(2), total), atol=STRUCTURE_TOL
    )
    expected = np.block(
        [
            [EXACT.exchange(spinor[:n, :n]), EXACT.exchange(spinor[:n, n:])],
            [EXACT.exchange(spinor[n:, :n]), EXACT.exchange(spinor[n:, n:])],
        ]
    )
    np.testing.assert_allclose(exchange.blocks[0], expected, atol=STRUCTURE_TOL)
    # Both matrices are Hermitian for a Hermitian density.
    np.testing.assert_allclose(
        exchange.blocks[0], adjoint(exchange.blocks[0]), atol=STRUCTURE_TOL
    )


def test_malformed_inputs_are_rejected() -> None:
    with pytest.raises(InputError, match="ao, ao, ao, ao"):
        ExactCoulombExchange(np.zeros((2, 2, 2)))
    with pytest.raises(InputError, match="aux, ao, ao"):
        DensityFittedCoulombExchange(np.zeros((3, 2, 4)))
    density = BlockMatrix(BlockStructure.restricted(), (np.eye(EXACT.n_ao + 1),))
    with pytest.raises(InputError, match="atomic orbitals"):
        coulomb_exchange_matrices(EXACT, density)

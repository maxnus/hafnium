"""Fourier transforms of functions and pairs: closed forms, quadrature, identities."""

import numpy as np
import pytest

from hafnium import _kernels
from hafnium.basis import BasisLibrary, BasisSet, Shell
from hafnium.integrals import RustEngine, ShellTable
from hafnium.molecule import Molecule
from tests.molecules import water
from tests.tolerances import INTEGRAL_TOL, QUADRATURE_TOL, RECIPROCAL_SPACE_TOL
from tests.vectors import G_VECTORS

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
ZERO = np.zeros((1, 3))


def table_for(molecule: Molecule, name: str) -> ShellTable:
    basis = BasisSet.for_molecule(molecule, name, library=LIBRARY)
    return ShellTable.from_basis(basis, molecule.positions_array())


def spd_table() -> ShellTable:
    """s, p and d shells with exponents of at least 0.8 on two centres."""
    molecule = Molecule.from_atoms(
        [("He", (0.3, -0.2, 0.4)), ("He", (-0.6, 0.5, -0.9))], unit="bohr"
    )
    shells = (
        Shell(0, (1.4, 0.8), ((0.6, 0.5),)),
        Shell(1, (0.9,), ((1.0,),)),
        Shell(2, (1.1,), ((1.0,),)),
    )
    basis = BasisSet("spd", ("He", "He"), shells * 2, (0, 0, 0, 1, 1, 1))
    return ShellTable.from_basis(basis, molecule.positions_array())


def grid_values(
    table: ShellTable, half_width: float, spacing: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Functions on a cubic grid: values ``[ao, point]``, points and volume element.

    The spherical functions are built from Cartesian monomials with the
    kernel's own transformation matrix, so only the transform itself is
    under test.
    """
    axis = np.arange(-half_width, half_width + spacing / 2, spacing)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = points.reshape(-1, 3)
    offsets = table.primitive_offsets
    rows: list[np.ndarray] = []
    for shell in range(table.n_shells):
        l = int(table.angular_momenta[shell])
        relative = points - table.centers[shell]
        distance_squared = np.einsum("gi,gi->g", relative, relative)
        radial = np.zeros(len(points))
        for k in range(offsets[shell], offsets[shell + 1]):
            radial += table.coefficients[k] * np.exp(
                -table.exponents[k] * distance_squared
            )
        components = [
            (i, j, l - i - j) for i in range(l, -1, -1) for j in range(l - i, -1, -1)
        ]
        cartesian = np.stack(
            [
                relative[:, 0] ** i * relative[:, 1] ** j * relative[:, 2] ** k * radial
                for i, j, k in components
            ]
        )
        rows.append(_kernels.spherical_transformation(l) @ cartesian)
    return np.concatenate(rows), points, spacing**3


def test_s_transform_matches_the_closed_form() -> None:
    exponent, center = 0.8, np.array([0.1, -0.4, 0.7])
    molecule = Molecule.from_atoms([("He", tuple(center.tolist()))], unit="bohr")
    basis = BasisSet("s", ("He",), (Shell(0, (exponent,), ((1.0,),)),), (0,))
    table = ShellTable.from_basis(basis, molecule.positions_array())
    values = ENGINE.fourier_transform(table, G_VECTORS)[:, 0]
    g_squared = np.einsum("gi,gi->g", G_VECTORS, G_VECTORS)
    expected = (
        (2 * exponent / np.pi) ** 0.75
        * (np.pi / exponent) ** 1.5
        * np.exp(-g_squared / (4 * exponent))
        * np.exp(-1j * G_VECTORS @ center)
    )
    np.testing.assert_allclose(values, expected, atol=INTEGRAL_TOL)


def test_pair_transform_at_zero_is_the_overlap() -> None:
    table = table_for(water(), "def2-svp")
    translations = np.array([[0.0, 0.0, 0.0], [0.4, -0.3, 0.9]])
    pair = ENGINE.pair_fourier_transform(table, table, translations, ZERO)
    assert pair.shape == (2, 1, table.n_functions, table.n_functions)
    assert pair.dtype == np.complex128
    overlap = ENGINE.overlap(table, table, translations)
    np.testing.assert_allclose(pair[:, 0], overlap, atol=INTEGRAL_TOL)


def test_transforms_match_numerical_quadrature() -> None:
    table = spd_table()
    values, points, volume = grid_values(table, 6.5, 0.2)
    phases = np.exp(-1j * points @ G_VECTORS.T)
    single = ENGINE.fourier_transform(table, G_VECTORS)
    np.testing.assert_allclose(
        single, volume * (values @ phases).T, atol=QUADRATURE_TOL
    )
    pair = ENGINE.pair_fourier_transform(table, table, ZERO, G_VECTORS)[0]
    for ig in range(len(G_VECTORS)):
        expected = volume * (values * phases[:, ig]) @ values.T
        np.testing.assert_allclose(pair[ig], expected, atol=QUADRATURE_TOL)


def test_pair_transform_conjugates_and_translates() -> None:
    molecule = water()
    table = table_for(molecule, "sto-3g")
    g = G_VECTORS[4:5]
    translation = np.array([0.6, -0.2, 0.5])
    both = ENGINE.pair_fourier_transform(
        table, table, translation[None], np.concatenate([g, -g])
    )[0]
    np.testing.assert_allclose(both[0].conj(), both[1], atol=INTEGRAL_TOL)
    moved = Molecule.from_atoms(
        zip(
            molecule.symbols,
            (np.asarray(molecule.positions) + translation).tolist(),
            strict=True,
        ),
        unit="bohr",
    )
    direct = ENGINE.pair_fourier_transform(table, table_for(moved, "sto-3g"), ZERO, g)
    np.testing.assert_allclose(both[0], direct[0, 0], atol=INTEGRAL_TOL)
    # Swapping the pair: ρ̃_{u v_T}(G) = exp(−i G·T) ρ̃_{v u_{−T}}(G).
    swapped = ENGINE.pair_fourier_transform(table, table, -translation[None], g)[0, 0]
    phase = np.exp(-1j * g[0] @ translation)
    np.testing.assert_allclose(both[0], phase * swapped.T, atol=INTEGRAL_TOL)


def test_metric_from_the_transform_matches_the_coulomb_kernel() -> None:
    # For s functions with radial transforms f_P(G) at centres a distance R
    # apart, (P|Q) = (2/π) ∫ f_P f_Q sin(GR)/(GR) exp(−G²/4ω²) dG, which
    # ties the transform to the verified Coulomb kernel.
    molecule = Molecule.from_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))], unit="bohr"
    )
    table = table_for(molecule, "sto-3g")
    g = np.linspace(0.0, 12.0, 6001)
    g_vectors = np.stack([np.zeros_like(g), np.zeros_like(g), g], axis=1)
    heights = np.asarray(table.centers)[:, 2]
    transforms = ENGINE.fourier_transform(table, g_vectors)
    radial = transforms * np.exp(1j * g[:, None] * heights[None, :])
    assert np.abs(np.imag(radial)).max() < INTEGRAL_TOL
    for omega in (None, 0.5):
        metric = ENGINE.coulomb_two_center(table, table, ZERO, omega=omega)[0]
        attenuation = 1.0 if omega is None else np.exp(-(g**2) / (4 * omega**2))
        for p in range(2):
            for q in range(2):
                distance = abs(heights[p] - heights[q])
                integrand = np.real(radial[:, p]) * np.real(radial[:, q]) * attenuation
                integrand *= np.sinc(g * distance / np.pi)
                expected = 2 / np.pi * np.trapezoid(integrand, g)
                assert metric[p, q] == pytest.approx(expected, abs=RECIPROCAL_SPACE_TOL)


def test_fourier_transform_rejects_bad_arguments() -> None:
    table = table_for(water(), "sto-3g")
    with pytest.raises(ValueError, match=r"G vectors have shape \[n, 3\]"):
        ENGINE.fourier_transform(table, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        ENGINE.pair_fourier_transform(
            table, table, ZERO, np.array([[np.nan, 0.0, 0.0]])
        )

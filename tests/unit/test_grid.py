"""Grids: quadratures, the partition, the assembly and basis values on the points."""

import math

import numpy as np
import pytest

from hafnium.backend import Array
from hafnium.basis import BasisSet, cartesian_components, solid_harmonic_coefficients
from hafnium.errors import InputError
from hafnium.grid import (
    BRAGG_RADII,
    LEBEDEV_SIZES,
    GridSpec,
    MolecularGrid,
    atomic_grid,
    becke_partition,
    build_grid,
    lebedev_degree,
    lebedev_grid,
    mura_knowles,
    nwchem_pruning,
    treutler_adjustment,
    treutler_ahlrichs,
)
from hafnium.integrals import RustEngine, ShellTable
from hafnium.integrals.periodic import function_radius
from hafnium.molecule import Molecule
from hafnium.scf import hartree_fock
from tests.molecules import ammonia_tilted, water
from tests.shell_tables import LIBRARY
from tests.tolerances import (
    ANGULAR_QUADRATURE_TOL,
    BASIS_CUTOFF_PRECISION,
    BASIS_VALUE_TOL,
    FINITE_DIFFERENCE_TOL,
    GRID_TOL,
    PARTITION_TOL,
    RADIAL_QUADRATURE_TOL,
)

ENGINE = RustEngine()
FINITE_DIFFERENCE_STEP = 1e-5


def monomial_integral(a: int, b: int, c: int) -> float:
    """``∮ x^a y^b z^c dΩ`` over the unit sphere."""
    if a % 2 or b % 2 or c % 2:
        return 0.0
    g = math.gamma
    return (
        2.0 * g((a + 1) / 2) * g((b + 1) / 2) * g((c + 1) / 2) / g((a + b + c + 3) / 2)
    )


def test_lebedev_rules_integrate_polynomials_to_their_degree() -> None:
    rng = np.random.default_rng(0)
    for size in LEBEDEV_SIZES:
        points, weights = lebedev_grid(size)
        assert points.shape == (size, 3)
        assert weights.shape == (size,)
        assert float(weights.sum()) == pytest.approx(1.0, abs=ANGULAR_QUADRATURE_TOL)
        np.testing.assert_allclose(
            np.sum(points**2, axis=1), 1.0, atol=ANGULAR_QUADRATURE_TOL
        )
        degree = lebedev_degree(size)
        for _ in range(10):
            a, b, c = (int(e) for e in rng.multinomial(degree, [1 / 3] * 3))
            value = (
                4
                * math.pi
                * float(
                    np.sum(
                        weights
                        * points[:, 0] ** a
                        * points[:, 1] ** b
                        * points[:, 2] ** c
                    )
                )
            )
            assert value == pytest.approx(
                monomial_integral(a, b, c), abs=ANGULAR_QUADRATURE_TOL
            )
    with pytest.raises(InputError, match="no Lebedev rule"):
        lebedev_grid(7)


def test_lebedev_rules_have_octahedral_symmetry() -> None:
    points, weights = lebedev_grid(302)
    table = np.column_stack([points, weights])
    order = np.lexsort(np.round(table.T[::-1], 12))
    reference = table[order]
    for mapped in (
        table[:, [1, 0, 2, 3]],
        table[:, [2, 1, 0, 3]],
        table * np.array([-1.0, 1.0, -1.0, 1.0]),
    ):
        np.testing.assert_allclose(
            mapped[np.lexsort(np.round(mapped.T[::-1], 12))],
            reference,
            atol=ANGULAR_QUADRATURE_TOL,
        )


@pytest.mark.parametrize("alpha", [0.3, 5.0, 50.0])
def test_radial_quadratures_integrate_gaussians(alpha: float) -> None:
    exact = math.sqrt(math.pi) / (4.0 * alpha**1.5)
    for radii, weights in (treutler_ahlrichs(75, 0.9), mura_knowles(75)):
        assert np.all(np.diff(radii) > 0.0)
        assert float(radii[0]) > 0.0
        assert np.all(weights > 0.0)
        value = float(np.sum(weights * radii**2 * np.exp(-alpha * radii**2)))
        assert value == pytest.approx(exact, rel=RADIAL_QUADRATURE_TOL)
    with pytest.raises(InputError, match="at least one"):
        treutler_ahlrichs(0)
    with pytest.raises(InputError, match="positive"):
        treutler_ahlrichs(10, -1.0)
    with pytest.raises(InputError, match="positive"):
        mura_knowles(10, 0.0)


def test_becke_partition_sums_to_one_and_owns_the_nuclei() -> None:
    molecule = water()
    centers = molecule.positions_array()
    radii = np.array([BRAGG_RADII[atom.atomic_number] for atom in molecule.atoms])
    adjustments = treutler_adjustment(radii)
    np.testing.assert_allclose(adjustments, -adjustments.T, atol=PARTITION_TOL)
    assert np.all(np.abs(adjustments) <= 0.5)
    rng = np.random.default_rng(1)
    points = np.concatenate([rng.normal(scale=2.0, size=(200, 3)), centers])
    for adjust in (None, adjustments):
        weights = becke_partition(points, centers, adjust)
        assert weights.shape == (3, points.shape[0])
        assert np.all(weights >= 0.0)
        np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=PARTITION_TOL)
        np.testing.assert_allclose(weights[:, -3:], np.eye(3), atol=PARTITION_TOL)
    with pytest.raises(InputError, match="coincide"):
        becke_partition(points, np.zeros((2, 3)))
    with pytest.raises(InputError, match="atom, atom"):
        becke_partition(points, centers, np.zeros((2, 2)))
    with pytest.raises(InputError, match="positive"):
        treutler_adjustment(np.array([1.0, 0.0]))


def test_grid_spec_follows_the_size_tables() -> None:
    default = GridSpec()
    assert (default.radial_count(1), default.angular_count(1)) == (50, 302)
    assert (default.radial_count(8), default.angular_count(8)) == (75, 302)
    assert (default.radial_count(14), default.angular_count(14)) == (80, 434)
    coarse = GridSpec(level=0)
    assert (coarse.radial_count(1), coarse.angular_count(1)) == (10, 50)
    fixed = GridSpec(radial_points=40, angular_points=86)
    assert (fixed.radial_count(8), fixed.angular_count(8)) == (40, 86)
    radii, _ = treutler_ahlrichs(75, 0.9)
    pruned = nwchem_pruning(8, radii, 302)
    assert set(pruned) == {50, 86, 266, 302}
    assert pruned[-1] == 266
    assert nwchem_pruning(8, radii, 38) == (38,) * 75
    assert set(nwchem_pruning(1, radii, 50)) == {50, 74}
    points, weights = atomic_grid(8, default)
    assert points.shape == (sum(pruned), 3)
    assert weights.shape == (sum(pruned),)
    assert points.shape[0] < 75 * 302
    unpruned, _ = atomic_grid(8, GridSpec(prune=False))
    assert unpruned.shape == (75 * 302, 3)
    for bad in (
        {"level": 10},
        {"radial_points": 0},
        {"angular_points": 7},
    ):
        with pytest.raises(InputError):
            GridSpec(**bad)  # type: ignore[arg-type]


def electron_count(molecule: Molecule, basis_name: str, spec: GridSpec) -> float:
    basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
    density = hartree_fock(ENGINE, basis, molecule).density.spin_trace().blocks[0]
    grid = build_grid(molecule, spec)
    table = ShellTable.from_basis(basis, molecule)
    radii = np.array([function_radius(shell, 1e-12) for shell in basis.shells])
    values = ENGINE.basis_values(table, grid.points, radii)[:, 0, :]
    return float(grid.integrate(np.sum((values @ density) * values, axis=1)))


def test_molecular_grid_integrates_the_electron_density() -> None:
    molecule = water()
    grid = build_grid(molecule)
    assert grid.n_points == 33698
    assert set(np.unique(grid.atoms)) == {0, 1, 2}
    assert electron_count(molecule, "def2-svp", GridSpec()) == pytest.approx(
        10.0, abs=GRID_TOL
    )
    assert electron_count(ammonia_tilted(), "cc-pvdz", GridSpec(level=5)) == (
        pytest.approx(10.0, abs=GRID_TOL)
    )
    helium = Molecule.from_atoms([("He", (0.3, -0.2, 0.1))])
    assert electron_count(helium, "sto-3g", GridSpec()) == pytest.approx(
        2.0, abs=GRID_TOL
    )


def direct_values(basis: BasisSet, molecule: Molecule, points: Array) -> Array:
    """``[g, ao]`` from the shells' primitives and the solid-harmonic table."""
    columns: list[Array] = []
    positions = molecule.positions_array()
    for shell, atom in zip(basis.shells, basis.shell_atoms, strict=True):
        l = shell.angular_momentum
        d: Array = points - positions[atom]
        r2: Array = np.sum(d * d, axis=1)
        monomials = np.stack(
            [
                d[:, 0] ** i * d[:, 1] ** j * d[:, 2] ** k
                for i, j, k in cartesian_components(l)
            ],
            axis=1,
        )
        harmonics = monomials @ solid_harmonic_coefficients(l).T
        for row in shell.primitive_coefficients():
            radial = sum(
                c * np.exp(-alpha * r2)
                for c, alpha in zip(row, shell.exponents, strict=True)
            )
            columns.append(harmonics * np.asarray(radial)[:, None])
    return np.concatenate(columns, axis=1)


def test_basis_values_match_a_direct_evaluation_and_their_derivatives() -> None:
    molecule = water()
    basis = BasisSet.for_atoms(molecule, "def2-svp", library=LIBRARY)
    table = ShellTable.from_basis(basis, molecule)
    radii = np.full(table.n_shells, np.inf)
    points = np.random.default_rng(2).normal(scale=1.5, size=(12, 3))
    values = ENGINE.basis_values(table, points, radii, order=2)
    assert values.shape == (12, 10, basis.n_functions)
    np.testing.assert_allclose(
        values[:, 0, :], direct_values(basis, molecule, points), atol=BASIS_VALUE_TOL
    )
    h = FINITE_DIFFERENCE_STEP
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = h
        up = ENGINE.basis_values(table, points + shift, radii, order=1)
        down = ENGINE.basis_values(table, points - shift, radii, order=1)
        np.testing.assert_allclose(
            (up[:, 0, :] - down[:, 0, :]) / (2 * h),
            values[:, 1 + axis, :],
            atol=FINITE_DIFFERENCE_TOL,
        )
        slot = 4
        for a in range(3):
            for b in range(a, 3):
                if a == axis:
                    np.testing.assert_allclose(
                        (up[:, 1 + b, :] - down[:, 1 + b, :]) / (2 * h),
                        values[:, slot, :],
                        atol=FINITE_DIFFERENCE_TOL,
                    )
                slot += 1


def test_basis_values_respect_the_cutoff_radii() -> None:
    molecule = water()
    basis = BasisSet.for_atoms(molecule, "def2-svp", library=LIBRARY)
    table = ShellTable.from_basis(basis, molecule)
    radii = np.array(
        [function_radius(shell, BASIS_CUTOFF_PRECISION) for shell in basis.shells]
    )
    points = np.concatenate(
        [np.random.default_rng(3).normal(scale=3.0, size=(50, 3)), [[40.0, 0.0, 0.0]]]
    )
    screened = ENGINE.basis_values(table, points, radii, order=1)
    full = ENGINE.basis_values(table, points, np.full(table.n_shells, np.inf), order=1)
    np.testing.assert_allclose(
        screened[:, 0, :], full[:, 0, :], atol=BASIS_CUTOFF_PRECISION
    )
    assert np.all(screened[-1] == 0.0)
    assert np.any(screened != full)
    with pytest.raises(ValueError, match="non-negative"):
        ENGINE.basis_values(table, points, -radii)
    with pytest.raises(ValueError, match="second order"):
        ENGINE.basis_values(table, points, radii, order=3)


def test_grid_containers_are_validated() -> None:
    grid = build_grid(water(), GridSpec(level=0))
    batches = list(grid.batches(1000))
    assert batches[0] == slice(0, 1000)
    assert batches[-1].stop == grid.n_points
    assert sum(batch.stop - batch.start for batch in batches) == grid.n_points
    with pytest.raises(InputError, match="at least one"):
        list(grid.batches(0))
    with pytest.raises(InputError, match="g, 3"):
        MolecularGrid(np.zeros((4, 2)), np.zeros(4), np.zeros(4, dtype=np.int64))
    with pytest.raises(InputError, match="shape \\[g\\]"):
        MolecularGrid(np.zeros((4, 3)), np.zeros(3), np.zeros(4, dtype=np.int64))

"""Grids: quadratures, the partition, the assembly and basis values on the points."""

import itertools
import math

import numpy as np
import pytest
from numpy.typing import NDArray

from hafnium.backend import Array
from hafnium.basis import BasisSet, cartesian_components, solid_harmonic_coefficients
from hafnium.errors import InputError
from hafnium.grid import (
    BRAGG_RADII,
    LEBEDEV_SIZES,
    PARTITION_SCHEMES,
    SHARED_SHELL_START,
    GridSpec,
    MolecularGrid,
    atomic_grid,
    becke_partition,
    build_cell_grid,
    build_grid,
    build_molecular_grid,
    lebedev_degree,
    lebedev_grid,
    mura_knowles,
    nearest_neighbour_distances,
    nwchem_pruning,
    partition_centers,
    partition_step,
    partition_weights,
    treutler_adjustment,
    treutler_ahlrichs,
)
from hafnium.integrals import RustEngine, ShellTable
from hafnium.integrals.periodic import function_radius
from hafnium.molecule import Cell, Molecule
from hafnium.scf import hartree_fock
from hafnium.symmetry import KPoints
from tests.cells import box, helium, lithium_hydride
from tests.molecules import ammonia_tilted, water
from tests.shell_tables import LIBRARY
from tests.tolerances import (
    ANGULAR_QUADRATURE_TOL,
    BASIS_CUTOFF_PRECISION,
    BASIS_VALUE_TOL,
    FINITE_DIFFERENCE_TOL,
    GRID_TOL,
    PARTITION_TOL,
    PERIODIC_GRID_TOL,
    RADIAL_QUADRATURE_TOL,
    STRUCTURE_TOL,
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
    # The arrays are copies: modifying them leaves the rule intact.
    points, weights = lebedev_grid(6)
    weights *= 2.0
    points[0, 0] = 0.5
    assert float(lebedev_grid(6)[1].sum()) == pytest.approx(
        1.0, abs=ANGULAR_QUADRATURE_TOL
    )
    assert float(lebedev_grid(6)[0][0, 0]) != 0.5


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


# ------------------------------------------------------------------ cells


def test_partition_steps_have_the_limits_and_the_symmetry() -> None:
    nu = np.linspace(-1.0, 1.0, 41)
    for scheme in PARTITION_SCHEMES:
        step = np.asarray(partition_step(nu, scheme))
        assert step[0] == pytest.approx(1.0, abs=PARTITION_TOL)
        assert step[-1] == pytest.approx(0.0, abs=PARTITION_TOL)
        np.testing.assert_allclose(step + step[::-1], 1.0, atol=PARTITION_TOL)
    # The Stratmann step is exactly one and zero beyond its transition.
    stratmann = partition_step(np.array([-0.64, -0.7, 0.64, 0.9]), "stratmann")
    assert list(np.asarray(stratmann)) == [1.0, 1.0, 0.0, 0.0]
    becke = np.asarray(partition_step(np.array([-0.64, 0.64]), "becke"))
    assert 0.0 < becke[1] < becke[0] < 1.0
    with pytest.raises(InputError, match="scheme"):
        partition_step(nu, "voronoi")  # type: ignore[arg-type]


def test_partition_kernel_matches_the_array_implementation() -> None:
    rng = np.random.default_rng(7)
    numbers = (1, 6, 8, 3, 14)
    centers = 2.0 * rng.normal(size=(len(numbers), 3))
    points = 3.0 * rng.normal(size=(300, 3))
    adjustments = treutler_adjustment(np.array([BRAGG_RADII[z] for z in numbers]))
    for scheme in PARTITION_SCHEMES:
        expected = np.asarray(
            becke_partition(points, centers, adjustments, scheme=scheme)
        )
        np.testing.assert_allclose(expected.sum(axis=0), 1.0, atol=PARTITION_TOL)
        for owner in range(len(numbers)):
            weights = partition_weights(
                points, centers, owner, adjustments, scheme=scheme
            )
            np.testing.assert_allclose(
                np.asarray(weights), expected[owner], atol=PARTITION_TOL
            )
    with pytest.raises(InputError, match="owner"):
        partition_weights(points, centers, len(numbers))
    with pytest.raises(InputError, match="coincide"):
        partition_weights(points, np.zeros((2, 3)), 0)
    with pytest.raises(InputError, match="adjustments"):
        partition_weights(points, centers, 0, adjustments[:2, :2])


def test_partition_centers_lead_with_the_cell_atoms() -> None:
    cell = lithium_hydride()
    atoms = np.asarray(cell.positions_array())
    positions, numbers = partition_centers(cell, 6.0)
    positions, numbers = np.asarray(positions), np.asarray(numbers)
    np.testing.assert_allclose(positions[:2], atoms, atol=STRUCTURE_TOL)
    assert list(numbers[:2]) == [3, 1]
    distances = np.linalg.norm(positions[:, None, :] - atoms[None, :, :], axis=-1)
    assert np.all(distances.min(axis=1) <= 6.0)
    # Every image of either atom within 6 Bohr of an atom is present.
    expected = 0
    for shift in np.array(list(itertools.product(range(-3, 4), repeat=3))) * 4.0:
        for atom in atoms:
            if np.linalg.norm(atom + shift - atoms, axis=1).min() <= 6.0:
                expected += 1
    assert positions.shape == (expected, 3)
    # Lithium and hydrogen sit 2√3 Bohr apart, closer than the 4 Bohr images.
    np.testing.assert_allclose(
        np.asarray(nearest_neighbour_distances(cell)),
        2.0 * np.sqrt(3.0),
        atol=STRUCTURE_TOL,
    )
    with pytest.raises(InputError, match="radius"):
        partition_centers(cell, 0.0)


def periodic_gaussian(cell: Cell, points: Array, alpha: float) -> Array:
    """``Σ_T exp(−α |r − A − T|²)`` over the images of the first atom."""
    center: NDArray[np.float64] = np.asarray(cell.positions_array(), dtype=np.float64)[
        0
    ]
    reach = np.sqrt(-np.log(1e-16) / alpha)
    offsets: NDArray[np.float64] = np.asarray(points, dtype=np.float64) - center
    extent = float(np.linalg.norm(offsets, axis=1).max())
    translations = np.asarray(cell.lattice.translations(reach + extent))
    values = np.zeros(offsets.shape[0])
    for start in range(0, translations.shape[0], 64):
        shifted = offsets[None, :, :] - translations[start : start + 64, None, :]
        values += np.exp(-alpha * np.sum(shifted * shifted, axis=-1)).sum(axis=0)
    return values


@pytest.mark.parametrize(
    "cell",
    [helium(3), lithium_hydride(), helium(2), helium(1)],
    ids=["He", "LiH", "slab", "chain"],
)
def test_cell_grid_integrates_periodic_gaussians(cell: Cell) -> None:
    """``∫_cell Σ_T g(r − T) = ∫ g`` for a Gaussian ``g`` on an atom, any dimension."""
    grid = build_grid(cell)
    assert np.all(np.asarray(grid.weights) != 0.0)
    for alpha in (0.4, 2.0):
        integral = float(grid.integrate(periodic_gaussian(cell, grid.points, alpha)))
        exact = (math.pi / alpha) ** 1.5
        assert integral == pytest.approx(exact, rel=PERIODIC_GRID_TOL)


def test_cell_grid_is_translation_covariant() -> None:
    cell = helium(3)
    grid = build_grid(cell)
    doubled = build_grid(cell.supercell((2, 1, 1)))
    assert doubled.n_points == 2 * grid.n_points
    np.testing.assert_allclose(
        np.sort(np.tile(np.asarray(grid.weights), 2)),
        np.sort(np.asarray(doubled.weights)),
        rtol=0.0,
        atol=PARTITION_TOL,
    )
    shift = np.asarray(cell.lattice.vectors[0])
    points = np.asarray(grid.points)
    expected = np.sort(np.concatenate([points, points + shift]), axis=0)
    np.testing.assert_allclose(
        np.sort(np.asarray(doubled.points), axis=0), expected, atol=STRUCTURE_TOL
    )


def test_cell_grid_options_shape_the_grid() -> None:
    cell = lithium_hydride()
    default = build_grid(cell)
    # The shared-shell rule adds points beyond the base rule; without it the
    # Stratmann partition still drops the points of zero weight.
    plain = build_grid(cell, GridSpec(shared_angular_points=50))
    assert plain.n_points < default.n_points
    molecular = sum(
        atomic_grid(atom.atomic_number, GridSpec())[1].shape[0] for atom in cell.atoms
    )
    assert plain.n_points < molecular
    assert np.all(np.asarray(plain.weights) != 0.0)
    # Becke's step keeps the points of the whole shared-shell atomic grids,
    # except where its products of steps underflow to zero.
    becke = build_grid(cell, GridSpec(partition="becke", partition_radius=8.0))
    shared = sum(
        atomic_grid(
            atom.atomic_number, GridSpec(), shared_from=SHARED_SHELL_START * nearest
        )[1].shape[0]
        for atom, nearest in zip(
            cell.atoms, np.asarray(nearest_neighbour_distances(cell)), strict=True
        )
    )
    assert default.n_points < becke.n_points <= shared
    assert build_grid(cell) is not default
    assert isinstance(build_cell_grid(cell), MolecularGrid)
    with pytest.raises(InputError, match="scheme"):
        GridSpec(partition="voronoi")  # type: ignore[arg-type]
    with pytest.raises(InputError, match="radius"):
        GridSpec(partition_radius=0.0)
    with pytest.raises(InputError, match="Lebedev"):
        GridSpec(shared_angular_points=7)


def test_molecule_in_a_box_keeps_its_grid_near_the_atoms() -> None:
    """Where no image is within reach the periodic grid is the molecular one."""
    molecule = water()
    # Unpruned with the base rule on the shared shells, so that the cell's
    # grid takes the same points as the molecule's.
    spec = GridSpec(partition="stratmann", prune=False, shared_angular_points=302)
    molecular = build_molecular_grid(molecule, spec)
    periodic = build_grid(box(molecule, 80.0), spec)
    centers = np.asarray(molecule.positions_array())

    def near(grid: MolecularGrid) -> tuple[Array, Array]:
        points, weights = np.asarray(grid.points), np.asarray(grid.weights)
        distance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
        keep = (distance.min(axis=1) < 10.0) & (weights != 0.0)
        order = np.lexsort(points[keep].T)
        return points[keep][order], weights[keep][order]

    points_m, weights_m = near(molecular)
    points_p, weights_p = near(periodic)
    np.testing.assert_allclose(points_p, points_m, atol=STRUCTURE_TOL)
    np.testing.assert_allclose(weights_p, weights_m, rtol=0.0, atol=PARTITION_TOL)


def test_bloch_basis_values_match_the_phased_sum() -> None:
    cell = helium(3)
    basis = BasisSet.for_atoms(cell, "cc-pvdz", library=LIBRARY)
    table = ShellTable.from_basis(basis, cell)
    points = np.asarray(build_grid(cell).points)[::500]
    translations = np.asarray(cell.lattice.translations(9.0))
    mesh = KPoints.mesh((2, 1, 1))
    kpts = np.asarray(mesh.cartesian(cell.lattice))
    phases = np.exp(1j * kpts @ translations.T)
    infinite = np.full(table.n_shells, np.inf)
    bloch = np.asarray(
        ENGINE.bloch_basis_values(
            table, points, infinite, translations, phases, order=1
        )
    )
    assert bloch.shape == (mesh.n_kpt, points.shape[0], 4, basis.n_functions)
    expected = np.zeros_like(bloch)
    for image, translation in enumerate(translations):
        values = np.asarray(
            ENGINE.basis_values(table, points - translation, infinite, order=1)
        )
        expected += phases[:, image][:, None, None, None] * values[None]
    np.testing.assert_allclose(bloch, expected, atol=BASIS_VALUE_TOL)
    # The Γ point of a single image is the plain evaluation.
    gamma = ENGINE.bloch_basis_values(
        table, points, infinite, translations[:1], np.ones((1, 1), dtype=complex)
    )
    np.testing.assert_allclose(
        np.asarray(gamma)[0],
        np.asarray(ENGINE.basis_values(table, points, infinite)),
        atol=BASIS_VALUE_TOL,
    )
    # Cutoff radii leave out the far images of every shell, each within the
    # precision of the radius.
    radii = np.array(
        [function_radius(shell, BASIS_CUTOFF_PRECISION) for shell in basis.shells]
    )
    screened = np.asarray(
        ENGINE.bloch_basis_values(table, points, radii, translations, phases)
    )
    neglected = np.abs(screened[:, :, 0, :] - bloch[:, :, 0, :]).max()
    assert 0.0 < neglected < translations.shape[0] * BASIS_CUTOFF_PRECISION
    with pytest.raises(ValueError, match="phases"):
        ENGINE.bloch_basis_values(table, points, infinite, translations, phases[:, :2])

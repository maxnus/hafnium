"""Periodic nuclear attraction: large box, band folding, range separation."""

import numpy as np
import pytest

from hafnium.basis import BasisLibrary, BasisSet
from hafnium.errors import InputError
from hafnium.integrals import (
    RustEngine,
    core_hamiltonian_k,
    kinetic_k,
    lattice_images,
    nuclear_attraction,
    nuclear_attraction_k,
    overlap,
    overlap_k,
)
from hafnium.integrals.periodic import (
    RangeSeparation,
    bloch_sum,
    nuclear_attraction_images,
    reaching_images,
)
from hafnium.molecule import (
    AtomicSystem,
    Cell,
    Lattice,
    default_omega,
    ewald_energy,
)
from hafnium.symmetry import KPoints
from tests.cells import CUBIC, box, helium
from tests.molecules import water
from tests.spectra import generalized_eigenvalues
from tests.tolerances import (
    INTEGRAL_TOL,
    LARGE_BOX_RATIO_WINDOW,
    LARGE_BOX_RESIDUAL_TOL,
    LARGE_VACUUM_RATIO_WINDOW,
    LARGE_VACUUM_RESIDUAL_TOL,
    LARGE_VACUUM_SLAB_TOL,
    LATTICE_SUM_LOOSE_PRECISION,
    LATTICE_SUM_LOOSE_TOL,
    SPECTRUM_TOL,
)
from tests.unit.test_ewald import POINT_CHARGE_CUBIC

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
# Meshes with partner points off the time-reversal invariant momenta, by dimension.
MESHES = {
    3: KPoints.mesh((3, 2, 1)),
    2: KPoints.mesh((3, 2)),
    1: KPoints.mesh((3,)),
}
SHIFTED_MESHES = {
    3: KPoints.mesh((2, 2, 2), shift=(0.5, 0.5, 0.5)),
    2: KPoints.mesh((2, 2), shift=(0.5, 0.5)),
    1: KPoints.mesh((2,), shift=(0.5,)),
}


def basis_for(system: AtomicSystem, name: str) -> BasisSet:
    return BasisSet.for_atoms(system, name, library=LIBRARY)


def bands(cell: Cell, basis: BasisSet, kpoints: KPoints) -> np.ndarray:
    """Sorted core-Hamiltonian band energies over all k-points of ``kpoints``."""
    s = overlap_k(ENGINE, basis, cell, kpoints)
    h = core_hamiltonian_k(ENGINE, basis, cell, kpoints)
    return np.sort(
        np.concatenate(
            [generalized_eigenvalues(hk, sk) for hk, sk in zip(h, s, strict=True)]
        )
    )


def test_large_box_at_gamma_approaches_the_molecular_matrix() -> None:
    # The regularized potential of a point charge in a cubic box with its
    # background is 1/r − α/L + (2π/3L³) r² + O(r⁴/L⁵), so after the Madelung
    # term the residual falls off with the inverse cube of the box length.
    molecule = water()
    basis = basis_for(molecule, "cc-pvdz")
    reference = nuclear_attraction(ENGINE, basis, molecule)
    s = overlap(ENGINE, basis, molecule)
    charge = sum(molecule.nuclear_charges)
    residuals: list[float] = []
    for length in (20.0, 40.0):
        v = nuclear_attraction_k(ENGINE, basis, box(molecule, length), KPoints.gamma())
        assert v.shape == (1, basis.n_functions, basis.n_functions)
        np.testing.assert_allclose(np.imag(v), 0.0, atol=INTEGRAL_TOL)
        madelung = charge * POINT_CHARGE_CUBIC / length * s
        residuals.append(float(np.max(np.abs(np.real(v[0]) - reference - madelung))))
    assert residuals[1] < LARGE_BOX_RESIDUAL_TOL
    low, high = LARGE_BOX_RATIO_WINDOW
    assert low < residuals[1] / residuals[0] < high


@pytest.mark.parametrize(
    ("multiples", "shift"),
    [((2, 1, 2), (0.0, 0.0, 0.0)), ((1, 2, 1), (0.0, 0.5, 0.0))],
)
def test_supercell_at_one_point_folds_the_primitive_bands(
    multiples: tuple[int, ...], shift: tuple[float, ...]
) -> None:
    cell = helium(3)
    mesh = KPoints.mesh(multiples, shift=shift)
    primitive = bands(cell, basis_for(cell, "cc-pvdz"), mesh)
    supercell = cell.supercell(multiples)
    folded = bands(
        supercell, basis_for(supercell, "cc-pvdz"), KPoints.from_fractional([shift])
    )
    assert folded.shape == primitive.shape
    np.testing.assert_allclose(folded, primitive, atol=SPECTRUM_TOL)


@pytest.mark.parametrize("dimension", [3, 2, 1])
def test_matrix_is_independent_of_the_range_separation(dimension: int) -> None:
    cell = helium(dimension)
    basis = basis_for(cell, "cc-pvdz")
    mesh = SHIFTED_MESHES[dimension]
    reference = nuclear_attraction_k(ENGINE, basis, cell, mesh)
    omega = default_omega(cell.lattice)
    for factor in (0.5, 2.0):
        other = nuclear_attraction_k(ENGINE, basis, cell, mesh, omega=factor * omega)
        np.testing.assert_allclose(other, reference, atol=INTEGRAL_TOL)


def vacuum_box(cell: Cell, length: float) -> Cell:
    """``cell`` with vectors of ``length`` Bohr added along its open directions."""
    vectors = cell.lattice.vectors + tuple(
        (length * direction[0], length * direction[1], length * direction[2])
        for direction in cell.lattice.perpendicular_vectors
    )
    return Cell(
        cell.atoms, charge=cell.charge, spin=cell.spin, lattice=Lattice(vectors)
    )


def madelung(lattice: Lattice) -> float:
    return -2.0 * ewald_energy(lattice, np.ones(1), np.zeros((1, 3)))


@pytest.mark.parametrize("dimension", [2, 1])
def test_reduced_dimension_kernels_match_a_box_with_growing_vacuum(
    dimension: int,
) -> None:
    # A box with vacuum L along the open directions adds to the potential of
    # every nucleus its images and a background. Against the slab or the
    # chain the difference is the Madelung constants' difference plus the
    # background's quadratic term, (2π/(AL)) (z − z_A)² for a slab and
    # (π/(aL²)) |r_⊥ − A_⊥|² for a chain, up to terms exponentially small in
    # L for the slab and of order 1/L⁴ for the chain.
    cell = helium(dimension)
    basis = basis_for(cell, "cc-pvdz")
    gamma = KPoints.gamma(dimension)
    reduced = np.real(nuclear_attraction_k(ENGINE, basis, cell, gamma)[0])
    s = np.real(overlap_k(ENGINE, basis, cell, gamma)[0])
    charge = sum(cell.nuclear_charges)
    residuals: list[np.ndarray] = []
    for length in (20.0, 40.0):
        padded = vacuum_box(cell, length)
        v = nuclear_attraction_k(
            ENGINE, basis_for(padded, "cc-pvdz"), padded, KPoints.gamma()
        )
        constant = charge * (madelung(padded.lattice) - madelung(cell.lattice))
        residuals.append(np.real(v[0]) - reduced - constant * s)
    if dimension == 2:
        np.testing.assert_allclose(
            20.0 * residuals[0], 40.0 * residuals[1], atol=LARGE_VACUUM_SLAB_TOL
        )
    else:
        assert np.max(np.abs(residuals[1])) < LARGE_VACUUM_RESIDUAL_TOL
        index = np.argmax(np.abs(residuals[0]))
        low, high = LARGE_VACUUM_RATIO_WINDOW
        assert low < residuals[1].flat[index] / residuals[0].flat[index] < high


@pytest.mark.parametrize("dimension", [3, 2, 1])
def test_matrices_are_hermitian_and_time_reversal_conjugates_them(
    dimension: int,
) -> None:
    cell = helium(dimension)
    basis = basis_for(cell, "cc-pvdz")
    mesh = MESHES[dimension]
    invariant = mesh.is_time_reversal_invariant
    partners = np.asarray(mesh.time_reversal_partners())
    v = nuclear_attraction_k(ENGINE, basis, cell, mesh)
    np.testing.assert_allclose(v, np.conj(np.swapaxes(v, 1, 2)), atol=INTEGRAL_TOL)
    np.testing.assert_allclose(
        np.take(v, partners, axis=0), np.conj(v), atol=INTEGRAL_TOL
    )
    imaginary = np.abs(np.imag(v))
    for k, is_invariant in enumerate(invariant):
        if is_invariant:
            np.testing.assert_allclose(imaginary[k], 0.0, atol=INTEGRAL_TOL)
        else:
            assert np.max(imaginary[k]) > INTEGRAL_TOL


def test_precision_controls_the_sums_and_they_converge() -> None:
    cell = helium(3)
    basis = basis_for(cell, "cc-pvdz")
    mesh = KPoints.mesh((2, 2, 2))
    reference = nuclear_attraction_k(ENGINE, basis, cell, mesh)
    coarse = nuclear_attraction_k(
        ENGINE, basis, cell, mesh, precision=LATTICE_SUM_LOOSE_PRECISION
    )
    np.testing.assert_allclose(coarse, reference, atol=LATTICE_SUM_LOOSE_TOL)
    assert np.max(np.abs(coarse - reference)) > INTEGRAL_TOL
    tight = nuclear_attraction_k(ENGINE, basis, cell, mesh, precision=1e-14)
    np.testing.assert_allclose(tight, reference, atol=INTEGRAL_TOL)


def test_core_hamiltonian_is_the_kinetic_plus_the_attraction() -> None:
    cell = helium(3)
    basis = basis_for(cell, "cc-pvdz")
    mesh = KPoints.mesh((2, 1, 1))
    h = core_hamiltonian_k(ENGINE, basis, cell, mesh)
    t = kinetic_k(ENGINE, basis, cell, mesh)
    v = nuclear_attraction_k(ENGINE, basis, cell, mesh)
    np.testing.assert_allclose(h, t + v, atol=INTEGRAL_TOL)
    # The images are real and their Bloch sum is the matrix.
    separation = RangeSeparation.for_lattice(cell.lattice)
    translations = lattice_images(basis, cell)
    images = nuclear_attraction_images(ENGINE, basis, cell, translations, separation)
    assert images.dtype == np.float64
    assert images.shape == (translations.shape[0], basis.n_functions, basis.n_functions)
    np.testing.assert_allclose(
        bloch_sum(images, translations, mesh, cell.lattice), v, atol=INTEGRAL_TOL
    )


def test_short_range_radius_and_reaching_images() -> None:
    separation = RangeSeparation.for_lattice(CUBIC, precision=1e-10)
    assert separation.omega == pytest.approx(default_omega(CUBIC))
    radius = separation.short_range_radius
    assert np.exp(-((separation.omega * radius) ** 2)) <= 1e-10
    assert np.exp(-((separation.omega * 0.8 * radius) ** 2)) > 1e-10
    # The real-space sum sees every nucleus the pair densities reach.
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    images = reaching_images(CUBIC, positions[:1], positions[1:], np.array([[3.5]]))
    rounded = {tuple(row) for row in np.round(images, 8).tolist()}
    assert rounded == {(0.0, 0.0, 0.0), (-4.0, 0.0, 0.0)}
    np.testing.assert_array_equal(images[0], 0.0)


def test_inputs_are_validated() -> None:
    cell = helium(3)
    basis = basis_for(cell, "cc-pvdz")
    gamma = KPoints.gamma()
    with pytest.raises(InputError, match="omega is positive"):
        nuclear_attraction_k(ENGINE, basis, cell, gamma, omega=-1.0)
    with pytest.raises(InputError, match="precision"):
        core_hamiltonian_k(ENGINE, basis, cell, gamma, precision=0.0)
    with pytest.raises(InputError, match="do not fit"):
        nuclear_attraction_k(ENGINE, basis, cell, KPoints.gamma(2))
    with pytest.raises(InputError, match="one reach per pair"):
        reaching_images(CUBIC, np.zeros((1, 3)), np.zeros((1, 3)), np.ones((2, 1)))

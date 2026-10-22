"""Periodic nuclear attraction: large box, band folding, range separation."""

import math

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
from hafnium.molecule import AtomicSystem, Cell, Lattice, Molecule, default_omega
from hafnium.symmetry import KPoints
from tests.cells import CUBIC, helium
from tests.molecules import water
from tests.spectra import generalized_eigenvalues
from tests.tolerances import (
    INTEGRAL_TOL,
    LARGE_BOX_RESIDUAL_TOL,
    LATTICE_SUM_LOOSE_PRECISION,
    LATTICE_SUM_LOOSE_TOL,
    SPECTRUM_TOL,
)
from tests.unit.test_ewald import POINT_CHARGE_CUBIC

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()


def basis_for(system: AtomicSystem, name: str) -> BasisSet:
    return BasisSet.for_atoms(system, name, library=LIBRARY)


def cubic_box(molecule: Molecule, length: float) -> Cell:
    vectors = tuple(tuple(row) for row in (length * np.eye(3)).tolist())
    return Cell.from_molecule(molecule, Lattice(vectors))


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
        v = nuclear_attraction_k(
            ENGINE, basis, cubic_box(molecule, length), KPoints.gamma()
        )
        assert v.shape == (1, basis.n_functions, basis.n_functions)
        np.testing.assert_allclose(np.imag(v), 0.0, atol=INTEGRAL_TOL)
        madelung = charge * POINT_CHARGE_CUBIC / length * s
        residuals.append(float(np.max(np.abs(np.real(v[0]) - reference - madelung))))
    assert residuals[1] < LARGE_BOX_RESIDUAL_TOL
    assert 0.09 < residuals[1] / residuals[0] < 0.16


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


def test_matrix_is_independent_of_the_range_separation() -> None:
    cell = helium(3)
    basis = basis_for(cell, "cc-pvdz")
    mesh = KPoints.mesh((2, 2, 2), shift=(0.5, 0.5, 0.5))
    reference = nuclear_attraction_k(ENGINE, basis, cell, mesh)
    omega = default_omega(cell.lattice)
    for factor in (0.5, 2.0):
        other = nuclear_attraction_k(ENGINE, basis, cell, mesh, omega=factor * omega)
        np.testing.assert_allclose(other, reference, atol=INTEGRAL_TOL)


def test_matrices_are_hermitian_and_time_reversal_conjugates_them() -> None:
    cell = helium(3)
    basis = basis_for(cell, "cc-pvdz")
    mesh = KPoints.mesh((3, 2, 1))
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


def test_range_separation_kernels_and_cutoffs() -> None:
    separation = RangeSeparation.for_lattice(CUBIC, precision=1e-10)
    assert separation.omega == pytest.approx(default_omega(CUBIC))
    radius = separation.short_range_radius
    assert math.erfc(separation.omega * radius) <= 1e-10
    assert math.erfc(separation.omega * 0.8 * radius) > 1e-10
    g_vectors = separation.reciprocal_vectors(CUBIC, charge_product=4.0)
    np.testing.assert_array_equal(g_vectors[0], 0.0)
    lengths = np.linalg.norm(g_vectors, axis=1)
    assert np.all(np.diff(lengths) >= -INTEGRAL_TOL)
    more = separation.reciprocal_vectors(CUBIC, charge_product=4e6)
    assert more.shape[0] > g_vectors.shape[0]
    kernel = separation.long_range_kernel(CUBIC, g_vectors)
    assert kernel[0] == pytest.approx(-math.pi / separation.omega**2)
    g_squared = lengths[1:] ** 2
    expected = (
        4.0 * math.pi * np.exp(-g_squared / (4.0 * separation.omega**2)) / g_squared
    )
    np.testing.assert_allclose(kernel[1:], expected, atol=INTEGRAL_TOL)
    # The real-space sum sees every nucleus the pair densities reach.
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    images = reaching_images(CUBIC, positions[:1], (2.0,), positions[1:], (1.5,))
    rounded = {tuple(row) for row in np.round(images, 8).tolist()}
    assert rounded == {(0.0, 0.0, 0.0), (-4.0, 0.0, 0.0)}
    np.testing.assert_array_equal(images[0], 0.0)


def test_inputs_are_validated() -> None:
    cell = helium(3)
    basis = basis_for(cell, "cc-pvdz")
    gamma = KPoints.gamma()
    with pytest.raises(InputError, match="not implemented yet"):
        nuclear_attraction_k(
            ENGINE, basis_for(helium(2), "sto-3g"), helium(2), KPoints.gamma(2)
        )
    with pytest.raises(InputError, match="omega is positive"):
        nuclear_attraction_k(ENGINE, basis, cell, gamma, omega=-1.0)
    with pytest.raises(InputError, match="precision"):
        core_hamiltonian_k(ENGINE, basis, cell, gamma, precision=0.0)
    with pytest.raises(InputError, match="do not fit"):
        nuclear_attraction_k(ENGINE, basis, cell, KPoints.gamma(2))
    with pytest.raises(InputError, match="not implemented yet"):
        RangeSeparation.for_lattice(Lattice(((3.0, 0.0, 0.0),))).long_range_kernel(
            Lattice(((3.0, 0.0, 0.0),)), np.zeros((1, 3))
        )
    with pytest.raises(InputError, match="one radius each"):
        reaching_images(CUBIC, np.zeros((1, 3)), (1.0, 2.0), np.zeros((1, 3)), (1.0,))

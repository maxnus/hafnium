"""Bloch sums of one-electron integrals: large box, band folding, images."""

import math

import numpy as np
import pytest

from hafnium.backend import Array
from hafnium.basis import BasisLibrary, BasisSet, Shell
from hafnium.errors import InputError
from hafnium.integrals import (
    RustEngine,
    ShellTable,
    bloch_sum,
    kinetic,
    kinetic_k,
    lattice_images,
    overlap,
    overlap_k,
)
from hafnium.integrals.periodic import atom_radii, function_radius
from hafnium.molecule import Atom, AtomicSystem, Cell, Lattice, Molecule
from hafnium.symmetry import KPoints
from tests.cells import helium
from tests.molecules import water
from tests.spectra import generalized_eigenvalues
from tests.tolerances import (
    ENERGY_TOL,
    INTEGRAL_TOL,
    LATTICE_SUM_LOOSE_PRECISION,
    LATTICE_SUM_LOOSE_TOL,
    SPECTRUM_TOL,
)

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()


def basis_for(system: AtomicSystem, name: str) -> BasisSet:
    return BasisSet.for_atoms(system, name, library=LIBRARY)


def box(molecule: Molecule, length: float, dimension: int) -> Cell:
    """``molecule`` repeated on an orthogonal lattice of side ``length`` in Bohr."""
    vectors = tuple(tuple(row) for row in (length * np.eye(3))[:dimension].tolist())
    return Cell.from_molecule(molecule, Lattice(vectors))


def bands(cell: Cell, basis: BasisSet, kpoints: KPoints) -> np.ndarray:
    """Sorted kinetic band energies over all k-points of ``kpoints``."""
    s = overlap_k(ENGINE, basis, cell, kpoints)
    t = kinetic_k(ENGINE, basis, cell, kpoints)
    return np.sort(
        np.concatenate(
            [generalized_eigenvalues(tk, sk) for tk, sk in zip(t, s, strict=True)]
        )
    )


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_large_box_at_gamma_reproduces_the_molecular_matrices(dimension: int) -> None:
    molecule = water()
    basis = basis_for(molecule, "cc-pvdz")
    gamma = KPoints.gamma(dimension)
    large = box(molecule, 30.0, dimension)
    s = overlap_k(ENGINE, basis, large, gamma)
    t = kinetic_k(ENGINE, basis, large, gamma)
    n = basis.n_functions
    assert s.shape == t.shape == (1, n, n)
    assert s.dtype == np.complex128
    np.testing.assert_allclose(s[0], overlap(ENGINE, basis, molecule), atol=ENERGY_TOL)
    np.testing.assert_allclose(t[0], kinetic(ENGINE, basis, molecule), atol=ENERGY_TOL)
    # In a small box the images overlap and the lattice sum differs.
    small = overlap_k(ENGINE, basis, box(molecule, 6.0, dimension), gamma)
    assert not np.allclose(small[0], overlap(ENGINE, basis, molecule), atol=ENERGY_TOL)


@pytest.mark.parametrize(
    ("dimension", "multiples", "shift"),
    [
        (1, (3,), (0.0,)),
        (1, (2,), (0.5,)),
        (2, (2, 2), (0.5, 0.0)),
        (3, (2, 2, 1), (0.0, 0.5, 0.0)),
        (3, (2, 1, 2), (0.0, 0.0, 0.0)),
    ],
)
def test_supercell_at_one_point_folds_the_primitive_bands(
    dimension: int, multiples: tuple[int, ...], shift: tuple[float, ...]
) -> None:
    # The supercell at the fractional point ``shift`` of its own reciprocal
    # basis unfolds onto the primitive mesh with that shift.
    cell = helium(dimension)
    mesh = KPoints.mesh(multiples, shift=shift)
    primitive = bands(cell, basis_for(cell, "cc-pvdz"), mesh)
    supercell = cell.supercell(multiples)
    point = KPoints.from_fractional([shift])
    folded = bands(supercell, basis_for(supercell, "cc-pvdz"), point)
    assert folded.shape == primitive.shape
    np.testing.assert_allclose(folded, primitive, atol=SPECTRUM_TOL)


def test_matrices_are_hermitian_and_time_reversal_conjugates_them() -> None:
    cell = helium(3)
    basis = basis_for(cell, "cc-pvdz")
    mesh = KPoints.mesh((3, 2, 1))
    invariant = mesh.is_time_reversal_invariant
    assert not all(invariant)
    partners = np.asarray(mesh.time_reversal_partners())
    matrices: list[Array] = [
        overlap_k(ENGINE, basis, cell, mesh),
        kinetic_k(ENGINE, basis, cell, mesh),
    ]
    for matrix in matrices:
        adjoint = np.conj(np.swapaxes(matrix, 1, 2))
        np.testing.assert_allclose(matrix, adjoint, atol=INTEGRAL_TOL)
        np.testing.assert_allclose(
            np.take(matrix, partners, axis=0), np.conj(matrix), atol=INTEGRAL_TOL
        )
        imaginary = np.abs(np.imag(matrix))
        for k, is_invariant in enumerate(invariant):
            if is_invariant:
                np.testing.assert_allclose(imaginary[k], 0.0, atol=INTEGRAL_TOL)
            else:
                assert np.max(imaginary[k]) > INTEGRAL_TOL
            assert np.linalg.eigvalsh(matrix[k]).min() > 0.0


def test_images_follow_the_precision_and_the_sums_are_converged() -> None:
    cell = helium(3)
    basis = basis_for(cell, "cc-pvdz")
    loose = lattice_images(basis, cell, precision=LATTICE_SUM_LOOSE_PRECISION)
    tight = lattice_images(basis, cell)
    assert 1 < loose.shape[0] < tight.shape[0]
    np.testing.assert_array_equal(tight[0], 0.0)
    lengths = np.linalg.norm(tight, axis=1)
    assert np.all(np.diff(lengths) >= -INTEGRAL_TOL)
    rounded: list[tuple[float, ...]] = [
        tuple(value) for value in np.round(tight, 8).tolist()
    ]
    negated: list[tuple[float, ...]] = [
        tuple(value) for value in np.round(-tight, 8).tolist()
    ]
    assert set(negated) == set(rounded)
    mesh = KPoints.mesh((2, 2, 2), shift=(0.5, 0.5, 0.5))
    reference = overlap_k(ENGINE, basis, cell, mesh)
    coarse = overlap_k(ENGINE, basis, cell, mesh, precision=LATTICE_SUM_LOOSE_PRECISION)
    np.testing.assert_allclose(coarse, reference, atol=LATTICE_SUM_LOOSE_TOL)
    assert np.max(np.abs(coarse - reference)) > INTEGRAL_TOL
    # Every image within twice the longest kept translation adds nothing.
    wider = cell.lattice.translations(2.0 * float(lengths.max()))
    assert wider.shape[0] > tight.shape[0]
    table = ShellTable.from_basis(basis, cell)
    direct = bloch_sum(ENGINE.overlap(table, table, wider), wider, mesh, cell.lattice)
    np.testing.assert_allclose(direct, reference, atol=INTEGRAL_TOL)


def test_moving_an_atom_by_a_lattice_vector_changes_phases_only() -> None:
    cell = helium(3)
    basis = basis_for(cell, "cc-pvdz")
    mesh = KPoints.mesh((3, 2, 1))
    shift = np.asarray(cell.lattice.vectors[0]) + np.asarray(cell.lattice.vectors[2])
    position = (np.asarray(cell.atoms[1].position) + shift).tolist()
    moved_atom = Atom("He", (position[0], position[1], position[2]))
    moved = Cell((cell.atoms[0], moved_atom), lattice=cell.lattice)
    on_moved = np.asarray([label.atom == 1 for label in basis.function_labels()])
    phase = np.exp(-1j * (mesh.cartesian(cell.lattice) @ shift))
    scale = np.where(on_moved[np.newaxis, :], phase[:, np.newaxis], 1.0)
    for function in (overlap_k, kinetic_k):
        before = function(ENGINE, basis, cell, mesh)
        after = function(ENGINE, basis, moved, mesh)
        expected = np.conj(scale)[:, :, np.newaxis] * before * scale[:, np.newaxis, :]
        np.testing.assert_allclose(after, expected, atol=INTEGRAL_TOL)


def test_function_radius_bounds_the_function_and_follows_the_precision() -> None:
    shell = Shell(2, (0.3, 2.0), ((0.6, 0.5),))
    precision = 1e-8
    radius = function_radius(shell, precision)
    coefficients = shell.primitive_coefficients()[0]

    def terms(r: float) -> list[float]:
        angular = math.sqrt(5.0 / (4.0 * math.pi))
        return [
            angular * abs(c) * r**2 * math.exp(-a * r * r)
            for c, a in zip(coefficients, shell.exponents, strict=True)
        ]

    assert sum(terms(radius)) <= precision
    assert max(terms(0.8 * radius)) > precision / shell.n_primitives
    assert function_radius(shell, 1e-4) < radius < function_radius(shell, 1e-12)
    # A function that never reaches the precision keeps its natural scale.
    tight = Shell(0, (50.0,), ((1.0,),))
    assert function_radius(tight, 100.0) == pytest.approx(
        1.0 / math.sqrt(50.0), abs=INTEGRAL_TOL
    )
    # The outer root is found even when the precision is close to the
    # maximum of the function.
    f_shell = Shell(3, (0.5,), ((1.0,),))
    peak = math.sqrt(3.0 / (2.0 * 0.5))
    coefficient = f_shell.primitive_coefficients()[0][0]
    at_peak = (
        math.sqrt(7.0 / (4.0 * math.pi))
        * coefficient
        * peak**3
        * math.exp(-0.5 * peak**2)
    )
    near = function_radius(f_shell, 0.999 * at_peak)
    assert peak < near < 1.2 * peak
    assert function_radius(f_shell, 1.001 * at_peak) == pytest.approx(
        peak, abs=INTEGRAL_TOL
    )
    with pytest.raises(InputError, match="precision"):
        function_radius(shell, 0.0)
    cell = helium(3)
    radii = atom_radii(basis_for(cell, "cc-pvdz"), precision)
    assert len(radii) == 2
    assert radii[0] == radii[1] > 0.0


def test_inputs_are_validated() -> None:
    cell = helium(2)
    basis = basis_for(cell, "sto-3g")
    with pytest.raises(InputError, match="do not fit"):
        overlap_k(ENGINE, basis, cell, KPoints.gamma(3))
    with pytest.raises(InputError, match="built for the atoms"):
        lattice_images(basis_for(water(), "sto-3g"), cell)
    with pytest.raises(InputError, match="one image per translation"):
        bloch_sum(np.zeros((2, 3, 3)), np.zeros((1, 3)), KPoints.gamma(2), cell.lattice)
    with pytest.raises(InputError, match=r"\[image, 3\]"):
        bloch_sum(np.zeros((2, 3, 3)), np.zeros((2, 2)), KPoints.gamma(2), cell.lattice)
    with pytest.raises(InputError, match="precision"):
        kinetic_k(ENGINE, basis, cell, KPoints.gamma(2), precision=-1.0)

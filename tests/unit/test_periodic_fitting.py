"""Density fitting with momentum transfer: invariances, large box, kernels."""

import math

import numpy as np
import pytest

from hafnium.basis import BasisLibrary, BasisSet, Shell
from hafnium.errors import InputError
from hafnium.integrals import (
    BlochImages,
    DensityFitting,
    PeriodicDensityFitting,
    RustEngine,
    ShellTable,
    density_fitting_k,
    overlap,
)
from hafnium.integrals.periodic import (
    RangeSeparation,
    function_absolute_integral,
    integer_coordinates,
)
from hafnium.molecule import AtomicSystem, Molecule, default_omega
from hafnium.symmetry import KPoints
from tests.cells import CUBIC, box, helium, helium_atom
from tests.molecules import water
from tests.tolerances import (
    INTEGRAL_TOL,
    LARGE_BOX_FITTING_RESIDUAL_TOL,
    LARGE_BOX_RATIO_WINDOW,
    LATTICE_SUM_LOOSE_PRECISION,
    LATTICE_SUM_LOOSE_TOL,
)
from tests.unit.test_ewald import POINT_CHARGE_CUBIC

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
AUXILIARY = "def2-universal-jkfit"


def basis_for(system: AtomicSystem, name: str) -> BasisSet:
    return BasisSet.for_atoms(system, name, library=LIBRARY)


def fitting_for(
    cell: AtomicSystem, basis_name: str, mesh: KPoints, **keywords: float
) -> PeriodicDensityFitting:
    assert hasattr(cell, "lattice")
    separation = RangeSeparation.for_lattice(cell.lattice, **keywords)  # type: ignore[attr-defined]
    return density_fitting_k(
        ENGINE,
        basis_for(cell, basis_name),
        basis_for(cell, AUXILIARY),
        cell,  # type: ignore[arg-type]
        mesh,
        separation=separation,
    )


def test_metric_and_tensor_are_hermitian_and_the_metric_is_positive() -> None:
    cell = helium(3)
    mesh = KPoints.mesh((2, 1, 2), shift=(0.5, 0.0, 0.0))
    fit = fitting_for(cell, "cc-pvdz", mesh)
    metric, tensor = fit.metric, fit.three_center
    assert metric.shape == (mesh.n_kpt, fit.n_aux, fit.n_aux)
    assert tensor.shape == (mesh.n_kpt, mesh.n_kpt, fit.n_aux, fit.n_ao, fit.n_ao)
    np.testing.assert_allclose(
        metric, np.conj(np.swapaxes(metric, 1, 2)), atol=INTEGRAL_TOL
    )
    assert all(np.linalg.eigvalsh(metric[q]).min() > 0.0 for q in range(mesh.n_kpt))
    swapped = np.conj(np.swapaxes(np.swapaxes(tensor, 0, 1), 3, 4))
    np.testing.assert_allclose(tensor, swapped, atol=INTEGRAL_TOL)
    # The transfer of (k1, k2) is k2 − k1 and that of (k2, k1) its negative.
    points = np.asarray(mesh.fractional)
    transfers = np.asarray(fit.transfers.fractional)
    for k1 in range(mesh.n_kpt):
        for k2 in range(mesh.n_kpt):
            q = transfers[int(fit.transfer_index[k1, k2])]
            distance = np.abs((q - (points[k2] - points[k1]) + 0.5) % 1.0 - 0.5)
            assert np.max(distance) < INTEGRAL_TOL
    # At Γ alone the metric and the tensor are real.
    gamma = fitting_for(cell, "cc-pvdz", KPoints.gamma())
    np.testing.assert_allclose(np.imag(gamma.metric), 0.0, atol=INTEGRAL_TOL)
    np.testing.assert_allclose(np.imag(gamma.three_center), 0.0, atol=INTEGRAL_TOL)


def test_integrals_are_independent_of_the_range_separation() -> None:
    cell = helium_atom()
    mesh = KPoints.mesh((2, 2, 1))
    reference = fitting_for(cell, "cc-pvdz", mesh)
    omega = default_omega(cell.lattice)
    for factor in (0.5, 2.0):
        other = fitting_for(cell, "cc-pvdz", mesh, omega=factor * omega)
        np.testing.assert_allclose(other.metric, reference.metric, atol=INTEGRAL_TOL)
        np.testing.assert_allclose(
            other.three_center, reference.three_center, atol=INTEGRAL_TOL
        )


def test_precision_controls_the_sums_and_they_converge() -> None:
    cell = helium_atom()
    mesh = KPoints.mesh((2, 2, 1))
    reference = fitting_for(cell, "cc-pvdz", mesh)
    coarse = fitting_for(cell, "cc-pvdz", mesh, precision=LATTICE_SUM_LOOSE_PRECISION)
    np.testing.assert_allclose(
        coarse.metric, reference.metric, atol=LATTICE_SUM_LOOSE_TOL
    )
    np.testing.assert_allclose(
        coarse.three_center, reference.three_center, atol=LATTICE_SUM_LOOSE_TOL
    )
    assert np.max(np.abs(coarse.three_center - reference.three_center)) > INTEGRAL_TOL
    tight = fitting_for(cell, "cc-pvdz", mesh, precision=1e-14)
    np.testing.assert_allclose(tight.metric, reference.metric, atol=INTEGRAL_TOL)
    np.testing.assert_allclose(
        tight.three_center, reference.three_center, atol=INTEGRAL_TOL
    )


def test_large_box_at_gamma_approaches_the_molecular_fitting() -> None:
    # Two charge distributions in a cubic box with their backgrounds interact
    # as 1/r − α/L + (2π/3L³) r² + ..., so after the Madelung term with the
    # charges of the auxiliary functions and of the pair densities the
    # residual falls off with the inverse cube of the box length.
    molecule = water()
    basis = basis_for(molecule, "sto-3g")
    auxiliary = basis_for(molecule, AUXILIARY)
    molecular = DensityFitting.build(ENGINE, basis, auxiliary, molecule)
    charges = np.real(
        ENGINE.fourier_transform(
            ShellTable.from_basis(auxiliary, molecule), np.zeros((1, 3))
        )[0]
    )
    s = overlap(ENGINE, basis, molecule)
    residuals: list[tuple[float, float]] = []
    for length in (40.0, 80.0):
        fit = fitting_for(box(molecule, length), "sto-3g", KPoints.gamma())
        madelung = POINT_CHARGE_CUBIC / length
        metric = molecular.metric - madelung * np.outer(charges, charges)
        three = molecular.three_center - madelung * charges[:, None, None] * s
        residuals.append(
            (
                float(np.max(np.abs(np.real(fit.metric[0]) - metric))),
                float(np.max(np.abs(np.real(fit.three_center[0, 0]) - three))),
            )
        )
    low, high = LARGE_BOX_RATIO_WINDOW
    for far, near in zip(residuals[1], residuals[0], strict=True):
        assert far < LARGE_BOX_FITTING_RESIDUAL_TOL
        assert low < far / near < high


def test_fitted_integrals_conserve_momentum_and_follow_the_tensor() -> None:
    cell = helium_atom()
    mesh = KPoints.mesh((2, 1, 2))
    fit = fitting_for(cell, "cc-pvdz", mesh)
    tensor = fit.df_tensor()
    assert tensor.shape == fit.three_center.shape
    quadruple = (1, 3, 2, 0)
    k1, k2, k3, k4 = quadruple
    assert int(fit.transfer_index[k1, k2]) == int(fit.transfer_index[k4, k3])
    eri = fit.eri(quadruple)
    expected = np.einsum("Puv,Pxw->uvwx", tensor[k1, k2], np.conj(tensor[k4, k3]))
    np.testing.assert_allclose(eri, expected, atol=INTEGRAL_TOL)
    # (u k1 v k2 | w k3 x k4) = conj((v k2 u k1 | x k4 w k3)).
    swapped = fit.eri((k2, k1, k4, k3))
    np.testing.assert_allclose(
        eri, np.conj(np.transpose(swapped, (1, 0, 3, 2))), atol=INTEGRAL_TOL
    )
    with pytest.raises(InputError, match="conserve crystal momentum"):
        fit.eri((0, 1, 0, 0))


def test_bloch_kernel_reduces_to_the_plain_kernel_for_one_image() -> None:
    cell = helium(3)
    orbital = ShellTable.from_basis(basis_for(cell, "cc-pvdz"), cell)
    fitting = ShellTable.from_basis(basis_for(cell, AUXILIARY), cell)
    zero = np.zeros((1, 3))
    origin = np.zeros((1, 3), dtype=np.int64)
    ones = (np.ones((1, 1), dtype=complex), np.ones((1, 1), dtype=complex))
    images = BlochImages(zero, zero, (origin, origin), ones)
    assert images.n_kpt == 1
    summed = ENGINE.coulomb_three_center_bloch(fitting, orbital, orbital, images)
    plain = ENGINE.coulomb_three_center(fitting, orbital, orbital, np.zeros((1, 2, 3)))
    np.testing.assert_allclose(summed[0, 0], plain[0], atol=INTEGRAL_TOL)
    # A step that reaches no image adds nothing.
    far = BlochImages(zero, zero + 5.0, (origin, origin + 1), ones)
    assert np.all(
        ENGINE.coulomb_three_center_bloch(fitting, orbital, orbital, far) == 0.0
    )
    with pytest.raises(InputError, match="three lattice coordinates"):
        BlochImages(zero, zero, (origin, np.zeros((2, 3), dtype=np.int64)), ones)
    with pytest.raises(InputError, match=r"\[kpt, image\]"):
        BlochImages(zero, zero, (origin, origin), (zero, zero))
    with pytest.raises(ValueError, match="distinct"):
        ENGINE.coulomb_three_center_bloch(
            fitting,
            orbital,
            orbital,
            BlochImages(
                np.zeros((2, 3)),
                zero,
                (np.zeros((2, 3), dtype=np.int64), origin),
                (np.ones((1, 2), dtype=complex), np.ones((1, 2), dtype=complex)),
            ),
        )


def test_integer_coordinates_label_the_images() -> None:
    images = CUBIC.translations(9.0)
    coordinates = integer_coordinates(CUBIC, images)
    assert coordinates.dtype == np.int64
    np.testing.assert_allclose(coordinates * 4.0, images, atol=INTEGRAL_TOL)
    rows = {tuple(row) for row in coordinates.tolist()}
    assert len(rows) == images.shape[0]


def test_absolute_integral_bounds_the_transform() -> None:
    molecule = Molecule.from_atoms([("He", (0.0, 0.0, 0.0))])
    g_vectors = np.array([[0.0, 0.0, 0.0], [0.4, -0.3, 0.2], [1.5, 0.5, -0.7]])
    for shell in (
        Shell(0, (0.9, 0.3), ((0.6, 0.5),)),
        Shell(1, (0.7,), ((1.0,),)),
        Shell(2, (1.1, 0.4), ((0.4, 0.7),)),
    ):
        basis = BasisSet("one", ("He",), (shell,), (0,))
        table = ShellTable.from_basis(basis, molecule)
        transform = ENGINE.fourier_transform(table, g_vectors)
        bound = function_absolute_integral(shell)
        assert np.max(np.abs(transform)) <= bound
        if shell.angular_momentum == 0:
            # For an s function the bound is the integral itself.
            assert float(np.real(transform[0, 0])) == pytest.approx(
                bound, rel=math.sqrt(INTEGRAL_TOL)
            )


def test_inputs_are_validated() -> None:
    cell = helium(2)
    with pytest.raises(InputError, match="not implemented yet"):
        fitting_for(cell, "sto-3g", KPoints.gamma(2))
    cell = helium_atom()
    with pytest.raises(InputError, match="do not fit"):
        fitting_for(cell, "sto-3g", KPoints.gamma(2))
    with pytest.raises(InputError, match="uniform mesh"):
        fitting_for(
            cell, "sto-3g", KPoints.from_fractional([(0.0, 0.0, 0.0), (0.3, 0.0, 0.0)])
        )
    fit = fitting_for(cell, "sto-3g", KPoints.gamma())
    with pytest.raises(InputError, match=r"\[q, aux, aux\]"):
        PeriodicDensityFitting(
            fit.kpoints,
            fit.transfers,
            fit.transfer_index,
            fit.metric[0],
            fit.three_center,
        )
    with pytest.raises(InputError, match=r"\[kpt, kpt, aux, ao, ao\]"):
        PeriodicDensityFitting(
            fit.kpoints,
            fit.transfers,
            fit.transfer_index,
            fit.metric,
            fit.three_center[0],
        )

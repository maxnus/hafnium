"""k-point meshes, time-reversal pairing and weights."""

import numpy as np
import pytest

from hafnium.errors import InputError
from hafnium.molecule import Lattice
from hafnium.symmetry import KPoints
from tests.tolerances import STRUCTURE_TOL


def test_gamma_centred_mesh_follows_the_convention() -> None:
    mesh = KPoints.mesh((2, 3))
    assert mesh.n_kpt == 6
    assert mesh.dimension == 2
    assert mesh.fractional == (
        (0.0, 0.0),
        (0.0, 1 / 3),
        (0.0, 2 / 3),
        (0.5, 0.0),
        (0.5, 1 / 3),
        (0.5, 2 / 3),
    )
    assert mesh.weights == (1 / 6,) * 6
    assert KPoints.gamma().fractional == ((0.0, 0.0, 0.0),)
    assert KPoints.gamma(1) == KPoints.mesh((1,))


def test_shifted_meshes_and_monkhorst_pack() -> None:
    shifted = KPoints.mesh((2, 2, 1), shift=(0.5, 0.5, 0.0))
    assert shifted.fractional[0] == (0.25, 0.25, 0.0)
    assert shifted.fractional[-1] == (0.75, 0.75, 0.0)
    # even sizes are shifted by half a step, odd sizes contain Γ
    assert KPoints.monkhorst_pack((2, 3, 4)) == KPoints.mesh(
        (2, 3, 4), shift=(0.5, 0.0, 0.5)
    )
    assert KPoints.monkhorst_pack((3,)) == KPoints.mesh((3,))
    with pytest.raises(InputError, match="0 or 1/2"):
        KPoints.mesh((2, 2), shift=(0.25, 0.0))
    with pytest.raises(InputError, match="0 or 1/2"):
        KPoints.mesh((2, 2), shift=(0.5,))
    with pytest.raises(InputError, match="positive sizes"):
        KPoints.mesh((2, 0))
    with pytest.raises(InputError, match="positive sizes"):
        KPoints.mesh(())


def test_coordinates_are_wrapped_and_validated() -> None:
    points = KPoints.from_fractional([(1.25, -0.5, 2.0), (-1e-17, 0.999, 0.0)])
    assert points.fractional[0] == (0.25, 0.5, 0.0)
    assert points.fractional[1] == (0.0, 0.999, 0.0)
    assert points.weights == (0.5, 0.5)
    weighted = KPoints.from_fractional([(0.0,), (0.5,)], weights=(0.25, 0.75))
    assert weighted.weights == (0.25, 0.75)
    with pytest.raises(InputError, match="at least one point"):
        KPoints.from_fractional([])
    with pytest.raises(InputError, match="same number"):
        KPoints.from_fractional([(0.0, 0.0), (0.0,)])
    with pytest.raises(InputError, match="one to three"):
        KPoints.from_fractional([(0.0, 0.0, 0.0, 0.0)])
    with pytest.raises(InputError, match="weights"):
        KPoints.from_fractional([(0.0,), (0.5,)], weights=(0.5,))
    with pytest.raises(InputError, match="sum to one"):
        KPoints.from_fractional([(0.0,), (0.5,)], weights=(0.5, 0.6))
    with pytest.raises(InputError, match="positive"):
        KPoints.from_fractional([(0.0,), (0.5,)], weights=(1.0, 0.0))
    with pytest.raises(InputError, match="finite"):
        KPoints.from_fractional([(float("inf"),)])


@pytest.mark.parametrize(
    ("size", "shift"),
    [
        ((2, 2, 2), None),
        ((3, 2, 1), None),
        ((2, 4, 3), (0.5, 0.5, 0.0)),
        ((5,), (0.5,)),
    ],
)
def test_time_reversal_partners_of_meshes(
    size: tuple[int, ...], shift: tuple[float, ...] | None
) -> None:
    mesh = KPoints.mesh(size, shift=shift)
    partners = mesh.time_reversal_partners()
    points = mesh.fractional_array()
    for index, partner in enumerate(partners):
        assert partners[partner] == index
        difference = (points[index] + points[partner] + 0.5) % 1.0 - 0.5
        assert np.all(np.abs(difference) <= STRUCTURE_TOL)
    invariant = mesh.is_time_reversal_invariant
    assert all(
        (partner == index) == flag
        for index, (partner, flag) in enumerate(zip(partners, invariant, strict=True))
    )


def test_unshifted_even_mesh_is_entirely_time_reversal_invariant() -> None:
    assert all(KPoints.mesh((2, 2, 2)).is_time_reversal_invariant)
    assert KPoints.mesh((3, 3, 3)).is_time_reversal_invariant.count(True) == 1
    assert not any(KPoints.mesh((2, 2), shift=(0.5, 0.5)).is_time_reversal_invariant)


def test_partners_must_be_present_and_unique() -> None:
    with pytest.raises(InputError, match="no time-reversal partner"):
        KPoints.from_fractional([(0.0,), (0.3,)]).time_reversal_partners()
    with pytest.raises(InputError, match="duplicate"):
        KPoints.from_fractional([(0.3,), (0.7,), (0.7,)]).time_reversal_partners()
    pair = KPoints.from_fractional([(0.3, 0.1), (0.7, 0.9)])
    assert pair.time_reversal_partners() == (1, 0)
    assert pair.is_time_reversal_invariant == (False, False)


def test_cartesian_coordinates_use_the_reciprocal_lattice() -> None:
    lattice = Lattice(((3.0, 0.2, 0.1), (-0.4, 2.5, 0.3)))
    mesh = KPoints.mesh((2, 3))
    cartesian = mesh.cartesian(lattice)
    assert cartesian.shape == (6, 3)
    np.testing.assert_allclose(
        cartesian, mesh.fractional_array() @ lattice.reciprocal_vectors_array()
    )
    with pytest.raises(InputError, match="do not fit"):
        KPoints.gamma(3).cartesian(lattice)


def test_mesh_size_recovers_meshes_and_rejects_other_sets() -> None:
    for size, shift in [((2, 3), None), ((3,), (0.5,)), ((2, 1, 2), (0.5, 0.0, 0.5))]:
        mesh = KPoints.mesh(size, shift=shift)
        assert mesh.mesh_size() == size
        rng = np.random.default_rng(len(size))
        order = [int(i) for i in rng.permutation(mesh.n_kpt)]
        shuffled = KPoints.from_fractional([mesh.fractional[i] for i in order])
        assert shuffled.mesh_size() == size
    assert KPoints.gamma().mesh_size() == (1, 1, 1)
    assert KPoints.from_fractional([(0.25,), (0.75,)]).mesh_size() == (2,)
    not_meshes: list[list[tuple[float, ...]]] = [
        [(0.0, 0.0), (0.3, 0.0), (0.7, 0.0)],
        [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)],
        [(0.0,), (0.25,), (0.5,)],
    ]
    for points in not_meshes:
        with pytest.raises(InputError, match="uniform mesh"):
            KPoints.from_fractional(points).mesh_size()
    with pytest.raises(InputError, match="equal weights"):
        KPoints.from_fractional([(0.0,), (0.5,)], weights=(0.25, 0.75)).mesh_size()

"""Functionals: metadata, named combinations and the evaluation on densities."""

import math

import numpy as np
import pytest

from hafnium.backend import Array
from hafnium.errors import InputError
from hafnium.xc import (
    NAMED_FUNCTIONALS,
    DensityOnGrid,
    Functional,
    Rung,
    component_info,
    evaluate_functional,
)
from tests.tolerances import FINITE_DIFFERENCE_TOL, STRUCTURE_TOL

FINITE_DIFFERENCE_STEP = 1e-4
"""Relative step of the central differences in every variable."""
# One point per row: rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b.
POINTS = np.array(
    [
        [0.3, 0.2, 0.05, 0.02, 0.04, 0.4, 0.3],
        [1.5, 1.5, 2.0, 1.9, 2.0, 5.0, 5.0],
        [0.01, 0.004, 1e-4, 2e-5, 3e-5, 0.02, 0.01],
        [0.7, 0.1, 0.3, 0.05, 0.02, 1.0, 0.2],
    ]
)


def density_at(points: Array) -> DensityOnGrid:
    return DensityOnGrid(
        np.ascontiguousarray(points[:, 0:2].T),
        np.ascontiguousarray(points[:, 2:5].T),
        np.ascontiguousarray(points[:, 5:7].T),
    )


def test_component_info_follows_libxc() -> None:
    pbe = component_info("GGA_X_PBE")
    assert (pbe.id, pbe.name, pbe.rung) == (101, "gga_x_pbe", Rung.GGA)
    assert pbe.exact_exchange == 0.0
    assert pbe.range_separation is None
    b3lyp = component_info("hyb_gga_xc_b3lyp")
    assert b3lyp.id == 402
    assert b3lyp.exact_exchange == pytest.approx(0.2)
    assert component_info("mgga_c_tpss").rung is Rung.META_GGA
    assert not component_info("mgga_x_r2scan").needs_laplacian
    rsh = component_info("hyb_gga_xc_wb97x_v")
    assert rsh.range_separation is not None
    omega, alpha, beta = rsh.range_separation
    assert (omega, alpha) == pytest.approx((0.3, 0.167))
    assert alpha + beta == pytest.approx(1.0)
    assert rsh.vv10 == (6.0, 0.01)
    with pytest.raises(InputError, match="unknown functional"):
        component_info("gga_x_nonsense")


def test_named_functionals_have_the_expected_rungs_and_exchange() -> None:
    expected = {
        "hf": (Rung.LDA, 1.0, False),
        "svwn": (Rung.LDA, 0.0, False),
        "pbe": (Rung.GGA, 0.0, False),
        "blyp": (Rung.GGA, 0.0, False),
        "b3lyp": (Rung.GGA, 0.2, False),
        "b3lyp5": (Rung.GGA, 0.2, False),
        "pbe0": (Rung.GGA, 0.25, False),
        "tpss": (Rung.META_GGA, 0.0, False),
        "r2scan": (Rung.META_GGA, 0.0, False),
        "m06-l": (Rung.META_GGA, 0.0, False),
        "m06-2x": (Rung.META_GGA, 0.54, False),
        "wb97x-v": (Rung.GGA, 0.167, True),
        "wb97m-v": (Rung.META_GGA, 0.15, True),
    }
    assert set(expected) == set(NAMED_FUNCTIONALS)
    for name, (rung, exact, separated) in expected.items():
        functional = Functional.named(name)
        assert functional.rung is rung, name
        assert functional.exact_exchange == pytest.approx(exact), name
        assert functional.is_range_separated is separated, name
        assert functional.is_hybrid is (exact != 0.0), name
        assert functional.needs_gradient is (rung is not Rung.LDA)
        assert functional.needs_tau is (rung is Rung.META_GGA)
    assert Functional.named("HF").components == ()
    assert Functional.named("wb97x-v").vv10 == (6.0, 0.01)
    assert Functional.named("wb97x-v").without_vv10().vv10 is None
    assert Functional.named("svwn").components == (("lda_x", 1.0), ("lda_c_vwn", 1.0))
    with pytest.raises(InputError, match="unknown functional"):
        Functional.named("b3lyp-d3")


def test_from_libxc_combines_fractions_and_guards_range_separation() -> None:
    mixed = Functional.from_libxc("gga_x_pbe", "gga_c_pbe", weights=(0.75, 1.0))
    assert mixed.components == (("gga_x_pbe", 0.75), ("gga_c_pbe", 1.0))
    assert mixed.exact_exchange == 0.0
    assert mixed.name == "gga_x_pbe + gga_c_pbe"
    pbe0 = Functional(
        "pbe0 by hand", (("gga_x_pbe", 0.75), ("gga_c_pbe", 1.0)), exact_exchange=0.25
    )
    assert "0.25 exact exchange" in pbe0.describe()
    rsh = Functional.from_libxc("hyb_gga_xc_wb97x_v")
    assert (rsh.omega, rsh.long_range_exchange) == pytest.approx((0.3, 0.833))
    assert "long-range exact exchange (omega 0.3)" in rsh.describe()
    with pytest.raises(InputError, match="weight one and alone"):
        Functional.from_libxc("hyb_gga_xc_wb97x_v", "hyb_mgga_xc_wb97m_v")
    with pytest.raises(InputError, match="weight one and alone"):
        Functional.from_libxc("hyb_gga_xc_wb97x_v", weights=(0.5,))
    with pytest.raises(InputError, match="one weight"):
        Functional.from_libxc("gga_x_pbe", weights=(1.0, 1.0))
    with pytest.raises(InputError, match="at least one"):
        Functional.from_libxc()


def test_functional_validation() -> None:
    with pytest.raises(InputError, match="positive range-separation"):
        Functional("bad", (("gga_x_pbe", 1.0),), long_range_exchange=0.5)
    with pytest.raises(InputError, match="omega is zero"):
        Functional("bad", (("gga_x_pbe", 1.0),), omega=0.3)
    with pytest.raises(InputError, match="components or exact exchange"):
        Functional("empty")
    with pytest.raises(InputError, match="finite"):
        Functional("bad", (("gga_x_pbe", math.nan),))
    with pytest.raises(InputError, match="unknown functional"):
        Functional("bad", (("gga_x_pbe_wrong", 1.0),))


def test_slater_exchange_matches_the_closed_form() -> None:
    densities = np.array([0.1, 0.5, 2.0])
    density = DensityOnGrid(np.stack([densities / 2, densities / 2]))
    values = evaluate_functional(Functional.from_libxc("lda_x"), density)
    fermi = (3.0 * densities / math.pi) ** (1 / 3)
    np.testing.assert_allclose(
        values.energy_per_particle, -0.75 * fermi, atol=STRUCTURE_TOL
    )
    np.testing.assert_allclose(
        values.vrho, np.stack([-fermi, -fermi]), atol=STRUCTURE_TOL
    )
    assert values.vsigma is None
    assert values.vtau is None


@pytest.mark.parametrize("name", ["pbe", "b3lyp", "tpss"])
def test_derivatives_match_finite_differences_of_the_energy_density(name: str) -> None:
    functional = Functional.named(name)

    def energy_density(points: Array) -> Array:
        values = evaluate_functional(functional, density_at(points))
        return (points[:, 0] + points[:, 1]) * values.energy_per_particle

    values = evaluate_functional(functional, density_at(POINTS))
    columns = {0: values.vrho[0], 1: values.vrho[1]}
    if values.vsigma is not None:
        columns.update({2: values.vsigma[0], 3: values.vsigma[1], 4: values.vsigma[2]})
    if values.vtau is not None:
        columns.update({5: values.vtau[0], 6: values.vtau[1]})
    assert (values.vtau is not None) is functional.needs_tau
    for column, expected in columns.items():
        step = FINITE_DIFFERENCE_STEP * POINTS[:, column]
        up, down = POINTS.copy(), POINTS.copy()
        up[:, column] += step
        down[:, column] -= step
        derivative = (energy_density(up) - energy_density(down)) / (2 * step)
        np.testing.assert_allclose(
            derivative, expected, rtol=FINITE_DIFFERENCE_TOL, atol=FINITE_DIFFERENCE_TOL
        )


def test_evaluation_requires_the_inputs_of_the_rung() -> None:
    density = density_at(POINTS)
    only_rho = DensityOnGrid(density.rho)
    with pytest.raises(InputError, match="gradients"):
        evaluate_functional(Functional.named("pbe"), only_rho)
    with pytest.raises(InputError, match="kinetic energy"):
        evaluate_functional(
            Functional.named("tpss"), DensityOnGrid(density.rho, density.sigma)
        )
    zeros = evaluate_functional(Functional.named("hf"), only_rho)
    assert not np.any(zeros.energy_per_particle)
    assert zeros.vrho.shape == (2, 4)
    with pytest.raises(InputError, match="2, g"):
        DensityOnGrid(np.zeros((3, 4)))
    with pytest.raises(InputError, match="3, g"):
        DensityOnGrid(np.zeros((2, 4)), np.zeros((2, 4)))
    with pytest.raises(InputError, match="2, g"):
        DensityOnGrid(np.zeros((2, 4)), np.zeros((3, 4)), np.zeros((3, 4)))
    assert density.n_points == 4

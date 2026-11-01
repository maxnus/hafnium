"""Functionals: metadata, named combinations and the evaluation on densities."""

# JAX's own annotations leave parts of its API unknown to a strict checker.
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

import dataclasses
import math

import numpy as np
import pytest

from hafnium.backend import Array, use_backend
from hafnium.errors import InputError
from hafnium.xc import (
    NAMED_FUNCTIONALS,
    DensityOnGrid,
    Functional,
    NoncollinearDensity,
    NoncollinearOptions,
    Rung,
    component_info,
    evaluate_functional,
    evaluate_noncollinear,
)
from tests.tolerances import (
    FINITE_DIFFERENCE_TOL,
    SPIN_ROTATION_QUADRATURE_TOL,
    STRUCTURE_TOL,
)

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


# ---------------------------------------------------------------- noncollinear

NONCOLLINEAR_STEP = 1e-6
"""Relative step of the central differences in the multi-collinear variables."""


def noncollinear_at(
    points: Array, *, polarization: float, twist: float
) -> NoncollinearDensity:
    """Spin densities of ``points`` as a noncollinear density.

    The alpha and beta quantities of every row are scaled to the
    ``polarization`` ``|m|/ρ`` and laid along a direction that differs per
    point; ``twist`` adds spin components of ``m``, ``∇m`` and ``τ_m`` that
    are not parallel to each other.
    """
    n = int(points.shape[0])
    rho = points[:, 0] + points[:, 1]
    s = polarization * rho
    axes = np.asarray(
        [[0.3, -0.5, 0.4], [0.0, 0.0, 1.0], [-0.6, 0.1, 0.2], [0.5, 0.5, 0.1]],
        dtype=np.float64,
    )[:n]
    axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    others: Array = np.asarray(
        [[0.8, 0.6, 0.15], [1.0, 0.0, 0.0], [0.2, 0.9, -0.3], [-0.4, 0.7, 0.6]],
        dtype=np.float64,
    )[:n]
    along: Array = np.einsum("ga,ga->g", others, axes)
    others = others - along[:, None] * axes
    others = others / np.linalg.norm(others, axis=1)[:, None]
    grad_rho = np.stack(
        [
            0.4 * np.sqrt(points[:, 2]),
            -0.2 * np.sqrt(points[:, 3]),
            0.3 * np.sqrt(points[:, 4]),
        ]
    )
    grad_s = polarization * grad_rho * np.asarray([[0.9], [0.7], [1.1]])
    tau = points[:, 5] + points[:, 6]
    tau_s = polarization * tau * 0.8
    directions: Array = axes + twist * 0.1 * others
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    m: Array = s * directions.T
    grad_m: Array = np.einsum("ga,ig->aig", axes, grad_s) + twist * 0.05 * np.einsum(
        "ga,ig->aig", others, grad_rho
    )
    tau_m: Array = tau_s * axes.T + twist * 0.1 * tau * others.T
    return NoncollinearDensity(rho, m, grad_rho, grad_m, tau, tau_m)


def collinear_of(density: NoncollinearDensity) -> DensityOnGrid:
    """The same density as collinear spin densities, ``m`` taken along its axis."""
    s = np.linalg.norm(density.m, axis=0)
    unit = density.m / s
    assert density.grad_rho is not None
    assert density.grad_m is not None
    assert density.tau is not None
    assert density.tau_m is not None
    grad_s: Array = np.einsum("ag,aig->ig", unit, density.grad_m)
    grad_alpha: Array = 0.5 * (density.grad_rho + grad_s)
    grad_beta: Array = 0.5 * (density.grad_rho - grad_s)
    tau_s: Array = np.einsum("ag,ag->g", unit, density.tau_m)
    return DensityOnGrid(
        np.stack([0.5 * (density.rho + s), 0.5 * (density.rho - s)]),
        np.stack(
            [
                np.sum(grad_alpha * grad_alpha, axis=0),
                np.sum(grad_alpha * grad_beta, axis=0),
                np.sum(grad_beta * grad_beta, axis=0),
            ]
        ),
        np.stack([0.5 * (density.tau + tau_s), 0.5 * (density.tau - tau_s)]),
    )


@pytest.mark.parametrize("name", ["svwn", "pbe", "tpss"])
def test_collinear_densities_recover_the_spin_polarized_functional(name: str) -> None:
    """Spin variables along one axis per point reduce the sphere to the collinear form.

    An LDA is exact by its closed form; a GGA or meta-GGA at moderate
    polarization by the quadrature, whose error is far below rounding
    there, and at strong polarization by the collinear branch.
    """
    functional = Functional.named(name)
    for polarization, options in [
        (0.4, NoncollinearOptions()),
        (0.995, NoncollinearOptions(collinear_threshold=0.99)),
    ]:
        density = noncollinear_at(POINTS, polarization=polarization, twist=0.0)
        values = evaluate_noncollinear(functional, density, options)
        reference = evaluate_functional(functional, collinear_of(density))
        rho = collinear_of(density).rho
        np.testing.assert_allclose(
            values.energy_density,
            (rho[0] + rho[1]) * reference.energy_per_particle,
            atol=STRUCTURE_TOL,
        )
        np.testing.assert_allclose(
            values.v_rho,
            0.5 * (reference.vrho[0] + reference.vrho[1]),
            atol=STRUCTURE_TOL,
        )
        unit = density.m / np.linalg.norm(density.m, axis=0)
        np.testing.assert_allclose(
            values.v_m,
            0.5 * (reference.vrho[0] - reference.vrho[1]) * unit,
            atol=STRUCTURE_TOL,
        )
        assert (values.v_grad_rho is None) is (name == "svwn")
        assert (values.v_tau is None) is (name != "tpss")
        if values.v_tau is not None and reference.vtau is not None:
            np.testing.assert_allclose(
                values.v_tau,
                0.5 * (reference.vtau[0] + reference.vtau[1]),
                atol=STRUCTURE_TOL,
            )


@pytest.mark.parametrize("name", ["svwn", "pbe", "tpss"])
@pytest.mark.parametrize("threshold", [None, 0.3])
def test_noncollinear_derivatives_match_finite_differences(
    name: str, threshold: float | None
) -> None:
    """On the sphere, and in the collinear branch when the threshold lies below.

    The spin variables are twisted out of each other's directions, so the
    rotation of the local axis with ``m`` enters the derivatives of the
    collinear branch.
    """
    functional = Functional.named(name)
    options = NoncollinearOptions(lebedev_points=110, collinear_threshold=threshold)
    density = noncollinear_at(POINTS, polarization=0.5, twist=1.0)
    values = evaluate_noncollinear(functional, density, options)
    fields = [
        ("rho", values.v_rho),
        ("m", values.v_m),
        ("grad_rho", values.v_grad_rho),
        ("grad_m", values.v_grad_m),
        ("tau", values.v_tau),
        ("tau_m", values.v_tau_m),
    ]
    for field, derivative in fields:
        if derivative is None:
            continue
        array = getattr(density, field)
        for index in np.ndindex(*array.shape[:-1]):
            step = NONCOLLINEAR_STEP * np.maximum(np.abs(array[index]), 0.1)
            up = dataclasses.replace(density, **{field: array.copy()})
            down = dataclasses.replace(density, **{field: array.copy()})
            getattr(up, field)[index] += step
            getattr(down, field)[index] -= step
            difference = (
                evaluate_noncollinear(functional, up, options).energy_density
                - evaluate_noncollinear(functional, down, options).energy_density
            ) / (2 * step)
            np.testing.assert_allclose(
                difference,
                derivative[index],
                rtol=FINITE_DIFFERENCE_TOL,
                atol=FINITE_DIFFERENCE_TOL,
            )


def test_lda_is_invariant_under_spin_rotations_and_the_sphere_nearly_so() -> None:
    """A rotation in spin space leaves the energy density unchanged.

    Exactly for an LDA, whose closed form depends on ``|m|`` alone; a GGA
    on the sphere only to the quadrature error of the rule, which carries
    the octahedral group alone.
    """
    density = noncollinear_at(POINTS, polarization=0.5, twist=1.0)
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    ) @ np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(0.4), -np.sin(0.4)],
            [0.0, np.sin(0.4), np.cos(0.4)],
        ]
    )
    assert density.grad_m is not None
    assert density.tau_m is not None
    rotated = dataclasses.replace(
        density,
        m=rotation @ density.m,
        grad_m=np.einsum("ab,big->aig", rotation, density.grad_m),
        tau_m=rotation @ density.tau_m,
    )
    for name, tolerance in [
        ("svwn", STRUCTURE_TOL),
        ("pbe", SPIN_ROTATION_QUADRATURE_TOL),
    ]:
        functional = Functional.named(name)
        before = evaluate_noncollinear(functional, density)
        after = evaluate_noncollinear(functional, rotated)
        np.testing.assert_allclose(
            after.energy_density, before.energy_density, atol=tolerance
        )
        np.testing.assert_allclose(after.v_rho, before.v_rho, atol=tolerance)
        np.testing.assert_allclose(after.v_m, rotation @ before.v_m, atol=tolerance)


def test_aligned_rules_agree_with_a_fine_lebedev_rule_and_report_failures() -> None:
    """The adaptive rules reach the fine fixed rule; a hopeless precision is counted."""
    functional = Functional.named("pbe")
    density = noncollinear_at(POINTS, polarization=0.6, twist=1.0)
    adaptive = evaluate_noncollinear(
        functional, density, NoncollinearOptions(precision=1e-12)
    )
    fixed = evaluate_noncollinear(
        functional, density, NoncollinearOptions(lebedev_points=5810)
    )
    assert adaptive.unconverged == 0
    np.testing.assert_allclose(
        adaptive.energy_density, fixed.energy_density, atol=1e-10
    )
    assert adaptive.v_grad_m is not None
    assert fixed.v_grad_m is not None
    np.testing.assert_allclose(adaptive.v_m, fixed.v_m, atol=1e-9)
    np.testing.assert_allclose(adaptive.v_grad_m, fixed.v_grad_m, atol=1e-9)
    hopeless = evaluate_noncollinear(
        functional, density, NoncollinearOptions(precision=1e-300, max_level=0)
    )
    assert hopeless.unconverged == density.n_points
    # The weights scale the tolerance to one, which the first comparison meets.
    light = evaluate_noncollinear(
        functional,
        density,
        NoncollinearOptions(precision=1e-300, max_level=1),
        weights=np.full(density.n_points, 1e-300),
    )
    assert light.unconverged == 0


def test_vanishing_spin_variables_and_inputs_are_validated() -> None:
    density = noncollinear_at(POINTS, polarization=0.0, twist=0.0)
    functional = Functional.named("pbe")
    values = evaluate_noncollinear(functional, density)
    reference = evaluate_functional(functional, collinear_of_zero(density))
    np.testing.assert_allclose(
        values.energy_density,
        density.rho * reference.energy_per_particle,
        atol=STRUCTURE_TOL,
    )
    assert not np.any(values.v_m)
    assert values.v_grad_m is not None
    assert not np.any(values.v_grad_m)
    zeros = evaluate_noncollinear(Functional.named("hf"), density)
    assert not np.any(zeros.energy_density)
    assert zeros.v_m.shape == (3, 4)
    with pytest.raises(InputError, match="gradients"):
        evaluate_noncollinear(functional, NoncollinearDensity(density.rho, density.m))
    with pytest.raises(InputError, match="kinetic"):
        evaluate_noncollinear(
            Functional.named("tpss"),
            NoncollinearDensity(
                density.rho, density.m, density.grad_rho, density.grad_m
            ),
        )
    with pytest.raises(InputError, match="magnetization has shape"):
        NoncollinearDensity(density.rho, np.zeros((2, 4)))
    with pytest.raises(InputError, match="come together"):
        NoncollinearDensity(density.rho, density.m, density.grad_rho)
    with pytest.raises(InputError, match="Lebedev"):
        NoncollinearOptions(lebedev_points=100)
    with pytest.raises(InputError, match="positive"):
        NoncollinearOptions(collinear_threshold=0.0)
    with pytest.raises(InputError, match="positive"):
        NoncollinearOptions(precision=0.0)
    with pytest.raises(InputError, match="level"):
        NoncollinearOptions(max_level=8)
    with pytest.raises(InputError, match="weights have shape"):
        evaluate_noncollinear(functional, density, weights=np.ones(3))


def collinear_of_zero(density: NoncollinearDensity) -> DensityOnGrid:
    """The unpolarized spin densities of a density without magnetization."""
    assert density.grad_rho is not None
    assert density.tau is not None
    sigma = 0.25 * np.sum(density.grad_rho * density.grad_rho, axis=0)
    return DensityOnGrid(
        np.stack([0.5 * density.rho, 0.5 * density.rho]),
        np.stack([sigma, sigma, sigma]),
        np.stack([0.5 * density.tau, 0.5 * density.tau]),
    )


# ------------------------------------------------------------ the JAX rule


def jax_density(points: Array, xp: object) -> DensityOnGrid:
    host = density_at(points)
    asarray = xp.asarray  # type: ignore[attr-defined]
    return DensityOnGrid(asarray(host.rho), asarray(host.sigma), asarray(host.tau))


@pytest.mark.parametrize("name", ["lda", "pbe", "b3lyp", "tpss"])
def test_jax_evaluation_matches_numpy(name: str) -> None:
    pytest.importorskip("jax")
    functional = (
        Functional.named(name) if name != "lda" else Functional.from_libxc("lda_x")
    )
    expected = evaluate_functional(functional, density_at(POINTS))
    with use_backend("jax") as backend:
        values = evaluate_functional(functional, jax_density(POINTS, backend.namespace))
    for field in ("energy_per_particle", "vrho", "vsigma", "vtau"):
        got, want = getattr(values, field), getattr(expected, field)
        assert (got is None) == (want is None)
        if got is not None:
            np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-13)


@pytest.mark.parametrize("name", ["lda", "pbe", "tpss"])
def test_jvp_matches_finite_differences_of_the_derivatives(name: str) -> None:
    jax = pytest.importorskip("jax")
    functional = (
        Functional.named(name) if name != "lda" else Functional.from_libxc("lda_x")
    )
    rng = np.random.default_rng(11)
    direction = rng.standard_normal(POINTS.shape) * POINTS

    def stacked(points: Array) -> Array:
        """Energy density and the derivatives the rung reads, as one ``[rows, g]``."""
        values = evaluate_functional(functional, density_at(points))
        rows = [
            (points[:, 0] + points[:, 1]) * values.energy_per_particle,
            *values.vrho,
        ]
        if values.vsigma is not None:
            rows.extend(values.vsigma)
        if values.vtau is not None:
            rows.extend(values.vtau)
        return np.stack(rows)

    step = FINITE_DIFFERENCE_STEP
    numeric = (
        stacked(POINTS + step * direction) - stacked(POINTS - step * direction)
    ) / (2.0 * step)
    with use_backend("jax") as backend:
        xp = backend.namespace

        def traced(points: Array) -> Array:
            density = DensityOnGrid(
                xp.linalg.matrix_transpose(points[:, 0:2]),
                xp.linalg.matrix_transpose(points[:, 2:5]),
                xp.linalg.matrix_transpose(points[:, 5:7]),
            )
            values = evaluate_functional(functional, density)
            total = points[:, 0] + points[:, 1]
            rows = [total * values.energy_per_particle, values.vrho[0], values.vrho[1]]
            if values.vsigma is not None:
                rows.extend(values.vsigma[k] for k in range(3))
            if values.vtau is not None:
                rows.extend(values.vtau[k] for k in range(2))
            return xp.stack(rows)

        _, tangent = jax.jvp(traced, (xp.asarray(POINTS),), (xp.asarray(direction),))
        compiled = jax.jit(traced)(xp.asarray(POINTS))
    np.testing.assert_allclose(
        np.asarray(tangent),
        numeric,
        rtol=FINITE_DIFFERENCE_TOL,
        atol=FINITE_DIFFERENCE_TOL,
    )
    np.testing.assert_allclose(np.asarray(compiled), stacked(POINTS), atol=1e-13)


def test_grad_of_the_integrated_energy_is_the_potential() -> None:
    jax = pytest.importorskip("jax")
    functional = Functional.named("pbe")
    weights = np.array([0.5, 1.0, 2.0, 0.25])
    with use_backend("jax") as backend:
        xp = backend.namespace
        host = density_at(POINTS)
        sigma = xp.asarray(host.sigma)

        def energy(rho: Array) -> Array:
            values = evaluate_functional(functional, DensityOnGrid(rho, sigma))
            return xp.sum(
                xp.asarray(weights) * (rho[0] + rho[1]) * values.energy_per_particle
            )

        gradient = jax.grad(energy)(xp.asarray(host.rho))
    expected = evaluate_functional(functional, host)
    np.testing.assert_allclose(
        np.asarray(gradient), weights * expected.vrho, atol=1e-12
    )

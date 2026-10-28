"""Kohn–Sham through the structured SCF: consistency across classes and terms."""

import dataclasses

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.grid import GridSpec
from hafnium.integrals import IntegralEngine
from hafnium.molecule import Molecule
from hafnium.scf import (
    EnergyTerms,
    ExactCoulombExchange,
    ExchangeCorrelation,
    Hamiltonian,
    KohnShamOptions,
    ScfOptions,
    ScfResult,
    hartree_fock,
    kohn_sham,
    orbital_gradient,
    solve_scf,
)
from hafnium.structured import BlockMatrix, BlockStructure, Ring
from hafnium.symmetry import SymmetrySpec, detect_point_group
from hafnium.xc import Functional, NoncollinearOptions
from tests.hamiltonians import ENGINE
from tests.molecules import water, water_cation
from tests.shell_tables import LIBRARY
from tests.tolerances import (
    CONVERGED_DENSITY_TOL,
    DENSITY_FITTING_ENERGY_TOL,
    ENERGY_TOL,
    FINITE_DIFFERENCE_TOL,
    GRID_TOL,
    NONCOLLINEAR_LIMIT_TOL,
    SPIN_TOL,
)
from tests.unit.test_scf import check_state

COLLINEAR_CLASSES = ["RHF", "complex RHF", "UHF", "complex UHF", "paired UHF"]
SPINOR_CLASSES = ["GHF", "complex GHF", "paired GHF"]


@pytest.fixture(scope="module")
def water_basis() -> tuple[Molecule, BasisSet]:
    molecule = water()
    return molecule, BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY)


@pytest.fixture(scope="module")
def restricted_pbe(water_basis: tuple[Molecule, BasisSet]) -> ScfResult:
    molecule, basis = water_basis
    return kohn_sham(ENGINE, basis, molecule, "pbe")


def test_exact_exchange_alone_is_hartree_fock(
    water_basis: tuple[Molecule, BasisSet],
) -> None:
    molecule, basis = water_basis
    reference = hartree_fock(ENGINE, basis, molecule)
    result = kohn_sham(ENGINE, basis, molecule, "hf")
    assert result.energy == pytest.approx(reference.energy, abs=ENERGY_TOL)
    assert result.energies.exchange_correlation == 0.0
    assert result.energies.exchange == pytest.approx(
        reference.energies.exchange, abs=ENERGY_TOL
    )


def test_exchange_fractions_add_up(water_basis: tuple[Molecule, BasisSet]) -> None:
    """Half the exchange through each builder, both with the full kernel, is HF."""
    molecule, basis = water_basis
    two_body = ExactCoulombExchange.from_molecule(ENGINE, basis, molecule)
    split = Functional(
        "split", (), exact_exchange=0.5, long_range_exchange=0.5, omega=0.4
    )
    exchange_correlation = ExchangeCorrelation.from_molecule(
        ENGINE, basis, molecule, split, grid=GridSpec(level=0)
    )
    hamiltonian = Hamiltonian.from_molecule(
        ENGINE,
        basis,
        molecule,
        two_body,
        exchange_correlation=exchange_correlation,
        long_range_two_body=two_body,
    )
    result = solve_scf(hamiltonian, SymmetrySpec.from_name("RHF"))
    reference = hartree_fock(ENGINE, basis, molecule)
    assert result.energy == pytest.approx(reference.energy, abs=ENERGY_TOL)


@pytest.mark.parametrize("name", COLLINEAR_CLASSES)
def test_collinear_classes_recover_the_restricted_energy(
    name: str, water_basis: tuple[Molecule, BasisSet], restricted_pbe: ScfResult
) -> None:
    molecule, basis = water_basis
    result = kohn_sham(
        ENGINE, basis, molecule, "pbe", symmetry=SymmetrySpec.from_name(name)
    )
    assert result.energy == pytest.approx(restricted_pbe.energy, abs=ENERGY_TOL)
    assert result.energies.exchange == 0.0
    assert result.energies.exchange_correlation == pytest.approx(
        restricted_pbe.energies.exchange_correlation, abs=ENERGY_TOL
    )
    assert result.spin.square == pytest.approx(0.0, abs=SPIN_TOL)
    np.testing.assert_allclose(
        result.density.to_dense(),
        restricted_pbe.density.to_dense(),
        atol=CONVERGED_DENSITY_TOL,
    )


@pytest.mark.parametrize("name", SPINOR_CLASSES)
def test_spinor_classes_recover_the_restricted_energy(
    name: str, water_basis: tuple[Molecule, BasisSet], restricted_pbe: ScfResult
) -> None:
    """A closed shell has no magnetization, so the sphere is never sampled."""
    molecule, basis = water_basis
    result = kohn_sham(
        ENGINE, basis, molecule, "pbe", symmetry=SymmetrySpec.from_name(name)
    )
    assert result.energy == pytest.approx(restricted_pbe.energy, abs=ENERGY_TOL)
    assert result.energies.exchange_correlation == pytest.approx(
        restricted_pbe.energies.exchange_correlation, abs=ENERGY_TOL
    )
    assert result.spin.vector == pytest.approx((0.0, 0.0, 0.0), abs=SPIN_TOL)
    np.testing.assert_allclose(
        result.density.to_dense(),
        restricted_pbe.density.to_dense(),
        atol=CONVERGED_DENSITY_TOL,
    )


@pytest.mark.parametrize(
    ("name", "functional"),
    [("GHF", "svwn"), ("complex GHF", "svwn"), ("GHF", "pbe"), ("complex GHF", "tpss")],
)
def test_spinor_classes_recover_the_unrestricted_energy(
    name: str, functional: str
) -> None:
    """The doublet cation stays collinear in the generalized classes.

    An LDA takes the closed form at every point; a GGA or meta-GGA the
    aligned rules, exact for a collinear density to their precision.
    """
    cation = water_cation()
    basis = BasisSet.for_atoms(cation, "sto-3g", library=LIBRARY)
    # The meta-GGA's second derivatives are costly: a coarse grid for it.
    options = KohnShamOptions(grid=GridSpec(level=0 if functional == "tpss" else 3))
    unrestricted = kohn_sham(
        ENGINE,
        basis,
        cation,
        functional,
        symmetry=SymmetrySpec.from_name("UHF"),
        options=options,
    )
    result = kohn_sham(
        ENGINE,
        basis,
        cation,
        functional,
        symmetry=SymmetrySpec.from_name(name),
        options=options,
    )
    tolerance = ENERGY_TOL if functional == "svwn" else NONCOLLINEAR_LIMIT_TOL
    assert result.energy == pytest.approx(unrestricted.energy, abs=tolerance)
    assert abs(result.spin.vector[2]) == pytest.approx(0.5, abs=SPIN_TOL)
    assert result.spin.square == pytest.approx(unrestricted.spin.square, abs=SPIN_TOL)


def test_rotated_spin_state_is_degenerate_and_stationary() -> None:
    """A rotation in spin space leaves the generalized energy unchanged.

    The rules over spin directions are aligned with the magnetization, so
    the energy and potential of the rotated collinear density follow the
    rotation to the precision of the rules, and the rotated determinant is
    a stationary point of the complex class.
    """
    cation = water_cation()
    basis = BasisSet.for_atoms(cation, "sto-3g", library=LIBRARY)
    unrestricted = kohn_sham(
        ENGINE, basis, cation, "pbe", symmetry=SymmetrySpec.from_name("UHF")
    )
    two_body = ExactCoulombExchange.from_molecule(ENGINE, basis, cation)
    exchange_correlation = ExchangeCorrelation.from_molecule(
        ENGINE, basis, cation, Functional.named("pbe")
    )
    hamiltonian = Hamiltonian.from_molecule(
        ENGINE, basis, cation, two_body, exchange_correlation=exchange_correlation
    )
    n = hamiltonian.n_ao
    theta, phi = 0.7, 0.4
    # exp(-i theta (cos phi sigma_x + sin phi sigma_y) / 2) tilts the spin axis.
    cosine, sine = np.cos(theta / 2), np.sin(theta / 2)
    spin_rotation = np.array(
        [
            [cosine, -1j * sine * np.exp(-1j * phi)],
            [-1j * sine * np.exp(1j * phi), cosine],
        ]
    )
    rotation = np.kron(spin_rotation, np.eye(n))
    structure = BlockStructure.generalized(Ring.COMPLEX)
    rotated = unrestricted.density.embed(structure).map_blocks(
        lambda block: rotation @ block @ rotation.conj().T
    )
    _, energies = hamiltonian.fock(rotated)
    assert energies.total == pytest.approx(
        unrestricted.energy, abs=NONCOLLINEAR_LIMIT_TOL
    )
    result = solve_scf(
        hamiltonian, SymmetrySpec.from_name("complex GHF"), guess=rotated
    )
    check_state(result, hamiltonian)
    assert result.n_iterations <= 3
    assert result.energy == pytest.approx(
        unrestricted.energy, abs=NONCOLLINEAR_LIMIT_TOL
    )
    s_z = unrestricted.spin.vector[2]
    expected = (
        s_z * np.sin(theta) * np.sin(phi),
        -s_z * np.sin(theta) * np.cos(phi),
        s_z * np.cos(theta),
    )
    assert result.spin.vector == pytest.approx(expected, abs=SPIN_TOL)


def noncollinear_density(
    engine: IntegralEngine, basis: BasisSet, molecule: Molecule, directions: np.ndarray
) -> BlockMatrix:
    """Spinor density of occupied RHF orbitals of ``molecule``, one spin direction each.

    Orbital ``i`` takes the spinor along the unit vector ``directions[i]``,
    ``(cos(θ/2), e^{iφ} sin(θ/2))``; the orbitals being orthonormal the
    density is idempotent, and its magnetization turns wherever another
    orbital dominates.
    """
    reference = hartree_fock(engine, basis, molecule)
    orbitals = reference.mo_coefficients.spin_blocks(orbitals=True)[0]
    occupied = np.flatnonzero(np.asarray(reference.mo_occupations.blocks[0]) > 0.0)
    assert len(occupied) >= len(directions)
    n = int(orbitals.shape[0])
    density = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    for index, direction in zip(occupied, directions, strict=False):
        theta = np.arccos(direction[2])
        phi = np.arctan2(direction[1], direction[0])
        spinor = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
        psi = np.kron(spinor, np.asarray(orbitals[:, index]))
        density += np.outer(psi, psi.conj())
    return BlockMatrix(BlockStructure.generalized(Ring.COMPLEX), (density,))


def test_noncollinear_potential_is_the_derivative_of_the_energy(
    water_basis: tuple[Molecule, BasisSet],
) -> None:
    """``E(D + εX) − E(D − εX) ≈ 2ε Re tr(V X)`` for a noncollinear density.

    Half the closed-shell density is mixed in, so that no point is
    strongly polarized and one fixed rule over spin directions makes the
    energy a smooth function of the density.
    """
    molecule, basis = water_basis
    directions = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.0, 0.8],
            [0.0, -0.8, 0.6],
        ]
    )
    structure = BlockStructure.generalized(Ring.COMPLEX)
    closed = hartree_fock(ENGINE, basis, molecule).density.embed(structure)
    density = 0.5 * (noncollinear_density(ENGINE, basis, molecule, directions) + closed)
    term = ExchangeCorrelation.from_molecule(
        ENGINE,
        basis,
        molecule,
        Functional.named("tpss"),
        grid=GridSpec(level=0),
    )
    term = dataclasses.replace(
        term, noncollinear=NoncollinearOptions(lebedev_points=110)
    )
    energy, potential = term.evaluate(density)
    assert energy < 0.0
    rng = np.random.default_rng(7)
    n = 2 * term.n_ao
    direction = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    direction = (direction + direction.conj().T) / 2
    step = 1e-4
    (dense,) = density.blocks
    plus, _ = term.evaluate(BlockMatrix(structure, (dense + step * direction,)))
    minus, _ = term.evaluate(BlockMatrix(structure, (dense - step * direction,)))
    difference = (plus - minus) / (2 * step)
    expected = float(np.real(np.sum(potential.blocks[0] * direction.T)))
    assert difference == pytest.approx(expected, abs=FINITE_DIFFERENCE_TOL)


def test_converged_state_is_stationary(
    water_basis: tuple[Molecule, BasisSet], restricted_pbe: ScfResult
) -> None:
    molecule, basis = water_basis
    two_body = ExactCoulombExchange.from_molecule(ENGINE, basis, molecule)
    exchange_correlation = ExchangeCorrelation.from_molecule(
        ENGINE, basis, molecule, Functional.named("pbe")
    )
    hamiltonian = Hamiltonian.from_molecule(
        ENGINE, basis, molecule, two_body, exchange_correlation=exchange_correlation
    )
    check_state(restricted_pbe, hamiltonian)
    fock, energies = hamiltonian.fock(restricted_pbe.density)
    assert energies.total == pytest.approx(restricted_pbe.energy, abs=ENERGY_TOL)
    assert energies.exchange_correlation < 0.0
    structure = restricted_pbe.structure
    gradient = orbital_gradient(
        fock,
        restricted_pbe.density,
        hamiltonian.metric(structure),
        hamiltonian.orthogonalizer(structure),
    )
    assert all(
        np.abs(block).max() < ScfOptions().gradient_tolerance
        for block in gradient.blocks
    )
    assert EnergyTerms(1.0, 2.0, 3.0, -4.0).electronic == 1.0
    assert EnergyTerms(1.0, 2.0, 3.0, -4.0, -0.5).total == 1.5


def test_point_group_blocking_recovers_the_energy(restricted_pbe: ScfResult) -> None:
    detection = detect_point_group(water())
    molecule = detection.orient(water())
    basis = BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY)
    spec = SymmetrySpec.from_name("UHF", point_group=detection.group)
    result = kohn_sham(ENGINE, basis, molecule, "pbe", symmetry=spec)
    assert result.structure.irreps is not None
    assert result.energy == pytest.approx(restricted_pbe.energy, abs=ENERGY_TOL)
    assert result.electrons_per_irrep() == {"A1": 6.0, "B1": 2.0, "B2": 2.0}


def test_hybrid_open_shell_and_fitting() -> None:
    cation = water_cation()
    basis = BasisSet.for_atoms(cation, "def2-svp", library=LIBRARY)
    auxiliary = BasisSet.for_atoms(cation, "def2-universal-jkfit", library=LIBRARY)
    unrestricted = SymmetrySpec.from_name("UHF")
    exact = kohn_sham(ENGINE, basis, cation, "b3lyp", symmetry=unrestricted)
    assert exact.energies.exchange < 0.0 < -exact.energies.exchange_correlation
    assert exact.spin.square == pytest.approx(0.75, abs=0.05)
    fitted = kohn_sham(
        ENGINE, basis, cation, "b3lyp", symmetry=unrestricted, auxiliary=auxiliary
    )
    assert fitted.energy == pytest.approx(exact.energy, abs=DENSITY_FITTING_ENERGY_TOL)
    assert fitted.energy != exact.energy


def test_grid_level_moves_the_energy_little(
    water_basis: tuple[Molecule, BasisSet], restricted_pbe: ScfResult
) -> None:
    molecule, basis = water_basis
    fine = kohn_sham(
        ENGINE, basis, molecule, "pbe", options=KohnShamOptions(grid=GridSpec(level=5))
    )
    assert fine.energy == pytest.approx(restricted_pbe.energy, abs=GRID_TOL)
    assert fine.energy != restricted_pbe.energy
    plain = kohn_sham(ENGINE, basis, molecule, "pbe", options=ScfOptions(diis_size=6))
    assert plain.energy == pytest.approx(restricted_pbe.energy, abs=ENERGY_TOL)


def test_kohn_sham_inputs_are_validated(water_basis: tuple[Molecule, BasisSet]) -> None:
    molecule, basis = water_basis
    two_body = ExactCoulombExchange.from_molecule(ENGINE, basis, molecule)
    pbe = ExchangeCorrelation.from_molecule(
        ENGINE, basis, molecule, Functional.named("pbe")
    )
    rsh = ExchangeCorrelation.from_molecule(
        ENGINE, basis, molecule, Functional.named("wb97x-v").without_vv10()
    )
    with pytest.raises(InputError, match="VV10"):
        ExchangeCorrelation.from_molecule(
            ENGINE, basis, molecule, Functional.named("wb97x-v")
        )
    with pytest.raises(InputError, match="long-range exact exchange: pass"):
        Hamiltonian.from_molecule(
            ENGINE, basis, molecule, two_body, exchange_correlation=rsh
        )
    with pytest.raises(InputError, match="no long-range exact exchange"):
        Hamiltonian.from_molecule(
            ENGINE,
            basis,
            molecule,
            two_body,
            exchange_correlation=pbe,
            long_range_two_body=two_body,
        )
    with pytest.raises(InputError, match="pass an exchange"):
        Hamiltonian.from_molecule(
            ENGINE, basis, molecule, two_body, long_range_two_body=two_body
        )
    other = BasisSet.for_atoms(molecule, "def2-svp", library=LIBRARY)
    with pytest.raises(InputError, match="covers"):
        Hamiltonian.from_molecule(
            ENGINE,
            other,
            molecule,
            ExactCoulombExchange.from_molecule(ENGINE, other, molecule),
            exchange_correlation=pbe,
        )
    spatial = BlockMatrix(BlockStructure.spatial(), (np.eye(basis.n_functions),))
    with pytest.raises(InputError, match="density with spin"):
        pbe.evaluate(spatial)
    density = BlockStructure.restricted()
    with pytest.raises(InputError, match="atomic orbitals"):
        pbe.evaluate(
            solve_scf(
                Hamiltonian.from_molecule(
                    ENGINE,
                    other,
                    molecule,
                    ExactCoulombExchange.from_molecule(ENGINE, other, molecule),
                ),
                SymmetrySpec.from_name("RHF"),
            ).density
        )
    assert density.is_collinear
    with pytest.raises(InputError, match="one cutoff radius"):
        dataclasses.replace(pbe, radii=pbe.radii[:-1])
    with pytest.raises(InputError, match="positive"):
        KohnShamOptions(basis_cutoff=0.0)
    with pytest.raises(InputError, match="at least one"):
        KohnShamOptions(batch_size=0)
    with pytest.raises(InputError, match="unknown functional"):
        kohn_sham(ENGINE, basis, molecule, "pbe-d3")

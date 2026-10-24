"""Kohn–Sham through the structured SCF: consistency across classes and terms."""

import dataclasses

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.grid import GridSpec
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
from hafnium.structured import BlockStructure
from hafnium.symmetry import SymmetrySpec, detect_point_group
from hafnium.xc import Functional
from tests.hamiltonians import ENGINE
from tests.molecules import water, water_cation
from tests.shell_tables import LIBRARY
from tests.tolerances import (
    CONVERGED_DENSITY_TOL,
    DENSITY_FITTING_ENERGY_TOL,
    ENERGY_TOL,
    GRID_TOL,
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
def test_spinor_classes_are_rejected(
    name: str, water_basis: tuple[Molecule, BasisSet]
) -> None:
    molecule, basis = water_basis
    with pytest.raises(InputError, match="collinear"):
        kohn_sham(ENGINE, basis, molecule, "pbe", symmetry=SymmetrySpec.from_name(name))


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

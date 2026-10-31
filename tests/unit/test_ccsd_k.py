"""DF-CCSD over k-points: classes agree and the energy folds onto the supercell."""

import pytest

from hafnium.cc import (
    CcsdOptions,
    CcsdPtResult,
    ccsd_from_periodic_fitting,
    ccsd_pt_from_periodic_fitting,
    df_ccsd,
    df_ccsd_pt,
)
from hafnium.errors import InputError
from hafnium.molecule import Molecule
from hafnium.mp import mp2_from_periodic_fitting
from hafnium.scf import ScfResult, solve_scf
from hafnium.structured import Symmetry
from hafnium.symmetry import KPoints, SymmetrySpec
from tests.cells import box, helium
from tests.hamiltonians import ENGINE
from tests.tolerances import CCSD_ENERGY_TOL, MP2_ENERGY_TOL
from tests.unit.test_mp2 import TIGHT
from tests.unit.test_mp2_k import Setup, setup
from tests.unit.test_scf import CLASSES, OPEN_SHELL_CLASSES

OPTIONS = CcsdOptions(energy_tolerance=1e-12, amplitude_tolerance=1e-10)


@pytest.fixture(scope="module")
def helium_pair() -> Setup:
    return setup(helium(3), KPoints.mesh((3, 1, 1)))


@pytest.fixture(scope="module")
def restricted_helium(helium_pair: Setup) -> tuple[ScfResult, CcsdPtResult]:
    _, _, _, mesh, fitting, hamiltonian = helium_pair
    reference = solve_scf(
        hamiltonian, SymmetrySpec.from_name("RHF").with_kpoints(mesh), options=TIGHT
    )
    result = ccsd_pt_from_periodic_fitting(reference, fitting, options=OPTIONS)
    return reference, result


@pytest.mark.parametrize("name", CLASSES)
def test_every_class_gives_the_closed_shell_energy_over_k(
    name: str, helium_pair: Setup, restricted_helium: tuple[ScfResult, CcsdPtResult]
) -> None:
    _, _, _, mesh, fitting, hamiltonian = helium_pair
    reference = solve_scf(
        hamiltonian, SymmetrySpec.from_name(name).with_kpoints(mesh), options=TIGHT
    )
    result = ccsd_pt_from_periodic_fitting(reference, fitting, options=OPTIONS)
    assert result.converged
    assert result.correlation_energy == pytest.approx(
        restricted_helium[1].correlation_energy, abs=CCSD_ENERGY_TOL
    )
    assert result.triples_correction == pytest.approx(
        restricted_helium[1].triples_correction, abs=CCSD_ENERGY_TOL
    )
    assert result.triples_correction < 0.0
    assert result.history[0].correlation_energy == pytest.approx(
        mp2_from_periodic_fitting(reference, fitting).correlation_energy,
        abs=MP2_ENERGY_TOL,
    )
    structure = result.doubles.structure
    reversible = "complex" not in name and name != "paired GHF"
    assert (Symmetry.TIME_REVERSAL in structure.symmetries) == reversible
    assert (Symmetry.SPIN_FLIP in structure.symmetries) == ("RHF" in name)
    assert result.singles.structure.symmetries == structure.symmetries


def test_supercell_at_gamma_folds_the_energy_per_cell(
    helium_pair: Setup, restricted_helium: tuple[ScfResult, CcsdPtResult]
) -> None:
    cell, _, _, mesh, _, _ = helium_pair
    supercell, basis, auxiliary, gamma, fitting, hamiltonian = setup(
        cell.supercell((3, 1, 1)), KPoints.gamma()
    )
    reference = solve_scf(
        hamiltonian, SymmetrySpec.from_name("RHF").with_kpoints(gamma), options=TIGHT
    )
    folded = df_ccsd_pt(
        ENGINE, basis, supercell, auxiliary, reference, kpoints=gamma, options=OPTIONS
    )
    assert folded.correlation_energy / mesh.n_kpt == pytest.approx(
        restricted_helium[1].correlation_energy, abs=CCSD_ENERGY_TOL
    )
    assert folded.reference_energy / mesh.n_kpt == pytest.approx(
        restricted_helium[0].energy, abs=CCSD_ENERGY_TOL
    )
    assert folded.triples_correction / mesh.n_kpt == pytest.approx(
        restricted_helium[1].triples_correction, abs=CCSD_ENERGY_TOL
    )
    assert folded.correlation_energy == pytest.approx(
        ccsd_from_periodic_fitting(
            reference, fitting, options=OPTIONS
        ).correlation_energy,
        abs=CCSD_ENERGY_TOL,
    )


def test_open_shell_classes_agree_and_fold_over_k() -> None:
    """A doublet cell: the open-shell classes agree and fold onto the supercell."""
    molecule = Molecule.from_atoms(
        [("He", (0.0, 0.0, 0.0)), ("H", (1.5, 0.0, 0.0))], unit="bohr", spin=1
    )
    cell = box(molecule, 8.0)
    mesh = KPoints.mesh((3, 1, 1))
    _, _, _, _, fitting, hamiltonian = setup(cell, mesh, "sto-3g")
    unrestricted = solve_scf(
        hamiltonian, SymmetrySpec.from_name("UHF").with_kpoints(mesh), options=TIGHT
    )
    energies: list[float] = []
    corrections: list[float] = []
    for name in OPEN_SHELL_CLASSES:
        spec = SymmetrySpec.from_name(name).with_kpoints(mesh)
        reference = solve_scf(
            hamiltonian,
            spec,
            guess=unrestricted.density.embed(spec.structure()),
            options=TIGHT,
        )
        result = ccsd_pt_from_periodic_fitting(reference, fitting, options=OPTIONS)
        energies.append(result.correlation_energy)
        corrections.append(result.triples_correction)
    assert energies[0] < 0.0
    assert all(e == pytest.approx(energies[0], abs=CCSD_ENERGY_TOL) for e in energies)
    assert all(
        c == pytest.approx(corrections[0], abs=CCSD_ENERGY_TOL) for c in corrections
    )
    supercell, basis, auxiliary, gamma, _, folded_hamiltonian = setup(
        cell.supercell((3, 1, 1)), KPoints.gamma(), "sto-3g"
    )
    reference = solve_scf(
        folded_hamiltonian,
        SymmetrySpec.from_name("UHF").with_kpoints(gamma),
        options=TIGHT,
    )
    folded = df_ccsd_pt(
        ENGINE, basis, supercell, auxiliary, reference, kpoints=gamma, options=OPTIONS
    )
    assert folded.correlation_energy / 3 == pytest.approx(
        energies[0], abs=CCSD_ENERGY_TOL
    )
    assert folded.triples_correction / 3 == pytest.approx(
        corrections[0], abs=CCSD_ENERGY_TOL
    )


def test_periodic_inputs_are_validated(helium_pair: Setup) -> None:
    cell, basis, auxiliary, mesh, _, hamiltonian = helium_pair
    reference = solve_scf(hamiltonian, SymmetrySpec.from_name("RHF").with_kpoints(mesh))
    with pytest.raises(InputError, match="needs the k-points"):
        df_ccsd(ENGINE, basis, cell, auxiliary, reference)

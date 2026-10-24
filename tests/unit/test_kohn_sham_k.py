"""Periodic Kohn–Sham through the structured SCF: classes, cells, terms and inputs."""

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.grid import GridSpec
from hafnium.molecule import Cell, Molecule
from hafnium.scf import (
    ExchangeCorrelation,
    Hamiltonian,
    KohnShamOptions,
    KPointDensityFittedCoulombExchange,
    ScfResult,
    hartree_fock,
    kohn_sham,
)
from hafnium.structured import BlockMatrix
from hafnium.symmetry import KPoints, SymmetrySpec, detect_space_group
from hafnium.xc import Functional
from tests.cells import box, diamond_helium, helium
from tests.hamiltonians import ENGINE
from tests.molecules import water
from tests.shell_tables import LIBRARY
from tests.tolerances import (
    ENERGY_TOL,
    MOLECULAR_LIMIT_TOL,
    PERIODIC_LEVEL_TOL,
    REFERENCE_GRADIENT_TOL,
    STRUCTURE_TOL,
)

AUXILIARY = "def2-universal-jkfit"
COLLINEAR_CLASSES = ["RHF", "complex RHF", "UHF", "complex UHF", "paired UHF"]
OPTIONS = KohnShamOptions(gradient_tolerance=REFERENCE_GRADIENT_TOL)


def bases(cell: Cell, name: str) -> tuple[BasisSet, BasisSet]:
    return (
        BasisSet.for_atoms(cell, name, library=LIBRARY),
        BasisSet.for_atoms(cell, AUXILIARY, library=LIBRARY),
    )


@pytest.fixture(scope="module")
def helium_cell() -> tuple[Cell, BasisSet, BasisSet, KPoints]:
    cell = helium(3)
    basis, auxiliary = bases(cell, "cc-pvdz")
    return cell, basis, auxiliary, KPoints.mesh((2, 1, 1))


def converge(
    cell: Cell,
    basis: BasisSet,
    auxiliary: BasisSet,
    functional: str,
    symmetry: SymmetrySpec,
    options: KohnShamOptions = OPTIONS,
) -> ScfResult:
    result = kohn_sham(
        ENGINE,
        basis,
        cell,
        functional,
        symmetry=symmetry,
        auxiliary=auxiliary,
        options=options,
    )
    assert result.converged
    return result


def test_exact_exchange_alone_is_hartree_fock(
    helium_cell: tuple[Cell, BasisSet, BasisSet, KPoints],
) -> None:
    cell, basis, auxiliary, mesh = helium_cell
    symmetry = SymmetrySpec.from_name("RHF").with_kpoints(mesh)
    reference = hartree_fock(
        ENGINE, basis, cell, symmetry=symmetry, auxiliary=auxiliary, options=OPTIONS
    )
    result = converge(cell, basis, auxiliary, "hf", symmetry)
    assert result.energy == pytest.approx(reference.energy, abs=ENERGY_TOL)
    assert result.energies.exchange_correlation == 0.0


@pytest.mark.parametrize("name", COLLINEAR_CLASSES)
def test_collinear_classes_recover_the_restricted_energy(
    helium_cell: tuple[Cell, BasisSet, BasisSet, KPoints], name: str
) -> None:
    cell, basis, auxiliary, mesh = helium_cell
    restricted = converge(
        cell, basis, auxiliary, "svwn", SymmetrySpec.from_name("RHF").with_kpoints(mesh)
    )
    result = converge(
        cell, basis, auxiliary, "svwn", SymmetrySpec.from_name(name).with_kpoints(mesh)
    )
    assert result.energy == pytest.approx(restricted.energy, abs=ENERGY_TOL)
    assert result.energies.exchange == 0.0
    assert result.energies.exchange_correlation < 0.0


def test_spinor_classes_are_rejected(
    helium_cell: tuple[Cell, BasisSet, BasisSet, KPoints],
) -> None:
    cell, basis, auxiliary, mesh = helium_cell
    with pytest.raises(InputError, match="collinear"):
        kohn_sham(
            ENGINE,
            basis,
            cell,
            "pbe",
            symmetry=SymmetrySpec.from_name("GHF").with_kpoints(mesh),
            auxiliary=auxiliary,
        )


def test_supercell_at_gamma_folds_the_mesh(
    helium_cell: tuple[Cell, BasisSet, BasisSet, KPoints],
) -> None:
    """The energy per cell at a 2×1×1 mesh is that of the doubled cell at Γ."""
    cell, basis, auxiliary, mesh = helium_cell
    primitive = converge(
        cell, basis, auxiliary, "pbe", SymmetrySpec.from_name("RHF").with_kpoints(mesh)
    )
    doubled = cell.supercell((2, 1, 1))
    basis_2, auxiliary_2 = bases(doubled, "cc-pvdz")
    folded = converge(
        doubled,
        basis_2,
        auxiliary_2,
        "pbe",
        SymmetrySpec.from_name("RHF").with_kpoints(KPoints.gamma()),
    )
    assert folded.energy == pytest.approx(2.0 * primitive.energy, abs=ENERGY_TOL)


def test_space_group_wedge_recovers_the_energy() -> None:
    cell = diamond_helium()
    basis, auxiliary = bases(cell, "cc-pvdz")
    mesh = KPoints.mesh((2, 2, 2))
    full = converge(
        cell, basis, auxiliary, "svwn", SymmetrySpec.from_name("RHF").with_kpoints(mesh)
    )
    group = detect_space_group(cell)
    reduced = converge(
        cell,
        basis,
        auxiliary,
        "svwn",
        SymmetrySpec.from_name("RHF").with_kpoints(mesh).with_space_group(group),
    )
    assert reduced.energy == pytest.approx(full.energy, abs=ENERGY_TOL)
    axis = reduced.structure.k_axis
    assert axis is not None
    assert len(axis.stored(True)) < mesh.n_kpt


def test_hybrid_takes_a_fraction_of_the_fitted_exchange(
    helium_cell: tuple[Cell, BasisSet, BasisSet, KPoints],
) -> None:
    cell, basis, auxiliary, mesh = helium_cell
    symmetry = SymmetrySpec.from_name("RHF").with_kpoints(mesh)
    hybrid = converge(cell, basis, auxiliary, "b3lyp", symmetry)
    pure = converge(cell, basis, auxiliary, "blyp", symmetry)
    assert hybrid.energies.exchange < 0.0
    assert pure.energies.exchange == 0.0
    assert hybrid.energies.exchange_correlation < 0.0


def test_grid_level_moves_the_energy_little(
    helium_cell: tuple[Cell, BasisSet, BasisSet, KPoints],
) -> None:
    cell, basis, auxiliary, mesh = helium_cell
    symmetry = SymmetrySpec.from_name("RHF").with_kpoints(mesh)
    coarse = converge(cell, basis, auxiliary, "pbe", symmetry)
    fine = converge(
        cell,
        basis,
        auxiliary,
        "pbe",
        symmetry,
        KohnShamOptions(
            grid=GridSpec(level=5), gradient_tolerance=REFERENCE_GRADIENT_TOL
        ),
    )
    assert fine.energy == pytest.approx(coarse.energy, abs=PERIODIC_LEVEL_TOL)
    assert fine.energy != coarse.energy


def test_molecule_in_a_box_recovers_the_molecular_term() -> None:
    """At Γ in a box of 80 Bohr the periodic term is the molecular one on its grid."""
    molecule = water()
    basis = BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY)
    # Unpruned with the base rule on the shared shells, so that the cell's
    # grid takes the same points as the molecule's.
    spec = GridSpec(partition="stratmann", prune=False, shared_angular_points=302)
    functional = Functional.named("pbe")
    state = kohn_sham(
        ENGINE, basis, molecule, functional, options=KohnShamOptions(grid=spec)
    )
    molecular = ExchangeCorrelation.from_molecule(
        ENGINE, basis, molecule, functional, grid=spec
    )
    energy, potential = molecular.evaluate(state.density)
    boxed = box(molecule, 80.0)
    gamma = KPoints.gamma()
    periodic = ExchangeCorrelation.from_cell(
        ENGINE,
        BasisSet.for_atoms(boxed, "sto-3g", library=LIBRARY),
        boxed,
        gamma,
        functional,
        grid=spec,
    )
    assert periodic.is_periodic
    assert periodic.translations is not None
    assert periodic.translations.shape[0] == 1
    structure = SymmetrySpec.from_name("RHF").with_kpoints(gamma).structure()
    density = BlockMatrix(structure, (np.asarray(state.density.blocks[0])[None],))
    energy_k, potential_k = periodic.evaluate(density)
    assert energy_k == pytest.approx(energy, abs=MOLECULAR_LIMIT_TOL)
    np.testing.assert_allclose(
        np.asarray(potential_k.blocks[0])[0],
        np.asarray(potential.blocks[0]),
        atol=MOLECULAR_LIMIT_TOL,
    )
    with pytest.raises(InputError, match="serves a cell"):
        periodic.evaluate(state.density)
    with pytest.raises(InputError, match="serves a molecule"):
        molecular.evaluate(density)


def test_periodic_inputs_are_validated(
    helium_cell: tuple[Cell, BasisSet, BasisSet, KPoints],
) -> None:
    cell, basis, auxiliary, mesh = helium_cell
    symmetry = SymmetrySpec.from_name("RHF").with_kpoints(mesh)
    with pytest.raises(InputError, match="k-point set"):
        kohn_sham(ENGINE, basis, cell, "pbe", auxiliary=auxiliary)
    with pytest.raises(InputError, match="auxiliary basis"):
        kohn_sham(ENGINE, basis, cell, "pbe", symmetry=symmetry)
    with pytest.raises(InputError, match="range-separated"):
        kohn_sham(
            ENGINE,
            basis,
            cell,
            Functional.named("wb97x-v").without_vv10(),
            symmetry=symmetry,
            auxiliary=auxiliary,
        )
    two_body = KPointDensityFittedCoulombExchange.from_cell(
        ENGINE, basis, auxiliary, cell, mesh
    )
    functional = Functional.named("pbe")
    other = ExchangeCorrelation.from_cell(
        ENGINE, basis, cell, KPoints.mesh((3, 1, 1)), functional
    )
    with pytest.raises(InputError, match="other k-points"):
        Hamiltonian.from_cell(
            ENGINE, basis, cell, mesh, two_body, exchange_correlation=other
        )
    molecule = Molecule(cell.atoms)
    molecular = ExchangeCorrelation.from_molecule(
        ENGINE,
        BasisSet.for_atoms(molecule, "cc-pvdz", library=LIBRARY),
        molecule,
        functional,
    )
    with pytest.raises(InputError, match="for a molecule"):
        Hamiltonian.from_cell(
            ENGINE, basis, cell, mesh, two_body, exchange_correlation=molecular
        )
    with pytest.raises(InputError, match="range-separated"):
        Hamiltonian.from_cell(
            ENGINE,
            basis,
            cell,
            mesh,
            two_body,
            exchange_correlation=ExchangeCorrelation.from_cell(
                ENGINE, basis, cell, mesh, Functional.named("wb97x-v").without_vv10()
            ),
        )
    periodic = ExchangeCorrelation.from_cell(ENGINE, basis, cell, mesh, functional)
    with pytest.raises(InputError, match="together"):
        ExchangeCorrelation(
            functional,
            periodic.grid,
            ENGINE,
            periodic.table,
            periodic.radii,
            kpoints=mesh,
        )
    with pytest.raises(InputError, match="translations"):
        ExchangeCorrelation(
            functional,
            periodic.grid,
            ENGINE,
            periodic.table,
            periodic.radii,
            kpoints=mesh,
            lattice=cell.lattice,
            translations=np.zeros((0, 3)),
        )
    assert periodic.translations is not None
    assert np.allclose(np.asarray(periodic.translations)[0], 0.0, atol=STRUCTURE_TOL)

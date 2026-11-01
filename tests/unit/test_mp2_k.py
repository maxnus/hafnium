"""DF-MP2 over k-points: classes agree, band folding, the transformation (ADR-0031)."""

import dataclasses

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.errors import InputError, StructureError
from hafnium.integrals import PeriodicDensityFitting, density_fitting_k
from hafnium.molecule import Cell, Molecule
from hafnium.mp import df_mp2, mp2_from_df_tensor, mp2_from_periodic_fitting
from hafnium.orbitals import OrbitalSpaces, Space, transform_periodic_fitting
from hafnium.scf import ExactCoulombExchange, Hamiltonian, ScfResult, solve_scf
from hafnium.structured import BlockMatrix, BlockVector, Symmetry
from hafnium.symmetry import KPoints, SymmetrySpec
from tests.cells import box, helium
from tests.hamiltonians import ENGINE
from tests.integrals_k import fitting_and_two_body
from tests.molecules import water
from tests.shell_tables import LIBRARY
from tests.tolerances import MP2_ENERGY_TOL, STRUCTURE_TOL
from tests.unit.test_mp2 import TIGHT
from tests.unit.test_scf import CLASSES, OPEN_SHELL_CLASSES

AUXILIARY = "def2-universal-jkfit"
Setup = tuple[Cell, BasisSet, BasisSet, KPoints, PeriodicDensityFitting, Hamiltonian]


def setup(cell: Cell, mesh: KPoints, basis_name: str = "cc-pvdz") -> Setup:
    basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
    auxiliary = BasisSet.for_atoms(cell, AUXILIARY, library=LIBRARY)
    fitting, two_body = fitting_and_two_body(ENGINE, basis, auxiliary, cell, mesh)
    hamiltonian = Hamiltonian.from_cell(ENGINE, basis, cell, mesh, two_body)
    return cell, basis, auxiliary, mesh, fitting, hamiltonian


@pytest.fixture(scope="module")
def helium_pair() -> Setup:
    return setup(helium(3), KPoints.mesh((3, 1, 1)))


@pytest.fixture(scope="module")
def restricted_helium(helium_pair: Setup) -> tuple[ScfResult, float]:
    _, _, _, mesh, fitting, hamiltonian = helium_pair
    reference = solve_scf(
        hamiltonian, SymmetrySpec.from_name("RHF").with_kpoints(mesh), options=TIGHT
    )
    return reference, mp2_from_periodic_fitting(reference, fitting).correlation_energy


@pytest.mark.parametrize("name", CLASSES)
def test_every_class_gives_the_closed_shell_energy_over_k(
    name: str, helium_pair: Setup, restricted_helium: tuple[ScfResult, float]
) -> None:
    _, _, _, mesh, fitting, hamiltonian = helium_pair
    reference = solve_scf(
        hamiltonian, SymmetrySpec.from_name(name).with_kpoints(mesh), options=TIGHT
    )
    result = mp2_from_periodic_fitting(reference, fitting)
    assert result.correlation_energy == pytest.approx(
        restricted_helium[1], abs=MP2_ENERGY_TOL
    )
    assert result.correlation_energy < 0.0
    structure = result.amplitudes.structure
    assert all(axis.momentum(0) == (0, 0, 0) for axis in structure.axes)
    # The mesh holds Γ, where a quaternionic block cannot derive by time reversal.
    reversible = "complex" not in name and name != "paired GHF"
    assert (Symmetry.TIME_REVERSAL in structure.symmetries) == reversible
    assert (Symmetry.SPIN_FLIP in structure.symmetries) == ("RHF" in name)
    if structure.symmetries:
        assert structure.n_stored < len(structure.keys)


def test_supercell_at_gamma_folds_the_energy_per_cell(
    helium_pair: Setup, restricted_helium: tuple[ScfResult, float]
) -> None:
    cell, _, _, mesh, _, _ = helium_pair
    supercell, basis, auxiliary, gamma, fitting, hamiltonian = setup(
        cell.supercell((3, 1, 1)), KPoints.gamma()
    )
    reference = solve_scf(
        hamiltonian, SymmetrySpec.from_name("RHF").with_kpoints(gamma), options=TIGHT
    )
    folded = df_mp2(ENGINE, basis, supercell, auxiliary, reference, kpoints=gamma)
    assert folded.correlation_energy / mesh.n_kpt == pytest.approx(
        restricted_helium[1], abs=MP2_ENERGY_TOL
    )
    assert folded.reference_energy / mesh.n_kpt == pytest.approx(
        restricted_helium[0].energy, abs=MP2_ENERGY_TOL
    )
    assert folded.correlation_energy == pytest.approx(
        mp2_from_periodic_fitting(reference, fitting).correlation_energy,
        abs=MP2_ENERGY_TOL,
    )


def test_open_shell_classes_agree_and_fold_over_k() -> None:
    """A doublet cell: the open-shell classes agree and fold onto the supercell.

    PySCF has no k-point UMP2 or GMP2 to compare with, so the unrestricted
    path is checked against the generalized one and against the supercell
    at Γ, which exercises the blocks derived by time reversal.
    """
    # Helium hydride in a box of 8 Bohr: the bands are narrow enough for
    # whole levels across the mesh (the cell of the k-point reference tests).
    molecule = Molecule.from_atoms(
        [("He", (0.0, 0.0, 0.0)), ("H", (1.5, 0.0, 0.0))], unit="bohr", spin=1
    )
    cell = box(molecule, 8.0)
    mesh = KPoints.mesh((3, 1, 1))
    _, _, _, _, fitting, hamiltonian = setup(cell, mesh, "sto-3g")
    # The generalized classes start from the unrestricted density: from a
    # spin-averaged guess their frontier levels are degenerate across the
    # mesh and no whole-level filling exists.
    unrestricted = solve_scf(
        hamiltonian, SymmetrySpec.from_name("UHF").with_kpoints(mesh), options=TIGHT
    )
    energies: list[float] = []
    for name in OPEN_SHELL_CLASSES:
        spec = SymmetrySpec.from_name(name).with_kpoints(mesh)
        reference = solve_scf(
            hamiltonian,
            spec,
            guess=unrestricted.density.embed(spec.structure()),
            options=TIGHT,
        )
        energies.append(
            mp2_from_periodic_fitting(reference, fitting).correlation_energy
        )
    assert energies[0] < 0.0
    assert all(e == pytest.approx(energies[0], abs=MP2_ENERGY_TOL) for e in energies)
    supercell, basis, auxiliary, gamma, _, folded_hamiltonian = setup(
        cell.supercell((3, 1, 1)), KPoints.gamma(), "sto-3g"
    )
    reference = solve_scf(
        folded_hamiltonian,
        SymmetrySpec.from_name("UHF").with_kpoints(gamma),
        options=TIGHT,
    )
    folded = df_mp2(ENGINE, basis, supercell, auxiliary, reference, kpoints=gamma)
    assert folded.correlation_energy / 3 == pytest.approx(
        energies[0], abs=MP2_ENERGY_TOL
    )


def test_transformation_matches_the_explicit_contraction_over_k(
    helium_pair: Setup,
) -> None:
    _, _, _, mesh, fitting, hamiltonian = helium_pair
    reference = solve_scf(hamiltonian, SymmetrySpec.from_name("UHF").with_kpoints(mesh))
    spaces = OrbitalSpaces.from_scf(reference)
    assert spaces.sizes(Space.OCCUPIED) == (2,) * 6
    assert spaces.sizes(Space.VIRTUAL) == (8,) * 6
    fitted = transform_periodic_fitting(fitting, spaces, Space.OCCUPIED, Space.VIRTUAL)
    signed = transform_periodic_fitting(
        fitting, spaces, Space.VIRTUAL, Space.OCCUPIED, signed=True
    )
    df_tensor, signature = fitting.df_tensor(), fitting.signature()
    occupied = spaces.materialized(Space.OCCUPIED)
    virtual = spaces.materialized(Space.VIRTUAL)
    n_kpt = mesh.n_kpt
    assert fitted.structure.n_stored == 2 * n_kpt * n_kpt
    for key in fitted.structure.stored_keys:
        transfer, bra, ket = key
        spin, k1 = divmod(bra, n_kpt)
        _, k2 = divmod(ket, n_kpt)
        assert int(fitting.transfer_index[k1, k2]) == transfer
        expected = np.einsum(
            "up,Puv,vq->Ppq",
            occupied[spin][k1].conj(),
            df_tensor[k1, k2],
            virtual[spin][k2],
        ) / np.sqrt(n_kpt)
        np.testing.assert_allclose(fitted.block(key), expected, atol=STRUCTURE_TOL)
        swapped = np.einsum(
            "up,Puv,vq->Ppq",
            virtual[spin][k2].conj(),
            df_tensor[k2, k1],
            occupied[spin][k1],
        ) / np.sqrt(n_kpt)
        np.testing.assert_allclose(
            signed.block((int(fitting.transfer_index[k2, k1]), ket, bra)),
            swapped * signature[k2, k1][:, None, None],
            atol=STRUCTURE_TOL,
        )
    # The energies at a derived point are those of its partner.
    np.testing.assert_allclose(
        spaces.sector_energies(Space.OCCUPIED, 2),
        spaces.sector_energies(Space.OCCUPIED, 1),
    )


def test_periodic_inputs_are_validated(helium_pair: Setup) -> None:
    cell, basis, auxiliary, mesh, fitting, hamiltonian = helium_pair
    reference = solve_scf(hamiltonian, SymmetrySpec.from_name("RHF").with_kpoints(mesh))
    with pytest.raises(InputError, match="needs the k-points"):
        df_mp2(ENGINE, basis, cell, auxiliary, reference)
    molecule = water()
    molecular_basis = BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY)
    molecular_auxiliary = BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY)
    with pytest.raises(InputError, match="no k-points"):
        df_mp2(
            ENGINE,
            molecular_basis,
            molecule,
            molecular_auxiliary,
            reference,
            kpoints=mesh,
        )
    with pytest.raises(InputError, match="different k-point sets"):
        transform_periodic_fitting(
            density_fitting_k(ENGINE, basis, auxiliary, cell, KPoints.mesh((2, 1, 1))),
            OrbitalSpaces.from_scf(reference),
            Space.OCCUPIED,
            Space.VIRTUAL,
        )
    with pytest.raises(InputError, match="belong to a cell"):
        mp2_from_df_tensor(reference, fitting.df_tensor()[0, 0])
    molecular = solve_scf(
        Hamiltonian.from_molecule(
            ENGINE,
            molecular_basis,
            molecule,
            ExactCoulombExchange.from_molecule(ENGINE, molecular_basis, molecule),
        ),
        SymmetrySpec.from_name("RHF"),
    )
    with pytest.raises(InputError, match="orbitals are molecular"):
        mp2_from_periodic_fitting(molecular, fitting)


def test_orbitals_at_invariant_points_must_be_real(helium_pair: Setup) -> None:
    """Two columns at Γ mixed with a complex coefficient are not real up to a phase."""
    _, _, _, mesh, fitting, hamiltonian = helium_pair
    reference = solve_scf(hamiltonian, SymmetrySpec.from_name("RHF").with_kpoints(mesh))
    coefficients = [np.array(block) for block in reference.mo_coefficients.blocks]
    gamma = coefficients[0][0]
    first, second = gamma[:, 0].copy(), gamma[:, 1].copy()
    gamma[:, 0] = (first + 1j * second) / np.sqrt(2)
    gamma[:, 1] = (first - 1j * second) / np.sqrt(2)
    mixed = dataclasses.replace(
        reference,
        mo_coefficients=BlockMatrix(reference.structure, tuple(coefficients)),
    )
    with pytest.raises(InputError, match="not real up to a phase"):
        OrbitalSpaces.from_scf(mixed)
    # Occupations that differ between the points of a block are a metal.
    occupations = reference.mo_occupations
    changed = [np.array(block) for block in occupations.blocks]
    changed[0][1, 1] = 0.0
    changed[0][1, 2] = 1.0
    metallic = dataclasses.replace(
        reference, mo_occupations=BlockVector(occupations.structure, tuple(changed))
    )
    with pytest.raises(InputError, match="differ between its k-points"):
        OrbitalSpaces.from_scf(metallic)
    fitted = transform_periodic_fitting(
        fitting, OrbitalSpaces.from_scf(reference), Space.OCCUPIED, Space.VIRTUAL
    )
    with pytest.raises(StructureError, match="vanishes by symmetry"):
        fitted.block((0, 0, 1))

"""DF-MP2 over block tensors: one formula for every class (ADR-0030)."""

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.correlated import (
    OrbitalSpaces,
    Space,
    transform_df_tensor,
)
from hafnium.errors import InputError, StructureError
from hafnium.integrals import DensityFitting
from hafnium.molecule import Molecule
from hafnium.mp import Mp2Result, df_mp2, mp2_from_df_tensor
from hafnium.scf import (
    AveragedOccupation,
    DensityFittedCoulombExchange,
    Hamiltonian,
    ScfOptions,
    ScfResult,
    solve_scf,
)
from hafnium.structured import (
    BlockMatrix,
    BlockStructure,
    BlockVector,
    KAxis,
    TensorAxis,
    TensorStructure,
)
from hafnium.symmetry import SymmetrySpec, detect_point_group
from tests.hamiltonians import ENGINE, hamiltonian_for
from tests.molecules import hydrogen, water, water_cation
from tests.shell_tables import LIBRARY
from tests.tolerances import ENERGY_TOL, MP2_ENERGY_TOL, STRUCTURE_TOL
from tests.unit.test_scf import CLASSES, OPEN_SHELL_CLASSES
from tests.unit.test_structure import sectors

AUXILIARY = "def2-universal-jkfit"
# The correlation energy is first order in the orbital error, so the
# references are converged well beyond the default gradient tolerance.
TIGHT = ScfOptions(energy_tolerance=1e-13, gradient_tolerance=1e-9)


def fitted_tensor(hamiltonian: Hamiltonian) -> np.ndarray:
    two_body = hamiltonian.two_body
    assert isinstance(two_body, DensityFittedCoulombExchange)
    return two_body.df_tensor


@pytest.fixture(scope="module")
def water_hamiltonian() -> Hamiltonian:
    return hamiltonian_for(water(), "sto-3g", AUXILIARY)[1]


@pytest.fixture(scope="module")
def cation_hamiltonian() -> Hamiltonian:
    return hamiltonian_for(water_cation(), "sto-3g", AUXILIARY)[1]


@pytest.fixture(scope="module")
def restricted_mp2(water_hamiltonian: Hamiltonian) -> Mp2Result:
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name("RHF"), options=TIGHT
    )
    return mp2_from_df_tensor(reference, fitted_tensor(water_hamiltonian))


def dense_mp2(reference: ScfResult, df_tensor: np.ndarray) -> float:
    """``¼ Σ |<ij||ab>|² / Δ`` over spin orbitals from the fitted four-centre ERIs."""
    n = df_tensor.shape[1]
    eri = np.einsum("Puv,Pwx->uvwx", df_tensor, df_tensor)
    c = reference.mo_coefficients.to_dense(orbitals=True)
    energies = reference.mo_energies.to_dense()
    occupied = reference.mo_occupations.to_dense() > 0.5
    integrals = np.zeros((c.shape[1],) * 4, dtype=c.dtype)
    for first in (c[:n], c[n:]):
        half = np.einsum("up,vq,uvwx->pqwx", first.conj(), first, eri)
        for second in (c[:n], c[n:]):
            integrals += np.einsum("pqwx,wr,xs->pqrs", half, second.conj(), second)
    virtual = ~occupied
    coulomb: np.ndarray = integrals[occupied][:, virtual][:, :, occupied][
        :, :, :, virtual
    ]
    antisymmetrized: np.ndarray = coulomb.transpose(0, 2, 1, 3) - coulomb.transpose(
        0, 2, 3, 1
    )
    e_occ, e_vir = energies[occupied], energies[virtual]
    denominators = (
        e_occ[:, None, None, None]
        + e_occ[None, :, None, None]
        - e_vir[None, None, :, None]
        - e_vir[None, None, None, :]
    )
    return float(np.sum(np.abs(antisymmetrized) ** 2 / denominators).real) / 4


@pytest.mark.parametrize("name", CLASSES)
def test_every_class_gives_the_closed_shell_energy(
    name: str, water_hamiltonian: Hamiltonian, restricted_mp2: Mp2Result
) -> None:
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name(name), options=TIGHT
    )
    result = mp2_from_df_tensor(reference, fitted_tensor(water_hamiltonian))
    assert result.correlation_energy == pytest.approx(
        restricted_mp2.correlation_energy, abs=MP2_ENERGY_TOL
    )
    assert result.reference_energy == pytest.approx(reference.energy)
    assert result.total_energy == pytest.approx(
        reference.energy + result.correlation_energy
    )
    assert result.correlation_energy < 0.0
    amplitudes = result.amplitudes
    assert amplitudes.n_axes == 4
    assert [axis.conjugated for axis in amplitudes.structure.axes] == [
        False,
        False,
        True,
        True,
    ]
    assert (amplitudes.structure.n_stored < len(amplitudes.structure.keys)) == (
        reference.structure.derived_partner is not None
    )


@pytest.mark.parametrize("name", OPEN_SHELL_CLASSES)
def test_open_shell_classes_agree(name: str, cation_hamiltonian: Hamiltonian) -> None:
    unrestricted = solve_scf(
        cation_hamiltonian, SymmetrySpec.from_name("UHF"), options=TIGHT
    )
    expected = mp2_from_df_tensor(unrestricted, fitted_tensor(cation_hamiltonian))
    reference = solve_scf(
        cation_hamiltonian, SymmetrySpec.from_name(name), options=TIGHT
    )
    result = mp2_from_df_tensor(reference, fitted_tensor(cation_hamiltonian))
    assert result.correlation_energy == pytest.approx(
        expected.correlation_energy, abs=MP2_ENERGY_TOL
    )
    assert result.correlation_energy == pytest.approx(
        dense_mp2(reference, fitted_tensor(cation_hamiltonian)), abs=ENERGY_TOL
    )


@pytest.mark.parametrize("name", ["RHF", "complex RHF", "paired UHF", "paired GHF"])
def test_mp2_matches_the_dense_spin_orbital_formula(
    name: str, water_hamiltonian: Hamiltonian
) -> None:
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name(name), options=TIGHT
    )
    result = mp2_from_df_tensor(reference, fitted_tensor(water_hamiltonian))
    assert result.correlation_energy == pytest.approx(
        dense_mp2(reference, fitted_tensor(water_hamiltonian)), abs=ENERGY_TOL
    )


def test_amplitudes_are_antisymmetric(restricted_mp2: Mp2Result) -> None:
    amplitudes = restricted_mp2.amplitudes
    dense = amplitudes.to_dense()
    np.testing.assert_allclose(
        amplitudes.permute("ijab->jiab").to_dense(), -dense, atol=STRUCTURE_TOL
    )
    np.testing.assert_allclose(
        amplitudes.permute("ijab->ijba").to_dense(), -dense, atol=STRUCTURE_TOL
    )
    assert dense.dtype == np.float64


@pytest.mark.parametrize("name", ["RHF", "UHF", "GHF", "paired GHF"])
def test_blocked_reference_gives_the_same_energy(name: str) -> None:
    molecule = water()
    detection = detect_point_group(molecule)
    oriented = detection.orient(molecule)
    basis = BasisSet.for_atoms(oriented, "sto-3g", library=LIBRARY)
    auxiliary = BasisSet.for_atoms(oriented, AUXILIARY, library=LIBRARY)
    two_body = DensityFittedCoulombExchange.from_molecule(
        ENGINE, basis, auxiliary, oriented
    )
    plain = Hamiltonian.from_molecule(ENGINE, basis, oriented, two_body)
    blocked = Hamiltonian.from_molecule(
        ENGINE, basis, oriented, two_body, point_group=detection.group
    )
    spec = SymmetrySpec.from_name(name)
    expected = df_mp2(
        ENGINE, basis, oriented, auxiliary, solve_scf(plain, spec, options=TIGHT)
    )
    reference = solve_scf(
        blocked, spec.with_point_group(detection.group), options=TIGHT
    )
    result = df_mp2(ENGINE, basis, oriented, auxiliary, reference)
    assert result.correlation_energy == pytest.approx(
        expected.correlation_energy, abs=MP2_ENERGY_TOL
    )
    axes = result.amplitudes.structure.axes
    assert all(axis.structure.irreps is not None for axis in axes)
    assert result.amplitudes.structure.n_stored > expected.amplitudes.structure.n_stored
    # The fitted tensor itself needs the adapted auxiliary basis.
    with pytest.raises(InputError, match="adapted"):
        mp2_from_df_tensor(reference, two_body.df_tensor)


def test_transformation_matches_the_explicit_contraction(
    water_hamiltonian: Hamiltonian,
) -> None:
    reference = solve_scf(water_hamiltonian, SymmetrySpec.from_name("complex UHF"))
    spaces = OrbitalSpaces.from_scf(reference)
    df_tensor = fitted_tensor(water_hamiltonian)
    fitted = transform_df_tensor(df_tensor, spaces, Space.OCCUPIED, Space.VIRTUAL)
    for position, (occupied, virtual) in enumerate(
        zip(spaces.occupied.blocks, spaces.virtual.blocks, strict=True)
    ):
        expected = np.einsum("up,Puv,vq->Ppq", occupied.conj(), df_tensor, virtual)
        np.testing.assert_allclose(
            fitted.block((0, position, position)), expected, atol=STRUCTURE_TOL
        )
    assert fitted.structure.keys == ((0, 0, 0), (0, 1, 1))
    assert spaces.ring is fitted.structure.ring
    assert spaces.sizes(Space.OCCUPIED) == (5, 5)
    assert spaces.sizes(Space.VIRTUAL) == (2, 2)
    assert spaces.space_of(fitted.structure.axes[1]) is Space.OCCUPIED
    assert spaces.space_of(fitted.structure.axes[2]) is Space.VIRTUAL
    with pytest.raises(StructureError, match="neither"):
        spaces.space_of(fitted.structure.axes[0])
    denominators = spaces.denominators(
        TensorStructure(fitted.structure.axes[1:], ring=spaces.ring)
    )
    e_occ = spaces.occupied_energies.blocks[0]
    e_vir = spaces.virtual_energies.blocks[0]
    np.testing.assert_allclose(
        denominators.block((0, 0)), e_occ[:, None] - e_vir[None, :], atol=STRUCTURE_TOL
    )


def test_orbital_spaces_reject_what_perturbation_theory_cannot_use(
    water_hamiltonian: Hamiltonian,
) -> None:
    atom = Molecule.from_atoms([("H", (0.0, 0.0, 0.0))], spin=1)
    _, hamiltonian = hamiltonian_for(atom, "sto-3g", AUXILIARY)
    averaged = solve_scf(
        hamiltonian,
        SymmetrySpec.from_name("UHF"),
        occupation=AveragedOccupation(0.5, 0.5),
    )
    with pytest.raises(InputError, match="fractionally"):
        OrbitalSpaces.from_scf(averaged)
    reference = solve_scf(water_hamiltonian, SymmetrySpec.from_name("RHF"))
    spaces = OrbitalSpaces.from_scf(reference)
    with pytest.raises(InputError, match="atomic orbitals"):
        transform_df_tensor(np.zeros((3, 2, 2)), spaces, Space.OCCUPIED, Space.VIRTUAL)
    with pytest.raises(InputError, match="aux, ao, ao"):
        transform_df_tensor(np.zeros((3, 7, 6)), spaces, Space.OCCUPIED, Space.VIRTUAL)
    with pytest.raises(InputError, match="adapted"):
        transform_df_tensor(
            fitted_tensor(water_hamiltonian),
            spaces,
            Space.OCCUPIED,
            Space.VIRTUAL,
            auxiliary=sectors((2, 1)),
        )
    # A k axis without the momenta of a uniform mesh gives no tensor axis.
    periodic = BlockStructure.restricted().with_k_axis(KAxis((0,), (1.0,)))
    unlabelled = OrbitalSpaces(
        BlockMatrix(periodic, (np.zeros((1, 2, 1)),)),
        BlockMatrix(periodic, (np.zeros((1, 2, 1)),)),
        BlockVector(periodic, (np.zeros((1, 1)),)),
        BlockVector(periodic, (np.zeros((1, 1)),)),
    )
    with pytest.raises(StructureError, match="uniform mesh"):
        unlabelled.axis(Space.OCCUPIED, conjugated=True)
    with pytest.raises(InputError, match="transform_periodic_fitting"):
        transform_df_tensor(
            np.zeros((3, 2, 2)), unlabelled, Space.OCCUPIED, Space.VIRTUAL
        )
    with pytest.raises(StructureError, match="coefficient column and an energy"):
        OrbitalSpaces(
            spaces.occupied,
            spaces.virtual,
            spaces.virtual_energies,
            spaces.occupied_energies,
        )
    assert TensorAxis(periodic.with_k_axis(None), (1, 1)).n_sectors == 2


def test_hydrogen_molecule_mp2_from_the_convenience_entry_point() -> None:
    molecule = hydrogen()
    basis = BasisSet.for_atoms(molecule, "cc-pvdz", library=LIBRARY)
    auxiliary = BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY)
    two_body = DensityFittedCoulombExchange.from_molecule(
        ENGINE, basis, auxiliary, molecule
    )
    hamiltonian = Hamiltonian.from_molecule(ENGINE, basis, molecule, two_body)
    reference = solve_scf(hamiltonian, SymmetrySpec.from_name("RHF"), options=TIGHT)
    result = df_mp2(ENGINE, basis, molecule, auxiliary, reference)
    fitting = DensityFitting.build(ENGINE, basis, auxiliary, molecule)
    assert result.correlation_energy == pytest.approx(
        dense_mp2(reference, fitting.df_tensor()), abs=ENERGY_TOL
    )
    assert -0.04 < result.correlation_energy < -0.02

"""The SCF over irrep sectors (ADR-0023): constrained results equal unconstrained."""

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.errors import InputError, StructureError, SymmetryError
from hafnium.molecule import Molecule
from hafnium.scf import (
    AufbauOccupation,
    ExactCoulombExchange,
    Hamiltonian,
    ScfOptions,
    ScfResult,
    atomic_density_guess,
    hartree_fock,
    orbital_gradient,
    solve_scf,
)
from hafnium.symmetry import KPoints, PointGroup, SymmetrySpec, detect_point_group
from tests import molecules
from tests.hamiltonians import ENGINE
from tests.shell_tables import LIBRARY
from tests.tolerances import (
    CONVERGED_DENSITY_TOL,
    ENERGY_TOL,
    MO_ENERGY_TOL,
    SPIN_TOL,
    STRUCTURE_TOL,
)
from tests.unit.test_point_group_detection import moved
from tests.unit.test_scf import CLASSES, check_state

# Energy above the ground state of a configuration fixed per irrep.
EXCITATION_GAIN = 0.1


def oriented(molecule: Molecule) -> tuple[Molecule, PointGroup]:
    detection = detect_point_group(molecule)
    return detection.orient(molecule), detection.group


def adapted_hamiltonian(
    molecule: Molecule, basis_name: str, group: PointGroup
) -> tuple[BasisSet, Hamiltonian]:
    basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
    two_body = ExactCoulombExchange.from_molecule(ENGINE, basis, molecule)
    return basis, Hamiltonian.from_molecule(
        ENGINE, basis, molecule, two_body, point_group=group
    )


@pytest.fixture(scope="module")
def water() -> tuple[Molecule, PointGroup, BasisSet, Hamiltonian]:
    molecule, group = oriented(molecules.water())
    basis, hamiltonian = adapted_hamiltonian(molecule, "sto-3g", group)
    return molecule, group, basis, hamiltonian


@pytest.fixture(scope="module")
def plain_water(water: tuple[Molecule, PointGroup, BasisSet, Hamiltonian]) -> ScfResult:
    return solve_scf(water[3], SymmetrySpec.from_name("RHF"))


@pytest.mark.parametrize("name", CLASSES)
def test_every_class_blocked_over_c2v_recovers_the_energy(
    name: str,
    water: tuple[Molecule, PointGroup, BasisSet, Hamiltonian],
    plain_water: ScfResult,
) -> None:
    _, group, _, hamiltonian = water
    spec = SymmetrySpec.from_name(name, point_group=group)
    result = solve_scf(hamiltonian, spec)
    assert result.structure == hamiltonian.structure(spec)
    assert result.structure.irreps is not None
    assert result.structure.irreps.labels == ("A1", "B1", "B2")
    assert result.structure.describe().endswith("in C2v")
    assert result.energy == pytest.approx(plain_water.energy, abs=ENERGY_TOL)
    assert result.electrons_per_irrep() == {"A1": 6.0, "B1": 2.0, "B2": 2.0}
    assert result.spin.square == pytest.approx(0.0, abs=SPIN_TOL)
    check_state(result, hamiltonian)
    np.testing.assert_allclose(
        result.density.to_dense(),
        plain_water.density.to_dense(),
        atol=CONVERGED_DENSITY_TOL,
    )
    np.testing.assert_allclose(
        np.sort(result.mo_energies.to_dense()),
        np.sort(plain_water.mo_energies.to_dense()),
        atol=MO_ENERGY_TOL,
    )
    # The orbitals in the atomic basis rebuild the density.
    c = result.mo_coefficients.to_dense(orbitals=True)
    occupations = result.mo_occupations.to_dense()
    np.testing.assert_allclose(
        (c * occupations[None, :]) @ c.conj().T,
        result.density.to_dense(),
        atol=STRUCTURE_TOL,
    )


def test_blocked_density_is_stationary_without_the_blocking(
    water: tuple[Molecule, PointGroup, BasisSet, Hamiltonian],
    plain_water: ScfResult,
) -> None:
    _, group, _, hamiltonian = water
    blocked = solve_scf(hamiltonian, SymmetrySpec.from_name("UHF", point_group=group))
    structure = SymmetrySpec.from_name("UHF").structure()
    density = blocked.density.embed(structure)
    fock, energies = hamiltonian.fock(density)
    assert energies.total == pytest.approx(blocked.energy, abs=ENERGY_TOL)
    gradient = orbital_gradient(
        fock,
        density,
        hamiltonian.metric(structure),
        hamiltonian.orthogonalizer(structure),
    )
    assert all(
        np.abs(block).max() < ScfOptions().gradient_tolerance
        for block in gradient.blocks
    )
    restarted = solve_scf(hamiltonian, SymmetrySpec.from_name("UHF"), guess=density)
    assert restarted.n_iterations <= 2
    # And the plain solution enters the blocked structure losslessly.
    back = plain_water.density.embed(
        blocked.structure.with_ring(blocked.structure.stored_blocks[0].ring)
    )
    assert back.structure.irreps == blocked.structure.irreps
    np.testing.assert_allclose(
        back.to_dense(), plain_water.density.to_dense(), atol=STRUCTURE_TOL
    )


def test_atomic_guess_is_symmetric(
    water: tuple[Molecule, PointGroup, BasisSet, Hamiltonian],
) -> None:
    _, group, basis, hamiltonian = water
    spec = SymmetrySpec.from_name("UHF", point_group=group)
    blocked = hamiltonian.structure(spec)
    guess = atomic_density_guess(ENGINE, basis, blocked)
    plain = atomic_density_guess(ENGINE, basis, spec.with_point_group(None).structure())
    embedded = plain.embed(blocked)
    for a, b in zip(guess.blocks, embedded.blocks, strict=True):
        np.testing.assert_allclose(a, b, atol=STRUCTURE_TOL)


def test_subgroups_and_the_convenience_entry_point(
    water: tuple[Molecule, PointGroup, BasisSet, Hamiltonian],
    plain_water: ScfResult,
) -> None:
    molecule, _, basis, _ = water
    result = hartree_fock(
        ENGINE,
        basis,
        molecule,
        symmetry=SymmetrySpec.from_name("RHF", point_group="C2"),
    )
    assert result.structure.irreps is not None
    assert result.structure.irreps.labels == ("A", "B")
    assert result.energy == pytest.approx(plain_water.energy, abs=ENERGY_TOL)
    assert result.electrons_per_irrep() == {"A": 6.0, "B": 4.0}
    # Cs reflects through xy, which water in the C2v frame does not have.
    with pytest.raises(SymmetryError, match="standard frame"):
        hartree_fock(
            ENGINE,
            basis,
            molecule,
            symmetry=SymmetrySpec.from_name("RHF", point_group="Cs"),
        )


def test_methane_in_d2_resolves_the_degenerate_shell() -> None:
    molecule, group = oriented(molecules.methane())
    assert group.name == "D2"
    _, hamiltonian = adapted_hamiltonian(molecule, "sto-3g", group)
    plain = solve_scf(hamiltonian, SymmetrySpec.from_name("RHF"))
    blocked = solve_scf(hamiltonian, SymmetrySpec.from_name("RHF", point_group=group))
    assert blocked.energy == pytest.approx(plain.energy, abs=ENERGY_TOL)
    assert blocked.electrons_per_irrep() == {"A": 4.0, "B1": 2.0, "B2": 2.0, "B3": 2.0}
    labels = dict(
        zip(blocked.structure.irrep_labels, blocked.mo_energies.blocks, strict=True)
    )
    # The t2 shell is one orbital in each B irrep at the same energy.
    assert labels["B1"][0] == pytest.approx(labels["B2"][0], abs=MO_ENERGY_TOL)
    assert labels["B2"][0] == pytest.approx(labels["B3"][0], abs=MO_ENERGY_TOL)


def test_electrons_fixed_per_irrep_give_an_excited_configuration(
    water: tuple[Molecule, PointGroup, BasisSet, Hamiltonian],
    plain_water: ScfResult,
) -> None:
    _, group, _, hamiltonian = water
    restricted = SymmetrySpec.from_name("RHF", point_group=group)
    excited = solve_scf(
        hamiltonian, restricted, occupation=AufbauOccupation(5, 5, {"A1": 4, "B2": 4})
    )
    assert excited.electrons_per_irrep() == {"A1": 4.0, "B1": 2.0, "B2": 4.0}
    assert excited.energy > plain_water.energy + EXCITATION_GAIN
    check_state(excited, hamiltonian)
    # The paired classes take the same totals and reach the same state.
    for name, counts in [
        ("paired UHF", {"A1": 4, "B2": (2, 2)}),
        ("paired GHF", {"A1": 4, "B2": 4}),
    ]:
        paired = solve_scf(
            hamiltonian,
            SymmetrySpec.from_name(name, point_group=group),
            occupation=AufbauOccupation(5, 5, counts),
        )
        assert paired.energy == pytest.approx(excited.energy, abs=ENERGY_TOL)
        assert paired.electrons_per_irrep() == {"A1": 4.0, "B1": 2.0, "B2": 4.0}
    unrestricted = SymmetrySpec.from_name("UHF", point_group=group)
    triplet = solve_scf(
        hamiltonian,
        unrestricted,
        occupation=AufbauOccupation(6, 4, {"B1": (1, 0), "B2": (2, 1)}),
    )
    assert triplet.electrons_per_irrep() == {"A1": 6.0, "B1": 1.0, "B2": 3.0}
    assert triplet.spin.vector[2] == pytest.approx(1.0, abs=SPIN_TOL)
    assert triplet.energy > plain_water.energy
    # A total splits with the odd electron in alpha.
    same = solve_scf(
        hamiltonian, unrestricted, occupation=AufbauOccupation(6, 4, {"B1": 1, "B2": 3})
    )
    assert same.energy == pytest.approx(triplet.energy, abs=ENERGY_TOL)
    generalized = solve_scf(
        hamiltonian,
        SymmetrySpec.from_name("GHF", point_group=group),
        occupation=AufbauOccupation(6, 4, {"B1": 1, "B2": 3}),
        guess=triplet.density.embed(
            hamiltonian.structure(SymmetrySpec.from_name("GHF", point_group=group))
        ),
    )
    assert generalized.energy == pytest.approx(triplet.energy, abs=ENERGY_TOL)


def test_fixed_irrep_electrons_are_validated(
    water: tuple[Molecule, PointGroup, BasisSet, Hamiltonian],
) -> None:
    _, group, _, hamiltonian = water
    restricted = SymmetrySpec.from_name("RHF", point_group=group)
    unrestricted = SymmetrySpec.from_name("UHF", point_group=group)
    generalized = SymmetrySpec.from_name("GHF", point_group=group)

    def run(spec: SymmetrySpec, occupation: AufbauOccupation) -> ScfResult:
        return solve_scf(hamiltonian, spec, occupation=occupation)

    with pytest.raises(InputError, match="labelled"):
        run(restricted, AufbauOccupation(5, 5, {"A2": 2}))
    with pytest.raises(InputError, match="even"):
        run(restricted, AufbauOccupation(5, 5, {"A1": 5}))
    with pytest.raises(InputError, match="equal alpha and beta"):
        run(restricted, AufbauOccupation(5, 5, {"A1": (3, 2)}))
    with pytest.raises(InputError, match="which holds"):
        run(unrestricted, AufbauOccupation(5, 5, {"A1": (6, 3)}))
    with pytest.raises(InputError, match="left over"):
        run(restricted, AufbauOccupation(5, 5, {"A1": 4, "B1": 2, "B2": 2}))
    with pytest.raises(InputError, match="total per irrep"):
        run(generalized, AufbauOccupation(5, 5, {"A1": (3, 3)}))
    with pytest.raises(InputError, match="no irrep sectors"):
        run(SymmetrySpec.from_name("RHF"), AufbauOccupation(5, 5, {"A1": 6}))


def test_fixed_irrep_electrons_are_integers_and_copied() -> None:
    with pytest.raises(InputError, match="non-negative"):
        AufbauOccupation(5, 5, {"A1": -2})
    with pytest.raises(InputError, match="total or"):
        AufbauOccupation(5, 5, {"A1": (1, 2, 3)})  # type: ignore[dict-item]
    with pytest.raises(InputError, match="integers"):
        AufbauOccupation(5, 5, {"A1": 2.0})  # type: ignore[dict-item]
    with pytest.raises(InputError, match="integers"):
        AufbauOccupation(5, 5, {"A1": True})
    counts = {"A1": 6}
    policy = AufbauOccupation(5, 5, counts)
    counts["B1"] = 2
    assert dict(policy.irrep_electrons or {}) == {"A1": 6}


def test_structures_and_hamiltonians_must_agree(
    water: tuple[Molecule, PointGroup, BasisSet, Hamiltonian],
) -> None:
    molecule, group, basis, hamiltonian = water
    spec = SymmetrySpec.from_name("RHF", point_group=group)
    plain = Hamiltonian.from_molecule(ENGINE, basis, molecule, hamiltonian.two_body)
    with pytest.raises(InputError, match="needs the symmetry adaptation"):
        solve_scf(plain, spec)
    with pytest.raises(InputError, match="built for C2v"):
        hamiltonian.structure(SymmetrySpec.from_name("RHF", point_group="C2"))
    with pytest.raises(InputError, match="do not match the Hamiltonian's symmetry"):
        plain.metric(hamiltonian.structure(spec))
    with pytest.raises(StructureError, match="embed"):
        solve_scf(
            hamiltonian,
            spec,
            guess=solve_scf(plain, spec.with_point_group(None)).density,
        )
    with pytest.raises(InputError, match="space group"):
        SymmetrySpec.from_name("RHF", point_group=group).with_kpoints(KPoints.gamma())
    tilted = moved(molecules.water(), 3)
    with pytest.raises(SymmetryError, match="standard frame"):
        hartree_fock(
            ENGINE,
            BasisSet.for_atoms(tilted, "sto-3g", library=LIBRARY),
            tilted,
            symmetry=spec,
        )

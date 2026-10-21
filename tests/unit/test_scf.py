"""The SCF driver across the Fukutome classes: literature values, consistency."""

import numpy as np
import pytest

from hafnium.basis import BasisSet, Shell
from hafnium.errors import ConvergenceError, InputError, StructureError
from hafnium.molecule import Molecule
from hafnium.scf import (
    AufbauOccupation,
    AveragedOccupation,
    Diis,
    ExactCoulombExchange,
    Hamiltonian,
    ScfOptions,
    ScfResult,
    atomic_density,
    atomic_density_guess,
    canonical_orthogonalizer,
    core_hamiltonian_guess,
    electrons_per_block,
    hartree_fock,
    orbital_gradient,
    solve_scf,
    spin_expectation,
)
from hafnium.structured import (
    BlockMatrix,
    BlockStructure,
    BlockVector,
    Ring,
    is_kramers_symmetric,
)
from hafnium.symmetry import SymmetrySpec
from tests.hamiltonians import ENGINE, hamiltonian_for
from tests.molecules import hydrogen, water, water_cation
from tests.random_matrices import random_antisymmetric
from tests.shell_tables import LIBRARY
from tests.tolerances import (
    CONVERGED_DENSITY_TOL,
    DENSITY_FITTING_ENERGY_TOL,
    ENERGY_TOL,
    LITERATURE_TOL,
    MO_ENERGY_TOL,
    SPIN_TOL,
    STRUCTURE_TOL,
)

CLASSES = [
    "RHF",
    "complex RHF",
    "UHF",
    "complex UHF",
    "paired UHF",
    "GHF",
    "complex GHF",
    "paired GHF",
]
OPEN_SHELL_CLASSES = ["UHF", "complex UHF", "GHF", "complex GHF"]
# Energy lowering that counts as a symmetry-broken solution in stretched H2.
SYMMETRY_BREAKING_GAIN = 1e-2


@pytest.fixture(scope="module")
def water_hamiltonian() -> Hamiltonian:
    return hamiltonian_for(water(), "sto-3g")[1]


@pytest.fixture(scope="module")
def restricted_water(water_hamiltonian: Hamiltonian) -> ScfResult:
    return solve_scf(water_hamiltonian, SymmetrySpec.from_name("RHF"))


def check_state(result: ScfResult, hamiltonian: Hamiltonian) -> None:
    """Hermitian, idempotent density with the right electron count (PLAN.md §8)."""
    overlap = hamiltonian.metric(result.structure)
    density = result.density
    assert result.converged
    assert all(
        np.allclose(block, block.conj().T, atol=STRUCTURE_TOL)
        for block in density.blocks
    )
    assert float(np.real(density.trace_product(overlap))) == pytest.approx(
        hamiltonian.n_electrons, abs=STRUCTURE_TOL
    )
    projected = density @ overlap @ density
    for block, again in zip(density.blocks, projected.blocks, strict=True):
        np.testing.assert_allclose(again, block, atol=STRUCTURE_TOL)


def test_h2_sto3g_matches_szabo_and_ostlund() -> None:
    # Szabo and Ostlund, Modern Quantum Chemistry, section 3.5.2 (R = 1.4 Bohr).
    molecule = hydrogen()
    basis = BasisSet.for_molecule(molecule, "sto-3g", library=LIBRARY)
    result = hartree_fock(ENGINE, basis, molecule)
    assert result.energies.electronic == pytest.approx(-1.8310, abs=LITERATURE_TOL)
    assert result.energy == pytest.approx(-1.1167, abs=LITERATURE_TOL)
    assert result.mo_energies.blocks[0][0] == pytest.approx(-0.5782, abs=LITERATURE_TOL)
    assert result.mo_energies.blocks[0][1] == pytest.approx(0.6703, abs=LITERATURE_TOL)


def test_hydrogen_atom_in_every_open_shell_class() -> None:
    # One s function: the energy is the expectation value of the core
    # Hamiltonian in the normalized STO-3G 1s function, -0.4666 Hartree.
    atom = Molecule.from_atoms([("H", (0.0, 0.0, 0.0))], spin=1)
    basis, hamiltonian = hamiltonian_for(atom, "sto-3g")
    for name in OPEN_SHELL_CLASSES:
        result = solve_scf(hamiltonian, SymmetrySpec.from_name(name))
        assert result.energy == pytest.approx(-0.4666, abs=LITERATURE_TOL)
        assert result.energies.coulomb == pytest.approx(-result.energies.exchange)
        assert result.spin.square == pytest.approx(0.75, abs=SPIN_TOL)
        assert result.spin.multiplicity == pytest.approx(2.0, abs=SPIN_TOL)
        assert abs(result.spin.vector[2]) == pytest.approx(0.5, abs=SPIN_TOL)
        check_state(result, hamiltonian)
        from_atoms = hartree_fock(
            ENGINE, basis, atom, symmetry=SymmetrySpec.from_name(name)
        )
        assert from_atoms.energy == pytest.approx(result.energy, abs=ENERGY_TOL)


@pytest.mark.parametrize("name", CLASSES)
def test_every_class_recovers_the_closed_shell_energy(
    name: str, water_hamiltonian: Hamiltonian, restricted_water: ScfResult
) -> None:
    spec = SymmetrySpec.from_name(name)
    result = solve_scf(water_hamiltonian, spec)
    assert result.structure == spec.structure()
    assert result.energy == pytest.approx(restricted_water.energy, abs=ENERGY_TOL)
    assert result.spin.square == pytest.approx(0.0, abs=SPIN_TOL)
    np.testing.assert_allclose(result.spin.vector, 0.0, atol=SPIN_TOL)
    check_state(result, water_hamiltonian)
    if spec.field.value == "paired" and result.structure.is_spinor:
        assert is_kramers_symmetric(result.density.blocks[0])
    np.testing.assert_allclose(
        result.density.to_dense(),
        restricted_water.density.to_dense(),
        atol=CONVERGED_DENSITY_TOL,
    )


@pytest.mark.parametrize("name", CLASSES[1:])
def test_embedded_solution_is_stationary_in_every_class(
    name: str, water_hamiltonian: Hamiltonian, restricted_water: ScfResult
) -> None:
    structure = SymmetrySpec.from_name(name).structure()
    density = restricted_water.density.embed(structure)
    fock, energies = water_hamiltonian.fock(density)
    assert energies.total == pytest.approx(restricted_water.energy, abs=ENERGY_TOL)
    overlap = water_hamiltonian.metric(structure)
    orthogonalizer = BlockMatrix(
        BlockStructure.spatial(),
        (canonical_orthogonalizer(water_hamiltonian.overlap),),
    ).embed(structure)
    gradient = orbital_gradient(fock, density, overlap, orthogonalizer)
    assert all(
        np.abs(block).max() < ScfOptions().gradient_tolerance
        for block in gradient.blocks
    )
    restarted = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name(name), guess=density
    )
    assert restarted.n_iterations <= 2
    assert restarted.energy == pytest.approx(restricted_water.energy, abs=ENERGY_TOL)


def test_symmetry_breaking_in_stretched_hydrogen() -> None:
    _, hamiltonian = hamiltonian_for(hydrogen(4.0), "sto-3g")
    restricted = solve_scf(hamiltonian, SymmetrySpec.from_name("RHF"))
    # Alpha on one atom, beta on the other.
    broken = BlockMatrix(
        BlockStructure.unrestricted(),
        (np.diag([1.0, 0.0]), np.diag([0.0, 1.0])),
    )
    unrestricted = solve_scf(hamiltonian, SymmetrySpec.from_name("UHF"), guess=broken)
    assert unrestricted.energy < restricted.energy - SYMMETRY_BREAKING_GAIN
    check_state(unrestricted, hamiltonian)
    # A broken-symmetry singlet is a half-and-half mixture: <S^2> near 1.
    assert 0.9 < unrestricted.spin.square < 1.0
    assert unrestricted.spin.vector[2] == pytest.approx(0.0, abs=SPIN_TOL)
    # The unrestricted solution embeds into the generalized class and stays.
    generalized = solve_scf(
        hamiltonian,
        SymmetrySpec.from_name("GHF"),
        guess=unrestricted.density.embed(BlockStructure.generalized()),
    )
    assert generalized.energy == pytest.approx(unrestricted.energy, abs=ENERGY_TOL)
    # From the symmetric core guess the unrestricted class stays restricted.
    symmetric = solve_scf(hamiltonian, SymmetrySpec.from_name("UHF"))
    assert symmetric.energy == pytest.approx(restricted.energy, abs=ENERGY_TOL)


@pytest.mark.parametrize("name", ["complex RHF", "complex UHF", "complex GHF"])
def test_complex_classes_return_from_a_complex_perturbation(
    name: str, water_hamiltonian: Hamiltonian, restricted_water: ScfResult
) -> None:
    structure = SymmetrySpec.from_name(name).structure()
    rng = np.random.default_rng(11)
    density = restricted_water.density.embed(structure)
    perturbed = density.map_blocks(
        lambda block: block + 0.05j * random_antisymmetric(rng, block.shape[0])
    )
    result = solve_scf(water_hamiltonian, SymmetrySpec.from_name(name), guess=perturbed)
    assert result.n_iterations > 2
    assert result.energy == pytest.approx(restricted_water.energy, abs=ENERGY_TOL)
    check_state(result, water_hamiltonian)


def test_open_shell_classes_agree_on_the_water_cation() -> None:
    molecule = water_cation()
    basis, hamiltonian = hamiltonian_for(molecule, "sto-3g")
    unrestricted = solve_scf(hamiltonian, SymmetrySpec.from_name("UHF"))
    check_state(unrestricted, hamiltonian)
    assert unrestricted.mo_occupations.blocks[0].sum() == 5
    assert unrestricted.mo_occupations.blocks[1].sum() == 4
    for name in OPEN_SHELL_CLASSES[1:]:
        result = solve_scf(hamiltonian, SymmetrySpec.from_name(name))
        assert result.energy == pytest.approx(unrestricted.energy, abs=ENERGY_TOL)
    from_atoms = hartree_fock(
        ENGINE, basis, molecule, symmetry=SymmetrySpec.from_name("UHF")
    )
    assert from_atoms.energy == pytest.approx(unrestricted.energy, abs=ENERGY_TOL)
    with pytest.raises(InputError, match="n_alpha == n_beta"):
        solve_scf(hamiltonian, SymmetrySpec.from_name("RHF"))
    with pytest.raises(InputError, match="even"):
        solve_scf(hamiltonian, SymmetrySpec.from_name("paired GHF"))


def test_density_fitting_reproduces_the_exact_energy_closely() -> None:
    molecule = water()
    basis = BasisSet.for_molecule(molecule, "def2-svp", library=LIBRARY)
    auxiliary = BasisSet.for_molecule(molecule, "def2-universal-jkfit", library=LIBRARY)
    exact = hartree_fock(ENGINE, basis, molecule)
    fitted = hartree_fock(ENGINE, basis, molecule, auxiliary=auxiliary)
    assert fitted.energy == pytest.approx(exact.energy, abs=DENSITY_FITTING_ENERGY_TOL)
    assert fitted.energy != pytest.approx(exact.energy, abs=ENERGY_TOL)


def test_diis_shortens_the_iteration() -> None:
    _, hamiltonian = hamiltonian_for(water(), "def2-svp")
    spec = SymmetrySpec()
    accelerated = solve_scf(hamiltonian, spec)
    plain = solve_scf(hamiltonian, spec, options=ScfOptions(diis_size=1))
    assert accelerated.n_iterations < plain.n_iterations
    assert accelerated.energy == pytest.approx(plain.energy, abs=ENERGY_TOL)
    history = accelerated.history
    assert [row.index for row in history] == list(range(len(history)))
    assert history[-1].gradient_norm < ScfOptions().gradient_tolerance
    assert abs(history[-1].energy - accelerated.energy) < ENERGY_TOL


def test_failure_to_converge_carries_the_last_state(
    water_hamiltonian: Hamiltonian,
) -> None:
    with pytest.raises(ConvergenceError, match="did not converge in 2") as excinfo:
        solve_scf(
            water_hamiltonian, SymmetrySpec(), options=ScfOptions(max_iterations=2)
        )
    last = excinfo.value.last
    assert isinstance(last, ScfResult)
    assert not last.converged
    assert last.n_iterations == 2
    # The carried state restarts the iteration where it stopped.
    finished = solve_scf(water_hamiltonian, SymmetrySpec(), guess=last.density)
    assert finished.converged


def test_guesses_and_options_are_validated(water_hamiltonian: Hamiltonian) -> None:
    n = water_hamiltonian.n_ao
    wrong_structure = BlockMatrix(BlockStructure.unrestricted(), (np.eye(n), np.eye(n)))
    with pytest.raises(StructureError, match="embed"):
        solve_scf(water_hamiltonian, SymmetrySpec(), guess=wrong_structure)
    wrong_size = BlockMatrix(BlockStructure.restricted(), (np.eye(n + 1),))
    with pytest.raises(StructureError, match="atomic orbitals"):
        solve_scf(water_hamiltonian, SymmetrySpec(), guess=wrong_size)
    with pytest.raises(InputError, match="iteration"):
        ScfOptions(max_iterations=0)
    with pytest.raises(InputError, match="positive"):
        ScfOptions(energy_tolerance=0.0)
    with pytest.raises(InputError, match="DIIS"):
        ScfOptions(diis_size=0)
    with pytest.raises(InputError, match="level shift"):
        ScfOptions(level_shift=-0.1)
    with pytest.raises(InputError, match="damping"):
        ScfOptions(damping=1.0)
    with pytest.raises(InputError, match="overlap threshold"):
        ScfOptions(overlap_threshold=0.0)
    with pytest.raises(InputError, match="no eigenvalue"):
        canonical_orthogonalizer(water_hamiltonian.overlap, threshold=100.0)
    with pytest.raises(InputError, match="match"):
        Hamiltonian(np.eye(n + 1), np.eye(n), water_hamiltonian.two_body, 0.0, 5, 5)


def test_spin_expectation_matches_the_collinear_formula() -> None:
    molecule = water_cation()
    _, hamiltonian = hamiltonian_for(molecule, "sto-3g")
    result = solve_scf(hamiltonian, SymmetrySpec.from_name("UHF"))
    alpha, beta = result.density.blocks
    overlap = hamiltonian.overlap
    s_z = 0.5
    expected = s_z * (s_z + 1) + 4 - np.trace(alpha @ overlap @ beta @ overlap)
    assert result.spin.square == pytest.approx(expected, abs=SPIN_TOL)
    assert result.spin.vector == pytest.approx((0.0, 0.0, s_z), abs=SPIN_TOL)
    assert result.spin.square > 0.75
    with pytest.raises(InputError, match="different bases"):
        spin_expectation(result.density, overlap[:-1, :-1])


def test_generalized_energy_is_invariant_under_spin_rotation() -> None:
    _, hamiltonian = hamiltonian_for(water_cation(), "sto-3g")
    collinear = solve_scf(hamiltonian, SymmetrySpec.from_name("GHF"))
    n = hamiltonian.n_ao
    theta = 0.7
    # exp(-i theta sigma_x / 2) rotates the spin about x by theta.
    rotation = np.kron(
        np.array(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ]
        ),
        np.eye(n),
    )
    structure = BlockStructure.generalized(Ring.COMPLEX)
    rotated = collinear.density.embed(structure).map_blocks(
        lambda block: rotation @ block @ rotation.conj().T
    )
    _, energies = hamiltonian.fock(rotated)
    assert energies.total == pytest.approx(collinear.energy, abs=ENERGY_TOL)
    spin = spin_expectation(rotated, hamiltonian.overlap)
    assert spin.square == pytest.approx(collinear.spin.square, abs=SPIN_TOL)
    s_z = collinear.spin.vector[2]
    assert spin.vector[2] == pytest.approx(s_z * np.cos(theta), abs=SPIN_TOL)
    assert abs(spin.vector[1]) == pytest.approx(abs(s_z * np.sin(theta)), abs=SPIN_TOL)
    assert spin.vector[0] == pytest.approx(0.0, abs=SPIN_TOL)
    # The rotated determinant is a stationary point of the complex class.
    noncollinear = solve_scf(
        hamiltonian, SymmetrySpec.from_name("complex GHF"), guess=rotated
    )
    assert noncollinear.n_iterations <= 2
    assert noncollinear.energy == pytest.approx(collinear.energy, abs=ENERGY_TOL)
    assert noncollinear.spin.vector[2] == pytest.approx(
        s_z * np.cos(theta), abs=SPIN_TOL
    )


@pytest.mark.parametrize(
    "options",
    [
        ScfOptions(level_shift=0.5),
        ScfOptions(damping=0.3),
        ScfOptions(diis_size=1, level_shift=0.2, damping=0.5),
    ],
)
def test_level_shift_and_damping_reach_the_same_state(
    options: ScfOptions, water_hamiltonian: Hamiltonian, restricted_water: ScfResult
) -> None:
    result = solve_scf(water_hamiltonian, SymmetrySpec(), options=options)
    assert result.energy == pytest.approx(restricted_water.energy, abs=ENERGY_TOL)
    check_state(result, water_hamiltonian)
    # The options change the path, not the destination.
    assert result.history[1].energy != restricted_water.history[1].energy


def test_linearly_dependent_functions_are_removed() -> None:
    molecule = water()
    basis = BasisSet.for_molecule(molecule, "sto-3g", library=LIBRARY)
    duplicated = BasisSet(
        "doubled",
        basis.atom_symbols,
        (basis.shells[0], *basis.shells),
        (0, *basis.shell_atoms),
    )
    _, reference = hamiltonian_for(molecule, "sto-3g")
    expected = solve_scf(reference, SymmetrySpec())
    two_body = ExactCoulombExchange.from_molecule(ENGINE, duplicated, molecule)
    hamiltonian = Hamiltonian.from_molecule(ENGINE, duplicated, molecule, two_body)
    assert np.linalg.eigvalsh(hamiltonian.overlap).min() == pytest.approx(
        0.0, abs=STRUCTURE_TOL
    )
    result = solve_scf(hamiltonian, SymmetrySpec())
    assert result.energy == pytest.approx(expected.energy, abs=ENERGY_TOL)
    assert result.mo_coefficients.shapes == ((8, 7),)
    assert result.mo_energies.blocks[0].shape == (7,)
    np.testing.assert_allclose(
        np.sort(result.mo_energies.blocks[0]),
        np.sort(expected.mo_energies.blocks[0]),
        atol=MO_ENERGY_TOL,
    )
    check_state(result, hamiltonian)
    orthogonalizer = canonical_orthogonalizer(hamiltonian.overlap)
    assert orthogonalizer.shape == (8, 7)
    np.testing.assert_allclose(
        orthogonalizer.T @ hamiltonian.overlap @ orthogonalizer,
        np.eye(7),
        atol=STRUCTURE_TOL,
    )


def test_core_guess_is_spin_averaged(water_hamiltonian: Hamiltonian) -> None:
    structure = BlockStructure.unrestricted()
    guess = core_hamiltonian_guess(water_hamiltonian, structure)
    np.testing.assert_allclose(guess.blocks[0], guess.blocks[1])
    overlap = water_hamiltonian.metric(structure)
    assert float(guess.trace_product(overlap)) == pytest.approx(10.0, abs=STRUCTURE_TOL)
    _, cation = hamiltonian_for(water_cation(), "sto-3g")
    guess = core_hamiltonian_guess(cation, structure)
    np.testing.assert_allclose(guess.blocks[0], guess.blocks[1])
    assert float(guess.trace_product(overlap)) == pytest.approx(9.0, abs=STRUCTURE_TOL)
    # Without a guess the driver starts from exactly this density, not from
    # the integer aufbau filling of the core orbitals.
    _, energies = cation.fock(guess)
    started = solve_scf(cation, SymmetrySpec.from_name("UHF"))
    assert started.history[0].energy == pytest.approx(energies.total, abs=ENERGY_TOL)


def test_atomic_densities_are_spherical_and_superpose_by_atom() -> None:
    carbon = Molecule.from_atoms([("C", (0.0, 0.0, 0.0))])
    basis = BasisSet.for_molecule(carbon, "sto-3g", library=LIBRARY)
    density = atomic_density(ENGINE, basis, 0)
    # STO-3G carbon: 1s, 2s, then 2p x y z. Two 2p electrons spread over
    # three orbitals give 1/3 per spin and no preferred direction.
    np.testing.assert_allclose(density[2:, 2:], np.eye(3) / 3, atol=STRUCTURE_TOL)
    np.testing.assert_allclose(density[:2, 2:], 0.0, atol=STRUCTURE_TOL)
    molecule = water()
    basis = BasisSet.for_molecule(molecule, "sto-3g", library=LIBRARY)
    structure = BlockStructure.generalized()
    guess = atomic_density_guess(ENGINE, basis, structure)
    assert guess.structure == structure
    block = guess.blocks[0]
    n = basis.n_functions
    np.testing.assert_allclose(block[:n, n:], 0.0)
    np.testing.assert_allclose(block[:n, :n], block[n:, n:])
    # Oxygen owns the first five functions, each hydrogen one; no cross terms.
    np.testing.assert_allclose(block[:5, 5:n], 0.0)
    np.testing.assert_allclose(block[5, 6], 0.0)
    _, hamiltonian = hamiltonian_for(molecule, "sto-3g")
    overlap = hamiltonian.metric(structure)
    assert float(guess.trace_product(overlap)) == pytest.approx(10.0, abs=STRUCTURE_TOL)


def test_atomic_guess_follows_the_shells_of_each_atom() -> None:
    # Two hydrogens with different shells: the densities are keyed by the
    # shells, not by the element, and placed in each atom's own functions.
    s_shell = Shell(0, (1.0,), ((1.0,),))
    p_shell = Shell(1, (0.8,), ((1.0,),))
    basis = BasisSet("mixed", ("H", "H"), (s_shell, s_shell, p_shell), (0, 1, 1))
    guess = atomic_density_guess(ENGINE, basis, BlockStructure.restricted())
    block = guess.blocks[0]
    assert block.shape == (5, 5)
    np.testing.assert_allclose(block[0, 1:], 0.0)
    assert block[0, 0] == pytest.approx(0.5, abs=STRUCTURE_TOL)
    assert block[1, 1] == pytest.approx(0.5, abs=STRUCTURE_TOL)
    np.testing.assert_allclose(block[2:, 2:], 0.0, atol=STRUCTURE_TOL)
    with pytest.raises(InputError, match="no basis functions"):
        atomic_density_guess(
            ENGINE,
            BasisSet("bare", ("H", "He"), (s_shell,), (0,)),
            BlockStructure.restricted(),
        )


def test_averaged_occupation_shares_degenerate_orbitals() -> None:
    energies = BlockVector(
        BlockStructure.restricted(), (np.array([-1.0, 0.0, 0.0, 0.0, 2.0]),)
    )
    occupations = AveragedOccupation(1.5, 1.5).occupy(energies).blocks[0]
    np.testing.assert_allclose(occupations, [1.0, 1 / 6, 1 / 6, 1 / 6, 0.0])
    occupations = AveragedOccupation(3.0, 3.0).occupy(energies).blocks[0]
    np.testing.assert_allclose(occupations, [1.0, 2 / 3, 2 / 3, 2 / 3, 0.0])
    spinor = BlockVector(BlockStructure.generalized(), (np.zeros(4),))
    np.testing.assert_allclose(
        AveragedOccupation(1.0, 0.0).occupy(spinor).blocks[0], 0.25
    )
    with pytest.raises(InputError, match="cannot place"):
        AveragedOccupation(3.0, 3.0).occupy(spinor)
    with pytest.raises(InputError, match="positive"):
        AveragedOccupation(1.0, 1.0, degeneracy_tolerance=0.0)


def test_aufbau_fills_the_lowest_orbitals_of_each_block() -> None:
    energies = BlockVector(
        BlockStructure.unrestricted(), (np.arange(4.0), np.arange(4.0))
    )
    occupations = AufbauOccupation(3, 1).occupy(energies)
    np.testing.assert_array_equal(occupations.blocks[0], [1.0, 1.0, 1.0, 0.0])
    np.testing.assert_array_equal(occupations.blocks[1], [1.0, 0.0, 0.0, 0.0])
    spinor = BlockVector(BlockStructure.generalized(), (np.arange(8.0),))
    assert AufbauOccupation(3, 1).occupy(spinor).blocks[0].sum() == 4
    with pytest.raises(InputError, match="n_alpha == n_beta"):
        AufbauOccupation(3, 1).occupy(
            BlockVector(BlockStructure.restricted(), (np.arange(4.0),))
        )
    with pytest.raises(InputError, match="cannot place"):
        AufbauOccupation(5, 5).occupy(energies)
    with pytest.raises(InputError, match="spatial"):
        electrons_per_block(BlockStructure.spatial(), 1, 1)
    with pytest.raises(InputError, match="non-negative"):
        AufbauOccupation(-1, 0)


def test_diis_solves_a_linear_model_exactly() -> None:
    # Errors that depend linearly on the stored matrices: the combination
    # with zero error is found exactly.
    structure = BlockStructure.restricted()
    focks = [
        BlockMatrix(structure, (np.array([[x, 0.0], [0.0, 1.0]]),)) for x in (1.0, 3.0)
    ]
    errors = [
        BlockMatrix(structure, (np.array([[x - 2.0, 0.0], [0.0, 0.0]]),))
        for x in (1.0, 3.0)
    ]
    diis = Diis(2)
    with pytest.raises(InputError, match="empty"):
        diis.extrapolate()
    diis = diis.push(focks[0], errors[0])
    assert diis.extrapolate() is focks[0]
    diis = diis.push(focks[1], errors[1])
    assert len(diis) == 2
    np.testing.assert_allclose(
        diis.extrapolate().blocks[0], [[2.0, 0.0], [0.0, 1.0]], atol=STRUCTURE_TOL
    )
    assert len(diis.push(focks[0], errors[0])) == 2
    with pytest.raises(InputError, match="at least one"):
        Diis(0)
    # Linearly dependent errors: the singular system still yields the
    # combination with zero model error.
    dependent = Diis(3).push(focks[0], errors[0]).push(focks[1], errors[1])
    dependent = dependent.push(focks[1], errors[1])
    np.testing.assert_allclose(
        dependent.extrapolate().blocks[0], [[2.0, 0.0], [0.0, 1.0]], atol=STRUCTURE_TOL
    )
    # Exactly zero errors (a one-function block): every stored matrix is
    # self-consistent and their average is returned, not NaN.
    zero = BlockMatrix(structure, (np.zeros((2, 2)),))
    settled = Diis(2).push(focks[0], zero).push(focks[1], zero)
    np.testing.assert_allclose(
        settled.extrapolate().blocks[0], [[2.0, 0.0], [0.0, 1.0]], atol=STRUCTURE_TOL
    )

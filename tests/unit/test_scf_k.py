"""Periodic Hartree–Fock: classes agree, band folding, occupations across k."""

import numpy as np
import pytest

from hafnium.basis import BasisLibrary, BasisSet
from hafnium.errors import InputError, StructureError
from hafnium.integrals import RustEngine, density_fitting_k, overlap_k
from hafnium.integrals.periodic import RangeSeparation
from hafnium.molecule import AtomicSystem, Cell, Lattice, Molecule, default_omega
from hafnium.scf import (
    DEFAULT_OVERLAP_THRESHOLD,
    AufbauOccupation,
    AveragedOccupation,
    Hamiltonian,
    KPointDensityFittedCoulombExchange,
    ScfResult,
    atomic_density_guess,
    canonical_orthogonalizer,
    hartree_fock,
    solve_scf,
    spin_expectation,
)
from hafnium.structured import BlockMatrix, BlockStructure, BlockVector, Ring
from hafnium.symmetry import KPoints, SymmetrySpec
from tests.cells import box, helium
from tests.tolerances import (
    ENERGY_TOL,
    FINAL_FOCK_RESIDUAL_TOL,
    SPIN_TOL,
    STRUCTURE_TOL,
)

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
AUXILIARY = "def2-universal-jkfit"
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


def basis_for(system: AtomicSystem, name: str) -> BasisSet:
    return BasisSet.for_atoms(system, name, library=LIBRARY)


def hamiltonian_for(
    cell: Cell, basis_name: str, mesh: KPoints, omega: float | None = None
) -> tuple[BasisSet, Hamiltonian]:
    basis = basis_for(cell, basis_name)
    separation = RangeSeparation.for_lattice(cell.lattice, omega=omega)
    two_body = KPointDensityFittedCoulombExchange.from_cell(
        ENGINE, basis, basis_for(cell, AUXILIARY), cell, mesh, separation=separation
    )
    return basis, Hamiltonian.from_cell(ENGINE, basis, cell, mesh, two_body)


def hydrogen_box() -> Cell:
    """One hydrogen atom per cubic cell of 8 Bohr, a doublet per cell.

    At 4 Bohr the 1s band is 0.7 Hartree wide and the aufbau occupation
    across the k-points changes between iterations; at 8 Bohr the band is
    narrow and the integer occupations settle.
    """
    molecule = Molecule.from_atoms([("H", (0.3, 0.1, 0.2))], unit="bohr", spin=1)
    return box(molecule, 8.0)


def check_state(result: ScfResult, hamiltonian: Hamiltonian) -> None:
    """Hermitian, idempotent density with the electron count per cell."""
    overlap = hamiltonian.metric(result.structure)
    density = result.density
    assert result.converged
    for block in density.blocks:
        np.testing.assert_allclose(
            block, np.conj(np.swapaxes(block, -1, -2)), atol=STRUCTURE_TOL
        )
    assert float(np.real(density.trace_product(overlap))) == pytest.approx(
        hamiltonian.n_electrons, abs=STRUCTURE_TOL
    )
    projected = density @ overlap @ density
    for block, again in zip(density.blocks, projected.blocks, strict=True):
        np.testing.assert_allclose(again, block, atol=STRUCTURE_TOL)
    # The orbital energies are the eigenvalues of the Fock matrix in the
    # metric at every stored k-point, F C = S C ε, up to the rebuild of the
    # Fock matrix from the orbitals' density.
    left = result.fock @ result.mo_coefficients
    right = overlap @ result.mo_coefficients
    for lhs, rhs, energies in zip(
        left.blocks, right.blocks, result.mo_energies.blocks, strict=True
    ):
        np.testing.assert_allclose(
            lhs, rhs * energies[..., None, :], atol=FINAL_FOCK_RESIDUAL_TOL
        )


@pytest.fixture(scope="module")
def helium_pair() -> tuple[BasisSet, Hamiltonian, KPoints]:
    mesh = KPoints.mesh((3, 1, 1))
    basis, hamiltonian = hamiltonian_for(helium(3), "cc-pvdz", mesh)
    return basis, hamiltonian, mesh


def test_every_class_recovers_the_closed_shell_energy_over_k(
    helium_pair: tuple[BasisSet, Hamiltonian, KPoints],
) -> None:
    basis, hamiltonian, mesh = helium_pair
    reference = solve_scf(hamiltonian, SymmetrySpec.from_name("RHF").with_kpoints(mesh))
    check_state(reference, hamiltonian)
    assert reference.spin.square == pytest.approx(0.0, abs=SPIN_TOL)
    # The points at 1/3 and 2/3 are partners, so the real classes store the
    # half zone (Γ and 1/3) as complex blocks, the complex classes every k.
    assert reference.density.blocks[0].dtype == np.complex128
    assert reference.density.blocks[0].shape[0] == 2
    for name in CLASSES:
        spec = SymmetrySpec.from_name(name).with_kpoints(mesh)
        result = solve_scf(hamiltonian, spec)
        assert result.energy == pytest.approx(reference.energy, abs=ENERGY_TOL)
        check_state(result, hamiltonian)
        from_atoms = hartree_fock(
            ENGINE,
            basis,
            helium(3),
            symmetry=spec,
            auxiliary=basis_for(helium(3), AUXILIARY),
        )
        assert from_atoms.energy == pytest.approx(reference.energy, abs=ENERGY_TOL)


def test_supercell_at_gamma_folds_the_primitive_energy(
    helium_pair: tuple[BasisSet, Hamiltonian, KPoints],
) -> None:
    _, hamiltonian, mesh = helium_pair
    primitive = solve_scf(hamiltonian, SymmetrySpec.from_name("RHF").with_kpoints(mesh))
    supercell = helium(3).supercell((3, 1, 1))
    gamma = KPoints.gamma()
    _, folded_hamiltonian = hamiltonian_for(supercell, "cc-pvdz", gamma)
    folded = solve_scf(
        folded_hamiltonian, SymmetrySpec.from_name("RHF").with_kpoints(gamma)
    )
    # The Γ-point blocks of the supercell are real.
    assert folded.density.blocks[0].dtype == np.float64
    assert folded.energy / 3 == pytest.approx(primitive.energy, abs=ENERGY_TOL)
    assert folded.energies.nuclear_repulsion / 3 == pytest.approx(
        primitive.energies.nuclear_repulsion, abs=ENERGY_TOL
    )
    assert folded.energies.exchange / 3 == pytest.approx(
        primitive.energies.exchange, abs=ENERGY_TOL
    )


def test_energy_is_independent_of_the_range_separation(
    helium_pair: tuple[BasisSet, Hamiltonian, KPoints],
) -> None:
    _, hamiltonian, mesh = helium_pair
    spec = SymmetrySpec.from_name("RHF").with_kpoints(mesh)
    reference = solve_scf(hamiltonian, spec)
    cell = helium(3)
    _, other = hamiltonian_for(
        cell, "cc-pvdz", mesh, omega=2.0 * default_omega(cell.lattice)
    )
    assert solve_scf(other, spec).energy == pytest.approx(
        reference.energy, abs=ENERGY_TOL
    )


def check_orthogonalizer(hamiltonian: Hamiltonian, structure: BlockStructure) -> None:
    """``X† S X`` is the identity at every stored k-point."""
    orthogonalizer = hamiltonian.orthogonalizer(structure)
    overlap = hamiltonian.metric(structure)
    identity = orthogonalizer.adjoint() @ overlap @ orthogonalizer
    for block in identity.blocks:
        eye = np.broadcast_to(np.eye(block.shape[-1]), block.shape)
        np.testing.assert_allclose(block, eye, atol=STRUCTURE_TOL)


def test_orthogonalizer_at_the_half_zone_is_complex(
    helium_pair: tuple[BasisSet, Hamiltonian, KPoints],
) -> None:
    _, hamiltonian, mesh = helium_pair
    structure = SymmetrySpec.from_name("RHF").with_kpoints(mesh).structure()
    assert hamiltonian.orthogonalizer(structure).blocks[0].dtype == np.complex128
    check_orthogonalizer(hamiltonian, structure)


def test_hydrogen_in_a_box_in_every_open_shell_class() -> None:
    cell = hydrogen_box()
    mesh = KPoints.mesh((2, 2, 2))
    basis, hamiltonian = hamiltonian_for(cell, "cc-pvdz", mesh)
    unrestricted = SymmetrySpec.from_name("UHF").with_kpoints(mesh)
    # Every point of the mesh is time-reversal invariant: the real classes
    # store each point as a real block, and so is the orthogonalizer, built
    # from the real overlap (a complex eigendecomposition would carry a
    # phase per eigenvector that its real part cannot drop).
    assert hamiltonian.orthogonalizer(unrestricted.structure()).blocks[0].dtype == (
        np.float64
    )
    check_orthogonalizer(hamiltonian, unrestricted.structure())
    reference = solve_scf(
        hamiltonian,
        unrestricted,
        guess=atomic_density_guess(ENGINE, basis, unrestricted.structure()),
    )
    check_state(reference, hamiltonian)
    assert reference.spin.square == pytest.approx(0.75, abs=SPIN_TOL)
    assert abs(reference.spin.vector[2]) == pytest.approx(0.5, abs=SPIN_TOL)
    for name in ("complex UHF", "GHF", "complex GHF"):
        spec = SymmetrySpec.from_name(name).with_kpoints(mesh)
        # From a spin-averaged guess the generalized classes converge to a
        # higher non-magnetic stationary state; the unrestricted density
        # embedded in the class leads to the doublet.
        result = solve_scf(
            hamiltonian, spec, guess=reference.density.embed(spec.structure())
        )
        check_state(result, hamiltonian)
        assert result.energy == pytest.approx(reference.energy, abs=ENERGY_TOL)
        assert result.spin.square == pytest.approx(0.75, abs=SPIN_TOL)
        assert abs(result.spin.vector[2]) == pytest.approx(0.5, abs=SPIN_TOL)
    with pytest.raises(InputError, match="n_alpha == n_beta"):
        solve_scf(hamiltonian, SymmetrySpec.from_name("RHF").with_kpoints(mesh))


def test_occupations_fill_levels_across_k_by_weight() -> None:
    mesh = KPoints.mesh((4, 1, 1))
    structure = SymmetrySpec.from_name("UHF").with_kpoints(mesh).structure()
    # Alpha stores the half zone, the points 0, 1/4 and 1/2, the second
    # carrying the weight of its partner at 3/4 too (1/4, 1/2, 1/4); beta
    # likewise; three levels per point.
    alpha = np.array([[0.0, 1.0, 1.1], [0.2, 1.5, 2.5], [0.4, 1.6, 3.0]])
    energies = BlockVector(structure, (alpha, alpha + 0.05))
    stored = structure.stored_weights(structure.stored_blocks[0])
    weights = np.asarray(stored[0]) + np.asarray(stored[1])
    assert weights.tolist() == [0.25, 0.5, 0.25]
    # Two alpha electrons per cell: the lowest level at every point (one
    # electron), then the levels at 1.0 and 1.1 (weight 1/4 each) and 1.5
    # (weight 1/2); one beta electron fills the lowest level.
    filled = AufbauOccupation(2, 1).occupy(energies)
    np.testing.assert_allclose(
        filled.blocks[0], [[1, 1, 1], [1, 1, 0], [1, 0, 0]], atol=STRUCTURE_TOL
    )
    np.testing.assert_allclose(filled.blocks[1], [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    assert float(np.sum(weights[:, None] * filled.blocks[0])) == pytest.approx(2.0)
    # With the level at 1.6 moved below 1.5 the count is reached inside the
    # level of weight 1/2.
    crossing = np.array([[0.0, 1.0, 1.1], [0.2, 1.5, 2.5], [0.4, 1.2, 3.0]])
    with pytest.raises(InputError, match="whole levels"):
        AufbauOccupation(2, 1).occupy(BlockVector(structure, (crossing, crossing)))
    with pytest.raises(InputError, match="cannot place"):
        AufbauOccupation(4, 1).occupy(energies)
    # The averaged policy shares the frontier among degenerate levels.
    degenerate = np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    shared = AveragedOccupation(2.0, 2.0).occupy(
        BlockVector(structure, (degenerate, degenerate))
    )
    np.testing.assert_allclose(shared.blocks[0][:, 0], 1.0)
    np.testing.assert_allclose(shared.blocks[0][:, 1:], 0.5)


def test_spatial_matrices_enter_structures_through_from_spatial() -> None:
    mesh = KPoints.mesh((3, 1, 1))
    axis = mesh.axis()
    assert axis.partners == (0, 2, 1)
    rng = np.random.default_rng(3)
    at_gamma = rng.standard_normal((3, 3))
    at_third = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    # A time-reversal symmetric matrix, X(−k) = conj(X(k)): real at Γ.
    symmetric = np.stack([at_gamma, at_third, np.conj(at_third)])
    matrix = BlockMatrix.from_spatial(symmetric, axis)
    assert matrix.structure == BlockStructure.spatial().with_k_axis(axis)
    assert matrix.blocks[0].shape == (2, 3, 3)
    np.testing.assert_allclose(matrix.materialized()[0], symmetric, atol=STRUCTURE_TOL)
    with pytest.raises(StructureError, match="time-reversal symmetric"):
        BlockMatrix.from_spatial(symmetric + 1j, axis)
    every = BlockMatrix.from_spatial(symmetric + 1j, axis, ring=Ring.COMPLEX)
    assert every.blocks[0].shape == (mesh.n_kpt, 3, 3)
    with pytest.raises(StructureError, match=r"\[kpt, n, m\]"):
        BlockMatrix.from_spatial(symmetric[:2], axis)
    # At Γ alone the real ring stores a real block.
    gamma = BlockMatrix.from_spatial(symmetric[:1], KPoints.gamma().axis())
    assert gamma.blocks[0].dtype == np.float64
    with pytest.raises(StructureError, match="two-dimensional"):
        BlockMatrix.from_spatial(symmetric)


def test_orthogonalizer_keeps_one_shape_across_k() -> None:
    rng = np.random.default_rng(1)
    base = rng.standard_normal((4, 4))
    overlap = base @ base.T + 4.0 * np.eye(4)
    # At the second k-point one direction is nearly dependent.
    dependent = overlap.copy()
    dependent[3, :] = dependent[2, :] + 1e-13
    dependent[:, 3] = dependent[:, 2] + 1e-13
    batch = np.stack([overlap, dependent])
    x = canonical_orthogonalizer(batch, DEFAULT_OVERLAP_THRESHOLD)
    assert x.shape == (2, 4, 3)
    for k in range(2):
        np.testing.assert_allclose(
            np.conj(x[k]).T @ batch[k] @ x[k], np.eye(3), atol=STRUCTURE_TOL
        )
    single = canonical_orthogonalizer(overlap, DEFAULT_OVERLAP_THRESHOLD)
    assert single.shape == (4, 4)
    np.testing.assert_allclose(
        single.T @ overlap @ single, np.eye(4), atol=STRUCTURE_TOL
    )


def test_k_point_inputs_are_validated(
    helium_pair: tuple[BasisSet, Hamiltonian, KPoints],
) -> None:
    basis, hamiltonian, mesh = helium_pair
    with pytest.raises(InputError, match="different k-point sets"):
        solve_scf(hamiltonian, SymmetrySpec.from_name("RHF"))
    with pytest.raises(InputError, match="do not match"):
        hamiltonian.metric(SymmetrySpec.from_name("RHF").structure())
    with pytest.raises(InputError, match="k-point set"):
        hartree_fock(
            ENGINE, basis, helium(3), auxiliary=basis_for(helium(3), AUXILIARY)
        )
    with pytest.raises(InputError, match="auxiliary basis"):
        hartree_fock(
            ENGINE, basis, helium(3), symmetry=SymmetrySpec().with_kpoints(mesh)
        )
    molecule = Molecule.from_atoms([("He", (0.0, 0.0, 0.0))])
    with pytest.raises(InputError, match="no k-points"):
        hartree_fock(
            ENGINE,
            basis_for(molecule, "sto-3g"),
            molecule,
            symmetry=SymmetrySpec().with_kpoints(mesh),
        )
    with pytest.raises(InputError, match=r"\[3, 10, 10\]"):
        Hamiltonian(
            hamiltonian.overlap[0],
            hamiltonian.core_hamiltonian,
            hamiltonian.two_body,
            0.0,
            1,
            1,
            kpoints=mesh,
        )
    with pytest.raises(InputError, match="different bases"):
        spin_expectation(
            solve_scf(hamiltonian, SymmetrySpec().with_kpoints(mesh)).density,
            hamiltonian.overlap[:1],
        )
    lattice = Lattice(((3.0, 0.0, 0.0),))
    assert lattice.dimension == 1


LOW_DIMENSION_MESHES = {2: KPoints.mesh((2, 2)), 1: KPoints.mesh((3,))}
LOW_DIMENSION_MULTIPLES = {2: (2, 1), 1: (3,)}
LOW_DIMENSION_FOLDED = {2: KPoints.mesh((1, 2)), 1: KPoints.gamma(1)}


@pytest.mark.parametrize("dimension", [2, 1])
def test_slab_and_chain_energies_are_range_separation_independent_and_fold(
    dimension: int,
) -> None:
    cell = helium(dimension)
    mesh = LOW_DIMENSION_MESHES[dimension]
    _, hamiltonian = hamiltonian_for(cell, "cc-pvdz", mesh)
    spec = SymmetrySpec.from_name("RHF").with_kpoints(mesh)
    reference = solve_scf(hamiltonian, spec)
    check_state(reference, hamiltonian)
    _, other = hamiltonian_for(
        cell, "cc-pvdz", mesh, omega=0.5 * default_omega(cell.lattice)
    )
    assert solve_scf(other, spec).energy == pytest.approx(
        reference.energy, abs=ENERGY_TOL
    )
    multiples = LOW_DIMENSION_MULTIPLES[dimension]
    folded_mesh = LOW_DIMENSION_FOLDED[dimension]
    _, folded_hamiltonian = hamiltonian_for(
        cell.supercell(multiples), "cc-pvdz", folded_mesh
    )
    folded = solve_scf(
        folded_hamiltonian, SymmetrySpec.from_name("RHF").with_kpoints(folded_mesh)
    )
    assert folded.energy / np.prod(multiples) == pytest.approx(
        reference.energy, abs=ENERGY_TOL
    )


def test_signed_contractions_match_the_fitted_integrals_of_a_chain() -> None:
    # At Γ of a chain the signature carries one −1; the builder's J and K
    # must still be the contractions of the explicit fitted integrals.
    cell = helium(1)
    gamma = KPoints.gamma(1)
    basis = basis_for(cell, "cc-pvdz")
    fit = density_fitting_k(ENGINE, basis, basis_for(cell, AUXILIARY), cell, gamma)
    assert int(np.count_nonzero(np.asarray(fit.signature()) < 0.0)) == 1
    two_body = KPointDensityFittedCoulombExchange.from_fitting(
        fit, overlap_k(ENGINE, basis, cell, gamma), 0.0
    )
    generator = np.random.default_rng(3)
    n = basis.n_functions
    random = generator.normal(size=(n, n)) + 1j * generator.normal(size=(n, n))
    density = (random + random.conj().T)[None, :, :]
    eri = fit.eri((0, 0, 0, 0))
    coulomb = np.einsum("uvwx,kwx->kuv", eri, density)
    exchange = np.einsum("uwxv,kwx->kuv", eri, density)
    np.testing.assert_allclose(two_body.coulomb(density), coulomb, atol=STRUCTURE_TOL)
    np.testing.assert_allclose(two_body.exchange(density), exchange, atol=STRUCTURE_TOL)


def test_fitted_builder_requires_a_matching_signature(
    helium_pair: tuple[BasisSet, Hamiltonian, KPoints],
) -> None:
    _, hamiltonian, _ = helium_pair
    two_body = hamiltonian.two_body
    assert isinstance(two_body, KPointDensityFittedCoulombExchange)
    with pytest.raises(InputError, match=r"\[kpt, kpt, aux\]"):
        KPointDensityFittedCoulombExchange(
            two_body.df_tensor,
            two_body.signature[0],
            two_body.weights,
            two_body.overlap,
            two_body.madelung,
        )

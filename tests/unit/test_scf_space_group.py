"""Hartree–Fock over the irreducible wedge of a space group (ADR-0024)."""

import numpy as np
import pytest

from hafnium.basis import BasisLibrary, BasisSet
from hafnium.errors import InputError
from hafnium.integrals import RustEngine
from hafnium.molecule import Cell, Molecule
from hafnium.scf import (
    Hamiltonian,
    KPointDensityFittedCoulombExchange,
    ScfResult,
    atomic_density_guess,
    hartree_fock,
    orbital_gradient,
    solve_scf,
)
from hafnium.symmetry import (
    KPoints,
    SpaceGroup,
    SpaceGroupAdaptation,
    SymmetrySpec,
    detect_space_group,
)
from tests.cells import diamond_helium, helium_chain, hexagonal_helium
from tests.tolerances import (
    CONVERGED_DENSITY_TOL,
    ENERGY_TOL,
    FINAL_FOCK_RESIDUAL_TOL,
    SPIN_TOL,
    STRUCTURE_TOL,
)
from tests.unit.test_scf_k import CLASSES, check_state

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
AUXILIARY = "def2-universal-jkfit"
Setup = tuple[Cell, BasisSet, KPoints, SpaceGroup, Hamiltonian, Hamiltonian]


def setup(cell: Cell, mesh: KPoints) -> Setup:
    """The cell's space group and its Hamiltonian with and without it."""
    basis = BasisSet.for_atoms(cell, "cc-pvdz", library=LIBRARY)
    auxiliary = BasisSet.for_atoms(cell, AUXILIARY, library=LIBRARY)
    group = detect_space_group(cell)
    two_body = KPointDensityFittedCoulombExchange.from_cell(
        ENGINE, basis, auxiliary, cell, mesh
    )
    plain = Hamiltonian.from_cell(ENGINE, basis, cell, mesh, two_body)
    reduced = Hamiltonian.from_cell(
        ENGINE, basis, cell, mesh, two_body, space_group=group
    )
    return cell, basis, mesh, group, plain, reduced


@pytest.fixture(scope="module")
def diamond() -> Setup:
    return setup(diamond_helium(), KPoints.mesh((2, 2, 2)))


@pytest.fixture(scope="module")
def slab() -> Setup:
    return setup(hexagonal_helium(), KPoints.mesh((3, 3)))


def solve(hamiltonian: Hamiltonian, basis: BasisSet, spec: SymmetrySpec) -> ScfResult:
    structure = hamiltonian.structure(spec)
    return solve_scf(
        hamiltonian, spec, guess=atomic_density_guess(ENGINE, basis, structure)
    )


def check_wedge_against_full_zone(
    fixture: Setup, name: str, stored: tuple[int, int]
) -> None:
    """The reduced state equals the full-zone state of the same class."""
    _, basis, mesh, group, plain, reduced = fixture
    spec = SymmetrySpec.from_name(name).with_kpoints(mesh)
    full = solve(plain, basis, spec)
    wedge = solve(reduced, basis, spec.with_space_group(group))
    assert wedge.structure.k_axis is not None
    assert wedge.structure.k_axis.operations == reduced.adaptation.operations()  # type: ignore[union-attr]
    half, every = stored
    assert wedge.density.blocks[0].shape[0] == (
        half if wedge.structure.half_zone(wedge.structure.stored_blocks[0]) else every
    )
    assert full.density.blocks[0].shape[0] > wedge.density.blocks[0].shape[0]
    check_state(wedge, reduced)
    assert wedge.energy == pytest.approx(full.energy, abs=ENERGY_TOL)
    assert wedge.spin.square == pytest.approx(full.spin.square, abs=SPIN_TOL)
    np.testing.assert_allclose(
        wedge.density.to_dense(), full.density.to_dense(), atol=CONVERGED_DENSITY_TOL
    )
    # The wedge density embeds losslessly into the full-zone structure, where
    # it is stationary, and the full-zone density projects onto the wedge.
    embedded = wedge.density.embed(full.structure)
    np.testing.assert_allclose(
        embedded.to_dense(), wedge.density.to_dense(), atol=STRUCTURE_TOL
    )
    fock, _ = plain.fock(embedded)
    gradient = orbital_gradient(
        fock,
        embedded,
        plain.metric(full.structure),
        plain.orthogonalizer(full.structure),
    )
    norm = np.sqrt(float(np.real(gradient.adjoint().trace_product(gradient))))
    assert norm < FINAL_FOCK_RESIDUAL_TOL
    projected = full.density.project(wedge.structure)
    for block, other in zip(projected.blocks, wedge.density.blocks, strict=True):
        np.testing.assert_allclose(block, other, atol=CONVERGED_DENSITY_TOL)


@pytest.mark.parametrize("name", CLASSES)
def test_every_class_over_the_wedge_of_a_non_symmorphic_crystal(
    diamond: Setup, name: str
) -> None:
    # Fd-3m with the inversion: three stars of the 2×2×2 mesh, all of its
    # points time-reversal invariant.
    assert diamond[3].name == "Fd-3m (227)"
    assert diamond[3].has_fractional_translations
    check_wedge_against_full_zone(diamond, name, (3, 3))


@pytest.mark.parametrize("name", CLASSES)
def test_every_class_over_the_wedge_of_a_hexagonal_slab(slab: Setup, name: str) -> None:
    # Three stars of the 3×3 mesh: Γ, the six points equivalent to (1/3, 0)
    # and the pair K, K'; the half zone alone would keep five points.
    assert slab[3].name == "P6/mmm (191)"
    check_wedge_against_full_zone(slab, name, (3, 3))


def test_a_chain_reduces_by_its_inversion() -> None:
    fixture = setup(helium_chain(), KPoints.mesh((4,)))
    _, _, mesh, group, _, reduced = fixture
    assert group.order == 16
    structure = reduced.structure(
        SymmetrySpec.from_name("complex RHF").with_kpoints(mesh).with_space_group(group)
    )
    assert structure.k_axis is not None
    # The inversion maps 1/4 onto 3/4 without conjugation, so the complex
    # class stores three points like the real one.
    assert structure.k_axis.stored(half_zone=False) == (0, 1, 2)
    assert structure.k_axis.derivation(3, half_zone=False).operation > 0
    check_wedge_against_full_zone(fixture, "complex RHF", (3, 3))


def test_the_entry_point_takes_the_space_group(slab: Setup) -> None:
    cell, basis, mesh, group, _, reduced = slab
    spec = SymmetrySpec.from_name("UHF").with_kpoints(mesh).with_space_group(group)
    result = hartree_fock(
        ENGINE,
        basis,
        cell,
        symmetry=spec,
        auxiliary=BasisSet.for_atoms(cell, AUXILIARY, library=LIBRARY),
    )
    assert result.structure == reduced.structure(spec)
    assert result.structure.describe() == (
        "unrestricted (real) at 9 k-points in P6/mmm (191)"
    )
    assert result.energy == pytest.approx(
        solve(reduced, basis, spec).energy, abs=ENERGY_TOL
    )
    molecule = Molecule.from_atoms([("He", (0.0, 0.0, 0.0))])
    with pytest.raises(InputError, match="no k-points or space group"):
        hartree_fock(
            ENGINE,
            BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY),
            molecule,
            symmetry=SymmetrySpec.from_name("RHF").with_space_group(group),
        )


def test_structures_and_hamiltonians_must_agree(diamond: Setup) -> None:
    cell, basis, mesh, group, plain, reduced = diamond
    spec = SymmetrySpec.from_name("RHF").with_kpoints(mesh)
    wedge = reduced.structure(spec.with_space_group(group))
    with pytest.raises(InputError, match="needs the space-group adaptation"):
        plain.structure(spec.with_space_group(group))
    with pytest.raises(InputError, match="point operations do not match"):
        plain.metric(wedge)
    # The plain structure is served by both Hamiltonians.
    assert reduced.structure(spec) == plain.structure(spec)
    assert reduced.metric(plain.structure(spec)).structure == plain.structure(spec)
    other = KPoints.mesh((3, 3, 3))
    adaptation = SpaceGroupAdaptation.from_basis(basis, cell, group, other)
    with pytest.raises(InputError, match="other k-points"):
        Hamiltonian(
            reduced.overlap,
            reduced.core_hamiltonian,
            reduced.two_body,
            reduced.nuclear_repulsion,
            reduced.n_alpha,
            reduced.n_beta,
            kpoints=mesh,
            adaptation=adaptation,
        )
    with pytest.raises(InputError, match="not invariant under operation"):
        Hamiltonian.from_cell(
            ENGINE,
            basis,
            cell,
            KPoints.mesh((2, 2, 1)),
            reduced.two_body,
            space_group=group,
        )

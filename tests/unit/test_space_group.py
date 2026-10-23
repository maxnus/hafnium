"""Space groups of cells, their action on the Bloch basis and the k-point wedge."""

from collections.abc import Callable

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.errors import InputError, SymmetryError
from hafnium.integrals import core_hamiltonian_k, overlap_k
from hafnium.molecule import Cell, Molecule
from hafnium.structured import KPointOperations
from hafnium.symmetry import (
    KPoints,
    SpaceGroup,
    SpaceGroupAdaptation,
    SymmetrySpec,
    detect_space_group,
)
from tests import cells
from tests.cells import diamond_helium, helium_chain, hexagonal_helium
from tests.hamiltonians import ENGINE
from tests.shell_tables import LIBRARY
from tests.tolerances import INTEGRAL_TOL, STRUCTURE_TOL


# (cell, name, order, whether an operation carries a fractional translation
# about the cell's origin, inversion): the inversion centre of the helium
# pair lies between the atoms and that of the lone atom on it, off the
# origin, so their inversions carry translations.
@pytest.mark.parametrize(
    ("build", "name", "order", "translations", "inversion"),
    [
        (cells.lithium_hydride, "Pm-3m (221)", 48, False, True),
        (diamond_helium, "Fd-3m (227)", 48, True, True),
        (hexagonal_helium, "P6/mmm (191)", 24, False, True),
        (helium_chain, "P4/mmm (123)", 16, False, True),
        (lambda: cells.helium(3), "P-1 (2)", 2, True, True),
        (cells.helium_atom, "P-1 (2)", 2, True, True),
    ],
)
def test_space_groups_are_detected(
    build: Callable[[], Cell],
    name: str,
    order: int,
    translations: bool,
    inversion: bool,
) -> None:
    cell = build()
    group = detect_space_group(cell)
    assert group.name == name
    assert group.order == order
    assert group.has_fractional_translations == translations
    assert group.has_inversion == inversion
    assert group.dimension == cell.dimension
    assert group.operations[0].is_identity
    for operation in group.operations:
        matrix = np.asarray(operation.matrix)
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=STRUCTURE_TOL)
        assert abs(np.linalg.det(matrix)) == pytest.approx(1.0, abs=STRUCTURE_TOL)
        assert operation.is_proper == (np.linalg.det(matrix) > 0)


def test_non_symmorphic_operations_carry_translations() -> None:
    group = detect_space_group(diamond_helium())
    translated = [
        operation for operation in group.operations if operation.has_translation
    ]
    assert len(translated) == 24
    assert all(not operation.is_identity for operation in translated)
    # The operations with no translation form the site group of an atom.
    site = SpaceGroup(
        group.international,
        group.number,
        group.point_group,
        tuple(op for op in group.operations if not op.has_translation),
        group.dimension,
        group.tolerance,
    )
    assert site.order == 24
    assert not site.has_fractional_translations


def test_lower_dimensional_cells_keep_their_open_directions() -> None:
    for cell in (hexagonal_helium(), helium_chain()):
        group = detect_space_group(cell)
        d = cell.dimension
        for operation in group.operations:
            w = np.asarray(operation.rotation)
            assert not np.any(w[:d, d:])
            assert not np.any(w[d:, :d])
            assert all(t == 0.0 for t in operation.translation[d:])


def test_k_point_images_and_the_restricted_subgroup() -> None:
    group = detect_space_group(cells.lithium_hydride())
    mesh = KPoints.mesh((2, 2, 2))
    images = group.kpoint_images(mesh)
    assert len(images) == group.order
    assert images[0] == tuple(range(8))
    for row in images:
        assert sorted(row) == list(range(8))
    # Γ, X, M and R: four stars.
    axis = mesh.axis(
        SpaceGroupAdaptation.from_basis(
            BasisSet.for_atoms(cells.lithium_hydride(), "sto-3g", library=LIBRARY),
            cells.lithium_hydride(),
            group,
            mesh,
        ).operations()
    )
    assert axis.stored(half_zone=True) == (0, 1, 3, 7)
    assert axis.stored(half_zone=False) == (0, 1, 3, 7)
    own, partner = axis.stored_weights(half_zone=True)
    assert own == pytest.approx((1 / 8, 3 / 8, 3 / 8, 1 / 8))
    assert partner == (0.0,) * 4
    # A mesh without the cubic symmetry keeps the tetragonal subgroup.
    flat = KPoints.mesh((2, 2, 1))
    with pytest.raises(InputError, match="not invariant under operation"):
        group.kpoint_images(flat)
    tetragonal = group.restricted_to(flat)
    assert tetragonal.order == 16
    assert tetragonal.name == group.name
    assert len(tetragonal.kpoint_images(flat)) == 16
    with pytest.raises(InputError, match="do not fit"):
        group.kpoint_images(KPoints.mesh((2, 2)))
    with pytest.raises(InputError, match="positive"):
        detect_space_group(cells.lithium_hydride(), tolerance=0.0)


@pytest.mark.parametrize(
    ("build", "basis_name", "mesh"),
    [
        # The fitting basis carries d functions, which the orbital bases do not.
        (diamond_helium, "def2-universal-jkfit", KPoints.mesh((2, 2, 2))),
        (diamond_helium, "cc-pvdz", KPoints.mesh((3, 3, 3))),
        (hexagonal_helium, "cc-pvdz", KPoints.mesh((3, 3))),
        (helium_chain, "cc-pvdz", KPoints.mesh((4,))),
        (cells.lithium_hydride, "sto-3g", KPoints.mesh((2, 2, 2))),
    ],
)
def test_bloch_matrices_at_rotated_k_points_are_the_rotated_matrices(
    build: Callable[[], Cell], basis_name: str, mesh: KPoints
) -> None:
    cell = build()
    group = detect_space_group(cell)
    basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
    adaptation = SpaceGroupAdaptation.from_basis(basis, cell, group, mesh)
    operations = adaptation.operations()
    assert isinstance(operations, KPointOperations)
    assert operations.n_operations == group.order
    assert operations.n_functions == basis.n_functions
    assert adaptation.operations() is operations
    assert operations == adaptation.operations()
    s = np.asarray(overlap_k(ENGINE, basis, cell, mesh))
    h = np.asarray(core_hamiltonian_k(ENGINE, basis, cell, mesh))
    for g in range(operations.n_operations):
        for k in range(mesh.n_kpt):
            u = np.asarray(operations.unitary(g, k))
            image = operations.images[g][k]
            for matrix in (s, h):
                np.testing.assert_allclose(
                    u @ matrix[k] @ u.conj().T, matrix[image], atol=INTEGRAL_TOL
                )


def test_adaptations_are_validated() -> None:
    cell = cells.lithium_hydride()
    group = detect_space_group(cell)
    mesh = KPoints.mesh((2, 2, 2))
    basis = BasisSet.for_atoms(cell, "sto-3g", library=LIBRARY)
    with pytest.raises(InputError, match="atom counts"):
        SpaceGroupAdaptation.from_basis(
            BasisSet.for_atoms(cells.helium_atom(), "sto-3g", library=LIBRARY),
            cell,
            group,
            mesh,
        )
    with pytest.raises(InputError, match="do not fit"):
        SpaceGroupAdaptation.from_basis(basis, cell, group, KPoints.mesh((2, 2)))
    with pytest.raises(InputError, match="not invariant"):
        SpaceGroupAdaptation.from_basis(basis, cell, group, KPoints.mesh((2, 2, 1)))
    # Atoms the group relates must carry equal shells: a cell of two helium
    # atoms with different bases is not symmetric under the swap.
    pair = Cell.from_molecule(
        Molecule.from_atoms(
            [("He", (0.0, 0.0, 0.0)), ("He", (2.0, 2.0, 2.0))], unit="bohr"
        ),
        cells.CUBIC,
    )
    pair_group = detect_space_group(pair)
    # Two equal atoms at the corner and the centre of the cube form a
    # body-centred lattice: the centring translation doubles the operations.
    assert pair_group.name == "Im-3m (229)"
    assert pair_group.order == 96
    mixed = BasisSet.for_atoms(pair, "sto-3g", library=LIBRARY)
    larger = BasisSet.for_atoms(pair, "cc-pvdz", library=LIBRARY)
    shells = mixed.shells[:1] + larger.shells[larger.n_shells // 2 :]
    atoms = mixed.shell_atoms[:1] + larger.shell_atoms[larger.n_shells // 2 :]
    unequal = BasisSet("mixed", mixed.atom_symbols, shells, atoms)
    with pytest.raises(SymmetryError, match="different basis functions"):
        SpaceGroupAdaptation.from_basis(unequal, pair, pair_group, mesh)
    # A group of another cell does not map the atoms.
    with pytest.raises(SymmetryError, match="does not map atom"):
        SpaceGroupAdaptation.from_basis(
            basis, cell, detect_space_group(diamond_helium()), mesh
        )


def test_specifications_name_the_space_group() -> None:
    cell = cells.lithium_hydride()
    group = detect_space_group(cell)
    mesh = KPoints.mesh((2, 2, 2))
    basis = BasisSet.for_atoms(cell, "sto-3g", library=LIBRARY)
    adaptation = SpaceGroupAdaptation.from_basis(basis, cell, group, mesh)
    spec = SymmetrySpec.from_name("UHF").with_kpoints(mesh).with_space_group(group)
    assert spec.space_group == group
    assert spec.with_kpoints(mesh).space_group == group
    structure = spec.structure(adaptation)
    assert structure.k_axis == mesh.axis(adaptation.operations())
    assert structure.describe() == "unrestricted (real) at 8 k-points in Pm-3m (221)"
    assert spec.with_space_group(None).structure() == structure.with_k_axis(mesh.axis())
    with pytest.raises(InputError, match="needs the space-group adaptation"):
        spec.structure()
    with pytest.raises(InputError, match="another group or k-point set"):
        spec.with_kpoints(KPoints.mesh((3, 3, 3))).structure(adaptation)
    with pytest.raises(InputError, match=r"pass SymmetrySpec\.with_kpoints"):
        SymmetrySpec.from_name("RHF").with_space_group(group).structure()
    with pytest.raises(InputError, match="give one of them"):
        SymmetrySpec.from_name("RHF", point_group="C2v").with_space_group(group)

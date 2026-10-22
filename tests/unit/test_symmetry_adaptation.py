"""Symmetry-adapted basis functions and orbital labels."""

import numpy as np
import pytest

from hafnium.basis import BasisSet, Shell
from hafnium.errors import InputError, SymmetryError
from hafnium.integrals import core_hamiltonian, overlap
from hafnium.molecule import Molecule
from hafnium.scf import hartree_fock
from hafnium.symmetry import PointGroup, SymmetryAdaptation, detect_point_group
from tests import molecules
from tests.hamiltonians import ENGINE
from tests.shell_tables import LIBRARY
from tests.tolerances import INTEGRAL_TOL, IRREP_WEIGHT_TOL
from tests.unit.test_point_group_detection import moved


def adaptation_for(
    molecule: Molecule, basis_name: str
) -> tuple[Molecule, BasisSet, SymmetryAdaptation]:
    """``molecule`` in its standard frame, its basis and the adaptation."""
    detection = detect_point_group(molecule)
    molecule = detection.orient(molecule)
    basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
    return (
        molecule,
        basis,
        SymmetryAdaptation.from_basis(basis, molecule, detection.group),
    )


@pytest.mark.parametrize(
    ("molecule", "basis_name"),
    [
        (molecules.water(), "cc-pvdz"),
        (molecules.water(), "def2-universal-jkfit"),
        (molecules.benzene(), "sto-3g"),
        (molecules.methane(), "def2-svp"),
        (molecules.diazene(), "def2-svp"),
    ],
)
def test_operators_are_invariant_and_block_diagonal(
    molecule: Molecule, basis_name: str
) -> None:
    molecule, basis, adaptation = adaptation_for(molecule, basis_name)
    n = basis.n_functions
    transformation = adaptation.transformation()
    assert transformation.shape == (n, n)
    assert sum(adaptation.dimensions) == n
    assert len(adaptation.function_irreps()) == n
    np.testing.assert_allclose(
        transformation.T @ transformation, np.eye(n), atol=INTEGRAL_TOL
    )
    for matrix in (
        overlap(ENGINE, basis, molecule),
        core_hamiltonian(ENGINE, basis, molecule),
    ):
        for index in range(adaptation.group.order):
            p = adaptation.representation(index)
            np.testing.assert_allclose(p @ p.T, np.eye(n), atol=INTEGRAL_TOL)
            np.testing.assert_allclose(p @ matrix @ p.T, matrix, atol=INTEGRAL_TOL)
        blocks = adaptation.to_irrep_blocks(matrix)
        full = transformation.T @ matrix @ transformation
        start = 0
        for block, dimension in zip(blocks, adaptation.dimensions, strict=True):
            assert block.shape == (dimension, dimension)
            np.testing.assert_allclose(
                full[start : start + dimension, start : start + dimension], block
            )
            full[start : start + dimension, start : start + dimension] = 0.0
            start += dimension
        np.testing.assert_allclose(full, 0.0, atol=INTEGRAL_TOL)
        np.testing.assert_allclose(
            adaptation.from_irrep_blocks(blocks), matrix, atol=INTEGRAL_TOL
        )


def test_water_salcs_and_orbital_labels() -> None:
    water = molecules.water()
    water, basis, adaptation = adaptation_for(water, "sto-3g")
    assert adaptation.labels == ("A1", "A2", "B1", "B2")
    assert adaptation.dimensions == (4, 0, 1, 2)
    assert adaptation.atom_images[1] == (0, 2, 1)
    result = hartree_fock(ENGINE, basis, water)
    s = overlap(ENGINE, basis, water)
    coefficients = result.mo_coefficients.blocks[0]
    weights = adaptation.orbital_weights(coefficients, s)
    np.testing.assert_allclose(np.sum(weights, axis=0), 1.0, atol=IRREP_WEIGHT_TOL)
    irreps = adaptation.orbital_irreps(coefficients, s)
    assert [adaptation.labels[i] for i in irreps] == [
        "A1",
        "A1",
        "B2",
        "A1",
        "B1",
        "A1",
        "B2",
    ]
    rotated, again = adaptation.symmetrize_orbitals(
        coefficients, result.mo_energies.blocks[0], s
    )
    assert again == irreps
    np.testing.assert_allclose(rotated, coefficients, atol=IRREP_WEIGHT_TOL)


def test_mixed_orbitals_are_rejected_and_degenerate_sets_resolved() -> None:
    water = molecules.water()
    water, basis, adaptation = adaptation_for(water, "sto-3g")
    result = hartree_fock(ENGINE, basis, water)
    s = overlap(ENGINE, basis, water)
    coefficients = np.array(result.mo_coefficients.blocks[0])
    angle = np.pi / 5
    mixed = coefficients.copy()
    mixed[:, 3] = (
        np.cos(angle) * coefficients[:, 3] + np.sin(angle) * coefficients[:, 4]
    )
    mixed[:, 4] = (
        -np.sin(angle) * coefficients[:, 3] + np.cos(angle) * coefficients[:, 4]
    )
    with pytest.raises(SymmetryError, match=r"orbitals \[3, 4\]"):
        adaptation.orbital_irreps(mixed, s)
    # Declared degenerate, the pair is rotated back into pure irreps.
    energies = np.array(result.mo_energies.blocks[0])
    energies[4] = energies[3]
    pure, irreps = adaptation.symmetrize_orbitals(mixed, energies, s)
    assert sorted(adaptation.labels[i] for i in irreps[3:5]) == ["A1", "B1"]
    assert adaptation.orbital_irreps(pure, s) == irreps
    density = coefficients[:, :5] @ coefficients[:, :5].T
    np.testing.assert_allclose(
        pure[:, :5] @ pure[:, :5].T, density, atol=IRREP_WEIGHT_TOL
    )
    with pytest.raises(InputError, match="ascending"):
        adaptation.symmetrize_orbitals(mixed, energies[::-1], s)
    with pytest.raises(InputError, match="rows"):
        adaptation.orbital_weights(coefficients[:-1], s)


def test_systems_outside_the_frame_or_with_unequal_bases_are_rejected() -> None:
    water = molecules.water()
    basis = BasisSet.for_atoms(water, "sto-3g", library=LIBRARY)
    group = PointGroup.named("C2v")
    rotated = moved(water, 9)
    with pytest.raises(SymmetryError, match="standard frame"):
        SymmetryAdaptation.from_basis(basis, rotated, group)
    with pytest.raises(SymmetryError, match="standard frame"):
        SymmetryAdaptation.from_basis(basis, water, PointGroup.named("D2h"))
    extra = Shell(0, (0.5,), ((1.0,),))
    unequal = BasisSet(
        basis.name,
        basis.atom_symbols,
        (*basis.shells, extra),
        (*basis.shell_atoms, 2),
    )
    with pytest.raises(SymmetryError, match="different basis functions"):
        SymmetryAdaptation.from_basis(unequal, water, group)
    with pytest.raises(InputError, match="atom counts"):
        SymmetryAdaptation.from_basis(basis, molecules.ammonia(), group)

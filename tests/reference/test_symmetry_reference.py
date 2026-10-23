"""Point groups, frames, orbital irreps and k-point wedges against PySCF."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.integrals import overlap
from hafnium.molecule import Cell, Molecule
from hafnium.scf import (
    AufbauOccupation,
    ExactCoulombExchange,
    Hamiltonian,
    KPointDensityFittedCoulombExchange,
    ScfOptions,
    atomic_density_guess,
    hartree_fock,
    solve_scf,
)
from hafnium.symmetry import (
    DEFAULT_DEGENERACY_TOLERANCE,
    KPoints,
    SymmetryAdaptation,
    SymmetrySpec,
    degenerate_sets,
    detect_point_group,
    detect_space_group,
)
from tests import molecules
from tests.cells import diamond_helium, hexagonal_helium
from tests.hamiltonians import ENGINE
from tests.shell_tables import LIBRARY
from tests.tolerances import ENERGY_TOL, GEOMETRY_TOL, MO_ENERGY_TOL, SPIN_TOL
from tests.unit.test_point_group_detection import moved

# PySCF keeps the linear groups and the full rotation group; their
# Abelian subgroups are requested explicitly.
PYSCF_SUBGROUP = {"Dooh": "D2h", "Coov": "C2v", "SO3": "D2h"}
# (molecule, the SCF class compared, whether the frame is compared): the
# frame is compared where PySCF's choice follows the same rules, which
# excludes Ci and linear molecules (arbitrary axes), C4v (either class of
# vertical mirrors) and the cubic groups, whose axes PySCF assigns by
# swapping x and y rather than flipping a sign when its candidate triple
# is left-handed. Irrep labels within a degenerate set are compared as
# multisets, which the remaining freedom of those frames does not change.
CASES: list[tuple[Callable[[], Molecule], str | None, bool]] = [
    (molecules.water, "rhf", True),
    (molecules.water_cation, "uhf", True),
    (molecules.ammonia, "rhf", True),
    (molecules.ammonia_tilted, "rhf", True),
    (molecules.methane, "rhf", False),
    (molecules.benzene, "rhf", True),
    (molecules.ethylene, "rhf", True),
    (molecules.hydrogen_peroxide, "rhf", True),
    (molecules.diazene, "rhf", True),
    (molecules.hypochlorous_acid, "rhf", True),
    (molecules.allene, "rhf", True),
    (molecules.nitrogen, "rhf", False),
    (molecules.hydrogen_cyanide, "rhf", False),
    (molecules.sulfur_hexafluoride, None, False),
    (molecules.boron_trifluoride, "rhf", True),
    (molecules.ethane, "rhf", True),
    (lambda: molecules.ethane(staggered=False), "rhf", True),
    (molecules.inversion_symmetric, None, False),
    (molecules.square_pyramid, None, False),
    (molecules.neon, "rhf", True),
]


@pytest.mark.reference
@pytest.mark.parametrize(("build", "kind", "frame_matters"), CASES)
def test_point_groups_and_orbital_irreps_match_pyscf(
    build: Callable[[], Molecule], kind: str | None, frame_matters: bool
) -> None:
    from tests.reference.pyscf_oracle import (
        orbital_label_reference,
        point_group_reference,
    )

    molecule = moved(build(), 11)
    detection = detect_point_group(molecule)
    oriented = detection.orient(molecule)
    top, sub, axes = point_group_reference(oriented)
    assert top == detection.full_group
    assert PYSCF_SUBGROUP.get(sub, sub) == detection.group.name
    if frame_matters:
        np.testing.assert_allclose(np.abs(axes), np.eye(3), atol=GEOMETRY_TOL)
    if kind is None:
        return
    basis = BasisSet.for_atoms(oriented, "sto-3g", library=LIBRARY)
    adaptation = SymmetryAdaptation.from_basis(basis, oriented, detection.group)
    expected = orbital_label_reference(
        oriented, basis, kind, subgroup=PYSCF_SUBGROUP.get(top)
    )
    assert expected["group"] == detection.group.name
    dimensions = dict(zip(adaptation.labels, adaptation.dimensions, strict=True))
    assert {label: n for label, n in dimensions.items() if n} == expected["dimensions"]
    result = hartree_fock(
        ENGINE, basis, oriented, symmetry=SymmetrySpec.from_name(kind)
    )
    assert result.energy == pytest.approx(expected["energy"], abs=ENERGY_TOL)
    s = overlap(ENGINE, basis, oriented)
    for spin, (labels, _) in enumerate(expected["spins"]):
        energies = result.mo_energies.blocks[spin]
        _, irreps = adaptation.symmetrize_orbitals(
            result.mo_coefficients.blocks[spin], energies, s
        )
        ours = [adaptation.labels[irrep] for irrep in irreps]
        # Within a degenerate set the assignment of members to irreps is
        # arbitrary, so the sets are compared as multisets.
        for start, end in degenerate_sets(energies, DEFAULT_DEGENERACY_TOLERANCE):
            assert sorted(ours[start:end]) == sorted(labels[start:end])


# (molecule, class, electrons fixed per irrep, basis): the blocked SCF of
# ADR-0023 against PySCF's symmetry-adapted classes, including configurations
# fixed per irrep (PySCF's ``irrep_nelec``) and the generalized class.
SCF_CASES: list[tuple[Callable[[], Molecule], str, dict[str, Any] | None, str]] = [
    (molecules.water, "rhf", None, "def2-svp"),
    (molecules.water, "rhf", {"A1": 4, "B2": 4}, "sto-3g"),
    (molecules.water_cation, "uhf", None, "def2-svp"),
    (molecules.water_cation, "uhf", {"B2": (1, 0)}, "sto-3g"),
    (molecules.water_cation, "ghf", None, "sto-3g"),
    (molecules.methane, "rhf", None, "sto-3g"),
    (molecules.ammonia, "rhf", None, "sto-3g"),
    (molecules.benzene, "rhf", None, "sto-3g"),
    (molecules.ethylene, "rhf", None, "sto-3g"),
    (molecules.diazene, "rhf", None, "sto-3g"),
    (molecules.nitrogen, "rhf", None, "sto-3g"),
]


@pytest.mark.reference
@pytest.mark.parametrize(("build", "kind", "irrep_nelec", "basis_name"), SCF_CASES)
def test_blocked_scf_matches_pyscf(
    build: Callable[[], Molecule],
    kind: str,
    irrep_nelec: dict[str, Any] | None,
    basis_name: str,
) -> None:
    from tests.reference.pyscf_oracle import symmetric_scf_reference

    detection = detect_point_group(build())
    molecule = detection.orient(build())
    basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
    expected = symmetric_scf_reference(
        molecule,
        basis,
        kind,
        subgroup=PYSCF_SUBGROUP.get(detection.full_group),
        irrep_nelec=irrep_nelec,
    )
    assert expected["group"] == detection.group.name
    spec = SymmetrySpec.from_name(kind, point_group=detection.group)
    if irrep_nelec is None:
        result = hartree_fock(ENGINE, basis, molecule, symmetry=spec)
    else:
        two_body = ExactCoulombExchange.from_molecule(ENGINE, basis, molecule)
        hamiltonian = Hamiltonian.from_molecule(
            ENGINE, basis, molecule, two_body, point_group=detection.group
        )
        result = solve_scf(
            hamiltonian,
            spec,
            occupation=AufbauOccupation(molecule.n_alpha, molecule.n_beta, irrep_nelec),
            guess=atomic_density_guess(ENGINE, basis, hamiltonian.structure(spec)),
        )
    assert result.energy == pytest.approx(expected["energy"], abs=ENERGY_TOL)
    assert result.spin.square == pytest.approx(expected["spin_square"], abs=SPIN_TOL)
    assert result.electrons_per_irrep() == expected["electrons"]
    structure = result.structure
    base = structure.sector_structure
    for index, (energies, labels) in enumerate(expected["spins"]):
        ours: list[tuple[float, str]] = []
        for sector in range(structure.n_sectors):
            position = sector * base.n_stored + index
            ours.extend(
                (float(energy), str(structure.irrep_labels[position]))
                for energy in result.mo_energies.blocks[position]
            )
        ours.sort()
        # PySCF lists the occupied orbitals first, so both are sorted by energy.
        theirs = sorted(zip((float(e) for e in energies), labels, strict=True))
        sorted_energies = np.asarray([energy for energy, _ in ours])
        np.testing.assert_allclose(
            sorted_energies, [energy for energy, _ in theirs], atol=MO_ENERGY_TOL
        )
        for start, end in degenerate_sets(
            sorted_energies, DEFAULT_DEGENERACY_TOLERANCE
        ):
            assert sorted(label for _, label in ours[start:end]) == sorted(
                label for _, label in theirs[start:end]
            )


# (cell, mesh): the wedge of a Γ-centred mesh under the space group against
# PySCF's symmetric KRHF, a non-symmorphic crystal and a hexagonal slab.
WEDGE_CASES: list[tuple[Callable[[], Cell], tuple[int, ...]]] = [
    (diamond_helium, (2, 2, 2)),
    (hexagonal_helium, (3, 3)),
]


@pytest.mark.reference
@pytest.mark.parametrize(("build", "mesh_size"), WEDGE_CASES)
def test_wedge_matches_pyscf(
    build: Callable[[], Cell], mesh_size: tuple[int, ...]
) -> None:
    from tests.reference.pyscf_oracle import symmetric_kscf_reference

    cell = build()
    basis = BasisSet.for_atoms(cell, "cc-pvdz", library=LIBRARY)
    auxiliary = BasisSet.for_atoms(cell, "def2-universal-jkfit", library=LIBRARY)
    mesh = KPoints.mesh(mesh_size)
    expected = symmetric_kscf_reference(cell, basis, auxiliary, mesh_size)
    group = detect_space_group(cell)
    # PySCF keeps the symmorphic operations, which form the site group of
    # the atom at the origin; for a slab its search omits the operations
    # that flip the normal, which do not change the wedge.
    symmorphic = [op for op in group.operations if not op.has_translation]
    if cell.dimension == 3:
        assert len(symmorphic) == expected["n_operations"]
    two_body = KPointDensityFittedCoulombExchange.from_cell(
        ENGINE, basis, auxiliary, cell, mesh
    )
    hamiltonian = Hamiltonian.from_cell(
        ENGINE, basis, cell, mesh, two_body, space_group=group
    )
    spec = SymmetrySpec.from_name("RHF").with_kpoints(mesh).with_space_group(group)
    result = solve_scf(
        hamiltonian,
        spec,
        guess=atomic_density_guess(ENGINE, basis, hamiltonian.structure(spec)),
        options=ScfOptions(gradient_tolerance=1e-8),
    )
    assert result.energy == pytest.approx(expected["energy"], abs=ENERGY_TOL)
    axis = result.structure.k_axis
    assert axis is not None
    stored = axis.stored(half_zone=True)
    assert len(stored) == expected["n_wedge"]
    own, partner = axis.stored_weights(half_zone=True)
    weights = [o + p for o, p in zip(own, partner, strict=True)]
    np.testing.assert_allclose(sorted(weights), sorted(expected["weights"]))
    # Match the wedge points by their fractional coordinates: the two
    # programs choose different representatives of a star, whose orbital
    # energies agree since the rotation is unitary.
    ours = {k: np.asarray(mesh.fractional[k]) for k in stored}
    for point, energies in zip(
        expected["fractional"], expected["mo_energies"], strict=True
    ):
        images = [
            np.asarray(mesh.fractional[axis.images[g][k]])
            for k in stored
            for g in range(len(axis.images))
        ]
        matches = [
            k
            for k in stored
            for g in range(len(axis.images))
            if np.allclose(
                (np.asarray(mesh.fractional[axis.images[g][k]]) - point + 0.5) % 1.0
                - 0.5,
                0.0,
                atol=GEOMETRY_TOL,
            )
            or np.allclose(
                (-np.asarray(mesh.fractional[axis.images[g][k]]) - point + 0.5) % 1.0
                - 0.5,
                0.0,
                atol=GEOMETRY_TOL,
            )
        ]
        assert matches, (point, images)
        position = stored.index(matches[0])
        np.testing.assert_allclose(
            np.sort(np.asarray(result.mo_energies.blocks[0][position])),
            np.sort(np.asarray(energies)),
            atol=MO_ENERGY_TOL,
        )
    assert ours

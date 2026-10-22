"""Point-group detection, the Abelian subgroup and the standard frame."""

from collections.abc import Callable

import numpy as np
import pytest

from hafnium.errors import InputError
from hafnium.molecule import Molecule
from hafnium.symmetry import Operation, detect_point_group
from tests import molecules
from tests.tolerances import GEOMETRY_TOL

# (molecule, full point group, Abelian subgroup)
CASES: list[tuple[Callable[[], Molecule], str, str]] = [
    (molecules.water, "C2v", "C2v"),
    (molecules.water_cation, "C2v", "C2v"),
    (molecules.ammonia, "C3v", "Cs"),
    (molecules.ammonia_tilted, "C1", "C1"),
    (molecules.methane, "Td", "D2"),
    (molecules.benzene, "D6h", "D2h"),
    (molecules.ethylene, "D2h", "D2h"),
    (molecules.hydrogen_peroxide, "C2", "C2"),
    (molecules.diazene, "C2h", "C2h"),
    (molecules.hypochlorous_acid, "Cs", "Cs"),
    (molecules.allene, "D2d", "D2"),
    (molecules.nitrogen, "Dooh", "D2h"),
    (molecules.hydrogen_cyanide, "Coov", "C2v"),
    (molecules.sulfur_hexafluoride, "Oh", "D2h"),
    (molecules.boron_trifluoride, "D3h", "C2v"),
    (molecules.ethane, "D3d", "C2h"),
    (lambda: molecules.ethane(staggered=False), "D3h", "C2v"),
    (molecules.inversion_symmetric, "Ci", "Ci"),
    (molecules.square_pyramid, "C4v", "C2v"),
    (molecules.neon, "SO3", "D2h"),
    (molecules.icosahedron, "Ih", "D2h"),
]


def moved(molecule: Molecule, seed: int) -> Molecule:
    """``molecule`` rotated and translated at random."""
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    positions = molecule.positions_array() @ rotation.T + 3.0 * rng.standard_normal(3)
    return Molecule.from_atoms(
        [
            (atom.symbol, row)
            for atom, row in zip(molecule.atoms, positions, strict=True)
        ],
        charge=molecule.charge,
        spin=molecule.spin,
        unit="bohr",
    )


def distances(molecule: Molecule) -> np.ndarray:
    positions = molecule.positions_array()
    return np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)


def assert_operations_hold(
    molecule: Molecule, operations: tuple[Operation, ...]
) -> None:
    """Every sign operation maps the atoms onto atoms of the same element."""
    positions = molecule.positions_array()
    for operation in operations:
        image = positions * np.asarray(operation.signs, dtype=float)
        for atom, row in enumerate(image):
            gaps: np.ndarray = np.linalg.norm(positions - np.asarray(row), axis=1)
            partner = int(np.argmin(gaps))
            assert gaps[partner] < GEOMETRY_TOL
            assert molecule.symbols[partner] == molecule.symbols[atom]


@pytest.mark.parametrize(("build", "full", "abelian"), CASES)
def test_groups_are_detected_in_any_frame(
    build: Callable[[], Molecule], full: str, abelian: str
) -> None:
    molecule = build()
    detection = detect_point_group(molecule)
    assert (detection.full_group, detection.group.name) == (full, abelian)
    for seed in (1, 2):
        rotated = moved(molecule, seed)
        detection = detect_point_group(rotated)
        assert (detection.full_group, detection.group.name) == (full, abelian)
        oriented = detection.orient(rotated)
        assert oriented.charge == molecule.charge
        assert oriented.spin == molecule.spin
        np.testing.assert_allclose(
            distances(oriented), distances(molecule), atol=GEOMETRY_TOL
        )
        assert_operations_hold(oriented, detection.group.operations)
        again = detect_point_group(oriented)
        assert again.is_standard
        assert (again.full_group, again.group.name) == (full, abelian)
        np.testing.assert_allclose(
            again.orient(oriented).positions_array(),
            oriented.positions_array(),
            atol=GEOMETRY_TOL,
        )


def test_standard_frames_follow_the_conventions() -> None:
    # Planar C2v: the molecular plane is yz and the twofold axis z.
    water = detect_point_group(moved(molecules.water(), 3))
    positions = water.orient(moved(molecules.water(), 3)).positions_array()
    np.testing.assert_allclose(positions[:, 0], 0.0, atol=GEOMETRY_TOL)
    assert abs(positions[1, 1]) > 1.0
    assert abs(positions[1, 2]) > 0.1
    # Planar D2h: the plane is yz and z passes through the carbons.
    ethylene = moved(molecules.ethylene(), 4)
    positions = detect_point_group(ethylene).orient(ethylene).positions_array()
    np.testing.assert_allclose(positions[:, 0], 0.0, atol=GEOMETRY_TOL)
    np.testing.assert_allclose(positions[:2, 1], 0.0, atol=GEOMETRY_TOL)
    # A principal axis of order six is z and a perpendicular twofold axis
    # through the first atom is x.
    benzene = moved(molecules.benzene(), 5)
    positions = detect_point_group(benzene).orient(benzene).positions_array()
    np.testing.assert_allclose(positions[:, 2], 0.0, atol=GEOMETRY_TOL)
    np.testing.assert_allclose(positions[0, 1:], 0.0, atol=GEOMETRY_TOL)
    # The twofold axes of a tetrahedron are the coordinate axes closest to
    # the input frame: a methane already on them keeps its frame.
    assert detect_point_group(molecules.methane()).is_standard
    # A C3v molecule descends to Cs with the mirror plane as xy.
    ammonia = moved(molecules.ammonia(), 6)
    positions = detect_point_group(ammonia).orient(ammonia).positions_array()
    assert np.sum(np.abs(positions[:, 2]) < GEOMETRY_TOL) == 2
    # A linear molecule lies along z.
    hcn = moved(molecules.hydrogen_cyanide(), 7)
    positions = detect_point_group(hcn).orient(hcn).positions_array()
    np.testing.assert_allclose(positions[:, :2], 0.0, atol=GEOMETRY_TOL)


def test_c1_keeps_its_frame_and_ci_is_centred() -> None:
    tilted = molecules.ammonia_tilted()
    detection = detect_point_group(tilted)
    assert detection.is_standard
    assert detection.orient(tilted) == tilted
    shifted = moved(molecules.inversion_symmetric(), 8)
    detection = detect_point_group(shifted)
    centre = detection.orient(shifted).positions_array()
    np.testing.assert_allclose(centre[:4] + centre[4:], 0.0, atol=GEOMETRY_TOL)


def test_tolerance_controls_the_detection() -> None:
    # One hydrogen of ammonia moved out of its mirror plane by 1e-3 Bohr
    # leaves no symmetry element at the default tolerance.
    ammonia = molecules.ammonia()
    atoms = [(atom.symbol, atom.position) for atom in ammonia.atoms]
    symbol, (x, y, z) = atoms[1]
    atoms[1] = (symbol, (x, y + 1e-3, z))
    distorted = Molecule.from_atoms(atoms, unit="bohr")
    assert detect_point_group(distorted).full_group == "C1"
    loose = detect_point_group(distorted, tolerance=1e-2)
    assert (loose.full_group, loose.group.name) == ("C3v", "Cs")
    with pytest.raises(InputError, match="positive"):
        detect_point_group(ammonia, tolerance=0.0)

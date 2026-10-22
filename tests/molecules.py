"""Molecules shared by the test suites, given in Angstrom."""

import math

from hafnium.molecule import Molecule

SQRT3 = math.sqrt(3.0)


def water() -> Molecule:
    """Water near its equilibrium geometry, C2v with the molecular plane yz."""
    return Molecule.from_atoms(
        [
            ("O", (0.0, 0.0, 0.1173)),
            ("H", (0.0, 0.7572, -0.4692)),
            ("H", (0.0, -0.7572, -0.4692)),
        ]
    )


def ammonia_tilted() -> Molecule:
    """Ammonia in a general orientation, with no spatial symmetry."""
    return Molecule.from_atoms(
        [
            ("N", (0.1, -0.05, 0.0)),
            ("H", (0.9, 0.3, 0.4)),
            ("H", (-0.5, 0.8, -0.2)),
            ("H", (-0.4, -0.7, 0.6)),
        ]
    )


def water_cation() -> Molecule:
    """The water radical cation, a doublet."""
    molecule = water()
    return Molecule(molecule.atoms, charge=1, spin=1)


def hydrogen(distance_bohr: float = 1.4) -> Molecule:
    """H2 along z at the given bond length in Bohr."""
    return Molecule.from_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, distance_bohr))], unit="bohr"
    )


def ammonia() -> Molecule:
    """Ammonia, C3v with the threefold axis along z and one hydrogen in the xz plane."""
    half = 0.94 * SQRT3 / 2
    return Molecule.from_atoms(
        [
            ("N", (0.0, 0.0, 0.1)),
            ("H", (0.94, 0.0, -0.3)),
            ("H", (-0.47, half, -0.3)),
            ("H", (-0.47, -half, -0.3)),
        ]
    )


def methane() -> Molecule:
    """Methane, Td with the twofold axes along the coordinate axes."""
    return Molecule.from_atoms(
        [("C", (0.0, 0.0, 0.0))]
        + [("H", (0.63 * x, 0.63 * y, 0.63 * x * y)) for x in (1, -1) for y in (1, -1)]
    )


def benzene() -> Molecule:
    """Benzene in the xy plane, D6h with the first carbon on the x axis."""
    return Molecule.from_atoms(
        [
            (
                "C",
                (
                    1.39 * math.cos(k * math.pi / 3),
                    1.39 * math.sin(k * math.pi / 3),
                    0.0,
                ),
            )
            for k in range(6)
        ]
        + [
            (
                "H",
                (
                    2.47 * math.cos(k * math.pi / 3),
                    2.47 * math.sin(k * math.pi / 3),
                    0.0,
                ),
            )
            for k in range(6)
        ]
    )


def ethylene() -> Molecule:
    """Ethylene in the yz plane with the double bond along z, D2h."""
    return Molecule.from_atoms(
        [
            ("C", (0.0, 0.0, 0.67)),
            ("C", (0.0, 0.0, -0.67)),
            ("H", (0.0, 0.92, 1.23)),
            ("H", (0.0, -0.92, 1.23)),
            ("H", (0.0, 0.92, -1.23)),
            ("H", (0.0, -0.92, -1.23)),
        ]
    )


def hydrogen_peroxide() -> Molecule:
    """Hydrogen peroxide, C2 with the twofold axis along z."""
    return Molecule.from_atoms(
        [
            ("O", (0.0, 0.73, 0.0)),
            ("O", (0.0, -0.73, 0.0)),
            ("H", (0.8, 0.9, 0.5)),
            ("H", (-0.8, -0.9, 0.5)),
        ]
    )


def diazene() -> Molecule:
    """Trans-diazene in the yz plane, C2h with the twofold axis along x."""
    return Molecule.from_atoms(
        [
            ("N", (0.0, 0.62, 0.0)),
            ("N", (0.0, -0.62, 0.0)),
            ("H", (0.0, 1.0, 0.95)),
            ("H", (0.0, -1.0, -0.95)),
        ]
    )


def hypochlorous_acid() -> Molecule:
    """Hypochlorous acid in the xy plane, Cs."""
    return Molecule.from_atoms(
        [("O", (0.0, 0.0, 0.0)), ("H", (0.96, 0.0, 0.0)), ("Cl", (-0.5, 1.6, 0.0))]
    )


def allene() -> Molecule:
    """Allene along z with perpendicular CH2 planes, D2d."""
    return Molecule.from_atoms(
        [
            ("C", (0.0, 0.0, 0.0)),
            ("C", (0.0, 0.0, 1.3)),
            ("C", (0.0, 0.0, -1.3)),
            ("H", (0.93, 0.0, 1.86)),
            ("H", (-0.93, 0.0, 1.86)),
            ("H", (0.0, 0.93, -1.86)),
            ("H", (0.0, -0.93, -1.86)),
        ]
    )


def nitrogen() -> Molecule:
    """N2 along z."""
    return Molecule.from_atoms([("N", (0.0, 0.0, 0.55)), ("N", (0.0, 0.0, -0.55))])


def hydrogen_cyanide() -> Molecule:
    """HCN along z."""
    return Molecule.from_atoms(
        [("H", (0.0, 0.0, -1.06)), ("C", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.16))]
    )


def sulfur_hexafluoride() -> Molecule:
    """SF6 with the fluorines on the coordinate axes, Oh."""
    return Molecule.from_atoms(
        [("S", (0.0, 0.0, 0.0))]
        + [
            ("F", tuple(1.56 * sign * float(k == axis) for k in range(3)))
            for axis in range(3)
            for sign in (1, -1)
        ]
    )


def boron_trifluoride() -> Molecule:
    """BF3 in the xy plane with one fluorine on the x axis, D3h."""
    return Molecule.from_atoms(
        [
            ("B", (0.0, 0.0, 0.0)),
            ("F", (1.3, 0.0, 0.0)),
            ("F", (-0.65, 1.3 * SQRT3 / 2, 0.0)),
            ("F", (-0.65, -1.3 * SQRT3 / 2, 0.0)),
        ]
    )


def ethane(*, staggered: bool = True) -> Molecule:
    """Ethane along z, D3d when staggered and D3h when eclipsed."""
    offset = math.pi / 3 if staggered else 0.0
    angles = (0.0, 2 * math.pi / 3, 4 * math.pi / 3)
    return Molecule.from_atoms(
        [("C", (0.0, 0.0, 0.77)), ("C", (0.0, 0.0, -0.77))]
        + [("H", (1.02 * math.cos(a), 1.02 * math.sin(a), 1.13)) for a in angles]
        + [
            ("H", (1.02 * math.cos(a + offset), 1.02 * math.sin(a + offset), -1.13))
            for a in angles
        ]
    )


def inversion_symmetric() -> Molecule:
    """A C2H2Cl2F2 conformer whose only symmetry element is the inversion centre."""
    half = [
        ("C", (0.7, 0.1, 0.0)),
        ("F", (1.0, 1.2, 0.4)),
        ("Cl", (1.4, -0.6, -1.0)),
        ("H", (0.9, 0.2, 1.0)),
    ]
    return Molecule.from_atoms(
        half + [(symbol, (-x, -y, -z)) for symbol, (x, y, z) in half]
    )


def square_pyramid() -> Molecule:
    """Five hydrogens, one above a square, C4v."""
    return Molecule.from_atoms(
        [("H", (0.0, 0.0, 1.0))]
        + [
            ("H", (x, y, 0.0))
            for x, y in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
        ],
        spin=1,
    )


def neon() -> Molecule:
    """A neon atom off the origin."""
    return Molecule.from_atoms([("Ne", (0.3, 0.2, 0.1))])

"""Molecules shared by the test suites, given in Angstrom."""

from hafnium.molecule import Molecule


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

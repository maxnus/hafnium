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

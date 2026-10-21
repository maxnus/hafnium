"""Unit conversions at the library boundary (CONVENTIONS.md §3).

Everything inside the library is in atomic units. Constants are CODATA 2018.
"""

BOHR_IN_ANGSTROM = 0.529177210903
"""One Bohr radius in Ångström."""

ANGSTROM_IN_BOHR = 1.0 / BOHR_IN_ANGSTROM
"""One Ångström in Bohr radii."""

HARTREE_IN_EV = 27.211386245988
"""One Hartree in electronvolts."""


def angstrom_to_bohr(length_angstrom: float) -> float:
    """Convert a length from Ångström to Bohr."""
    return length_angstrom * ANGSTROM_IN_BOHR


def bohr_to_angstrom(length: float) -> float:
    """Convert a length from Bohr to Ångström."""
    return length * BOHR_IN_ANGSTROM

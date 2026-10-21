"""One-electron integrals of a molecule in its atomic basis."""

from __future__ import annotations

from hafnium.backend import Array, host_array
from hafnium.basis import BasisSet
from hafnium.integrals.engine import IntegralEngine
from hafnium.integrals.shell_table import ShellTable
from hafnium.molecule import Molecule

_NO_TRANSLATION = host_array([[0.0, 0.0, 0.0]])


def _table(basis: BasisSet, molecule: Molecule) -> ShellTable:
    return ShellTable.from_basis(basis, molecule.positions_array())


def overlap(engine: IntegralEngine, basis: BasisSet, molecule: Molecule) -> Array:
    """Overlap matrix ``[ao, ao]``."""
    table = _table(basis, molecule)
    return engine.overlap(table, table, _NO_TRANSLATION)[0]


def kinetic(engine: IntegralEngine, basis: BasisSet, molecule: Molecule) -> Array:
    """Kinetic energy matrix ``[ao, ao]``."""
    table = _table(basis, molecule)
    return engine.kinetic(table, table, _NO_TRANSLATION)[0]


def nuclear_attraction(
    engine: IntegralEngine,
    basis: BasisSet,
    molecule: Molecule,
    *,
    omega: float | None = None,
) -> Array:
    """Nuclear attraction matrix ``[ao, ao]`` of all nuclei of ``molecule``.

    With ``omega`` the long-range kernel ``erf(ω r) / r`` replaces ``1 / r``.
    """
    table = _table(basis, molecule)
    return engine.nuclear_attraction(
        table,
        table,
        _NO_TRANSLATION,
        host_array(molecule.nuclear_charges),
        molecule.positions_array(),
        omega=omega,
    )[0]


def core_hamiltonian(
    engine: IntegralEngine, basis: BasisSet, molecule: Molecule
) -> Array:
    """Core Hamiltonian ``[ao, ao]``: kinetic energy plus nuclear attraction."""
    return kinetic(engine, basis, molecule) + nuclear_attraction(
        engine, basis, molecule
    )

"""One-electron integrals of a molecule in its atomic basis, and their derivatives.

The results are arrays of the active backend (ADR-0036). A derivative with
respect to the centre of a basis function is given for the bra,
``⟨∂_A μ | O | ν⟩``; the ket derivative is its transpose for the symmetric
operators of a molecule (ADR-0035).
"""

from __future__ import annotations

from hafnium.backend import Array, from_host, host_array
from hafnium.basis import BasisSet
from hafnium.integrals.engine import CoulombKernel, IntegralEngine, PointCharges
from hafnium.integrals.shell_table import ShellTable
from hafnium.molecule import Molecule


def _no_translation() -> Array:
    """Return the single zero translation of a molecule, fresh for every call."""
    return host_array([[0.0, 0.0, 0.0]])


def _table(basis: BasisSet, molecule: Molecule) -> ShellTable:
    return ShellTable.from_basis(basis, molecule)


def overlap(engine: IntegralEngine, basis: BasisSet, molecule: Molecule) -> Array:
    """Overlap matrix ``[ao, ao]``."""
    table = _table(basis, molecule)
    return from_host(engine.overlap(table, table, _no_translation())[0])


def kinetic(engine: IntegralEngine, basis: BasisSet, molecule: Molecule) -> Array:
    """Kinetic energy matrix ``[ao, ao]``."""
    table = _table(basis, molecule)
    return from_host(engine.kinetic(table, table, _no_translation())[0])


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
    return _attraction(engine, table, molecule, omega)


def _attraction(
    engine: IntegralEngine, table: ShellTable, molecule: Molecule, omega: float | None
) -> Array:
    attraction = engine.nuclear_attraction(
        table,
        table,
        _no_translation(),
        PointCharges.from_system(molecule),
        kernel=CoulombKernel(omega),
    )[0]
    return from_host(attraction)


def core_hamiltonian(
    engine: IntegralEngine, basis: BasisSet, molecule: Molecule
) -> Array:
    """Core Hamiltonian ``[ao, ao]``: kinetic energy plus nuclear attraction."""
    table = _table(basis, molecule)
    kinetic_energy = from_host(engine.kinetic(table, table, _no_translation())[0])
    return kinetic_energy + _attraction(engine, table, molecule, None)


def overlap_derivative(
    engine: IntegralEngine, basis: BasisSet, molecule: Molecule
) -> Array:
    """Differentiate the overlap with respect to the bra centre.

    ``⟨∂_A μ | ν⟩`` as ``[3, ao, ao]``.
    """
    table = _table(basis, molecule)
    return from_host(engine.overlap_derivative(table, table, _no_translation())[0])


def kinetic_derivative(
    engine: IntegralEngine, basis: BasisSet, molecule: Molecule
) -> Array:
    """Differentiate the kinetic energy with respect to the bra centre.

    ``⟨∂_A μ | −½∇² | ν⟩`` as ``[3, ao, ao]``.
    """
    table = _table(basis, molecule)
    return from_host(engine.kinetic_derivative(table, table, _no_translation())[0])


def nuclear_attraction_derivative(
    engine: IntegralEngine,
    basis: BasisSet,
    molecule: Molecule,
    *,
    omega: float | None = None,
) -> tuple[Array, Array]:
    """Differentiate the nuclear attraction of all nuclei of ``molecule``.

    With respect to the bra centre, ``⟨∂_A μ | V | ν⟩`` as ``[3, ao, ao]``,
    and with respect to the position of each nucleus, ``⟨μ | ∂_K V_K | ν⟩``
    as ``[atom, 3, ao, ao]``. With ``omega`` the long-range kernel
    ``erf(ω r) / r`` replaces ``1 / r``.
    """
    table = _table(basis, molecule)
    by_center, by_nucleus = engine.nuclear_attraction_derivative(
        table,
        table,
        _no_translation(),
        PointCharges.from_system(molecule),
        kernel=CoulombKernel(omega),
    )
    return from_host(by_center[0]), from_host(by_nucleus[0])

"""Spherically averaged atoms and the superposition of atomic densities.

The default initial guess of :func:`~hafnium.scf.hartree_fock`: every
element of the molecule is converged as a neutral, spin-averaged atom with
fractional occupation of its open shell in the functions of its own centre,
and the atomic densities are placed on the diagonal of the molecular
density. The result is spin-symmetric and carries the nuclear charge rather
than the electron count; the first Fock build of the molecular iteration
corrects both.
"""

from __future__ import annotations

from hafnium.backend import Array, array_namespace
from hafnium.basis import BasisSet
from hafnium.integrals import IntegralEngine
from hafnium.molecule import Atom, Molecule, atomic_number
from hafnium.scf.coulomb_exchange import ExactCoulombExchange
from hafnium.scf.driver import ScfOptions, solve_scf
from hafnium.scf.hamiltonian import Hamiltonian
from hafnium.scf.occupation import AveragedOccupation
from hafnium.structured import BlockMatrix, BlockStructure
from hafnium.symmetry import SymmetrySpec


def atomic_density(
    engine: IntegralEngine, basis: BasisSet, atom: int, *, options: ScfOptions | None
) -> Array:
    """Per-spin density ``[ao_A, ao_A]`` of the neutral spherical atom ``atom``.

    The atom is converged in the restricted class with its electrons
    averaged over both spins and over the degenerate orbitals of its open
    shell, in the shells of ``basis`` centred on it.
    """
    shells = tuple(
        shell
        for shell, centre in zip(basis.shells, basis.shell_atoms, strict=True)
        if centre == atom
    )
    symbol = basis.atom_symbols[atom]
    n_electrons = atomic_number(symbol)
    spherical = Molecule((Atom(symbol, (0.0, 0.0, 0.0)),), spin=n_electrons % 2)
    atomic_basis = BasisSet(basis.name, (symbol,), shells, (0,) * len(shells))
    two_body = ExactCoulombExchange.from_molecule(engine, atomic_basis, spherical)
    hamiltonian = Hamiltonian.from_molecule(engine, atomic_basis, spherical, two_body)
    result = solve_scf(
        hamiltonian,
        SymmetrySpec.from_name("RHF"),
        occupation=AveragedOccupation(n_electrons / 2, n_electrons / 2),
        options=options,
    )
    return result.density.blocks[0]


def atomic_density_guess(
    engine: IntegralEngine,
    basis: BasisSet,
    structure: BlockStructure,
    *,
    options: ScfOptions | None = None,
) -> BlockMatrix:
    """Superposition of atomic densities in ``structure``.

    Each element is converged once; the per-spin atomic densities are
    placed on the diagonal of the molecular density, which is then embedded
    into ``structure``.
    """
    offsets = basis.shell_offsets()
    starts = [
        offsets[index]
        for index, centre in enumerate(basis.shell_atoms)
        if index == 0 or centre != basis.shell_atoms[index - 1]
    ]
    ends = [*starts[1:], basis.n_functions]
    densities: dict[str, Array] = {}
    columns: list[Array] = []
    xp = None
    for atom, symbol in enumerate(basis.atom_symbols):
        if symbol not in densities:
            densities[symbol] = atomic_density(engine, basis, atom, options=options)
        block = densities[symbol]
        xp = array_namespace(block)
        before = xp.zeros((starts[atom], block.shape[1]), dtype=xp.float64)
        after = xp.zeros(
            (basis.n_functions - ends[atom], block.shape[1]), dtype=xp.float64
        )
        columns.append(xp.concat([before, block, after], axis=0))
    assert xp is not None
    total = xp.concat(columns, axis=1)
    return BlockMatrix(BlockStructure.spatial(), (total,)).embed(structure)

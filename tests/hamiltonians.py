"""Hamiltonians of test molecules with exact or fitted two-electron terms."""

from hafnium.basis import BasisSet
from hafnium.integrals import RustEngine
from hafnium.molecule import Molecule
from hafnium.scf import (
    DensityFittedCoulombExchange,
    ExactCoulombExchange,
    Hamiltonian,
)
from tests.shell_tables import LIBRARY

ENGINE = RustEngine()


def hamiltonian_for(
    molecule: Molecule, basis_name: str, auxiliary_name: str | None = None
) -> tuple[BasisSet, Hamiltonian]:
    """Basis set and Hamiltonian of ``molecule`` in the named basis sets."""
    basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
    if auxiliary_name is None:
        two_body = ExactCoulombExchange.from_molecule(ENGINE, basis, molecule)
    else:
        auxiliary = BasisSet.for_atoms(molecule, auxiliary_name, library=LIBRARY)
        two_body = DensityFittedCoulombExchange.from_molecule(
            ENGINE, basis, auxiliary, molecule
        )
    return basis, Hamiltonian.from_molecule(ENGINE, basis, molecule, two_body)

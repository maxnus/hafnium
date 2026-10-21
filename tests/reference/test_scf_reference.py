"""Hartree–Fock energies and orbital energies against PySCF, and stored data."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.basis import BasisLibrary, BasisSet
from hafnium.integrals import RustEngine
from hafnium.molecule import Molecule
from hafnium.scf import (
    DensityFittedCoulombExchange,
    ExactCoulombExchange,
    Hamiltonian,
    ScfResult,
    atomic_density_guess,
    solve_scf,
)
from hafnium.structured import BlockStructure, Ring
from hafnium.symmetry import SymmetrySpec
from tests.molecules import ammonia_tilted, water, water_cation
from tests.tolerances import ENERGY_TOL, MO_ENERGY_TOL

LIBRARY = BasisLibrary(offline=True)
ENGINE = RustEngine()
AUXILIARY = "def2-universal-jkfit"

MOLECULES = {
    "water": water(),
    "water_cation": water_cation(),
    "ammonia_tilted": ammonia_tilted(),
}
# (molecule, basis, class): closed shells in the restricted class, the
# doublet cation in the unrestricted and generalized classes.
CASES = [
    ("water", "sto-3g", "rhf"),
    ("water", "def2-svp", "rhf"),
    ("water", "cc-pvdz", "rhf"),
    ("ammonia_tilted", "def2-svp", "rhf"),
    ("water_cation", "sto-3g", "uhf"),
    ("water_cation", "def2-svp", "uhf"),
    ("water_cation", "def2-svp", "ghf"),
]
STORED_CASES = CASES[:1] + CASES[4:5]
SYMMETRY = {
    "rhf": SymmetrySpec.from_name("RHF"),
    "uhf": SymmetrySpec.from_name("UHF"),
    "ghf": SymmetrySpec.from_name("GHF"),
}


def hamiltonian_for(
    molecule: Molecule, basis: BasisSet, auxiliary: BasisSet | None = None
) -> Hamiltonian:
    two_body = (
        ExactCoulombExchange.from_molecule(ENGINE, basis, molecule)
        if auxiliary is None
        else DensityFittedCoulombExchange.from_molecule(
            ENGINE, basis, auxiliary, molecule
        )
    )
    return Hamiltonian.from_molecule(ENGINE, basis, molecule, two_body)


def converge(
    hamiltonian: Hamiltonian, basis: BasisSet, kind: str, guess: str
) -> ScfResult:
    """Converge from the ``"core"`` or ``"atoms"`` guess."""
    symmetry = SYMMETRY[kind]
    if guess == "core":
        return solve_scf(hamiltonian, symmetry)
    start = atomic_density_guess(ENGINE, basis, symmetry.structure())
    return solve_scf(hamiltonian, symmetry, guess=start)


def compute(
    name: str, basis_name: str, kind: str, guess: str = "atoms"
) -> tuple[Molecule, BasisSet, ScfResult]:
    molecule = MOLECULES[name]
    basis = BasisSet.for_molecule(molecule, basis_name, library=LIBRARY)
    return (
        molecule,
        basis,
        converge(hamiltonian_for(molecule, basis), basis, kind, guess),
    )


def check_against(result: ScfResult, expected: dict[str, Any]) -> None:
    assert result.energy == pytest.approx(expected["energy"], abs=ENERGY_TOL)
    mo_energies = result.mo_energies.to_dense()
    if result.structure.derived_partner is not None:
        mo_energies = mo_energies[: mo_energies.shape[0] // 2]
    np.testing.assert_allclose(
        np.sort(mo_energies),
        np.sort(np.asarray(expected["mo_energies"]).ravel()),
        atol=MO_ENERGY_TOL,
    )


@pytest.mark.reference
@pytest.mark.parametrize("guess", ["core", "atoms"])
@pytest.mark.parametrize(("name", "basis_name", "kind"), CASES)
def test_hartree_fock_matches_pyscf(
    name: str, basis_name: str, kind: str, guess: str
) -> None:
    from tests.reference.pyscf_oracle import scf_reference

    molecule, basis, result = compute(name, basis_name, kind, guess)
    check_against(result, scf_reference(molecule, basis, kind))


@pytest.mark.reference
@pytest.mark.parametrize("kind", ["rhf", "uhf"])
def test_density_fitted_hartree_fock_matches_pyscf(kind: str) -> None:
    from tests.reference.pyscf_oracle import scf_reference

    molecule = MOLECULES["water" if kind == "rhf" else "water_cation"]
    basis = BasisSet.for_molecule(molecule, "def2-svp", library=LIBRARY)
    auxiliary = BasisSet.for_molecule(molecule, AUXILIARY, library=LIBRARY)
    hamiltonian = hamiltonian_for(molecule, basis, auxiliary)
    result = converge(hamiltonian, basis, kind, "atoms")
    check_against(result, scf_reference(molecule, basis, kind, auxiliary=auxiliary))


@pytest.mark.reference
def test_complex_generalized_state_matches_pyscf_from_the_same_guess() -> None:
    """Both codes return from a complex perturbation to the real cation energy."""
    from tests.reference.pyscf_oracle import scf_reference

    molecule = MOLECULES["water_cation"]
    basis = BasisSet.for_molecule(molecule, "sto-3g", library=LIBRARY)
    hamiltonian = hamiltonian_for(molecule, basis)
    real = solve_scf(hamiltonian, SYMMETRY["ghf"])
    rng = np.random.default_rng(5)
    n = 2 * basis.n_functions
    x = rng.standard_normal((n, n))
    guess = real.density.embed(BlockStructure.generalized(Ring.COMPLEX)).map_blocks(
        lambda block: block + 0.05j * (x - x.T)
    )
    result = solve_scf(hamiltonian, SymmetrySpec.from_name("complex GHF"), guess=guess)
    expected = scf_reference(molecule, basis, "ghf", guess=guess.blocks[0])
    check_against(result, expected)
    assert result.energy == pytest.approx(real.energy, abs=ENERGY_TOL)


@pytest.mark.reference
@pytest.mark.parametrize(("name", "basis_name", "kind"), STORED_CASES)
def test_hartree_fock_matches_stored_data(
    name: str, basis_name: str, kind: str
) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data") / f"scf_{name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["method"] == kind
    _, _, result = compute(name, basis_name, kind)
    check_against(result, stored)

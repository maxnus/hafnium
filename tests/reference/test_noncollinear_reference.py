"""The multi-collinear functional and the generalized Kohn–Sham classes versus PySCF."""

import json
from importlib import resources
from typing import Any

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.grid import MolecularGrid
from hafnium.scf import ExchangeCorrelation, ScfResult, kohn_sham
from hafnium.structured import BlockStructure, Ring
from hafnium.symmetry import SymmetrySpec
from hafnium.xc import Functional, NoncollinearOptions, evaluate_noncollinear
from tests.hamiltonians import ENGINE
from tests.molecules import water, water_cation
from tests.reference.pyscf_oracle import (
    gks_potential_reference,
    gks_reference,
    noncollinear_functional_reference,
)
from tests.shell_tables import LIBRARY
from tests.tolerances import (
    KOHN_SHAM_ENERGY_TOL,
    LIBXC_REL_TOL,
    MO_ENERGY_TOL,
    NONCOLLINEAR_LIMIT_TOL,
    STRUCTURE_TOL,
)
from tests.unit.test_kohn_sham import noncollinear_density
from tests.unit.test_xc import POINTS, noncollinear_at

# (Hafnium name, PySCF code): one functional per rung.
FUNCTIONALS = [("svwn", "LDA,VWN"), ("pbe", "PBE"), ("tpss", "TPSS")]
# (molecule, basis, Hafnium name, PySCF code, PySCF's noncollinear scheme) of
# the converged states compared with PySCF's GKS, and the recorded one.
GKS_CASES = [
    ("water_cation", "def2-svp", "svwn", "LDA,VWN", "ncol"),
    ("water_cation", "def2-svp", "pbe", "PBE", "mcol"),
]
STORED_CASES = [("water_cation", "sto-3g", "svwn", "LDA,VWN", "ncol")]
# The spin directions of the five occupied orbitals of the test density.
DIRECTIONS = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.6, 0.0, 0.8],
        [0.0, -0.8, 0.6],
    ]
)


@pytest.mark.reference
@pytest.mark.parametrize(("name", "code"), FUNCTIONALS)
def test_kernel_matches_mcfun_on_the_sphere(name: str, code: str) -> None:
    """Same rule, no collinear branch: the two implementations agree to rounding."""
    functional = Functional.named(name)
    density = noncollinear_at(POINTS, polarization=0.6, twist=1.0)
    options = NoncollinearOptions(lebedev_points=770, collinear_threshold=None)
    values = evaluate_noncollinear(functional, density, options)
    energy, derivatives = noncollinear_functional_reference(code, density)
    np.testing.assert_allclose(values.energy_density, energy, rtol=LIBXC_REL_TOL)
    ours = (
        values.v_rho,
        values.v_m,
        values.v_grad_rho,
        values.v_grad_m,
        values.v_tau,
        values.v_tau_m,
    )
    for mine, theirs in zip(ours, derivatives, strict=True):
        assert (mine is None) is (theirs is None)
        if mine is not None and theirs is not None:
            np.testing.assert_allclose(
                mine, theirs, rtol=LIBXC_REL_TOL, atol=LIBXC_REL_TOL
            )


@pytest.mark.reference
@pytest.mark.parametrize(
    ("name", "code", "collinear"),
    [("svwn", "LDA,VWN", "ncol"), ("pbe", "PBE", "mcol"), ("tpss", "TPSS", "mcol")],
)
def test_term_matches_pyscf_for_a_noncollinear_density(
    name: str, code: str, collinear: str
) -> None:
    """Energy and spinor potential of one noncollinear density on PySCF's grid.

    Five orbitals of water with five spin directions. PySCF's ``ncol`` is
    the closed form Hafnium takes for an LDA; for the other rungs both
    codes sample the same Lebedev rule with the collinear branch off,
    since the two define it differently.
    """
    molecule = water()
    basis = BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY)
    density = noncollinear_density(ENGINE, basis, molecule, DIRECTIONS)
    (dense,) = density.spin_blocks()
    expected = gks_potential_reference(
        molecule, basis, code, np.asarray(dense), collinear=collinear
    )
    grid = MolecularGrid(
        np.asarray(expected["grid_points"]),
        np.asarray(expected["grid_weights"]),
        np.zeros(expected["grid_weights"].shape[0], dtype=np.int64),
    )
    term = ExchangeCorrelation.from_molecule(
        ENGINE, basis, molecule, Functional.named(name), grid=grid
    )
    term = dataclasses_replace(
        term, NoncollinearOptions(lebedev_points=770, collinear_threshold=None)
    )
    energy, potential = term.evaluate(density)
    assert energy == pytest.approx(expected["energy"], abs=KOHN_SHAM_ENERGY_TOL)
    np.testing.assert_allclose(
        potential.blocks[0], expected["potential"], atol=STRUCTURE_TOL
    )


def dataclasses_replace(
    term: ExchangeCorrelation, options: NoncollinearOptions
) -> ExchangeCorrelation:
    import dataclasses

    return dataclasses.replace(term, noncollinear=options)


def converge(molecule_name: str, basis_name: str, functional: str) -> ScfResult:
    molecule = {"water_cation": water_cation, "water": water}[molecule_name]()
    basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
    result = kohn_sham(
        ENGINE,
        basis,
        molecule,
        functional,
        symmetry=SymmetrySpec.from_name("complex GHF"),
    )
    assert result.converged
    assert result.structure == BlockStructure.generalized(Ring.COMPLEX)
    return result


def check_against(result: ScfResult, expected: dict[str, Any]) -> None:
    assert result.energy == pytest.approx(
        expected["energy"], abs=NONCOLLINEAR_LIMIT_TOL
    )
    ours = np.sort(np.asarray(result.mo_energies.blocks[0]))
    theirs = np.sort(np.asarray(expected["mo_energies"]))
    np.testing.assert_allclose(ours, theirs, atol=MO_ENERGY_TOL)


@pytest.mark.reference
@pytest.mark.parametrize(
    ("molecule_name", "basis_name", "name", "code", "collinear"), GKS_CASES
)
def test_generalized_kohn_sham_matches_pyscf_gks(
    molecule_name: str, basis_name: str, name: str, code: str, collinear: str
) -> None:
    """The doublet cation in the generalized class, a collinear state on both sides.

    PySCF's ``ncol`` LDA is the closed form Hafnium uses; its ``mcol`` GGA
    agrees for a collinear density at every point of either branch.
    """
    molecule = water_cation()
    basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
    expected = gks_reference(molecule, basis, code, collinear=collinear)
    check_against(converge(molecule_name, basis_name, name), expected)


@pytest.mark.reference
@pytest.mark.parametrize(
    ("molecule_name", "basis_name", "name", "code", "collinear"), STORED_CASES
)
def test_generalized_kohn_sham_matches_stored_data(
    molecule_name: str, basis_name: str, name: str, code: str, collinear: str
) -> None:
    """Regression against values recorded from PySCF (runs without PySCF)."""
    data_file = (
        resources.files("tests.reference.data")
        / f"gks_{molecule_name}_{basis_name}.json"
    )
    stored: dict[str, Any] = json.loads(data_file.read_text(encoding="utf-8"))
    assert stored["provenance"]["program"] == "pyscf"
    assert stored["provenance"]["method"] == "gks"
    assert stored["provenance"]["functional"] == code
    assert stored["provenance"]["collinear"] == collinear
    check_against(converge(molecule_name, basis_name, name), stored)

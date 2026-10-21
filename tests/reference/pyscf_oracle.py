"""Build PySCF molecules that mirror a Hafnium basis set exactly.

PySCF is the oracle for integrals (it ships libcint). Coordinates are passed
in Bohr (ADR-0006) and the basis is passed shell by shell from the Hafnium
data, so the comparison does not depend on PySCF's own basis library.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.molecule import Molecule

pyscf = pytest.importorskip("pyscf")
gto: Any = pyscf.gto
df: Any = pytest.importorskip("pyscf.df")


def pyscf_molecule(molecule: Molecule, basis: BasisSet) -> Any:
    """A ``pyscf.gto.Mole`` with the same atoms, order and shells."""
    atoms: list[list[Any]] = []
    basis_input: dict[str, list[Any]] = {}
    for index, atom in enumerate(molecule.atoms):
        label = f"{atom.symbol}{index}"
        atoms.append([label, list(atom.position)])
        shells = [
            shell
            for shell, centre in zip(basis.shells, basis.shell_atoms, strict=True)
            if centre == index
        ]
        basis_input[label] = [
            [
                shell.angular_momentum,
                *[
                    [exponent, *[row[k] for row in shell.coefficients]]
                    for k, exponent in enumerate(shell.exponents)
                ],
            ]
            for shell in shells
        ]
    return gto.M(
        atom=atoms,
        basis=basis_input,
        unit="Bohr",
        charge=molecule.charge,
        spin=molecule.spin,
        cart=False,
        verbose=0,
    )


def one_electron_reference(
    molecule: Molecule, basis: BasisSet
) -> dict[str, np.ndarray]:
    """Overlap, kinetic and nuclear attraction matrices from libcint."""
    mol = pyscf_molecule(molecule, basis)
    return {
        "overlap": np.asarray(mol.intor("int1e_ovlp")),
        "kinetic": np.asarray(mol.intor("int1e_kin")),
        "nuclear_attraction": np.asarray(mol.intor("int1e_nuc")),
    }


def attenuated_attraction_reference(
    molecule: Molecule, basis: BasisSet, omega: float
) -> np.ndarray:
    """``⟨μ| Σ_C −Z_C erf(ω r_C)/r_C |ν⟩`` from three-centre integrals.

    ``erf(ω r)/r`` is the potential of the unit-charge Gaussian
    ``(ω²/π)^{3/2} exp(−ω² r²)``, which is ``(ω²/2π)^{3/4}`` times PySCF's
    unit-normalized s function of exponent ``ω²``.
    """
    mol = pyscf_molecule(molecule, basis)
    charges = [[f"ghost-{atom.symbol}", list(atom.position)] for atom in molecule.atoms]
    aux_basis = {
        f"ghost-{symbol}": [[0, [omega**2, 1.0]]] for symbol in set(molecule.symbols)
    }
    aux = gto.M(atom=charges, basis=aux_basis, unit="Bohr", verbose=0)
    three_centre = np.asarray(df.incore.aux_e2(mol, aux, intor="int3c2e", aosym="s1"))
    scale = (omega**2 / (2.0 * np.pi)) ** 0.75
    total = np.zeros(three_centre.shape[:2])
    for index, charge in enumerate(molecule.nuclear_charges):
        total -= charge * three_centre[:, :, index]
    return total * scale


def density_fitting_reference(
    molecule: Molecule, basis: BasisSet, auxiliary: BasisSet, omega: float | None = None
) -> dict[str, np.ndarray]:
    """Metric ``(P|Q)`` and three-centre ``(P|uv)`` integrals from libcint.

    With ``omega`` the range-separated kernel is set on both molecules;
    libcint honours it for two-electron integrals.
    """
    mol = pyscf_molecule(molecule, basis)
    aux = pyscf_molecule(molecule, auxiliary)
    if omega is None:
        metric = np.asarray(aux.intor("int2c2e"))
        three = np.asarray(df.incore.aux_e2(mol, aux, intor="int3c2e", aosym="s1"))
    else:
        with mol.with_range_coulomb(omega), aux.with_range_coulomb(omega):
            metric = np.asarray(aux.intor("int2c2e"))
            three = np.asarray(df.incore.aux_e2(mol, aux, intor="int3c2e", aosym="s1"))
    return {
        "metric": metric,
        "three_center": np.ascontiguousarray(np.moveaxis(three, 2, 0)),
    }


def fitted_repulsion_reference(
    molecule: Molecule, basis: BasisSet, auxiliary: BasisSet
) -> np.ndarray:
    """Fitted ``(uv|wx)`` from PySCF's Cholesky-decomposed metric."""
    mol = pyscf_molecule(molecule, basis)
    aux = pyscf_molecule(molecule, auxiliary)
    lib: Any = pyscf.lib
    cderi = np.asarray(df.incore.cholesky_eri(mol, auxmol=aux))
    n_ao = mol.nao
    full = np.stack([lib.unpack_tril(row) for row in cderi])
    return np.einsum("Puv,Pwx->uvwx", full, full).reshape(n_ao, n_ao, n_ao, n_ao)


def four_center_reference(
    molecule: Molecule, basis: BasisSet, omega: float | None = None
) -> np.ndarray:
    """``(uv|wx)`` in chemists' notation from libcint, full or long-range."""
    mol = pyscf_molecule(molecule, basis)
    if omega is None:
        return np.asarray(mol.intor("int2e"))
    with mol.with_range_coulomb(omega):
        return np.asarray(mol.intor("int2e"))


def mixed_four_center_reference(
    molecule: Molecule, first: BasisSet, second: BasisSet, pattern: str
) -> np.ndarray:
    """Four-centre integrals whose indices run over two basis sets.

    ``pattern`` names the basis of each index: ``"abab"`` is ``(a b|a b)``
    with ``a`` in ``first`` and ``b`` in ``second``.
    """
    mol_first = pyscf_molecule(molecule, first)
    mol_second = pyscf_molecule(molecule, second)
    joined = gto.conc_mol(mol_first, mol_second)
    ranges = {"a": (0, mol_first.nbas), "b": (mol_first.nbas, joined.nbas)}
    slices = tuple(bound for letter in pattern for bound in ranges[letter])
    return np.asarray(joined.intor("int2e", shls_slice=slices))

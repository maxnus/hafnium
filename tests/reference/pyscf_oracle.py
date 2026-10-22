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
from hafnium.molecule import AtomicSystem, Cell, Molecule
from hafnium.symmetry import KPoints

pyscf = pytest.importorskip("pyscf")
gto: Any = pyscf.gto
df: Any = pytest.importorskip("pyscf.df")


def atom_label(system: AtomicSystem, index: int) -> str:
    """The per-atom label PySCF keys basis input by, ``"O0"``, ``"H1"``, ..."""
    return f"{system.atoms[index].symbol}{index}"


def basis_input(system: AtomicSystem, basis: BasisSet) -> dict[str, list[Any]]:
    """``basis`` in PySCF's internal format, keyed by :func:`atom_label`."""
    result: dict[str, list[Any]] = {}
    for index in range(system.n_atoms):
        shells = [
            shell
            for shell, centre in zip(basis.shells, basis.shell_atoms, strict=True)
            if centre == index
        ]
        result[atom_label(system, index)] = [
            [
                shell.angular_momentum,
                *[
                    [exponent, *[row[k] for row in shell.coefficients]]
                    for k, exponent in enumerate(shell.exponents)
                ],
            ]
            for shell in shells
        ]
    return result


def pyscf_molecule(molecule: Molecule, basis: BasisSet) -> Any:
    """A ``pyscf.gto.Mole`` with the same atoms, order and shells."""
    atoms = [
        [atom_label(molecule, index), list(atom.position)]
        for index, atom in enumerate(molecule.atoms)
    ]
    return gto.M(
        atom=atoms,
        basis=basis_input(molecule, basis),
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


def fourier_transform_reference(
    molecule: Molecule, basis: BasisSet, g_vectors: np.ndarray
) -> dict[str, np.ndarray]:
    """Transforms ``[G, ao]`` of functions and ``[G, ao, ao]`` of pairs.

    PySCF's ``gto.ft_ao`` evaluates them analytically with the kernel
    ``exp(−iG·r)`` of CONVENTIONS.md §3.
    """
    ft_ao: Any = pytest.importorskip("pyscf.gto.ft_ao")
    mol = pyscf_molecule(molecule, basis)
    g = np.ascontiguousarray(g_vectors, dtype=float)
    return {
        "single": np.asarray(ft_ao.ft_ao(mol, g)),
        "pair": np.asarray(ft_ao.ft_aopair(mol, g)),
    }


def mixed_fourier_transform_reference(
    molecule: Molecule, first: BasisSet, second: BasisSet, g_vectors: np.ndarray
) -> np.ndarray:
    """Pair transforms ``[G, ao_first, ao_second]`` between two basis sets."""
    ft_ao: Any = pytest.importorskip("pyscf.gto.ft_ao")
    mol_first = pyscf_molecule(molecule, first)
    joined = gto.conc_mol(mol_first, pyscf_molecule(molecule, second))
    slices = (0, mol_first.nbas, mol_first.nbas, joined.nbas)
    g = np.ascontiguousarray(g_vectors, dtype=float)
    return np.asarray(ft_ao.ft_aopair(joined, g, shls_slice=slices))


def scf_reference(
    molecule: Molecule,
    basis: BasisSet,
    kind: str,
    *,
    auxiliary: BasisSet | None = None,
    guess: np.ndarray | None = None,
) -> dict[str, Any]:
    """Converged ``kind`` (``"rhf"``, ``"uhf"`` or ``"ghf"``) energy and orbitals.

    With ``auxiliary`` the two-electron terms are density-fitted in that
    basis, passed as basis input because a ``Mole`` assigned to
    ``with_df.auxmol`` is replaced by the default auxiliary basis.
    ``guess`` is a starting density in PySCF's layout (spin blocks stacked
    for UHF, the spin-orbital matrix with alpha first for GHF).
    """
    scf: Any = pytest.importorskip("pyscf.scf")
    mol = pyscf_molecule(molecule, basis)
    mean_field = {"rhf": scf.RHF, "uhf": scf.UHF, "ghf": scf.GHF}[kind](mol)
    if auxiliary is not None:
        mean_field = mean_field.density_fit(auxbasis=basis_input(molecule, auxiliary))
    mean_field.conv_tol = 1e-12
    mean_field.conv_tol_grad = 1e-8
    mean_field.max_cycle = 200
    energy = float(mean_field.kernel(dm0=guess))
    assert mean_field.converged
    spin_square = 0.0 if kind == "rhf" else float(mean_field.spin_square()[0])
    return {
        "energy": energy,
        "mo_energies": np.asarray(mean_field.mo_energy),
        "mo_occupations": np.asarray(mean_field.mo_occ),
        "spin_square": spin_square,
    }


def padded_lattice_vectors(cell: Cell, vacuum: float) -> np.ndarray:
    """Three lattice vectors ``[3, 3]`` for PySCF, padded orthogonally.

    A chain or a slab gets vectors of length ``vacuum`` Bohr orthogonal to
    its periodic directions, so that PySCF's reciprocal vectors of the
    periodic directions coincide with the cell's and no image along the
    padding overlaps the cell.
    """
    vectors = [np.asarray(vector) for vector in cell.lattice.vectors]
    if len(vectors) == 1:
        (a,) = vectors
        trial = np.eye(3)[np.argmin(np.abs(a))]
        first = np.cross(a, trial)
        vectors.append(vacuum * first / np.linalg.norm(first))
    if len(vectors) == 2:
        normal = np.cross(vectors[0], vectors[1])
        vectors.append(vacuum * normal / np.linalg.norm(normal))
    return np.asarray(vectors)


def pyscf_cell(cell: Cell, basis: BasisSet, *, vacuum: float = 80.0) -> Any:
    """A ``pyscf.pbc.gto.Cell`` with the same atoms, order, shells and lattice.

    PySCF's lattice sums are cut off at a precision of 1e-14, below the
    integral tolerance, so that the comparison is limited by the kernels and
    not by the images included.
    """
    pbc_gto: Any = pytest.importorskip("pyscf.pbc.gto")
    atoms = [
        [atom_label(cell, index), list(atom.position)]
        for index, atom in enumerate(cell.atoms)
    ]
    reference = pbc_gto.Cell(
        atom=atoms,
        a=padded_lattice_vectors(cell, vacuum),
        basis=basis_input(cell, basis),
        unit="Bohr",
        dimension=cell.dimension,
        low_dim_ft_type="inf_vacuum",
        precision=1e-14,
        verbose=0,
    )
    reference.build()
    return reference


def bloch_one_electron_reference(
    cell: Cell, basis: BasisSet, kpoints: KPoints
) -> dict[str, np.ndarray]:
    """Overlap and kinetic matrices ``[kpt, ao, ao]`` at ``kpoints`` from PySCF.

    PySCF forms the same Bloch sums ``Σ_T exp(i k·T) M_T`` over libcint's
    integrals, with the k-points passed in Cartesian coordinates.
    """
    reference = pyscf_cell(cell, basis)
    kpts = np.asarray(kpoints.cartesian(cell.lattice))
    return {
        "overlap": np.asarray(reference.pbc_intor("int1e_ovlp", kpts=kpts)),
        "kinetic": np.asarray(reference.pbc_intor("int1e_kin", kpts=kpts)),
    }


def ewald_reference(cell: Cell, *, vacuum: float = 80.0) -> float:
    """Nuclear repulsion of ``cell`` from PySCF's Ewald sum, in Hartree.

    PySCF needs three lattice vectors; a slab gets a vacuum vector of
    ``vacuum`` Bohr along ``z`` and must lie in the ``xy`` plane, since its
    two-dimensional Ewald sum assumes that orientation. The analytic sum
    does not depend on the vacuum. Chains are not supported by PySCF.
    """
    pbc_gto: Any = pytest.importorskip("pyscf.pbc.gto")
    vectors = [list(vector) for vector in cell.lattice.vectors]
    if cell.dimension == 2:
        assert all(vector[2] == 0.0 for vector in vectors)
        vectors.append([0.0, 0.0, vacuum])
    elif cell.dimension != 3:
        raise ValueError("PySCF has Ewald sums in two and three dimensions only")
    atoms = [[atom.symbol, list(atom.position)] for atom in cell.atoms]
    reference = pbc_gto.Cell(
        atom=atoms,
        a=np.asarray(vectors),
        unit="Bohr",
        basis="sto-3g",
        dimension=cell.dimension,
        precision=1e-12,
        verbose=0,
    )
    reference.build()
    return float(reference.energy_nuc())

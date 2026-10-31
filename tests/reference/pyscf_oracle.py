"""Build PySCF molecules that mirror a Hafnium basis set exactly.

PySCF is the oracle for integrals (it ships libcint). Coordinates are passed
in Bohr (ADR-0006) and the basis is passed shell by shell from the Hafnium
data, so the comparison does not depend on PySCF's own basis library.
"""

from __future__ import annotations

import contextlib
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.molecule import AtomicSystem, Cell, Molecule
from hafnium.symmetry import KPoints
from hafnium.xc import NoncollinearDensity

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


def mp2_reference(
    molecule: Molecule, basis: BasisSet, auxiliary: BasisSet, kind: str
) -> dict[str, float]:
    """Density-fitted ``kind`` Hartree–Fock and DF-MP2 energies in one auxiliary basis.

    The reference is converged to a gradient of 1e-10, since the
    second-order energy is first order in the orbital error.
    """
    scf: Any = pytest.importorskip("pyscf.scf")
    dfmp2: Any = pytest.importorskip("pyscf.mp.dfmp2")
    dfump2: Any = pytest.importorskip("pyscf.mp.dfump2")
    dfgmp2: Any = pytest.importorskip("pyscf.mp.dfgmp2")
    mol = pyscf_molecule(molecule, basis)
    mean_field = {"rhf": scf.RHF, "uhf": scf.UHF, "ghf": scf.GHF}[kind](mol)
    mean_field = mean_field.density_fit(auxbasis=basis_input(molecule, auxiliary))
    mean_field.conv_tol = 1e-13
    mean_field.conv_tol_grad = 1e-10
    mean_field.max_cycle = 300
    reference_energy = float(mean_field.kernel())
    assert mean_field.converged
    perturbation = {"rhf": dfmp2.DFMP2, "uhf": dfump2.DFUMP2, "ghf": dfgmp2.DFGMP2}[
        kind
    ](mean_field)
    correlation_energy = float(perturbation.kernel()[0])
    return {
        "reference_energy": reference_energy,
        "correlation_energy": correlation_energy,
    }


def kmp2_reference(
    cell: Cell, basis: BasisSet, auxiliary: BasisSet, kpoints: KPoints, kind: str
) -> dict[str, float]:
    """Density-fitted ``kind`` (``"krhf"``) k-point HF and MP2 energies per cell.

    PySCF's ``KMP2`` with Gaussian density fitting in the same auxiliary
    basis; the reference is converged to a gradient of 1e-10, since the
    second-order energy is first order in the orbital error. PySCF's
    ``KUMP2`` raises ``NotImplementedError`` and no generalized k-point
    MP2 exists, so only the restricted class is served.
    """
    pbc_scf: Any = pytest.importorskip("pyscf.pbc.scf")
    kmp2: Any = pytest.importorskip("pyscf.pbc.mp.kmp2")
    if kind != "krhf":
        raise ValueError("PySCF implements k-point MP2 for KRHF only")
    reference = pyscf_cell(cell, basis)
    kpts = np.asarray(kpoints.cartesian(cell.lattice))
    mean_field = pbc_scf.KRHF(reference, kpts=kpts).density_fit(
        auxbasis=basis_input(cell, auxiliary)
    )
    mean_field.conv_tol = 1e-13
    mean_field.conv_tol_grad = 1e-10
    mean_field.max_cycle = 300
    reference_energy = float(mean_field.kernel())
    assert mean_field.converged
    correlation_energy = float(kmp2.KMP2(mean_field).kernel()[0])
    return {
        "reference_energy": reference_energy,
        "correlation_energy": correlation_energy,
    }


def ccsd_reference(
    molecule: Molecule, basis: BasisSet, auxiliary: BasisSet, kind: str
) -> dict[str, float]:
    """Density-fitted ``kind`` Hartree–Fock, DF-CCSD and (T) energies in one basis.

    The reference is converged to a gradient of 1e-10 and the amplitudes
    to a norm of 1e-10. PySCF has DF-CCSD for RHF (``cc.dfccsd``) and UHF
    (``cc.dfuccsd``) references; a GHF reference runs ``cc.gccsd`` on the
    four-centre integrals assembled from the same fitted tensor
    (``with_df.get_eri``), which is the same approximation. The (T)
    correction is ``ccsd_t`` of the same object.
    """
    scf: Any = pytest.importorskip("pyscf.scf")
    dfccsd: Any = pytest.importorskip("pyscf.cc.dfccsd")
    dfuccsd: Any = pytest.importorskip("pyscf.cc.dfuccsd")
    gccsd: Any = pytest.importorskip("pyscf.cc.gccsd")
    mol = pyscf_molecule(molecule, basis)
    mean_field = {"rhf": scf.RHF, "uhf": scf.UHF, "ghf": scf.GHF}[kind](mol)
    mean_field = mean_field.density_fit(auxbasis=basis_input(molecule, auxiliary))
    mean_field.conv_tol = 1e-13
    mean_field.conv_tol_grad = 1e-10
    mean_field.max_cycle = 300
    reference_energy = float(mean_field.kernel())
    assert mean_field.converged
    if kind == "rhf":
        coupled_cluster = dfccsd.RCCSD(mean_field)
    elif kind == "uhf":
        coupled_cluster = dfuccsd.UCCSD(mean_field)
    else:
        mean_field._eri = mean_field.with_df.get_eri()
        coupled_cluster = gccsd.GCCSD(mean_field)
    coupled_cluster.conv_tol = 1e-12
    coupled_cluster.conv_tol_normt = 1e-10
    coupled_cluster.max_cycle = 200
    correlation_energy = float(coupled_cluster.kernel()[0])
    assert coupled_cluster.converged
    return {
        "reference_energy": reference_energy,
        "correlation_energy": correlation_energy,
        "triples_correction": float(coupled_cluster.ccsd_t()),
    }


def kccsd_reference(
    cell: Cell,
    basis: BasisSet,
    auxiliary: BasisSet,
    kpoints: KPoints,
    kind: str,
    *,
    exchange_correction: bool = True,
) -> dict[str, float]:
    """Density-fitted ``kind`` k-point HF, CCSD and (T) energies per cell.

    PySCF's ``KRCCSD``, ``KUCCSD`` or ``KGCCSD`` (``"krhf"``, ``"kuhf"``,
    ``"kghf"``) with Gaussian density fitting in the same auxiliary basis,
    the reference converged to a gradient of 1e-10 and the amplitudes to
    a norm of 1e-10, and the (T) correction of ``ccsd_t`` for the two
    classes that have it (``KUCCSD`` has none, so its result carries no
    ``triples_correction``). By default PySCF's k-point CCSD classes rebuild the
    Fock matrix without the Ewald exchange correction of the mean field
    (``exxdiv=None``) and only shift the orbital energies of the
    denominators by the Madelung constant for convergence, so their
    energy is that of a Hamiltonian whose occupied orbital energies differ
    from the reference's, 5 mHartree per cell for the helium test cell,
    while ``KMP2`` keeps the corrected energies. ``keep_exxdiv`` makes
    ``KRCCSD`` use the mean field's Fock matrix as Hafnium does; ``KUCCSD``
    and ``KGCCSD`` have no such option, so they are compared without the
    correction on either side (``exchange_correction`` false, PySCF's
    ``exxdiv=None`` in the mean field too), where the three classes agree.
    ``KGCCSD`` also shifts the occupied orbital energies of its
    denominators by the Madelung constant whatever the mean field's
    setting, which its (T) inherits; the (T) here is evaluated with the
    diagonal of the Fock matrix of the equations instead.
    """
    pbc_scf: Any = pytest.importorskip("pyscf.pbc.scf")
    pbc_cc: Any = pytest.importorskip("pyscf.pbc.cc")
    if kind != "krhf" and exchange_correction:
        raise ValueError(
            "PySCF keeps the exchange correction in the Fock matrix of KRCCSD only; "
            "compare the other classes without it"
        )
    reference = pyscf_cell(cell, basis)
    kpts = np.asarray(kpoints.cartesian(cell.lattice))
    method = {"krhf": pbc_scf.KRHF, "kuhf": pbc_scf.KUHF, "kghf": pbc_scf.KGHF}[kind]
    mean_field = method(
        reference, kpts=kpts, exxdiv="ewald" if exchange_correction else None
    ).density_fit(auxbasis=basis_input(cell, auxiliary))
    mean_field.conv_tol = 1e-13
    mean_field.conv_tol_grad = 1e-10
    mean_field.max_cycle = 300
    reference_energy = float(mean_field.kernel())
    assert mean_field.converged
    coupled_cluster = {
        "krhf": pbc_cc.KRCCSD,
        "kuhf": pbc_cc.KUCCSD,
        "kghf": pbc_cc.KGCCSD,
    }[kind](mean_field)
    if kind == "krhf":
        coupled_cluster.keep_exxdiv = True
    coupled_cluster.conv_tol = 1e-12
    coupled_cluster.conv_tol_normt = 1e-10
    coupled_cluster.max_cycle = 200
    correlation_energy = float(coupled_cluster.kernel()[0])
    assert coupled_cluster.converged
    result = {
        "reference_energy": reference_energy,
        "correlation_energy": correlation_energy,
    }
    if kind == "krhf":
        # The kernel writes a scratch file into the working directory.
        with tempfile.TemporaryDirectory() as scratch, contextlib.chdir(scratch):
            result["triples_correction"] = float(coupled_cluster.ccsd_t())
    elif kind == "kghf":
        # KGCCSD shifts the occupied orbital energies of its denominators
        # by the Madelung constant whatever the mean field's ``exxdiv``, a
        # convergence device for the CCSD iteration that leaves its fixed
        # point alone but changes (T), whose denominators enter the energy;
        # the denominators of the Fock operator the equations use are its
        # diagonal, as KRCCSD keeps them with ``keep_exxdiv``.
        eris = coupled_cluster.ao2mo()
        eris.mo_energy = [eris.fock[k].diagonal().real for k in range(len(kpts))]
        result["triples_correction"] = float(coupled_cluster.ccsd_t(eris=eris))
    return result


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
    not by the images included. A slab takes PySCF's analytic
    two-dimensional Coulomb kernel (``low_dim_ft_type=None``, the truncated
    kernel of Ismail-Beigi), which converges exponentially with the vacuum
    and needs the slab in the ``xy`` plane for the Coulomb terms; a chain
    takes ``"inf_vacuum"``, PySCF's only option for dimension one, which
    integrates the open directions of the three-dimensional kernel on
    Gauss–Chebyshev nodes of a fixed count but drops the ``G = 0`` term
    instead of regularizing it, so its energies are not exact (with it a
    slab's ``energy_nuc`` comes out at millions of Hartree).
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
        low_dim_ft_type="inf_vacuum" if cell.dimension == 1 else None,
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


def periodic_nuclear_attraction_reference(
    cell: Cell, basis: BasisSet, kpoints: KPoints
) -> np.ndarray:
    """Nuclear attraction ``[kpt, ao, ao]`` from PySCF's density-fitted potential.

    ``GDF.get_nuc`` splits every nucleus into a compensating Gaussian
    charge, summed with the point charge in real space, and the smooth
    remainder in reciprocal space, and keeps the finite part
    ``π Z_tot / (ω² Ω)`` times the overlap at ``G = 0``: the regularization
    of ADR-0012, so the matrices compare directly in three dimensions.
    """
    pbc_df: Any = pytest.importorskip("pyscf.pbc.df")
    reference = pyscf_cell(cell, basis)
    kpts = np.asarray(kpoints.cartesian(cell.lattice))
    return np.asarray(pbc_df.GDF(reference, kpts=kpts).get_nuc(kpts))


def madelung_reference(cell: Cell, basis: BasisSet, kpoints: KPoints) -> float:
    """The probe-charge Madelung constant of the mesh from PySCF's Ewald sum."""
    tools: Any = pytest.importorskip("pyscf.pbc.tools")
    reference = pyscf_cell(cell, basis)
    kpts = np.asarray(kpoints.cartesian(cell.lattice))
    return float(tools.pbc.madelung(reference, kpts))


def periodic_fitted_eri_reference(
    cell: Cell,
    basis: BasisSet,
    auxiliary: BasisSet,
    kpoints: KPoints,
    quadruples: list[tuple[int, int, int, int]],
) -> dict[str, np.ndarray]:
    """Fitted ``(u k1 v k2 | w k3 x k4)`` from PySCF's Gaussian density fitting.

    PySCF's ``GDF`` with the same auxiliary basis fits the pair densities
    with the Coulomb metric and the range-separated kernel of its
    ``rsdf_builder``, the ``G = 0`` term of the metric carrying the
    ``−π/ω²`` finite part of ADR-0012, so its ``get_eri`` compares directly
    with :meth:`hafnium.integrals.PeriodicDensityFitting.eri`: chemists'
    notation with the first function of each pair conjugated.
    """
    pbc_df: Any = pytest.importorskip("pyscf.pbc.df")
    reference = pyscf_cell(cell, basis)
    kpts = np.asarray(kpoints.cartesian(cell.lattice))
    fitting = pbc_df.GDF(reference, kpts=kpts)
    fitting.auxbasis = basis_input(cell, auxiliary)
    fitting.build()
    n = reference.nao
    return {
        "".join(str(k) for k in quadruple): np.asarray(
            fitting.get_eri([kpts[k] for k in quadruple], compact=False)
        ).reshape(n, n, n, n)
        for quadruple in quadruples
    }


def kscf_reference(
    cell: Cell,
    basis: BasisSet,
    auxiliary: BasisSet,
    kpoints: KPoints,
    kind: str,
) -> dict[str, Any]:
    """Converged k-point ``kind`` (``"krhf"``, ``"kuhf"``, ``"kghf"``) state from PySCF.

    The two-electron terms are density-fitted in ``auxiliary`` with the
    Ewald exchange correction (PySCF's default ``exxdiv``), the same
    construction as ADR-0017 and ADR-0019, so the energy per cell and the
    orbital energies at every k-point compare directly; for a slab PySCF's
    two-dimensional kernels differ from those of ADR-0021 by constants that
    cancel in the energy of a neutral cell.

    PySCF's k-point classes take ``cell.spin`` as ``n_alpha − n_beta`` over
    the whole mesh, the spin per cell times the number of k-points, and
    the ``spin_square`` of ``KUHF`` is likewise that of the supercell
    (``spin_square_supercell`` here). ``KGHF`` has no ``spin_square`` and
    converges from its default guess to a higher non-magnetic state, so it
    starts from the converged ``KUHF`` density and reports that run's
    ``spin_square``.
    """
    pbc_scf: Any = pytest.importorskip("pyscf.pbc.scf")
    reference = pyscf_cell(cell, basis)
    reference.spin = cell.spin * kpoints.n_kpt
    kpts = np.asarray(kpoints.cartesian(cell.lattice))
    auxbasis = basis_input(cell, auxiliary)

    def solve(method: Any, guess: np.ndarray | None = None) -> tuple[Any, float]:
        mean_field = method(reference, kpts=kpts).density_fit(auxbasis=auxbasis)
        mean_field.conv_tol = 1e-12
        mean_field.conv_tol_grad = 1e-8
        mean_field.max_cycle = 200
        energy = float(mean_field.kernel(dm0=guess))
        assert mean_field.converged
        return mean_field, energy

    if kind == "krhf":
        mean_field, energy = solve(pbc_scf.KRHF)
        spin_square = 0.0
    elif kind == "kuhf":
        mean_field, energy = solve(pbc_scf.KUHF)
        spin_square = float(mean_field.spin_square()[0])
    else:
        unrestricted, _ = solve(pbc_scf.KUHF)
        spin_square = float(unrestricted.spin_square()[0])
        alpha, beta = (np.asarray(part) for part in unrestricted.make_rdm1())
        n = alpha.shape[-1]
        guess = np.zeros((len(kpts), 2 * n, 2 * n), dtype=np.result_type(alpha, beta))
        guess[:, :n, :n] = alpha
        guess[:, n:, n:] = beta
        mean_field, energy = solve(pbc_scf.KGHF, guess)
    return {
        "energy": energy,
        "mo_energies": np.asarray(mean_field.mo_energy),
        "spin_square_supercell": spin_square,
    }


def symmetric_basis_input(molecule: Molecule, basis: BasisSet) -> dict[str, list[Any]]:
    """``basis`` keyed by element, as PySCF's symmetry detection needs it.

    PySCF treats atoms with different basis keys as different elements, so
    the per-atom keys of :func:`basis_input` would hide the symmetry.
    """
    per_atom = basis_input(molecule, basis)
    by_element: dict[str, list[Any]] = {}
    for index, atom in enumerate(molecule.atoms):
        shells = per_atom[atom_label(molecule, index)]
        if by_element.setdefault(atom.symbol, shells) != shells:
            raise ValueError(f"atoms of {atom.symbol} carry different shells")
    return by_element


def plain_atoms(molecule: Molecule) -> list[list[Any]]:
    return [[atom.symbol, list(atom.position)] for atom in molecule.atoms]


def pyscf_symmetric_molecule(
    molecule: Molecule, basis: BasisSet, *, subgroup: str | None = None
) -> Any:
    """A ``pyscf.gto.Mole`` with symmetry detected, in ``molecule``'s own frame."""
    return gto.M(
        atom=plain_atoms(molecule),
        basis=symmetric_basis_input(molecule, basis),
        unit="Bohr",
        charge=molecule.charge,
        spin=molecule.spin,
        symmetry=True,
        symmetry_subgroup=subgroup,
        verbose=0,
    )


def point_group_reference(molecule: Molecule) -> tuple[str, str, np.ndarray]:
    """PySCF's full group, its Abelian subgroup and that subgroup's axes."""
    symm: Any = pytest.importorskip("pyscf.symm")
    top, _, axes = symm.detect_symm(plain_atoms(molecule))
    sub, sub_axes = symm.subgroup(top, axes)
    return str(top), str(sub), np.asarray(sub_axes)


def orbital_label_reference(
    molecule: Molecule, basis: BasisSet, kind: str, *, subgroup: str | None = None
) -> dict[str, Any]:
    """PySCF's symmetry-adapted RHF or UHF: energy, irrep dimensions and labels."""
    symm: Any = pytest.importorskip("pyscf.symm")
    scf: Any = pytest.importorskip("pyscf.scf")
    mol = pyscf_symmetric_molecule(molecule, basis, subgroup=subgroup)
    mf = scf.RHF(mol) if kind == "rhf" else scf.UHF(mol)
    mf.conv_tol = 1e-12
    energy = float(mf.kernel())
    coefficients = [mf.mo_coeff] if kind == "rhf" else list(mf.mo_coeff)
    energies = [mf.mo_energy] if kind == "rhf" else list(mf.mo_energy)
    spins = [
        (
            [
                str(label)
                for label in symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, c)
            ],
            np.asarray(e),
        )
        for c, e in zip(coefficients, energies, strict=True)
    ]
    return {
        "group": str(mol.groupname),
        "energy": energy,
        "dimensions": {
            str(name): int(block.shape[1])
            for name, block in zip(mol.irrep_name, mol.symm_orb, strict=True)
        },
        "spins": spins,
    }


def symmetric_scf_reference(
    molecule: Molecule,
    basis: BasisSet,
    kind: str,
    *,
    subgroup: str | None = None,
    irrep_nelec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """PySCF's symmetry-adapted ``kind`` SCF in ``molecule``'s own frame.

    Returns the energy, the electrons per irrep summed over spins, and per
    spin block the orbital energies in ascending order with the irrep label
    of every orbital; a generalized state has one spin block.
    """
    scf: Any = pytest.importorskip("pyscf.scf")
    mol = pyscf_symmetric_molecule(molecule, basis, subgroup=subgroup)
    mf = {"rhf": scf.RHF, "uhf": scf.UHF, "ghf": scf.GHF}[kind](mol)
    if irrep_nelec is not None:
        mf.irrep_nelec = irrep_nelec
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    energy = float(mf.kernel())
    assert mf.converged
    names = {
        int(i): str(name) for i, name in zip(mol.irrep_id, mol.irrep_name, strict=True)
    }
    # Per irrep PySCF reports a total (RHF, GHF) or ``(n_alpha, n_beta)`` (UHF).
    electrons = {
        str(label): float(np.sum(np.asarray(count, dtype=float)))
        for label, count in mf.get_irrep_nelec().items()
    }
    orbsym = mf.get_orbsym(mf.mo_coeff)
    if kind == "uhf":
        spins = [
            (np.asarray(e), [names[int(i)] for i in o])
            for e, o in zip(mf.mo_energy, orbsym, strict=True)
        ]
    else:
        spins = [(np.asarray(mf.mo_energy), [names[int(i)] for i in orbsym])]
    return {
        "group": str(mol.groupname),
        "energy": energy,
        "electrons": electrons,
        "spins": spins,
        "spin_square": 0.0 if kind == "rhf" else float(mf.spin_square()[0]),
    }


def symmetric_kscf_reference(
    cell: Cell, basis: BasisSet, auxiliary: BasisSet, mesh: tuple[int, ...]
) -> dict[str, Any]:
    """Converged ``KRHF`` over the irreducible wedge of ``mesh`` from PySCF.

    PySCF reduces the Γ-centred mesh by the symmorphic point operations
    its own search finds (``cell.symmorphic``, which it also forces for
    chains and slabs) and by time reversal, and runs the SCF at the
    points of the wedge with their weights (``kpts_ibz``, ``weights_ibz``).
    A slab needs a three-dimensional mesh with one point along the padded
    direction. The orbital energies are those of the wedge points, in the
    order of ``kpts_scaled_ibz``, whose periodic coordinates are returned.
    """
    pbc_scf: Any = pytest.importorskip("pyscf.pbc.scf")
    space_group: Any = pytest.importorskip("pyscf.pbc.symm.space_group")
    reference = pyscf_cell(cell, basis)
    group = space_group.SpaceGroup(reference)
    group.verbose = 0
    group.build(dump_info=False)
    reference.space_group_symmetry = True
    reference.symmorphic = True
    reference.build()
    padded_mesh = tuple(mesh) + (1,) * (3 - len(mesh))
    kpts = reference.make_kpts(
        padded_mesh, space_group_symmetry=True, time_reversal_symmetry=True
    )
    mean_field = pbc_scf.KRHF(reference, kpts).density_fit(
        auxbasis=basis_input(cell, auxiliary)
    )
    mean_field.conv_tol = 1e-12
    mean_field.conv_tol_grad = 1e-8
    mean_field.max_cycle = 200
    energy = float(mean_field.kernel())
    assert mean_field.converged
    return {
        "energy": energy,
        "n_operations": int(group.nop),
        "point_group": str(group.groupname["point_group_symbol"]),
        "n_wedge": int(kpts.nkpts_ibz),
        "weights": np.asarray(kpts.weights_ibz),
        "fractional": np.asarray(kpts.kpts_scaled_ibz)[:, : cell.dimension],
        "mo_energies": np.asarray(mean_field.mo_energy),
    }


def dft_reference(
    molecule: Molecule,
    basis: BasisSet,
    functional: str,
    kind: str,
    *,
    auxiliary: BasisSet | None = None,
    grid: tuple[int, tuple[int, int] | None, bool] = (3, None, True),
) -> dict[str, Any]:
    """Converged ``kind`` (``"rks"`` or ``"uks"``) state with PySCF's ``functional``.

    ``grid`` is ``(level, atom_grid, prune)`` in PySCF's terms: the size
    level, an optional ``(radial, angular)`` size for every atom, and whether
    the NWChem pruning is applied; the other grid settings are PySCF's
    defaults, which :mod:`hafnium.grid` follows. The VV10 nonlocal
    correlation of the ωB97 functionals is switched off (``nlc = 0``), since
    Hafnium does not evaluate it. With ``auxiliary`` the Coulomb and exact
    exchange terms are density-fitted in that basis.
    """
    dft: Any = pytest.importorskip("pyscf.dft")
    mol = pyscf_molecule(molecule, basis)
    mean_field = {"rks": dft.RKS, "uks": dft.UKS}[kind](mol)
    mean_field.xc = functional
    mean_field.nlc = 0
    level, atom_grid, prune = grid
    mean_field.grids.level = level
    if atom_grid is not None:
        mean_field.grids.atom_grid = atom_grid
    if not prune:
        mean_field.grids.prune = None
    if auxiliary is not None:
        mean_field = mean_field.density_fit(auxbasis=basis_input(molecule, auxiliary))
    mean_field.conv_tol = 1e-12
    mean_field.conv_tol_grad = 1e-8
    mean_field.max_cycle = 200
    energy = float(mean_field.kernel())
    assert mean_field.converged
    spin_square = 0.0 if kind == "rks" else float(mean_field.spin_square()[0])
    return {
        "energy": energy,
        "mo_energies": np.asarray(mean_field.mo_energy),
        "mo_occupations": np.asarray(mean_field.mo_occ),
        "spin_square": spin_square,
        "n_grid_nonzero": int(np.count_nonzero(mean_field.grids.weights)),
    }


def kdft_reference(
    cell: Cell,
    basis: BasisSet,
    auxiliary: BasisSet,
    kpoints: KPoints,
    functional: str,
    kind: str,
    *,
    level: int = 3,
) -> dict[str, Any]:
    """Converged k-point ``kind`` (``"krks"``, ``"kuks"``) state with ``functional``.

    The exchange–correlation term is integrated on PySCF's periodic Becke
    grids at ``level`` (``pbc.dft.gen_grid.BeckeGrids``: the atomic grids
    truncated to the cell and partitioned among the atoms whose grids reach
    into it), whose points and weights are returned as ``grid_points`` and
    ``grid_weights`` so that Hafnium can integrate on the same grid; its
    uniform grids would need thousands of points per direction for an
    all-electron basis. The two-electron terms are density-fitted in
    ``auxiliary`` with the Ewald exchange correction as in
    :func:`kscf_reference`, and ``cell.spin`` is the spin over the mesh.
    """
    pbc_dft: Any = pytest.importorskip("pyscf.pbc.dft")
    reference = pyscf_cell(cell, basis)
    reference.spin = cell.spin * kpoints.n_kpt
    kpts = np.asarray(kpoints.cartesian(cell.lattice))
    method = {"krks": pbc_dft.KRKS, "kuks": pbc_dft.KUKS}[kind]
    mean_field = method(reference, kpts=kpts).density_fit(
        auxbasis=basis_input(cell, auxiliary)
    )
    mean_field.xc = functional
    mean_field.grids = pbc_dft.gen_grid.BeckeGrids(reference)
    mean_field.grids.level = level
    mean_field.conv_tol = 1e-12
    mean_field.conv_tol_grad = 1e-8
    mean_field.max_cycle = 200
    energy = float(mean_field.kernel())
    assert mean_field.converged
    return {
        "energy": energy,
        "mo_energies": np.asarray(mean_field.mo_energy),
        "grid_points": np.asarray(mean_field.grids.coords),
        "grid_weights": np.asarray(mean_field.grids.weights),
    }


def gks_reference(
    molecule: Molecule,
    basis: BasisSet,
    functional: str,
    *,
    collinear: str = "ncol",
    collinear_threshold: float | None = 0.99,
    spin_samples: int = 770,
) -> dict[str, Any]:
    """Converged GKS state with PySCF's ``functional`` and noncollinear scheme.

    ``collinear`` is ``"ncol"`` (the locally collinear LDA) or ``"mcol"``
    (the multi-collinear approach through ``mcfun`` with ``spin_samples``
    directions and the collinear branch from ``collinear_threshold``, or
    without it for ``None``); the grid is PySCF's default at level 3 and the
    VV10 term is off as in :func:`dft_reference`.
    """
    dft: Any = pytest.importorskip("pyscf.dft")
    if collinear == "mcol":
        pytest.importorskip("mcfun")
    mol = pyscf_molecule(molecule, basis)
    mean_field = dft.GKS(mol)
    mean_field.xc = functional
    mean_field.nlc = 0
    mean_field.collinear = collinear
    mean_field._numint.collinear_thrd = collinear_threshold
    mean_field.spin_samples = spin_samples
    mean_field.conv_tol = 1e-12
    mean_field.conv_tol_grad = 1e-8
    mean_field.max_cycle = 200
    energy = float(mean_field.kernel())
    assert mean_field.converged
    return {
        "energy": energy,
        "mo_energies": np.asarray(mean_field.mo_energy),
        "mo_occupations": np.asarray(mean_field.mo_occ),
        "n_grid_nonzero": int(np.count_nonzero(mean_field.grids.weights)),
    }


def gks_potential_reference(
    molecule: Molecule,
    basis: BasisSet,
    functional: str,
    density: np.ndarray,
    *,
    collinear: str = "mcol",
    collinear_threshold: float | None = None,
    spin_samples: int = 770,
) -> dict[str, Any]:
    """PySCF's exchange–correlation energy and potential of a spinor ``density``.

    ``density`` is ``[2n, 2n]`` with the alpha functions first, as PySCF's
    GHF lays it out. The term is integrated on PySCF's grid at level 3,
    whose points and weights are returned without the zero-weight padding
    so that Hafnium can integrate on the same grid; the noncollinear
    scheme is as in :func:`gks_reference`.
    """
    dft: Any = pytest.importorskip("pyscf.dft")
    numint2c: Any = pytest.importorskip("pyscf.dft.numint2c")
    if collinear == "mcol":
        pytest.importorskip("mcfun")
    mol = pyscf_molecule(molecule, basis)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build(sort_grids=False, with_non0tab=False)
    integrator = numint2c.NumInt2C()
    integrator.collinear = collinear
    integrator.collinear_thrd = collinear_threshold
    integrator.spin_samples = spin_samples
    electrons, energy, potential = integrator.nr_vxc(
        mol, grids, functional, np.asarray(density), hermi=1
    )
    kept = grids.weights != 0.0
    return {
        "energy": float(energy),
        "electrons": float(np.real(electrons)),
        "potential": np.asarray(potential),
        "grid_points": np.asarray(grids.coords[kept]),
        "grid_weights": np.asarray(grids.weights[kept]),
    }


def noncollinear_functional_reference(
    functional: str, density: NoncollinearDensity, *, spin_samples: int = 770
) -> tuple[np.ndarray, tuple[np.ndarray | None, ...]]:
    """``mcfun``'s multi-collinear energy density and derivatives through libxc.

    The sphere alone is sampled (no collinear branch). Returns the energy
    density ``[g]`` and ``(v_rho, v_m, v_grad_rho, v_grad_m, v_tau, v_tau_m)``
    in Hafnium's layouts, ``None`` where the rung does not read a variable.
    """
    numint2c: Any = pytest.importorskip("pyscf.dft.numint2c")
    pytest.importorskip("mcfun")
    integrator = numint2c.NumInt2C()
    integrator.collinear = "mcol"
    integrator.collinear_thrd = None
    integrator.spin_samples = spin_samples
    xctype = integrator._xc_type(functional)
    rho = np.asarray(density.rho)
    n = rho.shape[0]
    if xctype == "LDA":
        rho_tm = np.stack([rho, *np.asarray(density.m)])
    else:
        n_var = 5 if xctype == "MGGA" else 4
        total = np.zeros((n_var, n))
        spins = np.zeros((3, n_var, n))
        total[0], total[1:4] = rho, np.asarray(density.grad_rho)
        spins[:, 0], spins[:, 1:4] = np.asarray(density.m), np.asarray(density.grad_m)
        if xctype == "MGGA":
            total[4], spins[:, 4] = np.asarray(density.tau), np.asarray(density.tau_m)
        rho_tm = np.concatenate([total[None], spins])
    evaluate = integrator.mcfun_eval_xc_adapter(functional)
    exc, vxc = evaluate(functional, rho_tm, deriv=1)[:2]
    energy = np.asarray(exc) * rho
    vxc = np.asarray(vxc)
    if xctype == "LDA":
        return energy, (vxc[0, 0], vxc[1:4, 0], None, None, None, None)
    kinetic = xctype == "MGGA"
    return energy, (
        vxc[0, 0],
        vxc[1:4, 0],
        vxc[0, 1:4],
        vxc[1:4, 1:4],
        vxc[0, 4] if kinetic else None,
        vxc[1:4, 4] if kinetic else None,
    )


def grid_reference(
    molecule: Molecule, level: int = 3, *, prune: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """PySCF's grid points ``[g, 3]`` and weights ``[g]`` at ``level``.

    PySCF pads its grid to a multiple of eight points with zero weights,
    which are dropped here; the remaining points keep their order, atom by
    atom and radial shell by radial shell.
    """
    dft: Any = pytest.importorskip("pyscf.dft")
    mol = pyscf_molecule(molecule, BasisSet.for_atoms(molecule, "sto-3g"))
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    if not prune:
        grids.prune = None
    grids.build(sort_grids=False, with_non0tab=False)
    table = dft.gen_grid.gen_atomic_grids(
        mol, radi_method=grids.radi_method, level=level, prune=grids.prune
    )
    n = sum(len(table[mol.atom_symbol(atom)][1]) for atom in range(mol.natm))
    return np.asarray(grids.coords[:n]), np.asarray(grids.weights[:n])


@dataclass(frozen=True)
class FunctionalValues:
    """libxc's energy per particle and first derivatives in Hafnium's layout."""

    energy_per_particle: np.ndarray
    vrho: np.ndarray
    vsigma: np.ndarray | None
    vtau: np.ndarray | None


def functional_reference(
    name: str, rho: np.ndarray, sigma: np.ndarray, tau: np.ndarray
) -> FunctionalValues:
    """libxc's values of ``name`` at spin-polarized points through PySCF.

    ``rho`` is ``[2, g]``, ``sigma`` ``[3, g]`` and ``tau`` ``[2, g]`` in the
    layout of :class:`hafnium.xc.DensityOnGrid`; the gradients PySCF takes
    are reconstructed with ``∇ρ_α`` along ``x`` and ``∇ρ_β`` in the ``xy``
    plane. Returns the energy per particle and ``vrho``, ``vsigma`` and
    ``vtau`` in Hafnium's layout, ``None`` where the rung does not have them.
    """
    libxc: Any = pytest.importorskip("pyscf.dft.libxc")
    n = rho.shape[1]
    grad_a = np.zeros((3, n))
    grad_a[0] = np.sqrt(sigma[0])
    grad_b = np.zeros((3, n))
    grad_b[0] = sigma[1] / grad_a[0]
    grad_b[1] = np.sqrt(np.maximum(sigma[2] - grad_b[0] ** 2, 0.0))
    rows = {"LDA": 1, "GGA": 4, "MGGA": 6}[libxc.xc_type(name)]
    full_a = np.vstack([rho[0], grad_a, np.zeros(n), tau[0]])
    full_b = np.vstack([rho[1], grad_b, np.zeros(n), tau[1]])
    arg = (full_a[:rows], full_b[:rows]) if rows > 1 else (full_a[0], full_b[0])
    exc, vxc = libxc.eval_xc(name, arg, spin=1, deriv=1)[:2]
    vxc = list(vxc)
    return FunctionalValues(
        np.asarray(exc),
        np.asarray(vxc[0]).T,
        np.asarray(vxc[1]).T if rows > 1 else None,
        np.asarray(vxc[3]).T if rows > 4 else None,
    )

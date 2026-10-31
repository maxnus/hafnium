"""Record reference integrals from PySCF (run by hand).

    uv run python -m tests.reference.generate_reference_data

Rewrites ``tests/reference/data/one_electron_<molecule>_<basis>.json``,
``density_fitting_<molecule>_<basis>.json``,
``four_center_<molecule>_<basis>.json``, ``fourier_<molecule>_<basis>.json``,
``scf_<molecule>_<basis>.json``, ``ewald_<cell>_nuclei.json``,
``bloch_<cell>_<basis>.json``, ``nuclear_<cell>_<basis>.json``,
``fitting_<cell>_<basis>.json``, ``kscf_<cell>_<basis>.json``,
``dft_<molecule>_<basis>.json``, ``mp2_<molecule>_<basis>.json``,
``kmp2_<cell>_<basis>.json``, ``ccsd_<molecule>_<basis>.json`` and
``kccsd_<cell>_<basis>_<class>.json`` with the PySCF values and their provenance.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from hafnium.basis import BasisSet
from hafnium.molecule import AtomicSystem
from tests.reciprocal_vectors import G_VECTORS
from tests.reference.pyscf_oracle import (
    bloch_one_electron_reference,
    ccsd_reference,
    density_fitting_reference,
    dft_reference,
    ewald_reference,
    four_center_reference,
    fourier_transform_reference,
    gks_reference,
    kccsd_reference,
    kdft_reference,
    kmp2_reference,
    kscf_reference,
    madelung_reference,
    mp2_reference,
    one_electron_reference,
    periodic_fitted_eri_reference,
    periodic_nuclear_attraction_reference,
    pyscf,
    scf_reference,
)
from tests.reference.test_bloch_reference import CELLS as BLOCH_CELLS
from tests.reference.test_bloch_reference import MESHES as BLOCH_MESHES
from tests.reference.test_bloch_reference import STORED_CASES as BLOCH_CASES
from tests.reference.test_ccsd_reference import STORED_CASES as CCSD_CASES
from tests.reference.test_density_fitting_reference import AUXILIARY
from tests.reference.test_density_fitting_reference import (
    STORED_CASES as DENSITY_FITTING_CASES,
)
from tests.reference.test_dft_reference import STORED_CASES as DFT_CASES
from tests.reference.test_ewald_reference import CELLS
from tests.reference.test_ewald_reference import STORED_CASES as EWALD_CASES
from tests.reference.test_fitting_reference import AUXILIARY as FITTING_AUXILIARY
from tests.reference.test_fitting_reference import CELLS as FITTING_CELLS
from tests.reference.test_fitting_reference import MESHES as FITTING_MESHES
from tests.reference.test_fitting_reference import QUADRUPLES
from tests.reference.test_fitting_reference import STORED_CASES as FITTING_CASES
from tests.reference.test_four_center_reference import (
    STORED_CASES as FOUR_CENTER_CASES,
)
from tests.reference.test_fourier_reference import STORED_CASES as FOURIER_CASES
from tests.reference.test_kccsd_reference import STORED_CASES as KCCSD_CASES
from tests.reference.test_kdft_reference import STORED_CASES as KDFT_CASES
from tests.reference.test_kdft_reference import STORED_LEVEL as KDFT_LEVEL
from tests.reference.test_kmp2_reference import STORED_CASES as KMP2_CASES
from tests.reference.test_kscf_reference import AUXILIARY as KSCF_AUXILIARY
from tests.reference.test_kscf_reference import CELLS as KSCF_CELLS
from tests.reference.test_kscf_reference import MESHES as KSCF_MESHES
from tests.reference.test_kscf_reference import STORED_CASES as KSCF_CASES
from tests.reference.test_mp2_reference import AUXILIARY as MP2_AUXILIARY
from tests.reference.test_mp2_reference import STORED_CASES as MP2_CASES
from tests.reference.test_noncollinear_reference import STORED_CASES as GKS_CASES
from tests.reference.test_nuclear_reference import CELLS as NUCLEAR_CELLS
from tests.reference.test_nuclear_reference import MESHES as NUCLEAR_MESHES
from tests.reference.test_nuclear_reference import STORED_CASES as NUCLEAR_CASES
from tests.reference.test_one_electron_reference import (
    LIBRARY,
    MOLECULES,
    STORED_CASES,
)
from tests.reference.test_scf_reference import MOLECULES as SCF_MOLECULES
from tests.reference.test_scf_reference import STORED_CASES as SCF_CASES

DIRECTORY = Path(__file__).parent / "data"


def record(
    kind: str,
    system: AtomicSystem,
    name: str,
    basis_name: str,
    settings: str,
    matrices: dict[str, np.ndarray],
    **extra: Any,
) -> None:
    """Write ``<kind>_<name>_<basis>.json`` with provenance."""
    provenance = {
        "program": "pyscf",
        "version": pyscf.__version__,
        "settings": settings,
        "date": datetime.date.today().isoformat(),
        "molecule": name,
        "basis": basis_name,
        **extra,
        "positions_bohr": [list(p) for p in system.positions],
    }
    data = {"provenance": provenance}
    data.update({key: matrix.tolist() for key, matrix in matrices.items()})
    path = DIRECTORY / f"{kind}_{name}_{basis_name}.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    """Write one JSON file per case."""
    DIRECTORY.mkdir(exist_ok=True)
    record_molecules()
    record_cells()
    record_kohn_sham()
    record_generalized_kohn_sham()
    record_periodic_kohn_sham()
    record_mp2()
    record_k_point_mp2()
    record_ccsd()
    record_k_point_ccsd()


def record_molecules() -> None:
    """Write the molecular cases."""
    for name, basis_name in STORED_CASES:
        molecule = MOLECULES[name]
        basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
        record(
            "one_electron",
            molecule,
            name,
            basis_name,
            "int1e_ovlp, int1e_kin, int1e_nuc; cart=False; unit=Bohr",
            one_electron_reference(molecule, basis),
        )
    for name, basis_name in DENSITY_FITTING_CASES:
        molecule = MOLECULES[name]
        basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
        auxiliary = BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY)
        record(
            "density_fitting",
            molecule,
            name,
            basis_name,
            "int2c2e and int3c2e (df.incore.aux_e2); cart=False; Bohr",
            density_fitting_reference(molecule, basis, auxiliary),
            auxiliary_basis=AUXILIARY,
        )
    for name, basis_name in FOUR_CENTER_CASES:
        molecule = MOLECULES[name]
        basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
        record(
            "four_center",
            molecule,
            name,
            basis_name,
            "int2e; cart=False; unit=Bohr",
            {"eri": four_center_reference(molecule, basis)},
        )
    for name, basis_name in FOURIER_CASES:
        molecule = MOLECULES[name]
        basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
        transforms = fourier_transform_reference(molecule, basis, G_VECTORS)
        parts = {
            f"{key}_{part}": getattr(np, part)(value)
            for key, value in transforms.items()
            for part in ("real", "imag")
        }
        record(
            "fourier",
            molecule,
            name,
            basis_name,
            "gto.ft_ao.ft_ao and ft_aopair with exp(-iG.r); cart=False; Bohr",
            {"g_vectors": G_VECTORS, **parts},
        )
    for name, basis_name, kind in SCF_CASES:
        molecule = SCF_MOLECULES[name]
        basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
        reference = scf_reference(molecule, basis, kind)
        record(
            "scf",
            molecule,
            name,
            basis_name,
            f"scf.{kind.upper()} with conv_tol=1e-12, conv_tol_grad=1e-8; Bohr",
            {key: np.asarray(value) for key, value in reference.items()},
            method=kind,
            charge=molecule.charge,
            spin=molecule.spin,
        )


def record_cells() -> None:
    """Write the periodic cases."""
    for name in EWALD_CASES:
        cell = CELLS[name]()
        record(
            "ewald",
            cell,
            name,
            "nuclei",
            "Cell.energy_nuc() with precision=1e-12; Bohr",
            {"energy": np.asarray(ewald_reference(cell))},
            dimension=cell.dimension,
            lattice_vectors_bohr=[list(vector) for vector in cell.lattice.vectors],
        )
    for name, basis_name in BLOCH_CASES:
        cell = BLOCH_CELLS[name]()
        basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
        mesh = BLOCH_MESHES[name]
        matrices = bloch_one_electron_reference(cell, basis, mesh)
        parts = {
            f"{key}_{part}": getattr(np, part)(value)
            for key, value in matrices.items()
            for part in ("real", "imag")
        }
        record(
            "bloch",
            cell,
            name,
            basis_name,
            "pbc_intor int1e_ovlp and int1e_kin at Cartesian kpts; precision=1e-14; "
            "low_dim_ft_type=inf_vacuum; cart=False; Bohr",
            parts,
            dimension=cell.dimension,
            lattice_vectors_bohr=[list(vector) for vector in cell.lattice.vectors],
            kpoints_fractional=[list(point) for point in mesh.fractional],
        )
    for name, basis_name in NUCLEAR_CASES:
        cell = NUCLEAR_CELLS[name]()
        basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
        mesh = NUCLEAR_MESHES[name]
        matrix = periodic_nuclear_attraction_reference(cell, basis, mesh)
        record(
            "nuclear",
            cell,
            name,
            basis_name,
            "df.GDF.get_nuc at Cartesian kpts and tools.pbc.madelung; "
            "precision=1e-14; cart=False; Bohr",
            {
                "nuclear_attraction_real": np.real(matrix),
                "nuclear_attraction_imag": np.imag(matrix),
                "madelung": np.asarray(madelung_reference(cell, basis, mesh)),
            },
            dimension=cell.dimension,
            lattice_vectors_bohr=[list(vector) for vector in cell.lattice.vectors],
            kpoints_fractional=[list(point) for point in mesh.fractional],
        )
    for name, basis_name in FITTING_CASES:
        cell = FITTING_CELLS[name]()
        basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
        auxiliary = BasisSet.for_atoms(cell, FITTING_AUXILIARY, library=LIBRARY)
        mesh = FITTING_MESHES[name]
        integrals = periodic_fitted_eri_reference(
            cell, basis, auxiliary, mesh, QUADRUPLES[name]
        )
        parts = {
            f"eri_{key}_{part}": getattr(np, part)(value)
            for key, value in integrals.items()
            for part in ("real", "imag")
        }
        record(
            "fitting",
            cell,
            name,
            basis_name,
            "df.GDF.get_eri with the same auxiliary basis at Cartesian kpts; "
            "precision=1e-14; cart=False; Bohr",
            parts,
            auxiliary_basis=FITTING_AUXILIARY,
            dimension=cell.dimension,
            lattice_vectors_bohr=[list(vector) for vector in cell.lattice.vectors],
            kpoints_fractional=[list(point) for point in mesh.fractional],
        )
    for name, basis_name, kind in KSCF_CASES:
        cell = KSCF_CELLS[name]()
        basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
        auxiliary = BasisSet.for_atoms(cell, KSCF_AUXILIARY, library=LIBRARY)
        mesh = KSCF_MESHES[name]
        reference = kscf_reference(cell, basis, auxiliary, mesh, kind)
        record(
            "kscf",
            cell,
            name,
            basis_name,
            f"pbc.scf.{kind.upper()} with density fitting in the same auxiliary "
            "basis, exxdiv=ewald, conv_tol=1e-12, conv_tol_grad=1e-8; KGHF from "
            "the KUHF density; spin_square of the supercell of the mesh; Bohr",
            {key: np.asarray(value) for key, value in reference.items()},
            method=kind,
            auxiliary_basis=KSCF_AUXILIARY,
            charge=cell.charge,
            spin=cell.spin,
            dimension=cell.dimension,
            lattice_vectors_bohr=[list(vector) for vector in cell.lattice.vectors],
            kpoints_fractional=[list(point) for point in mesh.fractional],
        )


if __name__ == "__main__":
    main()


def record_kohn_sham() -> None:
    """Write the Kohn–Sham cases, with PySCF's default grids."""
    for name, basis_name, kind, _, code in DFT_CASES:
        molecule = SCF_MOLECULES[name]
        basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
        reference = dft_reference(molecule, basis, code, kind)
        record(
            "dft",
            molecule,
            name,
            basis_name,
            f"dft.{kind.upper()} with xc={code!r}, nlc=0, grids.level=3; "
            "conv_tol=1e-12, conv_tol_grad=1e-8; Bohr",
            {key: np.asarray(value) for key, value in reference.items()},
            method=kind,
            functional=code,
            charge=molecule.charge,
            spin=molecule.spin,
        )


def record_generalized_kohn_sham() -> None:
    """Write the generalized Kohn–Sham cases with PySCF's noncollinear schemes."""
    for name, basis_name, _, code, collinear in GKS_CASES:
        molecule = SCF_MOLECULES[name]
        basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
        reference = gks_reference(molecule, basis, code, collinear=collinear)
        record(
            "gks",
            molecule,
            name,
            basis_name,
            f"dft.GKS with xc={code!r}, collinear={collinear!r}, nlc=0, "
            "grids.level=3; conv_tol=1e-12, conv_tol_grad=1e-8; Bohr",
            {key: np.asarray(value) for key, value in reference.items()},
            method="gks",
            functional=code,
            collinear=collinear,
            charge=molecule.charge,
            spin=molecule.spin,
        )


def record_periodic_kohn_sham() -> None:
    """Write the periodic Kohn–Sham cases with PySCF's grid at the stored level."""
    for name, basis_name, kind, _, code in KDFT_CASES:
        cell = KSCF_CELLS[name]()
        basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
        auxiliary = BasisSet.for_atoms(cell, KSCF_AUXILIARY, library=LIBRARY)
        mesh = KSCF_MESHES[name]
        reference = kdft_reference(
            cell, basis, auxiliary, mesh, code, kind, level=KDFT_LEVEL
        )
        record(
            "kdft",
            cell,
            name,
            basis_name,
            f"pbc.dft.{kind.upper()} with xc={code!r} on BeckeGrids level "
            f"{KDFT_LEVEL} (points and weights recorded), density fitting in the "
            "same auxiliary basis, exxdiv=ewald, conv_tol=1e-12, "
            "conv_tol_grad=1e-8; Bohr",
            {key: np.asarray(value) for key, value in reference.items()},
            method=kind,
            functional=code,
            auxiliary_basis=KSCF_AUXILIARY,
            charge=cell.charge,
            spin=cell.spin,
            dimension=cell.dimension,
            lattice_vectors_bohr=[list(vector) for vector in cell.lattice.vectors],
            kpoints_fractional=[list(point) for point in mesh.fractional],
        )


MP2_CLASSES = {
    "rhf": "mp.dfmp2.DFMP2",
    "uhf": "mp.dfump2.DFUMP2",
    "ghf": "mp.dfgmp2.DFGMP2",
}


def record_mp2() -> None:
    """Write the density-fitted MP2 cases."""
    for name, basis_name, kind in MP2_CASES:
        molecule = SCF_MOLECULES[name]
        basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
        auxiliary = BasisSet.for_atoms(molecule, MP2_AUXILIARY, library=LIBRARY)
        reference = mp2_reference(molecule, basis, auxiliary, kind)
        record(
            "mp2",
            molecule,
            name,
            basis_name,
            f"scf.{kind.upper()}.density_fit with conv_tol=1e-13, "
            f"conv_tol_grad=1e-10, then {MP2_CLASSES[kind]} in the same auxiliary "
            "basis; Bohr",
            {key: np.asarray(value) for key, value in reference.items()},
            method=kind,
            auxiliary=MP2_AUXILIARY,
            charge=molecule.charge,
            spin=molecule.spin,
        )


def record_k_point_mp2() -> None:
    """Write the k-point density-fitted MP2 cases."""
    for name, basis_name, kind in KMP2_CASES:
        cell = KSCF_CELLS[name]()
        basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
        auxiliary = BasisSet.for_atoms(cell, KSCF_AUXILIARY, library=LIBRARY)
        mesh = KSCF_MESHES[name]
        reference = kmp2_reference(cell, basis, auxiliary, mesh, kind)
        record(
            "kmp2",
            cell,
            name,
            basis_name,
            f"pbc.scf.{kind.upper()}.density_fit with conv_tol=1e-13, "
            "conv_tol_grad=1e-10, then pbc.mp.KMP2 in the same auxiliary basis; Bohr",
            {key: np.asarray(value) for key, value in reference.items()},
            method=kind,
            auxiliary=KSCF_AUXILIARY,
            mesh=list(mesh.mesh_size()),
            charge=cell.charge,
            spin=cell.spin,
        )


CCSD_CLASSES = {
    "rhf": "cc.dfccsd.RCCSD",
    "uhf": "cc.dfuccsd.UCCSD",
    "ghf": "cc.gccsd.GCCSD on with_df.get_eri",
}


def record_ccsd() -> None:
    """Write the density-fitted CCSD cases."""
    for name, basis_name, kind in CCSD_CASES:
        molecule = SCF_MOLECULES[name]
        basis = BasisSet.for_atoms(molecule, basis_name, library=LIBRARY)
        auxiliary = BasisSet.for_atoms(molecule, MP2_AUXILIARY, library=LIBRARY)
        reference = ccsd_reference(molecule, basis, auxiliary, kind)
        record(
            "ccsd",
            molecule,
            name,
            basis_name,
            f"scf.{kind.upper()}.density_fit with conv_tol=1e-13, "
            f"conv_tol_grad=1e-10, then {CCSD_CLASSES[kind]} in the same auxiliary "
            "basis with conv_tol=1e-12, conv_tol_normt=1e-10; Bohr",
            {key: np.asarray(value) for key, value in reference.items()},
            method=kind,
            auxiliary=MP2_AUXILIARY,
            charge=molecule.charge,
            spin=molecule.spin,
        )


def record_k_point_ccsd() -> None:
    """Write the k-point density-fitted CCSD cases, one file per class."""
    for name, basis_name, kind, exchange_correction in KCCSD_CASES:
        cell = KSCF_CELLS[name]()
        basis = BasisSet.for_atoms(cell, basis_name, library=LIBRARY)
        auxiliary = BasisSet.for_atoms(cell, KSCF_AUXILIARY, library=LIBRARY)
        mesh = KSCF_MESHES[name]
        reference = kccsd_reference(
            cell, basis, auxiliary, mesh, kind, exchange_correction=exchange_correction
        )
        exxdiv = "ewald" if exchange_correction else "None"
        keep = " with keep_exxdiv=True" if kind == "krhf" else ""
        record(
            "kccsd",
            cell,
            name,
            f"{basis_name}_{kind}",
            f"pbc.scf.{kind.upper()}(exxdiv={exxdiv}).density_fit with "
            f"conv_tol=1e-13, conv_tol_grad=1e-10, then pbc.cc.{kind[1].upper()}CCSD"
            f"{keep} in the same auxiliary basis with conv_tol=1e-12, "
            "conv_tol_normt=1e-10; Bohr",
            {key: np.asarray(value) for key, value in reference.items()},
            method=kind,
            exchange_correction=exchange_correction,
            auxiliary=KSCF_AUXILIARY,
            mesh=list(mesh.mesh_size()),
            charge=cell.charge,
            spin=cell.spin,
        )

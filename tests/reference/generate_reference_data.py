"""Record reference integrals from PySCF (run by hand).

    uv run python -m tests.reference.generate_reference_data

Rewrites ``tests/reference/data/one_electron_<molecule>_<basis>.json``,
``density_fitting_<molecule>_<basis>.json``,
``four_center_<molecule>_<basis>.json``, ``fourier_<molecule>_<basis>.json``,
``scf_<molecule>_<basis>.json`` and ``ewald_<cell>_nuclei.json`` with the
PySCF values and their provenance.
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
    density_fitting_reference,
    ewald_reference,
    four_center_reference,
    fourier_transform_reference,
    one_electron_reference,
    pyscf,
    scf_reference,
)
from tests.reference.test_density_fitting_reference import AUXILIARY
from tests.reference.test_density_fitting_reference import (
    STORED_CASES as DENSITY_FITTING_CASES,
)
from tests.reference.test_ewald_reference import CELLS
from tests.reference.test_ewald_reference import STORED_CASES as EWALD_CASES
from tests.reference.test_four_center_reference import (
    STORED_CASES as FOUR_CENTER_CASES,
)
from tests.reference.test_fourier_reference import STORED_CASES as FOURIER_CASES
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


if __name__ == "__main__":
    main()

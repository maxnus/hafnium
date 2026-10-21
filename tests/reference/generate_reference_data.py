"""Record reference integrals from PySCF (run by hand).

    uv run python -m tests.reference.generate_reference_data

Rewrites ``tests/reference/data/one_electron_<molecule>_<basis>.json``,
``density_fitting_<molecule>_<basis>.json``,
``four_center_<molecule>_<basis>.json`` and
``fourier_<molecule>_<basis>.json`` with the PySCF values and their
provenance.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from hafnium.basis import BasisSet
from tests.reference.pyscf_oracle import (
    density_fitting_reference,
    four_center_reference,
    fourier_transform_reference,
    one_electron_reference,
    pyscf,
)
from tests.reference.test_density_fitting_reference import AUXILIARY
from tests.reference.test_density_fitting_reference import (
    STORED_CASES as DENSITY_FITTING_CASES,
)
from tests.reference.test_four_center_reference import (
    STORED_CASES as FOUR_CENTER_CASES,
)
from tests.reference.test_fourier_reference import STORED_CASES as FOURIER_CASES
from tests.reference.test_one_electron_reference import (
    LIBRARY,
    MOLECULES,
    STORED_CASES,
)
from tests.vectors import G_VECTORS

DIRECTORY = Path(__file__).parent / "data"


def record(
    kind: str,
    name: str,
    basis_name: str,
    settings: str,
    matrices: dict[str, np.ndarray],
    **extra: Any,
) -> None:
    """Write ``<kind>_<name>_<basis>.json`` with provenance."""
    molecule = MOLECULES[name]
    provenance = {
        "program": "pyscf",
        "version": pyscf.__version__,
        "settings": settings,
        "date": datetime.date.today().isoformat(),
        "molecule": name,
        "basis": basis_name,
        **extra,
        "positions_bohr": [list(p) for p in molecule.positions],
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
        basis = BasisSet.for_molecule(molecule, basis_name, library=LIBRARY)
        record(
            "one_electron",
            name,
            basis_name,
            "int1e_ovlp, int1e_kin, int1e_nuc; cart=False; unit=Bohr",
            one_electron_reference(molecule, basis),
        )
    for name, basis_name in DENSITY_FITTING_CASES:
        molecule = MOLECULES[name]
        basis = BasisSet.for_molecule(molecule, basis_name, library=LIBRARY)
        auxiliary = BasisSet.for_molecule(molecule, AUXILIARY, library=LIBRARY)
        record(
            "density_fitting",
            name,
            basis_name,
            "int2c2e and int3c2e (df.incore.aux_e2); cart=False; Bohr",
            density_fitting_reference(molecule, basis, auxiliary),
            auxiliary_basis=AUXILIARY,
        )
    for name, basis_name in FOUR_CENTER_CASES:
        molecule = MOLECULES[name]
        basis = BasisSet.for_molecule(molecule, basis_name, library=LIBRARY)
        record(
            "four_center",
            name,
            basis_name,
            "int2e; cart=False; unit=Bohr",
            {"eri": four_center_reference(molecule, basis)},
        )
    for name, basis_name in FOURIER_CASES:
        molecule = MOLECULES[name]
        basis = BasisSet.for_molecule(molecule, basis_name, library=LIBRARY)
        transforms = fourier_transform_reference(molecule, basis, G_VECTORS)
        parts = {
            f"{key}_{part}": getattr(np, part)(value)
            for key, value in transforms.items()
            for part in ("real", "imag")
        }
        record(
            "fourier",
            name,
            basis_name,
            "gto.ft_ao.ft_ao and ft_aopair with exp(-iG.r); cart=False; Bohr",
            {"g_vectors": G_VECTORS, **parts},
        )


if __name__ == "__main__":
    main()

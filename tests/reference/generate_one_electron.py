"""Record one-electron reference integrals from PySCF (run by hand).

    uv run python -m tests.reference.generate_one_electron

Rewrites ``tests/reference/data/one_electron_<molecule>_<basis>.json`` with
the libcint values and their provenance.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from hafnium.basis import BasisSet
from tests.reference.pyscf_oracle import one_electron_reference, pyscf
from tests.reference.test_one_electron_reference import (
    LIBRARY,
    MOLECULES,
    STORED_CASES,
)


def main() -> None:
    """Write one JSON file per case."""
    directory = Path(__file__).parent / "data"
    directory.mkdir(exist_ok=True)
    for name, basis_name in STORED_CASES:
        molecule = MOLECULES[name]
        basis = BasisSet.for_molecule(molecule, basis_name, library=LIBRARY)
        matrices = one_electron_reference(molecule, basis)
        record = {
            "provenance": {
                "program": "pyscf",
                "version": pyscf.__version__,
                "settings": "int1e_ovlp, int1e_kin, int1e_nuc; cart=False; unit=Bohr",
                "date": datetime.date.today().isoformat(),
                "molecule": name,
                "basis": basis_name,
                "positions_bohr": [list(p) for p in molecule.positions],
            },
            **{key: matrix.tolist() for key, matrix in matrices.items()},
        }
        path = directory / f"one_electron_{name}_{basis_name}.json"
        path.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()

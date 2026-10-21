"""Shells, normalization, the basis library and basis set assembly."""

import json
import math
from pathlib import Path

import pytest

from hafnium.basis import BasisLibrary, BasisSet, Shell
from hafnium.basis.bse import basis_url, normalize_name
from hafnium.basis.library import bundled_basis_sets, shells_from_element
from hafnium.basis.shell import primitive_overlap
from hafnium.errors import BasisSetError
from hafnium.molecule import Molecule
from tests.tolerances import STRUCTURE_TOL

WATER = Molecule.from_xyz(
    "O 0 0 0.11779\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n"
)


def test_shell_counts_and_validation() -> None:
    shell = Shell(2, (1.0, 0.5), ((0.3, 0.7), (1.0, 0.0)))
    assert (shell.n_primitives, shell.n_contractions, shell.n_functions) == (2, 2, 10)
    with pytest.raises(BasisSetError, match="positive"):
        Shell(0, (0.0,), ((1.0,),))
    with pytest.raises(BasisSetError, match="one coefficient per primitive"):
        Shell(0, (1.0, 2.0), ((1.0,),))
    with pytest.raises(BasisSetError, match="at least one contraction"):
        Shell(0, (1.0,), ())


@pytest.mark.parametrize("angular_momentum", [0, 1, 2, 3])
def test_primitive_coefficients_give_unit_norm(angular_momentum: int) -> None:
    shell = Shell(angular_momentum, (5.0, 1.2, 0.3), ((0.15, 0.55, 0.45),))
    (row,) = shell.primitive_coefficients()
    power = angular_momentum + 1.5
    norm = sum(
        di * dj * math.gamma(power) / (2.0 * (ai + aj) ** power)
        for di, ai in zip(row, shell.exponents, strict=True)
        for dj, aj in zip(row, shell.exponents, strict=True)
    )
    assert norm == pytest.approx(1.0, abs=STRUCTURE_TOL)
    assert primitive_overlap(2.0, 2.0, angular_momentum) == pytest.approx(1.0)


def test_bundled_sets_are_available_offline() -> None:
    assert {"sto-3g", "def2-svp", "cc-pvdz", "def2-universal-jkfit"} <= set(
        bundled_basis_sets()
    )
    library = BasisLibrary(Path("/nonexistent"), offline=True)
    shells = library.shells("STO-3G", [8])
    assert [s.angular_momentum for s in shells[8]] == [0, 0, 1]
    assert sum(s.n_functions for s in shells[8]) == 5


@pytest.mark.parametrize(
    ("name", "n_functions", "max_l"),
    [("sto-3g", 7, 1), ("def2-svp", 24, 2), ("cc-pvdz", 24, 2)],
)
def test_basis_set_for_water(name: str, n_functions: int, max_l: int) -> None:
    library = BasisLibrary(Path("/nonexistent"), offline=True)
    basis = BasisSet.for_molecule(WATER, name, library=library)
    assert basis.n_functions == n_functions
    assert basis.max_angular_momentum == max_l
    assert basis.shell_atoms[0] == 0
    assert basis.shell_atoms[-1] == 2
    offsets = basis.shell_offsets()
    assert offsets[0] == 0
    assert offsets[-1] + basis.shells[-1].n_functions == n_functions
    labels = basis.function_labels(WATER)
    assert len(labels) == n_functions
    assert [label.m for label in labels if label.angular_momentum == 1][:3] == [
        -1,
        0,
        1,
    ]
    assert labels[0].symbol == "O"


def test_combined_sp_shells_are_split_and_cartesian_rejected() -> None:
    element = {
        "electron_shells": [
            {
                "function_type": "gto",
                "angular_momentum": [0, 1],
                "exponents": ["2.0", "0.5"],
                "coefficients": [["0.6", "0.4"], ["0.3", "0.7"]],
            }
        ]
    }
    shells = shells_from_element(element)
    assert [(s.angular_momentum, s.coefficients) for s in shells] == [
        (0, ((0.6, 0.4),)),
        (1, ((0.3, 0.7),)),
    ]
    element["electron_shells"][0]["function_type"] = "gto_cartesian"
    with pytest.raises(BasisSetError, match="spherical"):
        shells_from_element(element)
    with pytest.raises(BasisSetError, match="core potentials"):
        shells_from_element({"ecp_potentials": [], "electron_shells": []})


def test_library_fetches_once_and_then_uses_the_cache(tmp_path: Path) -> None:
    calls: list[str] = []
    bundled = json.loads(
        (
            Path(__file__).parents[2] / "hafnium" / "basis" / "data" / "sto-3g.json"
        ).read_text()
    )

    def fake_fetch(url: str) -> str:
        calls.append(url)
        return json.dumps({"elements": {"1": bundled["elements"]["1"]}})

    library = BasisLibrary(tmp_path, fetch=fake_fetch)
    first = library.shells("not-bundled", [1])
    second = library.shells("not-bundled", [1])
    assert len(calls) == 1
    assert calls[0] == basis_url("not-bundled", [1])
    assert (tmp_path / "not-bundled" / "1.json").is_file()
    assert first == second
    assert sum(s.n_functions for s in first[1]) == 1
    with pytest.raises(BasisSetError, match="does not define element"):
        library.shells("not-bundled", [2])


def test_offline_library_refuses_to_download(tmp_path: Path) -> None:
    library = BasisLibrary(tmp_path, offline=True)
    with pytest.raises(BasisSetError, match="offline"):
        library.shells("aug-cc-pvtz", [6])


def test_names_and_urls() -> None:
    assert normalize_name(" Def2-SVP ") == "def2-svp"
    assert basis_url("cc-pVDZ", [8, 1, 1]) == (
        "https://www.basissetexchange.org/api/basis/cc-pvdz/format/json/?elements=1,8"
    )

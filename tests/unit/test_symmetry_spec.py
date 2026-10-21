"""The Fukutome classes and their structures."""

import pytest

from hafnium.errors import InputError
from hafnium.structured import BlockStructure, Ring
from hafnium.symmetry import FUKUTOME_CLASSES, Field, SpinSymmetry, SymmetrySpec


def test_all_eight_classes_are_listed() -> None:
    acronyms = {fukutome.acronym for fukutome in FUKUTOME_CLASSES.values()}
    assert acronyms == {"TICS", "CCW", "ASCW", "ASDW", "ASW", "TSCW", "TSDW", "TSW"}


@pytest.mark.parametrize(
    ("name", "acronym", "structure"),
    [
        ("RHF", "TICS", BlockStructure.restricted(Ring.REAL)),
        ("complex rhf", "CCW", BlockStructure.restricted(Ring.COMPLEX)),
        ("UHF", "ASDW", BlockStructure.unrestricted(Ring.REAL)),
        ("complex UHF", "ASW", BlockStructure.unrestricted(Ring.COMPLEX)),
        ("paired UHF", "ASCW", BlockStructure.paired_unrestricted()),
        ("real GHF", "TSDW", BlockStructure.generalized(Ring.REAL)),
        ("complex GHF", "TSW", BlockStructure.generalized(Ring.COMPLEX)),
        ("paired GHF", "TSCW", BlockStructure.paired_generalized()),
        ("ascw", "ASCW", BlockStructure.paired_unrestricted()),
    ],
)
def test_names_map_to_classes_and_structures(
    name: str, acronym: str, structure: BlockStructure
) -> None:
    spec = SymmetrySpec.from_name(name)
    assert spec.fukutome.acronym == acronym
    assert spec.structure() == structure


def test_preserved_symmetries_follow_the_hierarchy() -> None:
    real_rhf = SymmetrySpec(SpinSymmetry.RESTRICTED, Field.REAL).fukutome.preserved
    assert set(real_rhf) == {"S^2", "S_z", "K", "Theta"}
    assert (
        SymmetrySpec(SpinSymmetry.GENERALIZED, Field.COMPLEX).fukutome.preserved == ()
    )
    assert SymmetrySpec(SpinSymmetry.UNRESTRICTED, Field.PAIRED).fukutome.preserved == (
        "S_z",
        "Theta",
    )


def test_restricted_paired_is_rejected() -> None:
    with pytest.raises(InputError, match="real RHF"):
        SymmetrySpec(SpinSymmetry.RESTRICTED, Field.PAIRED)


def test_unknown_names_are_rejected() -> None:
    with pytest.raises(InputError, match="unknown"):
        SymmetrySpec.from_name("ROHF")


def test_electron_count_validation() -> None:
    SymmetrySpec.from_name("UHF").validate_electrons(3, 2)
    SymmetrySpec.from_name("paired GHF").validate_electrons(3, 1)
    with pytest.raises(InputError, match="n_alpha == n_beta"):
        SymmetrySpec.from_name("RHF").validate_electrons(3, 2)
    with pytest.raises(InputError, match="paired UHF"):
        SymmetrySpec.from_name("paired UHF").validate_electrons(3, 2)
    with pytest.raises(InputError, match="even"):
        SymmetrySpec.from_name("paired GHF").validate_electrons(2, 1)
    with pytest.raises(InputError, match="non-negative"):
        SymmetrySpec.from_name("UHF").validate_electrons(-1, 0)

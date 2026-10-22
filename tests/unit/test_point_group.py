"""Abelian point groups: tables, characters, products and harmonic parities."""

import pytest

from hafnium.errors import InputError, SymmetryError
from hafnium.symmetry import (
    ABELIAN_GROUP_NAMES,
    Irrep,
    Operation,
    PointGroup,
    solid_harmonic_parity,
)

COTTON_LABELS = {
    "C1": ("A",),
    "Ci": ("Ag", "Au"),
    "Cs": ("A'", 'A"'),
    "C2": ("A", "B"),
    "C2h": ("Ag", "Bg", "Au", "Bu"),
    "C2v": ("A1", "A2", "B1", "B2"),
    "D2": ("A", "B1", "B2", "B3"),
    "D2h": ("Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"),
}


@pytest.mark.parametrize("name", ABELIAN_GROUP_NAMES)
def test_groups_are_abelian_with_orthogonal_characters(name: str) -> None:
    group = PointGroup.named(name)
    assert group.labels == COTTON_LABELS[name]
    assert group.order == group.n_irreps == len(group.operations)
    assert group.operations[0] is Operation.IDENTITY
    table = group.characters()
    for i, row in enumerate(table):
        assert row[0] == 1
        for j, other in enumerate(table):
            inner = sum(a * b for a, b in zip(row, other, strict=True))
            assert inner == (group.order if i == j else 0)
    for first in group.operations:
        for second in group.operations:
            assert first.compose(second) is second.compose(first)
            assert first.compose(second) in group.operations


def test_named_is_case_insensitive_and_rejects_other_groups() -> None:
    assert PointGroup.named("c2v").name == "C2v"
    assert PointGroup.named(" D2H ") == PointGroup.named("D2h")
    with pytest.raises(InputError, match="non-Abelian"):
        PointGroup.named("C3v")
    with pytest.raises(InputError, match="unknown"):
        PointGroup.named("Dinfh")


def test_operations_are_sign_matrices() -> None:
    assert Operation.C2_Z.signs == (-1, -1, 1)
    assert Operation.SIGMA_XY.signs == (1, 1, -1)
    assert Operation.C2_Z.compose(Operation.SIGMA_XY) is Operation.INVERSION
    assert Operation.SIGMA_XZ.compose(Operation.SIGMA_YZ) is Operation.C2_Z
    assert Operation.INVERSION.symbol == "i"
    assert Operation.C2_X.character((1, 0, 0)) == 1
    assert Operation.C2_X.character((0, 1, 0)) == -1


@pytest.mark.parametrize(
    ("name", "parity", "label"),
    [
        ("C2v", (1, 0, 0), "B1"),
        ("C2v", (0, 1, 0), "B2"),
        ("C2v", (0, 0, 1), "A1"),
        ("C2v", (1, 1, 0), "A2"),
        ("D2h", (1, 0, 0), "B3u"),
        ("D2h", (0, 1, 0), "B2u"),
        ("D2h", (0, 0, 1), "B1u"),
        ("D2h", (1, 1, 0), "B1g"),
        ("D2h", (1, 0, 1), "B2g"),
        ("D2h", (1, 1, 1), "Au"),
        ("C2h", (1, 0, 0), "Bu"),
        ("C2h", (0, 0, 1), "Au"),
        ("C2h", (1, 0, 1), "Bg"),
        ("D2", (0, 0, 1), "B1"),
        ("Cs", (1, 0, 0), "A'"),
        ("Cs", (0, 0, 1), 'A"'),
        ("Ci", (1, 1, 0), "Ag"),
    ],
)
def test_functions_of_a_parity_span_the_expected_irrep(
    name: str, parity: tuple[int, int, int], label: str
) -> None:
    group = PointGroup.named(name)
    assert group.labels[group.irrep_of(parity)] == label


def test_direct_products() -> None:
    d2h = PointGroup.named("D2h")
    labels = d2h.labels
    assert labels[d2h.product(labels.index("B1u"), labels.index("B2u"))] == "B3g"
    assert labels[d2h.product(labels.index("Au"), labels.index("Au"))] == "Ag"
    c2v = PointGroup.named("C2v")
    assert c2v.labels[c2v.product(2, 3)] == "A2"
    assert c2v.labels[c2v.product(1, 2)] == "B2"


def test_constructor_validates_closure_and_characters() -> None:
    with pytest.raises(SymmetryError, match="closed"):
        PointGroup(
            "bad",
            (Operation.IDENTITY, Operation.C2_Z, Operation.C2_X),
            (Irrep("A", (0, 0, 0)), Irrep("B", (1, 0, 0)), Irrep("C", (0, 0, 1))),
        )
    with pytest.raises(SymmetryError, match="identity"):
        PointGroup("bad", (Operation.C2_Z,), (Irrep("A", (0, 0, 0)),))
    with pytest.raises(SymmetryError, match="distinct"):
        PointGroup(
            "bad",
            (Operation.IDENTITY, Operation.C2_Z),
            (Irrep("A", (0, 0, 0)), Irrep("A'", (0, 0, 1))),
        )


# The parities of PySCF's spherical functions (pyscf.symm.param), d and f
# shells in the order m = -l, ..., l; p shells are x, y, z.
PYSCF_PARITIES = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(1, 1, 0), (0, 1, 1), (0, 0, 0), (1, 0, 1), (0, 0, 0)],
    3: [(0, 1, 0), (1, 1, 1), (0, 1, 0), (0, 0, 1), (1, 0, 0), (0, 0, 1), (1, 0, 0)],
    4: [
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 0),
        (0, 1, 1),
        (0, 0, 0),
        (1, 0, 1),
        (0, 0, 0),
        (1, 0, 1),
        (0, 0, 0),
    ],
}


@pytest.mark.parametrize("l", sorted(PYSCF_PARITIES))
def test_solid_harmonic_parities_match_pyscf(l: int) -> None:
    from hafnium.basis.basis_set import m_values

    assert [solid_harmonic_parity(l, m) for m in m_values(l)] == PYSCF_PARITIES[l]
    with pytest.raises(InputError, match="no harmonic"):
        solid_harmonic_parity(l, l + 1)

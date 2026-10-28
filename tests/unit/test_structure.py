"""Validation of block structures."""

import numpy as np
import pytest

from hafnium.errors import StructureError
from hafnium.structured import (
    Block,
    BlockStructure,
    IrrepSectors,
    KAxis,
    Partner,
    Ring,
    Spin,
)
from hafnium.symmetry import PointGroup

C2V = PointGroup.named("C2v").characters()
CS = PointGroup.named("Cs").characters()


def test_constructors_describe_themselves() -> None:
    assert BlockStructure.spatial().describe() == "spatial (real)"
    assert BlockStructure.restricted(Ring.COMPLEX).describe() == "restricted (complex)"
    assert BlockStructure.unrestricted().describe() == "unrestricted (real)"
    assert BlockStructure.paired_unrestricted().describe() == "paired unrestricted"
    assert (
        BlockStructure.generalized(Ring.COMPLEX).describe() == "generalized (complex)"
    )
    assert BlockStructure.paired_generalized().describe() == "paired generalized"


def test_stored_and_derived_blocks() -> None:
    restricted = BlockStructure.restricted()
    assert restricted.n_stored == 1
    assert restricted.derived_partner is Partner.COPY
    assert BlockStructure.paired_unrestricted().derived_partner is Partner.CONJUGATE
    assert BlockStructure.unrestricted().derived_partner is None
    assert BlockStructure.paired_generalized().is_complex
    assert not BlockStructure.generalized().is_complex


def test_with_ring() -> None:
    assert BlockStructure.unrestricted().with_ring(
        Ring.COMPLEX
    ) == BlockStructure.unrestricted(Ring.COMPLEX)
    with pytest.raises(StructureError):
        BlockStructure.paired_unrestricted().with_ring(Ring.REAL)
    with pytest.raises(StructureError):
        BlockStructure.generalized().with_ring(Ring.QUATERNION)


def test_invalid_blocks_are_rejected() -> None:
    with pytest.raises(StructureError, match="together"):
        Block(Spin.BETA, Ring.REAL, derived_from=0)
    with pytest.raises(StructureError, match="weight"):
        Block(Spin.ALPHA, Ring.REAL, weight=0)
    with pytest.raises(StructureError, match="spinor"):
        Block(Spin.ALPHA, Ring.QUATERNION)
    with pytest.raises(StructureError, match="cannot be derived"):
        Block(Spin.SPINOR, Ring.REAL, derived_from=0, partner=Partner.COPY)


def test_blocks_are_normalized_to_a_tuple() -> None:
    from_list = BlockStructure([Block(Spin.NONE, Ring.REAL)])  # type: ignore[arg-type]
    assert from_list == BlockStructure.spatial()
    assert hash(from_list) == hash(BlockStructure.spatial())


def test_beta_before_alpha_is_rejected() -> None:
    with pytest.raises(StructureError, match="alpha block precedes"):
        BlockStructure((Block(Spin.BETA, Ring.REAL), Block(Spin.ALPHA, Ring.REAL)))


def test_invalid_structures_are_rejected() -> None:
    with pytest.raises(StructureError, match="at least one"):
        BlockStructure(())
    with pytest.raises(StructureError, match="once"):
        BlockStructure((Block(Spin.ALPHA, Ring.REAL), Block(Spin.ALPHA, Ring.REAL)))
    with pytest.raises(StructureError, match="unsupported"):
        BlockStructure((Block(Spin.ALPHA, Ring.REAL),))
    with pytest.raises(StructureError, match="follow its source"):
        BlockStructure(
            (
                Block(Spin.ALPHA, Ring.REAL, derived_from=1, partner=Partner.COPY),
                Block(Spin.BETA, Ring.REAL),
            )
        )
    with pytest.raises(StructureError, match="ring"):
        BlockStructure(
            (
                Block(Spin.ALPHA, Ring.REAL),
                Block(Spin.BETA, Ring.COMPLEX, derived_from=0, partner=Partner.COPY),
            )
        )


def sectors(dimensions: tuple[int, ...]) -> IrrepSectors:
    """Irrep sectors of a basis split into blocks of ``dimensions`` functions."""
    n = sum(dimensions)
    eye = np.eye(n)
    start = 0
    blocks: list[np.ndarray] = []
    for dimension in dimensions:
        blocks.append(eye[:, start : start + dimension])
        start += dimension
    labels = ("A1", "B1", "B2")[: len(dimensions)]
    characters = {"A1": C2V[0], "B1": C2V[2], "B2": C2V[3]}
    return IrrepSectors(
        "C2v", labels, tuple(blocks), tuple(characters[label] for label in labels)
    )


def test_irrep_sectors_repeat_the_spin_pattern() -> None:
    pattern = BlockStructure.restricted()
    structure = pattern.with_irreps(sectors((3, 1, 2)))
    assert structure.n_sectors == 3
    assert structure.sector_structure == pattern
    assert structure.with_irreps(None) == pattern
    assert structure.describe() == "restricted (real) in C2v"
    assert [block.irrep for block in structure.blocks] == [0, 0, 1, 1, 2, 2]
    assert [block.derived_from for block in structure.blocks] == [
        None,
        0,
        None,
        2,
        None,
        4,
    ]
    assert structure.irrep_labels == ("A1", "B1", "B2")
    assert structure.sector_slice(1) == slice(1, 2)
    assert BlockStructure.unrestricted().with_irreps(sectors((2, 2))).sector_slice(
        1
    ) == slice(2, 4)
    assert structure.with_ring(Ring.COMPLEX) == BlockStructure.restricted(
        Ring.COMPLEX
    ).with_irreps(sectors((3, 1, 2)))
    assert pattern.irrep_labels == (None,)
    assert pattern.n_sectors == 1
    assert pattern.sector_slice(0) == slice(0, 1)


def test_irrep_sectors_compare_by_value() -> None:
    assert sectors((2, 1)) == sectors((2, 1))
    assert hash(sectors((2, 1))) == hash(sectors((2, 1)))
    assert sectors((2, 1)) != sectors((1, 2))
    assert BlockStructure.restricted().with_irreps(
        sectors((2, 1))
    ) == BlockStructure.restricted().with_irreps(sectors((2, 1)))
    with pytest.raises(StructureError, match="orthogonal"):
        IrrepSectors("Cs", ("A'", "A''"), (np.ones((2, 1)), np.ones((2, 1))), CS)
    with pytest.raises(StructureError, match="at least one column"):
        IrrepSectors("Cs", ("A'", "A''"), (np.eye(2), np.zeros((2, 0))), CS)
    with pytest.raises(StructureError, match="every spanned irrep"):
        IrrepSectors("Cs", ("A'",), (np.eye(3)[:, :2],), CS[:1])
    with pytest.raises(StructureError, match="distinct"):
        IrrepSectors("Cs", ("A'", "A'"), (np.eye(2)[:, :1], np.eye(2)[:, 1:]), CS)
    with pytest.raises(StructureError, match="labelled"):
        sectors((2, 1)).index("A2")


def test_irrep_sectors_know_their_characters() -> None:
    three = sectors((2, 1, 1))
    assert three.n_operations == 4
    assert three.sector_of(C2V[0]) == 0
    assert three.sector_of(C2V[1]) is None
    assert three.sector_of(C2V[3]) == 2
    other = (C2V[0], C2V[1], C2V[2])
    assert three != IrrepSectors("C2v", three.labels, three.coefficients, other)
    with pytest.raises(StructureError, match="character rows"):
        IrrepSectors("Cs", ("A'", "A''"), (np.eye(2)[:, :1], np.eye(2)[:, 1:]), CS[:1])
    with pytest.raises(StructureError, match="±1"):
        IrrepSectors(
            "Cs", ("A'", "A''"), (np.eye(2)[:, :1], np.eye(2)[:, 1:]), ((1, 1), (1, 0))
        )
    with pytest.raises(StructureError, match="distinct characters"):
        IrrepSectors(
            "Cs", ("A'", "A''"), (np.eye(2)[:, :1], np.eye(2)[:, 1:]), (CS[0], CS[0])
        )
    with pytest.raises(StructureError, match="one character per operation"):
        IrrepSectors(
            "Cs", ("A'", "A''"), (np.eye(2)[:, :1], np.eye(2)[:, 1:]), ((1, 1), (1,))
        )


def test_irrep_sectors_are_validated() -> None:
    two = sectors((2, 1))
    with pytest.raises(StructureError, match="only in a structure with irrep"):
        BlockStructure((Block(Spin.NONE, Ring.REAL, irrep=0),))
    with pytest.raises(StructureError, match="molecular"):
        BlockStructure.restricted().with_k_axis(KAxis.gamma()).with_irreps(two)
    with pytest.raises(StructureError, match="do not repeat"):
        BlockStructure((Block(Spin.NONE, Ring.REAL, irrep=0),), irreps=two)
    with pytest.raises(StructureError, match="irrep index"):
        BlockStructure(
            (
                Block(Spin.NONE, Ring.REAL, irrep=0),
                Block(Spin.NONE, Ring.REAL, irrep=0),
            ),
            irreps=two,
        )
    with pytest.raises(StructureError, match="does not repeat the spin pattern"):
        BlockStructure(
            (
                Block(Spin.ALPHA, Ring.REAL, irrep=0),
                Block(
                    Spin.BETA, Ring.REAL, derived_from=0, partner=Partner.COPY, irrep=0
                ),
                Block(Spin.ALPHA, Ring.REAL, irrep=1),
                Block(Spin.BETA, Ring.REAL, irrep=1),
            ),
            irreps=two,
        )
    with pytest.raises(StructureError, match="non-negative"):
        Block(Spin.NONE, Ring.REAL, irrep=-1)

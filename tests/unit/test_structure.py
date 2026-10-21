"""Validation of block structures."""

import pytest

from hafnium.errors import StructureError
from hafnium.structured import Block, BlockStructure, Partner, Ring, Spin


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

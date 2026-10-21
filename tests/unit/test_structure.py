"""Validation of block structures."""

import pytest

from hafnium.errors import StructureError
from hafnium.structured import Block, Partner, Ring, Spin, Structure


def test_constructors_describe_themselves() -> None:
    assert Structure.spatial().describe() == "spatial (real)"
    assert Structure.restricted(Ring.COMPLEX).describe() == "restricted (complex)"
    assert Structure.unrestricted().describe() == "unrestricted (real)"
    assert Structure.paired_unrestricted().describe() == "paired unrestricted"
    assert Structure.generalized(Ring.COMPLEX).describe() == "generalized (complex)"
    assert Structure.paired_generalized().describe() == "paired generalized"


def test_stored_and_derived_blocks() -> None:
    restricted = Structure.restricted()
    assert restricted.n_stored == 1
    assert restricted.derived_partner is Partner.COPY
    assert Structure.paired_unrestricted().derived_partner is Partner.CONJUGATE
    assert Structure.unrestricted().derived_partner is None
    assert Structure.paired_generalized().is_complex
    assert not Structure.generalized().is_complex


def test_with_ring() -> None:
    assert Structure.unrestricted().with_ring(Ring.COMPLEX) == Structure.unrestricted(
        Ring.COMPLEX
    )
    with pytest.raises(StructureError):
        Structure.paired_unrestricted().with_ring(Ring.REAL)
    with pytest.raises(StructureError):
        Structure.generalized().with_ring(Ring.QUATERNION)


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
    from_list = Structure([Block(Spin.NONE, Ring.REAL)])  # type: ignore[arg-type]
    assert from_list == Structure.spatial()
    assert hash(from_list) == hash(Structure.spatial())


def test_beta_before_alpha_is_rejected() -> None:
    with pytest.raises(StructureError, match="alpha block precedes"):
        Structure((Block(Spin.BETA, Ring.REAL), Block(Spin.ALPHA, Ring.REAL)))


def test_invalid_structures_are_rejected() -> None:
    with pytest.raises(StructureError, match="at least one"):
        Structure(())
    with pytest.raises(StructureError, match="once"):
        Structure((Block(Spin.ALPHA, Ring.REAL), Block(Spin.ALPHA, Ring.REAL)))
    with pytest.raises(StructureError, match="unsupported"):
        Structure((Block(Spin.ALPHA, Ring.REAL),))
    with pytest.raises(StructureError, match="follow its source"):
        Structure(
            (
                Block(Spin.ALPHA, Ring.REAL, derived_from=1, partner=Partner.COPY),
                Block(Spin.BETA, Ring.REAL),
            )
        )
    with pytest.raises(StructureError, match="ring"):
        Structure(
            (
                Block(Spin.ALPHA, Ring.REAL),
                Block(Spin.BETA, Ring.COMPLEX, derived_from=0, partner=Partner.COPY),
            )
        )

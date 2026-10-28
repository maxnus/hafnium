"""Block tensors (ADR-0030): allowed and derived blocks, contractions, dense form."""

from collections.abc import Iterator

import numpy as np
import pytest

from hafnium.errors import StructureError
from hafnium.structured import (
    BlockStructure,
    BlockTensor,
    KAxis,
    Partner,
    Ring,
    TensorAxis,
    TensorStructure,
    contract,
)
from hafnium.symmetry import SymmetrySpec
from tests.tolerances import STRUCTURE_TOL
from tests.unit.test_scf import CLASSES
from tests.unit.test_structure import sectors

SECTORS = sectors((2, 1, 1))


def structure_named(name: str, with_sectors: bool) -> BlockStructure:
    structure = SymmetrySpec.from_name(name).structure()
    return structure.with_irreps(SECTORS) if with_sectors else structure


def ring_of(*structures: BlockStructure) -> Ring:
    return (
        Ring.COMPLEX
        if any(structure.is_complex for structure in structures)
        else Ring.REAL
    )


def axis(
    structure: BlockStructure, sizes: Iterator[int], *, conjugated: bool = False
) -> TensorAxis:
    """An axis with the next sizes for the stored blocks, derived ones following."""
    expanded: list[int] = []
    for block in structure.blocks:
        expanded.append(
            next(sizes) if block.is_stored else expanded[block.derived_from or 0]
        )
    return TensorAxis(structure, tuple(expanded), conjugated)


def random_sizes(rng: np.random.Generator) -> Iterator[int]:
    while True:
        yield int(rng.integers(1, 4))


def random_member(rng: np.random.Generator, structure: TensorStructure) -> BlockTensor:
    """A random tensor of the structure: a random dense array projected onto it."""
    shape = tuple(axis.extent for axis in structure.axes)
    dense = rng.standard_normal(shape)
    if structure.ring is Ring.COMPLEX:
        dense = dense + 1j * rng.standard_normal(shape)
    return BlockTensor.from_dense(structure, dense, check=False)


def fitted_axes(
    name: str, with_sectors: bool, rng: np.random.Generator
) -> tuple[TensorAxis, TensorAxis, TensorAxis, Ring]:
    """Auxiliary, occupied (conjugated) and virtual axes of a class."""
    mo = structure_named(name, with_sectors)
    aux = BlockStructure.spatial()
    if with_sectors:
        aux = aux.with_irreps(SECTORS)
    sizes = random_sizes(rng)
    return (
        axis(aux, iter([3, 2, 1] if with_sectors else [4])),
        axis(mo, sizes, conjugated=True),
        axis(mo, sizes),
        ring_of(mo),
    )


# ------------------------------------------------------------------ structures


def test_axes_are_validated() -> None:
    unrestricted = BlockStructure.unrestricted()
    with pytest.raises(StructureError, match="2 sectors"):
        TensorAxis(unrestricted, (1,))
    with pytest.raises(StructureError, match="non-negative"):
        TensorAxis(unrestricted, (1, -1))
    with pytest.raises(StructureError, match="as many functions as its source"):
        TensorAxis(BlockStructure.restricted(), (2, 3))
    with pytest.raises(StructureError, match="k-points"):
        TensorAxis(unrestricted.with_k_axis(KAxis.gamma()), (1, 1))
    occupied = TensorAxis(unrestricted, (2, 1), conjugated=True)
    assert occupied.extent == 3
    assert occupied.offset(1) == 2
    assert occupied.conjugate().conjugated is False
    assert occupied.same_space(occupied.conjugate())
    assert occupied.partner(0) == 0
    assert TensorAxis(BlockStructure.restricted(), (2, 2)).partner(0) == 1


def test_spin_charges_select_the_blocks() -> None:
    unrestricted = BlockStructure.unrestricted()
    bra = TensorAxis(unrestricted, (2, 1), conjugated=True)
    ket = TensorAxis(unrestricted, (2, 1))
    assert TensorStructure((bra, ket)).keys == ((0, 0), (1, 1))
    # Two ket indices of one spin carry charge two; alpha with beta cancels.
    assert TensorStructure((ket, ket)).keys == ((0, 1), (1, 0))
    assert TensorStructure((bra, bra, ket, ket)).keys == (
        (0, 0, 0, 0),
        (0, 1, 0, 1),
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (1, 0, 1, 0),
        (1, 1, 1, 1),
    )
    spinor = TensorAxis(BlockStructure.generalized(), (3,))
    assert TensorStructure((spinor.conjugate(), spinor)).keys == ((0, 0),)
    assert TensorStructure(()).keys == ((),)
    with pytest.raises(StructureError, match="sectors, got"):
        TensorStructure((bra, ket)).is_allowed((0,))


def test_irrep_characters_select_the_blocks() -> None:
    # C2v with the sectors A1, B1, B2: B1 ⊗ B2 = A2, which the basis does not
    # span, so a block B1 B2 of two ket indices is forbidden while B1 B1 is not.
    spatial = BlockStructure.spatial().with_irreps(SECTORS)
    ket = TensorAxis(spatial, (1, 1, 1))
    assert TensorStructure((ket, ket)).keys == ((0, 0), (1, 1), (2, 2))
    assert TensorStructure((ket, ket, ket)).keys == (
        (0, 0, 0),
        (0, 1, 1),
        (0, 2, 2),
        (1, 0, 1),
        (1, 1, 0),
        (2, 0, 2),
        (2, 2, 0),
    )
    # An empty sector removes its blocks.
    assert TensorStructure((ket, TensorAxis(spatial, (1, 0, 1)))).keys == (
        (0, 0),
        (2, 2),
    )


def test_partner_rules_derive_blocks() -> None:
    restricted = BlockStructure.restricted()
    bra = TensorAxis(restricted, (2, 2), conjugated=True)
    ket = TensorAxis(restricted, (2, 2))
    structure = TensorStructure((bra, ket))
    assert structure.partner is Partner.COPY
    assert structure.keys == ((0, 0), (1, 1))
    assert structure.stored_keys == ((0, 0),)
    assert structure.derivation((1, 1)) == ((0, 0), False)
    assert structure.position((1, 1)) is None
    four = TensorStructure((bra, bra, ket, ket))
    assert four.stored_keys == ((0, 0, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0))
    assert four.derivation((1, 0, 1, 0)) == ((0, 1, 0, 1), False)
    paired = BlockStructure.paired_unrestricted()
    conjugating = TensorStructure(
        (
            TensorAxis(paired, (2, 2), conjugated=True),
            TensorAxis(paired, (2, 2)),
        ),
        ring=Ring.COMPLEX,
    )
    assert conjugating.partner is Partner.CONJUGATE
    assert conjugating.derivation((1, 1)) == ((0, 0), True)
    assert (
        TensorStructure(
            (TensorAxis(BlockStructure.unrestricted(), (1, 1)),) * 0
        ).partner
        is None
    )


def test_structures_are_validated() -> None:
    restricted = TensorAxis(BlockStructure.restricted(), (1, 1))
    paired = TensorAxis(BlockStructure.paired_unrestricted(), (1, 1))
    complex_spatial = TensorAxis(BlockStructure.spatial(Ring.COMPLEX), (2,))
    with pytest.raises(StructureError, match="one partner rule"):
        TensorStructure((restricted, paired.conjugate()), ring=Ring.COMPLEX)
    with pytest.raises(StructureError, match="hold real blocks"):
        TensorStructure(
            (paired.conjugate(), paired, complex_spatial), ring=Ring.COMPLEX
        )
    with pytest.raises(StructureError, match=r"ring=Ring\.COMPLEX"):
        TensorStructure((paired.conjugate(), paired))
    with pytest.raises(StructureError, match="real or the complex"):
        TensorStructure((restricted,), ring=Ring.QUATERNION)
    sectored = TensorAxis(BlockStructure.spatial().with_irreps(SECTORS), (1, 1, 1))
    with pytest.raises(StructureError, match="or none"):
        TensorStructure((sectored, restricted))
    other = TensorAxis(BlockStructure.spatial().with_irreps(sectors((1, 1))), (1, 1))
    assert TensorStructure((sectored, other)).n_axes == 2
    assert TensorStructure((sectored,)).describe().startswith("1-index tensor over")


# ------------------------------------------------------------------- tensors


def test_blocks_are_validated_and_filled() -> None:
    unrestricted = BlockStructure.unrestricted()
    structure = TensorStructure(
        (
            TensorAxis(unrestricted, (2, 1), conjugated=True),
            TensorAxis(unrestricted, (2, 1)),
        )
    )
    with pytest.raises(StructureError, match="stores 2 blocks"):
        BlockTensor(structure, (np.zeros((2, 2)),))
    with pytest.raises(StructureError, match="has shape"):
        BlockTensor(structure, (np.zeros((2, 2)), np.zeros((2, 1))))
    with pytest.raises(StructureError, match="must be float64"):
        BlockTensor(structure, (np.zeros((2, 2), dtype=complex), np.zeros((1, 1))))
    with pytest.raises(StructureError, match="not stored"):
        BlockTensor.from_blocks(structure, {(0, 1): np.zeros((2, 1))})
    with pytest.raises(StructureError, match=r"ring=Ring\.COMPLEX"):
        BlockTensor.from_blocks(structure, {(0, 0): np.zeros((2, 2), dtype=complex)})
    partial = BlockTensor.from_blocks(structure, {(1, 1): np.ones((1, 1))})
    complex_structure = TensorStructure(structure.axes, ring=Ring.COMPLEX)
    upcast = BlockTensor.from_blocks(complex_structure, {(1, 1): np.ones((1, 1))})
    assert upcast.block((1, 1)).dtype == np.complex128
    assert upcast.block((0, 0)).dtype == np.complex128
    np.testing.assert_array_equal(partial.block((0, 0)), np.zeros((2, 2)))
    assert BlockTensor.zeros(structure).sum() == 0.0
    counted = BlockTensor.from_function(
        structure, lambda key: np.full(structure.shape(key), float(sum(key)))
    )
    assert counted.block((1, 1))[0, 0] == 2.0
    with pytest.raises(StructureError, match="vanishes by symmetry"):
        counted.block((0, 1))
    with pytest.raises(StructureError, match="without axes"):
        _ = counted.scalar


@pytest.mark.parametrize("with_sectors", [False, True])
@pytest.mark.parametrize("name", CLASSES)
def test_contraction_matches_the_dense_einsum(name: str, with_sectors: bool) -> None:
    rng = np.random.default_rng(hash((name, with_sectors)) % 2**32)
    aux, occupied, virtual, ring = fitted_axes(name, with_sectors, rng)
    fitted = random_member(rng, TensorStructure((aux, occupied, virtual), ring=ring))
    swapped = random_member(
        rng,
        TensorStructure((aux, virtual.conjugate(), occupied.conjugate()), ring=ring),
    )
    coulomb = contract("Pia,Pbj->iajb", fitted, swapped.conj())
    expected: np.ndarray = np.einsum(
        "Pia,Pbj->iajb", fitted.to_dense(), swapped.to_dense().conj()
    )
    np.testing.assert_allclose(coulomb.to_dense(), expected, atol=STRUCTURE_TOL)
    assert coulomb.structure.ring is ring
    assert [axis.conjugated for axis in coulomb.structure.axes] == [
        True,
        False,
        True,
        False,
    ]
    if coulomb.structure.partner is not None:
        assert coulomb.structure.n_stored < len(coulomb.structure.keys)
    # Free indices from both operands, a derived block of the first operand.
    field = random_member(
        rng,
        TensorStructure((occupied.conjugate(), virtual.conjugate()), ring=ring),
    )
    folded = contract("iajb,jb->ia", coulomb, field)
    np.testing.assert_allclose(
        folded.to_dense(),
        np.einsum("iajb,jb->ia", expected, field.to_dense()),
        atol=STRUCTURE_TOL,
    )
    # An outer product sums over nothing.
    outer = contract("ia,jb->iajb", field.conj(), field)
    np.testing.assert_allclose(
        outer.to_dense(),
        np.einsum("ia,jb->iajb", field.to_dense().conj(), field.to_dense()),
        atol=STRUCTURE_TOL,
    )
    # The full contraction is a scalar over every block, derived ones included.
    total = contract("iajb,iajb->", coulomb.conj(), coulomb).scalar
    assert complex(total) == pytest.approx(
        complex(np.sum(np.abs(expected) ** 2)), abs=STRUCTURE_TOL
    )


@pytest.mark.parametrize("with_sectors", [False, True])
@pytest.mark.parametrize("name", CLASSES)
def test_permutation_sum_and_algebra_match_the_dense_form(
    name: str, with_sectors: bool
) -> None:
    rng = np.random.default_rng(hash((name, with_sectors, 1)) % 2**32)
    _, occupied, virtual, ring = fitted_axes(name, with_sectors, rng)
    structure = TensorStructure((occupied, virtual, occupied, virtual), ring=ring)
    tensor = random_member(rng, structure)
    dense = tensor.to_dense()
    np.testing.assert_allclose(
        tensor.permute("iajb->ijba").to_dense(),
        np.transpose(dense, (0, 2, 3, 1)),
        atol=STRUCTURE_TOL,
    )
    assert tensor.permute("iajb->ijba").structure == structure.permuted((0, 2, 3, 1))
    assert complex(tensor.sum()) == pytest.approx(
        complex(dense.sum()), abs=STRUCTURE_TOL
    )
    np.testing.assert_allclose(tensor.conj().to_dense(), dense.conj())
    assert tensor.conj().structure == structure.conjugated()
    np.testing.assert_allclose((tensor + tensor).to_dense(), 2 * dense)
    np.testing.assert_allclose((tensor - tensor).to_dense(), 0 * dense)
    np.testing.assert_allclose((-tensor).to_dense(), -dense)
    np.testing.assert_allclose((tensor * tensor).to_dense(), dense * dense)
    np.testing.assert_allclose((3 * tensor).to_dense(), (tensor * 3).to_dense())
    np.testing.assert_allclose((tensor / 2).to_dense(), dense / 2)
    np.testing.assert_allclose(
        ((tensor * tensor) / tensor).to_dense(), dense, atol=STRUCTURE_TOL
    )
    squared = tensor.map_blocks(lambda block: block * block)
    np.testing.assert_allclose(squared.to_dense(), dense * dense)
    # The dense form is a member: reading it back is exact.
    again = BlockTensor.from_dense(structure, dense)
    np.testing.assert_allclose(again.to_dense(), dense)


def test_mismatched_structures_and_dense_arrays_are_rejected() -> None:
    rng = np.random.default_rng(3)
    aux, occupied, virtual, ring = fitted_axes("RHF", False, rng)
    fitted = random_member(rng, TensorStructure((aux, occupied, virtual), ring=ring))
    other = random_member(
        rng, TensorStructure((aux, occupied.conjugate(), virtual), ring=ring)
    )
    with pytest.raises(StructureError, match="different structures"):
        _ = fitted + other
    with pytest.raises(StructureError, match="different structures"):
        _ = fitted * other
    with pytest.raises(StructureError, match="different structures"):
        _ = fitted / other
    dense = fitted.to_dense()
    noisy = dense + 0.1 * rng.standard_normal(dense.shape)
    with pytest.raises(StructureError, match="not a member"):
        BlockTensor.from_dense(fitted.structure, noisy)
    assert BlockTensor.from_dense(fitted.structure, noisy, check=False).n_axes == 3
    with pytest.raises(StructureError, match="has shape"):
        BlockTensor.from_dense(fitted.structure, dense[:-1])
    with pytest.raises(StructureError, match="distinct axes"):
        fitted.permute("Pii->iiP")
    with pytest.raises(StructureError, match="same axes"):
        fitted.permute("Pia->Pib")
    with pytest.raises(StructureError, match="one arrow"):
        fitted.permute("Pia")


def test_contraction_subscripts_are_validated() -> None:
    rng = np.random.default_rng(4)
    aux, occupied, virtual, ring = fitted_axes("UHF", False, rng)
    fitted = random_member(rng, TensorStructure((aux, occupied, virtual), ring=ring))
    with pytest.raises(StructureError, match="two operands"):
        contract("Pia,Pjb,ij->ab", fitted, fitted)
    with pytest.raises(StructureError, match="distinct letters"):
        contract("Pi,Pjb->ijb", fitted, fitted)
    with pytest.raises(StructureError, match="repeats"):
        contract("Pia,Qjb->iaia", fitted, fitted)
    with pytest.raises(StructureError, match="exactly one operand"):
        contract("Pia,Pjb->Piajb", fitted, fitted)
    with pytest.raises(StructureError, match="neither summed"):
        contract("Pia,Qjb->iajb", fitted, fitted)
    with pytest.raises(StructureError, match="conjugate one operand"):
        contract("Pia,Pjb->iajb", fitted, fitted)
    other = random_member(
        rng,
        TensorStructure(
            (TensorAxis(BlockStructure.spatial(), (5,)), occupied, virtual), ring=ring
        ),
    )
    with pytest.raises(StructureError, match="different sectors"):
        contract("Pia,Pbj->iajb", fitted, other.conj())
    with pytest.raises(StructureError, match="letters"):
        contract("Pia,P1b->iab", fitted, fitted)

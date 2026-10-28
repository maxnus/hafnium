"""Block tensors (ADR-0030): allowed and derived blocks, contractions, dense form."""

from collections.abc import Iterator

import numpy as np
import pytest

from hafnium.errors import StructureError
from hafnium.structured import (
    BlockStructure,
    BlockTensor,
    KAxis,
    Ring,
    Symmetry,
    TensorAxis,
    TensorStructure,
    contract,
)
from hafnium.symmetry import KPoints, SymmetrySpec
from tests.tolerances import STRUCTURE_TOL
from tests.unit.test_scf import CLASSES
from tests.unit.test_structure import sectors

SECTORS = sectors((2, 1, 1))
MESH = KPoints.mesh((3, 1, 1)).axis()
"""Γ and a pair of partners, so time reversal relates blocks."""
# (irrep sectors, k-points): the two are exclusive.
LABELLINGS = [(False, False), (True, False), (False, True)]


def structure_named(
    name: str, with_sectors: bool, over_k: bool = False
) -> BlockStructure:
    structure = SymmetrySpec.from_name(name).structure()
    if with_sectors:
        return structure.with_irreps(SECTORS)
    return structure.with_k_axis(MESH) if over_k else structure


def ring_of(*structures: BlockStructure) -> Ring:
    return (
        Ring.COMPLEX
        if any(
            structure.is_complex
            or (structure.k_axis is not None and not structure.k_axis.all_invariant)
            for structure in structures
        )
        else Ring.REAL
    )


def axis(
    structure: BlockStructure, sizes: Iterator[int], *, conjugated: bool = False
) -> TensorAxis:
    """An axis with the next sizes for the stored blocks, derived ones following."""
    n_kpt = 1 if structure.k_axis is None else structure.k_axis.n_kpt
    expanded: list[int] = []
    for block in structure.blocks:
        if block.is_stored:
            # One size per block: the orbital spaces of a state hold as
            # many orbitals at every point.
            expanded.extend([next(sizes)] * n_kpt)
        else:
            start = (block.derived_from or 0) * n_kpt
            expanded.extend(expanded[start : start + n_kpt])
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
    member = BlockTensor.from_dense(structure, dense, check=False)
    # A block a symmetry maps onto its own conjugate is real.
    blocks = {
        key: np.real(block) + 0.0 * block if structure.is_self_conjugate(key) else block
        for key, block in zip(structure.stored_keys, member.blocks, strict=True)
    }
    return BlockTensor.from_blocks(structure, blocks)


def fitted_axes(
    name: str, with_sectors: bool, rng: np.random.Generator, over_k: bool = False
) -> tuple[TensorAxis, TensorAxis, TensorAxis, Ring]:
    """Auxiliary (conjugated), occupied (conjugated) and virtual axes of a class.

    Over k-points the auxiliary axis runs over the momentum transfers as a
    real spatial structure, whose blocks at ``−q`` are the conjugates of
    those at ``q``: the random operands then carry the time-reversal
    relation that the products of a fitted tensor have whatever the gauge
    of its auxiliary index (ADR-0031), which the result assumes.
    """
    mo = structure_named(name, with_sectors, over_k)
    aux = BlockStructure.spatial()
    if with_sectors:
        aux = aux.with_irreps(SECTORS)
    if over_k:
        aux = aux.with_k_axis(MESH)
    sizes = random_sizes(rng)
    aux_sizes = [3, 2, 1] if with_sectors else [4, 4, 4] if over_k else [4]
    return (
        axis(aux, iter(aux_sizes), conjugated=True),
        axis(mo, sizes, conjugated=True),
        axis(mo, sizes),
        ring_of(mo, aux),
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
    with pytest.raises(StructureError, match="uniform mesh"):
        TensorAxis(unrestricted.with_k_axis(KAxis((0,), (1.0,))), (1, 1))
    over_k = TensorAxis(
        unrestricted.with_k_axis(KPoints.mesh((2, 1, 1)).axis()), (1, 2, 1, 2)
    )
    assert over_k.n_sectors == 4
    assert over_k.n_kpt == 2
    assert over_k.sector(1, 1) == 3
    assert (over_k.block_index(3), over_k.kpoint(3)) == (1, 1)
    assert over_k.momentum(1) == (2, 0, 0)
    assert over_k.modulus == (4, 2, 2)
    assert over_k.image(Symmetry.TIME_REVERSAL, 1) == 1
    with pytest.raises(StructureError, match="as many functions as its source"):
        TensorAxis(
            BlockStructure.restricted().with_k_axis(KPoints.mesh((2, 1, 1)).axis()),
            (1, 2, 1, 3),
        )
    with pytest.raises(StructureError, match="time reversal relates"):
        TensorAxis(BlockStructure.unrestricted().with_k_axis(MESH), (1, 2, 3, 1, 2, 3))
    occupied = TensorAxis(unrestricted, (2, 1), conjugated=True)
    assert occupied.extent == 3
    assert occupied.offset(1) == 2
    assert occupied.conjugate().conjugated is False
    assert occupied.same_space(occupied.conjugate())
    assert occupied.image(Symmetry.SPIN_FLIP, 0) is None
    assert occupied.image(Symmetry.TIME_REVERSAL, 1) == 1
    restricted = TensorAxis(BlockStructure.restricted(), (2, 2))
    assert restricted.image(Symmetry.SPIN_FLIP, 0) == 1


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
    assert structure.symmetries == (Symmetry.SPIN_FLIP,)
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
    assert conjugating.symmetries == (Symmetry.TIME_REVERSAL,)
    assert conjugating.orbit((0, 0)) == (((0, 0), False), ((1, 1), True))
    assert conjugating.derivation((1, 1)) == ((0, 0), True)
    assert TensorStructure(()).symmetries == ()
    assert (
        TensorStructure(
            (
                TensorAxis(BlockStructure.unrestricted(), (1, 1), conjugated=True),
                TensorAxis(BlockStructure.unrestricted(), (1, 1)),
            )
        ).symmetries
        == ()
    )


def test_structures_are_validated() -> None:
    restricted = TensorAxis(BlockStructure.restricted(), (1, 1))
    paired = TensorAxis(BlockStructure.paired_unrestricted(), (1, 1))
    complex_spatial = TensorAxis(BlockStructure.spatial(Ring.COMPLEX), (2,))
    # Time reversal is a symmetry of real and paired axes alike, the spin
    # flip of restricted ones only; a complex axis without partners leaves
    # no symmetry, so every block is stored.
    mixed = TensorStructure((restricted, paired.conjugate()), ring=Ring.COMPLEX)
    assert mixed.symmetries == (Symmetry.TIME_REVERSAL,)
    assert mixed.orbit((0, 1)) == (((0, 1), False), ((0, 0), True))
    assert (
        TensorStructure(
            (paired.conjugate(), paired, complex_spatial), ring=Ring.COMPLEX
        ).symmetries
        == ()
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


@pytest.mark.parametrize(("with_sectors", "over_k"), LABELLINGS)
@pytest.mark.parametrize("name", CLASSES)
def test_contraction_matches_the_dense_einsum(
    name: str, with_sectors: bool, over_k: bool
) -> None:
    rng = np.random.default_rng(hash((name, with_sectors, over_k)) % 2**32)
    aux, occupied, virtual, ring = fitted_axes(name, with_sectors, rng, over_k)
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
    if coulomb.structure.symmetries:
        assert coulomb.structure.n_stored < len(coulomb.structure.keys)
    if over_k:
        # Every stored block conserves crystal momentum, and time reversal
        # derives the blocks of the partner points of a real or paired state.
        axes = coulomb.structure.axes
        for key in coulomb.structure.keys:
            momenta = [axes[n].momentum(s) or (0,) for n, s in enumerate(key)]
            total = [
                sum(m[d] for m in momenta[0::2]) - sum(m[d] for m in momenta[1::2])
                for d in range(3)
            ]
            assert all(t % n == 0 for t, n in zip(total, (6, 2, 2), strict=True))
        # The mesh holds Γ, where a quaternionic block cannot derive by
        # time reversal.
        reversible = "complex" not in name and name != "paired GHF"
        assert (Symmetry.TIME_REVERSAL in coulomb.structure.symmetries) == reversible
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


@pytest.mark.parametrize(("with_sectors", "over_k"), LABELLINGS)
@pytest.mark.parametrize("name", CLASSES)
def test_permutation_sum_and_algebra_match_the_dense_form(
    name: str, with_sectors: bool, over_k: bool
) -> None:
    rng = np.random.default_rng(hash((name, with_sectors, over_k, 1)) % 2**32)
    _, occupied, virtual, ring = fitted_axes(name, with_sectors, rng, over_k)
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
            (
                TensorAxis(BlockStructure.spatial(), (5,), conjugated=True),
                occupied,
                virtual,
            ),
            ring=ring,
        ),
    )
    with pytest.raises(StructureError, match="different sectors"):
        contract("Pia,Pbj->iajb", fitted, other.conj())
    with pytest.raises(StructureError, match="letters"):
        contract("Pia,P1b->iab", fitted, fitted)


def test_time_reversal_and_the_spin_flip_combine_over_k() -> None:
    restricted = BlockStructure.restricted().with_k_axis(MESH)
    bra = TensorAxis(restricted, (1,) * 6, conjugated=True)
    ket = TensorAxis(restricted, (1,) * 6)
    structure = TensorStructure((bra, ket), ring=Ring.COMPLEX)
    assert structure.symmetries == (Symmetry.SPIN_FLIP, Symmetry.TIME_REVERSAL)
    # Sectors: (alpha, Γ), (alpha, 1/3), (alpha, 2/3), (beta, Γ), ... The
    # block (alpha 1/3, alpha 1/3) has an orbit of four: its copy in beta and
    # the conjugates at 2/3.
    assert structure.orbit((1, 1)) == (
        ((1, 1), False),
        ((4, 4), False),
        ((2, 2), True),
        ((5, 5), True),
    )
    assert structure.derivation((5, 5)) == ((1, 1), True)
    # The Γ block is fixed by time reversal: an orbit of two.
    assert structure.orbit((0, 0)) == (((0, 0), False), ((3, 3), False))
    assert structure.stored_keys == ((0, 0), (1, 1))
    # Only pairs of one point are allowed by momentum, alpha with alpha by spin.
    assert len(structure.keys) == 6
    with pytest.raises(StructureError, match="one mesh"):
        TensorStructure(
            (
                bra,
                TensorAxis(
                    BlockStructure.restricted().with_k_axis(
                        KPoints.mesh((2, 1, 1)).axis()
                    ),
                    (1,) * 4,
                ),
            ),
            ring=Ring.COMPLEX,
        )
    with pytest.raises(StructureError, match="k-points, or none"):
        TensorStructure(
            (bra, TensorAxis(BlockStructure.restricted(), (1, 1))), ring=Ring.COMPLEX
        )
    with pytest.raises(StructureError, match=r"ring=Ring\.COMPLEX"):
        TensorStructure((bra, ket))
    # The Γ block is mapped onto its own conjugate: it must be real.
    assert structure.is_self_conjugate((0, 0))
    assert not structure.is_self_conjugate((1, 1))
    member = random_member(np.random.default_rng(7), structure)
    assert float(np.max(np.imag(member.block((0, 0))))) == 0.0
    blocks = dict(zip(structure.stored_keys, member.blocks, strict=True))
    blocks[(0, 0)] = blocks[(0, 0)] + 1e-3j
    with pytest.raises(StructureError, match="must be real"):
        BlockTensor.from_dense(
            structure, BlockTensor.from_blocks(structure, blocks).to_dense()
        )
    # A quaternionic block derives by time reversal only on a mesh without
    # invariant points, since its orbitals there map onto Kramers partners.
    paired = BlockStructure.paired_generalized()
    with_gamma = TensorAxis(paired.with_k_axis(MESH), (2,) * 3)
    assert TensorStructure((with_gamma,), ring=Ring.COMPLEX).symmetries == ()
    shifted = TensorAxis(
        paired.with_k_axis(KPoints.mesh((2,), shift=(0.5,)).axis()), (2,) * 2
    )
    assert TensorStructure((shifted,), ring=Ring.COMPLEX).symmetries == (
        Symmetry.TIME_REVERSAL,
    )

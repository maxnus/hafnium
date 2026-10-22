"""The k axis of block structures: metadata, storage rules and validation."""

import numpy as np
import pytest

from hafnium.errors import InputError, StructureError
from hafnium.scf import AufbauOccupation
from hafnium.structured import (
    BlockMatrix,
    BlockStructure,
    BlockVector,
    KAxis,
    Ring,
)
from hafnium.symmetry import KPoints, SymmetrySpec
from tests.random_matrices import random_hermitian
from tests.tolerances import EIGEN_TOL, STRUCTURE_TOL

CHAIN = KPoints.mesh((3,)).axis()
"""Γ and one pair: partners (0, 2, 1)."""
INVARIANT = KPoints.mesh((2, 2)).axis()
"""Four time-reversal invariant points."""
SHIFTED = KPoints.mesh((2, 2), shift=(0.5, 0.5)).axis()
"""Two pairs, no invariant point."""


def test_axis_metadata_follows_the_partner_map() -> None:
    assert CHAIN.partners == (0, 2, 1)
    assert CHAIN.n_kpt == 3
    assert CHAIN.invariant == (True, False, False)
    assert CHAIN.representatives == (0, 1)
    assert CHAIN.stored(half_zone=True) == (0, 1)
    assert CHAIN.stored(half_zone=False) == (0, 1, 2)
    own, partner = CHAIN.stored_weights(half_zone=True)
    assert own == pytest.approx((1 / 3, 1 / 3))
    assert partner == pytest.approx((0.0, 1 / 3))
    own, partner = CHAIN.stored_weights(half_zone=False)
    assert own == pytest.approx((1 / 3,) * 3)
    assert partner == (0.0, 0.0, 0.0)
    assert INVARIANT.all_invariant
    assert INVARIANT.representatives == (0, 1, 2, 3)
    assert not any(SHIFTED.invariant)
    assert len(SHIFTED.representatives) == 2
    assert KAxis.gamma() == KPoints.gamma().axis()


def test_malformed_axes_are_rejected() -> None:
    with pytest.raises(StructureError, match="at least one"):
        KAxis((), ())
    with pytest.raises(StructureError, match="weights, got"):
        KAxis((0,), (0.5, 0.5))
    with pytest.raises(StructureError, match="own inverse"):
        KAxis((1, 2, 0), (1 / 3,) * 3)
    with pytest.raises(StructureError, match="own inverse"):
        KAxis((0, 3), (0.5, 0.5))
    with pytest.raises(StructureError, match="sum to one"):
        KAxis((0, 1), (0.5, 0.6))
    with pytest.raises(StructureError, match="positive"):
        KAxis((0, 1), (1.0, 0.0))
    with pytest.raises(StructureError, match="equal weights"):
        KAxis((1, 0), (0.25, 0.75))
    with pytest.raises(InputError, match="no time-reversal partner"):
        KPoints.from_fractional([(0.0,), (0.3,)]).axis()


def test_structures_carry_the_axis() -> None:
    restricted = BlockStructure.restricted().with_k_axis(CHAIN)
    assert restricted.k_axis == CHAIN
    assert restricted != BlockStructure.restricted()
    assert restricted.describe() == "restricted (real) at 3 k-points"
    assert BlockStructure.spatial().with_k_axis(KAxis.gamma()).describe() == (
        "spatial (real) at 1 k-point"
    )
    assert restricted.with_ring(Ring.COMPLEX).k_axis == CHAIN
    assert restricted.with_k_axis(None) == BlockStructure.restricted()
    (alpha,) = restricted.stored_blocks
    assert restricted.stored_kpoints(alpha) == (0, 1)
    assert restricted.invariant_positions(alpha) == (0,)
    assert restricted.all_kpoints == (0, 1, 2)
    assert BlockStructure.restricted().stored_kpoints(alpha) is None
    assert BlockStructure.restricted().invariant_positions(alpha) == (0,)


@pytest.mark.parametrize(
    ("structure", "axis", "stored", "ring"),
    [
        (BlockStructure.restricted(), CHAIN, (0, 1), Ring.COMPLEX),
        (BlockStructure.restricted(), INVARIANT, (0, 1, 2, 3), Ring.REAL),
        (BlockStructure.restricted(), KAxis.gamma(), (0,), Ring.REAL),
        (BlockStructure.restricted(Ring.COMPLEX), CHAIN, (0, 1, 2), Ring.COMPLEX),
        (BlockStructure.paired_unrestricted(), SHIFTED, (0, 1, 2, 3), Ring.COMPLEX),
        (BlockStructure.paired_generalized(), SHIFTED, (0, 1), Ring.QUATERNION),
        (BlockStructure.generalized(), SHIFTED, (0, 1), Ring.COMPLEX),
    ],
)
def test_storage_follows_the_ring(
    structure: BlockStructure, axis: KAxis, stored: tuple[int, ...], ring: Ring
) -> None:
    with_axis = structure.with_k_axis(axis)
    block = with_axis.stored_blocks[0]
    assert with_axis.stored_kpoints(block) == stored
    assert with_axis.storage_ring(block) is ring
    zeros = BlockMatrix.zeros(with_axis, (3, 2))
    factor = 2 if with_axis.is_spinor else 1
    assert zeros.shapes[0] == (len(stored), 3 * factor, 2 * factor)
    assert zeros.blocks[0].dtype == (np.float64 if ring is Ring.REAL else np.complex128)
    identity = BlockMatrix.identity(with_axis, 3)
    assert identity.shapes[0] == (len(stored), 3 * factor, 3 * factor)
    assert np.allclose(identity.blocks[0][-1], np.eye(3 * factor))


def test_symmetry_spec_attaches_the_axis_of_a_k_point_set() -> None:
    mesh = KPoints.mesh((3,))
    structure = SymmetrySpec.from_name("paired GHF").structure(mesh)
    assert structure.k_axis == mesh.axis()
    assert (
        structure.with_k_axis(None) == SymmetrySpec.from_name("paired GHF").structure()
    )
    assert SymmetrySpec.from_name("UHF").structure().k_axis is None


def test_block_shapes_and_dtypes_are_validated_with_an_axis() -> None:
    restricted = BlockStructure.restricted().with_k_axis(CHAIN)
    with pytest.raises(StructureError, match=r"\[kpt, n, m\]"):
        BlockMatrix(restricted, (np.zeros((2, 2), dtype=complex),))
    with pytest.raises(StructureError, match="stores 2 k-points, got 3"):
        BlockMatrix(restricted, (np.zeros((3, 2, 2), dtype=complex),))
    with pytest.raises(StructureError, match="complex away from"):
        BlockMatrix(restricted, (np.zeros((2, 2, 2)),))
    invariant = BlockStructure.restricted().with_k_axis(INVARIANT)
    with pytest.raises(StructureError, match="complex entries"):
        BlockMatrix(invariant, (np.zeros((4, 2, 2), dtype=complex),))
    with pytest.raises(StructureError, match="two-dimensional"):
        BlockMatrix(BlockStructure.restricted(), (np.zeros((1, 2, 2)),))
    with pytest.raises(StructureError, match=r"\[kpt, mo\]"):
        BlockVector(restricted, (np.zeros(2),))
    with pytest.raises(StructureError, match="stores 2 k-points"):
        BlockVector(restricted, (np.zeros((3, 2)),))


def test_embedding_and_dense_forms_keep_the_k_point_set() -> None:
    rng = np.random.default_rng(7)
    restricted = BlockStructure.restricted().with_k_axis(CHAIN)
    dense = np.stack([random_hermitian(rng, 4) for _ in range(3)])
    matrix = BlockMatrix.from_dense(restricted, dense, check=False)
    with pytest.raises(StructureError, match="different k axes"):
        matrix.embed(BlockStructure.unrestricted().with_k_axis(SHIFTED))
    with pytest.raises(StructureError, match="different k axes"):
        matrix.project(BlockStructure.restricted())
    with pytest.raises(StructureError, match="holds 2 k-points, the structure 3"):
        BlockMatrix.from_dense(restricted, dense[:2], check=False)
    with pytest.raises(StructureError, match="three-dimensional"):
        BlockMatrix.from_dense(restricted, dense[0], check=False)
    assert matrix.to_dense().shape == (3, 4, 4)
    assert matrix.materialized()[1].shape == (3, 2, 2)
    assert np.allclose(matrix.materialized()[0][2], matrix.blocks[0][1].conj())


def test_occupation_policies_reject_k_axes_for_now() -> None:
    restricted = BlockStructure.restricted().with_k_axis(CHAIN)
    energies = BlockVector(restricted, (np.zeros((2, 3)),))
    with pytest.raises(InputError, match="k-points"):
        AufbauOccupation(2, 2).occupy(energies)


@pytest.mark.parametrize(
    "structure",
    [
        BlockStructure.restricted(),
        BlockStructure.unrestricted(),
        BlockStructure.paired_unrestricted(),
        BlockStructure.generalized(),
        BlockStructure.paired_generalized(),
    ],
)
def test_the_gamma_axis_reproduces_the_molecular_structure(
    structure: BlockStructure,
) -> None:
    rng = np.random.default_rng(8)
    n = 3
    molecular = BlockMatrix.from_dense(
        structure, random_hermitian(rng, 2 * n), check=False
    )
    at_gamma = BlockMatrix.from_dense(
        structure.with_k_axis(KAxis.gamma()),
        molecular.to_dense()[None],
        check=True,
    )
    for molecular_block, gamma_block in zip(
        molecular.blocks, at_gamma.blocks, strict=True
    ):
        assert gamma_block.dtype == molecular_block.dtype
        assert np.allclose(gamma_block[0], molecular_block, atol=STRUCTURE_TOL)
    assert np.allclose(at_gamma.trace(), molecular.trace(), atol=STRUCTURE_TOL)
    assert np.allclose(
        at_gamma.trace_product(at_gamma),
        molecular.trace_product(molecular),
        atol=STRUCTURE_TOL,
    )
    values, _ = molecular.eigh()
    values_gamma, _ = at_gamma.eigh()
    for w, w_gamma in zip(values.blocks, values_gamma.blocks, strict=True):
        assert np.allclose(w_gamma[0], w, atol=EIGEN_TOL)
    assert np.allclose(
        at_gamma.spin_trace().blocks[0][0],
        molecular.spin_trace().blocks[0],
        atol=STRUCTURE_TOL,
    )

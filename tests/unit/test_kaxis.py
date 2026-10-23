"""The k axis of block structures: metadata, storage rules and validation."""

import itertools

import numpy as np
import pytest

from hafnium.errors import InputError, StructureError
from hafnium.scf import AufbauOccupation
from hafnium.structured import (
    BlockMatrix,
    BlockStructure,
    BlockVector,
    KAxis,
    KDerivation,
    KPointOperations,
    Ring,
)
from hafnium.symmetry import KPoints, SymmetrySpec
from tests.random_matrices import random_hermitian
from tests.tolerances import EIGEN_TOL, GEOMETRY_TOL, STRUCTURE_TOL

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
    structure = SymmetrySpec.from_name("paired GHF").with_kpoints(mesh).structure()
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


def involution(
    mesh: KPoints, rotation: np.ndarray, n: int, seed: int
) -> KPointOperations:
    """An operation of order two on ``mesh``, represented as a crystal's would be.

    ``rotation`` is an integer matrix squaring to one that acts on the
    fractional coordinates, ``k' = k W⁻¹``. The matrix reverses the order
    of the ``n`` functions, and the phases are those of integer lattice
    shifts ``t_i`` of the functions, ``exp(−2πi k'·t_i)`` with ``k'`` the
    image point, with ``t_{n−1−i} = −Wᵀ t_i`` so that ``U(gk) U(k) = 1``
    and ``U(−k) = conj(U(k))`` hold as they do for a space group.
    """
    rng = np.random.default_rng(seed)
    fractional = np.asarray(mesh.fractional)
    d = fractional.shape[1]
    moved = (fractional @ np.linalg.inv(rotation)) % 1.0
    images: list[int] = []
    for k in range(mesh.n_kpt):
        distance = np.abs((fractional - moved[k] + 0.5) % 1.0 - 0.5).max(axis=1)
        assert distance.min() < GEOMETRY_TOL
        images.append(int(np.argmin(distance)))
    shifts = np.zeros((n, d), dtype=int)
    for i in range(n // 2):
        shifts[i] = rng.integers(-1, 2, size=d)
        shifts[n - 1 - i] = -rotation.T @ shifts[i]
    if n % 2:
        # The middle function maps onto itself: its shift satisfies
        # −Wᵀ t = t, the first such vector of small integers, or zero.
        fixed = [
            vector
            for vector in itertools.product((-1, 0, 1), repeat=d)
            if any(vector)
            and bool(np.all(-rotation.T @ np.asarray(vector) == np.asarray(vector)))
        ]
        shifts[n // 2] = fixed[0] if fixed else 0
    phases = np.exp(-2j * np.pi * (fractional[images] @ shifts.T))
    return KPointOperations(
        "involution",
        (tuple(range(mesh.n_kpt)), tuple(images)),
        (np.eye(n), np.eye(n)[::-1]),
        np.stack([np.ones((mesh.n_kpt, n), dtype=complex), phases]),
    )


SWAP = np.array([[0, 1], [1, 0]])
"""The exchange of the two directions of a square mesh."""


def swap_operations(n: int) -> KPointOperations:
    """The 2×2 mesh with the operation exchanging its second and third points."""
    return involution(KPoints.mesh((2, 2)), SWAP, n, n)


def test_point_operations_reduce_the_stored_points() -> None:
    operations = swap_operations(2)
    axis = KAxis(INVARIANT.partners, INVARIANT.weights, operations=operations)
    assert axis.operations is operations
    assert axis.images == operations.images
    assert axis.stored(half_zone=True) == (0, 1, 3)
    assert axis.stored(half_zone=False) == (0, 1, 3)
    assert axis.derivation(2, half_zone=False) == KDerivation(1, 1, False)
    assert axis.derivation(1, half_zone=True).is_stored
    assert [d.source for d in axis.derivations(half_zone=True)] == [0, 1, 1, 3]
    own, partner = axis.stored_weights(half_zone=True)
    assert own == pytest.approx((0.25, 0.5, 0.25))
    assert partner == (0.0, 0.0, 0.0)
    assert axis.same_points(INVARIANT)
    assert axis != INVARIANT
    assert not INVARIANT.same_points(SHIFTED)
    assert hash(axis) != hash(INVARIANT)
    # An inversion on the chain reaches the partner by rotation, so nothing is
    # conjugated: the complex ring stores the same points as the real ring.
    inversion = involution(KPoints.mesh((3,)), -np.eye(1, dtype=int), 2, 0)
    chain = KAxis(CHAIN.partners, CHAIN.weights, operations=inversion)
    assert chain.stored(half_zone=False) == (0, 1)
    assert chain.derivation(2, half_zone=True) == KDerivation(1, 1, False)
    own, partner = chain.stored_weights(half_zone=True)
    assert own == pytest.approx((1 / 3, 2 / 3))
    assert partner == (0.0, 0.0)
    assert KAxis(CHAIN.partners, CHAIN.weights).derivation(2, half_zone=True) == (
        KDerivation(1, 0, True)
    )
    with pytest.raises(StructureError, match="act on 4 points"):
        KAxis(CHAIN.partners, CHAIN.weights, operations=operations)
    with pytest.raises(StructureError, match="equal weights"):
        KAxis((0, 1, 2, 3), (0.1, 0.2, 0.3, 0.4), operations=operations)


def test_point_operations_rotate_blocks_and_are_validated() -> None:
    operations = swap_operations(3)
    u = np.asarray(operations.unitary(1, 1))
    phases = np.asarray(operations.phases[1, 1])
    np.testing.assert_allclose(u, np.asarray(operations.matrices[1]) * phases)
    # The representation property on the pair (1, 2) and at the fixed point 0.
    for k in (0, 1, 2, 3):
        image = operations.images[1][k]
        both = np.asarray(operations.unitary(1, image)) @ np.asarray(
            operations.unitary(1, k)
        )
        np.testing.assert_allclose(both, np.eye(3), atol=STRUCTURE_TOL)
    rng = np.random.default_rng(2)
    x = random_hermitian(rng, 3)
    rotated = operations.rotate(1, 1, x, operator=True)
    np.testing.assert_allclose(rotated, u @ x @ u.conj().T, atol=STRUCTURE_TOL)
    np.testing.assert_allclose(
        operations.unrotate(1, 1, rotated, operator=True), x, atol=STRUCTURE_TOL
    )
    columns = rng.standard_normal((3, 2))
    np.testing.assert_allclose(
        operations.rotate(1, 1, columns, operator=False),
        u @ columns,
        atol=STRUCTURE_TOL,
    )
    spinor = random_hermitian(rng, 6)
    both = np.kron(np.eye(2), u)
    np.testing.assert_allclose(
        operations.rotate(1, 1, spinor, operator=True),
        both @ spinor @ both.conj().T,
        atol=STRUCTURE_TOL,
    )
    with pytest.raises(StructureError, match="does not act on the 3 functions"):
        operations.rotate(1, 1, np.eye(4), operator=True)


def test_point_operations_compare_by_value_and_are_validated() -> None:
    operations = swap_operations(3)
    assert operations == swap_operations(3)
    assert operations != swap_operations(2)
    assert hash(operations) == hash(swap_operations(3))
    eye, matrix = operations.matrices
    phases = np.asarray(operations.phases)
    with pytest.raises(StructureError, match="one image row and one matrix"):
        KPointOperations("x", (), (), phases)
    with pytest.raises(StructureError, match="first operation is the identity"):
        KPointOperations("x", ((0, 2, 1, 3), (0, 1, 2, 3)), (eye, matrix), phases)
    with pytest.raises(StructureError, match="permutes"):
        KPointOperations("x", ((0, 1, 2, 3), (0, 0, 1, 3)), (eye, matrix), phases)
    with pytest.raises(StructureError, match="closed under composition"):
        KPointOperations("x", ((0, 1, 2, 3), (1, 2, 3, 0)), (eye, matrix), phases)
    with pytest.raises(StructureError, match="orthogonal"):
        KPointOperations("x", operations.images, (eye, 2.0 * matrix), phases)
    with pytest.raises(StructureError, match="identity matrix"):
        KPointOperations("x", operations.images, (matrix, matrix), phases)
    with pytest.raises(StructureError, match=r"\[operation, kpt, ao\]"):
        KPointOperations("x", operations.images, (eye, matrix), phases[:, :2])
    with pytest.raises(StructureError, match="unit modulus"):
        KPointOperations("x", operations.images, (eye, matrix), 2.0 * phases)
    with pytest.raises(StructureError, match="unit phases"):
        KPointOperations("x", operations.images, (eye, matrix), -phases)


def test_rotated_axes_derive_and_project_consistently() -> None:
    operations = swap_operations(2)
    axis = KAxis(INVARIANT.partners, INVARIANT.weights, operations=operations)
    structure = BlockStructure.unrestricted().with_k_axis(axis)
    rng = np.random.default_rng(4)
    dense = np.stack([random_hermitian(rng, 4) for _ in range(4)])
    matrix = BlockMatrix.from_dense(structure, dense, check=False)
    assert matrix.blocks[0].shape == (3, 2, 2)
    expanded = matrix.to_dense()
    u = np.asarray(operations.unitary(1, 1))
    both = np.kron(np.eye(2), u)
    # The point 2 is the image of point 1 under the operation.
    np.testing.assert_allclose(
        expanded[2], both @ expanded[1] @ both.conj().T, atol=STRUCTURE_TOL
    )
    again = BlockMatrix.from_dense(structure, expanded, check=True)
    for block, other in zip(matrix.blocks, again.blocks, strict=True):
        np.testing.assert_allclose(block, other, atol=STRUCTURE_TOL)
    with pytest.raises(StructureError, match="does not have the"):
        BlockMatrix.from_dense(structure, dense, check=True)
    # The same k-points without operations: embedding goes through the
    # dense form and is lossless, projection returns.
    plain = BlockStructure.unrestricted().with_k_axis(INVARIANT)
    embedded = matrix.embed(plain)
    assert embedded.blocks[0].shape == (4, 2, 2)
    np.testing.assert_allclose(embedded.to_dense(), expanded, atol=STRUCTURE_TOL)
    projected = embedded.project(structure)
    for block, other in zip(matrix.blocks, projected.blocks, strict=True):
        np.testing.assert_allclose(block, other, atol=STRUCTURE_TOL)
    with pytest.raises(StructureError, match="different k axes"):
        matrix.embed(BlockStructure.unrestricted().with_k_axis(CHAIN))
    spatial = BlockMatrix.from_spatial(dense[:, :2, :2], axis, check=False)
    assert spatial.blocks[0].shape == (3, 2, 2)
    with pytest.raises(StructureError, match="point operations"):
        BlockMatrix.from_spatial(dense[:, :2, :2], axis)
    np.testing.assert_allclose(
        BlockMatrix.from_spatial(spatial.materialized()[0], axis).blocks[0],
        spatial.blocks[0],
        atol=STRUCTURE_TOL,
    )

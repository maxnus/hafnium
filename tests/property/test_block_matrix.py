"""Property tests of BlockMatrix over all symmetry structures (PLAN.md §8).

Every test runs over the molecular structures and over structures with a
k axis: the Γ point alone, an entirely time-reversal invariant mesh, a
chain mesh with Γ and one pair, a shifted mesh with pairs only, and two
axes with point operations (ADR-0024): the invariant mesh with the
operation swapping two of its points, and the chain mesh with an
inversion that maps the pair onto itself, each an operation of order two
represented by the reversal of the functions with random Bloch phases.
With an axis the dense form is ``[kpt, 2n, m]`` and a trace is the
k-weighted sum of the traces at every point.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hafnium.backend import adjoint
from hafnium.errors import StructureError
from hafnium.structured import (
    BlockMatrix,
    BlockStructure,
    BlockVector,
    IrrepSectors,
    KAxis,
    Ring,
    Spin,
    density_matrix,
    is_kramers_symmetric,
    time_reverse,
)
from hafnium.symmetry import KPoints
from tests.random_matrices import random_hermitian, random_positive_definite
from tests.tolerances import EIGEN_TOL, STRUCTURE_TOL
from tests.unit.test_kaxis import SWAP, involution

STRUCTURES: dict[str, BlockStructure] = {
    "spatial-real": BlockStructure.spatial(Ring.REAL),
    "spatial-complex": BlockStructure.spatial(Ring.COMPLEX),
    "restricted-real": BlockStructure.restricted(Ring.REAL),
    "restricted-complex": BlockStructure.restricted(Ring.COMPLEX),
    "unrestricted-real": BlockStructure.unrestricted(Ring.REAL),
    "unrestricted-complex": BlockStructure.unrestricted(Ring.COMPLEX),
    "paired-unrestricted": BlockStructure.paired_unrestricted(),
    "generalized-real": BlockStructure.generalized(Ring.REAL),
    "generalized-complex": BlockStructure.generalized(Ring.COMPLEX),
    "paired-generalized": BlockStructure.paired_generalized(),
}

# (lower symmetry class, higher symmetry class): every member of the first is
# representable in the second. Follows the Fukutome hierarchy.
EMBEDDINGS: list[tuple[str, str]] = [
    ("spatial-real", "spatial-complex"),
    ("spatial-real", "restricted-real"),
    ("spatial-real", "unrestricted-real"),
    ("spatial-real", "generalized-real"),
    ("spatial-real", "paired-unrestricted"),
    ("spatial-real", "paired-generalized"),
    ("spatial-complex", "restricted-complex"),
    ("restricted-real", "restricted-complex"),
    ("restricted-real", "unrestricted-real"),
    ("restricted-real", "paired-unrestricted"),
    ("restricted-real", "generalized-real"),
    ("restricted-complex", "unrestricted-complex"),
    ("restricted-complex", "generalized-complex"),
    ("unrestricted-real", "unrestricted-complex"),
    ("unrestricted-real", "generalized-real"),
    ("unrestricted-complex", "generalized-complex"),
    ("paired-unrestricted", "unrestricted-complex"),
    ("paired-unrestricted", "paired-generalized"),
    ("generalized-real", "generalized-complex"),
    ("paired-generalized", "generalized-complex"),
]

PAIRED = ["paired-unrestricted", "paired-generalized"]

# Half-zone structures and the complex structure that stores every k-point
# of the same members.
FULL_ZONE: dict[str, str] = {
    "spatial-real": "spatial-complex",
    "restricted-real": "restricted-complex",
    "unrestricted-real": "unrestricted-complex",
    "paired-unrestricted": "unrestricted-complex",
    "generalized-real": "generalized-complex",
    "paired-generalized": "generalized-complex",
}

PLAIN_AXES: dict[str, KAxis | None] = {
    "molecular": None,
    "gamma": KAxis.gamma(),
    "invariant-mesh": KPoints.mesh((2, 2)).axis(),
    "chain-mesh": KPoints.mesh((3,)).axis(),
    "shifted-mesh": KPoints.mesh((2, 2), shift=(0.5, 0.5)).axis(),
}
# (plain axis, its mesh, the integer matrix of the operation besides the identity)
ROTATED_AXES: dict[str, tuple[str, KPoints, np.ndarray]] = {
    "rotated-mesh": ("invariant-mesh", KPoints.mesh((2, 2)), SWAP),
    "inverted-chain": ("chain-mesh", KPoints.mesh((3,)), -np.eye(1, dtype=int)),
}
AXES = sorted(PLAIN_AXES) + sorted(ROTATED_AXES)

seeds = st.integers(min_value=0, max_value=2**31 - 1)
sizes = st.integers(min_value=1, max_value=4)
names = st.sampled_from(sorted(STRUCTURES))
axes = st.sampled_from(AXES)
meshes = st.sampled_from(
    ["chain-mesh", "shifted-mesh", "rotated-mesh", "inverted-chain"]
)


def k_axis_for(axis: str, n: int) -> KAxis | None:
    """The axis ``axis`` for ``n`` spatial functions.

    A rotated axis carries an operation of order two represented as a
    crystal's would be, with lattice shifts fixed by ``n``.
    """
    if axis in PLAIN_AXES:
        return PLAIN_AXES[axis]
    plain_name, mesh, rotation = ROTATED_AXES[axis]
    plain = PLAIN_AXES[plain_name]
    assert plain is not None
    operations = involution(mesh, rotation, n, 1000 + n)
    return KAxis(plain.partners, plain.weights, operations=operations)


def structure(name: str, axis: str, n: int) -> BlockStructure:
    return STRUCTURES[name].with_k_axis(k_axis_for(axis, n))


def random_dense(rng: np.random.Generator, size: int, axis: str) -> np.ndarray:
    """A random Hermitian spin-orbital matrix, one per k-point of ``axis``."""
    k_axis = k_axis_for(axis, 1)
    if k_axis is None:
        return random_hermitian(rng, size)
    return np.stack([random_hermitian(rng, size) for _ in range(k_axis.n_kpt)])


def member(name: str, n: int, rng: np.random.Generator, axis: str) -> BlockMatrix:
    """A random Hermitian member of the structure, obtained by projection."""
    return BlockMatrix.from_dense(
        structure(name, axis, n), random_dense(rng, 2 * n, axis), check=False
    )


def overlap_member(n: int, rng: np.random.Generator, axis: str) -> BlockMatrix:
    """A positive-definite spatial matrix with the symmetry of ``axis``."""
    k_axis = k_axis_for(axis, n)
    count = 1 if k_axis is None else k_axis.n_kpt
    blocks: list[np.ndarray] = []
    for _ in range(count):
        x = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        s = x @ x.conj().T + n * np.eye(n)
        blocks.append(np.block([[s, np.zeros((n, n))], [np.zeros((n, n)), s]]))
    dense = blocks[0] if k_axis is None else np.stack(blocks)
    return BlockMatrix.from_dense(
        structure("spatial-real", axis, n), dense, check=False
    )


def each(array: np.ndarray, batched: bool) -> tuple[np.ndarray, ...]:
    """The matrices (or vectors) of a block, one per stored k-point."""
    return tuple(array) if batched else (array,)


def weighted(axis: str, values: np.ndarray) -> np.ndarray:
    """Sum per-k values with the weights of ``axis`` (identity without one)."""
    k_axis = k_axis_for(axis, 1)
    if k_axis is None:
        return values
    return np.sum(np.asarray(k_axis.weights) * values, axis=0)


def dense_trace(axis: str, dense: np.ndarray) -> np.ndarray:
    return weighted(axis, np.einsum("...ii->...", dense))


def close(a: BlockMatrix, b: BlockMatrix, tol: float = STRUCTURE_TOL) -> bool:
    return a.structure == b.structure and all(
        np.allclose(x, y, atol=tol, rtol=0)
        for x, y in zip(a.blocks, b.blocks, strict=True)
    )


@settings(max_examples=100, deadline=None)
@given(seed=seeds, n=sizes, name=names, axis=axes)
def test_dense_round_trip_is_lossless(seed: int, n: int, name: str, axis: str) -> None:
    a = member(name, n, np.random.default_rng(seed), axis)
    again = BlockMatrix.from_dense(a.structure, a.to_dense(), check=True)
    assert close(a, again)


@settings(max_examples=100, deadline=None)
@given(seed=seeds, n=sizes, name=names, axis=axes)
def test_block_algebra_matches_dense_algebra(
    seed: int, n: int, name: str, axis: str
) -> None:
    rng = np.random.default_rng(seed)
    a, b = member(name, n, rng, axis), member(name, n, rng, axis)
    ad, bd = a.to_dense(), b.to_dense()
    assert np.allclose((a @ b).to_dense(), ad @ bd, atol=STRUCTURE_TOL)
    assert np.allclose((a + b).to_dense(), ad + bd, atol=STRUCTURE_TOL)
    assert np.allclose((a - 2 * b).to_dense(), ad - 2 * bd, atol=STRUCTURE_TOL)
    assert np.allclose(a.adjoint().to_dense(), adjoint(ad), atol=STRUCTURE_TOL)
    if axis in ROTATED_AXES:
        # The conjugate of a rotated point carries the phases of −k, so the
        # elementwise conjugate is not a member of the structure.
        with pytest.raises(StructureError, match="not a member"):
            a.conj()
        with pytest.raises(StructureError, match="not a member"):
            a.transpose()
    else:
        assert np.allclose(a.conj().to_dense(), ad.conj(), atol=STRUCTURE_TOL)
        assert np.allclose(
            a.transpose().to_dense(),
            np.swapaxes(ad, -1, -2),
            atol=STRUCTURE_TOL,
        )
    assert np.allclose(a.trace(), dense_trace(axis, ad), atol=STRUCTURE_TOL)
    assert np.allclose(
        a.trace_product(b), dense_trace(axis, ad @ bd), atol=STRUCTURE_TOL
    )


@settings(max_examples=100, deadline=None)
@given(seed=seeds, n=sizes, pair=st.sampled_from(EMBEDDINGS), axis=axes)
def test_embedding_is_lossless_and_projection_inverts_it(
    seed: int, n: int, pair: tuple[str, str], axis: str
) -> None:
    lower, higher = pair
    a = member(lower, n, np.random.default_rng(seed), axis)
    embedded = a.embed(structure(higher, axis, n))
    assert np.allclose(embedded.to_dense(), a.to_dense(), atol=STRUCTURE_TOL)
    assert close(embedded.project(structure(lower, axis, n)), a)


@settings(max_examples=60, deadline=None)
@given(seed=seeds, n=sizes, name=names, axis=axes)
def test_projection_is_idempotent(seed: int, n: int, name: str, axis: str) -> None:
    a = member(name, n, np.random.default_rng(seed), axis)
    assert close(a.project(a.structure), a)


@settings(max_examples=100, deadline=None)
@given(seed=seeds, n=sizes, name=names, axis=axes)
def test_eigh_diagonalizes_every_block(seed: int, n: int, name: str, axis: str) -> None:
    f = member(name, n, np.random.default_rng(seed), axis)
    batched = k_axis_for(axis, 1) is not None
    values, vectors = f.eigh()
    for fb, wb, cb in zip(f.blocks, values.blocks, vectors.blocks, strict=True):
        for f_k, w, c in zip(
            each(fb, batched), each(wb, batched), each(cb, batched), strict=True
        ):
            assert np.allclose(f_k @ c, c * w[None, :], atol=EIGEN_TOL)
            assert np.allclose(adjoint(c) @ c, np.eye(c.shape[1]), atol=EIGEN_TOL)
            assert np.all(np.diff(w) >= -EIGEN_TOL)


@settings(max_examples=100, deadline=None)
@given(seed=seeds, n=sizes, name=names, axis=axes)
def test_eigh_with_overlap(seed: int, n: int, name: str, axis: str) -> None:
    rng = np.random.default_rng(seed)
    f = member(name, n, rng, axis)
    batched = k_axis_for(axis, 1) is not None
    s = overlap_member(n, rng, axis).embed(f.structure)
    values, vectors = f.eigh(s)
    for fb, sb, wb, cb in zip(
        f.blocks, s.blocks, values.blocks, vectors.blocks, strict=True
    ):
        for f_k, s_k, w, c in zip(
            each(fb, batched),
            each(sb, batched),
            each(wb, batched),
            each(cb, batched),
            strict=True,
        ):
            assert np.allclose(f_k @ c, s_k @ c * w[None, :], atol=EIGEN_TOL)
            assert np.allclose(adjoint(c) @ s_k @ c, np.eye(c.shape[1]), atol=EIGEN_TOL)


def even_occupations(n: int, vectors: BlockMatrix) -> BlockVector:
    """Fill the lowest orbitals of every block and k-point, whole Kramers pairs."""
    blocks: list[np.ndarray] = []
    for block, c in zip(vectors.structure.stored_blocks, vectors.blocks, strict=True):
        m = int(c.shape[-1])
        count = 2 * ((n + 1) // 2) if block.spin is Spin.SPINOR else (n + 1) // 2
        occupation = np.zeros(m)
        occupation[:count] = 1.0
        blocks.append(np.broadcast_to(occupation, (*c.shape[:-2], m)).copy())
    return BlockVector(vectors.structure, tuple(blocks))


@settings(max_examples=100, deadline=None)
@given(
    seed=seeds,
    n=sizes,
    name=st.sampled_from(PAIRED),
    with_overlap=st.booleans(),
    axis=axes,
)
def test_paired_structures_give_kramers_pairs(
    seed: int, n: int, name: str, with_overlap: bool, axis: str
) -> None:
    rng = np.random.default_rng(seed)
    f = member(name, n, rng, axis)
    k_axis = k_axis_for(axis, n)
    invariant = (True,) if k_axis is None else k_axis.invariant
    dense = f.to_dense()
    for k, flag in enumerate(invariant):
        if flag:
            matrix = dense if k_axis is None else dense[k]
            assert is_kramers_symmetric(matrix, tolerance=STRUCTURE_TOL)
    overlap = overlap_member(n, rng, axis).embed(f.structure) if with_overlap else None
    values, vectors = f.eigh(overlap)
    density = density_matrix(vectors, even_occupations(n, vectors))
    if name == "paired-generalized":
        (block,) = f.structure.stored_blocks
        (w,), (c,) = values.blocks, vectors.blocks
        for position in f.structure.invariant_positions(block):
            w_k, c_k = (w, c) if k_axis is None else (w[position], c[position])
            assert np.allclose(w_k[0::2], w_k[1::2], atol=EIGEN_TOL)
            assert np.allclose(c_k[:, 1::2], time_reverse(c_k[:, 0::2]), atol=EIGEN_TOL)
    dense_density = density.to_dense()
    for k, flag in enumerate(invariant):
        if flag:
            matrix = dense_density if k_axis is None else dense_density[k]
            assert is_kramers_symmetric(matrix, tolerance=STRUCTURE_TOL)


@settings(max_examples=60, deadline=None)
@given(seed=seeds, n=sizes, name=names, axis=axes)
def test_spin_trace_sums_the_spin_diagonal(
    seed: int, n: int, name: str, axis: str
) -> None:
    d = member(name, n, np.random.default_rng(seed), axis)
    dense = d.to_dense()
    total = d.spin_trace()
    assert total.structure.is_spatial
    assert total.structure.k_axis == d.structure.k_axis
    assert np.allclose(
        total.materialized()[0],
        dense[..., :n, :n] + dense[..., n:, n:],
        atol=STRUCTURE_TOL,
    )


@settings(max_examples=100, deadline=None)
@given(seed=seeds, n=sizes, name=st.sampled_from(sorted(FULL_ZONE)), axis=meshes)
def test_half_zone_storage_agrees_with_full_zone_storage(
    seed: int, n: int, name: str, axis: str
) -> None:
    """ADR-0007: storing the half zone and the full zone give the same results."""
    rng = np.random.default_rng(seed)
    k_axis = k_axis_for(axis, n)
    assert k_axis is not None
    half = member(name, n, rng, axis)
    full_structure = structure(FULL_ZONE[name], axis, n)
    full = half.embed(full_structure)
    # The complex structure stores every point, or with point operations
    # one representative per orbit of the operations alone.
    assert full.blocks[0].shape[0] == len(k_axis.stored(half_zone=False))
    other = member(name, n, rng, axis)
    assert np.allclose(half.trace(), full.trace(), atol=STRUCTURE_TOL)
    assert np.allclose(
        half.trace_product(other),
        full.trace_product(other.embed(full_structure)),
        atol=STRUCTURE_TOL,
    )
    assert close(
        (half @ other).embed(full_structure), full @ other.embed(full_structure)
    )
    overlap = overlap_member(n, rng, axis)
    values, vectors = half.eigh(overlap.embed(half.structure))
    full_values, full_vectors = full.eigh(overlap.embed(full_structure))
    for w, w_full in zip(
        values.materialized(), full_values.materialized(), strict=True
    ):
        assert np.allclose(w, w_full, atol=EIGEN_TOL)
    density = density_matrix(vectors, even_occupations(n, vectors))
    full_density = density_matrix(full_vectors, even_occupations(n, full_vectors))
    for d, d_full in zip(
        density.spin_blocks(), full_density.spin_blocks(), strict=True
    ):
        assert np.allclose(d, d_full, atol=EIGEN_TOL)


def test_rectangular_coefficients_embed_through_the_hierarchy() -> None:
    rng = np.random.default_rng(1)
    c = BlockMatrix(BlockStructure.restricted(), (rng.standard_normal((5, 3)),))
    unrestricted = c.embed(BlockStructure.unrestricted())
    assert unrestricted.shapes == ((5, 3), (5, 3))
    generalized = unrestricted.embed(BlockStructure.generalized())
    assert generalized.shapes == ((10, 6),)
    assert np.allclose(generalized.to_dense(), c.to_dense())
    assert generalized.column_split is None
    assert unrestricted.column_split == (3, 3)


def test_uneven_column_split_round_trips() -> None:
    rng = np.random.default_rng(2)
    c = BlockMatrix(
        BlockStructure.unrestricted(),
        (rng.standard_normal((4, 3)), rng.standard_normal((4, 1))),
    )
    again = BlockMatrix.from_dense(c.structure, c.to_dense(), column_split=(3, 1))
    assert close(c, again)


def test_embedding_into_a_more_symmetric_structure_is_rejected() -> None:
    rng = np.random.default_rng(3)
    u = member("unrestricted-real", 3, rng, "molecular")
    with pytest.raises(StructureError, match="does not have the restricted"):
        u.embed(BlockStructure.restricted())


def test_operations_require_matching_structures() -> None:
    rng = np.random.default_rng(4)
    r = member("restricted-real", 2, rng, "molecular")
    u = member("unrestricted-real", 2, rng, "molecular")
    with pytest.raises(StructureError, match="embed"):
        _ = r + u
    with pytest.raises(StructureError, match="embed"):
        _ = r @ u


def test_dtype_must_match_ring() -> None:
    with pytest.raises(StructureError, match="complex entries"):
        BlockMatrix(
            BlockStructure.spatial(Ring.REAL), (np.zeros((2, 2), dtype=complex),)
        )
    with pytest.raises(StructureError, match="real entries"):
        BlockMatrix(BlockStructure.spatial(Ring.COMPLEX), (np.zeros((2, 2)),))
    with pytest.raises(StructureError, match="floating"):
        BlockMatrix(BlockStructure.spatial(Ring.REAL), (np.zeros((2, 2), dtype=int),))
    with pytest.raises(StructureError, match="float64"):
        BlockMatrix(
            BlockStructure.spatial(Ring.REAL), (np.zeros((2, 2), dtype=np.float32),)
        )
    with pytest.raises(StructureError, match="complex128"):
        BlockMatrix(
            BlockStructure.spatial(Ring.COMPLEX),
            (np.zeros((2, 2), dtype=np.complex64),),
        )


def test_projection_casts_to_the_64_bit_dtypes() -> None:
    dense = np.eye(4, dtype=np.float32)
    assert (
        BlockMatrix.from_dense(BlockStructure.spatial(), dense).blocks[0].dtype
        == np.float64
    )
    complex_block = BlockMatrix.from_dense(
        BlockStructure.generalized(Ring.COMPLEX), dense
    )
    assert complex_block.blocks[0].dtype == np.complex128


def test_block_count_and_shape_are_validated() -> None:
    with pytest.raises(StructureError, match="stores 2 blocks"):
        BlockMatrix(BlockStructure.unrestricted(), (np.zeros((2, 2)),))
    with pytest.raises(StructureError, match="even"):
        BlockMatrix(BlockStructure.generalized(), (np.zeros((3, 3)),))
    with pytest.raises(StructureError, match="row counts differ"):
        BlockMatrix(BlockStructure.unrestricted(), (np.eye(2), np.eye(3)))


def test_elementwise_products_of_block_matrices_are_rejected() -> None:
    a = BlockMatrix(BlockStructure.spatial(), (np.eye(2),))
    with pytest.raises(StructureError, match="use @"):
        _ = a * a  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(StructureError, match="use @"):
        _ = a / a  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    assert np.allclose((np.float64(2.0) * a).blocks[0], 2 * np.eye(2))
    assert np.allclose((a.trace() * a).blocks[0], 4 * np.eye(2))


def test_odd_column_count_points_at_column_split() -> None:
    with pytest.raises(StructureError, match="column_split"):
        BlockMatrix.from_dense(BlockStructure.unrestricted(), np.zeros((4, 7)))


def test_embedding_outside_the_hierarchy_uses_the_verified_dense_path() -> None:
    block = np.diag([1.0, 2.0])
    equal = BlockMatrix(BlockStructure.unrestricted(), (block, block.copy()))
    restricted = equal.embed(BlockStructure.restricted())
    assert np.allclose(restricted.blocks[0], block)
    unequal = BlockMatrix(BlockStructure.unrestricted(), (block, 2 * block))
    with pytest.raises(StructureError, match="does not have the restricted"):
        unequal.embed(BlockStructure.restricted())


@pytest.mark.parametrize("with_overlap", [False, True])
def test_degenerate_clusters_are_split_into_kramers_pairs(with_overlap: bool) -> None:
    rng = np.random.default_rng(5)
    n = 3
    paired = BlockStructure.paired_generalized()
    # the identity: one 2n-fold cluster; a block-diagonal matrix: a four-fold one
    spatial = np.diag([1.0, 1.0, 2.0]).astype(complex)
    four_fold = np.block(
        [[spatial, np.zeros((n, n))], [np.zeros((n, n)), spatial.conj()]]
    )
    for f in (BlockMatrix.identity(paired, n), BlockMatrix(paired, (four_fold,))):
        overlap = None
        metric = np.eye(2 * n)
        if with_overlap:
            overlap = BlockMatrix(
                BlockStructure.spatial(), (random_positive_definite(rng, n),)
            ).embed(paired)
            metric = overlap.blocks[0]
        values, vectors = f.eigh(overlap)
        (w,), (c,) = values.blocks, vectors.blocks
        assert c.shape == (2 * n, 2 * n)
        assert np.allclose(f.blocks[0] @ c, metric @ c * w[None, :], atol=EIGEN_TOL)
        assert np.allclose(adjoint(c) @ metric @ c, np.eye(2 * n), atol=EIGEN_TOL)
        assert np.allclose(c[:, 1::2], time_reverse(c[:, 0::2]), atol=EIGEN_TOL)


def test_quaternionic_blocks_must_be_kramers_symmetric_to_diagonalize() -> None:
    rng = np.random.default_rng(6)
    not_symmetric = BlockMatrix(
        BlockStructure.paired_generalized(), (random_hermitian(rng, 4),)
    )
    with pytest.raises(StructureError, match="Θ-symmetric"):
        not_symmetric.eigh()


def test_paired_occupations_must_respect_kramers_pairs() -> None:
    paired = BlockStructure.paired_generalized()
    _, vectors = BlockMatrix.identity(paired, 2).eigh()
    with pytest.raises(StructureError, match="Kramers pair"):
        density_matrix(vectors, BlockVector(paired, (np.array([1.0, 0.0, 0.0, 0.0]),)))
    density = density_matrix(
        vectors, BlockVector(paired, (np.array([1.0, 1.0, 0.0, 0.0]),))
    )
    assert is_kramers_symmetric(density.blocks[0])


def test_time_reversal_is_checked_at_the_invariant_k_points_only() -> None:
    chain = PLAIN_AXES["chain-mesh"]
    assert chain is not None
    paired = BlockStructure.paired_generalized().with_k_axis(chain)
    _, vectors = BlockMatrix.identity(paired, 2).eigh()
    # Γ (position 0) must keep its Kramers pairs, the pair point need not.
    occupations = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    density = density_matrix(vectors, BlockVector(paired, (occupations,)))
    assert is_kramers_symmetric(density.blocks[0][0])
    with pytest.raises(StructureError, match="Kramers pair"):
        density_matrix(vectors, BlockVector(paired, (occupations[::-1],)))
    broken = BlockMatrix(
        paired, (np.stack([random_hermitian(np.random.default_rng(9), 4)] * 2),)
    )
    with pytest.raises(StructureError, match="at k-point 0"):
        broken.eigh()
    real = BlockStructure.generalized().with_k_axis(chain)
    rotated = np.array([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)
    coefficients = np.stack([rotated, rotated])
    with pytest.raises(StructureError, match="real at a time-reversal"):
        density_matrix(
            BlockMatrix(real, (coefficients,)),
            BlockVector(real, (np.array([[1.0, 0.0], [1.0, 0.0]]),)),
        )


def test_zeros_and_identity_have_the_right_block_shapes() -> None:
    assert BlockMatrix.zeros(BlockStructure.generalized(), (3, 2)).shapes == ((6, 4),)
    assert BlockMatrix.zeros(BlockStructure.unrestricted(), (3, 2)).shapes == (
        (3, 2),
        (3, 2),
    )
    identity = BlockMatrix.identity(BlockStructure.paired_generalized(), 3)
    assert identity.shapes == ((6, 6),)
    assert identity.blocks[0].dtype == np.complex128


# ------------------------------------------------------------- irrep sectors
#
# Irrep sectors (ADR-0023) multiply the molecular structures. A random
# orthogonal matrix split into two or three column blocks serves as the SALC
# basis; a member of a structure over sectors is a random Hermitian matrix
# projected onto it, which drops the couplings between sectors.


def random_sectors(rng: np.random.Generator, n: int) -> IrrepSectors:
    """Sectors of a random orthogonal basis, as many as ``n`` functions allow."""
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    splits = sorted(rng.choice(n - 1, size=min(2, n - 1), replace=False) + 1)
    blocks = tuple(np.ascontiguousarray(part) for part in np.split(q, splits, axis=1))
    return IrrepSectors("D2", ("A", "B1", "B2")[: len(blocks)], blocks)


def sectored(name: str, n: int, rng: np.random.Generator) -> BlockMatrix:
    """A random Hermitian member of the molecular structure ``name`` over sectors."""
    structure = STRUCTURES[name].with_irreps(random_sectors(rng, n))
    return BlockMatrix.from_dense(structure, random_hermitian(rng, 2 * n), check=False)


@settings(max_examples=100, deadline=None)
@given(seed=seeds, n=sizes, name=names)
def test_sectors_round_trip_through_the_dense_form(
    seed: int, n: int, name: str
) -> None:
    a = sectored(name, n, np.random.default_rng(seed))
    assert a.n_spatial == n
    assert a.structure.n_sectors == len(a.structure.irreps.labels)  # type: ignore[union-attr]
    again = BlockMatrix.from_dense(a.structure, a.to_dense(), check=True)
    assert close(a, again)
    assert close(a.project(a.structure), a)
    assert all(
        close(part, BlockMatrix.from_dense(part.structure, part.to_dense()))
        for part in a.sectors()
    )
    assert close(BlockMatrix.from_sectors(a.structure, a.sectors()), a)


@settings(max_examples=100, deadline=None)
@given(seed=seeds, n=sizes, name=names)
def test_sector_algebra_matches_dense_algebra(seed: int, n: int, name: str) -> None:
    rng = np.random.default_rng(seed)
    a = sectored(name, n, rng)
    b = BlockMatrix.from_dense(a.structure, random_hermitian(rng, 2 * n), check=False)
    ad, bd = a.to_dense(), b.to_dense()
    assert np.allclose((a @ b).to_dense(), ad @ bd, atol=STRUCTURE_TOL)
    assert np.allclose((a + b).to_dense(), ad + bd, atol=STRUCTURE_TOL)
    assert np.allclose(a.adjoint().to_dense(), adjoint(ad), atol=STRUCTURE_TOL)
    assert np.allclose(a.trace(), np.trace(ad), atol=STRUCTURE_TOL)
    assert np.allclose(a.trace_product(b), np.trace(ad @ bd), atol=STRUCTURE_TOL)
    total = a.spin_trace()
    assert total.structure == BlockStructure.spatial(
        Ring.COMPLEX if a.structure.is_complex else Ring.REAL
    ).with_irreps(a.structure.irreps)
    assert np.allclose(
        total.materialized()[0], ad[:n, :n] + ad[n:, n:], atol=STRUCTURE_TOL
    )


@settings(max_examples=100, deadline=None)
@given(seed=seeds, n=sizes, pair=st.sampled_from(EMBEDDINGS))
def test_sectors_embed_along_the_hierarchy_and_into_the_plain_structure(
    seed: int, n: int, pair: tuple[str, str]
) -> None:
    lower, higher = pair
    a = sectored(lower, n, np.random.default_rng(seed))
    irreps = a.structure.irreps
    embedded = a.embed(STRUCTURES[higher].with_irreps(irreps))
    assert np.allclose(embedded.to_dense(), a.to_dense(), atol=STRUCTURE_TOL)
    assert close(embedded.project(a.structure), a)
    plain = a.embed(STRUCTURES[lower])
    assert plain.structure.irreps is None
    assert np.allclose(plain.to_dense(), a.to_dense(), atol=STRUCTURE_TOL)
    assert close(plain.project(a.structure), a)
    assert close(plain.embed(a.structure), a)


@settings(max_examples=60, deadline=None)
@given(seed=seeds, n=sizes, name=names)
def test_sector_eigenvectors_diagonalize_the_dense_operator(
    seed: int, n: int, name: str
) -> None:
    rng = np.random.default_rng(seed)
    f = sectored(name, n, rng)
    s = overlap_member(n, rng, "molecular").project(f.structure)
    values, vectors = f.eigh(s)
    c, w = vectors.to_dense(orbitals=True), values.to_dense()
    fd, sd = f.to_dense(), s.to_dense()
    assert c.shape == (2 * n, w.shape[0])
    assert np.allclose(fd @ c, sd @ c * w[None, :], atol=EIGEN_TOL)
    assert np.allclose(adjoint(c) @ sd @ c, np.eye(c.shape[1]), atol=EIGEN_TOL)
    density = density_matrix(vectors, even_occupations(n, vectors))
    occupations = even_occupations(n, vectors).to_dense()
    assert np.allclose(
        density.to_dense(), (c * occupations[None, :]) @ adjoint(c), atol=EIGEN_TOL
    )


def test_sector_shapes_and_crossings_are_validated() -> None:
    rng = np.random.default_rng(3)
    irreps = random_sectors(rng, 4)
    structure = BlockStructure.unrestricted().with_irreps(irreps)
    zeros = BlockMatrix.zeros(structure, (4, 4))
    assert zeros.shapes == tuple((d, d) for d in irreps.dimensions for _ in range(2))
    assert BlockMatrix.identity(
        BlockStructure.generalized().with_irreps(irreps), 4
    ).shapes == tuple((2 * d, 2 * d) for d in irreps.dimensions)
    with pytest.raises(StructureError, match="square operator"):
        BlockMatrix.zeros(structure, (4, 2))
    with pytest.raises(StructureError, match="functions, got"):
        BlockMatrix(structure, tuple(np.zeros((4, 4)) for _ in zeros.blocks))
    with pytest.raises(StructureError, match="only square operators"):
        BlockMatrix.from_dense(structure, np.zeros((8, 4)), check=False)
    coefficients = zeros.map_blocks(lambda block: block[:, :1])
    with pytest.raises(StructureError, match="orbitals=True"):
        coefficients.embed(BlockStructure.unrestricted())
    with pytest.raises(StructureError, match="orbitals=True"):
        coefficients.to_dense()
    assert coefficients.to_dense(orbitals=True).shape == (8, 2 * irreps.n_irreps)
    with pytest.raises(StructureError, match="sectors"):
        BlockMatrix.from_sectors(structure, zeros.sectors()[:1])
    coupled = BlockMatrix.from_dense(
        BlockStructure.unrestricted(), random_hermitian(rng, 8), check=False
    )
    with pytest.raises(StructureError, match="couples the irreps"):
        coupled.embed(structure)

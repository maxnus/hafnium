"""Property tests of BlockMatrix over all symmetry structures (PLAN.md §8)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hafnium.backend import adjoint
from hafnium.errors import StructureError
from hafnium.structured import (
    BlockMatrix,
    BlockVector,
    Ring,
    Structure,
    density_matrix,
    is_kramers_symmetric,
    time_reverse,
)
from tests.tolerances import EIGEN_TOL, STRUCTURE_TOL

STRUCTURES: dict[str, Structure] = {
    "spatial-real": Structure.spatial(Ring.REAL),
    "spatial-complex": Structure.spatial(Ring.COMPLEX),
    "restricted-real": Structure.restricted(Ring.REAL),
    "restricted-complex": Structure.restricted(Ring.COMPLEX),
    "unrestricted-real": Structure.unrestricted(Ring.REAL),
    "unrestricted-complex": Structure.unrestricted(Ring.COMPLEX),
    "paired-unrestricted": Structure.paired_unrestricted(),
    "generalized-real": Structure.generalized(Ring.REAL),
    "generalized-complex": Structure.generalized(Ring.COMPLEX),
    "paired-generalized": Structure.paired_generalized(),
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

seeds = st.integers(min_value=0, max_value=2**31 - 1)
sizes = st.integers(min_value=1, max_value=4)
names = st.sampled_from(sorted(STRUCTURES))


def random_hermitian(rng: np.random.Generator, size: int) -> np.ndarray:
    x = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    return (x + x.conj().T) / 2


def random_positive_definite(rng: np.random.Generator, size: int) -> np.ndarray:
    x = rng.standard_normal((size, size))
    return x @ x.T + size * np.eye(size)


def member(name: str, n: int, rng: np.random.Generator) -> BlockMatrix:
    """A random Hermitian member of the structure, obtained by projection."""
    return BlockMatrix.from_dense(
        STRUCTURES[name], random_hermitian(rng, 2 * n), check=False
    )


def close(a: BlockMatrix, b: BlockMatrix, tol: float = STRUCTURE_TOL) -> bool:
    return a.structure == b.structure and all(
        np.allclose(x, y, atol=tol, rtol=0)
        for x, y in zip(a.blocks, b.blocks, strict=True)
    )


@settings(max_examples=60, deadline=None)
@given(seed=seeds, n=sizes, name=names)
def test_dense_round_trip_is_lossless(seed: int, n: int, name: str) -> None:
    a = member(name, n, np.random.default_rng(seed))
    again = BlockMatrix.from_dense(a.structure, a.to_dense(), check=True)
    assert close(a, again)


@settings(max_examples=60, deadline=None)
@given(seed=seeds, n=sizes, name=names)
def test_block_algebra_matches_dense_algebra(seed: int, n: int, name: str) -> None:
    rng = np.random.default_rng(seed)
    a, b = member(name, n, rng), member(name, n, rng)
    ad, bd = a.to_dense(), b.to_dense()
    assert np.allclose((a @ b).to_dense(), ad @ bd, atol=STRUCTURE_TOL)
    assert np.allclose((a + b).to_dense(), ad + bd, atol=STRUCTURE_TOL)
    assert np.allclose((a - 2 * b).to_dense(), ad - 2 * bd, atol=STRUCTURE_TOL)
    assert np.allclose(a.adjoint().to_dense(), adjoint(ad), atol=STRUCTURE_TOL)
    assert np.allclose(a.conj().to_dense(), ad.conj(), atol=STRUCTURE_TOL)
    assert np.allclose(a.trace(), np.trace(ad), atol=STRUCTURE_TOL)
    assert np.allclose(a.trace_product(b), np.trace(ad @ bd), atol=STRUCTURE_TOL)


@settings(max_examples=60, deadline=None)
@given(seed=seeds, n=sizes, pair=st.sampled_from(EMBEDDINGS))
def test_embedding_is_lossless_and_projection_inverts_it(
    seed: int, n: int, pair: tuple[str, str]
) -> None:
    lower, higher = pair
    a = member(lower, n, np.random.default_rng(seed))
    embedded = a.embed(STRUCTURES[higher])
    assert np.allclose(embedded.to_dense(), a.to_dense(), atol=STRUCTURE_TOL)
    assert close(embedded.project(STRUCTURES[lower]), a)


@settings(max_examples=60, deadline=None)
@given(seed=seeds, n=sizes, name=names)
def test_projection_is_idempotent(seed: int, n: int, name: str) -> None:
    a = member(name, n, np.random.default_rng(seed))
    assert close(a.project(a.structure), a)


@settings(max_examples=60, deadline=None)
@given(seed=seeds, n=sizes, name=names)
def test_eigh_diagonalizes_every_block(seed: int, n: int, name: str) -> None:
    f = member(name, n, np.random.default_rng(seed))
    values, vectors = f.eigh()
    for fb, w, c in zip(f.blocks, values.blocks, vectors.blocks, strict=True):
        assert np.allclose(fb @ c, c * w[None, :], atol=EIGEN_TOL)
        assert np.allclose(adjoint(c) @ c, np.eye(c.shape[1]), atol=EIGEN_TOL)
        assert np.all(np.diff(w) >= -EIGEN_TOL)


@settings(max_examples=60, deadline=None)
@given(seed=seeds, n=sizes, name=names)
def test_eigh_with_overlap(seed: int, n: int, name: str) -> None:
    rng = np.random.default_rng(seed)
    f = member(name, n, rng)
    overlap = BlockMatrix(Structure.spatial(), (random_positive_definite(rng, n),))
    s = overlap.embed(f.structure)
    values, vectors = f.eigh(s)
    for fb, sb, w, c in zip(
        f.blocks, s.blocks, values.blocks, vectors.blocks, strict=True
    ):
        assert np.allclose(fb @ c, sb @ c * w[None, :], atol=EIGEN_TOL)
        assert np.allclose(adjoint(c) @ sb @ c, np.eye(c.shape[1]), atol=EIGEN_TOL)


@settings(max_examples=60, deadline=None)
@given(seed=seeds, n=sizes, name=st.sampled_from(PAIRED), with_overlap=st.booleans())
def test_paired_structures_give_kramers_pairs(
    seed: int, n: int, name: str, with_overlap: bool
) -> None:
    rng = np.random.default_rng(seed)
    f = member(name, n, rng)
    assert is_kramers_symmetric(f.to_dense(), tolerance=STRUCTURE_TOL)
    overlap = None
    if with_overlap:
        overlap = BlockMatrix(
            Structure.spatial(), (random_positive_definite(rng, n),)
        ).embed(f.structure)
    values, vectors = f.eigh(overlap)
    if name == "paired-generalized":
        (w,), (c,) = values.blocks, vectors.blocks
        assert np.allclose(w[0::2], w[1::2], atol=EIGEN_TOL)
        assert np.allclose(c[:, 1::2], time_reverse(c[:, 0::2]), atol=EIGEN_TOL)
        occupations = np.zeros(2 * n)
        occupations[: 2 * ((n + 1) // 2)] = 1.0
        density = density_matrix(vectors, BlockVector(f.structure, (occupations,)))
    else:
        occupations = np.zeros(n)
        occupations[: (n + 1) // 2] = 1.0
        density = density_matrix(vectors, BlockVector(f.structure, (occupations,)))
    assert is_kramers_symmetric(density.to_dense(), tolerance=STRUCTURE_TOL)


@settings(max_examples=40, deadline=None)
@given(seed=seeds, n=sizes, name=names)
def test_spin_trace_sums_the_spin_diagonal(seed: int, n: int, name: str) -> None:
    d = member(name, n, np.random.default_rng(seed))
    dense = d.to_dense()
    total = d.spin_trace()
    assert total.structure.is_spatial
    assert np.allclose(
        total.blocks[0], dense[:n, :n] + dense[n:, n:], atol=STRUCTURE_TOL
    )


def test_rectangular_coefficients_embed_through_the_hierarchy() -> None:
    rng = np.random.default_rng(1)
    c = BlockMatrix(Structure.restricted(), (rng.standard_normal((5, 3)),))
    unrestricted = c.embed(Structure.unrestricted())
    assert unrestricted.shapes == ((5, 3), (5, 3))
    generalized = unrestricted.embed(Structure.generalized())
    assert generalized.shapes == ((10, 6),)
    assert np.allclose(generalized.to_dense(), c.to_dense())
    assert generalized.column_split is None
    assert unrestricted.column_split == (3, 3)


def test_uneven_column_split_round_trips() -> None:
    rng = np.random.default_rng(2)
    c = BlockMatrix(
        Structure.unrestricted(),
        (rng.standard_normal((4, 3)), rng.standard_normal((4, 1))),
    )
    again = BlockMatrix.from_dense(c.structure, c.to_dense(), column_split=(3, 1))
    assert close(c, again)


def test_embedding_into_a_more_symmetric_structure_is_rejected() -> None:
    rng = np.random.default_rng(3)
    u = member("unrestricted-real", 3, rng)
    with pytest.raises(StructureError, match="does not have the restricted"):
        u.embed(Structure.restricted())


def test_operations_require_matching_structures() -> None:
    rng = np.random.default_rng(4)
    r = member("restricted-real", 2, rng)
    u = member("unrestricted-real", 2, rng)
    with pytest.raises(StructureError, match="embed"):
        _ = r + u
    with pytest.raises(StructureError, match="embed"):
        _ = r @ u


def test_dtype_must_match_ring() -> None:
    with pytest.raises(StructureError, match="complex entries"):
        BlockMatrix(Structure.spatial(Ring.REAL), (np.zeros((2, 2), dtype=complex),))
    with pytest.raises(StructureError, match="real entries"):
        BlockMatrix(Structure.spatial(Ring.COMPLEX), (np.zeros((2, 2)),))
    with pytest.raises(StructureError, match="floating"):
        BlockMatrix(Structure.spatial(Ring.REAL), (np.zeros((2, 2), dtype=int),))


def test_block_count_and_shape_are_validated() -> None:
    with pytest.raises(StructureError, match="stores 2 blocks"):
        BlockMatrix(Structure.unrestricted(), (np.zeros((2, 2)),))
    with pytest.raises(StructureError, match="even"):
        BlockMatrix(Structure.generalized(), (np.zeros((3, 3)),))


def test_zeros_and_identity_have_the_right_block_shapes() -> None:
    assert BlockMatrix.zeros(Structure.generalized(), (3, 2)).shapes == ((6, 4),)
    assert BlockMatrix.zeros(Structure.unrestricted(), (3, 2)).shapes == (
        (3, 2),
        (3, 2),
    )
    identity = BlockMatrix.identity(Structure.paired_generalized(), 3)
    assert identity.shapes == ((6, 6),)
    assert identity.blocks[0].dtype == np.complex128

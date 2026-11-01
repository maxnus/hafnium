"""JAX as a backend: selection, 64-bit arrays and the block types as pytrees."""

# JAX's own annotations leave parts of its API unknown to a strict checker.
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import numpy as np
import pytest

from hafnium.backend import (
    Array,
    Backend,
    active_backend,
    array_namespace,
    default_namespace,
    from_host,
    host_namespace,
    set_backend,
    use_backend,
)
from hafnium.errors import BackendError
from hafnium.structured import (
    BlockMatrix,
    BlockStructure,
    BlockTensor,
    BlockVector,
    TensorAxis,
    TensorStructure,
)

pytest.importorskip("jax")

import jax


def test_unknown_backends_are_rejected() -> None:
    with pytest.raises(BackendError, match="unknown backend"):
        Backend.named("torch")
    with pytest.raises(BackendError), use_backend("torch"):
        pass


def test_selection_is_scoped_and_restored() -> None:
    assert active_backend().name == "numpy"
    with use_backend("jax") as outer:
        assert active_backend() is outer
        assert outer.name == "jax"
        assert default_namespace() is outer.namespace
        assert host_namespace() is not default_namespace()
        with use_backend("numpy"):
            assert active_backend().name == "numpy"
        assert active_backend() is outer
    assert active_backend().name == "numpy"
    assert Backend.named("jax") is outer


def test_set_backend_returns_the_previous_selection() -> None:
    previous = set_backend("jax")
    try:
        assert previous.name == "numpy"
        assert active_backend().name == "jax"
    finally:
        set_backend(previous)
    assert active_backend().name == "numpy"


def test_jax_arrays_are_double_precision() -> None:
    with use_backend("jax") as backend:
        xp = backend.namespace
        assert xp.asarray([1.0]).dtype == xp.float64
        assert xp.zeros((2,), dtype=xp.complex128).dtype == xp.complex128
        moved = from_host(np.ones((2, 3)))
        assert isinstance(moved, jax.Array)
        assert moved.dtype == xp.float64
        assert array_namespace(moved) is xp


def matrix_and_vector(backend: Backend) -> tuple[BlockMatrix, BlockVector]:
    xp = backend.namespace
    structure = BlockStructure.unrestricted()
    alpha = xp.asarray([[1.0, 2.0], [2.0, 5.0]])
    beta = xp.asarray([[3.0, 0.0], [0.0, 1.0]])
    matrix = BlockMatrix(structure, (alpha, beta))
    vector = BlockVector(structure, (xp.asarray([1.0, 0.0]), xp.asarray([0.5, 0.5])))
    return matrix, vector


def tensor(backend: Backend) -> BlockTensor:
    xp = backend.namespace
    structure = BlockStructure.unrestricted()
    axis = TensorAxis(structure, (2, 3))
    tensor_structure = TensorStructure((axis, axis))
    return BlockTensor.from_function(
        tensor_structure,
        lambda key: xp.full(tensor_structure.shape(key), float(sum(key) + 1)),
    )


def test_block_types_flatten_to_their_arrays() -> None:
    with use_backend("jax") as backend:
        matrix, vector = matrix_and_vector(backend)
        for value in (matrix, vector, tensor(backend)):
            leaves, definition = jax.tree_util.tree_flatten(value)
            assert len(leaves) == len(value.blocks)
            assert all(isinstance(leaf, jax.Array) for leaf in leaves)
            rebuilt = jax.tree_util.tree_unflatten(definition, leaves)
            assert type(rebuilt) is type(value)
            assert rebuilt.structure is value.structure
            assert all(
                a is b for a, b in zip(rebuilt.blocks, value.blocks, strict=True)
            )


def test_tree_map_keeps_the_structure() -> None:
    with use_backend("jax") as backend:
        matrix, _ = matrix_and_vector(backend)

        def double(x: jax.Array) -> jax.Array:
            return 2.0 * x

        doubled = jax.tree_util.tree_map(double, matrix)
        assert isinstance(doubled, BlockMatrix)
        assert doubled.structure is matrix.structure
        np.testing.assert_allclose(
            np.asarray(doubled.to_dense()), 2.0 * np.asarray(matrix.to_dense())
        )


def test_unflattening_skips_validation() -> None:
    """A transformation may rebuild a type around placeholders, not arrays."""
    with use_backend("jax") as backend:
        matrix, _ = matrix_and_vector(backend)
        _, definition = jax.tree_util.tree_flatten(matrix)
        placeholder = jax.tree_util.tree_unflatten(definition, [None, None])
        assert isinstance(placeholder, BlockMatrix)
        assert placeholder.blocks == (None, None)


def test_structured_operations_trace_under_jit() -> None:
    with use_backend("jax") as backend:
        matrix, vector = matrix_and_vector(backend)

        @jax.jit
        def energy(density: BlockMatrix, fock: BlockMatrix) -> Array:
            return (density @ fock).trace()

        @jax.jit
        def scaled(x: BlockMatrix, occupations: BlockVector) -> BlockMatrix:
            return BlockMatrix(
                x.structure,
                tuple(
                    block * occupation[None, :]
                    for block, occupation in zip(
                        x.blocks, occupations.blocks, strict=True
                    )
                ),
            )

        expected = (matrix @ matrix).trace()
        assert float(energy(matrix, matrix)) == pytest.approx(float(expected))
        result = scaled(matrix, vector)
        assert isinstance(result, BlockMatrix)
        np.testing.assert_allclose(
            np.asarray(result.blocks[0]), np.asarray(matrix.blocks[0]) * [[1.0, 0.0]]
        )


def test_jax_arrays_are_rejected_when_mixed_with_host_arrays() -> None:
    with use_backend("jax") as backend, pytest.raises(TypeError, match="Multiple"):
        array_namespace(backend.namespace.ones(2), np.ones(2))

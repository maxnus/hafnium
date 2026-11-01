"""JAX as a backend: 64-bit arrays and the structured types as pytrees (ADR-0036).

The block types are registered with their arrays as leaves and their
structure as static metadata, so ``jax.tree_util`` and the transformations
see through them. Unflattening bypasses the constructors' validation, as
JAX asks of pytree types: a transformation may rebuild a type around
placeholder leaves that are not arrays.
"""

# JAX's own annotations leave parts of its API unknown to a strict checker.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import Any

from hafnium.backend.active import Backend
from hafnium.errors import BackendError

_registered = False


def load() -> Backend:
    """Import JAX, switch it to 64-bit arrays and register the pytrees."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise BackendError(
            "the jax backend needs JAX installed (pip install hafnium[jax])"
        ) from error
    jax.config.update("jax_enable_x64", True)
    if jnp.asarray(1.0).dtype != jnp.float64:
        raise BackendError("JAX did not switch to 64-bit arrays")
    register_pytrees()
    return Backend("jax", jnp)


def register_pytrees() -> None:
    """Register the block types with ``jax.tree_util``, once."""
    global _registered
    if _registered:
        return
    from jax.tree_util import register_pytree_node

    from hafnium.structured import BlockMatrix, BlockTensor, BlockVector

    for cls in (BlockMatrix, BlockVector, BlockTensor):
        register_pytree_node(cls, _flatten, _unflattener(cls))
    _registered = True


def _flatten(value: Any) -> tuple[list[Any], Hashable]:
    """Flatten a block type to its arrays, with its structure as the metadata."""
    return list(value.blocks), value.structure


def _unflattener(cls: type[Any]) -> Callable[[Hashable, Iterable[Any]], Any]:
    def unflatten(structure: Hashable, children: Iterable[Any]) -> Any:
        value: Any = object.__new__(cls)
        object.__setattr__(value, "structure", structure)
        object.__setattr__(value, "blocks", tuple(children))
        return value

    return unflatten

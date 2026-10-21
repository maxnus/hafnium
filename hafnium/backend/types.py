"""Type aliases for arrays and array namespaces."""

from types import ModuleType
from typing import Any, TypeAlias

from numpy.typing import NDArray

Array: TypeAlias = NDArray[Any]
"""Nominal array type.

At runtime an array may belong to any backend that implements the array API
standard; static checking uses NumPy's array type because the backends do
not share a common static type. Shapes are documented in docstrings with the
index letters of CONVENTIONS.md §2.4.
"""

Namespace: TypeAlias = ModuleType
"""An array API namespace such as ``numpy``, ``cupy`` or ``jax.numpy``."""

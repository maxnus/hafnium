"""Array backend: the only package that imports NumPy directly.

Method code obtains the array namespace from the arrays it is given, via
:func:`array_namespace`, and uses only the operations listed in
:mod:`hafnium.backend.allowed`, so that the same code runs on NumPy, CuPy
and JAX arrays. New arrays that are not derived from existing ones are
created with :func:`default_namespace`.
"""

from hafnium.backend.linalg import adjoint, eigh, generalized_eigh
from hafnium.backend.namespace import array_namespace, default_namespace
from hafnium.backend.types import Array, Namespace

__all__ = [
    "Array",
    "Namespace",
    "adjoint",
    "array_namespace",
    "default_namespace",
    "eigh",
    "generalized_eigh",
]

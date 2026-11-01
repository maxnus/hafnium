"""Array backend: the only package that imports NumPy directly.

Method code obtains the array namespace from the arrays it is given, via
:func:`array_namespace`, and uses only the operations listed in
:mod:`hafnium.backend.allowed`, so that the same code runs on NumPy, CuPy
and JAX arrays. New arrays that are not derived from existing ones are
created with :func:`default_namespace`, the namespace of the active
:class:`Backend`, which :func:`use_backend` selects (ADR-0036). The setup
around the compiled kernels works on host arrays, :func:`host_namespace`,
and hands its results over with :func:`from_host`.
"""

from hafnium.backend.active import (
    Backend,
    active_backend,
    from_host,
    host_namespace,
    set_backend,
    use_backend,
)
from hafnium.backend.host import host_array, to_host
from hafnium.backend.linalg import adjoint, block_diag, eigh, generalized_eigh
from hafnium.backend.namespace import array_namespace, default_namespace
from hafnium.backend.types import Array, Namespace

__all__ = [
    "Array",
    "Backend",
    "Namespace",
    "active_backend",
    "adjoint",
    "array_namespace",
    "block_diag",
    "default_namespace",
    "eigh",
    "from_host",
    "generalized_eigh",
    "host_array",
    "host_namespace",
    "set_backend",
    "to_host",
    "use_backend",
]

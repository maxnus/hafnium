"""Hafnium: symmetry-structured quantum chemistry.

The package is in the bootstrap stage. Public API will be re-exported from
``hafnium.api`` once methods exist; for now only the version and the compiled
kernel smoke-test function are exposed.
"""

from importlib.metadata import version as _distribution_version

from hafnium._kernels import kernel_version

__version__ = _distribution_version("hafnium")

__all__ = ["__version__", "kernel_version"]

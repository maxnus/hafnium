"""Exchange–correlation functionals (M7, ADR-0026).

A :class:`Functional` is a linear combination of the functionals of libxc,
named as libxc names them and evaluated by the ``xcx`` crate in the
compiled layer, plus the fractions of exact exchange it asks for.
:func:`evaluate_functional` returns the energy per particle and the
derivatives of the energy density at every grid point for a collinear
:class:`DensityOnGrid`.
"""

from hafnium.xc.evaluation import DensityOnGrid, XcValues, evaluate_functional
from hafnium.xc.functional import (
    NAMED_FUNCTIONALS,
    ComponentInfo,
    Functional,
    Rung,
    component_info,
)

__all__ = [
    "NAMED_FUNCTIONALS",
    "ComponentInfo",
    "DensityOnGrid",
    "Functional",
    "Rung",
    "XcValues",
    "component_info",
    "evaluate_functional",
]

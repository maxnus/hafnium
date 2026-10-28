"""Exchange–correlation functionals (M7, ADR-0026).

A :class:`Functional` is a linear combination of the functionals of libxc,
named as libxc names them and evaluated by the ``xcx`` crate in the
compiled layer, plus the fractions of exact exchange it asks for.
:func:`evaluate_functional` returns the energy per particle and the
derivatives of the energy density at every grid point for a collinear
:class:`DensityOnGrid`; :func:`evaluate_noncollinear` the multi-collinear
form of a functional for a :class:`NoncollinearDensity` of the generalized
classes (ADR-0029).
"""

from hafnium.xc.evaluation import DensityOnGrid, XcValues, evaluate_functional
from hafnium.xc.functional import (
    NAMED_FUNCTIONALS,
    ComponentInfo,
    Functional,
    Rung,
    component_info,
)
from hafnium.xc.noncollinear import (
    DEFAULT_COLLINEAR_THRESHOLD,
    DEFAULT_MAX_LEVEL,
    DEFAULT_PRECISION,
    NoncollinearDensity,
    NoncollinearOptions,
    NoncollinearValues,
    evaluate_noncollinear,
)

__all__ = [
    "DEFAULT_COLLINEAR_THRESHOLD",
    "DEFAULT_MAX_LEVEL",
    "DEFAULT_PRECISION",
    "NAMED_FUNCTIONALS",
    "ComponentInfo",
    "DensityOnGrid",
    "Functional",
    "NoncollinearDensity",
    "NoncollinearOptions",
    "NoncollinearValues",
    "Rung",
    "XcValues",
    "component_info",
    "evaluate_functional",
    "evaluate_noncollinear",
]

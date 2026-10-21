"""Exact four-centre electron repulsion integrals of a molecule.

The integrals are ``eri[u, v, w, x] = (uv|wx)`` in chemists' notation
(CONVENTIONS.md §3), stored in full. Density fitting is the primary route
to two-electron terms (PLAN.md); this module serves tests, small systems
and the in-core exact path.
"""

from __future__ import annotations

from hafnium.backend import Array, host_array
from hafnium.basis import BasisSet
from hafnium.integrals.engine import IntegralEngine
from hafnium.integrals.shell_table import ShellTable
from hafnium.molecule import Molecule

DEFAULT_SCHWARZ_THRESHOLD = 1e-14
"""Shell quartets whose Schwarz bound is below this are not evaluated.

Every neglected integral is smaller than the threshold in magnitude. The
default sits below the integral accuracy target so that screening never
shows in a result; raise it where speed matters more.
"""


def eri(
    engine: IntegralEngine,
    basis: BasisSet,
    molecule: Molecule,
    *,
    omega: float | None = None,
    threshold: float = DEFAULT_SCHWARZ_THRESHOLD,
) -> Array:
    """Electron repulsion integrals ``eri[ao, ao, ao, ao]``, chemists' notation.

    With ``omega`` the long-range kernel ``erf(ω r) / r`` replaces ``1 / r``.
    Shell quartets whose Schwarz bound is below ``threshold`` are left at
    zero.
    """
    table = ShellTable.from_basis(basis, molecule.positions_array())
    return engine.coulomb_four_center(
        (table, table),
        (table, table),
        host_array([[[0.0] * 3] * 3]),
        omega=omega,
        threshold=threshold,
    )[0]

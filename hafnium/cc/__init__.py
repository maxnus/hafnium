"""Coupled-cluster methods over block tensors (M8).

CCSD with density fitting for every Fukutome class of the reference,
molecular or over the k-points of a cell, from the spin-orbital
equations the generator of ADR-0032 derives and writes once
(:func:`df_ccsd`, :func:`ccsd_from_df_tensor`,
:func:`ccsd_from_periodic_fitting`, :func:`solve_ccsd`).
"""

from hafnium.cc.ccsd import (
    CcsdIteration,
    CcsdOptions,
    CcsdResult,
    ccsd_from_df_tensor,
    ccsd_from_periodic_fitting,
    df_ccsd,
    solve_ccsd,
)
from hafnium.cc.diis import AmplitudeDiis
from hafnium.cc.hamiltonian import NormalOrderedHamiltonian

__all__ = [
    "AmplitudeDiis",
    "CcsdIteration",
    "CcsdOptions",
    "CcsdResult",
    "NormalOrderedHamiltonian",
    "ccsd_from_df_tensor",
    "ccsd_from_periodic_fitting",
    "df_ccsd",
    "solve_ccsd",
]

"""Coupled-cluster methods over block tensors (M8).

CCSD with density fitting for every Fukutome class of the reference,
molecular or over the k-points of a cell, from the spin-orbital
equations the generator of ADR-0032 derives and writes once
(:func:`df_ccsd`, :func:`ccsd_from_df_tensor`,
:func:`ccsd_from_periodic_fitting`, :func:`solve_ccsd`), and CCSD(T)
with the correction from the same generator over batches of occupied
triples (ADR-0033; :func:`df_ccsd_pt`, :func:`ccsd_pt_from_df_tensor`,
:func:`ccsd_pt_from_periodic_fitting`, :func:`solve_ccsd_pt`, or
:func:`triples_correction` for existing amplitudes), and the Λ equations
of CCSD with its one-particle density (ADR-0034; :func:`df_ccsd_lambda`,
:func:`ccsd_lambda_from_df_tensor`, :func:`ccsd_lambda_from_periodic_fitting`,
:func:`solve_ccsd_lambda`, or :func:`solve_lambda` for an existing result).
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
from hafnium.cc.ccsd_lambda import (
    CcsdLambdaResult,
    LambdaIteration,
    ccsd_lambda_from_df_tensor,
    ccsd_lambda_from_periodic_fitting,
    df_ccsd_lambda,
    solve_ccsd_lambda,
    solve_lambda,
)
from hafnium.cc.ccsd_pt import (
    CcsdPtResult,
    TriplesOptions,
    ccsd_pt_from_df_tensor,
    ccsd_pt_from_periodic_fitting,
    df_ccsd_pt,
    solve_ccsd_pt,
    triples_correction,
)
from hafnium.cc.diis import AmplitudeDiis
from hafnium.cc.hamiltonian import NormalOrderedHamiltonian

__all__ = [
    "AmplitudeDiis",
    "CcsdIteration",
    "CcsdLambdaResult",
    "CcsdOptions",
    "CcsdPtResult",
    "CcsdResult",
    "LambdaIteration",
    "NormalOrderedHamiltonian",
    "TriplesOptions",
    "ccsd_from_df_tensor",
    "ccsd_from_periodic_fitting",
    "ccsd_lambda_from_df_tensor",
    "ccsd_lambda_from_periodic_fitting",
    "ccsd_pt_from_df_tensor",
    "ccsd_pt_from_periodic_fitting",
    "df_ccsd",
    "df_ccsd_lambda",
    "df_ccsd_pt",
    "solve_ccsd",
    "solve_ccsd_lambda",
    "solve_ccsd_pt",
    "solve_lambda",
    "triples_correction",
]

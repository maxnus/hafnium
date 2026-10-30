"""Møller–Plesset perturbation theory over block tensors (M8).

Second-order energies with density fitting for every Fukutome class of
the reference, molecular or over the k-points of a cell, written once in
spin-orbital form over the block tensors of ADR-0030 and ADR-0031
(:func:`df_mp2`, :func:`mp2_from_df_tensor`, :func:`mp2_from_periodic_fitting`).
"""

from hafnium.mp.mp2 import (
    Mp2Result,
    df_mp2,
    mp2_from_df_tensor,
    mp2_from_periodic_fitting,
    second_order,
)

__all__ = [
    "Mp2Result",
    "df_mp2",
    "mp2_from_df_tensor",
    "mp2_from_periodic_fitting",
    "second_order",
]

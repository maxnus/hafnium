"""The triples of the (T) correction as contracted terms (ADR-0033).

The correction reads the connected triples ``W = <Φ_ijk^abc| H_N T2 |Φ>_C``
and the disconnected ``V = <Φ_ijk^abc| H_N (T1 + T2) |Φ>`` with
``<Φ_ijk^abc| = <Φ| a†_i a†_j a†_k a_c a_b a_a``, and is
``1/36 Σ conj(W + V) W / D`` over the orbital energy denominators
(:mod:`hafnium.cc.ccsd_t`).
"""

from __future__ import annotations

from fractions import Fraction

from hafnium.cc.generator.equations import (
    Equation,
    Projection,
    doubles_cluster,
    singles_cluster,
)


def ccsd_t_equations() -> dict[str, Equation]:
    """Derive the connected and the disconnected triples of (T), by name.

    ``connected_triples`` is ``<Φ_ijk^abc| H_N T2 |Φ>_C``, the triples
    the doubles generate through the two-electron part of the
    Hamiltonian, and ``disconnected_triples`` the terms of
    ``<Φ_ijk^abc| H_N (T1 + T2) |Φ>`` in which the cluster operator is not
    contracted with the Hamiltonian: the singles with ``<bc||jk>`` and the
    doubles with ``f[vo]``, which vanishes for a canonical reference.
    """
    projection = Projection.triples()
    one = Fraction(1)
    return {
        "connected_triples": projection.expectation(
            "connected_triples", [(one, [doubles_cluster])]
        ),
        "disconnected_triples": projection.expectation(
            "disconnected_triples",
            [(one, [singles_cluster]), (one, [doubles_cluster])],
            connected=False,
        ),
    }

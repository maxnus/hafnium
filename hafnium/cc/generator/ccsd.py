"""The spin-orbital CCSD equations as contracted terms (ADR-0032).

- ``E = <Φ| H_N e^T |Φ>_C``,
- ``0 = <Φ_i^a| H_N e^T |Φ>_C`` with ``<Φ_i^a| = <Φ| a†_i a_a``,
- ``0 = <Φ_ij^ab| H_N e^T |Φ>_C`` with ``<Φ_ij^ab| = <Φ| a†_i a†_j a_b a_a``,

the subscript marking the terms in which every cluster operator is
contracted with the Hamiltonian, ``e^T`` expanded into the products
``T1^m T2^n / (m! n!)`` with no more operators than the projection and
the Hamiltonian can contract (:mod:`hafnium.cc.generator.equations`).
"""

from __future__ import annotations

from hafnium.cc.generator.equations import Equation, Projection


def ccsd_equations() -> dict[str, Equation]:
    """Derive the energy, singles and doubles equations of CCSD, by name."""
    return {
        projection.name: projection.equation()
        for projection in (
            Projection.energy(),
            Projection.singles(),
            Projection.doubles(),
        )
    }

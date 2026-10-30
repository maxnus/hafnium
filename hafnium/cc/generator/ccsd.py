"""The spin-orbital CCSD equations as contracted terms (ADR-0032).

With the normal-ordered Hamiltonian ``H_N = Σ f_pq {p† q} + ¼ Σ <pq||rs>
{p† q† s r}`` and the cluster operators ``T1 = Σ t_i^a a†_a a_i`` and
``T2 = ¼ Σ t_ij^ab a†_a a†_b a_j a_i``, the CCSD energy and amplitude
equations are

- ``E = <Φ| H_N e^T |Φ>_C``,
- ``0 = <Φ_i^a| H_N e^T |Φ>_C`` with ``<Φ_i^a| = <Φ| a†_i a_a``,
- ``0 = <Φ_ij^ab| H_N e^T |Φ>_C`` with ``<Φ_ij^ab| = <Φ| a†_i a†_j a_b a_a``,

the subscript marking the terms in which every cluster operator is
contracted with the Hamiltonian. The general indices of ``H_N`` are
expanded into occupied and virtual ones, so that the terms refer to the
blocks ``f[oo]``, ``f[ov]``, ... and ``v[oovv]``, ... of the Hamiltonian,
and ``e^T`` is expanded into the products ``T1^m T2^n / (m! n!)`` with no
more operators than the projection and the Hamiltonian can contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import factorial

from hafnium.cc.generator.wick import (
    OCCUPIED,
    SPACES,
    VIRTUAL,
    Factor,
    Index,
    Operator,
    OperatorString,
    Term,
    connected_expectation,
    merged,
)


class Fresh:
    """Unique dummy indices."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, space: str) -> Index:
        """Return a new dummy index of ``space``."""
        self.count += 1
        return Index(space, f"_{self.count}")


def hamiltonian_strings(fresh: Fresh) -> list[OperatorString]:
    """``H_N`` split by the spaces of its indices, every string normal ordered."""
    strings: list[OperatorString] = []
    for space_p, space_q in product(SPACES, repeat=2):
        p, q = fresh(space_p), fresh(space_q)
        strings.append(
            OperatorString.normal_ordered(
                Fraction(1),
                Factor("f", (p, q)),
                (Operator(p, True), Operator(q, False)),
            )
        )
    for spaces in product(SPACES, repeat=4):
        p, q, r, s = (fresh(space) for space in spaces)
        strings.append(
            OperatorString.normal_ordered(
                Fraction(1, 4),
                Factor("v", (p, q, r, s)),
                (
                    Operator(p, True),
                    Operator(q, True),
                    Operator(s, False),
                    Operator(r, False),
                ),
            )
        )
    return strings


def singles_cluster(fresh: Fresh) -> OperatorString:
    """``T1 = Σ t1[i, a] a†_a a_i``."""
    i, a = fresh(OCCUPIED), fresh(VIRTUAL)
    return OperatorString(
        Fraction(1), Factor("t1", (i, a)), (Operator(a, True), Operator(i, False))
    )


def doubles_cluster(fresh: Fresh) -> OperatorString:
    """``T2 = ¼ Σ t2[i, j, a, b] a†_a a†_b a_j a_i``."""
    i, j = fresh(OCCUPIED), fresh(OCCUPIED)
    a, b = fresh(VIRTUAL), fresh(VIRTUAL)
    return OperatorString(
        Fraction(1, 4),
        Factor("t2", (i, j, a, b)),
        (Operator(a, True), Operator(b, True), Operator(j, False), Operator(i, False)),
    )


def cluster_products(
    fresh: Fresh, max_operators: int
) -> list[tuple[Fraction, list[OperatorString]]]:
    """List ``T1^m T2^n / (m! n!)`` up to ``max_operators`` operators.

    The empty product comes first.
    """
    products: list[tuple[Fraction, list[OperatorString]]] = []
    for n in range(max_operators // 4 + 1):
        for m in range((max_operators - 4 * n) // 2 + 1):
            strings = [singles_cluster(fresh) for _ in range(m)]
            strings += [doubles_cluster(fresh) for _ in range(n)]
            products.append((Fraction(1, factorial(m) * factorial(n)), strings))
    return products


@dataclass(frozen=True, slots=True)
class Projection:
    """``<Φ_μ|`` as the operators of the bra and the labels of its free indices."""

    operators: tuple[Operator, ...]
    free: str


def energy_projection() -> Projection:
    """``<Φ|``."""
    return Projection((), "")


def singles_projection() -> Projection:
    """``<Φ_i^a| = <Φ| a†_i a_a``."""
    i, a = Index(OCCUPIED, "i", True), Index(VIRTUAL, "a", True)
    return Projection((Operator(i, True), Operator(a, False)), "ia")


def doubles_projection() -> Projection:
    """``<Φ_ij^ab| = <Φ| a†_i a†_j a_b a_a``."""
    i, j = Index(OCCUPIED, "i", True), Index(OCCUPIED, "j", True)
    a, b = Index(VIRTUAL, "a", True), Index(VIRTUAL, "b", True)
    return Projection(
        (Operator(i, True), Operator(j, True), Operator(b, False), Operator(a, False)),
        "ijab",
    )


def projected_equation(projection: Projection) -> list[Term]:
    """``<Φ_μ| H_N e^T |Φ>_C`` as merged canonical terms."""
    fresh = Fresh()
    bra = OperatorString(Fraction(1), None, projection.operators)
    terms: list[Term] = []
    # The projection and the Hamiltonian contract at most 4 + 4 operators.
    for hamiltonian in hamiltonian_strings(fresh):
        for coefficient, clusters in cluster_products(
            fresh, len(projection.operators) + 4
        ):
            strings = [bra, hamiltonian, *clusters]
            scaled = [
                Term(term.coefficient * coefficient, term.factors)
                for term in connected_expectation(strings, 1, range(2, len(strings)))
            ]
            terms.extend(scaled)
    return merged(terms)


def ccsd_equations() -> dict[str, list[Term]]:
    """Derive the energy, singles and doubles equations of CCSD."""
    return {
        "energy": projected_equation(energy_projection()),
        "singles": projected_equation(singles_projection()),
        "doubles": projected_equation(doubles_projection()),
    }

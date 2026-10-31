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
    Factor,
    Index,
    Kind,
    Operator,
    OperatorString,
    Term,
    connected_expectation,
)
from hafnium.correlated import Space


class Fresh:
    """Unique dummy indices."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, space: Space) -> Index:
        """Return a new dummy index of ``space``."""
        self.count += 1
        return Index(space, f"_{self.count}")


def hamiltonian_strings(fresh: Fresh) -> list[OperatorString]:
    """``H_N`` split by the spaces of its indices, every string normal ordered."""
    strings: list[OperatorString] = []
    for space_p, space_q in product(Space, repeat=2):
        p, q = fresh(space_p), fresh(space_q)
        strings.append(
            OperatorString.normal_ordered(
                Fraction(1),
                Factor(Kind.FOCK, (p, q)),
                (Operator(p, True), Operator(q, False)),
            )
        )
    for spaces in product(Space, repeat=4):
        p, q, r, s = (fresh(space) for space in spaces)
        strings.append(
            OperatorString.normal_ordered(
                Fraction(1, 4),
                Factor(Kind.INTEGRAL, (p, q, r, s)),
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
    i, a = fresh(Space.OCCUPIED), fresh(Space.VIRTUAL)
    return OperatorString(
        Fraction(1),
        Factor(Kind.SINGLES, (i, a)),
        (Operator(a, True), Operator(i, False)),
    )


def doubles_cluster(fresh: Fresh) -> OperatorString:
    """``T2 = ¼ Σ t2[i, j, a, b] a†_a a†_b a_j a_i``."""
    i, j = fresh(Space.OCCUPIED), fresh(Space.OCCUPIED)
    a, b = fresh(Space.VIRTUAL), fresh(Space.VIRTUAL)
    return OperatorString(
        Fraction(1, 4),
        Factor(Kind.DOUBLES, (i, j, a, b)),
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
class Equation:
    """The merged canonical terms of ``<Φ_μ| H_N e^T |Φ>_C``.

    ``free`` holds the labels of the free indices in the order of the
    result's axes, ``"ijab"`` for the doubles residual ``[i, j, a, b]``.
    """

    name: str
    free: str
    terms: tuple[Term, ...]


@dataclass(frozen=True, slots=True)
class Projection:
    """``<Φ_μ|`` as the operators of the bra and the labels of its free indices."""

    name: str
    operators: tuple[Operator, ...]
    free: str

    @classmethod
    def energy(cls) -> Projection:
        """``<Φ|``."""
        return cls("energy", (), "")

    @classmethod
    def singles(cls) -> Projection:
        """``<Φ_i^a| = <Φ| a†_i a_a``."""
        i, a = Index(Space.OCCUPIED, "i", True), Index(Space.VIRTUAL, "a", True)
        return cls("singles", (Operator(i, True), Operator(a, False)), "ia")

    @classmethod
    def doubles(cls) -> Projection:
        """``<Φ_ij^ab| = <Φ| a†_i a†_j a_b a_a``."""
        i, j = Index(Space.OCCUPIED, "i", True), Index(Space.OCCUPIED, "j", True)
        a, b = Index(Space.VIRTUAL, "a", True), Index(Space.VIRTUAL, "b", True)
        return cls(
            "doubles",
            (
                Operator(i, True),
                Operator(j, True),
                Operator(b, False),
                Operator(a, False),
            ),
            "ijab",
        )

    def equation(self) -> Equation:
        """Derive ``<Φ_μ| H_N e^T |Φ>_C`` as merged canonical terms."""
        fresh = Fresh()
        bra = OperatorString(Fraction(1), None, self.operators)
        terms: list[Term] = []
        # The projection and the Hamiltonian contract at most 4 + 4 operators.
        for hamiltonian in hamiltonian_strings(fresh):
            for coefficient, clusters in cluster_products(
                fresh, len(self.operators) + 4
            ):
                strings = [bra, hamiltonian, *clusters]
                terms.extend(
                    Term(term.coefficient * coefficient, term.factors)
                    for term in connected_expectation(
                        strings, 1, range(2, len(strings))
                    )
                )
        return Equation(self.name, self.free, tuple(Term.merge(terms)))


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

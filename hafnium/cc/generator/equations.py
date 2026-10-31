"""Equations as expectation values of operator strings over the Fermi vacuum.

With the normal-ordered Hamiltonian ``H_N = Σ f_pq {p† q} + ¼ Σ <pq||rs>
{p† q† s r}`` and the cluster operators ``T1 = Σ t_i^a a†_a a_i`` and
``T2 = ¼ Σ t_ij^ab a†_a a†_b a_j a_i``, an equation is ``<Φ_μ| H_N Π T |Φ>``
for a projection ``<Φ_μ|`` and products of cluster operators, keeping the
terms in which every cluster operator is contracted with the Hamiltonian
or the others. The general indices of ``H_N`` are expanded into occupied
and virtual ones, so that the terms refer to the blocks ``f[oo]``,
``f[ov]``, ... and ``v[oovv]``, ... of the Hamiltonian.
:mod:`hafnium.cc.generator.ccsd` and :mod:`hafnium.cc.generator.ccsd_t`
state the equations of the two methods in these terms.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    expectation,
)
from hafnium.correlated import Space

Cluster = Callable[["DummyIndices"], OperatorString]
"""A cluster operator, built over fresh dummy indices."""


class DummyIndices:
    """A supply of distinct summation indices for one derivation."""

    def __init__(self) -> None:
        self.count = 0

    def new(self, space: Space) -> Index:
        """Return a dummy index of ``space`` not handed out before."""
        self.count += 1
        return Index(space, f"_{self.count}")


def hamiltonian_strings(dummies: DummyIndices) -> list[OperatorString]:
    """``H_N`` split by the spaces of its indices, every string normal ordered."""
    strings: list[OperatorString] = []
    for space_p, space_q in product(Space, repeat=2):
        p, q = dummies.new(space_p), dummies.new(space_q)
        strings.append(
            OperatorString.normal_ordered(
                Fraction(1),
                Factor(Kind.FOCK, (p, q)),
                (Operator(p, True), Operator(q, False)),
            )
        )
    for spaces in product(Space, repeat=4):
        p, q, r, s = (dummies.new(space) for space in spaces)
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


def singles_cluster(dummies: DummyIndices) -> OperatorString:
    """``T1 = Σ t1[i, a] a†_a a_i``."""
    i, a = dummies.new(Space.OCCUPIED), dummies.new(Space.VIRTUAL)
    return OperatorString(
        Fraction(1),
        Factor(Kind.SINGLES, (i, a)),
        (Operator(a, True), Operator(i, False)),
    )


def doubles_cluster(dummies: DummyIndices) -> OperatorString:
    """``T2 = ¼ Σ t2[i, j, a, b] a†_a a†_b a_j a_i``."""
    i, j = dummies.new(Space.OCCUPIED), dummies.new(Space.OCCUPIED)
    a, b = dummies.new(Space.VIRTUAL), dummies.new(Space.VIRTUAL)
    return OperatorString(
        Fraction(1, 4),
        Factor(Kind.DOUBLES, (i, j, a, b)),
        (Operator(a, True), Operator(b, True), Operator(j, False), Operator(i, False)),
    )


def cluster_products(max_operators: int) -> list[tuple[Fraction, list[Cluster]]]:
    """List ``T1^m T2^n / (m! n!)`` up to ``max_operators`` operators.

    The empty product comes first.
    """
    products: list[tuple[Fraction, list[Cluster]]] = []
    for n in range(max_operators // 4 + 1):
        for m in range((max_operators - 4 * n) // 2 + 1):
            clusters: list[Cluster] = [singles_cluster] * m + [doubles_cluster] * n
            products.append((Fraction(1, factorial(m) * factorial(n)), clusters))
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

    @classmethod
    def triples(cls) -> Projection:
        """``<Φ_ijk^abc| = <Φ| a†_i a†_j a†_k a_c a_b a_a``."""
        occupied = [Index(Space.OCCUPIED, label, True) for label in "ijk"]
        virtual = [Index(Space.VIRTUAL, label, True) for label in "abc"]
        return cls(
            "triples",
            (
                *(Operator(index, True) for index in occupied),
                *(Operator(index, False) for index in reversed(virtual)),
            ),
            "ijkabc",
        )

    def expectation(
        self,
        name: str,
        products: Sequence[tuple[Fraction, Sequence[Cluster]]],
        *,
        connected: bool = True,
    ) -> Equation:
        """Derive ``<Φ_μ| H_N Σ products |Φ>`` as merged canonical terms.

        ``products`` holds coefficients and products of cluster operators;
        the terms in which every cluster operator is contracted with the
        Hamiltonian are kept, or with ``connected`` false the others.
        """
        dummies = DummyIndices()
        bra = OperatorString(Fraction(1), None, self.operators)
        terms: list[Term] = []
        for hamiltonian in hamiltonian_strings(dummies):
            for coefficient, clusters in products:
                strings = [
                    bra,
                    hamiltonian,
                    *(cluster(dummies) for cluster in clusters),
                ]
                terms.extend(
                    Term(term.coefficient * coefficient, term.factors)
                    for term in expectation(
                        strings, 1, range(2, len(strings)), connected=connected
                    )
                )
        return Equation(name, self.free, tuple(Term.merge(terms)))

    def equation(self) -> Equation:
        """Derive ``<Φ_μ| H_N e^T |Φ>_C`` as merged canonical terms."""
        # The projection and the Hamiltonian contract at most 4 + 4 operators.
        return self.expectation(self.name, cluster_products(len(self.operators) + 4))

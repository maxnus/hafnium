"""Equations as expectation values of operator strings over the Fermi vacuum.

With the normal-ordered Hamiltonian ``H_N = Σ f_pq {p† q} + ¼ Σ <pq||rs>
{p† q† s r}`` and the cluster operators ``T1 = Σ t_i^a a†_a a_i`` and
``T2 = ¼ Σ t_ij^ab a†_a a†_b a_j a_i``, an equation is ``<Φ_μ| H_N Π T |Φ>``
for a projection ``<Φ_μ|`` and products of cluster operators, keeping the
terms in which every cluster operator is contracted with the Hamiltonian
or the others. The general indices of ``H_N`` are expanded into occupied
and virtual ones, so that the terms refer to the blocks ``f[oo]``,
``f[ov]``, ... and ``v[oovv]``, ... of the Hamiltonian.
The equations of the methods are stated in these terms at the end.
:func:`ccsd_equations` (ADR-0032) derives

- ``E = <Φ| H_N e^T |Φ>_C``,
- ``0 = <Φ_i^a| H_N e^T |Φ>_C`` with ``<Φ_i^a| = <Φ| a†_i a_a``,
- ``0 = <Φ_ij^ab| H_N e^T |Φ>_C`` with ``<Φ_ij^ab| = <Φ| a†_i a†_j a_b a_a``,

the subscript marking the terms in which every cluster operator is
contracted with the Hamiltonian, ``e^T`` expanded into the products
``T1^m T2^n / (m! n!)`` with no more operators than the projection and
the Hamiltonian can contract. :func:`ccsd_pt_equations` (ADR-0033)
derives the connected triples ``W = <Φ_ijk^abc| H_N T2 |Φ>_C`` and the
disconnected ``V = <Φ_ijk^abc| H_N (T1 + T2) |Φ>`` of the (T) correction
``1/36 Σ conj(W + V) W / D`` (:mod:`hafnium.cc.ccsd_pt`), with
``<Φ_ijk^abc| = <Φ| a†_i a†_j a†_k a_c a_b a_a``. :func:`ccsd_lambda_equations`
(ADR-0034) derives the Λ equations ``0 = <Φ| (1 + Λ) (H_N e^T)_C |Φ_μ>``
of the de-excitation operators ``Λ1 = Σ l1[i, a] a†_i a_a`` and
``Λ2 = ¼ Σ l2[i, j, a, b] a†_i a†_j a_b a_a``, the ket the adjoint of
each projection and the unlinked terms left out, and the blocks of the
one-particle density ``<Φ| (1 + Λ) ({p† q} e^T)_C |Φ>``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from itertools import product
from math import factorial

from hafnium.cc.generator.wick import (
    LABELS,
    Factor,
    Index,
    Kind,
    Operator,
    OperatorString,
    Term,
    expectation,
)
from hafnium.orbitals import Space


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


class Cluster(Enum):
    """The cluster operators ``T1`` and ``T2`` and their adjoints ``Λ1`` and ``Λ2``.

    A member is the operator string of its amplitude kind over fresh
    dummy indices: ``T1 = Σ t1[i, a] a†_a a_i`` and ``T2 = ¼ Σ t2[i, j, a, b]
    a†_a a†_b a_j a_i`` excite, creating the virtual indices of the
    amplitude and annihilating the occupied ones, and ``Λ1 = Σ l1[i, a]
    a†_i a_a`` and ``Λ2 = ¼ Σ l2[i, j, a, b] a†_i a†_j a_b a_a`` de-excite,
    the adjoints of the two.
    """

    SINGLES = Kind.SINGLES
    DOUBLES = Kind.DOUBLES
    LAMBDA_SINGLES = Kind.LAMBDA_SINGLES
    LAMBDA_DOUBLES = Kind.LAMBDA_DOUBLES

    @property
    def kind(self) -> Kind:
        """The kind of the amplitude tensor."""
        return self.value

    @property
    def rank(self) -> int:
        """The electrons the operator moves, the occupied slots of its amplitude."""
        return 1 if self in (Cluster.SINGLES, Cluster.LAMBDA_SINGLES) else 2

    @property
    def n_operators(self) -> int:
        """The creation and annihilation operators of the string."""
        return 2 * self.rank

    @property
    def excites(self) -> bool:
        """Whether the operator excites the reference, or de-excites it."""
        return self in (Cluster.SINGLES, Cluster.DOUBLES)

    def string(self, dummies: DummyIndices) -> OperatorString:
        """Return the operator over indices of ``dummies`` not handed out before."""
        occupied = [dummies.new(Space.OCCUPIED) for _ in range(self.rank)]
        virtual = [dummies.new(Space.VIRTUAL) for _ in range(self.rank)]
        created, annihilated = (
            (virtual, occupied) if self.excites else (occupied, virtual)
        )
        return OperatorString(
            Fraction(1, factorial(self.rank) ** 2),
            Factor(self.kind, (*occupied, *virtual)),
            (
                *(Operator(index, True) for index in created),
                *(Operator(index, False) for index in reversed(annihilated)),
            ),
        )


DE_EXCITATIONS: tuple[Cluster | None, ...] = (
    None,
    Cluster.LAMBDA_SINGLES,
    Cluster.LAMBDA_DOUBLES,
)
"""``1 + Λ1 + Λ2``, the bra of the Λ functional, ``None`` for the one."""


def cluster_products(max_operators: int) -> list[tuple[Fraction, list[Cluster]]]:
    """List ``T1^m T2^n / (m! n!)`` up to ``max_operators`` operators.

    The empty product comes first.
    """
    products: list[tuple[Fraction, list[Cluster]]] = []
    for n in range(max_operators // 4 + 1):
        for m in range((max_operators - 4 * n) // 2 + 1):
            clusters = [Cluster.SINGLES] * m + [Cluster.DOUBLES] * n
            products.append((Fraction(1, factorial(m) * factorial(n)), clusters))
    return products


@dataclass(frozen=True, slots=True)
class Equation:
    """The merged canonical terms of an expectation value with free indices.

    ``free`` holds the labels of the free indices in the order of the
    result's axes, ``"ijab"`` for the doubles residual ``[i, j, a, b]``;
    ``antisymmetric`` says that the result changes sign under the odd
    permutations of its occupied and of its virtual free indices, as a
    residual does and a density block does not.
    """

    name: str
    free: str
    terms: tuple[Term, ...]
    antisymmetric: bool = True


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
                    *(cluster.string(dummies) for cluster in clusters),
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

    @property
    def ket(self) -> tuple[Operator, ...]:
        """``|Φ_μ>`` as the operators of the ket, the adjoint of the bra."""
        return tuple(
            Operator(op.index, not op.creation) for op in reversed(self.operators)
        )

    def lambda_equation(self) -> Equation:
        """Derive ``<Φ| (1 + Λ) (H_N e^T)_C |Φ_μ>`` linked, as merged canonical terms.

        The equation of the de-excitation amplitudes of this projection
        (ADR-0034): the ket is the adjoint of the bra, ``Λ = Λ1 + Λ2``
        stands to the left of the Hamiltonian, and the unlinked terms, in
        which the Hamiltonian with its cluster operators closes on itself
        and an amplitude multiplies the energy, are left out. The result
        has the free indices of the projection and the structure of the
        de-excitation amplitudes.
        """
        dummies = DummyIndices()
        ket = OperatorString(Fraction(1), None, self.ket)
        terms: list[Term] = []
        for left in DE_EXCITATIONS:
            # The quasi-annihilators of Λ and of the Hamiltonian, at most
            # four, contract every quasi-creator of the ket and the clusters.
            n_left = 0 if left is None else left.n_operators
            products = cluster_products(n_left + 4 - len(self.operators))
            for hamiltonian in hamiltonian_strings(dummies):
                terms.extend(
                    _de_excited(left, hamiltonian, products, ket, dummies, linked=True)
                )
        return Equation(f"lambda_{self.name}", self.free, tuple(Term.merge(terms)))


def _de_excited(
    left: Cluster | None,
    operator: OperatorString,
    products: Sequence[tuple[Fraction, Sequence[Cluster]]],
    ket: OperatorString | None,
    dummies: DummyIndices,
    *,
    linked: bool,
) -> list[Term]:
    """Terms of ``<Φ| left operator Σ products ket |Φ>`` connected to ``operator``.

    ``left`` is a de-excitation or ``None`` for the one, ``ket`` an
    excitation or ``None`` for the reference; the clusters of every
    product must be contracted with ``operator``, and with ``linked`` the
    terms in which they close on it are dropped (:func:`expectation`).
    """
    terms: list[Term] = []
    for coefficient, clusters in products:
        strings = [operator, *(cluster.string(dummies) for cluster in clusters)]
        if ket is not None:
            strings.append(ket)
        position = 0
        if left is not None:
            strings.insert(0, left.string(dummies))
            position = 1
        end = len(strings) - (ket is not None)
        terms.extend(
            Term(term.coefficient * coefficient, term.factors)
            for term in expectation(
                strings, position, range(position + 1, end), linked=linked
            )
        )
    return terms


def density_block(p: Space, q: Space) -> Equation:
    """Derive ``<Φ| (1 + Λ) ({p† q} e^T)_C |Φ>`` over the spaces of ``p`` and ``q``.

    The block of the one-particle density of CCSD (ADR-0034) that the
    normal-ordered ``{p† q}`` contributes, as the tensor ``[p, q]`` with
    ``q`` conjugated, so that it pairs with ``f[p, q]`` in a trace,
    without the occupations of the reference; it is not antisymmetric in
    its indices.
    """
    first = Index(p, LABELS[p][0], True)
    second = Index(q, LABELS[q][1 if q is p else 0], True)
    operator = OperatorString.normal_ordered(
        Fraction(1), None, (Operator(first, True), Operator(second, False))
    )
    creators = sum(op.creates_quasiparticle for op in operator.operators)
    dummies = DummyIndices()
    terms: list[Term] = []
    for left in DE_EXCITATIONS:
        # The quasi-annihilators of Λ and of the operator contract every
        # quasi-creator of the operator and the clusters.
        n_left = 0 if left is None else left.n_operators
        products = cluster_products(max(n_left + 2 - 2 * creators, 0))
        terms.extend(_de_excited(left, operator, products, None, dummies, linked=False))
    return Equation(
        f"density_{p.value}_{q.value}",
        first.label + second.label,
        tuple(Term.merge(terms)),
        antisymmetric=False,
    )


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


def ccsd_pt_equations() -> dict[str, Equation]:
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
            "connected_triples", [(one, [Cluster.DOUBLES])]
        ),
        "disconnected_triples": projection.expectation(
            "disconnected_triples",
            [(one, [Cluster.SINGLES]), (one, [Cluster.DOUBLES])],
            connected=False,
        ),
    }


def ccsd_lambda_equations() -> dict[str, Equation]:
    """Derive the Λ equations of CCSD and the blocks of its density, by name.

    ``lambda_singles`` and ``lambda_doubles`` are the residuals of the
    de-excitation amplitudes, ``density_<p>_<q>`` the blocks of
    ``<Φ| (1 + Λ) ({p† q} e^T)_C |Φ>`` over the spaces of ``p`` and ``q``.
    """
    equations = [
        Projection.singles().lambda_equation(),
        Projection.doubles().lambda_equation(),
        *(density_block(p, q) for p, q in product(Space, repeat=2)),
    ]
    return {equation.name: equation for equation in equations}

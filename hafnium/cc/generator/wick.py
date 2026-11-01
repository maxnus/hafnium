"""Wick contractions of normal-ordered operator strings over the Fermi vacuum.

The working equations of coupled-cluster theory are vacuum expectation
values of products of normal-ordered strings of second-quantized
operators, ``<Φ| P H_N T T ... |Φ>``, with respect to the reference
determinant ``|Φ>``. Wick's theorem writes such a product as the sum over
all sets of pairwise contractions between operators of different strings,
and the expectation value keeps the fully contracted sets. Relative to
``|Φ>`` an operator creates a quasi-particle (``a†`` of a virtual orbital,
``a_i`` of an occupied one) or annihilates one, and the contraction of
two operators is non-zero only for a quasi-annihilator to the left of a
quasi-creator over the same orbital: ``a_a a†_b = δ_ab`` and
``a†_i a_j = δ_ij``.

A contracted term is a coefficient times a product of tensor factors
whose indices the contractions have identified. Terms are brought to a
canonical form, the dummy indices relabelled and the factors ordered so
that equal terms coincide, with the antisymmetry of the two-electron
integrals and of the doubles amplitudes taken into account, and merged.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from itertools import permutations, product

from hafnium.orbitals import Space

LABELS = {Space.OCCUPIED: "ijklmn", Space.VIRTUAL: "abcdef"}
"""The labels of the indices of each space, free ones first."""

SPACE_NAMES = {Space.OCCUPIED: "OCC", Space.VIRTUAL: "VIR"}
"""The names a program gives the spaces when keying a Hamiltonian block."""


class Kind(Enum):
    """The tensors a term is a product of, by the name they carry in a program."""

    FOCK = "f"
    INTEGRAL = "v"
    SINGLES = "t1"
    DOUBLES = "t2"
    LAMBDA_SINGLES = "l1"
    LAMBDA_DOUBLES = "l2"

    @property
    def antisymmetric_pairs(self) -> tuple[tuple[int, int], ...]:
        """Slot pairs that change the sign of the tensor when swapped.

        ``v[p, q, r, s] = <pq||rs>`` is antisymmetric in ``p, q`` and in
        ``r, s``, ``t2[i, j, a, b]`` and ``l2[i, j, a, b]`` in ``i, j`` and
        in ``a, b``.
        """
        if self in (Kind.INTEGRAL, Kind.DOUBLES, Kind.LAMBDA_DOUBLES):
            return ((0, 1), (2, 3))
        return ()

    @property
    def order(self) -> int:
        """Position of the kind among the factors of a canonical term."""
        return {
            Kind.FOCK: 0,
            Kind.INTEGRAL: 0,
            Kind.SINGLES: 1,
            Kind.DOUBLES: 2,
            Kind.LAMBDA_SINGLES: 3,
            Kind.LAMBDA_DOUBLES: 4,
        }[self]

    @property
    def of_hamiltonian(self) -> bool:
        """Whether the tensor is a block of the Hamiltonian, keyed by its slots."""
        return self in (Kind.FOCK, Kind.INTEGRAL)


@dataclass(frozen=True, slots=True)
class Index:
    """An orbital index of ``space``; a free index keeps its ``label``."""

    space: Space
    label: str
    free: bool = False


@dataclass(frozen=True, slots=True)
class Operator:
    """``a†`` (``creation``) or ``a`` of the orbital ``index``."""

    index: Index
    creation: bool

    @property
    def creates_quasiparticle(self) -> bool:
        """Whether the operator creates a quasi-particle on the reference.

        ``a†`` of a virtual orbital adds a particle, ``a`` of an occupied
        orbital a hole; the other two annihilate one.
        """
        return self.creation == (self.index.space is Space.VIRTUAL)

    def contracts_with(self, right: Operator) -> bool:
        """Whether ``self right`` has a non-vanishing contraction over the reference."""
        return (
            self.index.space is right.index.space
            and not self.creates_quasiparticle
            and right.creates_quasiparticle
        )


@dataclass(frozen=True, slots=True)
class Factor:
    """A tensor of ``kind`` with its ``indices`` in slot order."""

    kind: Kind
    indices: tuple[Index, ...]

    @property
    def slots(self) -> tuple[Space, ...]:
        """The spaces of the slots, the key of a Hamiltonian block."""
        return tuple(index.space for index in self.indices)

    @property
    def letters(self) -> str:
        """The labels of the indices in slot order, the subscripts of a program."""
        return "".join(index.label for index in self.indices)

    @property
    def expression(self) -> str:
        """The tensor in a program: ``f[OCC, VIR]``, ...

        The Hamiltonian blocks are keyed by the names of their slot spaces.
        """
        if self.kind.of_hamiltonian:
            return f"{self.kind.value}[{', '.join(SPACE_NAMES[s] for s in self.slots)}]"
        return self.kind.value

    def variants(self) -> list[tuple[int, Factor]]:
        """List the factor under each combination of its antisymmetric swaps, signed."""
        variants: list[tuple[int, Factor]] = [(1, self)]
        for pair in self.kind.antisymmetric_pairs:
            swapped: list[tuple[int, Factor]] = []
            for sign, factor in variants:
                indices = list(factor.indices)
                indices[pair[0]], indices[pair[1]] = indices[pair[1]], indices[pair[0]]
                swapped.append((-sign, Factor(factor.kind, tuple(indices))))
            variants.extend(swapped)
        return variants

    def __str__(self) -> str:
        return f"{self.kind.value}[{','.join(index.label for index in self.indices)}]"


@dataclass(frozen=True, slots=True)
class OperatorString:
    """``coefficient × factor × operators``, the operators in normal order.

    Build it with :meth:`normal_ordered`, which supplies the sign of the
    reordering; a string in normal order has no contractions within
    itself, so its operators can be written as the product they are.
    """

    coefficient: Fraction
    factor: Factor | None
    operators: tuple[Operator, ...]

    @classmethod
    def normal_ordered(
        cls,
        coefficient: Fraction,
        factor: Factor | None,
        operators: Sequence[Operator],
    ) -> OperatorString:
        """Bring ``operators`` to normal order, quasi-creators first, with the sign.

        The normal-ordered product ``{A B C ...}`` equals the reordered
        product times the parity of the permutation; the relative order
        within the creators and within the annihilators is kept.
        """
        creators = [op for op in operators if op.creates_quasiparticle]
        annihilators = [op for op in operators if not op.creates_quasiparticle]
        inversions = 0
        annihilators_seen = 0
        for op in operators:
            if op.creates_quasiparticle:
                inversions += annihilators_seen
            else:
                annihilators_seen += 1
        sign = -1 if inversions % 2 else 1
        return cls(coefficient * sign, factor, (*creators, *annihilators))


Key = tuple[object, ...]


@dataclass(frozen=True, slots=True)
class Term:
    """``coefficient × Π factors`` with summation over the indices that are not free."""

    coefficient: Fraction
    factors: tuple[Factor, ...]

    @property
    def free_indices(self) -> tuple[Index, ...]:
        """The free indices in order of first appearance."""
        seen: list[Index] = []
        for factor in self.factors:
            for index in factor.indices:
                if index.free and index not in seen:
                    seen.append(index)
        return tuple(seen)

    @property
    def key(self) -> Key:
        """The comparable form of the factors that canonical terms are merged by."""
        return tuple(
            (
                factor.kind.value,
                tuple((i.space.value, i.label, i.free) for i in factor.indices),
            )
            for factor in self.factors
        )

    def __str__(self) -> str:
        return f"{self.coefficient} {' '.join(str(factor) for factor in self.factors)}"

    def canonical(self) -> Term:
        """Relabel the dummies and order the factors so that equal terms coincide.

        Every ordering of the factors of equal kind and every sign-carrying
        permutation of the antisymmetric slots is tried, the dummies are
        relabelled in order of appearance, and the smallest representation
        is taken; its sign goes into the coefficient.
        """
        free = {index.label for index in self.free_indices}
        pools = {
            space: [c for c in labels if c not in free]
            for space, labels in LABELS.items()
        }
        best: tuple[Key, int, tuple[Factor, ...]] | None = None
        for ordering in _orderings(self.factors):
            for variant in product(*(factor.variants() for factor in ordering)):
                sign = 1
                for s, _ in variant:
                    sign *= s
                factors = _relabelled([f for _, f in variant], pools)
                form = Term(self.coefficient, factors).key
                if best is None or form < best[0]:
                    best = (form, sign, factors)
        assert best is not None
        return Term(self.coefficient * best[1], best[2])

    def substituted(self, mapping: dict[str, str]) -> Term:
        """Rename the free indices by label."""
        return Term(
            self.coefficient,
            tuple(
                Factor(
                    factor.kind,
                    tuple(
                        Index(index.space, mapping.get(index.label, index.label), True)
                        if index.free
                        else index
                        for index in factor.indices
                    ),
                )
                for factor in self.factors
            ),
        )

    @classmethod
    def merge(cls, terms: Iterable[Term]) -> list[Term]:
        """Canonicalize ``terms`` and add up those that coincide, dropping zeros."""
        totals: dict[Key, Term] = {}
        for term in terms:
            form = term.canonical()
            previous = totals.get(form.key)
            totals[form.key] = (
                form
                if previous is None
                else Term(previous.coefficient + form.coefficient, form.factors)
            )
        return [term for term in totals.values() if term.coefficient != 0]


# -------------------------------------------------------------- contractions


def full_contractions(
    groups: Sequence[Sequence[Operator]],
) -> Iterator[tuple[int, list[tuple[int, int]]]]:
    """Enumerate the full contractions of the product of the operator ``groups``.

    Operators are numbered through the concatenated product. Only
    operators of different groups contract, since every group is in
    normal order. Yields the sign of each full contraction with its pairs
    ``(left, right)``: the leftmost uncontracted operator is paired with
    an operator to its right, which brings the pair together across the
    uncontracted operators in between, one sign change each.
    """
    operators = [(g, op) for g, group in enumerate(groups) for op in group]

    def recurse(
        remaining: list[int], sign: int, pairs: list[tuple[int, int]]
    ) -> Iterator[tuple[int, list[tuple[int, int]]]]:
        if not remaining:
            yield sign, pairs
            return
        first = remaining[0]
        group_first, op_first = operators[first]
        if op_first.creates_quasiparticle:
            return
        for offset, other in enumerate(remaining[1:]):
            group_other, op_other = operators[other]
            if group_other == group_first or not op_first.contracts_with(op_other):
                continue
            rest = [k for k in remaining if k not in (first, other)]
            yield from recurse(
                rest, sign * (-1 if offset % 2 else 1), [*pairs, (first, other)]
            )

    yield from recurse(list(range(len(operators))), 1, [])


def expectation(
    strings: Sequence[OperatorString],
    hamiltonian: int,
    clusters: Iterable[int],
    *,
    connected: bool = True,
    linked: bool = False,
) -> list[Term]:
    """``<Φ| Π strings |Φ>`` keeping the terms connected to the Hamiltonian.

    A term is connected when every string at a position in ``clusters``
    is contracted with the string at position ``hamiltonian`` at least
    once, the connectedness of the coupled-cluster equations; the other
    strings (the projection) contract freely. With ``connected`` false
    the terms that are not connected are kept instead, the disconnected
    products the (T) correction needs. With ``linked`` the terms in which
    the Hamiltonian and the clusters form a closed part, contracted with
    none of the other strings, are dropped as well: the unlinked products
    of the Λ equations, a de-excitation amplitude times the energy. The
    terms are returned uncanonicalized.
    """
    operators = [(s, op) for s, string in enumerate(strings) for op in string.operators]
    required = set(clusters)
    coefficient = Fraction(1)
    for string in strings:
        coefficient *= string.coefficient
    terms: list[Term] = []
    for sign, pairs in full_contractions([string.operators for string in strings]):
        touching = {
            operators[q][0] if operators[p][0] == hamiltonian else operators[p][0]
            for p, q in pairs
            if hamiltonian in (operators[p][0], operators[q][0])
        }
        if (required <= touching) != connected:
            continue
        if linked and _component(hamiltonian, pairs, operators) <= required | {
            hamiltonian
        }:
            continue
        representative = _identified(operators, pairs)
        factors = tuple(
            Factor(
                string.factor.kind,
                tuple(
                    representative.get(index, index) for index in string.factor.indices
                ),
            )
            for string in strings
            if string.factor is not None
        )
        terms.append(Term(coefficient * sign, factors))
    return terms


def _component(
    start: int,
    pairs: Iterable[tuple[int, int]],
    operators: Sequence[tuple[int, Operator]],
) -> set[int]:
    """Collect the strings reachable from ``start`` through the contractions."""
    neighbours: dict[int, set[int]] = {}
    for p, q in pairs:
        a, b = operators[p][0], operators[q][0]
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    component = {start}
    frontier = [start]
    while frontier:
        for other in neighbours.get(frontier.pop(), ()):
            if other not in component:
                component.add(other)
                frontier.append(other)
    return component


def _identified(
    operators: Sequence[tuple[int, Operator]], pairs: Iterable[tuple[int, int]]
) -> dict[Index, Index]:
    """Map every contracted index onto its class representative, a free one if any."""
    parent: dict[Index, Index] = {}

    def find(index: Index) -> Index:
        while parent.get(index, index) != index:
            index = parent[index]
        return index

    for p, q in pairs:
        a, b = find(operators[p][1].index), find(operators[q][1].index)
        if a != b:
            if b.free:
                a, b = b, a
            parent[b] = a
    return {index: find(index) for index in parent}


# ----------------------------------------------------------- canonical form


def _orderings(factors: Sequence[Factor]) -> Iterator[tuple[Factor, ...]]:
    """Every ordering of the factors that keeps the kinds in :attr:`Kind.order`."""
    kinds = sorted(
        {factor.kind for factor in factors}, key=lambda k: (k.order, k.value)
    )
    groups = [[f for f in factors if f.kind is kind] for kind in kinds]
    for arrangement in product(*(permutations(group) for group in groups)):
        yield tuple(f for group in arrangement for f in group)


def _relabelled(
    factors: Sequence[Factor], pools: dict[Space, list[str]]
) -> tuple[Factor, ...]:
    """Give the dummies labels from ``pools`` in order of first appearance."""
    names: dict[Index, Index] = {}
    used: dict[Space, int] = dict.fromkeys(Space, 0)
    result: list[Factor] = []
    for factor in factors:
        indices: list[Index] = []
        for index in factor.indices:
            if index.free:
                indices.append(index)
                continue
            if index not in names:
                label = pools[index.space][used[index.space]]
                used[index.space] += 1
                names[index] = Index(index.space, label)
            indices.append(names[index])
        result.append(Factor(factor.kind, tuple(indices)))
    return tuple(result)

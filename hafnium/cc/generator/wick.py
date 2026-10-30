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
from fractions import Fraction
from itertools import permutations, product

OCCUPIED = "o"
VIRTUAL = "v"
SPACES = (OCCUPIED, VIRTUAL)

OCCUPIED_LABELS = "ijklmn"
VIRTUAL_LABELS = "abcdef"
LABELS = {OCCUPIED: OCCUPIED_LABELS, VIRTUAL: VIRTUAL_LABELS}

ANTISYMMETRIC_PAIRS: dict[str, tuple[tuple[int, int], ...]] = {
    "v": ((0, 1), (2, 3)),
    "t2": ((0, 1), (2, 3)),
}
"""Slot pairs of a factor that change its sign when swapped.

``v[p, q, r, s] = <pq||rs>`` is antisymmetric in ``p, q`` and in ``r, s``,
``t2[i, j, a, b]`` in ``i, j`` and in ``a, b``.
"""

FACTOR_ORDER = {"f": 0, "v": 0, "t1": 1, "t2": 2}
"""Order of the factors of a canonical term: the Hamiltonian first."""


@dataclass(frozen=True, slots=True)
class Index:
    """An orbital index of ``space``; a free index keeps its ``label``."""

    space: str
    label: str
    free: bool = False

    def __post_init__(self) -> None:
        if self.space not in SPACES:
            raise ValueError(f"an index is occupied or virtual, got {self.space!r}")


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
        return self.creation == (self.index.space == VIRTUAL)


def contractible(left: Operator, right: Operator) -> bool:
    """Whether ``left right`` has a non-vanishing contraction over the reference."""
    return (
        left.index.space == right.index.space
        and not left.creates_quasiparticle
        and right.creates_quasiparticle
    )


@dataclass(frozen=True, slots=True)
class Factor:
    """A tensor ``name`` with its ``indices`` in slot order."""

    name: str
    indices: tuple[Index, ...]


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
            if group_other == group_first or not contractible(op_first, op_other):
                continue
            rest = [k for k in remaining if k not in (first, other)]
            yield from recurse(
                rest, sign * (-1 if offset % 2 else 1), [*pairs, (first, other)]
            )

    yield from recurse(list(range(len(operators))), 1, [])


def connected_expectation(
    strings: Sequence[OperatorString], hamiltonian: int, clusters: Iterable[int]
) -> list[Term]:
    """``<Φ| Π strings |Φ>`` keeping the terms connected to the Hamiltonian.

    Every string at a position in ``clusters`` must be contracted with the
    string at position ``hamiltonian`` at least once, the connectedness of
    the coupled-cluster equations; the other strings (the projection)
    contract freely. The terms are returned uncanonicalized.
    """
    operators = [(s, op) for s, string in enumerate(strings) for op in string.operators]
    required = set(clusters)
    coefficient = Fraction(1)
    for string in strings:
        coefficient *= string.coefficient
    terms: list[Term] = []
    for sign, pairs in full_contractions([string.operators for string in strings]):
        linked = {
            operators[q][0] if operators[p][0] == hamiltonian else operators[p][0]
            for p, q in pairs
            if hamiltonian in (operators[p][0], operators[q][0])
        }
        if not required <= linked:
            continue
        representative = _identified(operators, pairs)
        factors = tuple(
            Factor(
                string.factor.name,
                tuple(
                    representative.get(index, index) for index in string.factor.indices
                ),
            )
            for string in strings
            if string.factor is not None
        )
        terms.append(Term(coefficient * sign, factors))
    return terms


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


def canonical(term: Term) -> Term:
    """Relabel the dummy indices and order the factors so that equal terms coincide.

    Every ordering of the factors of equal kind and every sign-carrying
    permutation of the antisymmetric slots is tried, the dummies are
    relabelled in order of appearance, and the smallest representation is
    taken; its sign goes into the coefficient.
    """
    free = {index.label for index in term.free_indices}
    pools = {space: [c for c in LABELS[space] if c not in free] for space in SPACES}
    best: tuple[tuple[object, ...], int, tuple[Factor, ...]] | None = None
    for ordering in _orderings(term.factors):
        for variant in product(*(_variants(factor) for factor in ordering)):
            sign = 1
            for s, _ in variant:
                sign *= s
            factors = _relabelled([f for _, f in variant], pools)
            form = key(factors)
            if best is None or form < best[0]:
                best = (form, sign, factors)
    assert best is not None
    return Term(term.coefficient * best[1], best[2])


def merged(terms: Iterable[Term]) -> list[Term]:
    """Canonicalize ``terms`` and add up those that coincide, dropping zeros."""
    totals: dict[tuple[object, ...], Term] = {}
    for term in terms:
        form = canonical(term)
        form_key = key(form.factors)
        previous = totals.get(form_key)
        totals[form_key] = (
            form
            if previous is None
            else Term(previous.coefficient + form.coefficient, form.factors)
        )
    return [term for term in totals.values() if term.coefficient != 0]


def substituted(term: Term, mapping: dict[str, str]) -> Term:
    """Rename the free indices of ``term`` by label."""
    return Term(
        term.coefficient,
        tuple(
            Factor(
                factor.name,
                tuple(
                    Index(index.space, mapping.get(index.label, index.label), True)
                    if index.free
                    else index
                    for index in factor.indices
                ),
            )
            for factor in term.factors
        ),
    )


def _orderings(factors: Sequence[Factor]) -> Iterator[tuple[Factor, ...]]:
    """Every ordering of the factors that keeps the kinds in :data:`FACTOR_ORDER`."""
    kinds = sorted(
        {factor.name for factor in factors}, key=lambda n: (FACTOR_ORDER[n], n)
    )
    groups = [[f for f in factors if f.name == kind] for kind in kinds]
    for arrangement in product(*(permutations(group) for group in groups)):
        yield tuple(f for group in arrangement for f in group)


def _variants(factor: Factor) -> list[tuple[int, Factor]]:
    """List the factor under every combination of its antisymmetric swaps, signed."""
    variants = [(1, factor)]
    for pair in ANTISYMMETRIC_PAIRS.get(factor.name, ()):
        swapped: list[tuple[int, Factor]] = []
        for sign, f in variants:
            indices = list(f.indices)
            indices[pair[0]], indices[pair[1]] = indices[pair[1]], indices[pair[0]]
            swapped.append((-sign, Factor(f.name, tuple(indices))))
        variants.extend(swapped)
    return variants


def _relabelled(
    factors: Sequence[Factor], pools: dict[str, list[str]]
) -> tuple[Factor, ...]:
    """Give the dummies labels from ``pools`` in order of first appearance."""
    names: dict[Index, Index] = {}
    used: dict[str, int] = dict.fromkeys(SPACES, 0)
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
        result.append(Factor(factor.name, tuple(indices)))
    return tuple(result)


def key(factors: Sequence[Factor]) -> tuple[object, ...]:
    """Return the comparable form of ``factors`` that canonical terms are merged by."""
    return tuple(
        (factor.name, tuple((i.space, i.label, i.free) for i in factor.indices))
        for factor in factors
    )


def describe(term: Term) -> str:
    """``coefficient name[labels] ...`` for messages and tests."""
    factors = " ".join(
        f"{factor.name}[{','.join(index.label for index in factor.indices)}]"
        for factor in term.factors
    )
    return f"{term.coefficient} {factors}"

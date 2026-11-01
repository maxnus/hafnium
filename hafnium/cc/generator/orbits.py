"""Terms grouped by the permutations of the free indices of an equation.

A residual is antisymmetric in its occupied free indices and in its
virtual ones, so its terms fall into orbits under the permutations of
those indices: a term and the images of its free indices, each image
carrying the sign of the permutation. One term per orbit is evaluated
by the generated program and the images are its permuted copies; the
images under the occupied and under the virtual permutations alone are
collected as well, for the factored sums of the triples (ADR-0033). A
result without antisymmetry, a density block, has one orbit per term.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

from hafnium.cc.generator.equations import Equation
from hafnium.cc.generator.wick import LABELS, Term
from hafnium.orbitals import Space

Permutation = tuple[tuple[str, str], ...]
"""The free indices a permutation moves, as pairs ``(from, to)``."""

Cosets = tuple[tuple[Permutation, int], ...]
"""Permutations with their parities, the identity first."""

IDENTITY: Cosets = (((), 1),)
"""The group of the identity alone, for a result without antisymmetry."""


def permutations_of(free: str) -> Cosets:
    """List the permutations of the free indices ``free`` with their parities.

    Every permutation of the occupied letters with every permutation of
    the virtual letters, the identity first; the parity is the product
    of the parities.
    """
    occupied, virtual = _letters_of(free)
    return tuple(
        ((*occupied_moved, *virtual_moved), occupied_parity * virtual_parity)
        for virtual_moved, virtual_parity in _permutations_within(virtual)
        for occupied_moved, occupied_parity in _permutations_within(occupied)
    )


def _letters_of(free: str) -> tuple[str, str]:
    """Split the free indices into the occupied and the virtual letters."""
    return (
        "".join(c for c in free if c in LABELS[Space.OCCUPIED]),
        "".join(c for c in free if c in LABELS[Space.VIRTUAL]),
    )


def _permutations_within(letters: str) -> Cosets:
    """List the permutations of ``letters`` alone with parities, identity first."""
    return tuple(
        (
            tuple(
                (source, target)
                for source, target in zip(letters, image, strict=True)
                if source != target
            ),
            _parity(image),
        )
        for image in permutations(letters)
    )


def _parity(image: tuple[str, ...]) -> int:
    inversions = sum(
        1 for p, q in combinations(range(len(image)), 2) if image[p] > image[q]
    )
    return -1 if inversions % 2 else 1


ANTISYMMETRIZERS: dict[frozenset[Permutation], str] = {
    frozenset({()}): "",
    frozenset({(), (("i", "j"), ("j", "i"))}): "_minus_ij",
    frozenset({(), (("a", "b"), ("b", "a"))}): "_minus_ab",
    frozenset({p for p, _ in permutations_of("ijab")}): "_minus_both",
}
"""The helper of a generated module applying the permutations of an orbit's cosets.

The helpers are defined by :data:`~hafnium.cc.generator.modules.CCSD_HELPERS`.
"""


@dataclass(frozen=True, slots=True)
class Orbit:
    """A representative term and the coset representatives of its images.

    ``cosets`` are representatives under the permutations of all free
    indices, ``occupied_cosets`` and ``virtual_cosets`` under those of
    the occupied and of the virtual indices alone.
    """

    term: Term
    cosets: Cosets
    occupied_cosets: Cosets
    virtual_cosets: Cosets

    @property
    def antisymmetrizer(self) -> str:
        """Name of the helper applying the coset representatives, empty for none."""
        return ANTISYMMETRIZERS[frozenset(p for p, _ in self.cosets)]

    @property
    def factored(self) -> bool:
        """Whether the images are the occupied images permuted by the virtual ones.

        That is the case when the stabilizer of the term is the product
        of its occupied and its virtual parts, so that the sum over the
        cosets factors as ``Σ_τ ε(τ) τ Σ_σ ε(σ) σ``; the number of
        cosets is then the product of the two counts.
        """
        return len(self.cosets) == len(self.occupied_cosets) * len(self.virtual_cosets)


def orbits(equation: Equation) -> list[Orbit]:
    """Group the terms of ``equation`` by the permutations of its free indices.

    The residual ``R`` satisfies ``R = ε(g) g(R)`` for every permutation
    ``g`` of its antisymmetric index pairs, so the coefficient of the
    image of a term under ``g`` is fixed by the term's own, which is
    checked; the orbit is then ``c Σ_cosets ε(g) g(x)`` over the images
    with distinct canonical forms, ``x`` the representative evaluated.
    The images under the occupied and under the virtual permutations
    alone are collected the same way, for the factored form. A result
    without antisymmetry, a density block, has one orbit per term.

    Raises
    ------
    ValueError
        If a term lacks an image, or an image has another coefficient
        than the antisymmetry of the residual requires.
    """
    by_key = {term.key: term for term in equation.terms}
    done: set[tuple[object, ...]] = set()
    result: list[Orbit] = []
    occupied, virtual = _letters_of(equation.free)
    if equation.antisymmetric:
        group = permutations_of(equation.free)
        within = (_permutations_within(occupied), _permutations_within(virtual))
    else:
        group = IDENTITY
        within = (IDENTITY, IDENTITY)
    for term in sorted(equation.terms, key=lambda t: t.key):
        if term.key in done:
            continue
        cosets: list[tuple[Permutation, int]] = []
        for permutation, parity in group:
            image = term.substituted(dict(permutation)).canonical()
            expected = parity * image.coefficient
            actual = by_key.get(image.key)
            if actual is None or actual.coefficient != expected:
                raise ValueError(
                    "the residual is not antisymmetric: the image of a term under "
                    f"{permutation} has coefficient {actual and actual.coefficient}, "
                    f"expected {expected}"
                )
            if image.key not in done:
                done.add(image.key)
                cosets.append((permutation, parity))
        result.append(
            Orbit(
                term,
                tuple(cosets),
                _cosets_within(term, within[0]),
                _cosets_within(term, within[1]),
            )
        )
    return result


def _cosets_within(term: Term, group: Cosets) -> Cosets:
    """Return the permutations of ``group`` with distinct images of ``term``."""
    seen: set[tuple[object, ...]] = set()
    cosets: list[tuple[Permutation, int]] = []
    for permutation, parity in group:
        key = term.substituted(dict(permutation)).canonical().key
        if key not in seen:
            seen.add(key)
            cosets.append((permutation, parity))
    return tuple(cosets)

"""The equation generator: Wick contractions, canonical terms, the emitted program."""

from fractions import Fraction
from pathlib import Path

import pytest

import hafnium.cc._ccsd_equations as generated
from hafnium.cc.generator.ccsd import (
    Fresh,
    ccsd_equations,
    hamiltonian_strings,
    singles_cluster,
    singles_projection,
)
from hafnium.cc.generator.emit import module_source, orbits, program
from hafnium.cc.generator.wick import (
    Factor,
    Index,
    Operator,
    OperatorString,
    Term,
    canonical,
    connected_expectation,
    describe,
    full_contractions,
    merged,
)

I, J, K = (Index("o", label, True) for label in "ijk")
A, B = (Index("v", label, True) for label in "ab")


@pytest.fixture(scope="module")
def equations() -> dict[str, list[Term]]:
    return ccsd_equations()


def test_committed_equations_are_current() -> None:
    """The generated module is regenerated to detect drift (PLAN §5.5)."""
    source = Path(generated.__file__).read_text(encoding="utf-8")
    assert source == module_source()


def test_energy_is_the_known_expression(equations: dict[str, list[Term]]) -> None:
    assert sorted(describe(term) for term in equations["energy"]) == [
        "1 f[i,a] t1[i,a]",
        "1/2 v[i,j,a,b] t1[i,a] t1[j,b]",
        "1/4 v[i,j,a,b] t2[i,j,a,b]",
    ]


def test_singles_equation_has_the_fourteen_terms(
    equations: dict[str, list[Term]],
) -> None:
    terms = {describe(term) for term in equations["singles"]}
    assert len(terms) == 14
    assert {
        "1 f[a,i]",
        "1 f[a,b] t1[i,b]",
        "-1 f[j,i] t1[j,a]",
        "-1 v[j,a,i,b] t1[j,b]",
        "1 f[j,b] t2[i,j,a,b]",
        "-1/2 v[j,a,b,c] t2[i,j,b,c]",
        "-1/2 v[j,k,i,b] t2[j,k,a,b]",
        "-1 f[j,b] t1[i,b] t1[j,a]",
        "1 v[j,k,b,c] t1[j,b] t2[i,k,a,c]",
        "-1 v[j,k,b,c] t1[i,b] t1[j,a] t1[k,c]",
    } <= terms


def test_doubles_equation_has_the_thirty_one_orbits(
    equations: dict[str, list[Term]],
) -> None:
    terms = equations["doubles"]
    assert len(terms) == 63
    groups = orbits(terms, "ijab")
    assert len(groups) == 31
    helpers = {describe(group.term): group.antisymmetrizer for group in groups}
    assert helpers["1 v[a,b,i,j]"] == ""
    assert helpers["1/2 v[a,b,c,d] t2[i,j,c,d]"] == ""
    assert helpers["1/2 v[k,l,i,j] t2[k,l,a,b]"] == ""
    assert helpers["1/4 v[k,l,c,d] t2[i,j,c,d] t2[k,l,a,b]"] == ""
    assert helpers["1 f[k,i] t2[j,k,a,b]"] == "_minus_ij"
    assert helpers["-1 f[a,c] t2[i,j,b,c]"] == "_minus_ab"
    assert helpers["-1 v[k,a,i,c] t2[j,k,b,c]"] == "_minus_both"
    assert sum(len(group.cosets) for group in groups) == 63
    assert all(
        {"", "_minus_ij", "_minus_ab", "_minus_both"} >= {g.antisymmetrizer}
        for g in groups
    )


def test_single_excitation_overlaps_with_itself() -> None:
    """``<Φ| a†_i a_a a†_a a_i |Φ> = 1``: one full contraction with a plus sign."""
    bra = (Operator(I, True), Operator(A, False))
    ket = (Operator(Index("v", "c"), True), Operator(Index("o", "k"), False))
    contractions = list(full_contractions([bra, ket]))
    assert contractions == [(1, [(0, 3), (1, 2)])]
    # The crossed pairing of the doubles bra with its ket carries the sign.
    bra = (Operator(I, True), Operator(J, True), Operator(B, False), Operator(A, False))
    c, d = Index("v", "c"), Index("v", "d")
    k, m = Index("o", "k"), Index("o", "m")
    ket = (Operator(c, True), Operator(d, True), Operator(m, False), Operator(k, False))
    signs = {tuple(pairs): sign for sign, pairs in full_contractions([bra, ket])}
    assert signs[(0, 7), (1, 6), (2, 5), (3, 4)] == 1
    assert signs[(0, 6), (1, 7), (2, 5), (3, 4)] == -1
    assert len(signs) == 4


def test_normal_ordering_carries_the_sign() -> None:
    string = OperatorString.normal_ordered(
        Fraction(1), None, (Operator(I, True), Operator(J, False))
    )
    assert string.coefficient == -1
    assert string.operators == (Operator(J, False), Operator(I, True))
    string = OperatorString.normal_ordered(
        Fraction(1), None, (Operator(A, True), Operator(B, False))
    )
    assert string.coefficient == 1
    assert string.operators == (Operator(A, True), Operator(B, False))
    # Two annihilators moved across two creators: an even permutation.
    k, m = Index("o", "k"), Index("o", "m")
    string = OperatorString.normal_ordered(
        Fraction(1, 4),
        None,
        (Operator(I, True), Operator(J, True), Operator(k, False), Operator(m, False)),
    )
    assert string.coefficient == Fraction(1, 4)
    assert string.operators[:2] == (Operator(k, False), Operator(m, False))


def test_disconnected_terms_are_dropped() -> None:
    """``<Φ_i^a| f_N T1 T1 |Φ>`` keeps only the terms with both T1 linked to f."""
    fresh = Fresh()
    projection = singles_projection()
    bra = OperatorString(Fraction(1), None, projection.operators)
    # One cluster operator: f[vo] contracts with the bra alone and is
    # unlinked, f[ov] has no quasi-creator for it.
    terms = merged(
        term
        for one_body in hamiltonian_strings(fresh)[:4]
        for term in connected_expectation(
            [bra, one_body, singles_cluster(fresh)], 1, [2]
        )
    )
    assert sorted(describe(t) for t in terms) == [
        "-1 f[j,i] t1[j,a]",
        "1 f[a,b] t1[i,b]",
    ]
    f_ov = hamiltonian_strings(fresh)[1]
    assert "".join(i.space for i in f_ov.factor.indices) == "ov"  # type: ignore[union-attr]
    two = [bra, f_ov, singles_cluster(fresh), singles_cluster(fresh)]
    # The two equivalent contractions of ``T1 T1`` double the term; the
    # ``1/2!`` of the exponential is applied by the equation.
    connected = merged(connected_expectation(two, 1, [2, 3]))
    assert [describe(t) for t in connected] == ["-2 f[j,b] t1[i,b] t1[j,a]"]
    # Dropping the requirement admits the product of the energy with ``t1[i,a]``.
    unlinked = merged(connected_expectation(two, 1, []))
    assert sorted(describe(t) for t in unlinked) == [
        "-2 f[j,b] t1[i,b] t1[j,a]",
        "2 f[j,b] t1[i,a] t1[j,b]",
    ]


def test_canonical_form_merges_equal_terms_and_cancels_opposite_ones() -> None:
    k, m = Index("o", "k"), Index("o", "m")
    c, d = Index("v", "c"), Index("v", "d")
    term = Term(
        Fraction(1),
        (
            Factor("v", (k, m, c, d)),
            Factor("t2", (I, J, c, d)),
            Factor("t2", (k, m, A, B)),
        ),
    )
    swapped = Term(
        Fraction(1),
        (
            Factor("t2", (m, k, B, A)),
            Factor("v", (m, k, d, c)),
            Factor("t2", (J, I, d, c)),
        ),
    )
    assert canonical(term) == canonical(swapped)
    assert describe(canonical(term)) == "1 v[k,l,c,d] t2[i,j,c,d] t2[k,l,a,b]"
    assert [describe(t) for t in merged([term, swapped])] == [
        "2 v[k,l,c,d] t2[i,j,c,d] t2[k,l,a,b]"
    ]
    opposite = Term(Fraction(1), (Factor("v", (m, k, c, d)), *term.factors[1:]))
    assert merged([term, opposite]) == []


def test_orbits_reject_a_residual_that_is_not_antisymmetric() -> None:
    k = Index("o", "k")
    lone = Term(Fraction(1), (Factor("f", (k, I)), Factor("t2", (J, k, A, B))))
    with pytest.raises(ValueError, match="antisymmetric"):
        orbits([canonical(lone)], "ijab")


def test_program_contracts_pairwise_over_shared_indices() -> None:
    k, m = Index("o", "k"), Index("o", "m")
    c, d = Index("v", "c"), Index("v", "d")
    term = Term(
        Fraction(1),
        (
            Factor("v", (k, m, c, d)),
            Factor("t1", (I, c)),
            Factor("t1", (J, d)),
            Factor("t1", (k, A)),
            Factor("t1", (m, B)),
        ),
    )
    lines = program(term, "ijab")
    assert len(lines) == 4
    assert lines[-1].startswith('x = contract("')
    assert lines[-1].split("->")[1].startswith("ijab")
    for line in lines:
        inputs = line.split('"')[1].split("->")[0].split(",")
        assert set(inputs[0]) & set(inputs[1]), line
    assert program(Term(Fraction(1), (Factor("v", (A, B, I, J)),)), "ijab") == [
        'x = v["vvoo"].permute("abij->ijab")'
    ]
    assert program(Term(Fraction(1), (Factor("f", (I, A)),)), "ia") == ['x = f["ov"]']

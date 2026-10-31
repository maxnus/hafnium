"""The equation generator: Wick contractions, canonical terms, the emitted program."""

from fractions import Fraction
from pathlib import Path

import pytest

import hafnium.cc._ccsd_equations as generated
import hafnium.cc._ccsd_t_equations as generated_ccsd_t
from hafnium.cc.generator.ccsd import ccsd_equations
from hafnium.cc.generator.ccsd_t import ccsd_t_equations
from hafnium.cc.generator.emit import (
    ccsd_module_source,
    ccsd_t_module_source,
    function_source,
    orbits,
    permutations_of,
    program,
)
from hafnium.cc.generator.equations import (
    DummyIndices,
    Equation,
    Projection,
    hamiltonian_strings,
    singles_cluster,
)
from hafnium.cc.generator.wick import (
    Factor,
    Index,
    Kind,
    Operator,
    OperatorString,
    Term,
    expectation,
    full_contractions,
)
from hafnium.correlated import Space

O, V = Space.OCCUPIED, Space.VIRTUAL
I, J, K = (Index(O, label, True) for label in "ijk")
A, B = (Index(V, label, True) for label in "ab")


@pytest.fixture(scope="module")
def equations() -> dict[str, Equation]:
    return ccsd_equations()


def test_committed_equations_are_current() -> None:
    """The generated modules are regenerated to detect drift (PLAN §5.5)."""
    source = Path(generated.__file__).read_text(encoding="utf-8")
    assert source == ccsd_module_source()
    source = Path(generated_ccsd_t.__file__).read_text(encoding="utf-8")
    assert source == ccsd_t_module_source()


def test_energy_is_the_known_expression(equations: dict[str, Equation]) -> None:
    assert sorted(str(term) for term in equations["energy"].terms) == [
        "1 f[i,a] t1[i,a]",
        "1/2 v[i,j,a,b] t1[i,a] t1[j,b]",
        "1/4 v[i,j,a,b] t2[i,j,a,b]",
    ]


def test_singles_equation_has_the_fourteen_terms(
    equations: dict[str, Equation],
) -> None:
    terms = {str(term) for term in equations["singles"].terms}
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
    equations: dict[str, Equation],
) -> None:
    equation = equations["doubles"]
    assert equation.free == "ijab"
    assert len(equation.terms) == 63
    groups = orbits(equation)
    assert len(groups) == 31
    helpers = {str(group.term): group.antisymmetrizer for group in groups}
    assert helpers["1 v[a,b,i,j]"] == ""
    assert helpers["1/2 v[a,b,c,d] t2[i,j,c,d]"] == ""
    assert helpers["1/2 v[k,l,i,j] t2[k,l,a,b]"] == ""
    assert helpers["1/4 v[k,l,c,d] t2[i,j,c,d] t2[k,l,a,b]"] == ""
    assert helpers["1 f[k,i] t2[j,k,a,b]"] == "_minus_ij"
    assert helpers["-1 f[a,c] t2[i,j,b,c]"] == "_minus_ab"
    assert helpers["-1 v[k,a,i,c] t2[j,k,b,c]"] == "_minus_both"
    assert sum(len(group.cosets) for group in groups) == 63
    # The term symmetric under the simultaneous swap of both pairs alone
    # has a stabilizer that is no product, so its images do not factor.
    factored = {str(group.term): group.factored for group in groups}
    assert not factored["1 v[k,l,c,d] t2[i,k,a,c] t2[j,l,b,d]"]
    assert sum(not value for value in factored.values()) == 1
    assert factored["1 f[k,i] t2[j,k,a,b]"]
    assert all(
        {"", "_minus_ij", "_minus_ab", "_minus_both"} >= {g.antisymmetrizer}
        for g in groups
    )


def test_single_excitation_overlaps_with_itself() -> None:
    """``<Φ| a†_i a_a a†_a a_i |Φ> = 1``: one full contraction with a plus sign."""
    bra = (Operator(I, True), Operator(A, False))
    ket = (Operator(Index(V, "c"), True), Operator(Index(O, "k"), False))
    contractions = list(full_contractions([bra, ket]))
    assert contractions == [(1, [(0, 3), (1, 2)])]
    # The crossed pairing of the doubles bra with its ket carries the sign.
    bra = (Operator(I, True), Operator(J, True), Operator(B, False), Operator(A, False))
    c, d = Index(V, "c"), Index(V, "d")
    k, m = Index(O, "k"), Index(O, "m")
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
    k, m = Index(O, "k"), Index(O, "m")
    string = OperatorString.normal_ordered(
        Fraction(1, 4),
        None,
        (Operator(I, True), Operator(J, True), Operator(k, False), Operator(m, False)),
    )
    assert string.coefficient == Fraction(1, 4)
    assert string.operators[:2] == (Operator(k, False), Operator(m, False))


def test_disconnected_terms_are_dropped() -> None:
    """``<Φ_i^a| f_N T1 T1 |Φ>`` keeps only the terms with both T1 linked to f."""
    dummies = DummyIndices()
    projection = Projection.singles()
    bra = OperatorString(Fraction(1), None, projection.operators)
    # One cluster operator: f[vo] contracts with the bra alone and is
    # unlinked, f[ov] has no quasi-creator for it.
    terms = Term.merge(
        term
        for one_body in hamiltonian_strings(dummies)[:4]
        for term in expectation([bra, one_body, singles_cluster(dummies)], 1, [2])
    )
    assert sorted(str(t) for t in terms) == [
        "-1 f[j,i] t1[j,a]",
        "1 f[a,b] t1[i,b]",
    ]
    f_ov = hamiltonian_strings(dummies)[1]
    assert f_ov.factor is not None
    assert f_ov.factor.slots == (O, V)
    two = [bra, f_ov, singles_cluster(dummies), singles_cluster(dummies)]
    # The two equivalent contractions of ``T1 T1`` double the term; the
    # ``1/2!`` of the exponential is applied by the equation.
    connected = Term.merge(expectation(two, 1, [2, 3]))
    assert [str(t) for t in connected] == ["-2 f[j,b] t1[i,b] t1[j,a]"]
    # Dropping the requirement admits the product of the energy with ``t1[i,a]``,
    # which is the disconnected term alone.
    unlinked = Term.merge(expectation(two, 1, []))
    assert sorted(str(t) for t in unlinked) == [
        "-2 f[j,b] t1[i,b] t1[j,a]",
        "2 f[j,b] t1[i,a] t1[j,b]",
    ]
    disconnected = Term.merge(expectation(two, 1, [2, 3], connected=False))
    assert [str(t) for t in disconnected] == ["2 f[j,b] t1[i,a] t1[j,b]"]


def test_canonical_form_merges_equal_terms_and_cancels_opposite_ones() -> None:
    k, m = Index(O, "k"), Index(O, "m")
    c, d = Index(V, "c"), Index(V, "d")
    term = Term(
        Fraction(1),
        (
            Factor(Kind.INTEGRAL, (k, m, c, d)),
            Factor(Kind.DOUBLES, (I, J, c, d)),
            Factor(Kind.DOUBLES, (k, m, A, B)),
        ),
    )
    swapped = Term(
        Fraction(1),
        (
            Factor(Kind.DOUBLES, (m, k, B, A)),
            Factor(Kind.INTEGRAL, (m, k, d, c)),
            Factor(Kind.DOUBLES, (J, I, d, c)),
        ),
    )
    assert term.canonical() == swapped.canonical()
    assert str(term.canonical()) == "1 v[k,l,c,d] t2[i,j,c,d] t2[k,l,a,b]"
    assert [str(t) for t in Term.merge([term, swapped])] == [
        "2 v[k,l,c,d] t2[i,j,c,d] t2[k,l,a,b]"
    ]
    opposite = Term(
        Fraction(1), (Factor(Kind.INTEGRAL, (m, k, c, d)), *term.factors[1:])
    )
    assert Term.merge([term, opposite]) == []


def test_orbits_reject_a_residual_that_is_not_antisymmetric() -> None:
    k = Index(O, "k")
    lone = Term(
        Fraction(1), (Factor(Kind.FOCK, (k, I)), Factor(Kind.DOUBLES, (J, k, A, B)))
    )
    with pytest.raises(ValueError, match="antisymmetric"):
        orbits(Equation("doubles", "ijab", (lone.canonical(),)))


def test_program_contracts_pairwise_over_shared_indices() -> None:
    k, m = Index(O, "k"), Index(O, "m")
    c, d = Index(V, "c"), Index(V, "d")
    term = Term(
        Fraction(1),
        (
            Factor(Kind.INTEGRAL, (k, m, c, d)),
            Factor(Kind.SINGLES, (I, c)),
            Factor(Kind.SINGLES, (J, d)),
            Factor(Kind.SINGLES, (k, A)),
            Factor(Kind.SINGLES, (m, B)),
        ),
    )
    lines = program(term, "ijab")
    assert len(lines) == 4
    assert lines[-1].startswith('x = contract("')
    assert lines[-1].split("->")[1].startswith("ijab")
    for line in lines:
        inputs = line.split('"')[1].split("->")[0].split(",")
        assert set(inputs[0]) & set(inputs[1]), line
    assert program(
        Term(Fraction(1), (Factor(Kind.INTEGRAL, (A, B, I, J)),)), "ijab"
    ) == ['x = v[VIR, VIR, OCC, OCC].permute("abij->ijab")']
    assert program(Term(Fraction(1), (Factor(Kind.FOCK, (I, A)),)), "ia") == [
        "x = f[OCC, VIR]"
    ]


def test_triples_equations_have_the_textbook_terms() -> None:
    """``W`` and ``V`` of (T): two orbits of nine images each (ADR-0033)."""
    equations = ccsd_t_equations()
    connected = equations["connected_triples"]
    assert connected.free == "ijkabc"
    terms = {str(term) for term in connected.terms}
    assert len(terms) == 18
    assert {"1 v[b,c,j,d] t2[i,k,a,d]", "-1 v[l,c,j,k] t2[i,l,a,b]"} <= terms
    groups = orbits(connected)
    assert [len(group.cosets) for group in groups] == [9, 9]
    assert sum(parity for group in groups for _, parity in group.cosets) == 2
    disconnected = equations["disconnected_triples"]
    terms = {str(term) for term in disconnected.terms}
    assert len(terms) == 18
    assert {"1 v[b,c,j,k] t1[i,a]", "1 f[a,i] t2[j,k,b,c]"} <= terms
    assert [len(group.cosets) for group in orbits(disconnected)] == [9, 9]
    # Every stabilizer is a transposition of occupied times one of virtual
    # indices, so the images factor: three occupied images summed, then
    # the virtual antisymmetrizer applied to the sum.
    for group in (*groups, *orbits(disconnected)):
        assert group.factored
        assert len(group.occupied_cosets) == len(group.virtual_cosets) == 3
        assert group.occupied_cosets[0] == group.virtual_cosets[0] == ((), 1)
    source = function_source(connected)
    assert source.count("contract(") == 6
    assert source.count("_windowed(") == 12
    assert 'r = s - s.permute("ijkabc->ijkbac") + s.permute("ijkabc->ijkbca")' in source
    assert "f = h.fock" not in source
    assert "f, v = h.fock, h.integrals" in function_source(disconnected)


def test_unfactored_orbit_over_windows_is_written_image_by_image() -> None:
    """A stabilizer that is no product keeps the per-image form after the groups."""
    l, m = Index(O, "l"), Index(O, "m")
    c, d, e = Index(V, "c", True), Index(V, "d"), Index(V, "e")
    # Symmetric under the simultaneous swap of ``i, j`` with ``a, b`` alone.
    term = Term(
        Fraction(1),
        (
            Factor(Kind.INTEGRAL, (l, m, d, e)),
            Factor(Kind.DOUBLES, (I, l, A, d)),
            Factor(Kind.DOUBLES, (J, m, B, e)),
            Factor(Kind.SINGLES, (K, c)),
        ),
    )
    images = [
        Term(term.coefficient * parity, term.substituted(dict(g)).factors)
        for g, parity in permutations_of("ijkabc")
    ]
    equation = Equation("connected_triples", "ijkabc", tuple(Term.merge(images)))
    (orbit,) = orbits(equation)
    assert not orbit.factored
    assert len(orbit.cosets) == 18
    assert len(orbit.occupied_cosets) == len(orbit.virtual_cosets) == 6
    source = function_source(equation)
    assert source.count("contract(") == 3 * 18
    assert "s = " not in source
    assert source.count("r = r ") == 17
    # Every image coincides with one other, so the merged coefficient is two.
    assert "    r = 2.0 * x\n" in source or "    r = -2.0 * x\n" in source


def test_permutations_carry_their_parities() -> None:
    assert permutations_of("") == (((), 1),)
    assert permutations_of("ia") == (((), 1),)
    doubles = permutations_of("ijab")
    assert doubles[0] == ((), 1)
    assert dict(doubles) == {
        (): 1,
        (("i", "j"), ("j", "i")): -1,
        (("a", "b"), ("b", "a")): -1,
        (("i", "j"), ("j", "i"), ("a", "b"), ("b", "a")): 1,
    }
    triples = permutations_of("ijkabc")
    assert len(triples) == 36
    assert sum(parity for _, parity in triples) == 0
    assert (
        (("i", "j"), ("j", "k"), ("k", "i")),
        1,
    ) in triples


def test_windowed_program_restricts_the_operands() -> None:
    d = Index(V, "d")
    term = Term(
        Fraction(1),
        (
            Factor(Kind.INTEGRAL, (B, Index(V, "c"), d, K)),
            Factor(Kind.DOUBLES, (I, J, A, d)),
        ),
    )
    lines = program(term, "ijkabc", {"i": "j", "j": "i", "k": "k"})
    # The call is split over lines as the formatter would split it.
    assert lines[0] == "x = contract("
    assert lines[-1] == ")"
    statement = "".join(line.strip() for line in lines)
    assert "_windowed(v[VIR, VIR, VIR, OCC], (None, None, None, k))" in statement
    assert "_windowed(t2, (j, i, None, None))" in statement
    assert program(term, "ijkabc") == [
        'x = contract("bcdk,ijad->ijkabc", v[VIR, VIR, VIR, OCC], t2)'
    ]

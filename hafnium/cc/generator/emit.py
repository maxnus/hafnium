"""Emit the CCSD equations as a program of pairwise contractions over block tensors.

The doubles residual is antisymmetric in ``i, j`` and in ``a, b``, so its
terms fall into orbits under the permutations of those pairs: a term and
the images of its free indices, each image carrying the sign of the
permutation. One term per orbit is evaluated and the images are its
permuted copies. Within a term the factors are contracted two at a time
in the cheapest order, with the occupied and virtual extents weighted
as :data:`EXTENT` says, and every index is shared by at most two factors
or free, so a pairwise :func:`~hafnium.structured.contract` sums exactly
the indices the two operands share.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

from hafnium.cc.generator.ccsd import Equation, ccsd_equations
from hafnium.cc.generator.wick import LABELS, Kind, Term
from hafnium.correlated import Space

INTERMEDIATES = ("x", "y", "z", "w")
"""Names of the intermediates of a term's program, the result in the first."""

EXTENT = {Space.OCCUPIED: 10, Space.VIRTUAL: 40}
"""Extents of the two spaces for ranking the contraction orders."""

Permutation = tuple[tuple[str, str], ...]

PERMUTATIONS: dict[str, tuple[tuple[Permutation, int], ...]] = {
    "": (((), 1),),
    "ia": (((), 1),),
    "ijab": (
        ((), 1),
        ((("i", "j"), ("j", "i")), -1),
        ((("a", "b"), ("b", "a")), -1),
        ((("i", "j"), ("j", "i"), ("a", "b"), ("b", "a")), 1),
    ),
}
"""The permutations of the free indices of each equation, with their parities."""

ANTISYMMETRIZERS: dict[frozenset[Permutation], str] = {
    frozenset({()}): "",
    frozenset({(), (("i", "j"), ("j", "i"))}): "_minus_ij",
    frozenset({(), (("a", "b"), ("b", "a"))}): "_minus_ab",
    frozenset({p for p, _ in PERMUTATIONS["ijab"]}): "_minus_both",
}
"""The helper applying the permutations of an orbit's coset representatives."""

PRELUDE = '''"""The spin-orbital CCSD equations, generated code (ADR-0032).

Do not edit: ``python -m hafnium.cc.generator`` derives the terms by
Wick's theorem and writes this program of pairwise contractions, and a
test regenerates it to detect drift. The functions
take the normal-ordered Hamiltonian of a reference in its orbital basis
and the amplitudes ``t1[i, a]`` and ``t2[i, j, a, b]`` of CONVENTIONS.md
§3, and return the connected energy ``<Φ| H_N e^T |Φ>`` as a tensor
without axes and the residuals ``<Φ_i^a| H_N e^T |Φ>`` and
``<Φ_ij^ab| H_N e^T |Φ>`` as tensors of the structure of the amplitudes.
"""

from __future__ import annotations

from hafnium.cc.hamiltonian import NormalOrderedHamiltonian
from hafnium.structured import BlockTensor, contract
'''

HELPERS = '''

def _minus_ij(x: BlockTensor) -> BlockTensor:
    """``(1 − P(ij)) x``."""
    return x - x.permute("ijab->jiab")


def _minus_ab(x: BlockTensor) -> BlockTensor:
    """``(1 − P(ab)) x``."""
    return x - x.permute("ijab->ijba")


def _minus_both(x: BlockTensor) -> BlockTensor:
    """``(1 − P(ij)) (1 − P(ab)) x``."""
    return _minus_ab(_minus_ij(x))
'''

SIGNATURES = {
    "energy": (
        "energy",
        "``<Φ| H_N e^T |Φ>`` connected, as a tensor without axes.",
    ),
    "singles": (
        "singles_residual",
        "``<Φ_i^a| H_N e^T |Φ>`` connected, as the tensor ``[i, a]``.",
    ),
    "doubles": (
        "doubles_residual",
        "``<Φ_ij^ab| H_N e^T |Φ>`` connected, as the tensor ``[i, j, a, b]``.",
    ),
}


@dataclass(frozen=True, slots=True)
class Orbit:
    """A representative term and the coset representatives of its images."""

    term: Term
    cosets: tuple[tuple[Permutation, int], ...]

    @property
    def antisymmetrizer(self) -> str:
        """Name of the helper applying the coset representatives, empty for none."""
        return ANTISYMMETRIZERS[frozenset(p for p, _ in self.cosets)]


def orbits(equation: Equation) -> list[Orbit]:
    """Group the terms of ``equation`` by the permutations of its free indices.

    The residual ``R`` satisfies ``R = ε(g) g(R)`` for every permutation
    ``g`` of its antisymmetric index pairs, so the coefficient of the
    image of a term under ``g`` is fixed by the term's own, which is
    checked; the orbit is then ``c Σ_cosets ε(g) g(x)`` over the images
    with distinct canonical forms, ``x`` the representative evaluated.

    Raises
    ------
    ValueError
        If a term lacks an image, or an image has another coefficient
        than the antisymmetry of the residual requires.
    """
    by_key = {term.key: term for term in equation.terms}
    done: set[tuple[object, ...]] = set()
    result: list[Orbit] = []
    for term in sorted(equation.terms, key=lambda t: t.key):
        if term.key in done:
            continue
        cosets: list[tuple[Permutation, int]] = []
        for permutation, parity in PERMUTATIONS[equation.free]:
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
        result.append(Orbit(term, tuple(cosets)))
    return result


# ------------------------------------------------------------------ programs


@dataclass(frozen=True, slots=True)
class _Operand:
    letters: str
    expression: str


def _cost(a: str, b: str) -> int:
    cost = 1
    for letter in set(a) | set(b):
        cost *= EXTENT[
            Space.OCCUPIED if letter in LABELS[Space.OCCUPIED] else Space.VIRTUAL
        ]
    return cost


def _contracted(a: str, b: str, output: str | None) -> str:
    """Return the letters left after summing those ``a`` and ``b`` share."""
    if output is not None:
        return output
    shared = set(a) & set(b)
    return "".join(c for c in a if c not in shared) + "".join(
        c for c in b if c not in shared
    )


def _orders(
    operands: tuple[str, ...], free: str
) -> Iterator[tuple[int, tuple[tuple[int, int, str], ...]]]:
    """Every sequence of pairwise contractions, as ``(cost, steps)``.

    A step ``(p, q, output)`` contracts operands ``p`` and ``q`` of the
    current list and appends the result with the letters ``output``.
    """
    if len(operands) == 1:
        yield 0, ()
        return
    for p, q in combinations(range(len(operands)), 2):
        a, b = operands[p], operands[q]
        output = _contracted(a, b, free if len(operands) == 2 else None)
        rest = (*(x for k, x in enumerate(operands) if k not in (p, q)), output)
        for cost, steps in _orders(rest, free):
            yield _cost(a, b) + cost, ((p, q, output), *steps)


def program(term: Term, free: str) -> list[str]:
    """Statements evaluating ``term`` without its coefficient into ``x``."""
    operands = [_Operand(f.letters, f.expression) for f in term.factors]
    if len(operands) == 1:
        (operand,) = operands
        expression = operand.expression
        if operand.letters != free:
            expression += f'.permute("{operand.letters}->{free}")'
        return [f"x = {expression}"]
    _, steps = min(_orders(tuple(o.letters for o in operands), free))
    lines: list[str] = []
    names = iter(INTERMEDIATES)
    for p, q, output in steps:
        a, b = operands[p], operands[q]
        name = (
            a.expression
            if a.expression in INTERMEDIATES
            else b.expression
            if b.expression in INTERMEDIATES
            else next(names)
        )
        lines.append(
            f'{name} = contract("{a.letters},{b.letters}->{output}", '
            f"{a.expression}, {b.expression})"
        )
        operands = [o for k, o in enumerate(operands) if k not in (p, q)]
        operands.append(_Operand(output, name))
    (final,) = operands
    if final.expression != "x":
        lines.append(f"x = {final.expression}")
    return lines


def _accumulation(orbit: Orbit, first: bool) -> str:
    coefficient = orbit.term.coefficient
    value = f"{orbit.antisymmetrizer}(x)" if orbit.antisymmetrizer else "x"
    magnitude = abs(coefficient)
    if magnitude != 1:
        value = f"{float(magnitude)!r} * {value}"
    if first:
        return f"r = {'-' if coefficient < 0 else ''}{value}"
    return f"r = r {'-' if coefficient < 0 else '+'} {value}"


def function_source(equation: Equation) -> str:
    """Write the Python function evaluating the terms of ``equation``."""
    function, docstring = SIGNATURES[equation.name]
    lines = [
        f"def {function}(",
        "    h: NormalOrderedHamiltonian, t1: BlockTensor, t2: BlockTensor",
        ") -> BlockTensor:",
        f'    """{docstring}"""',
        "    f, v = h.fock, h.integrals",
    ]
    for position, orbit in enumerate(orbits(equation)):
        lines.extend(f"    {line}" for line in program(orbit.term, equation.free))
        lines.append(f"    {_accumulation(orbit, position == 0)}")
    lines.append("    return r")
    return "\n".join(lines) + "\n"


def module_source() -> str:
    """Write the complete source of :mod:`hafnium.cc._ccsd_equations`."""
    equations = ccsd_equations()
    fock: set[str] = set()
    integrals: set[str] = set()
    for equation in equations.values():
        for term in equation.terms:
            for factor in term.factors:
                if factor.kind is Kind.FOCK:
                    fock.add(factor.slots)
                elif factor.kind is Kind.INTEGRAL:
                    integrals.add(factor.slots)
    constants = (
        "FOCK_BLOCKS = (" + ", ".join(f'"{s}"' for s in sorted(fock)) + ")\n"
        '"""The blocks of the Fock operator the equations read."""\n'
        "INTEGRAL_BLOCKS = (\n"
        + "".join(f'    "{s}",\n' for s in sorted(integrals))
        + ")\n"
        '"""The blocks of the antisymmetrized integrals the equations read."""\n'
    )
    functions = "".join(
        "\n\n" + function_source(equations[name])
        for name in ("energy", "singles", "doubles")
    )
    return PRELUDE + "\n" + constants + HELPERS + functions

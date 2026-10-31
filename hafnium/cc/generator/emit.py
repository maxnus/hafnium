"""Emit the equations as programs of pairwise contractions over block tensors.

A residual is antisymmetric in its occupied free indices and in its
virtual ones, so its terms fall into orbits under the permutations of
those indices: a term and the images of its free indices, each image
carrying the sign of the permutation. One term per orbit is evaluated
and the images are its permuted copies. Within a term the factors are
contracted two at a time in the cheapest order, with the occupied and
virtual extents weighted as :data:`EXTENT` says, and every index is
shared by at most two factors or free, so a pairwise
:func:`~hafnium.structured.contract` sums exactly the indices the two
operands share.

The triples of (T) are evaluated over windows of the occupied axis
(ADR-0033): the function takes one window per occupied free index, the
operands are restricted to the windows of the letters they carry, and
the image of a term under a permutation of the occupied indices is the
term evaluated over the permuted windows and then permuted.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations

from hafnium.cc.generator.ccsd import Equation, ccsd_equations, triples_equations
from hafnium.cc.generator.wick import LABELS, SPACE_NAMES, Kind, Term
from hafnium.correlated import Space

INTERMEDIATES = ("x", "y", "z", "w")
"""Names of the intermediates of a term's program, the result in the first."""

EXTENT = {Space.OCCUPIED: 10, Space.VIRTUAL: 40}
"""Extents of the two spaces for ranking the contraction orders."""

Permutation = tuple[tuple[str, str], ...]
"""The free indices a permutation moves, as pairs ``(from, to)``."""


def permutations_of(free: str) -> tuple[tuple[Permutation, int], ...]:
    """List the permutations of the free indices ``free`` with their parities.

    Every permutation of the occupied letters with every permutation of
    the virtual letters, the identity first; the parity is the product
    of the parities.
    """
    occupied = [c for c in free if c in LABELS[Space.OCCUPIED]]
    virtual = [c for c in free if c in LABELS[Space.VIRTUAL]]
    result: list[tuple[Permutation, int]] = []
    for virtual_image in permutations(virtual):
        for occupied_image in permutations(occupied):
            moved = tuple(
                (source, target)
                for source, target in (
                    *zip(occupied, occupied_image, strict=True),
                    *zip(virtual, virtual_image, strict=True),
                )
                if source != target
            )
            result.append((moved, _parity(occupied_image) * _parity(virtual_image)))
    return tuple(result)


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
"""The helper applying the permutations of an orbit's coset representatives."""

PRELUDE = '''"""{title}, generated code ({adr}).

Do not edit: ``python -m hafnium.cc.generator`` derives the terms by
Wick's theorem and writes this program of pairwise contractions, and a
test regenerates it to detect drift. {description}
"""

from __future__ import annotations

from hafnium.cc.hamiltonian import NormalOrderedHamiltonian
from hafnium.correlated import Space
from hafnium.structured import BlockTensor, {imports}

OCC, VIR = Space.OCCUPIED, Space.VIRTUAL
'''

CCSD_DESCRIPTION = """The functions
take the normal-ordered Hamiltonian of a reference in its orbital basis
and the amplitudes ``t1[i, a]`` and ``t2[i, j, a, b]`` of CONVENTIONS.md
§3, and return the connected energy ``<Φ| H_N e^T |Φ>`` as a tensor
without axes and the residuals ``<Φ_i^a| H_N e^T |Φ>`` and
``<Φ_ij^ab| H_N e^T |Φ>`` as tensors of the structure of the amplitudes."""

TRIPLES_DESCRIPTION = """The functions
take the normal-ordered Hamiltonian of a reference in its orbital basis,
the amplitudes ``t1[i, a]`` and ``t2[i, j, a, b]`` of CONVENTIONS.md §3
and three windows of the occupied axis, and return the connected
triples ``<Φ_ijk^abc| H_N T2 |Φ>_C`` and the disconnected ones
``<Φ_ijk^abc| H_N (T1 + T2) |Φ>`` over those windows as tensors
``[i, j, k, a, b, c]``, so that the triples are evaluated a batch of
occupied indices at a time."""

CCSD_HELPERS = '''

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

TRIPLES_HELPERS = '''

def _windowed(x: BlockTensor, windows: tuple[TensorAxis | None, ...]) -> BlockTensor:
    """Restrict the axes of ``x`` to the windows given, ``None`` leaving one whole."""
    for position, window in enumerate(windows):
        if window is not None:
            x = x.window(position, window)
    return x
'''


ATTRIBUTES = {Kind.FOCK: "fock", Kind.INTEGRAL: "integrals"}
"""The attribute of the Hamiltonian each of its kinds is read from."""


@dataclass(frozen=True, slots=True)
class Signature:
    """The function an equation is written as."""

    function: str
    docstring: str
    windowed: str = ""
    """The free indices the function takes a window each for."""


SIGNATURES = {
    "energy": Signature(
        "energy", "``<Φ| H_N e^T |Φ>`` connected, as a tensor without axes."
    ),
    "singles": Signature(
        "singles_residual",
        "``<Φ_i^a| H_N e^T |Φ>`` connected, as the tensor ``[i, a]``.",
    ),
    "doubles": Signature(
        "doubles_residual",
        "``<Φ_ij^ab| H_N e^T |Φ>`` connected, as the tensor ``[i, j, a, b]``.",
    ),
    "connected_triples": Signature(
        "connected_triples",
        "``<Φ_ijk^abc| H_N T2 |Φ>`` connected, as ``[i, j, k, a, b, c]``.\n\n"
        "The occupied indices run over the windows ``i``, ``j`` and ``k``.",
        "ijk",
    ),
    "disconnected_triples": Signature(
        "disconnected_triples",
        "``<Φ_ijk^abc| H_N (T1 + T2) |Φ>`` disconnected, as ``[i, j, k, a, b, c]``."
        "\n\nThe occupied indices run over the windows ``i``, ``j`` and ``k``.",
        "ijk",
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
        for permutation, parity in permutations_of(equation.free):
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


def _operands(term: Term, windows: Mapping[str, str] | None) -> list[_Operand]:
    """Return the factors of ``term`` as operands, restricted to ``windows``."""
    operands: list[_Operand] = []
    for factor in term.factors:
        expression = factor.expression
        if windows and any(letter in windows for letter in factor.letters):
            names = ", ".join(windows.get(letter, "None") for letter in factor.letters)
            expression = f"_windowed({expression}, ({names}))"
        operands.append(_Operand(factor.letters, expression))
    return operands


def program(
    term: Term, free: str, windows: Mapping[str, str] | None = None
) -> list[str]:
    """Statements evaluating ``term`` without its coefficient into ``x``.

    ``windows`` names the window variable every windowed free index of the
    term is restricted to.
    """
    operands = _operands(term, windows)
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
        lines.extend(
            _call(
                f"{name} = contract",
                [f'"{a.letters},{b.letters}->{output}"', a.expression, b.expression],
            )
        )
        operands = [o for k, o in enumerate(operands) if k not in (p, q)]
        operands.append(_Operand(output, name))
    (final,) = operands
    if final.expression != "x":
        lines.append(f"x = {final.expression}")
    return lines


LINE_LENGTH = 88
"""The line length of the formatter the generated modules are checked with."""

FUNCTION_INDENT = 4
"""Indentation of the statements of a function, which the lines leave out."""


def _call(head: str, arguments: list[str]) -> list[str]:
    """Write ``head(arguments)`` as the formatter would, over lines without indent.

    One line when it fits, else the arguments on one indented line, else
    one argument per line with a trailing comma.
    """
    one = f"{head}({', '.join(arguments)})"
    if FUNCTION_INDENT + len(one) <= LINE_LENGTH:
        return [one]
    hugged = ", ".join(arguments)
    if 2 * FUNCTION_INDENT + len(hugged) <= LINE_LENGTH:
        return [f"{head}(", f"    {hugged}", ")"]
    return [f"{head}(", *(f"    {argument}," for argument in arguments), ")"]


def _accumulation(coefficient: Fraction, value: str, first: bool) -> str:
    magnitude = abs(coefficient)
    if magnitude != 1:
        value = f"{float(magnitude)!r} * {value}"
    if first:
        return f"r = {'-' if coefficient < 0 else ''}{value}"
    return f"r = r {'-' if coefficient < 0 else '+'} {value}"


def _orbit_lines(orbit: Orbit, equation: Equation, first: bool) -> list[str]:
    """Write the statements adding an orbit to ``r``."""
    signature = SIGNATURES[equation.name]
    if not signature.windowed:
        lines = program(orbit.term, equation.free)
        value = f"{orbit.antisymmetrizer}(x)" if orbit.antisymmetrizer else "x"
        return [*lines, _accumulation(orbit.term.coefficient, value, first)]
    lines: list[str] = []
    for permutation, parity in orbit.cosets:
        # The image under ``g`` carries the label ``g(l)`` where the
        # representative carries ``l``, so it is the representative over
        # the windows of ``g(l)`` with the axis of ``m`` taken from the
        # slot of ``g⁻¹(m)``.
        mapping = dict(permutation)
        inverse = {target: source for source, target in permutation}
        windows = {letter: mapping.get(letter, letter) for letter in signature.windowed}
        lines.extend(program(orbit.term, equation.free, windows))
        image = "".join(inverse.get(letter, letter) for letter in equation.free)
        value = (
            "x" if image == equation.free else f'x.permute("{equation.free}->{image}")'
        )
        lines.append(_accumulation(orbit.term.coefficient * parity, value, first))
        first = False
    return lines


def function_source(equation: Equation) -> str:
    """Write the Python function evaluating the terms of ``equation``."""
    signature = SIGNATURES[equation.name]
    parameters = ["h: NormalOrderedHamiltonian", "t1: BlockTensor", "t2: BlockTensor"]
    parameters += [f"{letter}: TensorAxis" for letter in signature.windowed]
    lines = [f"def {signature.function}("]
    if signature.windowed:
        lines.extend(f"    {parameter}," for parameter in parameters)
    else:
        lines.append(f"    {', '.join(parameters)}")
    lines.append(") -> BlockTensor:")
    if "\n" in signature.docstring:
        lines.append('    """' + signature.docstring.replace("\n\n", "\n\n    "))
        lines.append('    """')
    else:
        lines.append(f'    """{signature.docstring}"""')
    kinds = {factor.kind for term in equation.terms for factor in term.factors}
    names = [kind for kind in (Kind.FOCK, Kind.INTEGRAL) if kind in kinds]
    lines.append(
        "    "
        + ", ".join(kind.value for kind in names)
        + " = "
        + ", ".join(f"h.{ATTRIBUTES[kind]}" for kind in names)
    )
    for position, orbit in enumerate(orbits(equation)):
        lines.extend(
            f"    {line}" for line in _orbit_lines(orbit, equation, position == 0)
        )
    lines.append("    return r")
    return "\n".join(lines) + "\n"


def _constants(equations: Mapping[str, Equation]) -> str:
    """Write the blocks of the Hamiltonian the equations read as constants."""
    fock: set[tuple[Space, ...]] = set()
    integrals: set[tuple[Space, ...]] = set()
    for equation in equations.values():
        for term in equation.terms:
            for factor in term.factors:
                if factor.kind is Kind.FOCK:
                    fock.add(factor.slots)
                elif factor.kind is Kind.INTEGRAL:
                    integrals.add(factor.slots)

    def listed(name: str, blocks: set[tuple[Space, ...]]) -> str:
        entries = [
            "(" + ", ".join(SPACE_NAMES[s] for s in block) + ")"
            for block in sorted(blocks, key=lambda b: [s.value for s in b])
        ]
        # A tuple of one entry stays on one line, as the formatter keeps it.
        if len(entries) == 1:
            return f"{name} = ({entries[0]},)\n"
        return f"{name} = (\n" + "".join(f"    {e},\n" for e in entries) + ")\n"

    return (
        listed("FOCK_BLOCKS", fock)
        + '"""The blocks of the Fock operator the equations read."""\n'
        + listed("INTEGRAL_BLOCKS", integrals)
        + '"""The blocks of the antisymmetrized integrals the equations read."""\n'
    )


def ccsd_module_source() -> str:
    """Write the complete source of :mod:`hafnium.cc._ccsd_equations`."""
    equations = ccsd_equations()
    prelude = PRELUDE.format(
        title="The spin-orbital CCSD equations",
        adr="ADR-0032",
        description=CCSD_DESCRIPTION,
        imports="contract",
    )
    functions = "".join(
        "\n\n" + function_source(equations[name])
        for name in ("energy", "singles", "doubles")
    )
    return prelude + "\n" + _constants(equations) + CCSD_HELPERS + functions


def triples_module_source() -> str:
    """Write the complete source of :mod:`hafnium.cc._triples_equations`."""
    equations = triples_equations()
    prelude = PRELUDE.format(
        title="The triples of the (T) correction",
        adr="ADR-0033",
        description=TRIPLES_DESCRIPTION,
        imports="TensorAxis, contract",
    )
    functions = "".join(
        "\n\n" + function_source(equations[name])
        for name in ("connected_triples", "disconnected_triples")
    )
    return prelude + "\n" + _constants(equations) + TRIPLES_HELPERS + functions

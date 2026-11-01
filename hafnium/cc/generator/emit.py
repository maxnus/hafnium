"""Write an equation as a Python function of pairwise contractions over block tensors.

Within a term the factors are contracted two at a time in the cheapest
order, with the occupied and virtual extents weighted as :data:`EXTENT`
says, and every index is shared by at most two factors or free, so a
pairwise :func:`~hafnium.structured.contract` sums exactly the indices
the two operands share. One term per orbit of the equation
(:mod:`hafnium.cc.generator.orbits`) is evaluated and its images are
added as permuted copies, through the helper the orbit names.

The triples of (T) are evaluated over windows of the occupied axis
(ADR-0033): the function takes one window per occupied free index, the
operands are restricted to the windows of the letters they carry, and
the image of a term under a permutation of the occupied indices is the
term evaluated over the permuted windows and then permuted. When the
stabilizer of a term is the product of its occupied and its virtual
parts, the sum over its images factors: the occupied images are summed
first and the virtual permutations are applied to the sum, once for
every orbit that shares them.

The functions are laid out as the formatter would lay them out, so that
the generated modules pass its check unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from hafnium.cc.generator.equations import Equation
from hafnium.cc.generator.orbits import Cosets, Orbit, Permutation, orbits
from hafnium.cc.generator.wick import LABELS, SPACE_NAMES, Kind, Term
from hafnium.orbitals import Space

INTERMEDIATES = ("x", "y", "z", "w")
"""Names of the intermediates of a term's program, the result in the first."""

EXTENT = {Space.OCCUPIED: 10, Space.VIRTUAL: 40}
"""Extents of the two spaces for ranking the contraction orders."""

ATTRIBUTES = {Kind.FOCK: "fock", Kind.INTEGRAL: "integrals"}
"""The attribute of the Hamiltonian each of its kinds is read from."""

TYPES = {"h": "NormalOrderedHamiltonian"}
"""The annotation of an operand of a function, a block tensor when not listed."""


@dataclass(frozen=True, slots=True)
class Signature:
    """The function an equation is written as."""

    function: str
    docstring: str
    operands: tuple[str, ...] = ("h", "t1", "t2")
    """The names of the parameters, the Hamiltonian ``h`` and the tensors."""
    windowed: str = ""
    """The free indices the function takes a window each for."""


# ------------------------------------------------------------------ programs


@dataclass(frozen=True, slots=True)
class _Operand:
    letters: str
    expression: str


def _space_of(letter: str) -> Space:
    return Space.OCCUPIED if letter in LABELS[Space.OCCUPIED] else Space.VIRTUAL


def _cost(a: str, b: str) -> int:
    cost = 1
    for letter in set(a) | set(b):
        cost *= EXTENT[_space_of(letter)]
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

VARIABLES = (*INTERMEDIATES, "s", "r")
"""The variables of a generated function, in the order they are declared.

Pyright infers the type of an undeclared variable from its assignments
and gives up beyond 64 of them, which the longer equations exceed, so
every function declares the variables it assigns first.
"""

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


def _accumulation(
    coefficient: Fraction, value: str, first: bool, target: str = "r"
) -> str:
    magnitude = abs(coefficient)
    if magnitude != 1:
        value = f"{float(magnitude)!r} * {value}"
    if first:
        return f"{target} = {'-' if coefficient < 0 else ''}{value}"
    return f"{target} = {target} {'-' if coefficient < 0 else '+'} {value}"


def _permuted(name: str, permutation: Permutation, free: str) -> str:
    """Write the image of the tensor ``name`` under ``permutation``.

    The image under ``g`` carries the label ``g(l)`` where the original
    carries ``l``, so the axis of ``m`` is taken from the slot of
    ``g⁻¹(m)``.
    """
    inverse = {target: source for source, target in permutation}
    image = "".join(inverse.get(letter, letter) for letter in free)
    return name if image == free else f'{name}.permute("{free}->{image}")'


def _helper_lines(orbit: Orbit, equation: Equation, first: bool) -> list[str]:
    """Write the statements adding an orbit to ``r`` through a permutation helper."""
    lines = program(orbit.term, equation.free)
    value = f"{orbit.antisymmetrizer}(x)" if orbit.antisymmetrizer else "x"
    return [*lines, _accumulation(orbit.term.coefficient, value, first)]


def _image_lines(
    orbit: Orbit,
    equation: Equation,
    windowed: str,
    cosets: Cosets,
    target: str,
    first: bool,
) -> list[str]:
    """Write the images of an orbit's representative under ``cosets`` into ``target``.

    Each image is the representative evaluated over the windows of the
    free indices ``windowed``, permuted, and permuted back, added with
    its parity.
    """
    lines: list[str] = []
    for permutation, parity in cosets:
        mapping = dict(permutation)
        windows = {letter: mapping.get(letter, letter) for letter in windowed}
        lines.extend(program(orbit.term, equation.free, windows))
        value = _permuted("x", permutation, equation.free)
        lines.append(
            _accumulation(orbit.term.coefficient * parity, value, first, target)
        )
        first = False
    return lines


def _windowed_lines(
    orbits_: list[Orbit], equation: Equation, windowed: str
) -> list[str]:
    """Write the statements of an equation evaluated over the windows ``windowed``.

    The orbits whose stabilizer factors are summed over their occupied
    images into ``s``, grouped by their virtual cosets, and the virtual
    antisymmetrizer of a group is applied to ``s`` once, as permuted
    copies; the other orbits are evaluated image by image into ``r``.
    """
    groups: dict[tuple[Permutation, ...], list[Orbit]] = {}
    plain: list[Orbit] = []
    for orbit in orbits_:
        if orbit.factored:
            key = tuple(permutation for permutation, _ in orbit.virtual_cosets)
            groups.setdefault(key, []).append(orbit)
        else:
            plain.append(orbit)
    lines: list[str] = []
    first = True
    for members in groups.values():
        for position, orbit in enumerate(members):
            lines.extend(
                _image_lines(
                    orbit, equation, windowed, orbit.occupied_cosets, "s", position == 0
                )
            )
        # The identity comes first among the cosets, so the sum starts
        # with ``s`` itself.
        expression = " ".join(
            f"{'-' if parity < 0 else '+'} {_permuted('s', permutation, equation.free)}"
            for permutation, parity in members[0].virtual_cosets
        ).removeprefix("+ ")
        lines.extend(
            [f"r = {expression}"] if first else [f"x = {expression}", "r = r + x"]
        )
        first = False
    for orbit in plain:
        lines.extend(_image_lines(orbit, equation, windowed, orbit.cosets, "r", first))
        first = False
    return lines


def _definition(name: str, parameters: list[str], returns: str) -> list[str]:
    """Write ``def name(parameters) -> returns:`` as the formatter would.

    One line when it fits, else the parameters on one indented line, else
    one parameter per line with a trailing comma.
    """
    one = f"def {name}({', '.join(parameters)}) -> {returns}:"
    if len(one) <= LINE_LENGTH:
        return [one]
    hugged = ", ".join(parameters)
    if FUNCTION_INDENT + len(hugged) <= LINE_LENGTH:
        return [f"def {name}(", f"    {hugged}", f") -> {returns}:"]
    return [f"def {name}(", *(f"    {p}," for p in parameters), f") -> {returns}:"]


def _docstring_lines(docstring: str) -> list[str]:
    """Write ``docstring`` indented as the body of a function."""
    if "\n" not in docstring:
        return [f'    """{docstring}"""']
    first, *rest = docstring.split("\n")
    return [
        f'    """{first}',
        *(f"    {line}" if line else "" for line in rest),
        '    """',
    ]


def _hamiltonian_lines(terms: Iterable[Term]) -> list[str]:
    """Write the line reading the parts of the Hamiltonian the ``terms`` use."""
    kinds = {factor.kind for term in terms for factor in term.factors}
    names = [kind for kind in (Kind.FOCK, Kind.INTEGRAL) if kind in kinds]
    if not names:
        return []
    return [
        "    "
        + ", ".join(kind.value for kind in names)
        + " = "
        + ", ".join(f"h.{ATTRIBUTES[kind]}" for kind in names)
    ]


def _declaration_lines(body: Iterable[str]) -> list[str]:
    """Declare the variables the statements of ``body`` assign (:data:`VARIABLES`)."""
    assigned = {line.partition(" = ")[0] for line in body}
    return [f"{name}: BlockTensor" for name in VARIABLES if name in assigned]


def _parameters(signature: Signature) -> list[str]:
    """Write the annotated parameters of a function: its operands, then its windows."""
    parameters = [
        f"{name}: {TYPES.get(name, 'BlockTensor')}" for name in signature.operands
    ]
    return parameters + [f"{letter}: TensorAxis" for letter in signature.windowed]


def function_source(equation: Equation, signature: Signature) -> str:
    """Write the Python function ``signature`` evaluating the terms of ``equation``."""
    groups = orbits(equation)
    body = (
        _windowed_lines(groups, equation, signature.windowed)
        if signature.windowed
        else [
            line
            for position, orbit in enumerate(groups)
            for line in _helper_lines(orbit, equation, position == 0)
        ]
    )
    lines = _definition(signature.function, _parameters(signature), "BlockTensor")
    lines.extend(_docstring_lines(signature.docstring))
    lines.extend(_hamiltonian_lines(equation.terms))
    lines.extend(f"    {line}" for line in (*_declaration_lines(body), *body))
    lines.append("    return r")
    return "\n".join(lines) + "\n"


def density_source(blocks: Sequence[Equation], signature: Signature) -> str:
    """Write the function ``signature`` evaluating ``blocks`` into a mapping by spaces.

    The function returns ``dict[tuple[Space, Space], BlockTensor]`` with
    the tensor of each equation under the spaces of its free indices.
    """
    lines = _definition(
        signature.function,
        _parameters(signature),
        "dict[tuple[Space, Space], BlockTensor]",
    )
    lines.extend(_docstring_lines(signature.docstring))
    lines.append("    blocks: dict[tuple[Space, Space], BlockTensor] = {}")
    bodies = [
        [
            line
            for position, orbit in enumerate(orbits(equation))
            for line in _helper_lines(orbit, equation, position == 0)
        ]
        for equation in blocks
    ]
    declared = _declaration_lines(line for body in bodies for line in body)
    lines.extend(f"    {line}" for line in declared)
    for equation, body in zip(blocks, bodies, strict=True):
        lines.extend(f"    {line}" for line in body)
        spaces = ", ".join(SPACE_NAMES[_space_of(letter)] for letter in equation.free)
        lines.append(f"    blocks[{spaces}] = r")
    lines.append("    return blocks")
    return "\n".join(lines) + "\n"

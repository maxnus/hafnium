"""The generated modules of the methods: their preludes, helpers and functions.

:func:`ccsd_module_source`, :func:`ccsd_pt_module_source` and
:func:`ccsd_lambda_module_source` write the complete source of
:mod:`hafnium.cc._ccsd_equations`, :mod:`hafnium.cc._ccsd_pt_equations`
and :mod:`hafnium.cc._ccsd_lambda_equations`: a prelude naming the
method, the blocks of the Hamiltonian its equations read as constants,
the helpers the programs call, and the functions of the equations with
the signatures :data:`SIGNATURES` gives them, one per residual and one
for the blocks of the density, written by :mod:`hafnium.cc.generator.emit`.
"""

from __future__ import annotations

from collections.abc import Mapping

from hafnium.cc.generator.emit import Signature, density_source, function_source
from hafnium.cc.generator.equations import (
    Equation,
    ccsd_equations,
    ccsd_lambda_equations,
    ccsd_pt_equations,
)
from hafnium.cc.generator.wick import SPACE_NAMES, Kind
from hafnium.orbitals import Space

PRELUDE = '''"""{title}, generated code ({adr}).

Do not edit: ``python -m hafnium.cc.generator`` derives the terms by
Wick's theorem and writes this program of pairwise contractions, and a
test regenerates it to detect drift. {description}
"""

from __future__ import annotations

from hafnium.cc.hamiltonian import NormalOrderedHamiltonian
from hafnium.orbitals import Space
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

LAMBDA_DESCRIPTION = """The functions
take the normal-ordered Hamiltonian of a reference in its orbital basis,
the amplitudes ``t1[i, a]`` and ``t2[i, j, a, b]`` of CONVENTIONS.md §3
and the de-excitation amplitudes ``l1[i, a]`` and ``l2[i, j, a, b]``,
and return the residuals ``<Φ| (1 + Λ) H_N e^T |Φ_i^a>`` and
``<Φ| (1 + Λ) H_N e^T |Φ_ij^ab>`` of the Λ equations, linked, as tensors
of the structure of the de-excitation amplitudes, and the blocks of the
one-particle density ``<Φ| (1 + Λ) {p† q} e^T |Φ>`` by the spaces of
``p`` and ``q``."""

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

LAMBDA_OPERANDS = ("h", "t1", "t2", "l1", "l2")
"""The operands of a function of the Λ equations."""


SIGNATURES: dict[str, Signature] = {
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
        windowed="ijk",
    ),
    "disconnected_triples": Signature(
        "disconnected_triples",
        "``<Φ_ijk^abc| H_N (T1 + T2) |Φ>`` disconnected, as ``[i, j, k, a, b, c]``."
        "\n\nThe occupied indices run over the windows ``i``, ``j`` and ``k``.",
        windowed="ijk",
    ),
    "lambda_singles": Signature(
        "lambda_singles_residual",
        "``<Φ| (1 + Λ) H_N e^T |Φ_i^a>`` linked, as the tensor ``[i, a]``.",
        LAMBDA_OPERANDS,
    ),
    "lambda_doubles": Signature(
        "lambda_doubles_residual",
        "``<Φ| (1 + Λ) H_N e^T |Φ_ij^ab>`` linked, as the tensor ``[i, j, a, b]``.",
        LAMBDA_OPERANDS,
    ),
    "density": Signature(
        "one_particle_density",
        "``<Φ| (1 + Λ) {p† q} e^T |Φ>`` connected, by the spaces of ``p`` and "
        "``q``.\n\nEach block is the tensor ``[p, q]`` with ``q`` conjugated, "
        "pairing with\n``f[p, q]`` in a trace, without the occupations of the "
        "reference.",
        ("t1", "t2", "l1", "l2"),
    ),
}
"""The functions of the generated modules by the name of their equation.

``density`` is the one function of the four density blocks.
"""


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
        "\n\n" + function_source(equations[name], SIGNATURES[name])
        for name in ("energy", "singles", "doubles")
    )
    return prelude + "\n" + _constants(equations) + CCSD_HELPERS + functions


def ccsd_pt_module_source() -> str:
    """Write the complete source of :mod:`hafnium.cc._ccsd_pt_equations`."""
    equations = ccsd_pt_equations()
    prelude = PRELUDE.format(
        title="The triples of the (T) correction",
        adr="ADR-0033",
        description=TRIPLES_DESCRIPTION,
        imports="TensorAxis, contract",
    )
    functions = "".join(
        "\n\n" + function_source(equations[name], SIGNATURES[name])
        for name in ("connected_triples", "disconnected_triples")
    )
    return prelude + "\n" + _constants(equations) + TRIPLES_HELPERS + functions


def ccsd_lambda_module_source() -> str:
    """Write the complete source of :mod:`hafnium.cc._ccsd_lambda_equations`."""
    equations = ccsd_lambda_equations()
    prelude = PRELUDE.format(
        title="The Λ equations of CCSD and its one-particle density",
        adr="ADR-0034",
        description=LAMBDA_DESCRIPTION,
        imports="contract",
    )
    residuals = "".join(
        "\n\n" + function_source(equations[name], SIGNATURES[name])
        for name in ("lambda_singles", "lambda_doubles")
    )
    density = "\n\n" + density_source(
        [equation for equation in equations.values() if not equation.antisymmetric],
        SIGNATURES["density"],
    )
    return prelude + "\n" + _constants(equations) + CCSD_HELPERS + residuals + density

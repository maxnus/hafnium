"""The triples of the (T) correction, generated code (ADR-0033).

Do not edit: ``python -m hafnium.cc.generator`` derives the terms by
Wick's theorem and writes this program of pairwise contractions, and a
test regenerates it to detect drift. The functions
take the normal-ordered Hamiltonian of a reference in its orbital basis,
the amplitudes ``t1[i, a]`` and ``t2[i, j, a, b]`` of CONVENTIONS.md §3
and three windows of the occupied axis, and return the connected
triples ``<Φ_ijk^abc| H_N T2 |Φ>_C`` and the disconnected ones
``<Φ_ijk^abc| H_N (T1 + T2) |Φ>`` over those windows as tensors
``[i, j, k, a, b, c]``, so that the triples are evaluated a batch of
occupied indices at a time.
"""

from __future__ import annotations

from hafnium.cc.hamiltonian import NormalOrderedHamiltonian
from hafnium.orbitals import Space
from hafnium.structured import BlockTensor, TensorAxis, contract

OCC, VIR = Space.OCCUPIED, Space.VIRTUAL

FOCK_BLOCKS = ((VIR, OCC),)
"""The blocks of the Fock operator the equations read."""
INTEGRAL_BLOCKS = (
    (OCC, VIR, OCC, OCC),
    (VIR, VIR, OCC, OCC),
    (VIR, VIR, OCC, VIR),
)
"""The blocks of the antisymmetrized integrals the equations read."""


def _windowed(x: BlockTensor, windows: tuple[TensorAxis | None, ...]) -> BlockTensor:
    """Restrict the axes of ``x`` to the windows given, ``None`` leaving one whole."""
    for position, window in enumerate(windows):
        if window is not None:
            x = x.window(position, window)
    return x


def connected_triples(
    h: NormalOrderedHamiltonian,
    t1: BlockTensor,
    t2: BlockTensor,
    i: TensorAxis,
    j: TensorAxis,
    k: TensorAxis,
) -> BlockTensor:
    """``<Φ_ijk^abc| H_N T2 |Φ>`` connected, as ``[i, j, k, a, b, c]``.

    The occupied indices run over the windows ``i``, ``j`` and ``k``.
    """
    v = h.integrals
    x: BlockTensor
    s: BlockTensor
    r: BlockTensor
    x = contract(
        "laij,klbc->ijkabc",
        _windowed(v[OCC, VIR, OCC, OCC], (None, None, i, j)),
        _windowed(t2, (k, None, None, None)),
    )
    s = -x
    x = contract(
        "laij,klbc->ijkabc",
        _windowed(v[OCC, VIR, OCC, OCC], (None, None, i, k)),
        _windowed(t2, (j, None, None, None)),
    )
    s = s + x.permute("ijkabc->ikjabc")
    x = contract(
        "laij,klbc->ijkabc",
        _windowed(v[OCC, VIR, OCC, OCC], (None, None, j, k)),
        _windowed(t2, (i, None, None, None)),
    )
    s = s - x.permute("ijkabc->kijabc")
    r = s - s.permute("ijkabc->ijkbac") + s.permute("ijkabc->ijkbca")
    x = contract(
        "abid,jkcd->ijkabc",
        _windowed(v[VIR, VIR, OCC, VIR], (None, None, i, None)),
        _windowed(t2, (j, k, None, None)),
    )
    s = -x
    x = contract(
        "abid,jkcd->ijkabc",
        _windowed(v[VIR, VIR, OCC, VIR], (None, None, j, None)),
        _windowed(t2, (i, k, None, None)),
    )
    s = s + x.permute("ijkabc->jikabc")
    x = contract(
        "abid,jkcd->ijkabc",
        _windowed(v[VIR, VIR, OCC, VIR], (None, None, k, None)),
        _windowed(t2, (i, j, None, None)),
    )
    s = s - x.permute("ijkabc->jkiabc")
    x = s - s.permute("ijkabc->ijkacb") + s.permute("ijkabc->ijkcab")
    r = r + x
    return r


def disconnected_triples(
    h: NormalOrderedHamiltonian,
    t1: BlockTensor,
    t2: BlockTensor,
    i: TensorAxis,
    j: TensorAxis,
    k: TensorAxis,
) -> BlockTensor:
    """``<Φ_ijk^abc| H_N (T1 + T2) |Φ>`` disconnected, as ``[i, j, k, a, b, c]``.

    The occupied indices run over the windows ``i``, ``j`` and ``k``.
    """
    f, v = h.fock, h.integrals
    x: BlockTensor
    s: BlockTensor
    r: BlockTensor
    x = contract(
        "ai,jkbc->ijkabc",
        _windowed(f[VIR, OCC], (None, i)),
        _windowed(t2, (j, k, None, None)),
    )
    s = x
    x = contract(
        "ai,jkbc->ijkabc",
        _windowed(f[VIR, OCC], (None, j)),
        _windowed(t2, (i, k, None, None)),
    )
    s = s - x.permute("ijkabc->jikabc")
    x = contract(
        "ai,jkbc->ijkabc",
        _windowed(f[VIR, OCC], (None, k)),
        _windowed(t2, (i, j, None, None)),
    )
    s = s + x.permute("ijkabc->jkiabc")
    r = s - s.permute("ijkabc->ijkbac") + s.permute("ijkabc->ijkbca")
    x = contract(
        "abij,kc->ijkabc",
        _windowed(v[VIR, VIR, OCC, OCC], (None, None, i, j)),
        _windowed(t1, (k, None)),
    )
    s = x
    x = contract(
        "abij,kc->ijkabc",
        _windowed(v[VIR, VIR, OCC, OCC], (None, None, i, k)),
        _windowed(t1, (j, None)),
    )
    s = s - x.permute("ijkabc->ikjabc")
    x = contract(
        "abij,kc->ijkabc",
        _windowed(v[VIR, VIR, OCC, OCC], (None, None, j, k)),
        _windowed(t1, (i, None)),
    )
    s = s + x.permute("ijkabc->kijabc")
    x = s - s.permute("ijkabc->ijkacb") + s.permute("ijkabc->ijkcab")
    r = r + x
    return r

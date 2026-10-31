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
from hafnium.correlated import Space
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
    x = contract(
        "laij,klbc->ijkabc",
        _windowed(v[OCC, VIR, OCC, OCC], (None, None, i, j)),
        _windowed(t2, (k, None, None, None)),
    )
    r = -x
    x = contract(
        "laij,klbc->ijkabc",
        _windowed(v[OCC, VIR, OCC, OCC], (None, None, i, k)),
        _windowed(t2, (j, None, None, None)),
    )
    r = r + x.permute("ijkabc->ikjabc")
    x = contract(
        "laij,klbc->ijkabc",
        _windowed(v[OCC, VIR, OCC, OCC], (None, None, j, k)),
        _windowed(t2, (i, None, None, None)),
    )
    r = r - x.permute("ijkabc->kijabc")
    x = contract(
        "laij,klbc->ijkabc",
        _windowed(v[OCC, VIR, OCC, OCC], (None, None, i, j)),
        _windowed(t2, (k, None, None, None)),
    )
    r = r + x.permute("ijkabc->ijkbac")
    x = contract(
        "laij,klbc->ijkabc",
        _windowed(v[OCC, VIR, OCC, OCC], (None, None, i, k)),
        _windowed(t2, (j, None, None, None)),
    )
    r = r - x.permute("ijkabc->ikjbac")
    x = contract(
        "laij,klbc->ijkabc",
        _windowed(v[OCC, VIR, OCC, OCC], (None, None, j, k)),
        _windowed(t2, (i, None, None, None)),
    )
    r = r + x.permute("ijkabc->kijbac")
    x = contract(
        "laij,klbc->ijkabc",
        _windowed(v[OCC, VIR, OCC, OCC], (None, None, i, j)),
        _windowed(t2, (k, None, None, None)),
    )
    r = r - x.permute("ijkabc->ijkbca")
    x = contract(
        "laij,klbc->ijkabc",
        _windowed(v[OCC, VIR, OCC, OCC], (None, None, i, k)),
        _windowed(t2, (j, None, None, None)),
    )
    r = r + x.permute("ijkabc->ikjbca")
    x = contract(
        "laij,klbc->ijkabc",
        _windowed(v[OCC, VIR, OCC, OCC], (None, None, j, k)),
        _windowed(t2, (i, None, None, None)),
    )
    r = r - x.permute("ijkabc->kijbca")
    x = contract(
        "abid,jkcd->ijkabc",
        _windowed(v[VIR, VIR, OCC, VIR], (None, None, i, None)),
        _windowed(t2, (j, k, None, None)),
    )
    r = r - x
    x = contract(
        "abid,jkcd->ijkabc",
        _windowed(v[VIR, VIR, OCC, VIR], (None, None, j, None)),
        _windowed(t2, (i, k, None, None)),
    )
    r = r + x.permute("ijkabc->jikabc")
    x = contract(
        "abid,jkcd->ijkabc",
        _windowed(v[VIR, VIR, OCC, VIR], (None, None, k, None)),
        _windowed(t2, (i, j, None, None)),
    )
    r = r - x.permute("ijkabc->jkiabc")
    x = contract(
        "abid,jkcd->ijkabc",
        _windowed(v[VIR, VIR, OCC, VIR], (None, None, i, None)),
        _windowed(t2, (j, k, None, None)),
    )
    r = r + x.permute("ijkabc->ijkacb")
    x = contract(
        "abid,jkcd->ijkabc",
        _windowed(v[VIR, VIR, OCC, VIR], (None, None, j, None)),
        _windowed(t2, (i, k, None, None)),
    )
    r = r - x.permute("ijkabc->jikacb")
    x = contract(
        "abid,jkcd->ijkabc",
        _windowed(v[VIR, VIR, OCC, VIR], (None, None, k, None)),
        _windowed(t2, (i, j, None, None)),
    )
    r = r + x.permute("ijkabc->jkiacb")
    x = contract(
        "abid,jkcd->ijkabc",
        _windowed(v[VIR, VIR, OCC, VIR], (None, None, i, None)),
        _windowed(t2, (j, k, None, None)),
    )
    r = r - x.permute("ijkabc->ijkcab")
    x = contract(
        "abid,jkcd->ijkabc",
        _windowed(v[VIR, VIR, OCC, VIR], (None, None, j, None)),
        _windowed(t2, (i, k, None, None)),
    )
    r = r + x.permute("ijkabc->jikcab")
    x = contract(
        "abid,jkcd->ijkabc",
        _windowed(v[VIR, VIR, OCC, VIR], (None, None, k, None)),
        _windowed(t2, (i, j, None, None)),
    )
    r = r - x.permute("ijkabc->jkicab")
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
    x = contract(
        "ai,jkbc->ijkabc",
        _windowed(f[VIR, OCC], (None, i)),
        _windowed(t2, (j, k, None, None)),
    )
    r = x
    x = contract(
        "ai,jkbc->ijkabc",
        _windowed(f[VIR, OCC], (None, j)),
        _windowed(t2, (i, k, None, None)),
    )
    r = r - x.permute("ijkabc->jikabc")
    x = contract(
        "ai,jkbc->ijkabc",
        _windowed(f[VIR, OCC], (None, k)),
        _windowed(t2, (i, j, None, None)),
    )
    r = r + x.permute("ijkabc->jkiabc")
    x = contract(
        "ai,jkbc->ijkabc",
        _windowed(f[VIR, OCC], (None, i)),
        _windowed(t2, (j, k, None, None)),
    )
    r = r - x.permute("ijkabc->ijkbac")
    x = contract(
        "ai,jkbc->ijkabc",
        _windowed(f[VIR, OCC], (None, j)),
        _windowed(t2, (i, k, None, None)),
    )
    r = r + x.permute("ijkabc->jikbac")
    x = contract(
        "ai,jkbc->ijkabc",
        _windowed(f[VIR, OCC], (None, k)),
        _windowed(t2, (i, j, None, None)),
    )
    r = r - x.permute("ijkabc->jkibac")
    x = contract(
        "ai,jkbc->ijkabc",
        _windowed(f[VIR, OCC], (None, i)),
        _windowed(t2, (j, k, None, None)),
    )
    r = r + x.permute("ijkabc->ijkbca")
    x = contract(
        "ai,jkbc->ijkabc",
        _windowed(f[VIR, OCC], (None, j)),
        _windowed(t2, (i, k, None, None)),
    )
    r = r - x.permute("ijkabc->jikbca")
    x = contract(
        "ai,jkbc->ijkabc",
        _windowed(f[VIR, OCC], (None, k)),
        _windowed(t2, (i, j, None, None)),
    )
    r = r + x.permute("ijkabc->jkibca")
    x = contract(
        "abij,kc->ijkabc",
        _windowed(v[VIR, VIR, OCC, OCC], (None, None, i, j)),
        _windowed(t1, (k, None)),
    )
    r = r + x
    x = contract(
        "abij,kc->ijkabc",
        _windowed(v[VIR, VIR, OCC, OCC], (None, None, i, k)),
        _windowed(t1, (j, None)),
    )
    r = r - x.permute("ijkabc->ikjabc")
    x = contract(
        "abij,kc->ijkabc",
        _windowed(v[VIR, VIR, OCC, OCC], (None, None, j, k)),
        _windowed(t1, (i, None)),
    )
    r = r + x.permute("ijkabc->kijabc")
    x = contract(
        "abij,kc->ijkabc",
        _windowed(v[VIR, VIR, OCC, OCC], (None, None, i, j)),
        _windowed(t1, (k, None)),
    )
    r = r - x.permute("ijkabc->ijkacb")
    x = contract(
        "abij,kc->ijkabc",
        _windowed(v[VIR, VIR, OCC, OCC], (None, None, i, k)),
        _windowed(t1, (j, None)),
    )
    r = r + x.permute("ijkabc->ikjacb")
    x = contract(
        "abij,kc->ijkabc",
        _windowed(v[VIR, VIR, OCC, OCC], (None, None, j, k)),
        _windowed(t1, (i, None)),
    )
    r = r - x.permute("ijkabc->kijacb")
    x = contract(
        "abij,kc->ijkabc",
        _windowed(v[VIR, VIR, OCC, OCC], (None, None, i, j)),
        _windowed(t1, (k, None)),
    )
    r = r + x.permute("ijkabc->ijkcab")
    x = contract(
        "abij,kc->ijkabc",
        _windowed(v[VIR, VIR, OCC, OCC], (None, None, i, k)),
        _windowed(t1, (j, None)),
    )
    r = r - x.permute("ijkabc->ikjcab")
    x = contract(
        "abij,kc->ijkabc",
        _windowed(v[VIR, VIR, OCC, OCC], (None, None, j, k)),
        _windowed(t1, (i, None)),
    )
    r = r + x.permute("ijkabc->kijcab")
    return r

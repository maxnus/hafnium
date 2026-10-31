"""The spin-orbital CCSD equations, generated code (ADR-0032).

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
from hafnium.correlated import Space
from hafnium.structured import BlockTensor, contract

OCC, VIR = Space.OCCUPIED, Space.VIRTUAL

FOCK_BLOCKS = (
    (OCC, OCC),
    (OCC, VIR),
    (VIR, OCC),
    (VIR, VIR),
)
"""The blocks of the Fock operator the equations read."""
INTEGRAL_BLOCKS = (
    (OCC, OCC, OCC, OCC),
    (OCC, OCC, OCC, VIR),
    (OCC, OCC, VIR, VIR),
    (OCC, VIR, OCC, OCC),
    (OCC, VIR, OCC, VIR),
    (OCC, VIR, VIR, VIR),
    (VIR, VIR, OCC, OCC),
    (VIR, VIR, OCC, VIR),
    (VIR, VIR, VIR, VIR),
)
"""The blocks of the antisymmetrized integrals the equations read."""


def _minus_ij(x: BlockTensor) -> BlockTensor:
    """``(1 − P(ij)) x``."""
    return x - x.permute("ijab->jiab")


def _minus_ab(x: BlockTensor) -> BlockTensor:
    """``(1 − P(ab)) x``."""
    return x - x.permute("ijab->ijba")


def _minus_both(x: BlockTensor) -> BlockTensor:
    """``(1 − P(ij)) (1 − P(ab)) x``."""
    return _minus_ab(_minus_ij(x))


def energy(
    h: NormalOrderedHamiltonian, t1: BlockTensor, t2: BlockTensor
) -> BlockTensor:
    """``<Φ| H_N e^T |Φ>`` connected, as a tensor without axes."""
    f, v = h.fock, h.integrals
    x = contract("ia,ia->", f[OCC, VIR], t1)
    r = x
    x = contract("ijab,ia->jb", v[OCC, OCC, VIR, VIR], t1)
    x = contract("jb,jb->", t1, x)
    r = r + 0.5 * x
    x = contract("ijab,ijab->", v[OCC, OCC, VIR, VIR], t2)
    r = r + 0.25 * x
    return r


def singles_residual(
    h: NormalOrderedHamiltonian, t1: BlockTensor, t2: BlockTensor
) -> BlockTensor:
    """``<Φ_i^a| H_N e^T |Φ>`` connected, as the tensor ``[i, a]``."""
    f, v = h.fock, h.integrals
    x = contract("ji,ja->ia", f[OCC, OCC], t1)
    r = -x
    x = contract("jb,ib->ji", f[OCC, VIR], t1)
    x = contract("ja,ji->ia", t1, x)
    r = r - x
    x = contract("jb,ijab->ia", f[OCC, VIR], t2)
    r = r + x
    x = f[VIR, OCC].permute("ai->ia")
    r = r + x
    x = contract("ab,ib->ia", f[VIR, VIR], t1)
    r = r + x
    x = contract("jkib,kb->ji", v[OCC, OCC, OCC, VIR], t1)
    x = contract("ja,ji->ia", t1, x)
    r = r - x
    x = contract("jkib,jkab->ia", v[OCC, OCC, OCC, VIR], t2)
    r = r - 0.5 * x
    x = contract("jkbc,kc->jb", v[OCC, OCC, VIR, VIR], t1)
    x = contract("ib,jb->ij", t1, x)
    x = contract("ja,ij->ia", t1, x)
    r = r - x
    x = contract("jkbc,ib->jkci", v[OCC, OCC, VIR, VIR], t1)
    x = contract("jkac,jkci->ia", t2, x)
    r = r - 0.5 * x
    x = contract("jkbc,ikbc->ji", v[OCC, OCC, VIR, VIR], t2)
    x = contract("ja,ji->ia", t1, x)
    r = r - 0.5 * x
    x = contract("jkbc,jb->kc", v[OCC, OCC, VIR, VIR], t1)
    x = contract("ikac,kc->ia", t2, x)
    r = r + x
    x = contract("jaib,jb->ia", v[OCC, VIR, OCC, VIR], t1)
    r = r - x
    x = contract("jabc,jc->ab", v[OCC, VIR, VIR, VIR], t1)
    x = contract("ib,ab->ia", t1, x)
    r = r - x
    x = contract("jabc,ijbc->ia", v[OCC, VIR, VIR, VIR], t2)
    r = r - 0.5 * x
    return r


def doubles_residual(
    h: NormalOrderedHamiltonian, t1: BlockTensor, t2: BlockTensor
) -> BlockTensor:
    """``<Φ_ij^ab| H_N e^T |Φ>`` connected, as the tensor ``[i, j, a, b]``."""
    f, v = h.fock, h.integrals
    x = contract("ki,jkab->ijab", f[OCC, OCC], t2)
    r = _minus_ij(x)
    x = contract("kc,ic->ki", f[OCC, VIR], t1)
    x = contract("jkab,ki->ijab", t2, x)
    r = r + _minus_ij(x)
    x = contract("kc,ijbc->kijb", f[OCC, VIR], t2)
    x = contract("ka,kijb->ijab", t1, x)
    r = r + _minus_ab(x)
    x = contract("ac,ijbc->ijab", f[VIR, VIR], t2)
    r = r - _minus_ab(x)
    x = contract("klij,ka->lija", v[OCC, OCC, OCC, OCC], t1)
    x = contract("lb,lija->ijab", t1, x)
    r = r + x
    x = contract("klij,klab->ijab", v[OCC, OCC, OCC, OCC], t2)
    r = r + 0.5 * x
    x = contract("klic,jc->klij", v[OCC, OCC, OCC, VIR], t1)
    x = contract("ka,klij->alij", t1, x)
    x = contract("lb,alij->ijab", t1, x)
    r = r + _minus_ij(x)
    x = contract("klic,jc->klij", v[OCC, OCC, OCC, VIR], t1)
    x = contract("klab,klij->ijab", t2, x)
    r = r + 0.5 * _minus_ij(x)
    x = contract("klic,jlbc->kijb", v[OCC, OCC, OCC, VIR], t2)
    x = contract("ka,kijb->ijab", t1, x)
    r = r - _minus_both(x)
    x = contract("klic,kc->li", v[OCC, OCC, OCC, VIR], t1)
    x = contract("jlab,li->ijab", t2, x)
    r = r - _minus_ij(x)
    x = contract("klcd,ic->kldi", v[OCC, OCC, VIR, VIR], t1)
    x = contract("jd,kldi->jkli", t1, x)
    x = contract("ka,jkli->ajli", t1, x)
    x = contract("lb,ajli->ijab", t1, x)
    r = r + x
    x = contract("klcd,ic->kldi", v[OCC, OCC, VIR, VIR], t1)
    x = contract("jd,kldi->jkli", t1, x)
    x = contract("klab,jkli->ijab", t2, x)
    r = r + 0.5 * x
    x = contract("klcd,ic->kldi", v[OCC, OCC, VIR, VIR], t1)
    x = contract("jlbd,kldi->jbki", t2, x)
    x = contract("ka,jbki->ijab", t1, x)
    r = r - _minus_both(x)
    x = contract("klcd,kd->lc", v[OCC, OCC, VIR, VIR], t1)
    x = contract("ic,lc->il", t1, x)
    x = contract("jlab,il->ijab", t2, x)
    r = r - _minus_ij(x)
    x = contract("klcd,ijcd->klij", v[OCC, OCC, VIR, VIR], t2)
    x = contract("ka,klij->alij", t1, x)
    x = contract("lb,alij->ijab", t1, x)
    r = r + 0.5 * x
    x = contract("klcd,lc->kd", v[OCC, OCC, VIR, VIR], t1)
    x = contract("ijbd,kd->ijbk", t2, x)
    x = contract("ka,ijbk->ijab", t1, x)
    r = r - _minus_ab(x)
    x = contract("klcd,klbd->cb", v[OCC, OCC, VIR, VIR], t2)
    x = contract("ijac,cb->ijab", t2, x)
    r = r - 0.5 * _minus_ab(x)
    x = contract("klcd,ijcd->klij", v[OCC, OCC, VIR, VIR], t2)
    x = contract("klab,klij->ijab", t2, x)
    r = r + 0.25 * x
    x = contract("klcd,jlcd->kj", v[OCC, OCC, VIR, VIR], t2)
    x = contract("ikab,kj->ijab", t2, x)
    r = r - 0.5 * _minus_ij(x)
    x = contract("klcd,ikac->ldia", v[OCC, OCC, VIR, VIR], t2)
    x = contract("jlbd,ldia->ijab", t2, x)
    r = r + _minus_ij(x)
    x = contract("kaij,kb->ijab", v[OCC, VIR, OCC, OCC], t1)
    r = r + _minus_ab(x)
    x = contract("kaic,jc->kaij", v[OCC, VIR, OCC, VIR], t1)
    x = contract("kb,kaij->ijab", t1, x)
    r = r + _minus_both(x)
    x = contract("kaic,jkbc->ijab", v[OCC, VIR, OCC, VIR], t2)
    r = r - _minus_both(x)
    x = contract("kacd,ic->kadi", v[OCC, VIR, VIR, VIR], t1)
    x = contract("jd,kadi->jkai", t1, x)
    x = contract("kb,jkai->ijab", t1, x)
    r = r + _minus_ab(x)
    x = contract("kacd,ic->kadi", v[OCC, VIR, VIR, VIR], t1)
    x = contract("jkbd,kadi->ijab", t2, x)
    r = r - _minus_both(x)
    x = contract("kacd,ijcd->kaij", v[OCC, VIR, VIR, VIR], t2)
    x = contract("kb,kaij->ijab", t1, x)
    r = r + 0.5 * _minus_ab(x)
    x = contract("kacd,kc->ad", v[OCC, VIR, VIR, VIR], t1)
    x = contract("ijbd,ad->ijab", t2, x)
    r = r - _minus_ab(x)
    x = v[VIR, VIR, OCC, OCC].permute("abij->ijab")
    r = r + x
    x = contract("abic,jc->ijab", v[VIR, VIR, OCC, VIR], t1)
    r = r + _minus_ij(x)
    x = contract("abcd,ic->abdi", v[VIR, VIR, VIR, VIR], t1)
    x = contract("jd,abdi->ijab", t1, x)
    r = r + x
    x = contract("abcd,ijcd->ijab", v[VIR, VIR, VIR, VIR], t2)
    r = r + 0.5 * x
    return r

"""The Λ equations of CCSD and its one-particle density, generated code (ADR-0034).

Do not edit: ``python -m hafnium.cc.generator`` derives the terms by
Wick's theorem and writes this program of pairwise contractions, and a
test regenerates it to detect drift. The functions
take the normal-ordered Hamiltonian of a reference in its orbital basis,
the amplitudes ``t1[i, a]`` and ``t2[i, j, a, b]`` of CONVENTIONS.md §3
and the de-excitation amplitudes ``l1[i, a]`` and ``l2[i, j, a, b]``,
and return the residuals ``<Φ| (1 + Λ) H_N e^T |Φ_i^a>`` and
``<Φ| (1 + Λ) H_N e^T |Φ_ij^ab>`` of the Λ equations, linked, as tensors
of the structure of the de-excitation amplitudes, and the blocks of the
one-particle density ``<Φ| (1 + Λ) {p† q} e^T |Φ>`` by the spaces of
``p`` and ``q``.
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


def lambda_singles_residual(
    h: NormalOrderedHamiltonian,
    t1: BlockTensor,
    t2: BlockTensor,
    l1: BlockTensor,
    l2: BlockTensor,
) -> BlockTensor:
    """``<Φ| (1 + Λ) H_N e^T |Φ_i^a>`` linked, as the tensor ``[i, a]``."""
    f, v = h.fock, h.integrals
    x: BlockTensor
    y: BlockTensor
    r: BlockTensor
    x = contract("ij,ja->ia", f[OCC, OCC], l1)
    r = -x
    x = f[OCC, VIR]
    r = r + x
    x = contract("ib,jb->ij", f[OCC, VIR], t1)
    x = contract("ja,ij->ia", l1, x)
    r = r - x
    x = contract("ib,jkbc->ijkc", f[OCC, VIR], t2)
    x = contract("jkac,ijkc->ia", l2, x)
    r = r - 0.5 * x
    x = contract("jk,jb->kb", f[OCC, OCC], t1)
    x = contract("ikab,kb->ia", l2, x)
    r = r - x
    x = contract("jb,ib->ji", t1, l1)
    x = contract("ja,ji->ia", f[OCC, VIR], x)
    r = r - x
    x = contract("jkbc,ikbc->ji", t2, l2)
    x = contract("ja,ji->ia", f[OCC, VIR], x)
    r = r - 0.5 * x
    x = contract("jb,kb->jk", f[OCC, VIR], t1)
    x = contract("jc,jk->ck", t1, x)
    x = contract("ikac,ck->ia", l2, x)
    r = r - x
    x = contract("jb,jkbc->kc", f[OCC, VIR], t2)
    x = contract("ikac,kc->ia", l2, x)
    r = r + x
    x = contract("bj,ijab->ia", f[VIR, OCC], l2)
    r = r + x
    x = contract("ba,ib->ia", f[VIR, VIR], l1)
    r = r + x
    x = contract("bc,jc->bj", f[VIR, VIR], t1)
    x = contract("ijab,bj->ia", l2, x)
    r = r + x
    x = contract("ijkl,jb->iklb", v[OCC, OCC, OCC, OCC], t1)
    x = contract("klab,iklb->ia", l2, x)
    r = r + 0.5 * x
    x = contract("jb,kb->jk", t1, l1)
    x = contract("ijka,jk->ia", v[OCC, OCC, OCC, VIR], x)
    r = r + x
    x = contract("jlbc,klbc->jk", t2, l2)
    x = contract("ijka,jk->ia", v[OCC, OCC, OCC, VIR], x)
    r = r + 0.5 * x
    x = contract("ijkb,jb->ik", v[OCC, OCC, OCC, VIR], t1)
    x = contract("ka,ik->ia", l1, x)
    r = r - x
    x = contract("ijkb,lb->ijkl", v[OCC, OCC, OCC, VIR], t1)
    y = contract("jc,klac->jkla", t1, l2)
    x = contract("ijkl,jkla->ia", x, y)
    r = r + x
    x = contract("ijkb,jlbc->iklc", v[OCC, OCC, OCC, VIR], t2)
    x = contract("klac,iklc->ia", l2, x)
    r = r - x
    x = contract("ijab,jb->ia", v[OCC, OCC, VIR, VIR], t1)
    r = r + x
    x = contract("jc,kc->jk", t1, l1)
    x = contract("kb,jk->bj", t1, x)
    x = contract("ijab,bj->ia", v[OCC, OCC, VIR, VIR], x)
    r = r - x
    x = contract("jc,klcd->jkld", t1, l2)
    x = contract("klbd,jkld->bj", t2, x)
    x = contract("ijab,bj->ia", v[OCC, OCC, VIR, VIR], x)
    r = r - 0.5 * x
    x = contract("jlcd,klcd->jk", t2, l2)
    x = contract("kb,jk->bj", t1, x)
    x = contract("ijab,bj->ia", v[OCC, OCC, VIR, VIR], x)
    r = r - 0.5 * x
    x = contract("jkbc,kc->jb", t2, l1)
    x = contract("ijab,jb->ia", v[OCC, OCC, VIR, VIR], x)
    r = r + x
    x = contract("ijbc,jb->ic", v[OCC, OCC, VIR, VIR], t1)
    x = contract("kc,ic->ki", t1, x)
    x = contract("ka,ki->ia", l1, x)
    r = r + x
    x = contract("ijbc,jb->ic", v[OCC, OCC, VIR, VIR], t1)
    x = contract("klcd,ic->kldi", t2, x)
    x = contract("klad,kldi->ia", l2, x)
    r = r + 0.5 * x
    x = contract("ijbc,kb->ijck", v[OCC, OCC, VIR, VIR], t1)
    y = contract("jd,klad->jkla", t1, l2)
    x = contract("lc,ijck->lijk", t1, x)
    y = contract("jkla,lijk->ia", y, x)
    x = y
    r = r + 0.5 * x
    x = contract("ijbc,klbc->ijkl", v[OCC, OCC, VIR, VIR], t2)
    y = contract("jd,klad->jkla", t1, l2)
    x = contract("ijkl,jkla->ia", x, y)
    r = r + 0.25 * x
    x = contract("ijbc,kb->ijck", v[OCC, OCC, VIR, VIR], t1)
    x = contract("jlcd,ijck->ldik", t2, x)
    x = contract("klad,ldik->ia", l2, x)
    r = r - x
    x = contract("ijbc,jkbc->ik", v[OCC, OCC, VIR, VIR], t2)
    x = contract("ka,ik->ia", l1, x)
    r = r + 0.5 * x
    x = contract("ibjk,jkab->ia", v[OCC, VIR, OCC, OCC], l2)
    r = r - 0.5 * x
    x = contract("ibja,jb->ia", v[OCC, VIR, OCC, VIR], l1)
    r = r - x
    x = contract("ibjc,kc->ibjk", v[OCC, VIR, OCC, VIR], t1)
    x = contract("jkab,ibjk->ia", l2, x)
    r = r - x
    x = contract("jc,jb->cb", t1, l1)
    x = contract("ibac,cb->ia", v[OCC, VIR, VIR, VIR], x)
    r = r + x
    x = contract("jkcd,jkbd->cb", t2, l2)
    x = contract("ibac,cb->ia", v[OCC, VIR, VIR, VIR], x)
    r = r + 0.5 * x
    x = contract("ibcd,jc->ibdj", v[OCC, VIR, VIR, VIR], t1)
    x = contract("kd,ibdj->kibj", t1, x)
    x = contract("jkab,kibj->ia", l2, x)
    r = r - 0.5 * x
    x = contract("ibcd,jkcd->ibjk", v[OCC, VIR, VIR, VIR], t2)
    x = contract("jkab,ibjk->ia", l2, x)
    r = r - 0.25 * x
    x = contract("jb,ilbc->jilc", t1, l2)
    x = contract("kc,jilc->kjil", t1, x)
    x = contract("jkla,kjil->ia", v[OCC, OCC, OCC, VIR], x)
    r = r - 0.5 * x
    x = contract("jkbc,ilbc->jkil", t2, l2)
    x = contract("jkla,jkil->ia", v[OCC, OCC, OCC, VIR], x)
    r = r - 0.25 * x
    x = contract("jklb,jb->kl", v[OCC, OCC, OCC, VIR], t1)
    x = contract("kc,kl->cl", t1, x)
    x = contract("ilac,cl->ia", l2, x)
    r = r + x
    x = contract("jklb,jkbc->lc", v[OCC, OCC, OCC, VIR], t2)
    x = contract("ilac,lc->ia", l2, x)
    r = r + 0.5 * x
    x = contract("jkab,jb->ka", v[OCC, OCC, VIR, VIR], t1)
    y = contract("kc,ic->ki", t1, l1)
    x = contract("ka,ki->ia", x, y)
    r = r + x
    x = contract("jkab,jb->ka", v[OCC, OCC, VIR, VIR], t1)
    y = contract("klcd,ilcd->ki", t2, l2)
    x = contract("ka,ki->ia", x, y)
    r = r + 0.5 * x
    x = contract("jkab,lb->jkal", v[OCC, OCC, VIR, VIR], t1)
    y = contract("jc,ilcd->jild", t1, l2)
    y = contract("kd,jild->kjil", t1, y)
    x = contract("jkal,kjil->ia", x, y)
    r = r + 0.5 * x
    x = contract("jc,ilcd->jild", t1, l2)
    x = contract("klbd,jild->kbji", t2, x)
    x = contract("jkab,kbji->ia", v[OCC, OCC, VIR, VIR], x)
    r = r - x
    x = contract("jkab,lb->jkal", v[OCC, OCC, VIR, VIR], t1)
    y = contract("jkcd,ilcd->jkil", t2, l2)
    x = contract("jkal,jkil->ia", x, y)
    r = r + 0.25 * x
    x = contract("jkbc,ic->jkbi", t2, l1)
    x = contract("jkab,jkbi->ia", v[OCC, OCC, VIR, VIR], x)
    r = r + 0.5 * x
    x = contract("jkbc,jb->kc", v[OCC, OCC, VIR, VIR], t1)
    x = contract("lc,kc->lk", t1, x)
    x = contract("kd,lk->dl", t1, x)
    x = contract("ilad,dl->ia", l2, x)
    r = r - x
    x = contract("jkbc,jb->kc", v[OCC, OCC, VIR, VIR], t1)
    x = contract("klcd,kc->ld", t2, x)
    x = contract("ilad,ld->ia", l2, x)
    r = r + x
    x = contract("jkbc,klbc->jl", v[OCC, OCC, VIR, VIR], t2)
    x = contract("jd,jl->dl", t1, x)
    x = contract("ilad,dl->ia", l2, x)
    r = r + 0.5 * x
    x = contract("jkbc,lb->jkcl", v[OCC, OCC, VIR, VIR], t1)
    x = contract("jkcd,jkcl->dl", t2, x)
    x = contract("ilad,dl->ia", l2, x)
    r = r + 0.5 * x
    x = contract("jc,ikbc->jikb", t1, l2)
    x = contract("jbka,jikb->ia", v[OCC, VIR, OCC, VIR], x)
    r = r - x
    x = contract("jbkc,jc->bk", v[OCC, VIR, OCC, VIR], t1)
    x = contract("ikab,bk->ia", l2, x)
    r = r - x
    x = contract("jbac,jc->ba", v[OCC, VIR, VIR, VIR], t1)
    x = contract("ib,ba->ia", l1, x)
    r = r - x
    x = contract("jbac,kc->jbak", v[OCC, VIR, VIR, VIR], t1)
    y = contract("jd,ikbd->jikb", t1, l2)
    x = contract("jbak,jikb->ia", x, y)
    r = r + x
    x = contract("jkcd,ikbd->jcib", t2, l2)
    x = contract("jbac,jcib->ia", v[OCC, VIR, VIR, VIR], x)
    r = r - x
    x = contract("jbcd,jc->bd", v[OCC, VIR, VIR, VIR], t1)
    x = contract("kd,bd->kb", t1, x)
    x = contract("ikab,kb->ia", l2, x)
    r = r + x
    x = contract("jbcd,jkcd->bk", v[OCC, VIR, VIR, VIR], t2)
    x = contract("ikab,bk->ia", l2, x)
    r = r + 0.5 * x
    x = contract("bcja,ijbc->ia", v[VIR, VIR, OCC, VIR], l2)
    r = r - 0.5 * x
    x = contract("bcad,jd->bcaj", v[VIR, VIR, VIR, VIR], t1)
    x = contract("ijbc,bcaj->ia", l2, x)
    r = r + 0.5 * x
    return r


def lambda_doubles_residual(
    h: NormalOrderedHamiltonian,
    t1: BlockTensor,
    t2: BlockTensor,
    l1: BlockTensor,
    l2: BlockTensor,
) -> BlockTensor:
    """``<Φ| (1 + Λ) H_N e^T |Φ_ij^ab>`` linked, as the tensor ``[i, j, a, b]``."""
    f, v = h.fock, h.integrals
    x: BlockTensor
    y: BlockTensor
    r: BlockTensor
    x = contract("ik,jkab->ijab", f[OCC, OCC], l2)
    r = _minus_ij(x)
    x = contract("ia,jb->ijab", f[OCC, VIR], l1)
    r = r + _minus_both(x)
    x = contract("ic,kc->ik", f[OCC, VIR], t1)
    x = contract("jkab,ik->ijab", l2, x)
    r = r + _minus_ij(x)
    x = contract("kc,ijbc->kijb", t1, l2)
    x = contract("ka,kijb->ijab", f[OCC, VIR], x)
    r = r + _minus_ab(x)
    x = contract("ca,ijbc->ijab", f[VIR, VIR], l2)
    r = r - _minus_ab(x)
    x = contract("ijkl,klab->ijab", v[OCC, OCC, OCC, OCC], l2)
    r = r + 0.5 * x
    x = contract("ijka,kb->ijab", v[OCC, OCC, OCC, VIR], l1)
    r = r + _minus_ab(x)
    x = contract("ijkc,lc->ijkl", v[OCC, OCC, OCC, VIR], t1)
    x = contract("klab,ijkl->ijab", l2, x)
    r = r + x
    x = v[OCC, OCC, VIR, VIR]
    r = r + x
    x = contract("ijac,kc->ijak", v[OCC, OCC, VIR, VIR], t1)
    x = contract("kb,ijak->ijab", l1, x)
    r = r - _minus_ab(x)
    x = contract("klcd,klbd->cb", t2, l2)
    x = contract("ijac,cb->ijab", v[OCC, OCC, VIR, VIR], x)
    r = r - 0.5 * _minus_ab(x)
    x = contract("ijcd,kc->ijdk", v[OCC, OCC, VIR, VIR], t1)
    x = contract("ld,ijdk->lijk", t1, x)
    x = contract("klab,lijk->ijab", l2, x)
    r = r + 0.5 * x
    x = contract("ijcd,klcd->ijkl", v[OCC, OCC, VIR, VIR], t2)
    x = contract("klab,ijkl->ijab", l2, x)
    r = r + 0.25 * x
    x = contract("kc,jlbc->kjlb", t1, l2)
    x = contract("ikla,kjlb->ijab", v[OCC, OCC, OCC, VIR], x)
    r = r + _minus_both(x)
    x = contract("iklc,kc->il", v[OCC, OCC, OCC, VIR], t1)
    x = contract("jlab,il->ijab", l2, x)
    r = r + _minus_ij(x)
    x = contract("kc,jc->kj", t1, l1)
    x = contract("ikab,kj->ijab", v[OCC, OCC, VIR, VIR], x)
    r = r - _minus_ij(x)
    x = contract("klcd,jlcd->kj", t2, l2)
    x = contract("ikab,kj->ijab", v[OCC, OCC, VIR, VIR], x)
    r = r - 0.5 * _minus_ij(x)
    x = contract("ikac,kc->ia", v[OCC, OCC, VIR, VIR], t1)
    x = contract("jb,ia->ijab", l1, x)
    r = r + _minus_both(x)
    x = contract("ikac,lc->ikal", v[OCC, OCC, VIR, VIR], t1)
    y = contract("kd,jlbd->kjlb", t1, l2)
    x = contract("ikal,kjlb->ijab", x, y)
    r = r - _minus_both(x)
    x = contract("ikac,klcd->iald", v[OCC, OCC, VIR, VIR], t2)
    x = contract("jlbd,iald->ijab", l2, x)
    r = r + _minus_both(x)
    x = contract("ikcd,kc->id", v[OCC, OCC, VIR, VIR], t1)
    x = contract("ld,id->li", t1, x)
    x = contract("jlab,li->ijab", l2, x)
    r = r - _minus_ij(x)
    x = contract("ikcd,klcd->il", v[OCC, OCC, VIR, VIR], t2)
    x = contract("jlab,il->ijab", l2, x)
    r = r - 0.5 * _minus_ij(x)
    x = contract("icka,jkbc->ijab", v[OCC, VIR, OCC, VIR], l2)
    r = r - _minus_both(x)
    x = contract("icab,jc->ijab", v[OCC, VIR, VIR, VIR], l1)
    r = r + _minus_ij(x)
    x = contract("icad,kd->icak", v[OCC, VIR, VIR, VIR], t1)
    x = contract("jkbc,icak->ijab", l2, x)
    r = r + _minus_both(x)
    x = contract("kc,ijcd->kijd", t1, l2)
    x = contract("ld,kijd->lkij", t1, x)
    x = contract("klab,lkij->ijab", v[OCC, OCC, VIR, VIR], x)
    r = r + 0.5 * x
    x = contract("klcd,ijcd->klij", t2, l2)
    x = contract("klab,klij->ijab", v[OCC, OCC, VIR, VIR], x)
    r = r + 0.25 * x
    x = contract("klac,kc->la", v[OCC, OCC, VIR, VIR], t1)
    y = contract("ld,ijbd->lijb", t1, l2)
    x = contract("la,lijb->ijab", x, y)
    r = r - _minus_ab(x)
    x = contract("klac,klcd->ad", v[OCC, OCC, VIR, VIR], t2)
    x = contract("ijbd,ad->ijab", l2, x)
    r = r - 0.5 * _minus_ab(x)
    x = contract("kd,ijcd->kijc", t1, l2)
    x = contract("kcab,kijc->ijab", v[OCC, VIR, VIR, VIR], x)
    r = r + x
    x = contract("kcad,kd->ca", v[OCC, VIR, VIR, VIR], t1)
    x = contract("ijbc,ca->ijab", l2, x)
    r = r + _minus_ab(x)
    x = contract("cdab,ijcd->ijab", v[VIR, VIR, VIR, VIR], l2)
    r = r + 0.5 * x
    return r


def one_particle_density(
    t1: BlockTensor, t2: BlockTensor, l1: BlockTensor, l2: BlockTensor
) -> dict[tuple[Space, Space], BlockTensor]:
    """``<Φ| (1 + Λ) {p† q} e^T |Φ>`` connected, by the spaces of ``p`` and ``q``.

    Each block is the tensor ``[p, q]`` with ``q`` conjugated, pairing with
    ``f[p, q]`` in a trace, without the occupations of the reference.
    """
    blocks: dict[tuple[Space, Space], BlockTensor] = {}
    x: BlockTensor
    r: BlockTensor
    x = contract("ia,ja->ij", t1, l1)
    r = -x
    x = contract("ikab,jkab->ij", t2, l2)
    r = r - 0.5 * x
    blocks[OCC, OCC] = r
    x = t1
    r = x
    x = contract("ib,jb->ij", t1, l1)
    x = contract("ja,ij->ia", t1, x)
    r = r - x
    x = contract("ib,jkbc->ijkc", t1, l2)
    x = contract("jkac,ijkc->ia", t2, x)
    r = r - 0.5 * x
    x = contract("ikbc,jkbc->ij", t2, l2)
    x = contract("ja,ij->ia", t1, x)
    r = r - 0.5 * x
    x = contract("ijab,jb->ia", t2, l1)
    r = r + x
    blocks[OCC, VIR] = r
    x = l1.permute("ia->ai")
    r = x
    blocks[VIR, OCC] = r
    x = contract("ib,ia->ab", t1, l1)
    r = x
    x = contract("ijbc,ijac->ab", t2, l2)
    r = r + 0.5 * x
    blocks[VIR, VIR] = r
    return blocks

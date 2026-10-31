"""The normal-ordered Hamiltonian of a reference in its orbital basis (ADR-0032)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from hafnium.correlated import FittedOrbitalTensors
from hafnium.structured import BlockTensor


@dataclass(frozen=True, slots=True, eq=False)
class NormalOrderedHamiltonian:
    """``H_N = Σ f_pq {p† q} + ¼ Σ <pq||rs> {p† q† s r}`` block by block.

    Parameters
    ----------
    fock
        ``f[p, q]`` keyed by the spaces of its slots, ``"oo"``, ``"ov"``,
        ``"vo"`` and ``"vv"``, with ``p`` conjugated.
    integrals
        ``<pq||rs>`` keyed by the spaces of its four slots, ``"oovv"`` and
        so on, with ``p`` and ``q`` conjugated (CONVENTIONS.md §3).
    """

    fock: Mapping[str, BlockTensor]
    integrals: Mapping[str, BlockTensor]

    @classmethod
    def from_fitted(
        cls,
        fitted: FittedOrbitalTensors,
        *,
        fock_blocks: Iterable[str],
        integral_blocks: Iterable[str],
    ) -> NormalOrderedHamiltonian:
        """Form the blocks an equation reads from the fitted tensors.

        The Fock blocks are those of the canonical orbitals of the
        reference (:meth:`~hafnium.correlated.OrbitalSpaces.fock_block`)
        over the ring of the fitted tensors, the integral blocks the
        antisymmetrized ones of
        :meth:`~hafnium.correlated.FittedOrbitalTensors.antisymmetrized`.
        """
        return cls(
            {
                slots: fitted.spaces.fock_block(slots, ring=fitted.ring)
                for slots in fock_blocks
            },
            {slots: fitted.antisymmetrized(slots) for slots in integral_blocks},
        )

"""The normal-ordered Hamiltonian of a reference in its orbital basis (ADR-0032)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from hafnium.orbitals import FittedOrbitalTensors, Space
from hafnium.structured import BlockTensor

FockBlock = tuple[Space, Space]
"""The spaces of the two slots of a Fock block."""
IntegralBlock = tuple[Space, Space, Space, Space]
"""The spaces of the four slots of a block of ``<pq||rs>``."""


@dataclass(frozen=True, slots=True, eq=False)
class NormalOrderedHamiltonian:
    """``H_N = Σ f_pq {p† q} + ¼ Σ <pq||rs> {p† q† s r}`` block by block.

    Parameters
    ----------
    fock
        ``f[p, q]`` keyed by the spaces of its two slots, with ``p``
        conjugated.
    integrals
        ``<pq||rs>`` keyed by the spaces of its four slots, with ``p`` and
        ``q`` conjugated (CONVENTIONS.md §3).
    """

    fock: Mapping[FockBlock, BlockTensor]
    integrals: Mapping[IntegralBlock, BlockTensor]

    @classmethod
    def from_fitted(
        cls,
        fitted: FittedOrbitalTensors,
        *,
        fock_blocks: Iterable[FockBlock],
        integral_blocks: Iterable[IntegralBlock],
    ) -> NormalOrderedHamiltonian:
        """Form the blocks an equation reads from the fitted tensors.

        The Fock blocks are those of the canonical orbitals of the
        reference (:meth:`~hafnium.orbitals.OrbitalSpaces.fock_block`)
        over the ring of the fitted tensors, the integral blocks the
        antisymmetrized ones of
        :meth:`~hafnium.orbitals.FittedOrbitalTensors.antisymmetrized`.
        """
        return cls(
            {
                slots: fitted.spaces.fock_block(slots, ring=fitted.ring)
                for slots in fock_blocks
            },
            {slots: fitted.antisymmetrized(slots) for slots in integral_blocks},
        )

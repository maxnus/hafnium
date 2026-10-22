"""Spin expectation values of a single determinant in any Fukutome class.

For a determinant with spin-orbital density ``D`` and atomic overlap
``S``, with ``Q = D (I ⊗ S)`` and spin blocks ``Q_στ``:
``⟨S_z⟩ = (tr Q_αα − tr Q_ββ) / 2``, ``⟨S_+⟩ = tr Q_βα`` so that
``⟨S_x⟩ = Re ⟨S_+⟩`` and ``⟨S_y⟩ = Im ⟨S_+⟩``, and

``⟨S²⟩ = 3N/4 + |⟨S⟩|² − tr(Q_tot²) / 2 + tr(Q²) / 4``

with ``Q_tot = Q_αα + Q_ββ``, from ``S² = Σ_i s_i² + Σ_{i≠j} s_i · s_j`` and
the identity ``Σ_a σ_a ⊗ σ_a = 2 SWAP − 1``. For collinear determinants
this reduces to ``S_z (S_z + 1) + N_β − tr(Q_αα Q_ββ)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hafnium.backend import Array, Namespace, array_namespace
from hafnium.errors import InputError
from hafnium.structured import BlockMatrix
from hafnium.structured.dense import block_diag


@dataclass(frozen=True, slots=True)
class SpinExpectation:
    """``⟨S_x⟩, ⟨S_y⟩, ⟨S_z⟩`` and ``⟨S²⟩`` in units of ħ."""

    vector: tuple[float, float, float]
    square: float

    @property
    def multiplicity(self) -> float:
        """``2S + 1`` with ``S`` solving ``S (S + 1) = ⟨S²⟩``."""
        return math.sqrt(1.0 + 4.0 * self.square)


def spin_expectation(density: BlockMatrix, overlap: Array) -> SpinExpectation:
    """Spin expectation values of the determinant with ``density``.

    Parameters
    ----------
    density
        Structured density; any class, taken through its spin-orbital form.
    overlap : Array[ao, ao]
    """
    if density.structure.k_axis is not None:
        raise InputError(
            "spin expectation values across k-points are not implemented yet"
        )
    dense = density.to_dense()
    xp = array_namespace(dense, overlap)
    n = int(overlap.shape[0])
    if dense.shape != (2 * n, 2 * n):
        raise InputError("the density and the overlap act on different bases")
    q = dense @ block_diag(overlap, overlap)
    trace_aa, trace_bb = xp.linalg.trace(q[:n, :n]), xp.linalg.trace(q[n:, n:])
    n_electrons = _real(xp, trace_aa + trace_bb)
    s_plus = xp.linalg.trace(q[n:, :n])
    vector = (
        _real(xp, s_plus),
        float(xp.imag(s_plus)),
        _real(xp, trace_aa - trace_bb) / 2,
    )
    q_total = q[:n, :n] + q[n:, n:]
    square = (
        0.75 * n_electrons
        + sum(component**2 for component in vector)
        - _real(xp, xp.linalg.trace(q_total @ q_total)) / 2
        + _real(xp, xp.linalg.trace(q @ q)) / 4
    )
    return SpinExpectation(vector, square)


def _real(xp: Namespace, value: Array) -> float:
    return float(xp.real(value))

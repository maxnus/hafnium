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

from hafnium.backend import Array, Namespace, array_namespace, block_diag
from hafnium.errors import InputError
from hafnium.structured import BlockMatrix


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

    Over k-points the traces are weighted sums over the full k set, so the
    values are per cell; orbitals at different k-points are orthogonal, so
    the two-body term of ``⟨S²⟩`` has no cross terms between k-points. The
    supercell of a mesh of ``n_kpt`` points, which PySCF's k-point classes
    report, has ``n_kpt (⟨S²⟩ − |⟨S⟩|²) + n_kpt² |⟨S⟩|²``.

    Parameters
    ----------
    density
        Structured density; any class, taken through its spin-orbital form.
    overlap : Array[ao, ao]
        ``[kpt, ao, ao]`` over the full k set with a k axis.
    """
    dense = density.to_dense()
    xp = array_namespace(dense, overlap)
    axis = density.structure.k_axis
    if axis is None:
        dense, overlap = xp.expand_dims(dense, axis=0), xp.expand_dims(overlap, axis=0)
    weights = xp.asarray((1.0,) if axis is None else axis.weights, dtype=xp.float64)
    n = int(overlap.shape[-1])
    if dense.shape[-2:] != (2 * n, 2 * n) or int(overlap.shape[0]) != int(
        dense.shape[0]
    ):
        raise InputError("the density and the overlap act on different bases")
    q = dense @ block_diag(overlap, overlap)

    def weighted(values: Array) -> Array:
        return xp.sum(weights * values)

    trace_aa = weighted(xp.linalg.trace(q[:, :n, :n]))
    trace_bb = weighted(xp.linalg.trace(q[:, n:, n:]))
    n_electrons = _real(xp, trace_aa + trace_bb)
    s_plus = weighted(xp.linalg.trace(q[:, n:, :n]))
    vector = (
        _real(xp, s_plus),
        float(xp.imag(s_plus)),
        _real(xp, trace_aa - trace_bb) / 2,
    )
    q_total = q[:, :n, :n] + q[:, n:, n:]
    square = (
        0.75 * n_electrons
        + sum(component**2 for component in vector)
        - _real(xp, weighted(xp.linalg.trace(q_total @ q_total))) / 2
        + _real(xp, weighted(xp.linalg.trace(q @ q))) / 4
    )
    return SpinExpectation(vector, square)


def _real(xp: Namespace, value: Array) -> float:
    return float(xp.real(value))

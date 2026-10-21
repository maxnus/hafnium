"""Direct inversion in the iterative subspace over structured matrices.

Pulay's method: the next Fock matrix is the combination ``Σ_i c_i F_i`` of
the stored matrices whose combined error ``Σ_i c_i e_i`` has the smallest
norm under ``Σ_i c_i = 1``. The history is an immutable value; each
iteration produces a new one.
"""

from __future__ import annotations

from dataclasses import dataclass

from hafnium.backend import Array, adjoint, array_namespace, eigh
from hafnium.errors import InputError
from hafnium.structured import BlockMatrix

NULL_SPACE_TOLERANCE = 1e-14
"""Eigenvalues of the DIIS system below this fraction of the largest are dropped.

Error vectors become linearly dependent near convergence or in small
problems; the minimum-norm solution of the singular system is then taken,
as PySCF does.
"""


@dataclass(frozen=True, slots=True, eq=False)
class Diis:
    """The last ``size`` Fock matrices with their error vectors.

    Parameters
    ----------
    size
        Maximum number of stored pairs; ``1`` disables extrapolation.
    """

    size: int
    focks: tuple[BlockMatrix, ...] = ()
    errors: tuple[BlockMatrix, ...] = ()

    def __post_init__(self) -> None:
        if self.size < 1:
            raise InputError("the DIIS subspace holds at least one matrix")
        if len(self.focks) != len(self.errors) or len(self.focks) > self.size:
            raise InputError("DIIS stores one error per Fock matrix, up to its size")

    def __len__(self) -> int:
        return len(self.focks)

    def push(self, fock: BlockMatrix, error: BlockMatrix) -> Diis:
        """Return the history extended by one pair, dropping the oldest if full."""
        focks = (*self.focks, fock)[-self.size :]
        errors = (*self.errors, error)[-self.size :]
        return Diis(self.size, focks, errors)

    def extrapolate(self) -> BlockMatrix:
        """Return the error-minimizing combination of the stored Fock matrices."""
        if not self.focks:
            raise InputError("nothing to extrapolate: the DIIS history is empty")
        if len(self.focks) == 1:
            return self.focks[0]
        coefficients = _coefficients(self.errors)
        combined = self.focks[0].map_blocks(lambda block: block * coefficients[0])
        for index, fock in enumerate(self.focks[1:], start=1):
            combined = combined + fock.map_blocks(
                lambda block, c=coefficients[index]: block * c
            )
        return combined


def _coefficients(errors: tuple[BlockMatrix, ...]) -> Array:
    """Solve the DIIS equations for the real overlap of the error vectors.

    The system ``[[B, -1], [-1ᵀ, 0]] [c, λ] = [0, -1]`` is solved through
    its eigendecomposition, dropping the null space (see
    :data:`NULL_SPACE_TOLERANCE`).
    """
    xp = array_namespace(*errors[0].blocks)
    m = len(errors)
    overlaps = xp.stack(
        [
            xp.stack([xp.real(a.adjoint().trace_product(b)) for b in errors])
            for a in errors
        ]
    )
    overlaps = overlaps / xp.max(xp.linalg.diagonal(overlaps))
    top = xp.concat([overlaps, -xp.ones((m, 1), dtype=xp.float64)], axis=1)
    bottom = xp.concat(
        [-xp.ones((1, m), dtype=xp.float64), xp.zeros((1, 1), dtype=xp.float64)],
        axis=1,
    )
    system = xp.concat([top, bottom], axis=0)
    right = xp.concat([xp.zeros(m, dtype=xp.float64), -xp.ones(1, dtype=xp.float64)])
    eigenvalues, eigenvectors = eigh(system)
    keep = xp.abs(eigenvalues) > NULL_SPACE_TOLERANCE * xp.max(xp.abs(eigenvalues))
    inverse = xp.where(keep, 1.0 / xp.where(keep, eigenvalues, 1.0), 0.0)
    solution = eigenvectors @ (inverse * (adjoint(eigenvectors) @ right))
    return solution[:m]

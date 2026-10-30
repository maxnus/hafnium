"""Direct inversion in the iterative subspace over cluster amplitudes.

Pulay's method as :class:`hafnium.scf.diis.Diis` applies it to Fock
matrices: the next amplitudes are the combination ``Σ_k c_k t_k`` of the
stored ones whose combined error ``Σ_k c_k e_k`` has the smallest norm
under ``Σ_k c_k = 1``, the error being the update the residual asked for.
The amplitudes of one iteration are a tuple of block tensors, the
singles and doubles, and the overlap of two errors is the real part of
their inner product over the stored blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

from hafnium.backend import array_namespace
from hafnium.errors import InputError
from hafnium.scf.diis import solve_diis
from hafnium.structured import BlockTensor

Amplitudes = tuple[BlockTensor, ...]


@dataclass(frozen=True, slots=True, eq=False)
class AmplitudeDiis:
    """The last ``size`` sets of amplitudes with their errors.

    Parameters
    ----------
    size
        Maximum number of stored pairs; ``1`` disables extrapolation.
    """

    size: int
    amplitudes: tuple[Amplitudes, ...] = ()
    errors: tuple[Amplitudes, ...] = ()

    def __post_init__(self) -> None:
        if self.size < 1:
            raise InputError("the DIIS subspace holds at least one set of amplitudes")
        if len(self.amplitudes) != len(self.errors) or len(self.amplitudes) > self.size:
            raise InputError(
                "DIIS stores one error per set of amplitudes, up to its size"
            )

    def __len__(self) -> int:
        return len(self.amplitudes)

    def push(self, amplitudes: Amplitudes, errors: Amplitudes) -> AmplitudeDiis:
        """Return the history extended by one pair, dropping the oldest if full."""
        if len(amplitudes) != len(errors) or (
            self.amplitudes and len(amplitudes) != len(self.amplitudes[0])
        ):
            raise InputError(
                "every amplitude tensor has an error tensor, as many as the history "
                "holds"
            )
        return AmplitudeDiis(
            self.size,
            (*self.amplitudes, amplitudes)[-self.size :],
            (*self.errors, errors)[-self.size :],
        )

    def extrapolate(self) -> Amplitudes:
        """Return the error-minimizing combination of the stored amplitudes."""
        if not self.amplitudes:
            raise InputError("nothing to extrapolate: the DIIS history is empty")
        if len(self.amplitudes) == 1:
            return self.amplitudes[0]
        xp = array_namespace(*(block for e in self.errors[0] for block in e.blocks))
        overlaps = xp.asarray(
            [[overlap(a, b) for b in self.errors] for a in self.errors],
            dtype=xp.float64,
        )
        coefficients = [float(c) for c in solve_diis(overlaps)]
        combined: list[BlockTensor] = []
        for position in range(len(self.amplitudes[0])):
            total = coefficients[0] * self.amplitudes[0][position]
            for c, amplitudes in zip(
                coefficients[1:], self.amplitudes[1:], strict=True
            ):
                total = total + c * amplitudes[position]
            combined.append(total)
        return tuple(combined)


def overlap(a: Amplitudes, b: Amplitudes) -> float:
    """Real part of ``Σ conj(a) b`` over the stored blocks of every tensor."""
    total = 0.0
    for x, y in zip(a, b, strict=True):
        for block_x, block_y in zip(x.blocks, y.blocks, strict=True):
            xp = array_namespace(block_x, block_y)
            total += float(xp.real(xp.sum(xp.conj(block_x) * block_y)))
    return total

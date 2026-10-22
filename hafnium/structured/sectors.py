"""Irrep sectors of a block structure: the SALC basis of each irrep.

An Abelian point group splits every spin block of a structure into one
block per irrep (PLAN.md §5.2, ADR-0023). In the basis of
symmetry-adapted linear combinations (SALCs) every operator that commutes
with the group is block-diagonal over the irreps, with no change of ring
or weight. The sectors are described by the orthonormal SALC coefficients
of each irrep the basis spans, ``U_Γ [ao, salc]``; together the columns
form an orthogonal matrix, so a spatial operator ``X`` has the irrep
blocks ``U_Γᵀ X U_Γ`` and is recovered as ``Σ_Γ U_Γ X_Γ U_Γᵀ``, and orbital
coefficients ``C_Γ [salc, mo]`` of a sector are ``U_Γ C_Γ`` in the atomic
basis. Irreps the basis does not span have no sector.

:class:`IrrepSectors` knows nothing about point groups beyond the name of
the group and the labels of its irreps. It is built by
:meth:`hafnium.symmetry.SymmetryAdaptation.irreps`, so that ``symmetry``
depends on ``structured`` and not the other way round.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hafnium.backend import Array, array_namespace, block_diag
from hafnium.errors import StructureError

ORTHONORMALITY_TOLERANCE = 1e-10
"""Largest deviation of the SALC coefficients from an orthogonal matrix."""


@dataclass(frozen=True, slots=True, eq=False)
class IrrepSectors:
    """The irreps a basis spans, with the SALC coefficients of each.

    Two sector sets are equal when they carry the same group, labels and
    coefficients, so structures built from equal adaptations compare equal.

    Parameters
    ----------
    group
        Name of the point group, for messages.
    labels
        Mulliken label of every sector, in the group's order of irreps.
    coefficients
        One ``[ao, salc]`` array per sector with at least one column; the
        columns of all sectors together form an orthogonal matrix.
    """

    group: str
    labels: tuple[str, ...]
    coefficients: tuple[Array, ...]

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.labels)
        coefficients = tuple(self.coefficients)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "coefficients", coefficients)
        if not labels:
            raise StructureError("irrep sectors need at least one irrep")
        if len(set(labels)) != len(labels):
            raise StructureError(f"irrep labels are distinct, got {labels}")
        if len(coefficients) != len(labels):
            raise StructureError(
                f"{len(labels)} irrep labels need {len(labels)} coefficient arrays"
            )
        xp = array_namespace(*coefficients)
        for label, block in zip(labels, coefficients, strict=True):
            if block.ndim != 2 or int(block.shape[1]) < 1:
                raise StructureError(
                    f"the SALC coefficients of {label} are [ao, salc] with at least "
                    "one column; irreps the basis does not span have no sector"
                )
            if xp.isdtype(block.dtype, "complex floating"):
                raise StructureError("SALC coefficients of an Abelian group are real")
        n = int(coefficients[0].shape[0])
        if any(int(block.shape[0]) != n for block in coefficients):
            raise StructureError("every sector acts on the same atomic basis")
        transformation = xp.concat(coefficients, axis=1)
        if int(transformation.shape[1]) != n:
            raise StructureError(
                f"the sectors hold {int(transformation.shape[1])} functions, the "
                f"basis {n}: the SALCs of every spanned irrep must be included"
            )
        deviation = xp.max(
            xp.abs(
                xp.linalg.matrix_transpose(transformation) @ transformation - xp.eye(n)
            )
        )
        if float(deviation) > ORTHONORMALITY_TOLERANCE:
            raise StructureError(
                "the SALC coefficients of all sectors form an orthogonal matrix "
                f"(deviation {float(deviation):.2e})"
            )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, IrrepSectors):
            return NotImplemented
        if self.group != other.group or self.labels != other.labels:
            return False
        if self.dimensions != other.dimensions or self.n_functions != other.n_functions:
            return False
        xp = array_namespace(*self.coefficients)
        return all(
            bool(xp.all(a == b))
            for a, b in zip(self.coefficients, other.coefficients, strict=True)
        )

    def __hash__(self) -> int:
        return hash((self.group, self.labels, self.dimensions))

    @property
    def n_irreps(self) -> int:
        """Number of sectors."""
        return len(self.labels)

    @property
    def n_functions(self) -> int:
        """Number of atomic basis functions."""
        return int(self.coefficients[0].shape[0])

    @property
    def dimensions(self) -> tuple[int, ...]:
        """Number of SALCs of each sector."""
        return tuple(int(block.shape[1]) for block in self.coefficients)

    def index(self, label: str) -> int:
        """Position of the sector with Mulliken label ``label``.

        Raises
        ------
        StructureError
            If no sector carries the label: the group has no such irrep, or
            the basis does not span it.
        """
        try:
            return self.labels.index(label)
        except ValueError:
            raise StructureError(
                f"no sector is labelled {label!r}; the basis spans the irreps "
                f"{list(self.labels)} of {self.group}"
            ) from None

    def basis(self, irrep: int, *, spinor: bool) -> Array:
        """SALC coefficients of ``irrep`` on the spatial or the spinor space.

        On the spinor space the alpha and beta functions are adapted alike:
        ``block_diag(U_Γ, U_Γ)`` with alpha components first.
        """
        block = self.coefficients[irrep]
        return block_diag(block, block) if spinor else block

    def to_atomic(
        self, parts: Sequence[Array], *, spinor: bool, orbitals: bool
    ) -> Array:
        """Bring one array per sector to the atomic basis.

        Operators (``orbitals`` false) are square ``[salc, salc]`` blocks
        and return ``Σ_Γ U_Γ X_Γ U_Γᵀ``; orbital coefficients ``[salc, mo]``
        return ``U_Γ C_Γ`` of every sector side by side, ``[ao, mo]``, in
        the order of the sectors. Leading batch axes are kept.
        """
        if len(parts) != self.n_irreps:
            raise StructureError(
                f"{self.n_irreps} sectors need {self.n_irreps} arrays, got {len(parts)}"
            )
        xp = array_namespace(*parts)
        columns: list[Array] = []
        total: Array | None = None
        for irrep, part in enumerate(parts):
            basis = self.basis(irrep, spinor=spinor)
            if int(part.shape[-2]) != int(basis.shape[1]):
                raise StructureError(
                    f"a block of the {self.labels[irrep]} sector has "
                    f"{int(part.shape[-2])} rows, the sector {int(basis.shape[1])} "
                    "functions"
                )
            if orbitals:
                columns.append(basis @ part)
                continue
            if int(part.shape[-1]) != int(basis.shape[1]):
                raise StructureError(
                    f"a block of the {self.labels[irrep]} sector is not square: "
                    "orbital coefficients enter the atomic basis with orbitals=True"
                )
            term = basis @ part @ xp.linalg.matrix_transpose(basis)
            total = term if total is None else total + term
        if orbitals:
            return xp.concat(columns, axis=-1)
        assert total is not None
        return total

    def from_atomic(self, matrix: Array, *, spinor: bool) -> tuple[Array, ...]:
        """Project an operator in the atomic basis onto the sectors.

        Returns ``U_Γᵀ X U_Γ`` per sector; the couplings between sectors,
        which vanish for an operator that commutes with the group, are
        dropped. Leading batch axes are kept.
        """
        n = 2 * self.n_functions if spinor else self.n_functions
        if matrix.ndim < 2 or tuple(int(s) for s in matrix.shape[-2:]) != (n, n):
            raise StructureError(
                f"an operator over the {'spinor' if spinor else 'atomic'} basis "
                f"is [..., {n}, {n}]"
            )
        xp = array_namespace(matrix)
        parts: list[Array] = []
        for irrep in range(self.n_irreps):
            basis = self.basis(irrep, spinor=spinor)
            parts.append(xp.linalg.matrix_transpose(basis) @ matrix @ basis)
        return tuple(parts)

"""Declarative symmetry specification of a mean-field state.

The eight Fukutome classes of Hartree–Fock determinants are the combinations
of a spin structure (restricted, unrestricted, generalized) with the
treatment of time reversal (real orbitals, complex orbitals, or complex
orbitals with Kramers pairing). The table follows Table I of Henderson,
Jiménez-Hoyos and Scuseria, arXiv:1710.00932, where Θ = −iσ_y K is time
reversal and K is complex conjugation. A specification may add the
k-point set of a periodic system or an Abelian point group of a molecule
(ADR-0023).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum

from hafnium.errors import InputError
from hafnium.structured import BlockStructure, Ring
from hafnium.symmetry.adaptation import SymmetryAdaptation
from hafnium.symmetry.kpoints import KPoints
from hafnium.symmetry.point_group import PointGroup


class SpinSymmetry(Enum):
    """Spin structure of the determinant."""

    RESTRICTED = "restricted"
    """Alpha and beta electrons share spatial orbitals (S² and S_z preserved)."""
    UNRESTRICTED = "unrestricted"
    """Separate alpha and beta orbitals, collinear spin (S_z preserved)."""
    GENERALIZED = "generalized"
    """Spinor orbitals mixing alpha and beta, noncollinear spin."""


class Field(Enum):
    """Treatment of time reversal and complex conjugation."""

    REAL = "real"
    """Real orbitals: invariant under complex conjugation K."""
    COMPLEX = "complex"
    """Complex orbitals with no time-reversal constraint."""
    PAIRED = "paired"
    """Complex orbitals invariant under time reversal Θ (Kramers pairs)."""


@dataclass(frozen=True, slots=True)
class FukutomeClass:
    """One of the eight symmetry classes of Hartree–Fock determinants."""

    acronym: str
    full_name: str
    common_name: str
    preserved: tuple[str, ...]
    """Symmetry operators preserved, among ``S^2``, ``S_z``, ``K`` and ``Theta``."""


FUKUTOME_CLASSES: dict[tuple[SpinSymmetry, Field], FukutomeClass] = {
    (SpinSymmetry.RESTRICTED, Field.REAL): FukutomeClass(
        "TICS",
        "time-reversal invariant closed shell",
        "real RHF",
        ("S^2", "S_z", "K", "Theta"),
    ),
    (SpinSymmetry.RESTRICTED, Field.COMPLEX): FukutomeClass(
        "CCW", "charge current wave", "complex RHF", ("S^2", "S_z")
    ),
    (SpinSymmetry.UNRESTRICTED, Field.REAL): FukutomeClass(
        "ASDW", "axial spin density wave", "real UHF", ("S_z", "K")
    ),
    (SpinSymmetry.UNRESTRICTED, Field.COMPLEX): FukutomeClass(
        "ASW", "axial spin wave", "complex UHF", ("S_z",)
    ),
    (SpinSymmetry.UNRESTRICTED, Field.PAIRED): FukutomeClass(
        "ASCW", "axial spin current wave", "paired UHF", ("S_z", "Theta")
    ),
    (SpinSymmetry.GENERALIZED, Field.REAL): FukutomeClass(
        "TSDW", "torsional spin density wave", "real GHF", ("K",)
    ),
    (SpinSymmetry.GENERALIZED, Field.COMPLEX): FukutomeClass(
        "TSW", "torsional spin wave", "complex GHF", ()
    ),
    (SpinSymmetry.GENERALIZED, Field.PAIRED): FukutomeClass(
        "TSCW", "torsional spin current wave", "paired GHF", ("Theta",)
    ),
}
"""The Fukutome classes keyed by (spin symmetry, field)."""

_COMMON_SPIN = {
    "rhf": SpinSymmetry.RESTRICTED,
    "uhf": SpinSymmetry.UNRESTRICTED,
    "ghf": SpinSymmetry.GENERALIZED,
}
_COMMON_FIELD = {"real": Field.REAL, "complex": Field.COMPLEX, "paired": Field.PAIRED}


@dataclass(frozen=True, slots=True)
class SymmetrySpec:
    """Symmetries imposed on a mean-field state.

    Parameters
    ----------
    spin
        Spin structure.
    field
        Treatment of time reversal. ``Field.PAIRED`` with a restricted spin
        structure is rejected: a restricted determinant that is invariant
        under time reversal can always be chosen real, so it is ``REAL``.
    kpoints
        The k-point set of a periodic system.
    point_group
        An Abelian point group imposed on a molecule, which must be in the
        group's standard frame (ADR-0022): the state is then blocked over
        the irreps of the group (ADR-0023). Not combined with k-points.

    Examples
    --------
    >>> SymmetrySpec.from_name("complex UHF").fukutome.acronym
    'ASW'
    """

    spin: SpinSymmetry = SpinSymmetry.RESTRICTED
    field: Field = Field.REAL
    kpoints: KPoints | None = dataclass_field(default=None, kw_only=True)
    point_group: PointGroup | None = dataclass_field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.spin is SpinSymmetry.RESTRICTED and self.field is Field.PAIRED:
            raise InputError(
                "a restricted determinant invariant under time reversal is real RHF; "
                "use Field.REAL"
            )
        if self.kpoints is not None and self.point_group is not None:
            raise InputError(
                "a point group blocks the state of a molecule; a cell carries its "
                "space group through the k-point set instead"
            )

    @classmethod
    def from_name(
        cls, name: str, *, point_group: PointGroup | str | None = None
    ) -> SymmetrySpec:
        """Build a specification from a common or Fukutome name.

        Accepted forms, case-insensitively: ``"RHF"``, ``"UHF"``, ``"GHF"``
        (real), ``"complex UHF"``, ``"paired GHF"`` and so on, or a Fukutome
        acronym such as ``"ASCW"``. ``point_group`` names an Abelian group
        to impose, see :meth:`with_point_group`.
        """
        words = name.strip().lower().split()
        if len(words) == 1 and words[0].upper() in {
            c.acronym for c in FUKUTOME_CLASSES.values()
        }:
            for (spin, field), fukutome in FUKUTOME_CLASSES.items():
                if fukutome.acronym == words[0].upper():
                    return cls(spin, field).with_point_group(point_group)
        if len(words) == 1:
            words = ["real", *words]
        if len(words) == 2 and words[0] in _COMMON_FIELD and words[1] in _COMMON_SPIN:
            return cls(
                _COMMON_SPIN[words[1]], _COMMON_FIELD[words[0]]
            ).with_point_group(point_group)
        raise InputError(f"unknown symmetry name {name!r}")

    def with_point_group(self, group: PointGroup | str | None) -> SymmetrySpec:
        """Return the same class with the Abelian point group ``group`` imposed.

        A name is looked up with :meth:`PointGroup.named`; ``None`` imposes
        no spatial symmetry.
        """
        if isinstance(group, str):
            group = PointGroup.named(group)
        return SymmetrySpec(
            self.spin, self.field, kpoints=self.kpoints, point_group=group
        )

    @property
    def fukutome(self) -> FukutomeClass:
        """The Fukutome class of this specification."""
        return FUKUTOME_CLASSES[(self.spin, self.field)]

    def with_kpoints(self, kpoints: KPoints | None) -> SymmetrySpec:
        """Return the same class over ``kpoints``, the translational sectors."""
        return SymmetrySpec(
            self.spin, self.field, kpoints=kpoints, point_group=self.point_group
        )

    def structure(self, adaptation: SymmetryAdaptation | None = None) -> BlockStructure:
        """Block structure of one-body operators (Fock, density) in this class.

        With ``kpoints`` the structure carries their k axis: the blocks
        become batches over the stored k-points, a half zone for the real
        and paired classes (ADR-0007, ADR-0015). With a ``point_group`` the
        blocks are multiplied by the irrep sectors of the basis, which the
        symmetry ``adaptation`` of the basis provides (ADR-0023);
        :meth:`hafnium.scf.Hamiltonian.structure` passes the adaptation of
        its Hamiltonian.

        Raises
        ------
        InputError
            If a point group is imposed without an adaptation, or with one
            built for another group.
        """
        structure = self._spin_structure()
        if self.kpoints is not None:
            return structure.with_k_axis(self.kpoints.axis())
        if self.point_group is None:
            return structure
        if adaptation is None:
            raise InputError(
                f"imposing the point group {self.point_group.name} needs the "
                "symmetry adaptation of the basis: pass Hamiltonian.adaptation, or "
                "build the structure with Hamiltonian.structure(spec)"
            )
        if adaptation.group != self.point_group:
            raise InputError(
                f"the specification imposes {self.point_group.name}, the symmetry "
                f"adaptation is built for {adaptation.group.name}"
            )
        return structure.with_irreps(adaptation.irreps())

    def _spin_structure(self) -> BlockStructure:
        if self.field is Field.PAIRED:
            if self.spin is SpinSymmetry.UNRESTRICTED:
                return BlockStructure.paired_unrestricted()
            return BlockStructure.paired_generalized()
        ring = Ring.REAL if self.field is Field.REAL else Ring.COMPLEX
        if self.spin is SpinSymmetry.RESTRICTED:
            return BlockStructure.restricted(ring)
        if self.spin is SpinSymmetry.UNRESTRICTED:
            return BlockStructure.unrestricted(ring)
        return BlockStructure.generalized(ring)

    def validate_electrons(self, n_alpha: int, n_beta: int) -> None:
        """Check that an electron count is representable in this class.

        Raises
        ------
        InputError
            For restricted or paired-unrestricted classes with
            ``n_alpha != n_beta``, or paired-generalized with an odd total.
        """
        if n_alpha < 0 or n_beta < 0:
            raise InputError("electron counts must be non-negative")
        if self.spin is SpinSymmetry.RESTRICTED and n_alpha != n_beta:
            raise InputError(
                f"a restricted state needs n_alpha == n_beta, "
                f"got {n_alpha} and {n_beta}"
            )
        if (
            self.spin is SpinSymmetry.UNRESTRICTED
            and self.field is Field.PAIRED
            and n_alpha != n_beta
        ):
            raise InputError(
                f"paired UHF needs n_alpha == n_beta, got {n_alpha} and {n_beta}"
            )
        if (
            self.spin is SpinSymmetry.GENERALIZED
            and self.field is Field.PAIRED
            and (n_alpha + n_beta) % 2
        ):
            raise InputError("paired GHF needs an even number of electrons")

"""Symmetry-adapted linear combinations of basis functions in the standard frame.

In the standard frame of an Abelian point group every operation is a sign
operation (ADR-0022). It maps a basis function on atom ``A`` to the same
function on the image atom ``g(A)``, times the character of the function's
parity under the operation (:func:`solid_harmonic_parity`), so the group
acts on the basis by signed permutations. The projection operator of an
irrep applied to a basis function is a signed sum over the orbit of its
atom, and the projections of all functions give the SALCs of the irrep as
an orthonormal set of coefficient vectors, orthogonal in the coefficient
space rather than in the overlap metric. Every operator that commutes with
the group, the overlap, the core Hamiltonian and the Fock matrix of a
symmetric density, is block-diagonal over the irreps in that basis.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from hafnium.backend import (
    Array,
    adjoint,
    array_namespace,
    default_namespace,
    generalized_eigh,
)
from hafnium.basis import BasisSet
from hafnium.errors import InputError, SymmetryError
from hafnium.molecule import AtomicSystem
from hafnium.structured import IrrepSectors
from hafnium.symmetry.detection import DEFAULT_SYMMETRY_TOLERANCE
from hafnium.symmetry.point_group import PointGroup, solid_harmonic_parity

DEFAULT_LABEL_TOLERANCE = 1e-8
"""Weight an orbital may carry outside its irrep and still be labelled."""

DEFAULT_DEGENERACY_TOLERANCE = 1e-6
"""Relative spacing below which orbital energies count as degenerate."""


@dataclass(frozen=True, slots=True, eq=False)
class SymmetryAdaptation:
    """The signed-permutation representation of a group on a basis and its SALCs.

    Build it with :meth:`from_basis`. The arrays are host arrays of the
    default namespace.

    Parameters
    ----------
    group
        The Abelian group in whose standard frame the system is given.
    atom_images
        ``[operation][atom]``: the atom each operation maps an atom onto.
    function_images
        ``[operation, ao]``: the basis function each operation maps a
        function onto.
    function_signs
        ``[operation, ao]``: the sign the function picks up, ``±1``.
    coefficients
        One ``[ao, salc]`` array per irrep, orthonormal columns; an irrep
        the basis does not span has no columns.
    """

    group: PointGroup
    atom_images: tuple[tuple[int, ...], ...]
    function_images: Array
    function_signs: Array
    coefficients: tuple[Array, ...]
    _sectors: IrrepSectors = field(init=False, repr=False)

    def __post_init__(self) -> None:
        spanned = [
            (label, block)
            for label, block in zip(self.labels, self.coefficients, strict=True)
            if int(block.shape[1]) > 0
        ]
        sectors = IrrepSectors(
            self.group.name,
            tuple(label for label, _ in spanned),
            tuple(block for _, block in spanned),
        )
        object.__setattr__(self, "_sectors", sectors)

    @classmethod
    def from_basis(
        cls,
        basis: BasisSet,
        system: AtomicSystem,
        group: PointGroup,
        *,
        tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
    ) -> SymmetryAdaptation:
        """Build the representation of ``group`` on ``basis`` over ``system``.

        Raises
        ------
        SymmetryError
            If an operation of the group does not map every atom onto an
            atom of the same element within ``tolerance`` Bohr about the
            origin, so that the system is not in the group's standard frame
            (see :meth:`PointGroupDetection.orient`), or if atoms the group
            relates carry different shells.
        """
        if basis.n_atoms != system.n_atoms:
            raise InputError("the basis set and the system have different atom counts")
        atom_images = tuple(
            _atom_images(system, operation.signs, operation.symbol, tolerance)
            for operation in group.operations
        )
        _require_equal_shells(basis, atom_images)
        xp = default_namespace()
        starts = _atom_starts(basis)
        labels = basis.function_labels()
        parities = [
            solid_harmonic_parity(label.angular_momentum, label.m) for label in labels
        ]
        images: list[list[int]] = []
        signs: list[list[float]] = []
        for operation, atoms in zip(group.operations, atom_images, strict=True):
            images.append(
                [
                    starts[atoms[label.atom]] + (index - starts[label.atom])
                    for index, label in enumerate(labels)
                ]
            )
            signs.append([float(operation.character(parity)) for parity in parities])
        function_images = xp.asarray(images, dtype=xp.int64)
        function_signs = xp.asarray(signs, dtype=xp.float64)
        return cls(
            group,
            atom_images,
            function_images,
            function_signs,
            _salcs(group, atom_images, function_images, function_signs, starts),
        )

    @property
    def n_functions(self) -> int:
        """Number of basis functions."""
        return int(self.function_images.shape[1])

    @property
    def dimensions(self) -> tuple[int, ...]:
        """Number of SALCs of each irrep."""
        return tuple(int(block.shape[1]) for block in self.coefficients)

    @property
    def labels(self) -> tuple[str, ...]:
        """Mulliken label of each irrep."""
        return self.group.labels

    def irreps(self) -> IrrepSectors:
        """Return the irrep sectors of the basis for the structured matrices.

        One sector per irrep the basis spans, with its SALCs, in the order
        of the group's irreps (ADR-0023); irreps without SALCs are left out.
        The same object is returned every time, so structures built from
        one adaptation share it.
        """
        return self._sectors

    def transformation(self) -> Array:
        """Return all SALCs as one orthogonal matrix ``[ao, ao]``, irrep by irrep."""
        xp = array_namespace(*self.coefficients)
        return xp.concat(self.coefficients, axis=1)

    def function_irreps(self) -> tuple[int, ...]:
        """Irrep index of every column of :meth:`transformation`."""
        return tuple(
            irrep
            for irrep, dimension in enumerate(self.dimensions)
            for _ in range(dimension)
        )

    def representation(self, operation: int) -> Array:
        """Matrix ``[ao, ao]`` of an operation on the basis, a signed permutation.

        ``P[g(μ), μ]`` is the sign of function ``μ``; an operator ``M``
        that commutes with the group satisfies ``P M Pᵀ = M``.
        """
        xp = array_namespace(self.function_images)
        rows = xp.take(
            xp.eye(self.n_functions), self.function_images[operation], axis=0
        )
        return rows * xp.expand_dims(self.function_signs[operation], 1)

    def to_irrep_blocks(self, matrix: Array) -> tuple[Array, ...]:
        """Return ``U_Γᵀ M U_Γ`` of an operator ``M`` ``[..., ao, ao]`` per irrep."""
        xp = array_namespace(matrix)
        return tuple(
            xp.linalg.matrix_transpose(block) @ matrix @ block
            for block in self.coefficients
        )

    def from_irrep_blocks(self, blocks: tuple[Array, ...]) -> Array:
        """Assemble an operator ``[..., ao, ao]`` from its irrep blocks."""
        xp = array_namespace(*blocks)
        total = None
        for coefficients, block in zip(self.coefficients, blocks, strict=True):
            part = coefficients @ block @ xp.linalg.matrix_transpose(coefficients)
            total = part if total is None else total + part
        assert total is not None
        return total

    def orbital_weights(self, mo_coefficients: Array, overlap: Array) -> Array:
        """Weight ``[irrep, mo]`` of every irrep in every orbital, summing to one.

        The weight of an irrep is the norm, in the overlap metric, of the
        orbital's component in that irrep's SALCs.
        """
        xp = array_namespace(mo_coefficients, overlap)
        if int(mo_coefficients.shape[-2]) != self.n_functions:
            raise InputError(
                f"the orbitals have {mo_coefficients.shape[-2]} rows, the basis "
                f"{self.n_functions} functions"
            )
        weights: list[Array] = []
        for block in self.coefficients:
            transposed = xp.linalg.matrix_transpose(block)
            components = transposed @ mo_coefficients
            metric = transposed @ overlap @ block
            weights.append(
                xp.real(xp.sum(xp.conj(components) * (metric @ components), axis=-2))
            )
        stacked = xp.stack(weights, axis=0)
        return stacked / xp.sum(stacked, axis=0)

    def orbital_irreps(
        self,
        mo_coefficients: Array,
        overlap: Array,
        *,
        tolerance: float = DEFAULT_LABEL_TOLERANCE,
    ) -> tuple[int, ...]:
        """Irrep index of every orbital of ``mo_coefficients`` ``[ao, mo]``.

        Raises
        ------
        SymmetryError
            If an orbital carries more than ``tolerance`` of its weight
            outside one irrep: it is symmetry-broken, or a member of a
            degenerate set that :meth:`symmetrize_orbitals` can resolve.
        """
        xp = array_namespace(mo_coefficients, overlap)
        weights = self.orbital_weights(mo_coefficients, overlap)
        largest = xp.max(weights, axis=0)
        mixed = [
            index
            for index in range(int(largest.shape[0]))
            if largest[index] < 1.0 - tolerance
        ]
        if mixed:
            raise SymmetryError(
                f"orbitals {mixed} are not symmetry-adapted (largest irrep weights "
                f"{[float(largest[index]) for index in mixed]}); degenerate sets are "
                "resolved by symmetrize_orbitals"
            )
        return tuple(int(index) for index in xp.argmax(weights, axis=0))

    def symmetrize_orbitals(
        self,
        mo_coefficients: Array,
        mo_energies: Array,
        overlap: Array,
        *,
        degeneracy_tolerance: float = DEFAULT_DEGENERACY_TOLERANCE,
        tolerance: float = DEFAULT_LABEL_TOLERANCE,
    ) -> tuple[Array, tuple[int, ...]]:
        """Rotate degenerate orbitals into pure irreps and label every orbital.

        Orbitals whose energies differ by less than ``degeneracy_tolerance``
        relative to the larger of one and the spectral range form a set;
        within each set the operator ``Σ_Γ Γ P_Γ`` of the irrep projectors
        is diagonalized, whose eigenvectors are the pure orbitals and whose
        eigenvalues are their irrep indices. Returns the rotated
        coefficients ``[ao, mo]`` and the irreps.

        Raises
        ------
        SymmetryError
            If a set is not invariant under the group, so that no rotation
            makes its members pure.
        """
        xp = array_namespace(mo_coefficients, mo_energies, overlap)
        projector = self._weighted_projector(mo_coefficients, overlap)
        metric = adjoint(mo_coefficients) @ overlap @ mo_coefficients
        columns: list[Array] = []
        irreps: list[int] = []
        for start, end in degenerate_sets(mo_energies, degeneracy_tolerance):
            block = projector[start:end, start:end]
            # The generalized problem in the orbitals' own overlap makes the
            # eigenvalues irrep indices for any set spanning an invariant
            # space, normalized or not; the rotated set is orthonormal.
            values, vectors = generalized_eigh(
                (block + adjoint(block)) / 2, metric[start:end, start:end]
            )
            rotated = mo_coefficients[:, start:end] @ vectors
            for position in range(end - start):
                value = float(values[position])
                if abs(value - round(value)) > tolerance:
                    raise SymmetryError(
                        f"orbitals {list(range(start, end))} do not span a space "
                        f"invariant under the group (projector eigenvalue {value}): "
                        "a single orbital is symmetry-broken, a degenerate set "
                        "incomplete"
                    )
                irreps.append(round(value))
            columns.append(rotated)
        return xp.concat(columns, axis=1), tuple(irreps)

    def _weighted_projector(self, mo_coefficients: Array, overlap: Array) -> Array:
        """Return ``Σ_Γ Γ ⟨p|P_Γ|q⟩`` over the orbitals, ``[mo, mo]``."""
        xp = array_namespace(mo_coefficients, overlap)
        total = None
        for irrep, block in enumerate(self.coefficients):
            transposed = xp.linalg.matrix_transpose(block)
            components = transposed @ mo_coefficients
            metric = transposed @ overlap @ block
            part = irrep * (adjoint(components) @ metric @ components)
            total = part if total is None else total + part
        assert total is not None
        return total


def degenerate_sets(energies: Array, tolerance: float) -> list[tuple[int, int]]:
    """Return ``(start, end)`` of the sets of nearly equal consecutive energies.

    ``energies`` ``[mo]`` are in ascending order; consecutive energies
    closer than ``tolerance`` times the larger of one and the spectral
    range belong to one set.
    """
    values = [float(energy) for energy in energies]
    if any(b < a for a, b in itertools.pairwise(values)):
        raise InputError("orbital energies are in ascending order")
    spread = max(1.0, values[-1] - values[0]) if values else 1.0
    sets: list[tuple[int, int]] = []
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or values[end] - values[end - 1] > tolerance * spread:
            sets.append((start, end))
            start = end
    return sets


def _atom_images(
    system: AtomicSystem, signs: tuple[int, int, int], symbol: str, tolerance: float
) -> tuple[int, ...]:
    """Return the atom each atom is mapped onto by the sign operation ``signs``."""
    xp = default_namespace()
    positions = system.positions_array(xp)
    moved = positions * xp.asarray(signs, dtype=xp.float64)
    distance = xp.linalg.vector_norm(
        xp.expand_dims(moved, 1) - xp.expand_dims(positions, 0), axis=-1
    )
    nearest = xp.argmin(distance, axis=1)
    images: list[int] = []
    for atom in range(system.n_atoms):
        image = int(nearest[atom])
        if (
            float(distance[atom, image]) > tolerance
            or system.symbols[image] != system.symbols[atom]
        ):
            raise SymmetryError(
                f"the operation {symbol} does not map atom {atom} onto an equivalent "
                "atom: the system is not in the standard frame of the group about "
                "the origin; orient it with PointGroupDetection.orient first"
            )
        images.append(image)
    return tuple(images)


def _require_equal_shells(
    basis: BasisSet, atom_images: tuple[tuple[int, ...], ...]
) -> None:
    shells = [
        tuple(
            shell
            for shell, centre in zip(basis.shells, basis.shell_atoms, strict=True)
            if centre == atom
        )
        for atom in range(basis.n_atoms)
    ]
    for images in atom_images:
        for atom, image in enumerate(images):
            if shells[atom] != shells[image]:
                raise SymmetryError(
                    f"atoms {atom} and {image} are related by the group but carry "
                    "different basis functions"
                )


def _atom_starts(basis: BasisSet) -> tuple[int, ...]:
    """Return the index of the first function of every atom."""
    offsets = basis.shell_offsets()
    starts = [basis.n_functions] * basis.n_atoms
    for shell, atom in reversed(list(enumerate(basis.shell_atoms))):
        starts[atom] = offsets[shell]
    return tuple(starts)


def _salcs(
    group: PointGroup,
    atom_images: tuple[tuple[int, ...], ...],
    function_images: Array,
    function_signs: Array,
    starts: tuple[int, ...],
) -> tuple[Array, ...]:
    """Project the functions of one atom per orbit onto each irrep.

    The projector of an irrep is ``Σ_g χ_Γ(g) P_g`` over the signed
    permutations ``P_g``; its columns at the representative functions,
    normalized where they do not vanish, are the SALCs of the irrep.
    """
    xp = array_namespace(function_images, function_signs)
    n_ao = int(function_images.shape[1])
    eye = xp.eye(n_ao)
    permutations = [
        xp.take(eye, function_images[g], axis=0) * xp.expand_dims(function_signs[g], 1)
        for g in range(group.order)
    ]
    representatives = xp.asarray(
        _orbit_representatives(atom_images, starts, n_ao), dtype=xp.int64
    )
    columns: list[Array] = []
    for row in group.characters():
        projector = xp.zeros((n_ao, n_ao))
        for character, permutation in zip(row, permutations, strict=True):
            projector = projector + character * permutation
        selected = xp.take(projector, representatives, axis=1)
        norms = xp.linalg.vector_norm(selected, axis=0)
        keep = xp.asarray(
            [k for k in range(int(norms.shape[0])) if float(norms[k]) > 1e-8],
            dtype=xp.int64,
        )
        columns.append(xp.take(selected, keep, axis=1) / xp.take(norms, keep, axis=0))
    return tuple(columns)


def _orbit_representatives(
    atom_images: tuple[tuple[int, ...], ...], starts: tuple[int, ...], n_ao: int
) -> list[int]:
    """Return the functions of the first atom of every orbit, in order."""
    n_atoms = len(starts)
    covered = [False] * n_atoms
    functions: list[int] = []
    for atom in range(n_atoms):
        if covered[atom]:
            continue
        for images in atom_images:
            covered[images[atom]] = True
        end = starts[atom + 1] if atom + 1 < n_atoms else n_ao
        functions.extend(range(starts[atom], end))
    return functions
